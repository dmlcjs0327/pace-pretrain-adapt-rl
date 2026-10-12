import os as _os
from pathlib import Path as _Path
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", "/opt/pretrain-then-adapt-rl-small-networks/experiments")
import inspect
import numpy as np
import os
import yaml
from collections import deque

import isaacgym

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments

import gym
from gym import spaces
from argparse import Namespace

from rl_games.common import env_configurations, vecenv

# Optional ceiling of the rl-games adaptive learning-rate schedule. rl-games fixes it at 1e-2,
# which every Study 2 condition reached within 10 iterations. AG_ADAPTIVE_MAX_LR=<value> sets it.
if os.environ.get("AG_ADAPTIVE_MAX_LR"):
    from rl_games.common import schedulers as _sch
    _orig_adaptive_init = _sch.AdaptiveScheduler.__init__

    def _adaptive_init(self, kl_threshold=0.008):
        _orig_adaptive_init(self, kl_threshold)
        self.max_lr = float(os.environ["AG_ADAPTIVE_MAX_LR"])

    _sch.AdaptiveScheduler.__init__ = _adaptive_init

# Smaller learning rate for the backbone: AG_BACKBONE_LR_SCALE=<s>
# puts the encoder and GRU into an Adam parameter group whose learning rate is s times the schedule's.
if os.environ.get("AG_BACKBONE_LR_SCALE"):
    import torch as _torch
    from rl_games.algos_torch import a2c_continuous as _a2c
    from rl_games.common import a2c_common as _ac
    _orig_agent_init = _a2c.A2CAgent.__init__

    def _agent_init(self, base_name, params):
        _orig_agent_init(self, base_name, params)
        scale = float(os.environ["AG_BACKBONE_LR_SCALE"])
        bb, rest = [], []
        for n, p in self.model.named_parameters():
            (bb if any(k in n for k in (".policy.proprio_mlp", ".policy.fusion_mlp", ".policy.gru")) else rest).append(p)
        self.optimizer = _torch.optim.Adam([{"params": rest, "lr_scale": 1.0}, {"params": bb, "lr_scale": scale}],
                                           float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        for g in self.optimizer.param_groups:
            g["lr"] = float(self.last_lr) * g["lr_scale"]
        print(f"[backbone lr] scale {scale}: {sum(p.numel() for p in bb)} backbone weights, {sum(p.numel() for p in rest)} others")

    _a2c.A2CAgent.__init__ = _agent_init
    _orig_update_lr = _ac.A2CBase.update_lr

    def _update_lr(self, lr):
        _orig_update_lr(self, lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)

    _ac.A2CBase.update_lr = _update_lr

import torch
import distutils

# Register custom network builder for the pretrain-adapt experiments (B1+).
# Idempotent — safe to import even when other YAMLs (`actor_critic`) are used.
from aerial_gym.rl_training.rl_games import network_pretrain_adapt as _net_pa
_net_pa.register()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class CommonMetricTensorboardWriter:
    TAG_MAP = {
        "rewards/step": "eval/return_raw/step",
        "rewards/time": "eval/return_raw/time",
        "shaped_rewards/step": "eval/return_scaled/step",
        "shaped_rewards/time": "eval/return_scaled/time",
        "episode_lengths/step": "eval/episode_length/step",
        "episode_lengths/time": "eval/episode_length/time",
    }

    PROGRESS_TAG_MAP = {
        "rewards/step": "eval/return_raw/progress",
        "shaped_rewards/step": "eval/return_scaled/progress",
        "episode_lengths/step": "eval/episode_length/progress",
    }

    def __init__(self, writer, algo):
        self.writer = writer
        self.algo = algo

    def _max_training_steps(self):
        if getattr(self.algo, "max_frames", -1) and self.algo.max_frames > 0:
            return float(self.algo.max_frames)
        if getattr(self.algo, "max_epochs", -1) and self.algo.max_epochs > 0:
            return float(self.algo.batch_size_envs * self.algo.max_epochs)
        return None

    def _progress_index(self, current_step):
        max_steps = self._max_training_steps()
        if max_steps is None:
            return None
        progress = min(1.0, max(0.0, float(current_step) / max_steps))
        return int(round(progress * 1000.0))

    def add_scalar(self, tag, value, step, *args, **kwargs):
        # Pass through: eval/*, grad_norm/* (B5.3 monitoring), pretrain_adapt/* (B2 aux loss).
        if tag.startswith("eval/") or tag.startswith("grad_norm/") or tag.startswith("pretrain_adapt/"):
            self.writer.add_scalar(tag, value, step, *args, **kwargs)
            return
        # Opt-in: keep the learning rate,
        # KL, and entropy traces of rl-games so that the optimizer state can be read
        # during training. Default behaviour is unchanged.
        if os.environ.get("AG_PASS_TRAIN_INFO") == "1" and (tag.startswith("info/") or tag.startswith("losses/")):
            self.writer.add_scalar(tag, value, step, *args, **kwargs)
            return
        mapped_tag = self.TAG_MAP.get(tag)
        if mapped_tag is None:
            return
        self.writer.add_scalar(mapped_tag, value, step, *args, **kwargs)
        progress_tag = self.PROGRESS_TAG_MAP.get(tag)
        progress_index = self._progress_index(step)
        if progress_tag is not None and progress_index is not None:
            self.writer.add_scalar(progress_tag, value, progress_index, *args, **kwargs)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __getattr__(self, item):
        return getattr(self.writer, item)

class CommonMetricObserver:
    def __init__(self, window_size=256):
        self.window_size = window_size

    def before_init(self, base_name, config, experiment_name):
        return

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer
        self.success_history = deque(maxlen=self.window_size)
        self.crash_history = deque(maxlen=self.window_size)
        self.timeout_history = deque(maxlen=self.window_size)
        self.flight_history = {}

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or len(done_indices) == 0:
            return
        required_keys = ("success", "crashes", "timeouts")
        if not all(key in infos for key in required_keys):
            return

        done_env_indices = (done_indices[:: self.algo.num_agents] // self.algo.num_agents).view(-1)
        for env_idx in done_env_indices.tolist():
            self.success_history.append(float(infos["success"][env_idx].item()))
            self.crash_history.append(float(infos["crashes"][env_idx].item()))
            self.timeout_history.append(float(infos["timeouts"][env_idx].item()))
        # Per-episode flight metrics reported by the task (keys "flight/*"), logging only.
        for key, val in infos.items():
            if isinstance(key, str) and key.startswith("flight/") and torch.is_tensor(val):
                vals = val[done_env_indices].detach().float().cpu().tolist()
                self.flight_history.setdefault(key, deque(maxlen=self.window_size)).extend(vals)

    def after_steps(self):
        return

    def after_clear_stats(self):
        self.success_history.clear()
        self.crash_history.clear()
        self.timeout_history.clear()
        for dq in self.flight_history.values():
            dq.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        # Critic warm-up: restore the requires_grad pattern of the adaptation mode after N iterations.
        cw = os.environ.get("AG_CRITIC_WARMUP")
        if cw and not getattr(self, "_warm_done", False) and epoch_num >= int(cw):
            net = self.algo.model.a2c_network
            for n, p in net.named_parameters():
                if n in net._rg_mask:
                    p.requires_grad = net._rg_mask[n]
            self._warm_done = True
            # rule drives the learning rate to its ceiling; the released weights would then take their first Adam
            # steps at the ceiling (KL 0.93 in a test). Restart the learning rate at its initial value, as at the
            # start of every run (policy learning-rate warm-up after critic warm-up, PIRLNav).
            self.algo.last_lr = float(self.algo.config["learning_rate"])
            self.algo.update_lr(self.algo.last_lr)
            print(f"[critic warm-up] actor and backbone released after iteration {epoch_num}; "
                  f"learning rate restarted at {self.algo.last_lr:g}", flush=True)
        # LP-FT: AG_UNFREEZE_AT=<N> unfreezes the backbone after N iterations of head-only training.
        unf = os.environ.get("AG_UNFREEZE_AT")
        if unf and not getattr(self, "_unfrozen", False) and epoch_num >= int(unf):
            pol = self.algo.model.a2c_network.policy
            for name in ("proprio_mlp", "fusion_mlp", "gru"):
                for p in getattr(pol, name).parameters():
                    p.requires_grad = True
            self._unfrozen = True
            # Same restart of the learning rate as after the critic warm-up (section 12): the backbone would
            # otherwise take its first Adam steps at the learning rate reached during head-only training.
            self.algo.last_lr = float(self.algo.config["learning_rate"])
            self.algo.update_lr(self.algo.last_lr)
            print(f"[LPFT] backbone unfrozen after iteration {epoch_num}; learning rate restarted at "
                  f"{self.algo.last_lr:g}", flush=True)
        # Actor learning-rate warm-up after release: for the BC
        # backbone, AG_ACTOR_WARMUP=<W> raises the learning-rate ceiling linearly from base/W to base over the W
        # iterations after each release (end of the critic warm-up, LP-FT unfreeze). The adaptive rule acts below
        # the ceiling; with a constant schedule the rate itself is scaled.
        aw = os.environ.get("AG_ACTOR_WARMUP")
        if aw:
            W = int(aw)
            sch = getattr(self.algo, "scheduler", None)
            adaptive = sch is not None and hasattr(sch, "max_lr")
            if not hasattr(self, "_aw_base"):
                self._aw_base = float(sch.max_lr) if adaptive else float(self.algo.config["learning_rate"])
            rel = [int(x) for x in (os.environ.get("AG_CRITIC_WARMUP"), os.environ.get("AG_UNFREEZE_AT")) if x]
            past = [r for r in rel if epoch_num >= r]
            if past:
                frac = min(1.0, (epoch_num - max(past) + 1) / W)
                cap = self._aw_base * frac
                if adaptive:
                    sch.max_lr = cap
                    if self.algo.last_lr > cap:
                        self.algo.last_lr = cap
                        self.algo.update_lr(cap)
                else:
                    self.algo.last_lr = cap
                    self.algo.update_lr(cap)
                if frac < 1.0 and (epoch_num - max(past)) % 10 == 0:
                    print(f"[actor warm-up] iteration {epoch_num}: learning-rate ceiling {cap:.2e}", flush=True)
        if self.writer is None or len(self.success_history) == 0:
            return

        success_rate = sum(self.success_history) / len(self.success_history)
        crash_rate = sum(self.crash_history) / len(self.crash_history)
        timeout_rate = sum(self.timeout_history) / len(self.timeout_history)
        max_steps = None
        if getattr(self.algo, "max_frames", -1) and self.algo.max_frames > 0:
            max_steps = float(self.algo.max_frames)
        elif getattr(self.algo, "max_epochs", -1) and self.algo.max_epochs > 0:
            max_steps = float(self.algo.batch_size_envs * self.algo.max_epochs)
        progress_index = None
        if max_steps is not None:
            progress_index = int(round(min(1.0, max(0.0, float(frame) / max_steps)) * 1000.0))

        self.writer.add_scalar("eval/success_rate/step", success_rate, frame)
        self.writer.add_scalar("eval/success_rate/time", success_rate, total_time)
        self.writer.add_scalar("eval/crash_rate/step", crash_rate, frame)
        self.writer.add_scalar("eval/crash_rate/time", crash_rate, total_time)
        self.writer.add_scalar("eval/timeout_rate/step", timeout_rate, frame)
        self.writer.add_scalar("eval/timeout_rate/time", timeout_rate, total_time)
        if progress_index is not None:
            self.writer.add_scalar("eval/success_rate/progress", success_rate, progress_index)
            self.writer.add_scalar("eval/crash_rate/progress", crash_rate, progress_index)
            self.writer.add_scalar("eval/timeout_rate/progress", timeout_rate, progress_index)
        for key, dq in self.flight_history.items():
            arr = np.asarray(dq, dtype=float)
            if arr.size and not np.all(np.isnan(arr)):
                self.writer.add_scalar(f"eval/{key}/step", float(np.nanmean(arr)), frame)

def _install_common_metric_tensorboard_patch():
    import rl_games.common.a2c_common as a2c_common

    if getattr(a2c_common, "_common_metric_patch_installed", False):
        return

    original_init = a2c_common.A2CBase.__init__

    def patched_init(self, base_name, params):
        original_init(self, base_name, params)
        if self.writer is not None and not isinstance(self.writer, CommonMetricTensorboardWriter):
            self.writer = CommonMetricTensorboardWriter(self.writer, self)

    def patched_write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        a_losses,
        c_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ):
        # Opt-in: record the optimizer state during training.
        if os.environ.get("AG_PASS_TRAIN_INFO") == "1" and self.writer is not None:
            from rl_games.algos_torch import torch_ext

            self.writer.add_scalar("info/last_lr", last_lr * lr_mul, frame)
            self.writer.add_scalar("info/kl", torch_ext.mean_list(kls).item(), frame)
            self.writer.add_scalar("losses/entropy", torch_ext.mean_list(entropies).item(), frame)
            self.writer.add_scalar("losses/a_loss", torch_ext.mean_list(a_losses).item(), frame)
            self.writer.add_scalar("losses/c_loss", torch_ext.mean_list(c_losses).item(), frame)
            # clipping (mean over the minibatches of the iteration, and the fraction that was clipped).
            for k, v in self.aux_loss_dict.items():
                self.writer.add_scalar("losses/" + k, torch_ext.mean_list(v).item(), frame)
            gn = getattr(self, "_ag_grad_norms", [])
            if gn:
                self.writer.add_scalar("info/grad_norm", float(sum(gn) / len(gn)), frame)
                self.writer.add_scalar("info/grad_clipped", float(sum(g > self.grad_norm for g in gn) / len(gn)), frame)
                self._ag_grad_norms = []
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    original_tgs = a2c_common.A2CBase.trancate_gradients_and_step

    def patched_tgs(self):
        # Read-only record of the total gradient norm before clipping; the update itself is unchanged.
        if os.environ.get("AG_PASS_TRAIN_INFO") == "1" and self.truncate_grads and not self.multi_gpu:
            with torch.no_grad():
                norms = [p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None]
                if norms:
                    if not hasattr(self, "_ag_grad_norms"):
                        self._ag_grad_norms = []
                    self._ag_grad_norms.append(float(torch.norm(torch.stack(norms))))
        return original_tgs(self)

    a2c_common.A2CBase.__init__ = patched_init
    a2c_common.A2CBase.write_stats = patched_write_stats
    a2c_common.A2CBase.trancate_gradients_and_step = patched_tgs
    a2c_common._common_metric_patch_installed = True

def _install_training_log_patch():
    """Replace rl_games fps-only log with mean reward and mean episode length (and epoch/frames)."""
    import rl_games.common.a2c_common as a2c_common
    _original_print_statistics = a2c_common.print_statistics

    def _custom_print_statistics(
        print_stats, curr_frames, step_time, step_inference_time, total_time,
        epoch_num, max_epochs, frame, max_frames,
    ):
        if not print_stats:
            return
        # Get caller's self (the agent) to read game_rewards / game_lengths
        try:
            caller_frame = inspect.currentframe().f_back
            agent = caller_frame.f_locals.get("self") if caller_frame else None
        except Exception:
            agent = None
        mean_reward = None
        mean_length = None
        if agent is not None and getattr(agent, "game_rewards", None):
            if getattr(agent.game_rewards, "current_size", 0) > 0:
                try:
                    mean_reward = agent.game_rewards.get_mean()
                    mean_reward = mean_reward[0] if hasattr(mean_reward, "__getitem__") else float(mean_reward)
                except Exception:
                    pass
            if getattr(agent, "game_lengths", None) and getattr(agent.game_lengths, "current_size", 0) > 0:
                try:
                    mean_length = float(agent.game_lengths.get_mean())
                except Exception:
                    pass
        # Prefer reward/ep_len; fallback to original fps line
        if mean_reward is not None:
            ep_str = f"  mean_ep_len: {mean_length:.1f}" if mean_length is not None else ""
            epoch_str = f"{epoch_num:.0f}" if max_epochs == -1 else f"{epoch_num:.0f}/{max_epochs:.0f}"
            frame_str = f"{frame:.0f}" if max_frames == -1 else f"{frame:.0f}/{max_frames:.0f}"
            print(f"epoch: {epoch_str}  frames: {frame_str}  mean_reward: {mean_reward:.4f}{ep_str}")
        else:
            _original_print_statistics(
                print_stats, curr_frames, step_time, step_inference_time, total_time,
                epoch_num, max_epochs, frame, max_frames,
            )

    a2c_common.print_statistics = _custom_print_statistics
# import warnings
# warnings.filterwarnings("error")

class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        return (
            observations["observations"],
            rewards,
            dones,
            infos,
        )

class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)
        # S20 (pre-registration v2, section 21): AG_OBS_EXPAND=1 standardizes the 13-dimensional state with the fixed
        # pretraining statistics and expands it with the fixed sensor model of ssl_pretrain/expand.py (128 inputs).
        # The network then uses identity statistics, so the stored observation is exactly what the policy saw.
        self._expand = os.environ.get("AG_OBS_EXPAND") == "1"
        if self._expand:
            from aerial_gym.ssl_pretrain.expand import expansion_params, OUT_DIM
            st = torch.load(os.environ["AG_OBS_EXPAND_STATS"], map_location="cpu")
            dev = getattr(self.env.env, "device", "cuda")  # the Aerial Gym task inside ExtractObsWrapper
            self._m = torch.as_tensor(st["state_mean"], dtype=torch.float32, device=dev)
            self._s = torch.as_tensor(st["state_std"], dtype=torch.float32, device=dev)
            self._A, self._b = expansion_params(dev)
            self._out_dim = OUT_DIM
            print(f"[runner] S20 observation expansion to {OUT_DIM} inputs")

    def _x(self, obs):
        if not self._expand:
            return obs
        from aerial_gym.ssl_pretrain.expand import expand
        x = torch.clamp((obs - self._m) / self._s, -5.0, 5.0)
        return expand(x, self._A, self._b)

    def step(self, actions):
        obs, rew, dones, infos = self.env.step(actions)
        return self._x(obs), rew, dones, infos

    def reset(self):
        return self._x(self.env.reset())

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        obs_dim = self._out_dim if self._expand else self.env.task_config.observation_space_dim
        info["observation_space"] = spaces.Box(
            np.ones(obs_dim) * -np.Inf,
            np.ones(obs_dim) * np.Inf,
        )
        print(info["action_space"], info["observation_space"])
        return info

env_configurations.register(
    "position_setpoint_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_px4",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_px4", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_acceleration_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_acceleration_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

# D1-D4 navigation task variants (proposal §5.3 main grid).
env_configurations.register(
    "navigation_task_d1",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task_d1", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task_d2",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task_d2", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task_d3",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task_d3", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task_d4",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task_d4", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

# SSL pretrain × adapt:
for _ssl_task in [
    # T1 = state_setpoint (regulation)
    "state_setpoint_baseQuad_clean",
    "state_setpoint_baseQuad_disturb",
    "state_setpoint_x500_clean",
    "state_setpoint_lmf2_disturb",
    "ss_lmf2_clean",                      # T1 × R3 (lmf2 clean)
    "traj_baseQuad",
    "traj_x500",
    "traj_lmf2",
    "recovery_baseQuad",
    "recovery_x500",
    "recovery_lmf2",
    "agg_baseQuad",
    "agg_x500",
    "agg_lmf2",
    # dynamics-varied T1/T2
    "ss_dr_x500", "ss_dr_lmf2", "recovery_dr_x500", "recovery_dr_lmf2",
    # difficulty ladder of T3
    "agg_d1_x500", "agg_d1_lmf2", "agg_d2_x500", "agg_d2_lmf2",
    # T3 with diverse initial states
    "agg_upset_x500", "agg_upset_lmf2",
    "psm_baseQuad", "psm_x500", "psm_lmf2",
    "gc_baseQuad", "gc_x500", "gc_lmf2",
]:
    env_configurations.register(
        _ssl_task,
        {
            "env_creator": (lambda name=_ssl_task: lambda **kwargs: task_registry.make_task(name, **kwargs))(),
            "vecenv_type": "AERIAL-RLGPU",
        },
    )

env_configurations.register(
    "position_setpoint_gc_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_gc_task", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_reconfigurable",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_reconfigurable", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_morphy",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_morphy", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_end_to_end",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_end_to_end", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)

# Second domain (pre-registration v2, section 20): IsaacGymEnvs Anymal, built from a local clone without hydra by
# my research/2026 Neural Networks/experiments/domain2/igenv.py. Only task names starting with "ige_" use it.
D2_DIR = _EXP_ROOT + "/domain2"
# (task, overrides, blind, strip_heights). Section 29: the second domain moves to the AnymalTerrain task
# of IsaacGymEnvs, so that the easy (flat) and hard (rough) tasks share one robot model, actuation, and input layout.
# The policy never sees the height scan: inputs 36:176 are removed, leaving the 48 inputs of the Anymal layout
# (base linear velocity 3, angular velocity 3, gravity 3, commands 3, joint positions 12, joint velocities 12, actions 12).
# The collector of section 29 is the shipped AnymalTerrain PPO with the height scan (privileged, 188 inputs).
_PLANE = {"env.terrain.terrainType": "plane", "env.terrain.curriculum": False}
D2_TASKS = {"ige_anymal_flat": ("Anymal", {}, False, False), "ige_anymal_blind": ("Anymal", {}, True, False),
            "ige_terrain_flat": ("AnymalTerrain", _PLANE, False, True),
            "ige_terrain_flat_blind": ("AnymalTerrain", _PLANE, True, True),
            "ige_terrain_rough_curr": ("AnymalTerrain", {}, False, True),
            "ige_terrain_rough": ("AnymalTerrain", {"env.terrain.curriculum": False}, False, True),
            "ige_terrain_collector": ("AnymalTerrain", {}, False, False)}

class IGERLGPUEnv(vecenv.IVecEnv):
    """rl_games vector env for IsaacGymEnvs tasks; 'blind' zeroes the base linear velocity (first three inputs),
    'strip' removes the height scan (inputs 36:176) of AnymalTerrain."""

    def __init__(self, config_name, num_actors, **kwargs):
        import sys as _sys
        if D2_DIR not in _sys.path:
            _sys.path.insert(0, D2_DIR)
        import igenv
        task, overrides, self.blind, self.strip = D2_TASKS[config_name]
        self.env = igenv.make(task, int(kwargs.get("num_envs", num_actors)), int(kwargs.get("seed", 0)), overrides)

    def _obs(self, o):
        o = o["obs"] if isinstance(o, dict) else o
        if self.strip:
            o = torch.cat([o[:, :36], o[:, 176:188]], dim=1)
        if self.blind:
            o = o.clone()
            o[:, 0:3] = 0.0
        return o

    def step(self, actions):
        o, r, d, info = self.env.step(actions)
        return self._obs(o), r, d, info

    def reset(self):
        return self._obs(self.env.reset())

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        obs_space = self.env.observation_space
        if self.strip:
            obs_space = gym.spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
        return {"action_space": self.env.action_space, "observation_space": obs_space}

vecenv.register("IGE-RLGPU", lambda config_name, num_actors, **kwargs: IGERLGPUEnv(config_name, num_actors, **kwargs))
for _name in D2_TASKS:
    env_configurations.register(_name, {"env_creator": lambda **kwargs: None, "vecenv_type": "IGE-RLGPU"})

def get_args():
    from isaacgym import gymutil

    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "1024",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "rl_games",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_task",
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Choose whether to use warp or Isaac Gym rendeing pipeline.",
        },
        {
            "name": "--total_env_steps",
            "type": int,
            "required": False,
            "help": "Override total environment steps for off-policy configs.",
        },
    ]

    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args

def update_config(config, args):

    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        # config['params']['config']['num_envs'] = args['num_envs']
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
        # When num_envs is small, batch_size = num_actors * horizon_length can be smaller
        # than minibatch_size; rl_games requires batch_size % minibatch_size == 0.
        horizon_length = config["params"]["config"].get("horizon_length", 32)
        batch_size = args["num_envs"] * horizon_length
        minibatch_size = config["params"]["config"].get("minibatch_size", 8192)
        if minibatch_size > batch_size or (batch_size % minibatch_size != 0):
            config["params"]["config"]["minibatch_size"] = batch_size
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config

def update_offpolicy_config(config, args):
    if args["task"] is not None:
        config["env"]["task_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["experiment"]["name"] = args["experiment_name"]
    if args["num_envs"] > 0:
        config["train"]["num_envs"] = args["num_envs"]
    config["train"]["headless"] = args["headless"]
    config["train"]["use_warp"] = args["use_warp"]
    if args.get("total_env_steps") is not None:
        config["train"]["total_env_steps"] = args["total_env_steps"]
    if args["seed"] > 0:
        config["experiment"]["seed"] = args["seed"]
    return config

if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())

    config_name = args["file"]

    print("Loading config: ", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        if "params" in config:
            config = update_config(config, args)

            from rl_games.torch_runner import Runner

            _install_common_metric_tensorboard_patch()
            runner = Runner(algo_observer=CommonMetricObserver())
            # B4 — register custom A2CAgent that adds aux loss step (proposal §4.3)
            from aerial_gym.rl_training.rl_games.agent_pretrain_adapt import A2CAgentPretrainAdapt
            runner.algo_factory.register_builder(
                "a2c_continuous_pretrain_adapt",
                lambda **kwargs: A2CAgentPretrainAdapt(**kwargs),
            )
            # Experiment D — per-step weight-update-norm logging agent (opt-in via
            # `algo.name: a2c_continuous_update_norm` in the yaml). Stock cells
            # are unaffected.
            from aerial_gym.rl_training.rl_games.agent_update_norm import A2CAgentUpdateNorm
            runner.algo_factory.register_builder(
                "a2c_continuous_update_norm",
                lambda **kwargs: A2CAgentUpdateNorm(**kwargs),
            )
            try:
                runner.load(config)
            except yaml.YAMLError as exc:
                print(exc)

            _install_training_log_patch()

            rank = int(os.getenv("LOCAL_RANK", "0"))
            if args["track"] and rank == 0:
                import wandb

                wandb.init(
                    project=args["wandb_project_name"],
                    entity=args["wandb_entity"],
                    sync_tensorboard=True,
                    config=config,
                    monitor_gym=True,
                    save_code=True,
                )
            runner.run(args)

            if args["track"] and rank == 0:
                wandb.finish()
        else:
            from aerial_gym.rl_training.offpolicy.train import run_experiment

            config = update_offpolicy_config(config, args)
            run_experiment(
                config,
                checkpoint=args.get("checkpoint"),
                play=args.get("play", False),
            )
