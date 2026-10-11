"""T3, the aggressive setpoint task: the target is drawn 4-8 m from the initial position (1-2 m and 2-4 m in the difficulty ladder) and the episode is short, so the vehicle has to reach it fast. The reward is that of PositionSetpointTask."""
import os

import torch

from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
    PositionSetpointTask,
)
from aerial_gym.utils.math import quat_apply_inverse, quat_axis

def _parse_curriculum(spec):
    """AG_T3_CURRICULUM = "100:1:2,200:2:4": up to RL iteration 100 the target distance is drawn from [1, 2] m, up to
    200 from [2, 4] m, afterwards from the task's own range (S14, pre-registration v2 section 18)."""
    out = []
    for part in spec.split(","):
        it, lo, hi = part.split(":")
        out.append((int(it), float(lo), float(hi)))
    return sorted(out)

def _parse_levels(spec):
    """AG_T3_SUCCESS_CURRICULUM = "1:2,2:4,4:8": distance ranges of the levels of the success-gated curriculum (V13,
    pre-registration v2 section 25)."""
    return [(float(lo), float(hi)) for lo, hi in (part.split(":") for part in spec.split(","))]

# V13 (section 25): the level rule of Aerial Gym's navigation task (navigation_task_config.curriculum): after every
# 2,048 finished episodes, one level up if the success rate exceeds 0.7, one level down if it is below 0.6.
SUCC_CHECK_EPISODES, SUCC_UP, SUCC_DOWN = 2048, 0.7, 0.6

@torch.jit.script
def _reward_relative_termination(pos_error, robot_quats, robot_angvels, crashes, radius):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """The reward of position_setpoint_task.compute_reward, term by term, with the termination distance of 8 m
    replaced by a per-vehicle radius (S15, pre-registration v2 section 18)."""
    dist = torch.norm(pos_error, dim=1)
    pos_reward = 3.0 * torch.exp(-8.0 * dist * dist) + 2.0 * torch.exp(-4.0 * dist * dist)
    dist_reward = (20 - dist) / 40.0
    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)
    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3
    total_reward = pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    crashes[:] = torch.where(dist > radius, torch.ones_like(crashes), crashes)
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)
    return total_reward, crashes

class AggressiveSetpointTask(PositionSetpointTask):
    """

    sample logic:
      - distance ∈ [d_min, d_max] (default [4, 8] m)
      - target = init_pos + dir * dist

    """

    def __init__(self, task_config, **kwargs):
        super().__init__(task_config, **kwargs)
        self.aggr_dist_min = float(getattr(task_config, "aggressive_dist_min", 4.0))
        self.aggr_dist_max = float(getattr(task_config, "aggressive_dist_max", 8.0))
        # Training-time remedies of S14 and S15 (section 18), off unless the runner sets them; evaluation never does.
        spec = os.environ.get("AG_T3_CURRICULUM", "")
        self._curriculum = _parse_curriculum(spec) if spec else []
        self._horizon = int(os.environ.get("AG_HORIZON", "32"))
        margin = os.environ.get("AG_T3_TERM_MARGIN", "")
        self._term_margin = float(margin) if margin else None
        self._d0 = torch.full((self.sim_env.num_envs,), 8.0, device=self.device)
        # V6 and V13 (section 25), training only, off unless the runner sets them; evaluation never does.
        prog = os.environ.get("AG_T3_PROGRESS", "")
        self._prog_k = float(prog) if prog else 0.0
        self._d_prev = None
        sc = os.environ.get("AG_T3_SUCCESS_CURRICULUM", "")
        self._levels = _parse_levels(sc) if sc else []
        self._level, self._lv_n, self._lv_s = 0, 0, 0.0
        self._sample_aggressive_targets(torch.arange(self.sim_env.num_envs, device=self.device))

    def _sample_aggressive_targets(self, env_ids):
        """
        """
        n = len(env_ids)
        if n == 0:
            return
        device = self.device
        if "robot_position" in self.obs_dict:
            init_pos = self.obs_dict["robot_position"][env_ids].clone()
        else:
            init_pos = torch.zeros(n, 3, device=device)
        # random unit direction
        direction = torch.randn(n, 3, device=device)
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        # distance ∈ [min, max]; with the curriculum of S14 the range depends on the RL iteration
        lo, hi = self.aggr_dist_min, self.aggr_dist_max
        if self._levels:  # V13: success-gated level
            lo, hi = self._levels[self._level]
        elif self._curriculum:
            it = self.counter // self._horizon
            for end, clo, chi in self._curriculum:
                if it < end:
                    lo, hi = clo, chi
                    break
        dist = lo + (hi - lo) * torch.rand(n, device=device)
        offset = direction * dist.unsqueeze(1)
        self.target_position[env_ids] = init_pos + offset
        self.target_position[env_ids, 2] = torch.clamp(
            self.target_position[env_ids, 2], min=0.5
        )
        # initial distance after the clamp, for the relative termination of S15
        self._d0[env_ids] = torch.norm(self.target_position[env_ids] - init_pos, dim=1)

    def reset(self):
        super().reset()
        self._sample_aggressive_targets(
            torch.arange(self.sim_env.num_envs, device=self.device)
        )
        # observations with the new targets
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._sample_aggressive_targets(env_ids)

    def compute_rewards_and_crashes(self, obs_dict):
        if self._term_margin is None:
            rewards, crashes = super().compute_rewards_and_crashes(obs_dict)
        else:
            pos_error_vehicle_frame = quat_apply_inverse(
                obs_dict["robot_vehicle_orientation"], (self.target_position - obs_dict["robot_position"])
            )
            rewards, crashes = _reward_relative_termination(pos_error_vehicle_frame, obs_dict["robot_orientation"],
                                                            obs_dict["robot_body_angvel"], obs_dict["crashes"],
                                                            self._d0 + self._term_margin)
        if self._prog_k:
            # V6 (section 25): progress term k (d_{t-1} - d_t), zero on the first step of an episode, not on crash steps
            d = torch.norm(self.target_position - obs_dict["robot_position"], dim=1)
            new = getattr(self, "_flight_new_episode", None)
            if self._d_prev is None or new is None:
                prog = torch.zeros_like(d)
            else:
                prog = torch.where(new, torch.zeros_like(d), self._d_prev - d)
            rewards = torch.where(crashes > 0, rewards, rewards + self._prog_k * prog)
            self._d_prev = d.clone()
        return rewards, crashes

    def step(self, actions):
        out = super().step(actions)
        if self._levels:
            # V13 (section 25): success = within 0.5 m of the target at any time of the episode (reached_05, the
            # criterion of the fixed-state evaluation), counted over finished episodes
            done = (self.infos["crashes"] + self.infos["timeouts"]) > 0
            nd = int(done.sum())
            if nd:
                self._lv_n += nd
                self._lv_s += float(self.infos["flight/reached_05"][done].sum())
                if self._lv_n >= SUCC_CHECK_EPISODES:
                    rate, old = self._lv_s / self._lv_n, self._level
                    if rate > SUCC_UP:
                        self._level = min(self._level + 1, len(self._levels) - 1)
                    elif rate < SUCC_DOWN:
                        self._level = max(self._level - 1, 0)
                    if self._level != old or os.environ.get("AG_T3_SUCC_DEBUG"):
                        print(f"[T3 success curriculum] iteration {self.counter // self._horizon}: success "
                              f"{rate:.3f} over {self._lv_n} episodes, level {old} -> {self._level} "
                              f"{self._levels[self._level]}", flush=True)
                    self._lv_n, self._lv_s = 0, 0.0
        return out
