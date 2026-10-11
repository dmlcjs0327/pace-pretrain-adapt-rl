"""rl_games A2CAgent subclass that adds a per-minibatch joint SSL aux loss
.

Each PPO minibatch's gradient step adds `lambda * SSL_loss` to the standard
PPO actor-critic loss BEFORE the single backward pass — matching CURL/SPR/ATC
canonical joint-aux paradigm. The aux head is wired into the network module
via `network.set_aux_module(...)`, and `agent.calc_gradients` caches the
current minibatch input_dict on the network so its `get_aux_loss()` hook can
re-run the backbone over the temporally-structured (num_seqs, seq_length, ...)
tensor and apply the SSL loss. rl_games' `calc_gradients` then auto-adds the
returned aux loss to the total loss in the same backward — so the aux gradient
flows through the same backbone params and the same optimizer as the PPO update.

Supported aux losses (SSL-aligned, post-GRU latent matching offline pretrain):
    FD / IDM_SSL / MAE / CPC_SSL / QM
    P0  — disabled (stock PPO)

Legacy P1/P2/P3 (pre-GRU z_t, separate per-epoch step) are retained as a
historical reference path for back-compat but are NOT recommended; new work
should use the SSL-aligned per-minibatch path.

YAML usage (extend ppo_pretrain_adapt.yaml):
    algo:
      name: a2c_continuous_pretrain_adapt
    config:
      ...
      aux_loss:
        name: MAE              # FD / IDM_SSL / MAE / CPC_SSL / QM / SPR_SSL / ...
        lambda: 0.1            # joint loss weight; main grid sweeps {0.01, 0.1, 0.5, 1.0, 2.0}
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from rl_games.algos_torch import a2c_continuous

from aerial_gym.nn.aux_losses import build_aux_loss
from aerial_gym.nn.freeze_policy import apply_full_ft, apply_ibf, trainable_param_count
from aerial_gym.nn.lora import wrap_linears_with_lora
from aerial_gym.utils.grad_monitor import per_layer_grad_norms

# The pretraining losses reused as auxiliary losses during RL (Aux-SSL).
# Identical loss semantics as the offline pretrain phase, applied on rollout sequences instead.
from aerial_gym.ssl_pretrain.losses_curl import CURLLoss as SSLCURLLoss
from aerial_gym.ssl_pretrain.losses import (
    ForwardDynamicsLoss as SSLForwardDynamicsLoss,
    InverseDynamicsLoss as SSLInverseDynamicsLoss,
    MaskedStateLoss as SSLMaskedStateLoss,
    ContrastivePredictiveLoss as SSLContrastivePredictiveLoss,
    QuasiMetricLoss as SSLQuasiMetricLoss,
    SelfPredictiveLoss as SSLSPRLoss,
    PIDMLoss as SSLPIDMLoss,
    VICRegLoss as SSLVICRegLoss,
    BYOLLoss as SSLBYOLLoss,
    FusionLoss as SSLFusionLoss,
)

# Names that route to the SSL-pretrain-aligned aux path (post-GRU latent, sequence loss).
# Distinct from the legacy P1/P2/P3 names which use pre-GRU z_t (kept for back-compat).
_SSL_AUX_NAMES = {
    "FD", "IDM_SSL", "MAE", "CPC_SSL", "QM",
    "SPR_SSL", "PIDM_SSL", "VICREG_SSL", "BYOL_SSL", "FUSION_SSL",
    "CURL",  # canonical CURL InfoNCE — uses CNN visual encoder
}

class A2CAgentPretrainAdapt(a2c_continuous.A2CAgent):
    """A2CAgent + per-minibatch joint SSL aux loss (CURL/SPR-style).

    The aux loss is added to the PPO loss inside `calc_gradients`, so a single
    backward pass updates the backbone via both PPO and aux gradients. This
    differs from the legacy per-epoch design (one separate aux step per
    `train_epoch`), which under-powered the aux signal by ~30-60x relative to
    canonical implementations.
    """

    def __init__(self, base_name, params):
        super().__init__(base_name=base_name, params=params)
        cfg = params["config"].get("aux_loss", {}) or {}
        self.aux_name = (cfg.get("name") or "P0").upper()
        self.aux_lambda = float(cfg.get("lambda", 0.1))
        # λ-schedule:
        # supported: None (fixed), "linear_decay_50_500" (λ * (1 - max(0, (epoch-50)/450))).
        self.aux_lambda_schedule = cfg.get("lambda_schedule", None)
        self._aux_lambda_init = self.aux_lambda
        self.aux_lr = float(cfg.get("lr", 3.0e-4))
        self.aux_k = int(cfg.get("k_step", 4))
        # Inner policy module (PretrainAdaptPolicy) — exposes _fuse(obs) → z_t
        # Path: model._orig_mod (when torch.compile) → a2c_network → policy
        self._policy_module = self._resolve_policy_module()

        # ---- Pretrain checkpoint load (proposal §4.5) ----
        # When `pretrain_ckpt` is given, load BACKBONE weights from a previous
        # rl_games checkpoint into self._policy_module BEFORE adaptation is
        # applied. This is the proposal's pretrain → adapt sequence: the
        # adapter (Full-FT/IBF/LoRA) wraps/freezes weights that already
        # encode pretrain knowledge.
        pre_cfg = params["config"].get("pretrain", {}) or {}
        self._pretrain_ckpt_path: Optional[str] = pre_cfg.get("ckpt")
        if self._pretrain_ckpt_path:
            self._load_pretrain_backbone(self._pretrain_ckpt_path)

        # ---- Adaptation strategy (proposal §2.4 / §4.4) ----
        # Applied right after model creation so PPO optimizer sees only the
        # post-adaptation trainable params. Re-grouping the existing
        # self.optimizer (rl_games default Adam) is the simplest path.
        adapt_cfg = params["config"].get("adaptation", {}) or {}
        self.adapt_name = (adapt_cfg.get("name") or "A1_FULLFT").upper()
        if self.adapt_name in ("A1", "A1_FULLFT", "FULL_FT", "FULL-FT"):
            apply_full_ft(self._policy_module)
            wrapped = []
        elif self.adapt_name in ("A2", "A2_IBF", "IBF"):
            frozen = apply_ibf(self._policy_module)
            wrapped = []
            print(f"[adapt=A2 IBF] frozen modules: {frozen}")
        elif self.adapt_name in ("A3", "A3_LORA", "LORA"):
            r = int(adapt_cfg.get("lora_r", 8))
            alpha = float(adapt_cfg.get("lora_alpha", 16.0))
            target = adapt_cfg.get("lora_target") or [
                "fusion_mlp.0", "fusion_mlp.2",
                "actor_trunk.0", "critic_trunk.0",
            ]
            # Freeze whole policy first; wrap_linears_with_lora un-freezes only the LoRA deltas
            for p in self._policy_module.parameters():
                p.requires_grad_(False)
            wrapped = wrap_linears_with_lora(
                self._policy_module, target, r=r, alpha=alpha, freeze_base=True,
            )
            # rl_games' optimizer was created over self.model.parameters() before
            # wrap; rebuild the param groups so LoRA params are included and
            # frozen base params are excluded.
            self.optimizer = self._rebuild_ppo_optimizer()
            print(f"[adapt=A3 LoRA r={r}] wrapped={wrapped}  trainable={trainable_param_count(self._policy_module):,}")
        else:
            raise ValueError(f"unknown adaptation.name: {self.adapt_name}")
        self._adapt_wrapped = wrapped

        if self.aux_name == "P0":
            self.aux_head: Optional[nn.Module] = None
            self.aux_optim: Optional[torch.optim.Optimizer] = None
            self._wire_aux_into_network(None)
            return

        z_dim = self._policy_module.cfg.fusion_hidden
        gru_dim = self._policy_module.cfg.gru_hidden
        action_dim = self._policy_module.cfg.action_dim
        proprio_dim = self._policy_module.cfg.proprio_dim
        if self.aux_name in ("P1", "IDM"):
            self.aux_head = build_aux_loss("IDM", z_dim=z_dim, action_dim=action_dim)
        elif self.aux_name in ("P2", "CPC", "CPCA"):
            self.aux_head = build_aux_loss("CPC", z_dim=z_dim, action_dim=action_dim, k_step=self.aux_k)
        elif self.aux_name in ("P3", "RP"):
            self.aux_head = build_aux_loss("RP", z_dim=z_dim, k_step=3)
        # ---- SSL-aligned aux heads (post-GRU latent, matching offline pretrain) ----
        elif self.aux_name == "FD":
            self.aux_head = SSLForwardDynamicsLoss(hidden_dim=gru_dim, action_dim=action_dim)
        elif self.aux_name == "IDM_SSL":
            self.aux_head = SSLInverseDynamicsLoss(hidden_dim=gru_dim, action_dim=action_dim)
        elif self.aux_name == "MAE":
            self.aux_head = SSLMaskedStateLoss(hidden_dim=gru_dim, state_dim=proprio_dim, mask_ratio=0.3)
        elif self.aux_name == "CPC_SSL":
            self.aux_head = SSLContrastivePredictiveLoss(
                hidden_dim=gru_dim, predict_steps=(1, 2, 4, 8), temperature=0.1,
            )
        elif self.aux_name == "QM":
            self.aux_head = SSLQuasiMetricLoss(hidden_dim=gru_dim, proj_dim=64, n_pairs_per_seq=16)
        elif self.aux_name == "SPR_SSL":
            self.aux_head = SSLSPRLoss(hidden_dim=gru_dim, action_dim=action_dim)
        elif self.aux_name == "PIDM_SSL":
            self.aux_head = SSLPIDMLoss(hidden_dim=gru_dim, action_dim=action_dim)
        elif self.aux_name == "VICREG_SSL":
            self.aux_head = SSLVICRegLoss(hidden_dim=gru_dim)
        elif self.aux_name == "BYOL_SSL":
            self.aux_head = SSLBYOLLoss(hidden_dim=gru_dim)
        elif self.aux_name == "FUSION_SSL":
            self.aux_head = SSLFusionLoss(hidden_dim=gru_dim, action_dim=action_dim, state_dim=proprio_dim)
        elif self.aux_name == "CURL":
            # CURL InfoNCE on backbone latent — canonical Laskin 2020 form.
            # Two augmented views of the same observation are forwarded
            # through the (CNN+proprio) backbone; bilinear InfoNCE pulls
            # matched (query, key) pairs together. Requires the backbone
            # to have a trainable visual encoder (use_cnn_vision_branch=True)
            # for the canonical paradigm to deliver its strength — otherwise
            # the aux gradient cannot reach the visual encoder.
            self.aux_head = SSLCURLLoss(hidden_dim=gru_dim, temperature=0.1)
        else:
            raise ValueError(f"unknown aux_loss.name: {self.aux_name}")
        self.aux_head = self.aux_head.to(self.ppo_device)

        # Per-minibatch joint paradigm: aux head is wired into the network so
        # rl_games' calc_gradients adds aux_loss to PPO loss in one backward pass.
        # No separate aux_optim is needed — backbone gets aux gradient from the
        # MAIN PPO optimizer (which sees aux_head params via the wiring below).
        if self.aux_name in _SSL_AUX_NAMES:
            self._wire_aux_into_network(self.aux_head)
            # Add aux head params to the main PPO optimizer so they receive updates
            self.optimizer = self._rebuild_ppo_optimizer_with_aux()
            self.aux_optim = None
            print(f"[aux={self.aux_name} per-minibatch joint] head_params={sum(p.numel() for p in self.aux_head.parameters()):,}  λ={self.aux_lambda}")

            # ---- EMA target backbone for canonical SPR/BYOL (Schwarzer 2021,
            # Grill 2020) — fixes the degenerate `z_target = z_full.detach()`
            # variant that empirically reduced AuxJoint(SPR/BYOL) to ≈ Scratch
            # on the proprio main grid.
            # VICReg gets augmentation-pair views (no EMA needed). All proprio
            # state augmentation = Gaussian noise σ=0.05 on the 13-D proprio
            # subvector; vision/VAE dims of the observation are passed through
            # unchanged (depth augmentation lives in the CURL-style pipeline).
            if self.aux_name in ("SPR_SSL", "BYOL_SSL", "CURL"):
                # CURL uses a momentum key encoder (He 2020 MoCo / Laskin 2020).
                # SPR/BYOL use an EMA target encoder. All three benefit from
                # a stop-grad target, so we share the EMA target backbone code.
                self._ema_target_backbone = copy.deepcopy(self._policy_module).to(self.ppo_device)
                for p in self._ema_target_backbone.parameters():
                    p.requires_grad = False
                self._ema_target_backbone.eval()
                # Decay: CURL uses τ=0.05 in original (very fast), SPR/BYOL use 0.99.
                # We default to 0.99 for all; CURL's faster decay is a design knob
                # we leave for follow-up tuning rather than hard-coding here.
                self._ema_decay = 0.99
                self._wire_target_backbone_into_network(self._ema_target_backbone)
                print(f"[aux={self.aux_name}] EMA target backbone built (decay={self._ema_decay})")
            else:
                self._ema_target_backbone = None
                self._ema_decay = None
        else:
            # Legacy P1/P2/P3 (kept for back-compat): separate optimizer, per-epoch step.
            self._wire_aux_into_network(None)
            backbone_modules = [self._policy_module.proprio_mlp, self._policy_module.fusion_mlp]
            if self._policy_module.vae_proj_mlp is not None:
                backbone_modules.append(self._policy_module.vae_proj_mlp)
            backbone = []
            for m in backbone_modules:
                backbone += [p for p in m.parameters() if p.requires_grad]
            aux_params = list(self.aux_head.parameters()) + backbone
            self.aux_optim = torch.optim.Adam(aux_params, lr=self.aux_lr) if aux_params else None
            print(f"[aux={self.aux_name} legacy per-epoch] head_params={sum(p.numel() for p in self.aux_head.parameters()):,}")

    # ------------------------------------------------------------------

    def _rebuild_ppo_optimizer(self) -> torch.optim.Optimizer:
        """Recreate rl_games' main optimizer over CURRENT trainable params.

        Used after LoRA wrapping replaces submodules so the param tree changes.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Fall back to adam with the current learning_rate
        return torch.optim.Adam(params, lr=float(self.last_lr), eps=1e-08)

    def _rebuild_ppo_optimizer_with_aux(self) -> torch.optim.Optimizer:
        """Like _rebuild_ppo_optimizer but ALSO includes aux_head params.

        Used in the per-minibatch joint paradigm so that the aux head receives
        gradient updates from the same backward as the PPO loss. Avoids
        duplicating params already in self.model.parameters().
        """
        seen = set()
        params = []
        for p in self.model.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        if self.aux_head is not None:
            for p in self.aux_head.parameters():
                if id(p) not in seen:
                    p.requires_grad_(True)
                    params.append(p)
                    seen.add(id(p))
        return torch.optim.Adam(params, lr=float(self.last_lr), eps=1e-08)

    def _wire_target_backbone_into_network(self, target_bb: nn.Module) -> None:
        """Inject the EMA target backbone into the rl_games Network so its
        get_aux_loss() hook can use it for canonical SPR/BYOL (forward the
        augmented obs through the target backbone, stop-grad)."""
        m = self.model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        net = getattr(m, "a2c_network", None)
        if net is not None and hasattr(net, "set_aux_target_backbone"):
            net.set_aux_target_backbone(target_bb)

    @torch.no_grad()
    def _ema_update_target(self) -> None:
        """Exponential-moving-average update of target backbone from online
        backbone (Schwarzer 2021 / Grill 2020). Called once per epoch."""
        if getattr(self, "_ema_target_backbone", None) is None:
            return
        decay = self._ema_decay
        for p_t, p_o in zip(self._ema_target_backbone.parameters(), self._policy_module.parameters()):
            p_t.data.mul_(decay).add_(p_o.data, alpha=1.0 - decay)
        for b_t, b_o in zip(self._ema_target_backbone.buffers(), self._policy_module.buffers()):
            b_t.data.copy_(b_o.data)

    def _wire_aux_into_network(self, head: Optional[nn.Module]) -> None:
        """Inject aux head + name + lambda into the rl_games Network so its
        get_aux_loss() hook (called per minibatch by calc_gradients) returns
        a non-None loss tensor that rl_games adds to the PPO loss.
        """
        # self.model wraps the Network (a2c_continuous models.py:78). Walk
        # through torch.compile / a2c_network wrappers to find the Network.
        m = self.model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        net = getattr(m, "a2c_network", None)
        if net is None or not hasattr(net, "set_aux_module"):
            print("[aux] target network has no set_aux_module(); per-minibatch aux disabled")
            return
        net.set_aux_module(head, self.aux_name if head is not None else "P0", self.aux_lambda)

    # ------------------------------------------------------------------

    def _load_pretrain_backbone(self, ckpt_path: str) -> None:
        """Load PretrainAdaptPolicy weights from an rl_games checkpoint.

        Strips rl_games' wrapping prefixes ([_orig_mod.]a2c_network.policy.*)
        and loads only the inner policy state. Adaptation (LoRA/IBF) is
        applied afterwards so the freeze/wrap sees the loaded weights.
        Also restores `running_mean_std` (obs normalizer) when present, so
        the adapt phase sees the SAME input distribution the pretrain phase
        was working in.

        ALSO restores PPO Adam optimizer state (momentum + variance estimates)
        from the pretrain ckpt. Without this, the fresh Adam optimizer
        produces a giant first update (un-tempered by accumulated 2nd moment
        statistics) that *catastrophically reshapes the pretrained backbone
        in the first ~50 epochs* — empirically observed in cell 1 (D1
        P0/A1/seed42): backbone L2 drift saturated by ep_50 (max|delta| ≈
        0.4-1.1 on weight tensors of magnitude ~1, then stayed frozen for
        the rest of 1144 epoch run while only actor/critic continued
        learning). Loading the pretrain Adam state preserves the smooth
        adaptation trajectory that transfer learning depends on.

        Optimizer load is best-effort: shape-mismatch (LoRA/IBF reshape the
        param tree) → silent skip → fresh optimizer (acceptable since those
        adapt strategies inherently limit backbone update magnitude via
        explicit freezing).
        """
        from pathlib import Path

        sd = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
        full_sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd

        PREFIXES = [
            "_orig_mod.a2c_network.policy.",
            "a2c_network.policy.",
            "policy.",
        ]
        inner = {}
        norm_state = {}
        for k, v in full_sd.items():
            for p in PREFIXES:
                if k.startswith(p):
                    inner[k[len(p):]] = v
                    break
            for sub in ("running_mean_std.", "value_mean_std."):
                # match `_orig_mod.running_mean_std.<rest>` or just `running_mean_std.<rest>`
                idx = k.find(sub)
                if idx != -1:
                    norm_state[k[idx:]] = v

        missing, unexpected = self._policy_module.load_state_dict(inner, strict=False)
        print(f"[pretrain] loaded {len(inner)} backbone tensors  missing={len(missing)}  unexpected={len(unexpected)}  src={Path(ckpt_path).name}")
        if missing:
            print(f"[pretrain] (first missing keys: {list(missing)[:3]})")

        # Restore obs/value normalizer state when both source and target use
        # `normalize_input=True` so adapt-phase observations land in the
        # distribution the pretrained policy expects.
        if hasattr(self.model, "running_mean_std") and self.model.running_mean_std is not None:
            try:
                tgt_state = self.model.running_mean_std.state_dict()
                ckpt_state = {k.replace("running_mean_std.", ""): v for k, v in norm_state.items() if k.startswith("running_mean_std.")}
                if ckpt_state and set(ckpt_state).issubset(set(tgt_state)):
                    self.model.running_mean_std.load_state_dict(ckpt_state, strict=False)
                    print(f"[pretrain] restored running_mean_std ({len(ckpt_state)} tensors)")
            except Exception as e:
                print(f"[pretrain] running_mean_std restore skipped: {e}")

        # Restore PPO Adam optimizer state (momentum + variance) — critical for
        # avoiding fresh-optimizer shock (proposal §4.5: pretrain → adapt
        # representation continuity).
        if isinstance(sd, dict) and "optimizer" in sd:
            try:
                self.optimizer.load_state_dict(sd["optimizer"])
                # Sanity: Adam momentum tensors exist for at least one param
                states_with_momentum = sum(
                    1 for s in self.optimizer.state.values()
                    if isinstance(s, dict) and "exp_avg" in s
                )
                print(f"[pretrain] restored PPO Adam optimizer state "
                      f"({states_with_momentum} param groups with momentum)")
            except Exception as e:
                print(f"[pretrain] optimizer state restore skipped: {e} "
                      f"(fresh optimizer will be used — expected for LoRA/IBF reshape)")
        else:
            print(f"[pretrain] no 'optimizer' key in ckpt — using fresh optimizer")

    def _resolve_policy_module(self):
        m = self.model
        # Walk through possible wrappers: torch.compile (_orig_mod), then a2c_network, then policy
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        net = getattr(m, "a2c_network", m)
        if not hasattr(net, "policy"):
            raise RuntimeError("Pretrain-adapt aux loss requires PretrainAdaptBuilder network (custom)")
        return net.policy

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def calc_gradients(self, input_dict):
        """Cache minibatch input_dict on the network so its get_aux_loss()
        hook (invoked by super().calc_gradients) can read the temporally
        structured (num_seqs * seq_length, ...) tensors.

        rl_games' prepare_dataset (a2c_common.py:1020-1066)
        builds dataset_dict (= input_dict here) WITHOUT a `seq_length` key.
        Without injection, get_aux_loss reads seq_len=1 and early-returns None
        on every minibatch, so the aux loss never enters the backward pass —
        the symptom is identical PPO rewards across (P, lambda) for the same
        seed.
        Inject self.seq_length (PPO RNN minibatch length) so the reshape into
        (num_seqs, seq_length, obs_dim) inside get_aux_loss works.
        """
        m = self.model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        net = getattr(m, "a2c_network", None)
        if net is not None and hasattr(net, "_aux_input_dict"):
            if self.is_rnn and input_dict.get("seq_length") is None:
                input_dict["seq_length"] = self.seq_length
            net._aux_input_dict = input_dict
        return super().calc_gradients(input_dict)

    # ------------------------------------------------------------------

    def _aux_step(self) -> Optional[float]:
        """Legacy per-epoch aux step (kept ONLY for back-compat with P1/P2/P3
        names that target the pre-GRU latent). SSL-aligned aux paths (FD,
        IDM_SSL, MAE, CPC_SSL, QM, SPR_SSL, PIDM_SSL, VICReg_SSL, BYOL_SSL,
        FUSION_SSL) now run per-minibatch via the network's get_aux_loss hook
        and SHOULD NOT enter this code path.
        """
        if self.aux_name == "P0" or self.aux_head is None:
            return None
        if self.aux_name in _SSL_AUX_NAMES:
            # SSL aux is handled per-minibatch in calc_gradients; nothing to do here.
            return None
        td = self.experience_buffer.tensor_dict
        obses = td["obses"]                  # [T, B, obs_dim]
        actions = td["actions"]              # [T, B, act_dim]
        rewards = td["rewards"]              # [T, B, 1] or [T, B]
        if rewards.dim() == 3:
            rewards = rewards.squeeze(-1)
        T, B = obses.shape[0], obses.shape[1]

        if self.normalize_input and hasattr(self.model, "running_mean_std"):
            # Apply same input normalization used for PPO so z_t matches what actor sees
            obs_flat = obses.reshape(T * B, -1)
            with torch.no_grad():
                obs_flat = self.model.norm_obs(obs_flat) if hasattr(self.model, "norm_obs") else obs_flat
            obses_norm = obs_flat.reshape(T, B, -1)
        else:
            obses_norm = obses

        # ---- Legacy aux path (P1/P2/P3): pre-GRU latent ----
        # Forward all timesteps through the pre-recurrent backbone
        flat = obses_norm.reshape(T * B, -1)
        z_flat = self._policy_module._fuse(flat)
        z_seq = z_flat.reshape(T, B, -1)

        if self.aux_name in ("P1", "IDM"):
            # pair (t, t+1) within same env
            z_t = z_seq[:-1].reshape(-1, z_seq.shape[-1])
            z_tp1 = z_seq[1:].reshape(-1, z_seq.shape[-1])
            a_t = actions[:-1].reshape(-1, actions.shape[-1])
            loss = self.aux_head(z_t, z_tp1, a_t)

        elif self.aux_name in ("P2", "CPC", "CPCA"):
            k = self.aux_k
            if T <= k:
                return None
            z_anchor = z_seq[: T - k].reshape(-1, z_seq.shape[-1])
            # pair k-step future actions
            future_actions = torch.stack([actions[i:i + k] for i in range(T - k)], dim=0)  # [T-k, k, B, A]
            future_actions = future_actions.permute(0, 2, 1, 3).reshape(-1, k, actions.shape[-1])
            z_target = z_seq[k:].reshape(-1, z_seq.shape[-1])
            loss = self.aux_head(z_anchor, future_actions, z_target)

        elif self.aux_name in ("P3", "RP"):
            rk = 3
            if T <= rk:
                return None
            z_anchor = z_seq[: T - rk].reshape(-1, z_seq.shape[-1])
            future_rewards = torch.stack([rewards[i:i + rk] for i in range(T - rk)], dim=0)  # [T-rk, rk, B]
            future_rewards = future_rewards.permute(0, 2, 1).reshape(-1, rk)
            loss = self.aux_head(z_anchor, future_rewards)
        else:
            return None

        # Apply λ schedule (e.g. "linear_decay_50_500": λ_init for ep≤50, linear to 0 by ep=500)
        eff_lambda = self.aux_lambda
        if self.aux_lambda_schedule == "linear_decay_50_500":
            ep = getattr(self, "epoch_num", 0)
            if ep <= 50:
                eff_lambda = self._aux_lambda_init
            elif ep < 500:
                eff_lambda = self._aux_lambda_init * (1.0 - (ep - 50) / 450.0)
            else:
                eff_lambda = 0.0
        scaled = eff_lambda * loss
        self.aux_optim.zero_grad()
        scaled.backward()
        # Gradient clipping (mirror PPO grad_norm)
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.aux_optim.param_groups for p in g["params"]],
                self.grad_norm,
            )
        self.aux_optim.step()
        return float(loss.detach().item())

    # ------------------------------------------------------------------

    def train_epoch(self):
        # Run stock PPO epoch first. With SSL-aligned aux, super().train_epoch()
        # itself adds aux_loss to PPO loss every minibatch via the model's
        # get_aux_loss() hook. With legacy P1/P2/P3, _aux_step() below runs
        # one separate per-epoch step (back-compat).
        result = super().train_epoch()
        # Canonical SPR/BYOL: EMA-update the target backbone after the PPO
        # epoch (Schwarzer 2021 / Grill 2020). No-op when target backbone
        # is None (i.e. all other aux modes).
        self._ema_update_target()
        aux_loss_val = self._aux_step()
        try:
            norms = per_layer_grad_norms(self._policy_module)
            if hasattr(self, "writer") and self.writer is not None:
                for k, v in norms.items():
                    self.writer.add_scalar(k, v, self.epoch_num)
            if norms.get("grad_norm/total", 0.0) > 10.0 and self.epoch_num % 10 == 0:
                print(f"[grad_warn] total={norms['grad_norm/total']:.2f} (>10) at epoch={self.epoch_num}")
        except Exception as e:
            if self.epoch_num == 1:
                print(f"[grad_monitor] disabled: {e}")
        # Periodic logging: SSL per-minibatch aux exposes itself via rl_games'
        # self.aux_loss_dict; legacy per-epoch aux returns the loss directly.
        if self.epoch_num % 10 == 0:
            if aux_loss_val is not None:
                print(f"[aux={self.aux_name} legacy] epoch={self.epoch_num} aux_loss={aux_loss_val:.4f}")
            elif self.aux_name in _SSL_AUX_NAMES:
                tracked = getattr(self, "aux_loss_dict", {})
                if tracked:
                    parts = []
                    for k, v in tracked.items():
                        try:
                            val = float(v) if not isinstance(v, list) else float(sum(v) / len(v))
                            parts.append(f"{k}={val:.4f}")
                        except Exception:
                            pass
                    if parts:
                        print(f"[aux={self.aux_name} joint] epoch={self.epoch_num} {' '.join(parts)}")
        return result

def register():
    """Register the custom algo with rl_games. Idempotent."""
    # rl_games' Runner registers algos in its own factory; the standard pattern
    # is to register from the runner.py side. We expose a no-op here so callers
    # can `from agent_pretrain_adapt import register` and call it.
    pass
