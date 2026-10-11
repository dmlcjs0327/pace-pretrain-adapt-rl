"""Auxiliary losses for pretrain-adapt experiments (proposal §4.3).

Three framework-agnostic auxiliary losses that consume the fusion latent
`z_t` exposed by `PretrainAdaptPolicy.forward_with_extras`:

- IDMLoss   (P1): inverse dynamics — predict a_t from (z_t, z_{t+1}).
- CPCALoss  (P2): action-conditional InfoNCE — predict z_{t+k} from
  (z_t, a_t..a_{t+k-1}) via contrastive temperature τ.
- RPLoss    (P3): UNREAL-style reward prediction — predict r_{t+1..k}
  from z_t with magnitude (MSE) + sign (3-class CE) heads.

Each module:
  - takes anchor latents `z_t` (and optional future tensors)
  - returns a scalar tensor (with `.backward()` available)
  - exposes a `.head_params()` helper for separate-checkpoint save (W9)

The losses are computed framework-agnostically, so the same instance can
be plugged into rl_games' `train_actor_critic` hook OR Sample Factory's
custom Learner `_calculate_losses`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# P1 — Inverse Dynamics Model (Pathak 2017 ICM; Hansen 2021 PAD)
# ---------------------------------------------------------------------------

@dataclass
class IDMConfig:
    z_dim: int = 128
    action_dim: int = 4
    hidden: int = 128
    activation: str = "elu"

class IDMLoss(nn.Module):
    """Predict a_t from (z_t, z_{t+1}) via MSE.

    Loss = mean ||h_phi(concat(z_t, z_{t+1})) - a_t||^2.
    """

    def __init__(self, cfg: IDMConfig):
        super().__init__()
        self.cfg = cfg
        act = nn.ELU if cfg.activation == "elu" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(cfg.z_dim * 2, cfg.hidden),
            act(),
            nn.Linear(cfg.hidden, cfg.action_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_tp1: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs shape [B, ...]. Output: scalar loss."""
        x = torch.cat([z_t, z_tp1], dim=-1)
        a_hat = self.net(x)
        return F.mse_loss(a_hat, actions)

    def head_params(self) -> Iterable[nn.Parameter]:
        return self.net.parameters()

# ---------------------------------------------------------------------------
# P2 — CPC|A: action-conditional InfoNCE (van den Oord 2018; Ye 2020)
# ---------------------------------------------------------------------------

@dataclass
class CPCAConfig:
    z_dim: int = 128
    action_dim: int = 4
    context_hidden: int = 128
    proj_dim: int = 64
    k_step: int = 1               # 1 or 4 in proposal
    temperature: float = 0.1
    max_batch: int = 4096          # cap for InfoNCE logits to avoid OOM (B^2 memory)

class CPCALoss(nn.Module):
    """Action-conditional contrastive prediction.

    Context c_t = g_rho(z_t, a_t..a_{t+k-1}) → projected by W_k → compared
    against projector(z_{t+k}) for both the positive (matching B index) and
    negatives (the rest of the batch).

    Loss = InfoNCE over the batch dimension.
    """

    def __init__(self, cfg: CPCAConfig):
        super().__init__()
        self.cfg = cfg
        # context encoder takes [z_t, flattened k actions]
        ctx_in = cfg.z_dim + cfg.action_dim * cfg.k_step
        self.context_encoder = nn.Sequential(
            nn.Linear(ctx_in, cfg.context_hidden),
            nn.ELU(),
            nn.Linear(cfg.context_hidden, cfg.context_hidden),
        )
        # projector to a smaller similarity space
        self.target_projector = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.proj_dim),
            nn.ELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
        )
        # transition matrix W_k (one shared across timesteps in the simple variant)
        self.W_k = nn.Parameter(torch.empty(cfg.context_hidden, cfg.proj_dim))
        nn.init.kaiming_uniform_(self.W_k, a=5 ** 0.5)

    def forward(
        self,
        z_t: torch.Tensor,                  # [B, z_dim]
        future_actions: torch.Tensor,        # [B, k, action_dim]
        z_tpk: torch.Tensor,                 # [B, z_dim]
    ) -> torch.Tensor:
        B = z_t.shape[0]
        # Subsample to keep [B, B] InfoNCE logits below cfg.max_batch^2 entries.
        if B > self.cfg.max_batch:
            idx = torch.randperm(B, device=z_t.device)[: self.cfg.max_batch]
            z_t = z_t[idx]
            future_actions = future_actions[idx]
            z_tpk = z_tpk[idx]
            B = self.cfg.max_batch
        a_flat = future_actions.reshape(B, -1)  # [B, k*action_dim]
        ctx = self.context_encoder(torch.cat([z_t, a_flat], dim=-1))   # [B, ctx]
        target = self.target_projector(z_tpk)                          # [B, proj]
        proj_ctx = ctx @ self.W_k                                       # [B, proj]
        logits = proj_ctx @ target.T / self.cfg.temperature             # [B, B]
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)

    def head_params(self) -> Iterable[nn.Parameter]:
        for p in self.context_encoder.parameters():
            yield p
        for p in self.target_projector.parameters():
            yield p
        yield self.W_k

# ---------------------------------------------------------------------------
# P3 — Reward Prediction (UNREAL — Jaderberg 2017)
# ---------------------------------------------------------------------------

@dataclass
class RPConfig:
    z_dim: int = 128
    hidden: int = 64
    k_step: int = 3               # predict next-k rewards
    sign_eps: float = 0.05        # threshold for positive/negative class boundary

class RPLoss(nn.Module):
    """Reward prediction with dual head (magnitude MSE + sign classification).

    Sign classes: 0 = negative (r < -eps), 1 = zero (-eps ≤ r ≤ eps),
    2 = positive (r > eps). MSE + sign-CE losses summed.
    """

    def __init__(self, cfg: RPConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.hidden),
            nn.ELU(),
        )
        self.magnitude_head = nn.Linear(cfg.hidden, cfg.k_step)
        self.sign_head = nn.Linear(cfg.hidden, cfg.k_step * 3)

    def _sign_class(self, r: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.sign_eps
        cls = torch.zeros_like(r, dtype=torch.long)
        cls = torch.where(r > eps, torch.full_like(cls, 2), cls)
        cls = torch.where(r < -eps, torch.full_like(cls, 0), cls)
        cls = torch.where(r.abs() <= eps, torch.full_like(cls, 1), cls)
        return cls

    def forward(
        self,
        z_t: torch.Tensor,                  # [B, z_dim]
        future_rewards: torch.Tensor,        # [B, k]
    ) -> torch.Tensor:
        B = z_t.shape[0]
        h = self.trunk(z_t)
        mag = self.magnitude_head(h)                                   # [B, k]
        sign_logits = self.sign_head(h).reshape(B, self.cfg.k_step, 3) # [B, k, 3]
        sign_target = self._sign_class(future_rewards)                  # [B, k]
        # MSE on magnitude + cross-entropy on sign
        mse_loss = F.mse_loss(mag, future_rewards)
        sign_loss = F.cross_entropy(
            sign_logits.reshape(-1, 3),
            sign_target.reshape(-1),
        )
        return mse_loss + sign_loss

    def head_params(self) -> Iterable[nn.Parameter]:
        return self.parameters()

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_aux_loss(name: str, **overrides) -> nn.Module:
    """Construct an aux loss by name. `overrides` map onto the dataclass fields."""
    name = name.upper()
    if name in ("P1", "IDM"):
        cfg = IDMConfig(**overrides)
        return IDMLoss(cfg)
    if name in ("P2", "CPC", "CPCA", "CPC|A"):
        cfg = CPCAConfig(**overrides)
        return CPCALoss(cfg)
    if name in ("P3", "RP"):
        cfg = RPConfig(**overrides)
        return RPLoss(cfg)
    raise ValueError(f"unknown aux loss: {name}")
