"""Regularizers and auxiliary losses added to the PPO loss.

Used through the get_aux_loss hook that rl-games calls once per PPO minibatch. Modes (network config reg_mode):
  l2sp     (lambda / 2) * sum ||theta - theta_0||^2 over the backbone (encoder and GRU), where theta_0 are the
           weights at the start of RL (L2-SP, Li et al. 2018, alpha = lambda; for a random backbone this is the
           L2-init penalty)
  aux_ssl  lambda * the pretraining loss of the backbone's objective, computed on windows of the offline flight
           data with the head of the pretraining checkpoint, which keeps training (SGI, Schwarzer et al. 2021)
  bc_reg   lambda * squared error between the actor mean on offline windows and the recorded commands
           (behavior-cloning regularization of fine-tuning, Wolczyk et al. 2024)
The offline windows are 16,384 windows of 33 steps of the v3 data of R2 and R3 (fixed seed), standardized with the
fixed statistics, as in pretraining; each call uses 256 of them.
"""
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _os.environ.get("PACE_DATA", "/opt/pretrain-then-adapt-rl-small-networks/data")
import copy

import h5py
import numpy as np
import torch
import torch.nn as nn

DATA = _DATA_ROOT + "/v3/"
FILES = [("ssl_v3_x500.h5", "x500"), ("ssl_v3_lmf2.h5", "lmf2_proprio_clean")]
WIN, N_WIN, BATCH = 33, 16384, 256

def load_windows(stats, seed=12345):
    rng = np.random.default_rng(seed)
    S, A = [], []
    for fn, g in FILES:
        with h5py.File(DATA + fn, "r") as f:
            s = f[g]["states"][:].reshape(2700, 4096, 13)
            a = f[g]["actions"][:].reshape(2700, 4096, 4)
        ok = np.arange(2700 - WIN + 1)
        ok = ok[(ok % 300) <= 300 - WIN]
        ts, ns = rng.choice(ok, N_WIN // 2), rng.integers(0, 4096, N_WIN // 2)
        S.append(np.stack([s[t:t + WIN, n] for t, n in zip(ts, ns)]))
        A.append(np.stack([a[t:t + WIN, n] for t, n in zip(ts, ns)]))
    S, A = np.concatenate(S), np.concatenate(A)
    xs = (S - np.array(stats["state_mean"])) / np.array(stats["state_std"])
    us = (A - np.array(stats["action_mean"])) / np.array(stats["action_std"])
    return (torch.tensor(np.clip(xs, -5, 5), dtype=torch.float32), torch.tensor(us, dtype=torch.float32),
            torch.tensor(A, dtype=torch.float32))

def backbone_names(policy):
    return [n for n, _ in policy.named_parameters() if n.startswith(("proprio_mlp.", "fusion_mlp.", "gru."))]

class PaceRegularizer(nn.Module):
    def __init__(self, policy, mode, lam, ckpt=None, stats_path=None):
        super().__init__()
        self.mode, self.lam = mode, float(lam)
        self._policy = [policy]  # not a submodule: its parameters belong to the policy
        if mode == "l2sp":
            self.names = backbone_names(policy)
            params = dict(policy.named_parameters())
            for i, n in enumerate(self.names):
                self.register_buffer(f"theta0_{i}", params[n].detach().clone())
            self.norm0 = sum(float((params[n].detach() ** 2).sum()) for n in self.names)
            return
        stats = torch.load(stats_path, map_location="cpu")
        xs, us, ar = load_windows(stats)
        # Not saved with the model: rebuilt deterministically from the data with a fixed seed,
        # and 16,384 windows would add about 45 MB to every checkpoint.
        self.register_buffer("xs", xs, persistent=False)
        self.register_buffer("us", us, persistent=False)
        self.register_buffer("ar", ar, persistent=False)
        if mode == "aux_ssl":
            from aerial_gym.ssl_pretrain.losses import (ContrastivePredictiveLoss, MaskedStateLoss, PIDMLoss,
                                                        SelfPredictiveLoss)
            from aerial_gym.ssl_pretrain.extra_losses import ForwardDynamicsLoss, InverseDynamicsLoss
            ck = torch.load(ckpt, map_location="cpu")
            self.objective = ck["objective"]
            self.mask_unit, self.mask_ratio = ck.get("mask_unit", "entry"), float(ck.get("mask_ratio", 0.3))
            self.head = {"MAE": lambda: MaskedStateLoss(128, 13, mask_ratio=0.3),
                         "CPC": lambda: ContrastivePredictiveLoss(128, predict_steps=(1, 2, 4, 8), temperature=0.1),
                         "SPR": lambda: SelfPredictiveLoss(128, 4, predict_steps=(1, 2, 3, 5)),
                         "PIDM": lambda: PIDMLoss(128, 4, predict_steps=4),
                         "IDM": lambda: InverseDynamicsLoss(128, 4),
                         "FDM": lambda: ForwardDynamicsLoss(128, 4, 13)}[self.objective]()
            if "head_state_dict" in ck:
                self.head.load_state_dict(ck["head_state_dict"])
            self.target = None
            if self.objective == "SPR":
                self.target = nn.ModuleDict({k: copy.deepcopy(getattr(policy, k)) for k in ("proprio_mlp", "fusion_mlp", "gru")})
                for p in self.target.parameters():
                    p.requires_grad = False

    @staticmethod
    def _feats(mods, x):
        h = mods["fusion_mlp"](mods["proprio_mlp"](x))
        z, _ = mods["gru"](h)
        return z

    def forward(self):
        pol = self._policy[0]
        if self.mode == "l2sp":
            params = dict(pol.named_parameters())
            num = sum(((params[n] - getattr(self, f"theta0_{i}")) ** 2).sum() for i, n in enumerate(self.names))
            return 0.5 * self.lam * num  # (alpha / 2) ||theta - theta_0||^2, as in L2-SP
        idx = torch.randint(0, self.xs.shape[0], (BATCH,), device=self.xs.device)
        x, u, ar = self.xs[idx], self.us[idx], self.ar[idx]
        mods = {k: getattr(pol, k) for k in ("proprio_mlp", "fusion_mlp", "gru")}
        if self.mode == "bc_reg":
            z = self._feats(mods, x)
            return self.lam * ((pol.actor_mu(pol.actor_trunk(z)) - ar) ** 2).mean()
        o = self.objective
        if o == "MAE":
            r = self.mask_ratio if self.mask_ratio >= 0 else float(np.random.choice([0.15, 0.3, 0.5, 0.75]))
            if self.mask_unit == "entry":
                mask = torch.rand(x.shape, device=x.device) < r
            else:
                mask = (torch.rand(x.shape[:2], device=x.device) < r).unsqueeze(-1).expand_as(x)
            loss = self.head.forward_with_z_full(self._feats(mods, x * (~mask).float()), x, mask)
        elif o == "SPR":
            z = self._feats(mods, x)
            with torch.no_grad():
                zt = self._feats(self.target, x)
            loss = self.head.forward_with_z_full(z, zt, u)
            with torch.no_grad():
                for k in ("proprio_mlp", "fusion_mlp", "gru"):
                    for pt, po in zip(self.target[k].parameters(), mods[k].parameters()):
                        pt.mul_(0.99).add_(po.detach(), alpha=0.01)
        elif o in ("PIDM", "IDM"):
            loss = self.head.forward_with_z_full(self._feats(mods, x), u)
        elif o == "FDM":
            loss = self.head.forward_with_z_full(self._feats(mods, x), x, u)
        else:  # CPC
            loss = self.head.forward_with_z_full(self._feats(mods, x))
        return self.lam * loss
