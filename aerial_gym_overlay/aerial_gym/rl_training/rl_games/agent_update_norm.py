"""Experiment D — per-step weight-update-norm logging agent.

RL-Theorist point: the catastrophic-forgetting hypothesis (LoRA wins because the
frozen backbone is preserved) and the slow-effective-LR hypothesis (LoRA / small-LR
FullFT win simply because the *magnitude* of per-step parameter movement is small)
make different predictions about the per-minibatch update norm. Logging the L2 norm
of every parameter update lets the two be told apart:
  - slow-effective-LR    => small per-step ||Δθ|| everywhere, uniformly.
  - catastrophic-forget  => backbone ||Δθ|| specifically near-zero (frozen / tiny)
                            while head ||Δθ|| is normal-sized.

This module subclasses the stock rl_games `A2CAgent` and wraps `train_actor_critic`
(the per-minibatch update entry point) to snapshot trainable weights before/after
each optimizer step and accumulate the per-step update L2 norm, split into the same
6 logical groups used by `utils/grad_monitor.py` plus a backbone/head split. Norms
are written to `<experiment_dir>/update_norm.csv` and TensorBoard.

Minimal-intrusion: this is an opt-in algo (`algo.name: a2c_continuous_update_norm`
in the yaml). Stock cells are unaffected. runner.py registers the builder.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import torch

try:
    from rl_games.algos_torch.a2c_continuous import A2CAgent
except Exception:  # pragma: no cover - import guard
    A2CAgent = object

# Backbone vs head parameter-name prefixes (PretrainAdaptPolicy layout).
_BACKBONE_PREFIXES = ["proprio_mlp", "vae_proj_mlp", "fusion_mlp", "gru"]
_HEAD_PREFIXES = ["actor_trunk", "actor_mu", "log_std", "critic_trunk", "critic_v"]
_GROUP_PREFIXES = {
    "proprio_mlp": ["proprio_mlp"],
    "vae_proj_mlp": ["vae_proj_mlp"],
    "fusion_mlp": ["fusion_mlp"],
    "gru": ["gru"],
    "actor": ["actor_trunk", "actor_mu", "log_std"],
    "critic": ["critic_trunk", "critic_v"],
}

def _matches(name: str, prefixes: List[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)

def _strip_model_prefix(name: str) -> str:
    """rl_games wraps the policy as model.a2c_network.policy.<policy params>.
    Strip leading wrapper components so the PretrainAdapt prefixes match.
    Iterate to ground-state so any ordering / nesting is handled.
    """
    changed = True
    while changed:
        changed = False
        for pre in ("model.", "_orig_mod.", "a2c_network.", "policy."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name

class A2CAgentUpdateNorm(A2CAgent):
    """Stock continuous-PPO agent + per-step weight-update-norm logging."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._un_csv_path = os.path.join(self.experiment_dir, "update_norm.csv")
        self._un_step_global = 0
        self._un_fields = (
            ["epoch", "minibatch", "global_step", "total", "backbone", "head"]
            + sorted(_GROUP_PREFIXES.keys())
        )
        # truncate / create CSV with header
        with open(self._un_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self._un_fields)
        # accumulators reset per epoch for a compact TensorBoard summary
        self._un_epoch_accum: Dict[str, float] = {}
        self._un_epoch_count = 0
        print(f"[update_norm] logging per-step weight-update L2 norm -> {self._un_csv_path}")

    # ------------------------------------------------------------------
    def _un_trainable_named_params(self):
        """Yield (clean_name, param) for trainable params of the policy model."""
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            yield _strip_model_prefix(name), p

    def _un_snapshot(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self._un_trainable_named_params()}

    def _un_delta_norms(self, before: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """L2 norm of (after - before) aggregated into groups + backbone/head/total."""
        sq: Dict[str, float] = {k: 0.0 for k in _GROUP_PREFIXES}
        sq["total"] = 0.0
        sq["backbone"] = 0.0
        sq["head"] = 0.0
        for name, p in self._un_trainable_named_params():
            if name not in before:
                continue
            d = float((p.detach() - before[name]).pow(2).sum().item())
            sq["total"] += d
            if _matches(name, _BACKBONE_PREFIXES):
                sq["backbone"] += d
            if _matches(name, _HEAD_PREFIXES):
                sq["head"] += d
            for g, prefixes in _GROUP_PREFIXES.items():
                if _matches(name, prefixes):
                    sq[g] += d
                    break
        return {k: v ** 0.5 for k, v in sq.items()}

    # ------------------------------------------------------------------
    def train_actor_critic(self, input_dict):
        """Per-minibatch PPO update; wrap to log ||Δθ|| of this exact step."""
        before = self._un_snapshot()
        result = super().train_actor_critic(input_dict)
        try:
            norms = self._un_delta_norms(before)
            self._un_step_global += 1
            mb = self._un_step_global  # running minibatch index
            row = [
                int(getattr(self, "epoch_num", 0)),
                mb,
                self._un_step_global,
                norms["total"], norms["backbone"], norms["head"],
            ] + [norms[g] for g in sorted(_GROUP_PREFIXES.keys())]
            with open(self._un_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v
                                        for v in row])
            for k, v in norms.items():
                self._un_epoch_accum[k] = self._un_epoch_accum.get(k, 0.0) + v
            self._un_epoch_count += 1
        except Exception as e:  # never let logging break training
            if self._un_step_global == 0:
                print(f"[update_norm] disabled: {e}")
        return result

    def train_epoch(self):
        result = super().train_epoch()
        # per-epoch mean of the per-step norms -> TensorBoard
        try:
            if self._un_epoch_count > 0 and getattr(self, "writer", None) is not None:
                for k, total in self._un_epoch_accum.items():
                    self.writer.add_scalar(
                        f"update_norm/{k}", total / self._un_epoch_count,
                        self.epoch_num,
                    )
            self._un_epoch_accum = {}
            self._un_epoch_count = 0
        except Exception:
            pass
        return result

def register():
    """Register the update-norm agent builder with an rl_games Runner.

    Call after constructing the Runner. Idempotent-ish — re-registering the
    same key just overwrites.
    """
    from rl_games.torch_runner import Runner  # noqa: F401  (type hint only)
    return A2CAgentUpdateNorm
