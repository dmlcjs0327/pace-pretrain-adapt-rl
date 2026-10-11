"""Freezing helpers for adaptation strategies (proposal §2.4 / §4.4).

A1 Full-FT — every param trainable (no-op via `unfreeze_all`).
A2 IBF (Input-Branch-Frozen) — freeze proprio MLP + (VAE projection MLP);
   keep fusion + GRU + heads trainable.
A3 LoRA — see `lora.py`; this module only handles freeze/unfreeze.

The helpers are framework-agnostic: they walk the nn.Module tree by attribute
path. For `PretrainAdaptPolicy`, the relevant attribute paths are:
   - `proprio_mlp`
   - `vae_proj_mlp`        (only present if cfg.use_vae_branch=True)
   - `fusion_mlp`
   - `gru`
   - `actor_trunk`, `actor_mu`
   - `critic_trunk`, `critic_v`
   - `log_std`
"""

from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn

def _set_requires_grad(module: nn.Module, flag: bool) -> int:
    """Set requires_grad on every parameter of `module`. Returns count."""
    n = 0
    for p in module.parameters():
        p.requires_grad_(flag)
        n += 1
    return n

def freeze_modules_by_path(root: nn.Module, paths: Iterable[str]) -> List[str]:
    """Set requires_grad=False on submodules at the listed dotted paths.

    Missing paths are silently skipped (e.g. `vae_proj_mlp` is None in simple env).
    Returns the list of paths that were actually frozen.
    """
    frozen: List[str] = []
    for path in paths:
        try:
            sub = root.get_submodule(path)
        except AttributeError:
            continue
        if sub is None:
            continue
        _set_requires_grad(sub, False)
        frozen.append(path)
    return frozen

def unfreeze_all(root: nn.Module) -> None:
    for p in root.parameters():
        p.requires_grad_(True)

# Strategy presets --------------------------------------------------------

def apply_full_ft(policy: nn.Module) -> None:
    """A1 — every param trainable."""
    unfreeze_all(policy)

def apply_ibf(policy: nn.Module) -> List[str]:
    """A2 — freeze proprio MLP and VAE projection MLP (input branches).

    Fusion + GRU + actor/critic heads remain trainable.
    """
    return freeze_modules_by_path(policy, ["proprio_mlp", "vae_proj_mlp"])

def trainable_param_count(root: nn.Module) -> int:
    return sum(p.numel() for p in root.parameters() if p.requires_grad)

def total_param_count(root: nn.Module) -> int:
    return sum(p.numel() for p in root.parameters())
