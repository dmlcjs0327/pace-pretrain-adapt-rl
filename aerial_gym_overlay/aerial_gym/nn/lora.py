"""LoRA — Low-Rank Adaptation for nn.Linear (proposal §4.6 W3).

Drop-in replacement for any `nn.Linear` that adds a low-rank update
W' = W + (alpha/r) · B·A on top of the (frozen) base weight, with B
zero-initialized so the first forward exactly equals the base layer.

A short helper `wrap_linears_with_lora` walks an nn.Module tree and replaces
selected `nn.Linear` modules in-place. The selection predicate is supplied
by the caller — typically: actor head's first Linear + critic head's first
Linear + fusion's two Linears (proposal §4.4 default for `LoRA r=8`).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """nn.Linear drop-in with low-rank update; base is frozen by default."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0,
                 freeze_base: bool = True):
        super().__init__()
        self.base = base_linear
        if freeze_base:
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.r = int(r)
        self.lora_A = nn.Parameter(torch.empty(self.r, d_in, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, self.r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base is frozen; LoRA update goes through (x @ A^T) @ B^T scaled.
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.4f}, base_frozen={not self.base.weight.requires_grad}"

def _name_in_target_paths(name: str, target_paths: Iterable[str]) -> bool:
    """Match either dotted prefix or exact name."""
    return any(name == p or name.startswith(p + ".") for p in target_paths)

def wrap_linears_with_lora(
    root: nn.Module,
    target_paths: Iterable[str],
    r: int = 8,
    alpha: float = 16.0,
    freeze_base: bool = True,
) -> List[str]:
    """Replace `nn.Linear` modules at `target_paths` with `LoRALinear` in-place.

    Returns the list of dotted names that were actually wrapped.
    """
    wrapped: List[str] = []
    target_paths = list(target_paths)
    # collect candidate (parent, attr_name, full_name) before mutating
    candidates = []
    for full_name, mod in root.named_modules():
        if isinstance(mod, nn.Linear) and _name_in_target_paths(full_name, target_paths):
            parent_name, _, attr = full_name.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            candidates.append((parent, attr, full_name, mod))
    for parent, attr, full_name, base_linear in candidates:
        new_mod = LoRALinear(base_linear, r=r, alpha=alpha, freeze_base=freeze_base)
        setattr(parent, attr, new_mod)
        wrapped.append(full_name)
    return wrapped

def lora_trainable_params(root: nn.Module) -> int:
    return sum(p.numel() for n, p in root.named_parameters() if p.requires_grad)
