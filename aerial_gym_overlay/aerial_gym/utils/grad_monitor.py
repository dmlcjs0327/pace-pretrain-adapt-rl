"""Per-layer gradient norm monitor (proposal §4.6 W11).

Records L2 norm of gradients aggregated by 6 logical groups of the
PretrainAdaptPolicy (proposal §4.2):
    proprio_mlp, vae_proj_mlp, fusion_mlp, gru, actor (trunk+mu+log_std),
    critic (trunk+v).

Designed to be called after `loss.backward()` and before `optimizer.step()`.
Returns a dict[str, float] suitable for TensorBoard `add_scalar` or W&B `log`.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

GROUP_PREFIXES = {
    "grad_norm/proprio_mlp": ["proprio_mlp"],
    "grad_norm/vae_proj_mlp": ["vae_proj_mlp"],
    "grad_norm/fusion_mlp": ["fusion_mlp"],
    "grad_norm/gru": ["gru"],
    "grad_norm/actor": ["actor_trunk", "actor_mu", "log_std"],
    "grad_norm/critic": ["critic_trunk", "critic_v"],
}

def _matches(name: str, prefixes) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)

def per_layer_grad_norms(policy: nn.Module) -> Dict[str, float]:
    """Compute L2 norm of grads for each predefined group.

    For LoRA-wrapped Linears, both `base.weight` (frozen, grad=None) and
    `lora_A`/`lora_B` (trainable) appear under the original module name.
    The grad sum naturally excludes None grads.
    """
    sums: Dict[str, float] = {k: 0.0 for k in GROUP_PREFIXES}
    sums["grad_norm/total"] = 0.0
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        sq = float(p.grad.detach().pow(2).sum().item())
        sums["grad_norm/total"] += sq
        for group, prefixes in GROUP_PREFIXES.items():
            if _matches(name, prefixes):
                sums[group] += sq
                break
    # convert sum-of-squares → L2 norm
    return {k: float(v) ** 0.5 for k, v in sums.items()}
