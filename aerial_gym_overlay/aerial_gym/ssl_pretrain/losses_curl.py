"""CURL InfoNCE loss for canonical AuxJoint comparison.

Reference: Laskin et al. 2020 [laskin2020curl] — Contrastive Unsupervised
Representations for Reinforcement Learning. Two image augmentations of the
same observation are encoded by a *query* network and a *key* network; the
key encoder is an EMA copy of the query (momentum encoder, He et al. 2020
MoCo style). Bilinear InfoNCE pulls matched query–key pairs together while
pushing non-matched pairs in the batch apart.

Adapted to our backbone interface so it can be plugged into either:
  - Offline pretrain (`trainer_vision_curl.py`)
  - Online AuxJoint (via `agent_pretrain_adapt.py` aux handler)

Shape conventions match `aux_losses.py` / `ssl_pretrain/losses.py`:
  forward_with_views(z_query, z_key) → scalar InfoNCE
where z_query and z_key are (B, T, hidden_dim) latents from two augmented
views of the same input sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class CURLLoss(nn.Module):
    """Bilinear InfoNCE on (z_query, z_key) latent pairs.

    Logits matrix L_ij = z_q_i^T W z_k_j, optimised by a symmetric
    cross-entropy with the identity matrix as target (matched pairs along
    the diagonal). Temperature softmax with `temperature` parameter.
    """

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        # Bilinear weight (CURL canonical — He 2020 MoCo also uses this form).
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * (hidden_dim ** -0.5))

    def forward_with_views(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """
        z_query, z_key: (B, T, hidden_dim)

        We pool over time first (mean) — this gives one representation per
        sequence, then InfoNCE over the batch. SPR-style temporal contrast
        would instead operate per-timestep; we keep the simpler CURL form
        since our hidden_dim already aggregates GRU recurrence.
        """
        # mean-pool over time → (B, hidden_dim)
        q = z_query.mean(dim=1)
        k = z_key.mean(dim=1).detach()  # key is detached (momentum-target style)

        # Normalise (CURL paper uses bilinear form on un-normalised latents; we
        # add L2 normalisation for numerical stability in our small hidden_dim).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Bilinear logits (B, B)
        Wq = q @ self.W                                # (B, hidden_dim)
        logits = Wq @ k.t()                            # (B, B)
        logits = logits / self.temperature

        B = q.shape[0]
        targets = torch.arange(B, device=q.device)
        # Symmetric InfoNCE (matches CURL/MoCo).
        loss_q2k = F.cross_entropy(logits, targets)
        loss_k2q = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_q2k + loss_k2q)
