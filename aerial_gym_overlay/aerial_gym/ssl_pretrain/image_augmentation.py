"""Image augmentation for depth observations — CURL / SPR style.

Used by both the offline CURL pretrain trainer and the online AuxJoint(CURL)
handler to produce two augmented views of the same depth image for the
InfoNCE contrastive loss (CURL) or for the asymmetric query/key encoders
(SPR).

Augmentations (all preserve the (B, 1, H, W) tensor shape and are
implemented in pure PyTorch so they run on GPU):
  - random_crop_pad: random crop with zero-pad reflection (CURL canonical)
  - additive_gaussian: per-pixel Gaussian noise (depth-noise simulation)
  - random_translate: small random pixel shift (a simpler alternative to crop)

Usage:
  view_a = aug_pipeline(depth)  # (B, 1, H, W) → (B, 1, H, W)
  view_b = aug_pipeline(depth)  # different random crop/noise

For depth specifically we keep colour jitter out (it has no physical
meaning on a 1-channel depth image) and instead emphasise translate +
Gaussian noise, both of which are plausible IMU/depth-sensor analogues.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

def random_crop_pad(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Pad the image with zeros, then random-crop back to the original shape.

    Effectively a per-batch random translation by an offset in
    [-pad, +pad] pixels on each axis. CURL [laskin2020curl] default
    augmentation for 84×84 Atari frames uses pad=4.
    """
    B, C, H, W = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
    # Random crop offsets per batch element.
    offs_h = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    offs_w = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    out = torch.empty_like(x)
    # Vectorise via gather where possible; simple loop is fine at B≤256.
    for b in range(B):
        out[b] = padded[b, :, offs_h[b]:offs_h[b] + H, offs_w[b]:offs_w[b] + W]
    return out

def additive_gaussian(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma^2). Depth values are clamped to [0, max]
    after noise (real depth sensors return non-negative ranges).

    sigma=0.05 in normalised-depth units corresponds to ≈ 5 cm of additive
    noise on a 10 m range — within the order of real D435 / D455 noise."""
    noise = torch.randn_like(x) * sigma
    out = x + noise
    return torch.clamp(out, min=0.0, max=10.0)

def aug_pipeline(x: torch.Tensor, crop_pad: int = 4, noise_sigma: float = 0.05) -> torch.Tensor:
    """Apply random_crop_pad followed by additive_gaussian. Returns a new
    tensor of the same shape as `x`. Stochastic — call twice on the same
    input to obtain two distinct views for contrastive learning."""
    x = random_crop_pad(x, pad=crop_pad)
    x = additive_gaussian(x, sigma=noise_sigma)
    return x
