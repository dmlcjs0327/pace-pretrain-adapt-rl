"""S20 (pre-registration v2, section 21): a fixed sensor model that makes the 13-dimensional state high-dimensional,
mixed, and noisy, so that compressing the input has value (as for images).

phi(x) = [tanh(A x + b) (64 mixed channels), xi (64 channels of standard normal noise, new at every step)],
x = the state standardized with the fixed pretraining statistics. A has entries N(0, 4/13), b entries N(0, 0.25),
drawn once from a generator with seed 2026. The same phi is used for pretraining and RL. In RL the environment
wrapper applies it (runner.py, AG_OBS_EXPAND=1), so the noise is part of the stored observation and the PPO update
sees the same input as the rollout.
"""
import torch

EXPAND_SEED = 2026
IN_DIM, MIX_DIM, NOISE_DIM = 13, 64, 64
OUT_DIM = MIX_DIM + NOISE_DIM

def expansion_params(device="cpu"):
    g = torch.Generator().manual_seed(EXPAND_SEED)
    A = torch.randn(MIX_DIM, IN_DIM, generator=g) * (2.0 / IN_DIM ** 0.5)
    b = torch.randn(MIX_DIM, generator=g) * 0.5
    return A.to(device), b.to(device)

def expand(x, A, b):
    """x: [..., 13] standardized state -> [..., 128]."""
    mixed = torch.tanh(x @ A.T + b)
    noise = torch.randn(*x.shape[:-1], NOISE_DIM, device=x.device, dtype=x.dtype)
    return torch.cat([mixed, noise], dim=-1)
