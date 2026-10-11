"""S10 support (pre-registration v2, section 17): goal-directedness of the pretraining data. Samples 3,000 windows of
33 steps per airframe (fixed seed), rotates the body velocity to the world frame, and stores the cosine between the
velocity and the goal-relative position for moving steps (speed > 0.3 m/s), for the original data and for the
target-swapped transform of S10 (dp_t - dp_1 + dp_1 of another window). Output: analysis/data_goal_cos.npz.
Section 25 (V3): also the task-agnostic data v3ag of R2 and R3 (keys R2ag_cos, R3ag_cos).
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import h5py
import numpy as np

D = _DATA_ROOT + "/v3/"
OUT = _EXP_ROOT + "/logs/study2/analysis/data_goal_cos.npz"

def rotate(q, v):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    return np.einsum("...ij,...j->...i", R, v)

rng = np.random.default_rng(0)
out = {}
for name, fn, g in [("R2", "ssl_v3_x500.h5", "x500"), ("R3", "ssl_v3_lmf2.h5", "lmf2_proprio_clean"),
                    ("R1", "ssl_v3_baseQuad.h5", None),
                    ("R2ag", "../v3ag/ssl_v3ag_x500.h5", "x500"), ("R3ag", "../v3ag/ssl_v3ag_lmf2.h5", "lmf2_proprio_clean")]:
    with h5py.File(D + fn, "r") as f:
        g = g or list(f.keys())[0]
        s = f[g]["states"]
        T = s.shape[0] // 4096
        ok = np.arange(T - 33)
        ok = ok[(ok % 300) <= 300 - 33]
        ts, vs = rng.choice(ok, 3000), rng.integers(0, 4096, 3000)
        S = np.stack([s[(t + np.arange(33)) * 4096 + v] for t, v in zip(ts, vs)])
    dp, vw = S[..., 0:3], rotate(S[..., 3:7], S[..., 7:10])
    sp, dist = np.linalg.norm(vw, axis=-1), np.linalg.norm(dp, axis=-1)
    m = sp > 0.3
    cos = (vw * dp).sum(-1) / (sp * dist + 1e-9)
    dps = dp - dp[:, :1] + dp[rng.permutation(len(S)), :1]
    coss = (vw * dps).sum(-1) / (sp * np.linalg.norm(dps, axis=-1) + 1e-9)
    t3 = (dist >= 4) & (sp >= 2) & (cos > 0.8)
    out[f"{name}_cos"], out[f"{name}_cos_swap"] = cos[m], coss[m]
    out[f"{name}_t3like"] = np.array([t3.mean(), t3.any(1).mean()])
    print(name, f"cos {cos[m].mean():+.3f} swapped {coss[m].mean():+.3f} T3-like steps {t3.mean():.3f}")
np.savez(OUT, **out)
