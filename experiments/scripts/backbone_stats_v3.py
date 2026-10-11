"""Representation statistics of the pretrained backbones (Study 2, Appendix B).
For 1,536 windows of 33 steps (512 per airframe, first 600 time steps of each file), standardized with the
statistics stored in each checkpoint, it reports the mean standard deviation of the final GRU features and
their effective rank (exponential of the entropy of the normalized squared singular values), plus the
final pretraining loss. A random backbone is included for reference."""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401
import glob, json
import h5py, numpy as np, torch
from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
D = _DATA_ROOT + "/"
CK = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3/"
FILES = [("v3/ssl_v3_x500.h5", "x500"), ("v3/ssl_v3_lmf2.h5", "lmf2_proprio_clean"), ("v3/ssl_v3_baseQuad.h5", "base_quadrotor")]
rng = np.random.default_rng(0)
X = []
for fn, g in FILES:
    with h5py.File(D + fn, "r") as f:
        s = f[g]["states"][:2700 * 4096].reshape(2700, 4096, 13)
    ts = rng.choice(np.arange(2700)[(np.arange(2700) % 300) <= 267], 512); ns = rng.integers(0, 4096, 512)
    X.append(np.stack([s[t:t + 33, n] for t, n in zip(ts, ns)]))
X = np.concatenate(X)
ref = torch.load(glob.glob(CK + "P_MAE__seed7__*.pt")[0], map_location="cpu")
mu, sd = np.array(ref["state_mean"]), np.array(ref["state_std"])
x = torch.tensor((X - mu) / sd, dtype=torch.float32)
def stats(bb):
    with torch.no_grad():
        z, _ = bb(x)
    Z = z[:, -1].numpy(); Zc = Z - Z.mean(0)
    sv = np.linalg.svd(Zc, compute_uv=False); p = sv ** 2 / np.sum(sv ** 2)
    return float(Z.std(0).mean()), float(np.exp(-(p * np.log(p + 1e-12)).sum()))
out = {}
for seed in (7, 42, 99):
    torch.manual_seed(seed)
    out[f"random/seed{seed}"] = dict(zip(("feature_std", "effective_rank"), stats(ProprioGRUBackbone(proprio_dim=13))))
for p in sorted(glob.glob(CK + "P_*__seed*__srcmixed__backbone.pt")):
    ck = torch.load(p, map_location="cpu")
    bb = ProprioGRUBackbone(proprio_dim=13); bb.load_state_dict(ck["backbone_state_dict"])
    fs, er = stats(bb)
    out[f"{ck['objective']}/seed{ck['seed']}"] = dict(feature_std=fs, effective_rank=er, epochs=len(ck["loss_history"]),
                                                      loss_first=ck["loss_history"][0], loss_last=ck["loss_history"][-1])
json.dump(out, open(_EXP_ROOT + "/logs/pretrain_v3/backbone_stats.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:14s} " + "  ".join(f"{a}={b:.4g}" for a, b in v.items()))
