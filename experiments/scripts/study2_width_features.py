"""Frozen-feature geometry per network width (M4, pre-registration v2, sections 22 and 23).

For every width, backbone (random under the seed, or MAE pretrained at that width), and seed 7, 42, 99, the features
(GRU output at the last step of 33-step windows) on the test windows of the S19 teacher data of the four T1/T2
settings (study2_s22_align.test_windows, 4,096 windows per setting, generator seed 5) give
  part_ratio     participation ratio (sum lambda)^2 / sum lambda^2 of the feature covariance (effective dimension)
  k90            number of principal components that explain 90% of the variance
  r2_action_lin  test R^2 of a ridge regression (RidgeCV, 70/30 split) from the features to the teacher's mean action
  r2_value_lin   the same for the teacher's value
each averaged over the four settings. Rows of the requested widths are replaced in
logs/study2/analysis/q_width_features.csv, other widths are kept.
Usage: python study2_width_features.py --widths 8
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import isaacgym  # noqa: F401  (must precede torch)
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
from study2_s22_align import TASKS, test_windows

CK = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3"
OUT = _EXP_ROOT + "/logs/study2/analysis/q_width_features.csv"

def feats(bb, x):
    with torch.no_grad():
        h, _ = bb(x)
        return h[:, -1].cpu().numpy()

def pr(F):
    F = F - F.mean(0)
    ev = np.clip(np.linalg.eigvalsh(F.T @ F / len(F)), 0, None)[::-1]
    k90 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1)
    return ev.sum() ** 2 / (ev ** 2).sum(), k90

def probe(F, y):
    n = len(F)
    tr = np.arange(n) < int(0.7 * n)
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(F[tr], y[tr])
    p = m.predict(F[~tr])
    y2 = y[~tr]
    return float(1 - ((p - y2) ** 2).sum() / ((y2 - y2.mean(0)) ** 2).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=float, nargs="+", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    data = {t: test_windows(t, rng, n=4096) for t in TASKS}
    rows = []
    for w in a.widths:
        for init in ("random", "MAE"):
            for s in (7, 42, 99):
                gh = int(round(128 * w))
                if init == "MAE":
                    bb = ProprioGRUBackbone(13, int(round(64 * w)), gh, 2, gh).cuda()
                    tag = "" if w == 1.0 else f"__w{w:g}"
                    bb.load_state_dict(torch.load(f"{CK}/P_MAE__seed{s}__srcmixed{tag}__backbone.pt",
                                                  map_location="cpu")["backbone_state_dict"])
                else:
                    torch.manual_seed(s)
                    bb = ProprioGRUBackbone(13, int(round(64 * w)), gh, 2, gh).cuda()
                pa, k9, ra, rv = [], [], [], []
                for t in TASKS:
                    x, ya, yv = data[t]
                    F = feats(bb, x)
                    p, k = pr(F)
                    pa.append(p); k9.append(k); ra.append(probe(F, ya)); rv.append(probe(F, yv))
                rows.append((w, init, s, np.mean(pa), np.mean(k9), np.mean(ra), np.mean(rv)))
                print(rows[-1], flush=True)
    new = pd.DataFrame(rows, columns=["width", "init", "seed", "part_ratio", "k90", "r2_action_lin", "r2_value_lin"])
    try:
        old = pd.read_csv(OUT)
        old = old[~old.width.isin(a.widths)]
    except FileNotFoundError:
        old = pd.DataFrame()
    pd.concat([old, new], ignore_index=True).sort_values(["width", "init", "seed"]).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
