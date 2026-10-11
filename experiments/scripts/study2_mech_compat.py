"""S11 companion analysis: does RL move the backbone away from the
solution of its pretraining objective, and does AuxSSL hold it there?

For checkpoints of finished runs (start of RL = pretrained backbone, then 25, 100, 300, 600 iterations) it computes,
on a fixed set of 4,096 offline windows of the R2 and R3 flight data (pace_regularizers.load_windows, fixed seed,
standardized with the fixed statistics):
  loss_fixed   the pretraining loss of the objective with the head of the pretraining checkpoint (the refitted head
               P_<obj>_head; for AuxSSL runs also the head that trained along, column loss_own)
  r2_lin       the share of variance of the objective's target that a ridge regression on the GRU output explains
               (MAE: the hidden entries of the states, input masked with a fixed mask of ratio 0.3; AP (run name PIDM):
               the next four commands), fitted on half of the windows and tested on the other half
Output: logs/study2/analysis/mech_compat.csv (incremental, one row per run and checkpoint).
Usage: python study2_mech_compat.py [--stages 2] [--inits MAE PIDM random] [--modes FullFT AuxSSL]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401  (must precede torch: aerial_gym imports it)
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, _AG_ROOT + "/aerial_gym/rl_training/rl_games")
from pace_regularizers import load_windows  # noqa: E402
from aerial_gym.ssl_pretrain.losses import MaskedStateLoss, PIDMLoss  # noqa: E402

RUNS = _AG_ROOT + "/aerial_gym/rl_training/rl_games/runs/"
CKPT = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3/"
STATS = _DATA_ROOT + "/v3/state_stats.pt"
A = _EXP_ROOT + "/logs/study2/analysis/"
L = _EXP_ROOT + "/logs/study2/ledger.csv"
ITERS = [0, 25, 100, 300, 600]
NW = 4096

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proprio_mlp = nn.Sequential(nn.Linear(13, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        self.fusion_mlp = nn.Sequential(nn.Linear(64, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU())
        self.gru = nn.GRU(128, 128, batch_first=True)

    def forward(self, x):
        z, _ = self.gru(self.fusion_mlp(self.proprio_mlp(x)))
        return z

def backbone_from_run(sd):
    pre = "a2c_network.policy."
    bb = Backbone()
    bb.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre) and k[len(pre):].split(".")[0] in
                        ("proprio_mlp", "fusion_mlp", "gru")})
    return bb

def own_head(sd):
    pre = "a2c_network.pace_reg.head."
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}

def ridge_r2(Z, Y, lam=1e-2):
    n = Z.shape[0] // 2
    Z1 = np.concatenate([Z, np.ones((Z.shape[0], 1))], 1)
    W = np.linalg.solve(Z1[:n].T @ Z1[:n] + lam * np.eye(Z1.shape[1]), Z1[:n].T @ Y[:n])
    P = Z1[n:] @ W
    return float(1 - ((P - Y[n:]) ** 2).sum() / ((Y[n:] - Y[n:].mean(0)) ** 2).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="+", default=[2])
    ap.add_argument("--inits", nargs="+", default=["MAE", "PIDM", "random"])
    ap.add_argument("--modes", nargs="+", default=["FullFT", "AuxSSL"])
    ap.add_argument("--tasks", nargs="+", default=["state_setpoint_x500_clean", "recovery_x500", "agg_x500",
                                                     "ss_lmf2_clean", "recovery_lmf2", "agg_lmf2"])
    a = ap.parse_args()
    torch.set_num_threads(4)
    stats = torch.load(STATS)
    xs, us, _ = load_windows(stats)
    xs, us = xs[:NW], us[:NW]
    g = torch.Generator().manual_seed(7)
    mask = torch.rand(xs.shape, generator=g) < 0.3
    out = A + "mech_compat.csv"
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    have = set(zip(old.run_id, old.iter)) if len(old) else set()
    led = pd.read_csv(L)
    led = led[(led.exit == 0) & led.stage.isin(a.stages) & led.init.isin(a.inits) & led["mode"].isin(a.modes)
              & led.task.isin(a.tasks)].drop_duplicates("run_id", keep="last")
    # stage-2 cells of R2 and R3 with seeds 7, 42, 99 are reused stage-1 runs
    if 2 in a.stages:
        l1 = pd.read_csv(L)
        l1 = l1[(l1.exit == 0) & (l1.stage == 1) & l1.seed.isin([7, 42, 99]) & l1.init.isin(a.inits)
                & l1["mode"].isin(a.modes) & l1.task.isin(a.tasks)].drop_duplicates("run_id", keep="last")
        led = pd.concat([led, l1])
    rows = []
    for _, r in led.iterrows():
        objs = ["MAE", "PIDM"] if r.init == "random" else [r.init]
        heads = {}
        for o in objs:
            hk = torch.load(CKPT + f"P_{o}_head__seed{r.seed}__srcmixed__backbone.pt", map_location="cpu")
            heads[o] = hk
        for it in ITERS:
            if (r.run_id, it) in have:
                continue
            if it == 0:
                if r.init == "random":
                    f = glob.glob(RUNS + r.run_id + "/nn/last_*_ep_25_rew_*.pth")  # nearest to the random start
                    if not f:
                        continue
                    bb = backbone_from_run(torch.load(f[0], map_location="cpu")["model"])
                    sd = None
                else:
                    bb = Backbone()
                    bb.load_state_dict(torch.load(CKPT + f"P_{r.init}__seed{r.seed}__srcmixed__backbone.pt",
                                                  map_location="cpu")["backbone_state_dict"])
                    sd = None
            else:
                f = glob.glob(RUNS + r.run_id + f"/nn/last_*_ep_{it}_rew_*.pth")
                if not f:
                    continue
                sd = torch.load(f[0], map_location="cpu")["model"]
                bb = backbone_from_run(sd)
            bb.eval()
            row = dict(run_id=r.run_id, stage=r.stage, task=r.task, init=r.init, mode=r["mode"], seed=r.seed, iter=it)
            with torch.no_grad():
                for o in objs:
                    hk = heads[o]
                    if o == "MAE":
                        head = MaskedStateLoss(128, 13, mask_ratio=0.3)
                        z = bb(xs * (~mask).float())
                        head.load_state_dict(hk["head_state_dict"])
                        row[f"loss_fixed_{o}"] = float(head.forward_with_z_full(z, xs, mask))
                        if sd is not None and r["mode"] == "AuxSSL" and own_head(sd) and o == ("MAE" if r.init == "random" else r.init):
                            head.load_state_dict(own_head(sd))
                            row[f"loss_own_{o}"] = float(head.forward_with_z_full(z, xs, mask))
                        Z = z.reshape(-1, 128).numpy()
                        Y = xs.reshape(-1, 13).numpy()
                        M = mask.reshape(-1, 13).numpy()
                        # variance of the hidden entries explained: fit all entries, score on hidden ones
                        n = Z.shape[0] // 2
                        Z1 = np.concatenate([Z, np.ones((Z.shape[0], 1))], 1)
                        W = np.linalg.solve(Z1[:n].T @ Z1[:n] + 1e-2 * np.eye(129), Z1[:n].T @ Y[:n])
                        P = Z1[n:] @ W
                        res = ((P - Y[n:]) ** 2)[M[n:]].sum()
                        tot = ((Y[n:] - Y[n:].mean(0)) ** 2)[M[n:]].sum()
                        row[f"r2_lin_{o}"] = float(1 - res / tot)
                    else:
                        head = PIDMLoss(128, 4, predict_steps=4)
                        z = bb(xs)
                        head.load_state_dict(hk["head_state_dict"])
                        row[f"loss_fixed_{o}"] = float(head.forward_with_z_full(z, us))
                        if sd is not None and r["mode"] == "AuxSSL" and own_head(sd) and o == ("MAE" if r.init == "random" else r.init):
                            head.load_state_dict(own_head(sd))
                            row[f"loss_own_{o}"] = float(head.forward_with_z_full(z, us))
                        T = xs.shape[1]
                        Zt = z[:, :T - 4].reshape(-1, 128).numpy()
                        Y = torch.cat([us[:, k + 1:T - 4 + k + 1] for k in range(4)], -1).reshape(-1, 16).numpy()
                        row[f"r2_lin_{o}"] = ridge_r2(Zt, Y)
            rows.append(row)
        if len(rows) and len(rows) % 100 == 0:
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
            print(len(rows), flush=True)
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
    print("done", len(rows))

if __name__ == "__main__":
    main()
