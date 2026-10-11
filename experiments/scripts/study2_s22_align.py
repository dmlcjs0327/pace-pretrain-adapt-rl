"""S22 (pre-registration v2, section 21): alignment of the initial representation with the policy that RL must learn.

For every backbone (eight objectives x seeds 7, 42, 99; random backbones with the default initialization under seeds
7, 42, 99; shuffled MAE and AP; target-swapped MAE and AP; time-shuffled MAE when it exists) the GRU output at the last
step of 33-step windows is computed on the test episode of the S19 teacher data of the four T1/T2 settings (4,096
windows per setting, fixed seed). Reported per backbone (mean over the four settings):
  cka_action  linear CKA between the features and the teacher's mean action at the last step
  cka_value   linear CKA between the features and the teacher's value at the last step
  erank       participation ratio (sum lambda)^2 / sum lambda^2 of the feature covariance
Then the Spearman rank correlation over backbone-seed units with
  g100_full   early gain of FullFT: fixed-state evaluation at 100 iterations in T1 and T2 minus that of training from
              scratch (same seed), in units of the final evaluation of training from scratch (mean over 4 settings)
  g600_frozen final gain of the frozen backbone: training return at 600 iterations minus training from scratch (same
              seed), in units of the scratch return (mean over 4 settings; stage 1 only)
Reference: for a linearized head, the initial decrease of the squared error is proportional to the alignment of the
feature kernel with the target (Arora et al., 2019; Saxe et al., 2014).
Output: logs/study2/analysis/s22_align.csv and s22_align.json.
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
import isaacgym  # noqa: F401  (must precede torch)
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
import study2_drift as D

TASKS = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
EPISODES = {"state_setpoint_x500_clean": 2, "ss_lmf2_clean": 2, "recovery_x500": 4, "recovery_lmf2": 4}
CK = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3"
OBJ = ["MAE", "CPC", "SPR", "PIDM", "IDM", "FDM", "BC", "MAEs"]
EXTRA = {"MAEshuf": 5, "PIDMshuf": 5, "MAEtswap": 8, "PIDMtswap": 8, "MAEtshuf": 17}  # init -> RL stage
A = D.L / "analysis"
WIN = 33

def test_windows(task, rng, n=4096):
    z = np.load(D.L / "s19" / f"teacher_{task}.npz")
    obs, act, val, done = z["obs"], z["act"], z["value"], z["done"]
    T, N = done.shape
    t_lo = T - T // EPISODES[task]
    ends = np.zeros((T + 1, N), dtype=np.int64)
    ends[1:] = np.cumsum(done, 0)
    t0 = np.arange(t_lo, T - WIN + 1)
    ok = (ends[t0 + WIN - 1] - ends[t0]) == 0
    tt, nn_ = np.nonzero(ok)
    pick = rng.choice(len(tt), size=min(n, len(tt)), replace=False)
    ts, ns = t0[tt[pick]], nn_[pick]
    idx = ts[:, None] + np.arange(WIN)[None]
    x = torch.tensor(obs[idx, ns[:, None]], dtype=torch.float32, device=D.dev)
    last = ts + WIN - 1
    return x, act[last, ns], val[last, ns]

def erank(F):
    F = F - F.mean(0)
    ev = torch.linalg.eigvalsh(F.T @ F / F.shape[0]).clamp(min=0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())

def backbones():
    out = []
    for o in OBJ + list(EXTRA):
        for s in (7, 42, 99):
            p = f"{CK}/P_{o}__seed{s}__srcmixed__backbone.pt"
            if os.path.exists(p):
                out.append((o, s, D.ssl_backbone(p)))
    for s in (7, 42, 99):
        torch.manual_seed(s)
        bb = ProprioGRUBackbone(proprio_dim=13, proprio_hidden=64, fusion_hidden=128, gru_hidden=128)
        out.append(("random", s, D.effective_backbone(bb.state_dict(), prefix="")))
    return out

def gains():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    ev = ev[(ev.disturb == 0) & ev.task.isin(TASKS) & ev.seed.isin([7, 42, 99])]
    sc = ev[(ev.stage == 1) & (ev.init == "random") & (ev["mode"] == "FullFT")]
    den = sc[sc.iter >= 550].groupby("task").ret.mean()
    sc100 = sc[sc.iter == 100].set_index(["task", "seed"]).ret
    rows = {}
    for init in OBJ + list(EXTRA):
        st = EXTRA.get(init, 1)
        e = ev[(ev.stage == st) & (ev.init == init) & (ev["mode"] == "FullFT") & (ev.iter == 100)].set_index(["task", "seed"]).ret
        for s in (7, 42, 99):
            v = [(e.get((t, s)) - sc100.get((t, s))) / den[t] for t in TASKS if (t, s) in e.index and (t, s) in sc100.index]
            if len(v) == len(TASKS):
                rows[(init, s)] = {"g100_full": float(np.mean(v))}
    for s in (7, 42, 99):
        rows.setdefault(("random", s), {})["g100_full"] = 0.0
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.stage == 1) & (b.K == 600) & b.task.isin(TASKS) & b.seed.isin([7, 42, 99])]
    scr = b[(b.init == "random") & (b["mode"] == "FullFT")].set_index(["task", "seed"]).ret
    dden = scr.groupby(level="task").mean()
    for init in OBJ + ["random"]:
        fz = b[(b.init == init) & (b["mode"] == "EncFrozen")].set_index(["task", "seed"]).ret
        for s in (7, 42, 99):
            v = [(fz.get((t, s)) - scr.get((t, s))) / dden[t] for t in TASKS if (t, s) in fz.index and (t, s) in scr.index]
            if len(v) == len(TASKS):
                rows.setdefault((init, s), {})["g600_frozen"] = float(np.mean(v))
    return rows

def main():
    rng = np.random.default_rng(22)
    data = {t: test_windows(t, rng) for t in TASKS}
    rows = []
    with torch.no_grad():
        for init, s, W in backbones():
            r = dict(init=init, seed=s)
            ca, cv, er = [], [], []
            for t in TASKS:
                x, ya, yv = data[t]
                F = D.features(W, x)
                ca.append(D.cka(F, torch.tensor(ya, device=D.dev)))
                cv.append(D.cka(F, torch.tensor(yv[:, None], device=D.dev)))
                er.append(erank(F))
            r.update(cka_action=float(np.mean(ca)), cka_value=float(np.mean(cv)), erank=float(np.mean(er)))
            rows.append(r)
    df = pd.DataFrame(rows)
    g = gains()
    for k in ("g100_full", "g600_frozen"):
        df[k] = [g.get((i, s), {}).get(k, np.nan) for i, s in zip(df.init, df.seed)]
    df.to_csv(A / "s22_align.csv", index=False)
    out = {}
    for x in ("cka_action", "cka_value", "erank"):
        for y in ("g100_full", "g600_frozen"):
            d = df[[x, y]].dropna()
            rho, p = spearmanr(d[x], d[y])
            out[f"{x}|{y}"] = dict(rho=round(float(rho), 3), n=int(len(d)))
    (A / "s22_align.json").write_text(json.dumps(out, indent=1))
    print(df.round(3).to_string(index=False))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
