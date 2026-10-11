"""V8 (pre-registration v2, section 25, stage 31): whitened frozen features and the observation at the heads.

RL (training return, R2 and R3 x T1-T3, seeds 7, 42, 99), for random, MAE, and AP (run name PIDM) backbones:
  Frozen       stage 1, mode EncFrozen
  whiten       stage 31, init <name>_whiten, mode EncFrozen (fixed whitening of the GRU output before the heads)
  skip         stage 31, init <name>_skip, mode EncFrozen (standardized observation appended to the head input)
  FullFT       stage 1, same backbone
Reported:
  - gain over training from scratch (Eq. gain, stage-1 scratch return units) at 100 and 600 iterations, seed t interval;
  - the Frozen-to-FullFT ratio at 600 iterations (return of the frozen variant / return of FullFT of the same backbone,
    per setting and seed, mean over the settings), for T1 and T2 and for all six settings;
  - the recovered share (ratio_variant - ratio_Frozen) / (1 - ratio_Frozen), per seed;
  - the supervised check run before RL (analysis/v8_students.csv): test R^2 of the value and the action of frozen
    students with whitened features or the observation at the heads, against plain frozen students
    (analysis/s19_students.csv, head width 64).
Prediction (pre-registered, from Proposition 1): whitening recovers more where the imbalance of directions is large,
the observation shortcut where information is missing.
Output: logs/study2/analysis/v8_features.json and a printed summary.
Usage: python study2_v8_features.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study2_review_stats as R  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")
S3 = [7, 42, 99]
T12 = ["state_setpoint_x500_clean", "recovery_x500", "ss_lmf2_clean", "recovery_lmf2"]
INITS = ("random", "MAE", "PIDM")

def r4(r):
    return None if r is None else dict(mean=round(r["mean"], 4), t95=[round(x, 4) for x in r["t95"]], p=round(r["p"], 4))

def ratio(b, init, variant, tasks):
    """Per seed: mean over settings of return(frozen variant) / return(FullFT, same backbone), 600 iterations."""
    f = b[(b.K == 600) & b.task.isin(tasks) & b.seed.isin(S3)]
    if variant == "Frozen":
        v = f[(f.stage == 1) & (f.init == init) & (f["mode"] == "EncFrozen")]
    else:
        v = f[(f.stage == 31) & (f.init == f"{init}_{variant}") & (f["mode"] == "EncFrozen")]
    ff = f[(f.stage == 1) & (f.init == init) & (f["mode"] == "FullFT")]
    m = v.groupby(["task", "seed"]).ret.mean().to_frame("v").join(ff.groupby(["task", "seed"]).ret.mean().to_frame("f"), how="inner")
    if m.empty:
        return pd.Series(dtype=float)
    return (m.v / m.f).groupby("seed").mean()

def main():
    b = R.load_budgets()
    tasks = sorted(b[(b.stage == 31)].task.unique())
    out = {"tasks": tasks}
    v31 = b[b.stage == 31].copy()
    v31["init"] = v31.init.str.replace("_whiten", "|whiten").str.replace("_skip", "|skip")
    base = b[(b.stage == 1) & b.seed.isin(S3) & b["mode"].isin(["EncFrozen", "FullFT"]) & b.init.isin(list(INITS))]
    comb = pd.concat([v31.assign(stage=900), base.assign(stage=900)])
    for init in INITS:
        for var in ("Frozen", "whiten", "skip"):
            a = (init, "EncFrozen") if var == "Frozen" else (f"{init}|{var}", "EncFrozen")
            for K in (100, 600):
                out[f"{init}|{var}|G_vs_scratch|K{K}"] = r4(R.contrast(comb, 900, K, S3, tasks, a, ("random", "FullFT")))
            if var != "Frozen":
                out[f"{init}|{var}|vs_plain_Frozen|K600"] = r4(R.contrast(comb, 900, 600, S3, tasks, a, (init, "EncFrozen")))
        for scope, ts in (("T1T2", T12), ("all", tasks)):
            rf = ratio(b, init, "Frozen", ts)
            out[f"{init}|ratio|{scope}|Frozen"] = round(float(rf.mean()), 4) if len(rf) else None
            for var in ("whiten", "skip"):
                rv = ratio(b, init, var, ts)
                if len(rv) and len(rf):
                    rec = ((rv - rf) / (1 - rf)).dropna()
                    out[f"{init}|ratio|{scope}|{var}"] = round(float(rv.mean()), 4)
                    out[f"{init}|recovered|{scope}|{var}"] = dict(mean=round(float(rec.mean()), 4),
                                                                   seeds={int(s): round(float(x), 4) for s, x in rec.items()})
    # supervised check (before RL)
    sv = pd.read_csv(A / "v8_students.csv")
    s0 = pd.read_csv(A / "s19_students.csv")
    s0 = s0[s0["mode"] == "Frozen"]
    for init in INITS:
        base_r = s0[s0.init == init][["r2_value", "r2_action"]].mean()
        out[f"supervised|{init}|Frozen"] = dict(value=round(float(base_r.r2_value), 4), action=round(float(base_r.r2_action), 4))
        for feat, hh in (("whiten", 64), ("skip", 64), ("whiten", 0), ("none", 0)):
            x = sv[(sv.init == init) & (sv.feat == feat) & (sv.head_hidden == hh)][["r2_value", "r2_action"]].mean()
            out[f"supervised|{init}|{feat}|head{hh}"] = dict(value=round(float(x.r2_value), 4), action=round(float(x.r2_action), 4))
    (A / "v8_features.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
