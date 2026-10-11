"""V13 (pre-registration v2, section 25, stage 36): success-gated distance curriculum in T3, ten seeds.

Fixed-state evaluation on the original T3 (no wind, reach rate `reached_05`), R2 and R3, seeds of the confirmation stage
(S10). Reported, as pre-registered: the reach rate of random, MAE, and AP (run name PIDM) with FullFT under the
success-gated curriculum (init names with the suffix _succ) at 100 and 300 iterations and at the end (mean of 550, 575,
600); the gain in reach rate of MAE and AP over training from scratch under the same curriculum, one value per seed
(mean over the two airframes), with a seed bootstrap (4,000 resamples) and the seed t interval; and, side by side, the
same numbers without curriculum (confirmation stage: stage 2 plus the stage-1 cells it reuses) on the same seeds.
Output: logs/study2/analysis/v13_success.json and a printed summary.
Usage: python study2_v13_success.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

A = Path(_EXP_ROOT + "/logs/study2/analysis")
S10 = [7, 13, 21, 42, 99, 123, 256, 512, 777, 2024]
T3 = ["agg_x500", "agg_lmf2"]
POINTS = {"100": [100], "300": [300], "end": [550, 575, 600]}
rng = np.random.default_rng(25)

def summ(v):
    v = np.asarray(v, float)
    b = [rng.choice(v, len(v)).mean() for _ in range(4000)]
    t = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v)) if len(v) > 1 else (np.nan, np.nan)
    return dict(mean=round(float(v.mean()), 4), boot=[round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                t95=[round(float(t[0]), 4), round(float(t[1]), 4)], n=int(len(v)))

def reach(ev, stages, init, its):
    e = ev[ev.stage.isin(stages) & ev.task.isin(T3) & (ev.init == init) & (ev["mode"] == "FullFT") & ev.seed.isin(S10)
           & ev.iter.isin(its) & (ev.disturb == 0)]
    e = e.drop_duplicates(["task", "init", "seed", "iter"], keep="first")
    return e.groupby(["seed", "task"]).reached_05.mean().groupby("seed").mean()

def main():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    out = {}
    for name, stages, tag in (("success", [36], "_succ"), ("none", [2, 1], "")):
        for p, its in POINTS.items():
            sc = reach(ev, stages, "random" + tag, its)
            out[f"{name}|{p}|reach_scratch"] = summ(sc.values) if len(sc) else None
            for o in ("MAE", "PIDM"):
                r = reach(ev, stages, o + tag, its)
                out[f"{name}|{p}|reach_{o}"] = summ(r.values) if len(r) else None
                g = (r - sc).dropna()
                out[f"{name}|{p}|gain_{o}"] = summ(g.values) if len(g) else None
    (A / "v13_success.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
