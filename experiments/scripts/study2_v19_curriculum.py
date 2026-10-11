"""V19 (pre-registration v2, section 31): initial-state diversity x success-gated curriculum x pretraining in T3.

Cells (FullFT, R2 and R3, seeds 7, 42, 99), each evaluated with the fixed-state evaluation of its own initial-state
distribution (no wind, reach rate of the first episode, `reached_05`):
  narrow, no curriculum      stage 1,  tasks agg_x500 / agg_lmf2,               init random / MAE
  narrow, curriculum         stage 36, tasks agg_x500 / agg_lmf2,               init random_succ / MAE_succ
  diverse, no curriculum     stage 42, tasks agg_upset_x500 / agg_upset_lmf2,   init random / MAE
  diverse, curriculum        stage 42, tasks agg_upset_x500 / agg_upset_lmf2,   init random_succ / MAE_succ
Measures, at 100 and 300 iterations and at the end (mean of 550, 575, 600):
  reach rate of every cell (mean over the two airframes, then over seeds);
  G = reach(MAE) - reach(scratch) under the same start and curriculum, one value per seed (mean over airframes);
  interaction = G(diverse, curriculum) - G(narrow, curriculum), and the same without curriculum, per seed;
  seed bootstrap (4,000 resamples) of the mean, and the per-seed values.
Secondary: the reach rate of the diverse cells by initial tilt bin (analysis/eval_tilt_bins.csv).
Output: logs/study2/analysis/v19_curriculum.json and a printed summary.
Usage: python study2_v19_curriculum.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(_EXP_ROOT + "/logs/study2/analysis")
SEEDS = [7, 42, 99]
CELLS = {("narrow", "none"): (1, ["agg_x500", "agg_lmf2"], ""),
         ("narrow", "curr"): (36, ["agg_x500", "agg_lmf2"], "_succ"),
         ("diverse", "none"): (42, ["agg_upset_x500", "agg_upset_lmf2"], ""),
         ("diverse", "curr"): (42, ["agg_upset_x500", "agg_upset_lmf2"], "_succ")}
POINTS = {"100": [100], "300": [300], "end": [550, 575, 600]}
rng = np.random.default_rng(31)

def boot(v):
    v = np.asarray(v, float)
    if len(v) == 0 or np.isnan(v).all():
        return None
    b = [rng.choice(v, len(v)).mean() for _ in range(4000)]
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]

def reach(ev, stage, tasks, init, its):
    e = ev[(ev.stage == stage) & ev.task.isin(tasks) & (ev.init == init) & (ev["mode"] == "FullFT") & ev.seed.isin(SEEDS)
           & ev.iter.isin(its) & (ev.disturb == 0)]
    if e.empty:
        return pd.Series(dtype=float)
    return e.groupby(["seed", "task"]).reached_05.mean().groupby("seed").mean()

def main():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    out = {}
    gains = {}
    for (start, cur), (st, tasks, tag) in CELLS.items():
        for p, its in POINTS.items():
            sc, ma = reach(ev, st, tasks, "random" + tag, its), reach(ev, st, tasks, "MAE" + tag, its)
            out[f"{start}|{cur}|{p}|reach_scratch"] = boot(sc.values)
            out[f"{start}|{cur}|{p}|reach_MAE"] = boot(ma.values)
            g = (ma - sc).dropna()
            gains[(start, cur, p)] = g
            out[f"{start}|{cur}|{p}|G"] = boot(g.values)
            out[f"{start}|{cur}|{p}|G_seeds"] = {int(s): round(float(x), 4) for s, x in g.items()}
    for cur in ("curr", "none"):
        for p in POINTS:
            d, n = gains[("diverse", cur, p)], gains[("narrow", cur, p)]
            inter = (d - n).dropna()
            out[f"interaction|{cur}|{p}"] = boot(inter.values)
            out[f"interaction|{cur}|{p}|seeds"] = {int(s): round(float(x), 4) for s, x in inter.items()}
    tb = A / "eval_tilt_bins.csv"
    if tb.exists():
        t = pd.read_csv(tb).drop_duplicates(["run_id", "iter", "disturb"], keep="last")
        t = t[(t.stage == 42) & t.seed.isin(SEEDS)]
        cols = [c for c in t.columns if c.startswith("reached_05_tilt")]
        for p, its in POINTS.items():
            tt = t[t.iter.isin(its)]
            for init in ("random", "MAE", "random_succ", "MAE_succ"):
                x = tt[tt.init == init]
                if len(x):
                    out[f"tilt|{init}|{p}"] = {c[len("reached_05_"):]: round(float(x.groupby(["seed", "task"])[c].mean()
                                                                               .groupby("seed").mean().mean()), 4) for c in cols}
    (A / "v19_curriculum.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
