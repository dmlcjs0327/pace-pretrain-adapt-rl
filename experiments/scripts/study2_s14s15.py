"""S14 (distance curriculum, stage 11) and S15 (relative termination, stage 12), pre-registration v2 section 18.

Both remedies act only during training; every run is evaluated on the original T3 (fixed-state evaluation, no
disturbance). Reported per remedy, against the same conditions without remedy (stage 1, seeds 7, 42, 99, R2 = X500,
R3 = LMF2):
- reach rate (0.5 m) and evaluation return, mean over checkpoints 550/575/600, per condition and airframe;
- reach rate at 25 and 100 iterations (early);
- gain G of each pretrained condition over training from scratch under the same remedy (difference of reach rate,
  seed bootstrap over the three seeds, airframes averaged), and the same gain without remedy.
Output: logs/study2/analysis/s14s15.json and a printed table.
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
T3 = ["agg_x500", "agg_lmf2"]
CONDS = [("random", "FullFT"), ("MAE", "FullFT"), ("PIDM", "FullFT"), ("MAE", "BackboneLoRA")]
B = 4000
rng = np.random.default_rng(18)

def main():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    ev = ev[(ev.disturb == 0) & ev.seed.isin(SEEDS) & ev.task.isin(T3)]
    ev = ev[ev.stage.isin([1, 11, 12])]
    ev = ev[ev.apply(lambda r: (r["init"], r["mode"]) in CONDS, axis=1)]
    term = ev[ev.iter >= 550].groupby(["stage", "task", "init", "mode", "seed"])[["reached_05", "ret"]].mean()
    early = ev[ev.iter.isin([25, 100])].pivot_table(index=["stage", "task", "init", "mode", "seed"], columns="iter", values="reached_05")
    out = {}
    names = {1: "none", 11: "S14_curriculum", 12: "S15_relterm"}
    for st in (1, 11, 12):
        if st not in term.index.get_level_values(0):
            continue
        T = term.loc[st]
        for (i, m) in CONDS:
            if (i, m) not in [(a, b) for (_, a, b, _) in T.index]:
                continue
            for t in T3:
                try:
                    x = T.loc[(t, i, m)].reindex(SEEDS)
                except KeyError:
                    continue
                out[f"{names[st]}|{i}|{m}|{t}|reach"] = [None if pd.isna(v) else round(float(v), 4) for v in x.reached_05]
                out[f"{names[st]}|{i}|{m}|{t}|ret"] = [None if pd.isna(v) else round(float(v), 1) for v in x.ret]
                if st in early.index.get_level_values(0):
                    try:
                        e = early.loc[(st, t, i, m)].reindex(SEEDS)
                        out[f"{names[st]}|{i}|{m}|{t}|reach_it25_100"] = [[None if pd.isna(v) else round(float(v), 4) for v in e[c]] for c in (25, 100) if c in e]
                    except KeyError:
                        pass
        R = T.reached_05.unstack("seed").reindex(columns=SEEDS)

        def g(i, m, ss):
            d = 0.0
            for t in T3:
                d += R.loc[(t, i, m)].reindex(ss).mean() - R.loc[(t, "random", "FullFT")].reindex(ss).mean()
            return d / len(T3)
        for (i, m) in CONDS[1:]:
            try:
                pt = g(i, m, SEEDS)
            except KeyError:
                continue
            bs = [g(i, m, list(rng.choice(SEEDS, 3))) for _ in range(B)]
            out[f"{names[st]}|G_reach_vs_scratch|{i}|{m}"] = [round(float(pt), 4), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 4).tolist()]
        out[f"{names[st]}|scratch_reach_mean"] = round(float(R.loc[[(t, "random", "FullFT") for t in T3]].mean().mean()), 4)
        out[f"{names[st]}|n_runs"] = int(len(T.reset_index()))
    (A / "s14s15.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
