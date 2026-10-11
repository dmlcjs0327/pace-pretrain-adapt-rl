"""S16 (pre-registration v2, section 18): MAE with LoRA and LP-FT trained for 1,800 iterations in T3 (stage 13),
R2 and R3, seeds {7, 42, 99}.

Prediction (direction only): take-off after 600 iterations is rare for LoRA and LP-FT.
Take-off criterion (as for S13, section 17): the 30-iteration running mean of the training return first exceeds the
highest 30-iteration mean of any scratch run in T3 of the screening and confirmation stages (104.5).
Reported per run: take-off iteration, highest 30-iteration mean up to 600 and up to 1,800, training reach rate
(within 0.5 m) in the windows (540, 600], (1140, 1200], (1740, 1800]. No threshold besides the pre-registered one.
Output: logs/study2/analysis/s16.json and a printed table.
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
T3 = ["agg_x500", "agg_lmf2"]

def main():
    s = pd.read_csv(A / "series.csv")
    t3 = s[s.task.isin(T3)]
    sc = t3[(t3.init == "random") & (t3["mode"] == "FullFT") & t3.stage.isin([1, 2])]
    best = float(sc.groupby("run_id").ret.apply(lambda r: r.rolling(30, min_periods=10).mean().max()).max())
    rows = []
    for rid, g in t3[t3.stage == 13].groupby("run_id"):
        g = g.sort_values("iter")
        m = g.ret.rolling(30, min_periods=10).mean().values
        it = g.iter.values
        w = lambda a, b: float(g[(g.iter > a) & (g.iter <= b)].reached_05.mean())
        rows.append(dict(run_id=rid, task=g.task.iloc[0], mode=g["mode"].iloc[0], seed=int(g.seed.iloc[0]),
                         takeoff=int(it[np.argmax(m > best)]) if (m > best).any() else None,
                         max30_600=float(np.nanmax(m[it <= 600])), max30_1800=float(np.nanmax(m)),
                         reach600=w(540, 600), reach1200=w(1140, 1200), reach1800=w(1740, 1800)))
    r = pd.DataFrame(rows).sort_values(["mode", "task", "seed"])
    out = {"best_scratch_30": round(best, 2), "runs": r.round(4).to_dict("records")}
    for mode, g in r.groupby("mode"):
        out[f"summary|{mode}"] = dict(n=len(g), first_takeoff_after_600=int((g.takeoff > 600).sum()),
                                      takeoff_by_600=int((g.takeoff <= 600).sum()), never=int(g.takeoff.isna().sum()),
                                      reach600=round(g.reach600.mean(), 4), reach1200=round(g.reach1200.mean(), 4),
                                      reach1800=round(g.reach1800.mean(), 4))
    (A / "s16.json").write_text(json.dumps(out, indent=1, default=str))
    print(r.round(3).to_string(index=False))
    for k, v in out.items():
        if k.startswith("summary"):
            print(k, v)

if __name__ == "__main__":
    main()
