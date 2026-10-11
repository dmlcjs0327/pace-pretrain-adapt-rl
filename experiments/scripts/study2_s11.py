"""S11 (pre-registration v2, section 17): MAE with Aux-SSL at auxiliary weights 0.1 and 0.01 (stage 9) against
lambda = 1 (stage 1 AuxSSL) and lambda = 0 (stage 1 FullFT), on the six screening settings and seeds {7, 42, 99}.

Prediction H_C: the smaller lambda, the more the gain over scratch recovers toward FullFT, monotonically.
Measures (main scale G, training return, units of the scratch return of the setting, mean over six settings,
seed bootstrap with 4,000 resamples):
- G(lambda) over scratch (random + FullFT) at 100, 300, 600 iterations
- D(lambda) = G(lambda) - G(FullFT), the loss of keeping the auxiliary loss
Output: logs/study2/analysis/s11.json and a printed summary.
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
B = 4000
rng = np.random.default_rng(11)
LAM = {"AuxSSL": 1.0, "AuxSSL01": 0.1, "AuxSSL001": 0.01, "FullFT": 0.0}

def main():
    b = pd.read_csv(A / "budgets.csv")
    b = b[b.seed.isin(SEEDS)]
    keep = ((b.stage == 1) & (((b.init == "MAE") & b["mode"].isin(["AuxSSL", "FullFT"])) |
                              ((b.init == "random") & (b["mode"] == "FullFT")))) | \
           ((b.stage == 9) & (b.init == "MAE") & b["mode"].isin(["AuxSSL01", "AuxSSL001"]))
    b = b[keep].copy()
    b["cond"] = np.where(b.init == "random", "scratch", b["mode"])
    tasks = sorted(b[b.stage == 9].task.unique())
    out = {"n_runs_stage9": int(b[b.stage == 9].run_id.nunique()), "tasks": tasks}
    idx = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))  # seed resamples, shared by all measures

    def ci(pt, bs):
        return [round(float(pt), 3), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 3).tolist()]
    for K in (100, 300, 600):
        tab = b[b.K == K].pivot_table(index=["task", "seed"], columns="cond", values="ret")
        X = {c: np.array([[tab.loc[(t, sd), c] for sd in SEEDS] for t in tasks]) for c in list(LAM) + ["scratch"]}
        den = X["scratch"].mean(1)  # scratch return of each setting, all seeds (fixed denominator)

        def G(c, ts=None):
            m = np.ones(len(tasks), bool) if ts is None else np.array([t in ts for t in tasks])
            pt = ((X[c].mean(1) - X["scratch"].mean(1)) / den)[m].mean()
            bs = ((X[c][:, idx].mean(-1) - X["scratch"][:, idx].mean(-1)) / den[:, None])[m].mean(0)
            return pt, bs
        for c in LAM:
            pt, bs = G(c)
            out[f"G|{c}|{K}"] = ci(pt, bs)
            if c != "FullFT":
                pf, bf = G("FullFT")
                out[f"D_vs_FullFT|{c}|{K}"] = ci(pt - pf, bs - bf)
        if K == 600:
            for grp, ts in (("T12", [t for t in tasks if not t.startswith("agg")]), ("T3", [t for t in tasks if t.startswith("agg")])):
                for c in ("AuxSSL", "AuxSSL01", "AuxSSL001"):
                    pt, bs = G(c, ts)
                    pf, bf = G("FullFT", ts)
                    out[f"D_vs_FullFT|{c}|600|{grp}"] = ci(pt - pf, bs - bf)
    (A / "s11.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
