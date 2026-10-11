"""V9 (pre-registration v2, section 25, stage 32): separate actor and critic backbones, each with its own memory.

Conditions (FullFT, T1 and T2 on R2 and R3, seeds 7, 13, 21, 42, 99): both random (dualnone), MAE in the actor only
(dualactor), MAE in the critic only (dualcritic), MAE in both (dualboth). Training return.
Reported:
  - G of each condition over dualnone (Eq. gain, in units of the final return of dualnone) at 100 and 600 iterations,
    seed t interval;
  - the share of the 100-iteration gain of dualboth that each single side keeps, G(side) / G(both), per seed and pooled;
  - for reference, dualnone and dualboth against the shared-backbone runs of stage 1/2 (random and MAE with FullFT,
    same settings and seeds where available).
Prediction (pre-registered, from the critic-first account): the critic-only MAE keeps most of the 100-iteration gain,
the actor-only MAE less.
Output: logs/study2/analysis/v9_dual.json and a printed summary.
Usage: python study2_v9_dual.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study2_review_stats as R  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")
S5 = [7, 13, 21, 42, 99]
T12 = ["state_setpoint_x500_clean", "recovery_x500", "ss_lmf2_clean", "recovery_lmf2"]

def r4(r):
    return None if r is None else dict(mean=round(r["mean"], 4), t95=[round(x, 4) for x in r["t95"]], p=round(r["p"], 4),
                                       seeds={int(s): round(float(v), 4) for s, v in r["seeds"].items()})

def main():
    b = R.load_budgets()
    d = b[(b.stage == 32)].copy()
    # the reference of R.contrast (its unit) is the condition named random|FullFT: here the dual random backbones
    d32 = d.assign(init=d.init.replace({"random_dualnone": "random"}))
    out = {}
    for K in (100, 600):
        for cond in ("dualactor", "dualcritic", "dualboth"):
            out[f"{cond}|vs_dualnone|K{K}"] = r4(R.contrast(d32, 32, K, S5, T12, (f"MAE_{cond}", "FullFT"), ("random", "FullFT")))
    for K in (100, 600):
        gb = out[f"dualboth|vs_dualnone|K{K}"]
        if gb:
            for cond in ("dualactor", "dualcritic"):
                gs = out[f"{cond}|vs_dualnone|K{K}"]
                if gs:
                    per = {s: gs["seeds"][s] / gb["seeds"][s] for s in gs["seeds"] if s in gb["seeds"] and abs(gb["seeds"][s]) > 1e-9}
                    out[f"{cond}|share_of_both|K{K}"] = dict(pooled=round(gs["mean"] / gb["mean"], 4) if gb["mean"] else None,
                                                             seeds={s: round(v, 4) for s, v in per.items()})
    shared = b[b.stage.isin([1, 2]) & (b["mode"] == "FullFT") & b.init.isin(["random", "MAE"])]
    comb = pd.concat([d.assign(stage=900), shared.drop_duplicates(["task", "init", "mode", "seed", "K"]).assign(stage=900)])
    for K in (100, 600):
        out[f"dualnone_vs_shared_scratch|K{K}"] = r4(R.contrast(comb, 900, K, S5, T12, ("random_dualnone", "FullFT"), ("random", "FullFT")))
        out[f"dualboth_vs_shared_MAE|K{K}"] = r4(R.contrast(comb, 900, K, S5, T12, ("MAE_dualboth", "FullFT"), ("MAE", "FullFT")))
    (A / "v9_dual.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
