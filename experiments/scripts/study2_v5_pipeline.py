"""V5 (pre-registration v2, section 25): pipeline sensitivity of the training-from-scratch baseline (stage 28).

Variants: (a) bptt32, backpropagation through the whole 32-step rollout; (b) sig05, initial action standard deviation
0.5; both for random and MAE with FullFT on the six settings of R2 and R3, seeds 7, 42, 99. (c) runnorm, the running
input normalization of rl-games, training from scratch only, same cells. (d) ent1e3, entropy coefficient 1e-3, random
and MAE with FullFT on T3 of R2 and R3, seeds 7, 42, 99. Reported, as pre-registered:
- the final return of training from scratch in each variant against the default training from scratch (stage 1),
  in units of the default final return, seed t intervals;
- within each variant, the gain of MAE over training from scratch of the same variant at 100 and 600 iterations;
- within each variant, the iterations to the final level of its own training from scratch in T1 and T2 (section 24
  method, exploratory), and the same numbers for the default pipeline on the same seeds;
- the T3 training reach rate at 600 iterations (runnorm and ent1e3 ask whether training from scratch takes off).
Output: logs/study2/analysis/v5_pipeline.json and a printed summary.
Usage: python study2_v5_pipeline.py
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
import study2_time_to_level as T  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")
S3 = [7, 42, 99]
T6 = ["state_setpoint_x500_clean", "recovery_x500", "agg_x500", "ss_lmf2_clean", "recovery_lmf2", "agg_lmf2"]
T12 = [t for t in T6 if not t.startswith("agg")]
T3 = ["agg_x500", "agg_lmf2"]

def r4(r):
    return None if r is None else [round(r["mean"], 4), round(r["t95"][0], 4), round(r["t95"][1], 4)]

def main():
    out = {}
    b = R.load_budgets()
    base = b[(b.stage == 1) & (b["mode"] == "FullFT") & b.init.isin(["random", "MAE"]) & b.seed.isin(S3)]
    for v in ("bptt32", "sig05", "runnorm", "ent1e3"):
        var = b[(b.stage == 28) & b.init.isin(["random_" + v, "MAE_" + v])]
        tasks = sorted(set(var.task))
        # (1) training from scratch of the variant against the default training from scratch
        comb = pd.concat([var.assign(stage=900), base.assign(stage=900)])
        for K in (100, 600):
            out[f"{v}|scratch_vs_default|K{K}"] = r4(R.contrast(comb, 900, K, S3, tasks, ("random_" + v, "FullFT"), ("random", "FullFT")))
        # (2) MAE against training from scratch inside the variant
        if ("MAE_" + v) in set(var.init):
            ren = var.assign(stage=900, init=var.init.str.replace("_" + v, "", regex=False))
            for K in (100, 600):
                out[f"{v}|MAE_vs_scratch_same_variant|K{K}"] = r4(R.contrast(ren, 900, K, S3, tasks, ("MAE", "FullFT"), ("random", "FullFT")))
                out[f"{v}|default_MAE_vs_default_scratch|K{K}|same_tasks"] = r4(R.contrast(base.assign(stage=900), 900, K, S3, tasks, ("MAE", "FullFT"), ("random", "FullFT")))
        # (2b) default MAE against the training from scratch of the variant (the stronger baseline)
        for K in (100, 600):
            out[f"{v}|default_MAE_vs_variant_scratch|K{K}"] = r4(R.contrast(comb, 900, K, S3, tasks, ("MAE", "FullFT"), ("random_" + v, "FullFT")))
        # (3) T3 training reach at 600 iterations
        for name, d in (("variant", var), ("default", base)):
            dd = d[(d.K == 600) & d.task.isin(T3)]
            if len(dd):
                out[f"{v}|T3_reach600|{name}"] = {i: round(float(g.reached_05.mean()), 4) for i, g in dd.groupby("init")}
    # (4) time to the final level inside the variants with MAE, T1 and T2, and the default on the same seeds
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret"])
    for v in ("bptt32", "sig05", None):
        if v is None:
            d = s[(s.stage == 1) & s.task.isin(T12) & s.seed.isin(S3) & (s["mode"] == "FullFT") & s.init.isin(["random", "MAE"])]
            key = "default"
        else:
            d = s[(s.stage == 28) & s.task.isin(T12) & s.init.isin(["random_" + v, "MAE_" + v])]
            d = d.assign(init=d.init.str.replace("_" + v, "", regex=False))
            key = v
        tab = T.table(d, [("MAE", "FullFT")], [0.8, 1.0])
        for q in (0.8, 1.0):
            out[f"{key}|speedup_MAE|q{q}"] = T.boot(tab[tab.init == "MAE"], q)
    (A / "v5_pipeline.json").write_text(json.dumps(out, indent=1))
    for k, val in out.items():
        print(k, val)

if __name__ == "__main__":
    main()
