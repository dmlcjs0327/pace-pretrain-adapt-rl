"""V6 (stage 29, progress reward), V7 (stage 30, restraint strength), and the fixed-state measures of V14 (stage 37).

V6: reach rate on the original T3 (fixed-state evaluation, no wind, mean of 550, 575, 600) of random, MAE, AP (PIDM)
    with FullFT trained with the progress term k in {10, 100} per m, seeds 7, 13, 21, 42, 99, R2 and R3; gain of MAE
    and AP over training from scratch under the same reward (per seed, mean over airframes, seed bootstrap); and the
    same without the progress term (confirmation stage) on the same seeds.
V7: gain over training from scratch (Eq. gain, training return) at 100 and 600 iterations for random and MAE backbones
    under Small-LR (backbone learning-rate factor 0.01 stage 1, 0.1 and 0.3 stage 30) and L2-SP (alpha 1e-2 stage 1,
    1e-3 and 1e-4 stage 30), six settings, seeds 7, 42, 99, seed t interval; FullFT is the end of both curves.
V14: T3 reach rate of target-masked backbones (stage 37) and their early (T1, T2, 100-iteration evaluation) and final
    (T3, 600-iteration training return) positions, as for target-swapped data (fig. 15c method).
Output: logs/study2/analysis/v6_v7_v14.json and a printed summary.
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
T3 = ["agg_x500", "agg_lmf2"]
S5 = [7, 13, 21, 42, 99]
S3 = [7, 42, 99]
rng = np.random.default_rng(29)

def boot(v):
    v = np.asarray(v, float)
    b = [rng.choice(v, len(v)).mean() for _ in range(4000)]
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]

def main():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    ev = ev[ev.disturb == 0]
    out = {}
    m = ev.task.isin(T3) & ev.iter.isin([550, 575, 600]) & ev.seed.isin(S5) & (ev["mode"] == "FullFT")

    def reach(stages, init):
        x = ev[m & ev.stage.isin(stages) & (ev.init == init)].drop_duplicates(["task", "seed", "iter"], keep="first")
        return x.groupby(["seed", "task"]).reached_05.mean().groupby("seed").mean()
    for k in (None, 10, 100):
        tag, st = ("", [2, 1]) if k is None else (f"_prog{k}", [29])
        sc = reach(st, "random" + tag)
        out[f"V6|k{k}|reach_scratch"] = boot(sc.values)
        for o in ("MAE", "PIDM"):
            r = reach(st, o + tag)
            out[f"V6|k{k}|reach_{o}"] = boot(r.values)
            out[f"V6|k{k}|gain_{o}"] = boot((r - sc).dropna().values)
    b = R.load_budgets()
    base = b[(b.stage == 1) & b.seed.isin(S3)]
    v30 = b[(b.stage == 30)]
    comb = pd.concat([base.assign(stage=900), v30.assign(stage=900)])
    tasks = sorted(b[b.stage == 30].task.unique())
    for init in ("random", "MAE"):
        for mode in ("DiscLR", "DiscLR01", "DiscLR03", "L2SP", "L2SP3", "L2SP4", "FullFT"):
            for K in (100, 600):
                r = R.contrast(comb, 900, K, S3, tasks, (init, mode), ("random", "FullFT"))
                out[f"V7|{init}|{mode}|K{K}"] = None if r is None else [round(r["mean"], 4), round(r["t95"][0], 4), round(r["t95"][1], 4)]
    sys.path.insert(0, "/tmp/claude-1000/-home-user-aerial-gym-simulator-my-research-2026-CoRL/89253667-86b1-4a9b-9acd-feb0ad9efaee/scratchpad/nc/v17/figsrc")
    import make_v11_mech_figs as M
    evm = M.ev_table()
    for alt in ("MAEtmask",):
        p = M.positions(evm, alt, 37)
        out[f"V14|{alt}|position"] = None if p is None else {k: [round(x, 3) for x in v] for k, v in p.items()}
    x = ev[ev.task.isin(T3) & ev.iter.isin([550, 575, 600]) & ev.seed.isin(S3) & (ev["mode"] == "FullFT")]
    for st, ini in ((1, "MAE"), (8, "MAEtswap"), (37, "MAEtmask"), (1, "PIDM"), (8, "PIDMtswap"), (37, "PIDMtmask"), (1, "random")):
        y = x[(x.stage == st) & (x.init == ini)]
        out[f"V14|reach|{ini}"] = round(100 * float(y.groupby(["seed", "task"]).reached_05.mean().mean()), 1) if len(y) else None
    (A / "v6_v7_v14.json").write_text(json.dumps(out, indent=1))
    for k_, v in out.items():
        print(k_, v)

if __name__ == "__main__":
    main()
