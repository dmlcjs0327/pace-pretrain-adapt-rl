"""V14 (pre-registration v2, section 26) and V15 (section 27), the parts that need only the training logs.

V14, target-masked data (stage 37) next to the original (stage 1) and the target-swapped data (stage 8), FullFT,
R2 and R3, seeds 7, 42, 99, with the scale and seed bootstrap of S10 (study2_s10.py):
- G(x, random) and the position P = G(x, random) / G(original, random) on the T3 training return at 600 iterations.
- If the fixed-state evaluation of stage 37 exists: the early position (T1 and T2, 100 iterations) and the T3 reach rate.
V15, random backbone with the CPC auxiliary loss (stage 38) against training from scratch (stage 1, random + FullFT),
with seed-level t intervals at 100 and 600 iterations, next to the random backbone with the MAE auxiliary loss and the
CPC backbone with Aux-SSL (both stage 1), on the six settings of R2 and R3 and seeds 7, 42, 99.
Output: logs/study2/analysis/v14_v15.json and a printed summary.
Usage: python study2_v14_v15.py
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
T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
T3 = ["agg_x500", "agg_lmf2"]
SEEDS = [7, 42, 99]
B = 4000
rng = np.random.default_rng(26)
SRC = {"random": 1, "MAE": 1, "PIDM": 1, "MAEtswap": 8, "PIDMtswap": 8, "MAEtmask": 37, "PIDMtmask": 37}

def pick(df):
    keep = np.zeros(len(df), bool)
    for i, st in SRC.items():
        keep |= ((df.stage == st) & (df.init == i)).values
    return df[keep & df.seed.isin(SEEDS).values & df["mode"].eq("FullFT").values]

def main():
    out = {}
    b = pd.read_csv(A / "budgets.csv")
    tr = pick(b[(b.K == 600) & b.task.isin(T3)]).pivot_table(index=["task", "seed"], columns="init", values="ret")
    den_t = tr["random"].groupby("task").mean()
    tabs = [("final_T3_ret600", tr, T3, den_t)]
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    ev = pick(ev[ev.disturb == 0])
    reach = None
    if "MAEtmask" in set(ev.init):
        e100 = ev[(ev.iter == 100) & ev.task.isin(T12)].pivot_table(index=["task", "seed"], columns="init", values="ret")
        eterm = ev[(ev.iter >= 550) & ev.task.isin(T12)].groupby(["task", "init", "seed"]).ret.mean().unstack("init")
        tabs.append(("early_T12_eval100", e100, T12, eterm["random"].groupby("task").mean()))
        reach = ev[(ev.iter >= 550) & ev.task.isin(T3)].groupby(["task", "init", "seed"]).reached_05.mean().unstack("init")

    def gain(tab, tasks, den, x, ss):
        g = 0.0
        for t in tasks:
            X = tab.loc[t].reindex(ss)
            g += (X[x].mean() - X["random"].mean()) / den[t]
        return g / len(tasks)

    def boot(fn):
        pt = fn(SEEDS)
        bs = [fn(list(rng.choice(SEEDS, len(SEEDS)))) for _ in range(B)]
        return [round(float(pt), 4), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 4).tolist()]

    for base in ("MAE", "PIDM"):
        for name, tab, tasks, den in tabs:
            out[f"{base}|G_vs_random|{name}"] = boot(lambda ss: gain(tab, tasks, den, base, ss))
            for var in ("tswap", "tmask"):
                x = base + var
                if x not in tab.columns:
                    continue
                out[f"{x}|G_vs_random|{name}"] = boot(lambda ss: gain(tab, tasks, den, x, ss))

                def pos(ss):
                    o = gain(tab, tasks, den, base, ss)
                    return gain(tab, tasks, den, x, ss) / o if abs(o) > 1e-9 else np.nan
                out[f"{x}|P|{name}"] = boot(pos)
    if reach is not None:
        for i in SRC:
            if i in reach.columns:
                out[f"reach|T3mean|{i}"] = round(float(reach.loc[T3][i].mean()), 4)

    # V15
    bb = R.load_budgets()
    t6 = sorted(bb[(bb.stage == 38)].task.unique())
    for K in (100, 600):
        b2 = bb[bb.K == K]
        ref = b2[(b2.stage == 1) & (b2.init == "random") & (b2["mode"] == "FullFT")]
        for lab, st, cell in (("random+CPCaux(st38)", 38, ("random", "AuxSSLcpc")),
                              ("random+MAEaux(st1)", 1, ("random", "AuxSSL")),
                              ("CPC+AuxSSL(st1)", 1, ("CPC", "AuxSSL")),
                              ("CPC+FullFT(st1)", 1, ("CPC", "FullFT"))):
            rows = b2[(b2.stage == st) & (b2.init == cell[0]) & (b2["mode"] == cell[1])]
            if len(rows) < len(t6) * 3:
                out[f"V15|{lab}|K{K}"] = f"incomplete {len(rows)}/{len(t6) * 3}"
                continue
            comb = pd.concat([rows.assign(stage=900), ref.assign(stage=900)])
            r = R.contrast(comb, 900, K, R.S3, t6, cell, ("random", "FullFT"))
            out[f"V15|{lab}|K{K}"] = [round(r["mean"], 4), round(r["t95"][0], 4), round(r["t95"][1], 4), round(r["p"], 4)]
    (A / "v14_v15.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
