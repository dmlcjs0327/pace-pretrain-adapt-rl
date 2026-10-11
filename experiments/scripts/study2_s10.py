"""S10 (pre-registration v2, section 17): target-swapped backbones (stage 8) against the original pretrained backbone
(stage 1), the random backbone with the same mode, and scratch, on the same six settings and seeds {7, 42, 99}.

Reported, as pre-registered:
- position P = G(swapped, random) / G(original, random) with the scale of S7 (Eq. position of the manuscript):
  early = fixed-state evaluation return at 100 iterations in T1 and T2 (in units of the scratch terminal evaluation
  return), final = training return at 600 iterations in T3 (in units of the scratch training return); seed bootstrap.
- T3 reach rate (fixed-state evaluation, no disturbance, mean over checkpoints 550/575/600) per condition and airframe.
- T1 and T2 gain at 100 iterations of each condition over scratch (same units as above).
Output: logs/study2/analysis/s10.json and a printed summary.
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
T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
T3 = ["agg_x500", "agg_lmf2"]
SEEDS = [7, 42, 99]
B = 4000
rng = np.random.default_rng(17)

def main():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    ev = ev[(ev.disturb == 0) & ev.seed.isin(SEEDS) & ev["mode"].eq("FullFT")]
    ev = ev[((ev.stage == 1) & ev.init.isin(["random", "MAE", "PIDM"])) | ((ev.stage == 8) & ev.init.isin(["MAEtswap", "PIDMtswap"]))]
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.K == 600) & b.seed.isin(SEEDS) & b["mode"].eq("FullFT") & b.task.isin(T3)]
    b = b[((b.stage == 1) & b.init.isin(["random", "MAE", "PIDM"])) | ((b.stage == 8) & b.init.isin(["MAEtswap", "PIDMtswap"]))]

    e100 = ev[ev.iter == 100].pivot_table(index=["task", "seed"], columns="init", values="ret")
    eterm = ev[ev.iter >= 550].groupby(["task", "init", "seed"]).ret.mean().unstack("init")
    den_e = eterm["random"].groupby("task").mean()
    tr = b.pivot_table(index=["task", "seed"], columns="init", values="ret")
    den_t = tr["random"].groupby("task").mean()
    reach = ev[ev.iter >= 550].groupby(["task", "init", "seed"]).reached_05.mean().unstack("init")

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

    out = {"n_runs_stage8": int(ev[ev.stage == 8].run_id.nunique())}
    for base in ("MAE", "PIDM"):
        sw = base + "tswap"
        for name, tab, tasks, den in (("early_T12_eval100", e100, T12, den_e), ("final_T3_ret600", tr, T3, den_t)):
            if sw not in tab.columns:
                continue
            out[f"{sw}|G_vs_random|{name}"] = boot(lambda ss: gain(tab, tasks, den, sw, ss))
            out[f"{base}|G_vs_random|{name}"] = boot(lambda ss: gain(tab, tasks, den, base, ss))

            def pos(ss):
                o = gain(tab, tasks, den, base, ss)
                return gain(tab, tasks, den, sw, ss) / o if abs(o) > 1e-9 else np.nan
            out[f"{sw}|P|{name}"] = boot(pos)
    for t in T3:
        for i in ("random", "MAE", "PIDM", "MAEtswap", "PIDMtswap"):
            if i in reach.columns:
                v = reach.loc[t][i].reindex(SEEDS)
                out[f"reach|{t}|{i}"] = [round(float(x), 4) if pd.notna(x) else None for x in v]
    for i in ("MAE", "PIDM", "MAEtswap", "PIDMtswap"):
        if i in reach.columns:
            out[f"reach|T3mean|{i}"] = round(float(reach.loc[T3][i].mean()), 4)
    out["reach|T3mean|random"] = round(float(reach.loc[T3]["random"].mean()), 4)
    (A / "s10.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
