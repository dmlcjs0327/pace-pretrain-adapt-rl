"""Summary of the mechanism probes of Study 2 (exploratory, pre-registration sections 6 and 7).

Reads analysis/{results.json, budgets.csv, probes_info.csv, probes_dyn.csv, drift.csv} and writes
analysis/mechanism.json with
  signatures   for the two combinations of objective and mode with the largest and the two with the smallest
               primary measure (delta against the random counterpart) at 600 iterations (ties by objective, then
               mode; the rule of the transplant probe) and for the random counterparts of the same modes: improvement, airframe accuracy, mean
               R^2 of the future states 10 steps ahead (velocity, tilt, body rate; negative values set to 0),
               effective rank, and fraction of dead units, each at the start of RL and at 600 iterations, the
               final CKA under both input pipelines, and the Jacobian norms at 600 iterations
  correlations Spearman rank correlations over the pretrained runs, with a seed bootstrap CI, between the
               delta of a run (its return minus the mean return of its random counterpart, in units of the
               scratch return of the setting) and each probe value at 600
               iterations, and between the Jacobian norm and the command change in flight
Input pipeline of the information and plasticity probes: 'rl' (as the policy sees its input) unless --pipe pre.
Usage: python study2_mechanism.py [--stage 1] [--pipe rl]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pathlib import Path

A = Path(_EXP_ROOT + "/logs/study2/analysis")
rng = np.random.default_rng(2026)
FUT = ["r2_vel_10", "r2_tilt_10", "r2_rate_10"]

def ranked_combos(res, stage):
    R = res["delta"][str(stage)]["vs_random"]
    c = [(v["600"][0], *k.split("|")) for k, v in R.items() if "600" in v]
    best = sorted(c, key=lambda x: (-x[0], x[1], x[2]))[:2]
    worst = sorted(c, key=lambda x: (x[0], x[1], x[2]))[:2]
    return best, worst

def per_run(stage, pipe):
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.stage == stage) & (b.K == 600)]
    info = pd.read_csv(A / "probes_info.csv")
    dyn = pd.read_csv(A / "probes_dyn.csv")
    drift = pd.read_csv(A / "drift.csv")
    info = info[info.stage == stage].copy()
    for c in [c for c in info if c.startswith("r2_")]:
        info[c] = info[c].clip(lower=0)
    info["future"] = info[[f"{c}_{pipe}" for c in FUT]].mean(1)
    rand = b[b.init == "random"].groupby(["task", "mode"]).ret.mean()
    scr = b[(b.init == "random") & (b["mode"] == "FullFT")].groupby("task").ret.mean()
    rows = []
    for _, r in b.iterrows():
        i0 = info[(info.run_id == r.run_id) & (info.iter == 0)]
        i6 = info[(info.run_id == r.run_id) & (info.iter == 600)]
        d0 = dyn[(dyn.run_id == r.run_id) & (dyn.iter == 0)]
        d6 = dyn[(dyn.run_id == r.run_id) & (dyn.iter == 600)]
        k6 = drift[(drift.run_id == r.run_id) & (drift.iter == 600)]
        if i0.empty or i6.empty or d6.empty:
            continue
        imp = 100 * (r.ret / rand[(r.task, r["mode"])] - 1) if (r.task, r["mode"]) in rand.index else np.nan
        dl = (r.ret - rand[(r.task, r["mode"])]) / scr[r.task] if (r.task, r["mode"]) in rand.index and r.task in scr.index else np.nan
        rows.append(dict(run_id=r.run_id, task=r.task, init=r.init, mode=r["mode"], seed=r.seed, ret=r.ret, improvement=imp, delta=dl,
                         action_rate=r.get("action_rate", np.nan),
                         acc0=float(i0[f"acc_airframe_{pipe}"].iloc[0]), acc6=float(i6[f"acc_airframe_{pipe}"].iloc[0]),
                         fut0=float(i0.future.iloc[0]), fut6=float(i6.future.iloc[0]),
                         erank0=float(d0[f"erank_{pipe}"].iloc[0]), erank6=float(d6[f"erank_{pipe}"].iloc[0]),
                         dead0=float(d0[f"dead_{pipe}"].iloc[0]), dead6=float(d6[f"dead_{pipe}"].iloc[0]),
                         cka_pre=float(k6.cka.iloc[0]) if len(k6) else np.nan, cka_rl=float(d6.cka_rl.iloc[0]),
                         jac_norm=float(i6.jac_norm.iloc[0]), jac_std=float(i6.jac_std.iloc[0])))
    return pd.DataFrame(rows)

def boot_spearman(df, x, y, B=2000):
    d = df[[x, y, "seed"]].dropna()
    if len(d) < 6:
        return None
    pt = spearmanr(d[x], d[y]).correlation
    seeds = d.seed.unique()
    bs = []
    for _ in range(B):
        pick = rng.choice(seeds, len(seeds))
        g = pd.concat([d[d.seed == s] for s in pick])
        if g[x].nunique() > 1 and g[y].nunique() > 1:
            bs.append(spearmanr(g[x], g[y]).correlation)
    return [round(float(pt), 3), *np.round(np.percentile(bs, [2.5, 97.5]), 3).tolist()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--pipe", default="rl", choices=["rl", "pre"])
    a = ap.parse_args()
    res = json.loads((A / "results.json").read_text())
    runs = per_run(a.stage, a.pipe)
    best, worst = ranked_combos(res, a.stage)
    cols = ["delta", "improvement", "acc0", "acc6", "fut0", "fut6", "erank0", "erank6", "dead0", "dead6", "cka_pre", "cka_rl", "jac_norm", "jac_std"]
    sig = {}
    for grp, lst in (("best", best), ("worst", worst)):
        for v, o, m in lst:
            g = runs[(runs.init == o) & (runs["mode"] == m)]
            sig[f"{grp}:{o}+{m}"] = {**{c: round(float(g[c].mean()), 3) for c in cols}, "n": int(len(g)), "improvement_prereg": v}
    for m in sorted({m for _, _, m in best + worst}):
        g = runs[(runs.init == "random") & (runs["mode"] == m)]
        sig[f"random+{m}"] = {**{c: round(float(g[c].mean()), 3) for c in cols if c not in ("improvement", "delta")}, "n": int(len(g))}
    pre = runs[runs.init != "random"]
    cor = {f"delta~{c}": boot_spearman(pre, c, "delta") for c in ["cka_pre", "cka_rl", "erank6", "dead6", "acc6", "fut6", "jac_std"]}
    cor["action_rate~jac_std"] = boot_spearman(runs, "jac_std", "action_rate")
    out = {"stage": a.stage, "pipe": a.pipe, "n_runs": int(len(runs)), "signatures": sig, "correlations": cor}
    (A / f"mechanism_{a.pipe}.json").write_text(json.dumps(out, indent=1))
    runs.to_csv(A / f"mechanism_runs_{a.pipe}.csv", index=False)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
