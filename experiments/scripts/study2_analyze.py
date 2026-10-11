"""Analysis of Study 2 (pre-registered in experiments/configs/study2_prereg_v2.md).

Primary measure (v2, section 6): delta[stage][A][B][K] = [point, lo, hi], the mean over settings of
  (R_e(A) - R_e(B)) / R_e(random, FullFT), with R_e the seed mean of the terminal return in setting e, i.e. the
  difference in units of the return of training from scratch; seed bootstrap (joint across settings).
  A, B are 'init|mode'. Stored for every pretrained condition against its random counterpart ('vs_random'),
  against training from scratch ('vs_scratch'), and against FullFT with the same backbone ('vs_fullft'; LP-FT
  report of section 10, S1), and in stage 2 for D1-D4 with all seeds and with the
  seven new seeds only ('decisions').
floor[stage]: settings where every condition reaches the 0.5 m ball in fewer than 5% of the episodes at 600
  iterations (mean over seeds), reported as not testable (section 6).

Reads logs/study2/analysis/{series,budgets}.csv (study2_parse.py) and writes analysis/results.json with
  improvement[stage][objective][mode][K] = [point, lo, hi]   percent over the random counterpart of the
      same mode, mean over settings; 95% percentile bootstrap that resamples whole seeds jointly across
      settings (4,000 draws)
  prob_improvement[stage][objective][mode][K] = [point, lo, hi]   probability that a run with the pretrained
      backbone ends above a run with the random counterpart in the same setting (all seed pairs, ties count
      one half; Agarwal et al. 2021), mean over settings, same seed bootstrap. Added after the pre-registration
      as a descriptive measure that is robust to the bimodal return (section 7 of the pre-registration)
  mode_return[stage][init][mode][K]      terminal return normalized per setting by random+FullFT at 600
  flight[stage][task][init][mode]        flight metrics at K = 600, mean over settings of the task and seeds
  traces[stage][init][mode]              median learning rate, KL, action noise per iteration
Conditions that are not complete yet are skipped.
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

A = Path(_EXP_ROOT + "/logs/study2/analysis")
# Extended grid (pre-registration v2, section 11); PIDM is the run name of the action-prediction objective (AP).
OBJ = ["MAE", "CPC", "SPR", "PIDM", "IDM", "FDM", "BC", "MAEs"]
MODES = ["FullFT", "LateFT", "EncFrozen", "BackboneLoRA", "DiscLR", "L2SP", "LPFT", "AuxSSL", "BCreg"]
FLIGHT = ["ret", "final_dist_m", "mean_dist_m", "frac_within_05", "reached_05", "reach_time_s", "diverged",
          "collided", "timed_out", "max_tilt_deg", "mean_angvel", "action_rate", "action_sat", "noise", "lr"]
rng = np.random.default_rng(2026)
B = 4000

def paired_arrays(b, stage, obj, mode, K):
    d = b[(b.stage == stage) & (b.K == K) & (b["mode"] == mode) & (b.init.isin([obj, "random"]))]
    piv = d.pivot_table(index=["task", "seed"], columns="init", values="ret")
    if obj not in piv or "random" not in piv or piv.isna().any().any():
        return None
    tasks = sorted(set(piv.index.get_level_values(0)))
    seeds = sorted(set(piv.index.get_level_values(1)))
    if any((t, s) not in piv.index for t in tasks for s in seeds):
        return None
    P = np.array([[piv.loc[(t, s), obj] for s in seeds] for t in tasks])
    R = np.array([[piv.loc[(t, s), "random"] for s in seeds] for t in tasks])
    return P, R

def prob_improvement(b, stage, obj, mode, K):
    arr = paired_arrays(b, stage, obj, mode, K)
    if arr is None:
        return None
    P, R = arr
    n = P.shape[1]

    def stat(idx):
        p, r = P[:, idx][:, :, None], R[:, idx][:, None, :]
        return float(np.mean((p > r) + 0.5 * (p == r)))
    pt = stat(np.arange(n))
    bs = [stat(rng.integers(0, n, n)) for _ in range(B)]
    return [round(pt, 3), *np.round(np.percentile(bs, [2.5, 97.5]), 3).tolist()]

def delta(b, stage, Ac, Bc, K, seeds=None, col="ret", den_col="ret"):
    """Mean over settings of (X(Ac) - X(Bc)) / R(random, FullFT) with a joint seed bootstrap; Ac, Bc = (init, mode).
    X is the terminal return (col='ret') or a secondary measure of section 12 ('peak', 'auc', 'decline'); the
    denominator is always the terminal return of random+FullFT."""
    d = b[(b.stage == stage) & (b.K == K)]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    d = d.assign(cond=d.init + "|" + d["mode"])
    names = ["|".join(Ac), "|".join(Bc), "random|FullFT"]
    piv = d[d.cond.isin(names)].pivot_table(index=["task", "seed"], columns="cond", values=col)
    den = d[d.cond == "random|FullFT"].pivot_table(index=["task", "seed"], columns="cond", values=den_col)
    if any(n not in piv for n in names) or piv[names].isna().any().any() or "random|FullFT" not in den:
        return None
    tasks = sorted(set(piv.index.get_level_values(0)))
    sd = sorted(set(piv.index.get_level_values(1)))
    if any((t, x) not in piv.index or (t, x) not in den.index for t in tasks for x in sd):
        return None
    M = {n: np.array([[piv.loc[(t, x), n] for x in sd] for t in tasks]) for n in names[:2]}
    Z = np.array([[den.loc[(t, x), "random|FullFT"] for x in sd] for t in tasks])

    def stat(idx):
        z = Z[:, idx].mean(1)
        return float(np.mean((M[names[0]][:, idx].mean(1) - M[names[1]][:, idx].mean(1)) / z))
    pt = stat(np.arange(len(sd)))
    bs = [stat(rng.integers(0, len(sd), len(sd))) for _ in range(B)]
    return [round(pt, 4), *np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist()]

def improvement(b, stage, obj, mode, K):
    d = b[(b.stage == stage) & (b.K == K) & (b["mode"] == mode) & (b.init.isin([obj, "random"]))]
    piv = d.pivot_table(index=["task", "seed"], columns="init", values="ret")
    if obj not in piv or "random" not in piv or piv.isna().any().any():
        return None
    tasks = sorted(set(piv.index.get_level_values(0)))
    seeds = sorted(set(piv.index.get_level_values(1)))
    if any((t, s) not in piv.index for t in tasks for s in seeds):
        return None
    P = np.array([[piv.loc[(t, s), obj] for s in seeds] for t in tasks])
    R = np.array([[piv.loc[(t, s), "random"] for s in seeds] for t in tasks])

    def stat(idx):
        return float(np.mean(100 * (P[:, idx].mean(1) / R[:, idx].mean(1) - 1)))
    pt = stat(np.arange(len(seeds)))
    bs = [stat(rng.integers(0, len(seeds), len(seeds))) for _ in range(B)]
    return [round(pt, 1), *np.round(np.percentile(bs, [2.5, 97.5]), 1).tolist()]

SECONDARY = ["peak", "auc", "decline"]
TASKS_R2R3 = ["state_setpoint_x500_clean", "recovery_x500", "agg_x500", "ss_lmf2_clean", "recovery_lmf2", "agg_lmf2"]

def with_reused(df, sel_path=None):
    """Stage-2 cells that equal a finished stage-1 run (same setting, initialization, mode, seed, and budget) are not
    rerun (run_study2.reusable); copy those stage-1 rows as stage-2 rows so that stage 2 is complete."""
    sel_path = sel_path or A.parent / "stage2_selection.json"
    if not sel_path.exists() or "stage" not in df:
        return df
    sel = json.loads(sel_path.read_text())
    if int(sel["iters"]) != 600:
        return df
    keep = {(i, "BCreg" if (i, m) == ("BC", "AuxSSL") else m)
            for i in ("random", "MAE", sel["objective"]) for m in ("FullFT", sel["mode"])}
    s1 = df[(df.stage == 1) & df.seed.isin([7, 42, 99]) & df.task.isin(TASKS_R2R3)]
    s1 = s1[[(i, m) in keep for i, m in zip(s1.init, s1["mode"])]]
    have = set(zip(*[df[df.stage == 2][c] for c in ("task", "init", "mode", "seed")]))
    s1 = s1[[k not in have for k in zip(s1.task, s1.init, s1["mode"], s1.seed)]]
    return pd.concat([df, s1.assign(stage=2)], ignore_index=True)

def add_secondary(b, s):
    """Section 12: peak of the 30-iteration moving average, mean logged return over the run, and peak minus the
    terminal return (iterations 541-600), attached to the K = 600 rows of 600-iteration runs."""
    rows = []
    for rid, g in s[s.iters == 600].sort_values("iter").groupby("run_id"):
        r = g.ret.reset_index(drop=True)
        ma = r.rolling(30, min_periods=30).mean()
        term = r[(g.iter.values > 540) & (g.iter.values <= 600)].mean()
        rows.append(dict(run_id=rid, peak=float(ma.max()), auc=float(r.mean()), decline=float(ma.max() - term)))
    sec = pd.DataFrame(rows, columns=["run_id"] + SECONDARY)
    b = b.drop(columns=[c for c in SECONDARY if c in b]).merge(sec, on="run_id", how="left")
    b.loc[b.K != 600, SECONDARY] = np.nan
    return b

EARLY = ["e25", "e100"]

def add_early_eval(b):
    """Section 12: mean-action fixed-state evaluation return at checkpoints 25 and 100 (e25, e100)
    and the mean over 550/575/600 (eterm), without wind, attached to the K = 600 rows."""
    p = A / "eval_fixed.csv"
    if not p.exists():
        return b.assign(e25=np.nan, e100=np.nan, eterm=np.nan)
    ev = pd.read_csv(p)
    ev = ev[ev.disturb == 0]
    piv = ev.pivot_table(index="run_id", columns="iter", values="ret")
    e = pd.DataFrame({"run_id": piv.index,
                      "e25": piv.get(25, pd.Series(np.nan, index=piv.index)).values,
                      "e100": piv.get(100, pd.Series(np.nan, index=piv.index)).values,
                      "eterm": piv.reindex(columns=[550, 575, 600]).mean(1, skipna=False).values})
    b = b.drop(columns=[c for c in ["e25", "e100", "eterm"] if c in b]).merge(e, on="run_id", how="left")
    b.loc[b.K != 600, ["e25", "e100", "eterm"]] = np.nan
    return b

def decision_set(b, stage, sel, seeds=None, col="ret"):
    """D1-D4 of section 6 for the selected objective and mode (BC runs Aux-SSL as BC-reg, section 11)."""
    inits, modes = ["MAE", sel["objective"]], ["FullFT", sel["mode"]]

    def cm(i, m):
        return (i, "BCreg" if (i, m) == ("BC", "AuxSSL") else m)
    den = "eterm" if col in EARLY else "ret"
    D = {}
    for m in modes:
        D[f"D1|{m}"] = delta(b, stage, ("MAE", m), ("random", m), 600, seeds, col, den)
        D[f"D2|{m}"] = delta(b, stage, cm(sel["objective"], m), ("random", m), 600, seeds, col, den)
    for i in inits:
        D[f"D3|{i}"] = delta(b, stage, cm(i, sel["mode"]), (i, "FullFT"), 600, seeds, col, den)
        for m in modes:
            D[f"D4|{i}|{m}"] = delta(b, stage, cm(i, m), ("random", "FullFT"), 600, seeds, col, den)
    return D

def main():
    b = pd.read_csv(A / "budgets.csv")
    s = pd.read_csv(A / "series.csv")
    b = with_reused(add_early_eval(add_secondary(b, s)))
    s = with_reused(s)
    R = {"delta": {}, "decisions": {}, "floor": {}, "improvement": {}, "prob_improvement": {}, "mode_return": {},
         "flight": {}, "traces": {}, "counts": {}, "secondary": {}, "decisions_secondary": {}, "s4": {}}
    for stage in sorted(b.stage.unique()):
        st = str(stage)
        R["counts"][st] = b[(b.stage == stage) & (b.K == 100)].groupby(["init", "mode"]).size().to_dict()
        R["counts"][st] = {f"{k[0]}|{k[1]}": v for k, v in R["counts"][st].items()}
        for o in OBJ:
            for m in MODES:
                for K in (100, 300, 600):
                    v = improvement(b, stage, o, m, K)
                    if v is not None:
                        R["improvement"].setdefault(st, {}).setdefault(o, {}).setdefault(m, {})[str(K)] = v
                    v = prob_improvement(b, stage, o, m, K)
                    if v is not None:
                        R["prob_improvement"].setdefault(st, {}).setdefault(o, {}).setdefault(m, {})[str(K)] = v
        for o in OBJ:
            for m in MODES:
                for K in (100, 300, 600):
                    for key, Bc in (("vs_random", ("random", m)), ("vs_scratch", ("random", "FullFT")),
                                    ("vs_fullft", (o, "FullFT"))):  # vs_fullft: the mode against FullFT, same backbone
                        if key == "vs_fullft" and m == "FullFT":
                            continue
                        v = delta(b, stage, (o, m), Bc, K)
                        if v is not None:
                            R["delta"].setdefault(st, {}).setdefault(key, {}).setdefault(f"{o}|{m}", {})[str(K)] = v
        for o in OBJ:
            for m in MODES:
                for key, Bc in (("vs_random", ("random", m)), ("vs_scratch", ("random", "FullFT"))):
                    for col in SECONDARY:
                        v = delta(b, stage, (o, m), Bc, 600, None, col)
                        if v is not None:
                            R["secondary"].setdefault(st, {}).setdefault(key, {}).setdefault(f"{o}|{m}", {})[col] = v
                    for col in EARLY:  # early fixed-state evaluation in units of the scratch evaluation return
                        v = delta(b, stage, (o, m), Bc, 600, None, col, den_col="eterm")
                        if v is not None:
                            R["secondary"].setdefault(st, {}).setdefault(key, {}).setdefault(f"{o}|{m}", {})[col] = v
        for m in MODES[1:]:  # random backbone: each mode against random+FullFT
            for K in (100, 300, 600):
                v = delta(b, stage, ("random", m), ("random", "FullFT"), K)
                if v is not None:
                    R["delta"].setdefault(st, {}).setdefault("vs_fullft", {}).setdefault(f"random|{m}", {})[str(K)] = v
        f6 = b[(b.stage == stage) & (b.K == 600)]
        if "reached_05" in f6:
            reach = f6.groupby(["task", "init", "mode"]).reached_05.mean().groupby("task").max()
            R["floor"][st] = sorted(reach[reach < 0.05].index.tolist())
        sel_path = A.parent / "stage2_selection.json"
        if stage == 2 and sel_path.exists():
            sel = json.loads(sel_path.read_text())
            new7 = [13, 21, 123, 256, 512, 777, 2024]
            for tag, seeds in (("all", None), ("new_seeds", new7)):
                R["decisions"][tag] = decision_set(b, stage, sel, seeds)
                R["decisions_secondary"][tag] = {col: decision_set(b, stage, sel, seeds, col) for col in SECONDARY + EARLY}
        if stage == 4 and sel_path.exists():
            # S4 (section 12): the same comparisons with C1 (stage 4) and with C3 on the same seeds and settings
            # (stage 1), side by side
            sel = json.loads(sel_path.read_text())
            for col in ["ret"] + SECONDARY + EARLY:
                R["s4"][col] = {"C1": decision_set(b, 4, sel, None, col), "C3_stage1": decision_set(b, 1, sel, [7, 42, 99], col)}
        ref = b[(b.stage == stage) & (b.init == "random") & (b["mode"] == "FullFT") & (b.K == 600)].groupby("task").ret.mean()
        bb = b[b.stage == stage].copy()
        bb["norm"] = bb.ret / bb.task.map(ref)
        for (i, m, K), g in bb.groupby(["init", "mode", "K"]):
            if g.norm.notna().all():
                R["mode_return"].setdefault(st, {}).setdefault(i, {}).setdefault(m, {})[str(K)] = round(float(g.groupby("task").norm.mean().mean()), 3)
        f6 = b[(b.stage == stage) & (b.K == 600)].copy()
        f6["T"] = f6.task.str.extract(r"^(state_setpoint|ss|recovery|agg)")[0].map({"state_setpoint": "T1", "ss": "T1", "recovery": "T2", "agg": "T3"})
        for (T, i, m), g in f6.groupby(["T", "init", "mode"]):
            R["flight"].setdefault(st, {}).setdefault(T, {}).setdefault(i, {})[m] = {c: (round(float(g[c].mean()), 4) if c in g and g[c].notna().any() else None) for c in FLIGHT}
        ss = s[s.stage == stage]
        for (i, m), g in ss.groupby(["init", "mode"]):
            med = g.groupby("iter")[["lr", "kl", "noise"]].median()
            R["traces"].setdefault(st, {}).setdefault(i, {})[m] = {c: med[c].round(6).tolist() for c in med}
    json.dump(R, open(A / "results.json", "w"))
    for st, dd in R["delta"].items():
        for key, conds in dd.items():
            print(f"stage {st} delta {key} (units of the scratch return), K=100/300/600:")
            for cnd, v in conds.items():
                print(f"  {cnd:22s} " + "  ".join(f"{K}:{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for K, x in v.items()))
    if R["decisions"]:
        print("decisions:", json.dumps(R["decisions"], indent=1))
    print("floor settings:", R["floor"])
    imp = R["improvement"]
    for st in imp:
        print(f"stage {st} improvement over random counterpart (%), K=100/300/600:")
        for o in imp[st]:
            for m in imp[st][o]:
                print(f"  {o:5s} {m:13s} " + "  ".join(f"{K}:{v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for K, v in imp[st][o][m].items()))
    print("counts:", R["counts"])

if __name__ == "__main__":
    main()
