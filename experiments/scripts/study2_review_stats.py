"""Re-analyses requested by the reviews of manuscript v12.3 (pre-registration v2, section 25, analysis only).

A  Confirmation (stage 2): the ten pre-registered contrasts D1-D4 (section 6) with seed-level values. For seed s,
   delta_s = mean over settings e of (R_e,s(A) - R_e,s(B)) / mean_s R_e(random, FullFT); the mean of delta_s equals the
   point estimate of section 6. Reported: per-seed values, Student-t 95% interval over seeds, two-sided p, Holm
   adjustment over the ten contrasts, and the probability of improvement P(A > B) over paired (setting, seed) cells
   with a seed bootstrap. For the nine-setting aggregate and for each task family (T1, T2, T3), ten seeds and the seven
   new seeds, at 600 and 100 iterations.
B  Screening (stage 1, three seeds): every mode on MAE and AP (run name PIDM) against scratch and against FullFT of the
   same backbone, by task family, with seed values and t intervals (two degrees of freedom).
C  Selection rule of section 4 recomputed on the backbones common to all modes (seven, without BC), with bootstrap
   probabilities of selection (seeds resampled jointly over settings).
D  Break-even: number of RL runs over which one pretraining pays for itself in T1 and T2, from the iterations saved
   (time_to_level.csv) and measured costs (pretraining logs, run wall times of the ledger).
E  Time to the final level of scratch: median of ratios and ratio of medians over q, the seven new seeds alone, and,
   for runs that never reach the level, the share reaching it by iteration (cumulative incidence).
F  T3 per run: share of pretrained runs above the best scratch run of the same setting (Clopper-Pearson), and the
   exact upper bound of the take-off probability of compute-matched scratch.
G  Steady wind (section 8): D4-type contrasts of the fixed-state evaluation return and reach under winds L = 1, 2.
H  Second domain (stage 15): LoRA and FullFT against scratch per task with five seeds.
Output: logs/study2/analysis/review_stats.json and a printed report.
Usage: python study2_review_stats.py [--parts A B C D E F G H]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import argparse
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, _EXP_ROOT + "/scripts")
import study2_analyze as AN  # noqa: E402
import study2_supp as S  # noqa: E402

A_DIR = S.A
S10 = [7, 13, 21, 42, 99, 123, 256, 512, 777, 2024]
NEW7 = [13, 21, 123, 256, 512, 777, 2024]
S3 = [7, 42, 99]
FAM = {"T1": ("state_setpoint", "ss_"), "T2": ("recovery",), "T3": ("agg",)}
rng = np.random.default_rng(25)
B = 2000

def fam_tasks(tasks, fam):
    return [t for t in tasks if fam is None or t.startswith(FAM[fam])]

def cells(b, stage, K, seeds, tasks, Ac, Bc, col="ret"):
    d = b[(b.stage == stage) & (b.K == K) & b.seed.isin(seeds) & b.task.isin(tasks)]
    d = d.assign(cond=d.init + "|" + d["mode"])
    names = ["|".join(Ac), "|".join(Bc), "random|FullFT"]
    piv = d[d.cond.isin(names)].pivot_table(index=["task", "seed"], columns="cond", values=col)
    if any(n not in piv for n in names):
        return None
    piv = piv[list(dict.fromkeys(names))]  # Bc may be scratch itself (D4)
    full = pd.MultiIndex.from_product([sorted(set(tasks)), sorted(seeds)], names=["task", "seed"])
    piv = piv.reindex(full)
    if piv.isna().any().any():
        return None
    return piv, names

def contrast(b, stage, K, seeds, tasks, Ac, Bc, col="ret"):
    r = cells(b, stage, K, seeds, tasks, Ac, Bc, col)
    if r is None:
        return None
    piv, names = r
    z = piv[names[2]].groupby("task").mean()
    diff = (piv[names[0]] - piv[names[1]]) / piv.index.get_level_values("task").map(z).values
    per_seed = diff.groupby("seed").mean()
    n = len(per_seed)
    m, sd = float(per_seed.mean()), float(per_seed.std(ddof=1))
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else 0.0
    win = (piv[names[0]] > piv[names[1]]).astype(float) + 0.5 * (piv[names[0]] == piv[names[1]])
    sd_list = sorted(seeds)
    w_seed = win.groupby("seed").mean()
    bs = [float(w_seed.loc[rng.choice(sd_list, len(sd_list))].mean()) for _ in range(B)]
    return dict(mean=round(m, 4), t95=[round(m - tc * se, 4), round(m + tc * se, 4)], p=p,
                seeds={int(k): round(float(v), 4) for k, v in per_seed.items()},
                range=[round(float(per_seed.min()), 4), round(float(per_seed.max()), 4)],
                prob_improvement=round(float(win.mean()), 3),
                pi95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n_seeds=n)

def holm(ps):
    keys = sorted(ps, key=ps.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out

def fmt(r):
    if r is None:
        return "n/a"
    return (f"{r['mean']:+.3f} t95 [{r['t95'][0]:+.3f}, {r['t95'][1]:+.3f}] p={r['p']:.4f} "
            f"range [{r['range'][0]:+.2f}, {r['range'][1]:+.2f}] PI {r['prob_improvement']:.2f} "
            f"[{r['pi95'][0]:.2f}, {r['pi95'][1]:.2f}]")

def load_budgets():
    b = pd.read_csv(A_DIR / "budgets.csv")
    return AN.with_reused(AN.add_early_eval(b))

def part_a(b, out):
    sel = json.load(open(A_DIR.parent / "stage2_selection.json"))
    o, mo = sel["objective"], sel["mode"]
    con = {}
    for m in ("FullFT", mo):
        con[f"D1|{m}"] = (("MAE", m), ("random", m))
        con[f"D2|{m}"] = ((o, m), ("random", m))
    for i in ("MAE", o):
        con[f"D3|{i}"] = ((i, mo), (i, "FullFT"))
        for m in ("FullFT", mo):
            con[f"D4|{i}|{m}"] = ((i, m), ("random", "FullFT"))
    T9 = sorted(b[(b.stage == 2) & (b.K == 600)].task.unique())
    res = {}
    for K in (600, 100):
        for stag, seeds in (("10 seeds", S10), ("7 new seeds", NEW7)):
            for fam in (None, "T1", "T2", "T3"):
                key = f"K{K}|{stag}|{fam or '9 settings'}"
                res[key] = {c: contrast(b, 2, K, seeds, fam_tasks(T9, fam), *ab) for c, ab in con.items()}
                ps = {c: r["p"] for c, r in res[key].items() if r is not None}
                if len(ps) == len(con):
                    for c, ph in holm(ps).items():
                        res[key][c]["p_holm"] = round(ph, 4)
    print("\n== A. Confirmatory contrasts D1-D4 (stage 2), seed-level t intervals, Holm over the ten contrasts")
    for key, rr in res.items():
        print(f"-- {key}")
        for c, r in rr.items():
            ph = "" if r is None or "p_holm" not in r else f" Holm p={r['p_holm']:.4f}"
            print(f"   {c:16s} {fmt(r)}{ph}")
    out["A"] = res

def part_b(b, out):
    T6 = sorted(b[(b.stage == 1) & (b.K == 600)].task.unique())
    modes = ["FullFT", "LateFT", "EncFrozen", "BackboneLoRA", "LPFT", "DiscLR", "L2SP", "AuxSSL"]
    res = {}
    print("\n== B. Screening (stage 1, seeds 7, 42, 99): against scratch | against FullFT of the same backbone")
    for K in (600, 100):
        for fam in (None, "T1", "T2", "T3"):
            for i in ("random", "MAE", "PIDM"):
                for m in modes:
                    if (i, m) == ("random", "FullFT"):
                        continue
                    k = f"K{K}|{fam or '6 settings'}|{i}|{m}"
                    res[k] = dict(vs_scratch=contrast(b, 1, K, S3, fam_tasks(T6, fam), (i, m), ("random", "FullFT")),
                                  vs_fullft=None if m == "FullFT" else
                                  contrast(b, 1, K, S3, fam_tasks(T6, fam), (i, m), (i, "FullFT")))
                    if K == 600 and fam is None:
                        print(f"   {i:6s} {m:12s} {fmt(res[k]['vs_scratch'])} | {fmt(res[k]['vs_fullft'])}")
    out["B"] = res

def part_c(b, out):
    PRE7 = ["MAE", "CPC", "SPR", "PIDM", "IDM", "FDM", "MAEs"]
    PRE8 = PRE7 + ["BC"]
    MODES = ["LateFT", "EncFrozen", "BackboneLoRA", "DiscLR", "L2SP", "LPFT", "AuxSSL"]
    d = b[(b.stage == 1) & (b.K == 600)]
    tasks = sorted(d.task.unique())

    def scores(seeds, common):
        dd = d[d.seed.isin(seeds)]
        cell = dd.groupby(["task", "init", "mode"]).ret.mean()
        ref = cell.xs(("random", "FullFT"), level=("init", "mode"))
        Sx = (cell / cell.index.get_level_values("task").map(ref).values).rename("S").reset_index()
        sc = {}
        for m in MODES:
            objs = PRE7 if (common or m == "AuxSSL") else PRE8
            v = Sx[(Sx["mode"] == m) & Sx.init.isin(objs)].S
            sc[m] = float(v.mean())
        return sc
    point = {"preregistered": scores(S3, False), "common_7": scores(S3, True)}
    prob = {}
    for tag, common in (("preregistered", False), ("common_7", True)):
        wins = []
        for _ in range(B):
            ss = list(rng.choice(S3, 3))
            # a seed drawn twice counts twice: duplicate its rows
            sc = scores_dup(d, ss, common, PRE7, PRE8, MODES)
            wins.append(max(sc, key=sc.get))
        prob[tag] = {m: round(wins.count(m) / B, 3) for m in MODES}
    out["C"] = dict(point=point, selection_probability=prob)
    print("\n== C. Mode selection, pre-registered averaging vs common seven backbones")
    for tag in point:
        top = sorted(point[tag].items(), key=lambda x: -x[1])[:4]
        print(f"   {tag:14s} " + ", ".join(f"{m} {v:.3f}" for m, v in top) + f" | P(selected) {prob[tag]}")

def scores_dup(d, seeds, common, PRE7, PRE8, MODES):
    dd = pd.concat([d[d.seed == s] for s in seeds])
    cell = dd.groupby(["task", "init", "mode"]).ret.mean()
    ref = cell.xs(("random", "FullFT"), level=("init", "mode"))
    Sx = (cell / cell.index.get_level_values("task").map(ref).values).rename("S").reset_index()
    return {m: float(Sx[(Sx["mode"] == m) & Sx.init.isin(PRE7 if (common or m == "AuxSSL") else PRE8)].S.mean())
            for m in MODES}

def pretrain_seconds():
    secs = []
    for f in glob.glob(_EXP_ROOT + "/logs/pretrain_v3/*.log"):
        if not re.search(r"/(MAE|PIDM)__seed\d+\.log$", f):
            continue
        last = [ln for ln in open(f, errors="ignore") if "epoch 25/25" in ln]
        if last:
            secs.append(float(re.search(r"elapsed=([\d.]+)s", last[-1]).group(1)))
    return secs

def part_d(out):
    t = json.load(open(A_DIR / "time_to_level.json"))
    led = pd.read_csv(A_DIR.parent / "ledger.csv")
    led = led[(led.exit == 0) & (led.stage == 2) & (led["mode"] == "FullFT") & ~led.task.str.startswith("agg")]
    run_s = float(led.wall_s.median())
    pre = pretrain_seconds()
    res = dict(pretrain_seconds_median=float(np.median(pre)) if pre else None, n_pretrain_logs=len(pre),
               rl_run_seconds_median_concurrent=run_s, rl_seconds_per_iteration=run_s / 600, by_q={})
    for init in ("MAE", "PIDM"):
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r = t.get(f"confirmation|{init}|q{q}")
            if not r or not pre:
                continue
            saved = r["saved"][0]
            per_run = saved * run_s / 600
            res["by_q"][f"{init}|q{q}"] = dict(iterations_saved=saved, seconds_saved_per_run=round(per_run, 1),
                                               break_even_runs=round(float(np.median(pre)) / per_run, 2) if per_run > 0 else None)
    out["D"] = res
    print("\n== D. Break-even (concurrent wall times; standalone timing replaces them when measured)")
    print(f"   pretraining {res['pretrain_seconds_median']} s (n={len(pre)}), RL run {run_s:.0f} s")
    for k, v in res["by_q"].items():
        print(f"   {k:12s} saved {v['iterations_saved']} it = {v['seconds_saved_per_run']} s/run -> break-even {v['break_even_runs']} runs")

def part_e(out):
    d = pd.read_csv(A_DIR / "time_to_level.csv")
    d1 = pd.read_csv(A_DIR / "time_to_level_screening.csv")
    res = {}
    for init in ("MAE", "PIDM"):
        for tag, seeds in (("10 seeds", S10), ("7 new seeds", NEW7)):
            h = d[(d.init == init) & d.seed.isin(seeds)]
            for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                ok = h[h[f"s{q}"].notna() & h[f"p{q}"].notna()]
                res[f"{init}|{tag}|q{q}"] = dict(
                    median_of_ratios=round(float((ok[f"s{q}"] / ok[f"p{q}"]).median()), 3),
                    ratio_of_medians=round(float(ok[f"s{q}"].median() / ok[f"p{q}"].median()), 3),
                    reached_pre=round(float(h[f"p{q}"].notna().mean()), 3),
                    reached_scratch=round(float(h[f"s{q}"].notna().mean()), 3), n=int(len(h)))
    ci = {}
    grid = list(range(0, 601, 25))
    for (init, mode), h in d1.groupby(["init", "mode"]):
        x = h["p1.0"].values
        ci[f"{init}|{mode}"] = [round(float(np.mean(np.nan_to_num(x, nan=1e9) <= g)), 3) for g in grid]
    sc = d1.drop_duplicates(["task", "seed"])["s1.0"].values
    ci["scratch"] = [round(float(np.mean(np.nan_to_num(sc, nan=1e9) <= g)), 3) for g in grid]
    res["cumulative_incidence_q1.0"] = dict(iterations=grid, curves=ci)
    out["E"] = res
    print("\n== E. Time to the final level of scratch")
    for k, v in res.items():
        if k.endswith("q1.0") or k.endswith("q0.8"):
            print(f"   {k:22s} {v}")

def part_f(b, out):
    from scipy.stats import beta
    res = {}
    ev = pd.read_csv(A_DIR / "eval_fixed.csv")
    ev = ev[(ev.disturb == 0) & ev.iter.isin([550, 575, 600])]
    reach = ev.groupby("run_id").reached_05.mean()
    d = b[(b.stage == 2) & (b.K == 600) & b.task.str.startswith("agg")].copy()
    d["reach"] = d.run_id.map(reach)
    for (init, mode), h in d.groupby(["init", "mode"]):
        if (init, mode) == ("random", "FullFT"):
            continue
        k = n = 0
        for task, g in h.groupby("task"):
            best = d[(d.task == task) & (d.init == "random") & (d["mode"] == "FullFT")].reach.max()
            k += int((g.reach > best).sum())
            n += int(g.reach.notna().sum())
        lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        res[f"{init}|{mode}"] = dict(above_best_scratch=k, n=n, share=round(k / max(n, 1), 3),
                                     cp95=[round(float(lo), 3), round(float(hi), 3)])
    s3 = b[(b.stage == 3) & b.task.str.startswith("agg") & (b.init == "random")]
    if len(s3):
        res["compute_matched_scratch_note"] = "see stage-3 runs at 3,000 iterations"
    # exact one-sided 95% upper bound of a probability after 0 successes in n trials
    for n0 in (3, 6):
        res[f"upper95_after_0_of_{n0}"] = round(float(1 - 0.05 ** (1 / n0)), 3)
    out["F"] = res
    print("\n== F. T3 per run (fixed-state reach, mean of 550/575/600)")
    for k, v in res.items():
        print(f"   {k:28s} {v}")

def part_g(b, out):
    ev = pd.read_csv(A_DIR / "eval_fixed.csv")
    ev = ev[ev.iter.isin([550, 575, 600])]
    res = {}
    for L in (0.0, 1.0, 2.0):
        e = ev[ev.disturb == L].groupby("run_id")[["ret", "reached_05", "final_dist_m"]].mean()
        bb = b[(b.stage == 2) & (b.K == 600)].copy()
        for c in e.columns:
            bb[f"w_{c}"] = bb.run_id.map(e[c])
        T9 = sorted(bb.task.unique())
        for fam in (None, "T1", "T2", "T3"):
            for i in ("MAE", "PIDM"):
                res[f"L{L:g}|{fam or '9 settings'}|{i}|FullFT"] = contrast(bb, 2, 600, S10, fam_tasks(T9, fam),
                                                                            (i, "FullFT"), ("random", "FullFT"),
                                                                            col="w_ret")
    out["G"] = res
    print("\n== G. Steady wind (fixed-state evaluation return, D4 FullFT, ten seeds)")
    for k, v in res.items():
        if "9 settings" in k:
            print(f"   {k:30s} {fmt(v)}")

def part_h(b, out):
    res = {}
    d = b[(b.stage == 15) & (b.K == 600)]
    seeds = sorted(d.seed.unique())
    for task in sorted(d.task.unique()):
        for i in ("random", "MAE", "PIDM"):
            for m in ("FullFT", "EncFrozen", "BackboneLoRA"):
                if (i, m) == ("random", "FullFT"):
                    continue
                res[f"{task}|{i}|{m}"] = contrast(b, 15, 600, seeds, [task], (i, m), ("random", "FullFT"))
    out["H"] = res
    print("\n== H. Second domain (quadruped), against scratch, five seeds")
    for k, v in res.items():
        print(f"   {k:40s} {fmt(v)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=list("ABCDEFGH"))
    a = ap.parse_args()
    p = A_DIR / "review_stats.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    b = load_budgets()
    for part in a.parts:
        {"A": lambda: part_a(b, out), "B": lambda: part_b(b, out), "C": lambda: part_c(b, out),
         "D": lambda: part_d(out), "E": lambda: part_e(out), "F": lambda: part_f(b, out),
         "G": lambda: part_g(b, out), "H": lambda: part_h(b, out)}[part]()
    p.write_text(json.dumps(out, indent=1))
    print("written", p)

if __name__ == "__main__":
    main()
