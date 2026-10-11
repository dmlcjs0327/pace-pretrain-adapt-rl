"""Supplementary analyses of Study 2.

S1 (LP-FT) is part of the screening grid and is reported by study2_analyze.py ('vs_fullft', 'vs_random', 'vs_scratch').
dynamics variation (stage 3, settings ss_dr_*, recovery_dr_*):
  for every backbone and mode, the gain over the random counterpart and over training from scratch in the four
  settings with hidden dynamics variation minus the same gain in the four fixed settings of the same task and
  airframe (stage 1), mean over the four settings; seed bootstrap (4,000 draws, joint across settings and stages).
compute-matched scratch (stage 3, random+FullFT trained to 3,000 iterations):
  for every combination of stage 1, [R_600(objective, mode) - R_K(scratch)] / R_600(scratch, stage 1), mean over the
  six settings, with K = 600 r + 600 c (cost not shared) and K = 600 r + 100 c (shared by the six settings); c is
  the cost of data collection and pretraining in 600-iteration FullFT runs (plus the head refit of Aux-SSL), r the
  time of a run of the mode relative to FullFT. Primary costs: standalone measurement (measure_cost.py, correction
  of 18:13); the costs measured with concurrent jobs are kept as a record. Returns at K: window (0.9K, K].
Writes analysis/supp.json. Usage: python study2_supp.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(_EXP_ROOT + "")
A = EXP / "logs" / "study2" / "analysis"
PRE_LOGS = EXP / "logs" / "pretrain_v3"
RL_RUN_S = 828.0   # one 600-iteration RL run at the measurement load (section 10)
# Pretraining cost in RL runs: section 10 (original four) and the 17:05 note (extension; FDM over three seeds, 17:12).
COST = {"MAE": 2.18, "PIDM": 2.23, "SPR": 2.66, "CPC": 2.78, "IDM": 2.92, "BC": 2.72, "MAEs": 2.19, "FDM": 3.00}
HEAD_REFIT = ["MAE", "CPC", "SPR", "PIDM"]
EXTRA_COST_MODES = ["AuxSSL", "L2SP", "BCreg"]
NEW_BACKBONES = ["IDM", "FDM", "BC", "MAEs"]
DR_PAIR = {"ss_dr_x500": "state_setpoint_x500_clean", "ss_dr_lmf2": "ss_lmf2_clean",
           "recovery_dr_x500": "recovery_x500", "recovery_dr_lmf2": "recovery_lmf2"}
rng = np.random.default_rng(2026)
B = 4000

def refit_cost():
    """Head refit time of Aux-SSL in RL runs, mean over seeds."""
    out = {}
    for o in HEAD_REFIT:
        t = []
        for f in glob.glob(str(PRE_LOGS / f"{o}_head__seed*.log")):
            e = re.findall(r"epoch (\d+)/(\d+) .*elapsed=([\d.]+)s", Path(f).read_text(errors="ignore"))
            if e and e[-1][0] == e[-1][1]:
                t.append(float(e[-1][2]))
        if t:
            out[o] = float(np.mean(t)) / RL_RUN_S
    return out

def mode_ratio():
    """r of the 17:05 note: median wall time of the mode in stage 1 / median of FullFT with the new backbones."""
    led = pd.read_csv(EXP / "logs" / "study2" / "ledger.csv")
    led = led[(led.stage.astype(str) == "1") & (led.exit == 0)]
    base = led[(led.init.isin(NEW_BACKBONES)) & (led["mode"] == "FullFT")].wall_s.median()
    return {m: float(led[led["mode"] == m].wall_s.median() / base) for m in EXTRA_COST_MODES
            if (led["mode"] == m).any() and np.isfinite(base)}

def window_return(s, K):
    """Mean logged return over (0.9K, K] per run."""
    w = s[(s.iter > 0.9 * K) & (s.iter <= K)]
    return w.groupby(["task", "seed"]).ret.mean()

def boot(fn, seeds):
    pt = fn(np.array(seeds))
    bs = [fn(rng.choice(seeds, len(seeds))) for _ in range(B)]
    return [round(float(pt), 4), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 4).tolist()]

def dynamics_variation(b):
    b6 = b[b.K == 600]
    out = {}
    for (i, m), _ in b6[(b6.stage == 3) & (b6.init != "random")].groupby(["init", "mode"]):
        def table(stage, tasks):
            d = b6[(b6.stage == stage) & (b6.task.isin(tasks))]
            piv = d.assign(c=d.init + "|" + d["mode"]).pivot_table(index=["task", "seed"], columns="c", values="ret")
            return piv
        dr, fx = table(3, list(DR_PAIR)), table(1, list(DR_PAIR.values()))
        need = [f"{i}|{m}", f"random|{m}", "random|FullFT"]
        if any(n not in dr or n not in fx for n in need):
            continue
        seeds = sorted(set(dr.index.get_level_values(1)) & set(fx.index.get_level_values(1)))
        res = {}
        for key, ref in (("vs_random", f"random|{m}"), ("vs_scratch", "random|FullFT")):
            def stat(pick, ref=ref):
                vals = []
                for t_dr, t_fx in DR_PAIR.items():
                    g = []
                    for piv, t in ((dr, t_dr), (fx, t_fx)):
                        x = piv.loc[t].reindex(pick)
                        g.append((x[f"{i}|{m}"].mean() - x[ref].mean()) / x["random|FullFT"].mean())
                    vals.append(g[0] - g[1])
                return np.nanmean(vals)
            res[key] = boot(stat, seeds)
        out[f"{i}|{m}"] = res
    return out

def costs_standalone():
    """Primary costs (correction of 18:13): standalone wall times from measure_cost.py, in FullFT RL runs."""
    p = A / "cost_standalone.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ff = d["t_rl"]["random|FullFT"]
    return {"c": {o: (d["t_collect"] + v) / ff for o, v in d["t_pre"].items()},
            "refit": {o: v / ff for o, v in d["t_refit"].items()},
            "r": {"L2SP": d["t_rl"]["random|L2SP"] / ff, "BCreg": d["t_rl"]["BC|BCreg"] / ff,
                  "AuxSSL": {k.split("|")[0]: v / ff for k, v in d["t_rl"].items() if k.endswith("|AuxSSL")}},
            "source": "standalone (measure_cost.py)"}

def costs_concurrent():
    """Record only: the costs measured with concurrent jobs (section 10 and the notes of 17:05 and 17:12)."""
    return {"c": COST, "refit": refit_cost(), "r": mode_ratio(), "source": "concurrent (record only)"}

def total_time(cost, i, m):
    """Total time in 600-iteration FullFT runs: (RL part r, pretraining part c) for backbone i in mode m."""
    c = cost["c"].get(i)
    if c is None:
        return None
    c = c + (cost["refit"].get(i, 0.0) if m == "AuxSSL" else 0.0)
    r = cost["r"].get(m, 1.0)
    if isinstance(r, dict):
        r = r.get(i)
        if r is None:
            return None
    return r, c

def compute_matched(s, b, cost):
    s3 = s[(s.stage == 3) & (s.init == "random") & (s["mode"] == "FullFT") & (s.iters >= 3000)]
    if s3.empty or cost is None:
        return {}
    b6 = b[(b.stage == 1) & (b.K == 600)]
    scr = b6[(b6.init == "random") & (b6["mode"] == "FullFT")].set_index(["task", "seed"]).ret
    out = {"cost": cost, "rows": {}}
    for (i, m), g in b6[b6.init != "random"].groupby(["init", "mode"]):
        rc = total_time(cost, i, m)
        if rc is None:
            continue
        rr, c = rc
        R = g.set_index(["task", "seed"]).ret
        row = {"x_not_shared": rr + c, "x_shared_6": rr + c / 6}
        for tag, K in (("not_shared", 600 * rr + 600 * c), ("shared_6", 600 * rr + 100 * c)):
            K = int(round(K))
            if K > s3.iter.max():
                row[tag] = {"K": K, "delta": None, "note": "beyond the horizon of the scratch runs"}
                continue
            RK = window_return(s3, K)
            seeds = sorted(set(R.index.get_level_values(1)) & set(RK.index.get_level_values(1)))
            tasks = sorted(set(R.index.get_level_values(0)) & set(RK.index.get_level_values(0)))
            if not seeds or not tasks:
                continue

            def stat(pick):
                return np.nanmean([(R.loc[t].reindex(pick).mean() - RK.loc[t].reindex(pick).mean()) / scr.loc[t].reindex(pick).mean()
                                   for t in tasks])
            row[tag] = {"K": K, "delta": boot(stat, seeds)}
        out["rows"][f"{i}|{m}"] = row
    return out

def best_of_n_expectation(w, fin, n):
    """Expected final return of the seed with the best early return in a uniformly random n-subset of the seeds:
    the k-th lowest early return (k = 1..m) is the best of the subset with probability C(k-1, n-1) / C(m, n)."""
    from math import comb
    order = sorted(w.index, key=lambda x: w[x])
    m = len(order)
    n = min(n, m)
    return sum(fin[x] * comb(k, n - 1) / comb(m, n) for k, x in enumerate(order))

def selection_baseline(s, b, cost):
    """S5 (section 13): pretrained condition (stage 2) against the expected final return of scratch with seed selection
    under the same total compute. Scratch strategy: n seeds for T_s iterations, keep the one with the best logged return
    over (0.9 T_s, T_s], continue it to 600; n is the largest with n T_s + 600 - T_s <= 600 B (n <= available seeds).
    Expectation over all n-subsets (exact). Values in units of the seed-mean final return of random+FullFT."""
    import study2_analyze as AN
    if cost is None:
        return {}
    AN.A = A
    s2 = AN.with_reused(s)
    s2 = s2[s2.stage == 2]
    b6 = AN.with_reused(b)
    b6 = b6[(b6.stage == 2) & (b6.K == 600)]
    scr = s2[(s2.init == "random") & (s2["mode"] == "FullFT")]
    if scr.empty:
        return {}
    out = {}
    for (i, m), g in b6[b6.init != "random"].groupby(["init", "mode"]):
        rc = total_time(cost, i, m)
        if rc is None:
            continue
        rr, c = rc
        for tag, Bud in (("not_shared", rr + c), ("shared_6", rr + c / 6)):
            for Ts in (50, 100):
                per_task = {}
                for task, gp in g.groupby("task"):
                    sc = scr[scr.task == task]
                    w = sc[(sc.iter > 0.9 * Ts) & (sc.iter <= Ts)].groupby("seed").ret.mean()
                    fin = sc[(sc.iter > 540) & (sc.iter <= 600)].groupby("seed").ret.mean()
                    n = int(min(len(fin), max(1, np.floor((600 * Bud - 600 + Ts) / Ts))))  # at most the seeds run
                    per_task[task] = dict(n=n, w=w, fin=fin, pre=gp.set_index("seed").ret)

                def stat(pick_s, pick_p):
                    vals = []
                    for d in per_task.values():
                        ss = sorted(set(x for x in pick_s if x in d["fin"].index))
                        pp = [x for x in pick_p if x in d["pre"].index]
                        if not ss or not pp:
                            continue
                        exp = best_of_n_expectation(d["w"][ss], d["fin"][ss], d["n"])
                        vals.append((d["pre"][pp].mean() - exp) / d["fin"].mean())
                    return np.mean(vals) if vals else np.nan
                seeds = sorted(set().union(*[set(d["fin"].index) for d in per_task.values()]))
                pt = stat(seeds, seeds)
                bs = [stat(list(rng.choice(seeds, len(seeds))), list(rng.choice(seeds, len(seeds)))) for _ in range(B)]
                out[f"{i}|{m}|{tag}|Ts{Ts}"] = {"B": round(Bud, 3), "n": int(np.median([d["n"] for d in per_task.values()])),
                                                "delta": [round(float(pt), 4), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 4).tolist()]}
    return out

def cumulative_learning(b, ev):
    """S6 (section 13): noise-free learning curve from the fixed-state evaluation (no wind) at 25, 50, 100, ..., 600;
    maximum drawdown and final-minus-best, in units of the final evaluation return of random+FullFT of the setting."""
    if ev is None or ev.empty:
        return {}
    e = ev[(ev.disturb == 0) & (ev.stage == 1)]
    piv = e.pivot_table(index=["run_id", "task", "init", "mode", "seed"], columns="iter", values="ret")
    its = [k for k in [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 575, 600] if k in piv.columns]
    if len(its) < 8:
        return {}
    piv = piv[its].dropna()
    final = piv[[k for k in (550, 575, 600) if k in its]].mean(1)
    curve = piv.values
    runmax = np.maximum.accumulate(curve, axis=1)
    dd = (runmax - curve).max(1)
    fmb = final.values - curve.max(1)
    r = piv.reset_index()[["run_id", "task", "init", "mode", "seed"]].assign(drawdown=dd, final_minus_best=fmb, final=final.values)
    ref = r[(r.init == "random") & (r["mode"] == "FullFT")].groupby("task").final.mean()
    r = r[r.task.isin(ref.index)]
    for c in ("drawdown", "final_minus_best"):
        r[c] = r[c] / r.task.map(ref)
    out = {}
    for (i, m), g in r.groupby(["init", "mode"]):
        cell = g.groupby(["task", "seed"])[["drawdown", "final_minus_best"]].mean()
        seeds = sorted(g.seed.unique())

        def stat(pick, col):
            x = cell[col].unstack("seed").reindex(columns=pick)
            return float(np.nanmean(x.mean(1)))
        out[f"{i}|{m}"] = {col: [round(stat(seeds, col), 4), *np.round(np.nanpercentile(
            [stat(list(rng.choice(seeds, len(seeds))), col) for _ in range(B)], [2.5, 97.5]), 4).tolist()]
            for col in ("drawdown", "final_minus_best")}
        out[f"{i}|{m}"]["n_runs"] = int(len(g))
    r.to_csv(A / "cumulative_learning_runs.csv", index=False)
    return out

def structure_control(b):
    """S7 (section 14): shuffled backbones (stage 5) against the original pretrained backbone, the random backbone,
    and scratch (stage 1, same settings and seeds), G in units of the scratch return (training return at K, early
    evaluation return in units of the scratch final evaluation return)."""
    import study2_analyze as AN
    AN.A = A
    s5 = b[b.stage == 5]
    if s5.empty:
        return {}
    bb = pd.concat([b[b.stage == 1], s5.assign(stage=1)], ignore_index=True)
    out = {}
    for base in ("MAE", "PIDM"):
        sh = base + "shuf"
        for m in ("FullFT", "BackboneLoRA"):
            row = {}
            for key, Bc in (("vs_original", (base, m)), ("vs_random", ("random", m)), ("vs_scratch", ("random", "FullFT"))):
                for K in (100, 300, 600):
                    v = AN.delta(bb, 1, (sh, m), Bc, K)
                    if v is not None:
                        row[f"{key}|ret{K}"] = v
                for col in ("e25", "e100"):
                    v = AN.delta(bb, 1, (sh, m), Bc, 600, None, col, den_col="eterm")
                    if v is not None:
                        row[f"{key}|{col}"] = v
            out[f"{sh}|{m}"] = row
    return out

LADDER = {"agg_d1_x500": ("d1", "R2"), "agg_d1_lmf2": ("d1", "R3"), "agg_d2_x500": ("d2", "R2"),
          "agg_d2_lmf2": ("d2", "R3"), "agg_x500": ("d3", "R2"), "agg_lmf2": ("d3", "R3")}

def difficulty_ladder(b):
    """S8 (section 15): per difficulty level (d1, d2: stage 6; d3: T3 of stage 1) and airframe, the reach rate
    (0.5 m), the final distance, and the terminal return of each condition, and the differences of each pretrained
    condition from its random counterpart and from scratch, with a seed bootstrap; difficulty = reach rate of scratch."""
    d = b[(b.K == 600) & b.task.isin(LADDER) & (((b.stage == 6)) | ((b.stage == 1) & b.task.isin(["agg_x500", "agg_lmf2"])))]
    d = d[d.init.isin(["random", "MAE", "PIDM"]) & d["mode"].isin(["FullFT", "BackboneLoRA"])]
    if d[d.stage == 6].empty:
        return {}
    out = {}
    for task, g in d.groupby("task"):
        lvl, af = LADDER[task]
        cell = g.pivot_table(index="seed", columns=["init", "mode"], values=["reached_05", "final_dist_m", "ret"])
        seeds = list(cell.index)
        row = {"scratch_reach": round(float(cell[("reached_05", "random", "FullFT")].mean()), 4)}
        for i in ("MAE", "PIDM"):
            for m in ("FullFT", "BackboneLoRA"):
                for metric in ("reached_05", "final_dist_m"):
                    for key, ref in (("vs_random", ("random", m)), ("vs_scratch", ("random", "FullFT"))):
                        if (metric, i, m) not in cell or (metric,) + ref not in cell:
                            continue
                        a, r = cell[(metric, i, m)], cell[(metric,) + ref]

                        def stat(p):
                            return float(a.reindex(p).mean() - r.reindex(p).mean())
                        bs = [stat(list(rng.choice(seeds, len(seeds)))) for _ in range(B)]
                        row[f"{i}|{m}|{metric}|{key}"] = [round(stat(seeds), 4), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 4).tolist()]
        out[f"{lvl}|{af}"] = row
    return out

def main():
    b = pd.read_csv(A / "budgets.csv")
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iters", "iter", "ret"])
    ev = pd.read_csv(A / "eval_fixed.csv") if (A / "eval_fixed.csv").exists() else None
    out = {"dynamics_variation": dynamics_variation(b), "compute_matched": compute_matched(s, b, costs_standalone()),
           "compute_matched_concurrent_record": compute_matched(s, b, costs_concurrent()),
           "selection_baseline": selection_baseline(s, b, costs_standalone()),
           "cumulative_learning": cumulative_learning(b, ev)}
    import study2_analyze as AN
    AN.A = A
    out["structure_control"] = structure_control(AN.add_early_eval(b))
    out["difficulty_ladder"] = difficulty_ladder(b)
    (A / "supp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])

if __name__ == "__main__":
    main()
