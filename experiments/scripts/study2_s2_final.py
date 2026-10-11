"""Stage-2 (confirmation) final report: coverage, D1-D4 (pre-registration section 6) with all ten seeds and the seven
new seeds, the floor rule of study2_analyze.py (every condition < 5% reach at 600), R1 (held-out airframe), task
families, and T3 per-run reach (training log and fixed-state evaluation) with the oracle best-of-n of scratch.
Run after study2_parse.py (and after run_eval_progress.sh 2 for the fixed-state columns)."""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, _EXP_ROOT + "/scripts")
import study2_supp as S, study2_analyze as AN

S10 = [7, 13, 21, 42, 99, 123, 256, 512, 777, 2024]
NEW7 = [13, 21, 123, 256, 512, 777, 2024]
CONDS = ["random|FullFT", "random|AuxSSL", "MAE|FullFT", "MAE|AuxSSL", "PIDM|FullFT", "PIDM|AuxSSL"]

def f(v):
    return "n/a" if v is None else f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"

b = AN.with_reused(AN.add_early_eval(pd.read_csv(S.A / "budgets.csv")))
sel = json.load(open(S.A.parent / "stage2_selection.json"))
d6 = b[(b.stage == 2) & (b.K == 600)].assign(cond=lambda x: x.init + "|" + x["mode"])
print("== coverage (stage 2, 600 iterations, reused runs included)")
print(pd.crosstab(d6.cond, d6.seed).reindex(CONDS).to_string())
seeds_done = [s for s in S10 if (d6[d6.seed == s].groupby("cond").size().reindex(CONDS).fillna(0) == 9).all()]
print("complete seeds:", seeds_done)

reach = d6.groupby(["task", "cond"]).reached_05.mean().groupby("task").max()
floor = sorted(reach[reach < 0.05].index)
print("\n== floor rule (max over conditions of mean training reach at 600 < 5%):", floor)
print(reach.round(3).to_string())

for col, lab in (("ret", "training return"), ("eterm", "fixed-state evaluation, mean of 550/575/600")):
    cov = d6.dropna(subset=[col]).groupby(["cond", "seed"]).size().unstack().reindex(CONDS)
    if col == "eterm":
        print(f"\n== coverage of {lab} (settings per condition and seed)")
        print(cov.to_string())
    # decisions only on a complete grid (pivot_table silently drops missing cells otherwise)
    if not (cov.reindex(columns=S10).fillna(0) == 9).all().all():
        print(f"\n== {lab}: grid incomplete, decisions not computed")
        continue
    for tag, seeds in (("10 seeds", S10), ("7 new seeds", NEW7)):
        for ftag, bb in (("9 settings", b), ("floor excluded", b[~b.task.isin(floor)])):
            D = AN.decision_set(bb, 2, sel, seeds, col=col) if col == "ret" else None
            if col == "eterm":
                D = {}
                for m in ("FullFT", "AuxSSL"):
                    D[f"D1|{m}"] = AN.delta(bb, 2, ("MAE", m), ("random", m), 600, seeds, "eterm", "eterm")
                    D[f"D2|{m}"] = AN.delta(bb, 2, ("PIDM", m), ("random", m), 600, seeds, "eterm", "eterm")
                for i in ("MAE", "PIDM"):
                    D[f"D3|{i}"] = AN.delta(bb, 2, (i, "AuxSSL"), (i, "FullFT"), 600, seeds, "eterm", "eterm")
                    for m in ("FullFT", "AuxSSL"):
                        D[f"D4|{i}|{m}"] = AN.delta(bb, 2, (i, m), ("random", "FullFT"), 600, seeds, "eterm", "eterm")
            print(f"\n== D1-D4, {lab}, {tag}, {ftag}")
            for k, v in D.items():
                print(f"  {k:16s} {f(v)}")

T = sorted(d6.task.unique())
fam = {"R1 held out": [t for t in T if "baseQuad" in t], "R2": [t for t in T if "x500" in t],
       "R3": [t for t in T if "lmf2" in t], "T1": [t for t in T if t.startswith(("state_setpoint", "ss_"))],
       "T2": [t for t in T if t.startswith("recovery")], "T3": [t for t in T if t.startswith("agg")],
       "R1 without floor": [t for t in T if "baseQuad" in t and t not in floor]}
print(f"\n== vs scratch by subset, complete seeds {seeds_done}, training return: 600 [CI] | 100")
for name, ts in fam.items():
    if not ts:
        continue
    sub = b[b.task.isin(ts)]
    print(" ", name)
    for c in CONDS[1:]:
        i, m = c.split("|")
        v6 = AN.delta(sub, 2, (i, m), ("random", "FullFT"), 600, seeds_done)
        v1 = AN.delta(sub, 2, (i, m), ("random", "FullFT"), 100, seeds_done)
        print(f"    {c:13s} {f(v6)}   100: {'n/a' if v1 is None else f'{v1[0]:+.2f}'}")

print("\n== T3 reach per run, training log at 600 (seed columns)")
for t in [x for x in T if x.startswith("agg")]:
    print(t)
    print(d6[d6.task == t].pivot_table(index="cond", columns="seed", values="reached_05").reindex(CONDS).round(2).to_string())

ev_path = S.A / "eval_fixed.csv"
ev = pd.read_csv(ev_path)
e2 = ev[(ev.disturb == 0) & (ev.iter >= 550)].copy()
# stage-2 runs reused from stage 1 (R2, R3, seeds 7, 42, 99) keep their stage-1 evaluation
reuse = e2[(e2.stage == 1) & e2.seed.isin([7, 42, 99]) & ~e2.task.str.contains("baseQuad")
           & e2["mode"].isin(["FullFT", "AuxSSL"]) & e2.init.isin(["random", "MAE", "PIDM"])]
e2 = pd.concat([e2[e2.stage == 2], reuse])
e2 = e2.assign(cond=e2.init + "|" + e2["mode"])
term = e2[e2.task.str.startswith("agg")].groupby(["task", "cond", "seed"]).reached_05.mean().reset_index()
print("\n== T3 reach per run, fixed-state evaluation (550/575/600, no wind)")
if term.empty:
    print("  not available yet")
else:
    for t in sorted(term.task.unique()):
        print(t)
        print(term[term.task == t].pivot_table(index="cond", columns="seed", values="reached_05").reindex(CONDS).round(2).to_string())
    print("\n  mean (min-max) over seeds, R2 and R3 tasks")
    for c in CONDS:
        g = term[(term.cond == c) & ~term.task.str.contains("baseQuad")]
        for t, gt in g.groupby("task"):
            print(f"    {c:13s} {t:10s} {100 * gt.reached_05.mean():5.1f}% ({100 * gt.reached_05.min():.0f}-{100 * gt.reached_05.max():.0f}%), n={len(gt)}")
    # oracle best of n scratch runs (exact expectation over n-subsets of the seeds), R2/R3 mean per seed as in Fig. 9b
    sc = term[(term.cond == "random|FullFT") & ~term.task.str.contains("baseQuad")]
    vals = sc.groupby("seed").reached_05.mean().sort_values().values
    m = len(vals)

    def best_of(n):
        return sum(math.comb(k, n - 1) / math.comb(m, n) * vals[k] for k in range(n - 1, m))
    print(f"\n  scratch, oracle best of n over {m} seeds (R2/R3 mean):",
          {n: round(100 * best_of(n), 2) for n in (1, 2, 3, 5, 10) if n <= m})
    for c in ("MAE|FullFT", "PIDM|FullFT"):
        g = term[(term.cond == c) & ~term.task.str.contains("baseQuad")].groupby("seed").reached_05.mean()
        print(f"  one {c} run: mean {100 * g.mean():.1f}%, min {100 * g.min():.1f}%, max {100 * g.max():.1f}%")
