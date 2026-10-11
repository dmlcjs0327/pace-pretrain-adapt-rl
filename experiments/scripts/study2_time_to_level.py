"""Iterations to reach the final level of training from scratch (pre-registration v2, section 24, exploratory).

For every setting and seed, F is the mean training return of training from scratch (random + FullFT) over iterations
541-600. The training return of each run is smoothed with a 10-iteration moving average, and the first iteration at
which it reaches q * F is recorded for q = 0.5, 0.6, ..., 1.0 (no single threshold is chosen). Reported: the median
speed-up (scratch iterations / pretrained iterations) and the median number of iterations saved, with a seed
bootstrap (2,000 resamples); the share of runs that reach the level within 600 iterations; and the change of the
scratch return over the last 100 iterations (convergence check).
  confirmation  stage 2, ten seeds, T1 and T2 on R1-R3, MAE and AP (run name PIDM) with FullFT. Cells that stage 2
                reused from stage 1 (same setting, backbone, mode, seed) are taken from stage 1.
  screening     stage 1, three seeds, T1 and T2 on R2 and R3, every backbone and mode, at q = 0.8 and 1.0.
Output: logs/study2/analysis/time_to_level.json and a printed summary.
Usage: python study2_time_to_level.py
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
QS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2",
       "state_setpoint_baseQuad_clean", "recovery_baseQuad"]
rng = np.random.default_rng(24)

def curves(df):
    out = {}
    for (init, mode), h in df.groupby(["init", "mode"]):
        c = h.sort_values("iter").set_index("iter").ret.interpolate(limit_direction="both")
        out[(init, mode)] = c.rolling(10, min_periods=1).mean()
    return out

def first(c, level):
    hit = c >= level
    return float(hit.idxmax()) if hit.any() else np.nan

def table(s, combos, qs):
    rows = []
    for (task, seed), g in s.groupby(["task", "seed"]):
        cur = curves(g)
        if ("random", "FullFT") not in cur:
            continue
        sc = cur[("random", "FullFT")]
        F = sc.loc[541:600].mean()
        late = (sc.loc[551:600].mean() - sc.loc[451:500].mean()) / abs(F)
        for cb in combos:
            if cb not in cur:
                continue
            r = dict(task=task, seed=seed, init=cb[0], mode=cb[1], late=late, final=cur[cb].loc[541:600].mean() / F)
            for q in qs:
                r[f"s{q}"], r[f"p{q}"] = first(sc, q * F), first(cur[cb], q * F)
            rows.append(r)
    return pd.DataFrame(rows)

def boot(h, q):
    seeds = sorted(h.seed.unique())
    ok = h[h[f"s{q}"].notna() & h[f"p{q}"].notna()]

    def stat(ss):
        x = pd.concat([ok[ok.seed == s] for s in ss])
        return np.median(x[f"s{q}"] / x[f"p{q}"]), np.median(x[f"s{q}"] - x[f"p{q}"])
    pt = stat(seeds)
    bs = np.array([stat(list(rng.choice(seeds, len(seeds)))) for _ in range(2000)])
    return dict(speedup=[round(pt[0], 3), *np.round(np.percentile(bs[:, 0], [2.5, 97.5]), 3).tolist()],
                saved=[round(pt[1], 1), *np.round(np.percentile(bs[:, 1], [2.5, 97.5]), 1).tolist()],
                iters_scratch=float(ok[f"s{q}"].median()), iters_pre=float(ok[f"p{q}"].median()),
                reached=round(float(h[f"p{q}"].notna().mean()), 3), n=int(len(ok)))

def main():
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret"])
    out = {}
    # confirmation: stage 2, filled with the stage-1 runs that stage 2 reused
    s2 = s[(s.stage == 2) & s.task.isin(T12) & (s["mode"] == "FullFT") & s.init.isin(["random", "MAE", "PIDM"])]
    have = set(zip(s2.task, s2.init, s2["mode"], s2.seed))
    s1 = s[(s.stage == 1) & s.task.isin(T12) & (s["mode"] == "FullFT") & s.init.isin(["random", "MAE", "PIDM"])]
    s1 = s1[[k not in have for k in zip(s1.task, s1.init, s1["mode"], s1.seed)]
            & s1.seed.isin(s2.seed.unique())]
    conf = pd.concat([s2, s1])
    d = table(conf, [("MAE", "FullFT"), ("PIDM", "FullFT")], QS)
    d.to_csv(A / "time_to_level.csv", index=False)
    out["confirmation|scratch late change"] = round(float(d.drop_duplicates(["task", "seed"]).late.median()), 4)
    for init in ("MAE", "PIDM"):
        h = d[d.init == init]
        out[f"confirmation|{init}|final over scratch"] = round(float(h.final.median()), 3)
        for q in QS:
            out[f"confirmation|{init}|q{q}"] = boot(h, q)
    # screening: every backbone and mode
    st1 = s[(s.stage == 1) & s.task.isin(T12) & s.seed.isin([7, 42, 99])]
    combos = sorted(set(zip(st1.init, st1["mode"])) - {("random", "FullFT")})
    d1 = table(st1, combos, [0.8, 1.0])
    d1.to_csv(A / "time_to_level_screening.csv", index=False)
    for (init, mode), h in d1.groupby(["init", "mode"]):
        for q in (0.8, 1.0):
            ok = h[h[f"s{q}"].notna() & h[f"p{q}"].notna()]
            out[f"screening|{init}|{mode}|q{q}"] = dict(
                speedup=round(float((ok[f"s{q}"] / ok[f"p{q}"]).median()), 3) if len(ok) else None,
                reached=round(float(h[f"p{q}"].notna().mean()), 3), n=int(len(h)))
    (A / "time_to_level.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k.startswith("confirmation") or k.endswith("q1.0"):
            print(k, v)

if __name__ == "__main__":
    main()
