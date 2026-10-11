"""V4 (pre-registration v2, section 25): iterations to the final level of the shuffled backbones on the ten seeds.

Same method as study2_time_to_level.py (section 24, exploratory measure): T1 and T2 on R1-R3, FullFT, the ten seeds of
the confirmation. The original backbones (MAE, AP = run name PIDM) and training from scratch come from stage 2 (and
the stage-1 cells that stage 2 reused). The shuffled backbones (MAEshuf, PIDMshuf) come from stage 5 (seeds 7, 42, 99)
and stage 27 (the other seeds). Reported per backbone: the median speed-up over training from scratch at q = 1.0 and
the other levels, with a seed bootstrap; and the share of the extra speed that the shuffled backbone keeps,
(speed-up of shuffled - 1) / (speed-up of original - 1), with the same bootstrap.
Output: logs/study2/analysis/v4_shuffle_speed.json and a printed summary.
Usage: python study2_v4_shuffle_speed.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study2_time_to_level as T  # noqa: E402

A = T.A
rng = np.random.default_rng(25)

def main():
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret"])
    s2 = s[(s.stage == 2) & s.task.isin(T.T12) & (s["mode"] == "FullFT") & s.init.isin(["random", "MAE", "PIDM"])]
    have = set(zip(s2.task, s2.init, s2["mode"], s2.seed))
    s1 = s[(s.stage == 1) & s.task.isin(T.T12) & (s["mode"] == "FullFT") & s.init.isin(["random", "MAE", "PIDM"])]
    s1 = s1[[k not in have for k in zip(s1.task, s1.init, s1["mode"], s1.seed)] & s1.seed.isin(s2.seed.unique())]
    sh = s[s.stage.isin([5, 27]) & s.task.isin(T.T12) & (s["mode"] == "FullFT") & s.init.isin(["MAEshuf", "PIDMshuf"])]
    sh = sh[sh.seed.isin(s2.seed.unique())]
    d = T.table(pd.concat([s2, s1, sh]), [(i, "FullFT") for i in ("MAE", "PIDM", "MAEshuf", "PIDMshuf")], T.QS)
    out = {"n_cells": {i: int((d.init == i).sum()) for i in ("MAE", "PIDM", "MAEshuf", "PIDMshuf")}}
    for i in ("MAE", "PIDM", "MAEshuf", "PIDMshuf"):
        for q in (0.8, 1.0):
            out[f"{i}|q{q}"] = T.boot(d[d.init == i], q)
    seeds = sorted(d.seed.unique())
    for base in ("MAE", "PIDM"):
        def share(ss, q=1.0):
            def med(init):
                x = pd.concat([d[(d.init == init) & (d.seed == s_)] for s_ in ss])
                x = x[x[f"s{q}"].notna() & x[f"p{q}"].notna()]
                return np.median(x[f"s{q}"] / x[f"p{q}"])
            o = med(base)
            return (med(base + "shuf") - 1) / (o - 1) if abs(o - 1) > 1e-9 else np.nan
        pt = share(seeds)
        bs = [share(list(rng.choice(seeds, len(seeds)))) for _ in range(2000)]
        out[f"{base}shuf|share_of_extra_speed|q1.0"] = [round(float(pt), 3), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 3).tolist()]
    (A / "v4_shuffle_speed.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
