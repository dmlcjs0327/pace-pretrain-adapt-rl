"""V17 (pre-registration v2, section 29, stage 40) and V18 (section 30, stage 41): the second domain on AnymalTerrain.

Gains G use the definition of the paper (study2_analyze.delta): difference of training returns averaged over (0.9K, K],
in units of the training return of training from scratch (random + FullFT) of the same setting, joint seed bootstrap
(4,000 resamples), seeds 7, 13, 21, 42, 99.
Stage 40 (flat ground, with and without the base velocity): per setting, MAE and AP (run name PIDM) with FullFT,
Frozen, LoRA against training from scratch and against their random counterparts, at 100 and 600 iterations; the
position P of the shuffled backbones (Eq. position); the speed-up to the final level of training from scratch
(section 24 method, levels 0.5-1.0); the floor rule of section 29 (scratch return at 600 iterations <= 0).
Stage 41 (rough terrain without the height scan, with and without the terrain curriculum): MAE and AP with FullFT and
Frozen against training from scratch and against random Frozen, at 100, 300, and 600 iterations.
Output: logs/study2/analysis/v17_v18_d2.json and a printed summary.
Usage: python study2_v17_v18_d2.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study2_analyze as AN  # noqa: E402
import study2_time_to_level as TL  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")
AN.A = A

def per_task(b, stage, Ks, modes):
    res = {}
    d = b[(b.stage == stage)].copy()
    for task, g in d.groupby("task"):
        f = g.assign(stage=1)
        sc = f[(f.init == "random") & (f["mode"] == "FullFT") & (f.K == 600)].ret
        res[f"{task}|scratch_ret600_mean"] = round(float(sc.mean()), 3) if len(sc) else None
        res[f"{task}|floor"] = bool(len(sc) and sc.mean() <= 0)
        for obj in ("MAE", "PIDM"):
            for mode in modes:
                for K in Ks:
                    v = AN.delta(f, 1, (obj, mode), ("random", "FullFT"), K)
                    if v is not None:
                        res[f"{task}|{obj}|{mode} vs scratch|K{K}"] = v
                    if mode != "FullFT":
                        v = AN.delta(f, 1, (obj, mode), ("random", mode), K)
                        if v is not None:
                            res[f"{task}|{obj}|{mode} vs random|{mode}|K{K}"] = v
                if (obj + "shuf") in set(f.init):
                    o = AN.delta(f, 1, (obj, "FullFT"), ("random", "FullFT"), K)
                    s = AN.delta(f, 1, (obj + "shuf", "FullFT"), ("random", "FullFT"), K)
                    if o and s and abs(o[0]) > 1e-9:
                        res[f"{task}|{obj}shuf|P|K{K}"] = round(s[0] / o[0], 3)
    return res

def speed(stage):
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret"])
    d = s[(s.stage == stage) & (s["mode"] == "FullFT")]
    out = {}
    if d.empty:
        return out
    tab = TL.table(d, [("MAE", "FullFT"), ("PIDM", "FullFT")], [0.5, 0.8, 1.0])
    for o in ("MAE", "PIDM"):
        for q in (0.5, 0.8, 1.0):
            out[f"{o}|speedup|q{q}"] = TL.boot(tab[tab.init == o], q)
    return out

def main():
    b = pd.read_csv(A / "budgets.csv")
    out = {"stage40": per_task(b, 40, (100, 600), ("FullFT", "EncFrozen", "BackboneLoRA")),
           "stage40_speed": speed(40),
           "stage41": per_task(b, 41, (100, 300, 600), ("FullFT", "EncFrozen"))}
    (A / "v17_v18_d2.json").write_text(json.dumps(out, indent=1))
    for part in out.values():
        for k, v in part.items():
            print(k, v)

if __name__ == "__main__":
    main()
