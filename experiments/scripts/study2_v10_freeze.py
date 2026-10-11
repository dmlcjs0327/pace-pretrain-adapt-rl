"""V10 (pre-registration v2, section 25): confirmatory test of freezing on the seven seeds the screening did not use.

Four comparisons, as pre-registered: Frozen MAE and Frozen AP (run names MAE, PIDM with mode EncFrozen, stage 33)
against training from scratch, and against FullFT of the same backbone (stage 2), on the nine settings (R1-R3 x
T1-T3) and seeds 13, 21, 123, 256, 512, 777, 2024. Main measure: the gain of Eq. (gain) at 600 iterations (training
return), mean over the nine settings, one value per seed. Test: two-sided paired t over the seven seeds, Holm over
the four comparisons, alpha 0.05. Also reported (descriptive): the same per task family and per airframe.
Output: logs/study2/analysis/v10_freeze.json and a printed summary.
Usage: python study2_v10_freeze.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study2_review_stats as R  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")

def main():
    b = R.load_budgets()
    b = b[b.K == 600]
    tasks = sorted(b[b.stage == 33].task.unique())
    fz = b[(b.stage == 33) & (b["mode"] == "EncFrozen") & b.init.isin(["MAE", "PIDM"])]
    ref = b[(b.stage == 2) & (b["mode"] == "FullFT") & b.init.isin(["random", "MAE", "PIDM"])]
    comb = R.pd.concat([fz.assign(stage=900), ref.assign(stage=900)])
    comps = {"Frozen MAE vs scratch": (("MAE", "EncFrozen"), ("random", "FullFT")),
             "Frozen AP vs scratch": (("PIDM", "EncFrozen"), ("random", "FullFT")),
             "Frozen MAE vs MAE FullFT": (("MAE", "EncFrozen"), ("MAE", "FullFT")),
             "Frozen AP vs AP FullFT": (("PIDM", "EncFrozen"), ("PIDM", "FullFT"))}
    out = {"n_tasks": len(tasks), "seeds": R.NEW7}
    res = {k: R.contrast(comb, 900, 600, R.NEW7, tasks, a, c) for k, (a, c) in comps.items()}
    hp = R.holm({k: v["p"] for k, v in res.items()})
    for k, v in res.items():
        out[k] = dict(mean=round(v["mean"], 4), t95=[round(x, 4) for x in v["t95"]], p=round(v["p"], 6),
                      holm_p=round(hp[k], 6), seeds={s: round(x, 4) for s, x in v["seeds"].items()})
    groups = {"T1": [t for t in tasks if t.startswith(("state_setpoint", "ss_"))],
              "T2": [t for t in tasks if t.startswith("recovery")],
              "T3": [t for t in tasks if t.startswith("agg")],
              "R1": [t for t in tasks if "baseQuad" in t],
              "R2": [t for t in tasks if "x500" in t],
              "R3": [t for t in tasks if "lmf2" in t]}
    for g, ts in groups.items():
        for k in ("Frozen MAE vs scratch", "Frozen AP vs scratch"):
            a, c = comps[k]
            r = R.contrast(comb, 900, 600, R.NEW7, ts, a, c)
            out[f"{g}|{k}"] = None if r is None else [round(r["mean"], 3), round(r["t95"][0], 3), round(r["t95"][1], 3)]
    (A / "v10_freeze.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
