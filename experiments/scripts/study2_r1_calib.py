"""R1 and R2 (pre-registration v2, section 23): the calibration rule of section 2 at width x4 (stage 19) and, because
R1 chose a candidate other than C3, at width x8 (stage 22).

Scratch runs (random + FullFT) at the given width, six settings x seeds 1001, 1002 x candidates C1-C4. The terminal return of a
run is its mean training return over iterations 541-600 (budgets.csv, K = 600). Rule of section 2: divide each run by
the mean of the eight runs of its setting, average over settings and seeds, and choose the largest candidate.
Output: logs/study2/analysis/r1_calib.json (stage 19) or r2_calib.json (stage 22) and a printed table (score per candidate, and the mean terminal return per
setting and candidate).
Usage: python study2_r1_calib.py [--stage 22]   (after study2_parse.py)
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import argparse
import json
import re
from pathlib import Path

import pandas as pd

A = Path(_EXP_ROOT + "/logs/study2/analysis")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=19, choices=[19, 22])
    a = ap.parse_args()
    w, out_name = (4, "r1_calib.json") if a.stage == 19 else (8, "r2_calib.json")
    b = pd.read_csv(A / "budgets.csv")
    d = b[(b.stage == a.stage) & (b.K == 600)].copy()
    d["cand"] = d.run_id.map(lambda r: re.search(rf"__randomw{w}(C\d)__", r).group(1))
    n = d.groupby("cand").size().to_dict()
    d["rel"] = d.ret / d.groupby("task").ret.transform("mean")
    score = d.groupby("cand").rel.mean().round(4).to_dict()
    complete = len(d) == 48
    out = {"runs": int(len(d)), "runs_per_candidate": n, "complete": complete, "score": score,
           "chosen": max(score, key=score.get) if complete else None,
           "terminal_return": d.groupby(["task", "cand"]).ret.mean().unstack().round(1).to_dict("index")}
    (A / out_name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
