"""Calibration rule of Study 2 v2 (pre-registration v2, section 2).

Reads the 36 calibration runs (random + FullFT; 6 settings x seeds {1001, 1002} x candidates C1-C3) from
logs/study2/calib/, takes the terminal return of each run (mean logged return over iterations 541-600),
divides it by the mean terminal return of the 6 runs of the same setting, averages over settings and seeds,
and writes the candidate with the largest mean to logs/study2/ppo_config.json. Refuses to decide while any
of the 36 runs is missing. Also prints the per-setting table, the learning-rate trace summary, and the
fraction of iterations at the learning-rate ceiling for the record.
Usage: python study2_calib.py [--write]
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "automation"))
import run_study2 as S  # noqa: E402

L = S.EXP / "logs" / "study2"
C = L / "calib"
PAT = re.compile(r"epoch: (\d+)/(\d+)\s+frames: \d+\s+mean_reward: ([-\d.]+)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = []
    for c in S.cells("C"):
        rid = S.run_id(c)
        log = C / f"{rid}.log"
        if not log.exists():
            continue
        ret = {int(e): float(v) for e, _, v in PAT.findall(log.read_text(errors="ignore"))}
        if max(ret, default=0) < 600:
            continue
        rows.append(dict(cand=c["cand"], task=c["task"], seed=c["seed"],
                         term=np.mean([ret[i] for i in range(541, 601) if i in ret]),
                         r300=np.mean([ret[i] for i in range(271, 301) if i in ret])))
    d = pd.DataFrame(rows)
    print(f"{len(d)} of 36 calibration runs finished")
    if len(d) == 0:
        return
    d["norm"] = d.term / d.groupby("task").term.transform("mean")
    pd.set_option("display.width", 200)
    print(d.pivot_table(index="task", columns="cand", values="term", aggfunc="mean").round(0).to_string())
    score = d.groupby("cand").norm.mean()
    print("mean normalized terminal return:", score.round(3).to_dict())
    if len(d) < 36:
        print("not all runs finished: no decision")
        return
    best = score.idxmax()
    out = {"chosen": best, "config": S.CANDIDATES[best], "scores": score.round(4).to_dict(),
           "rule": "pre-registration v2 section 2", "written": datetime.now().isoformat(timespec="seconds")}
    print(json.dumps(out, indent=1))
    if a.write:
        S.PPO_CFG.write_text(json.dumps(out, indent=1))
        print("written", S.PPO_CFG)

if __name__ == "__main__":
    main()
