"""Parse the finished Study 2 runs into two tables (logs/study2/analysis/).

series.csv   one row per run and iteration: return, learning rate, KL, action noise (from the entropy of
             the four-dimensional Gaussian policy), and the logged flight metrics.
budgets.csv  one row per run and budget K in {100, 300, 600}: every quantity averaged over (0.9K, K].
Usage: python study2_parse.py            (incremental: runs already parsed are kept)
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import csv
import glob
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

L = Path(_EXP_ROOT + "/logs/study2")
RUNS = _AG_ROOT + "/aerial_gym/rl_training/rl_games/runs/"
OUT = L / "analysis"
PAT = re.compile(r"epoch: (\d+)/(\d+)\s+frames: \d+\s+mean_reward: ([-\d.]+)")
HALF_LOG_2PIE = 0.5 * math.log(2 * math.pi * math.e)
SCALARS = {"info/last_lr": "lr", "info/kl": "kl", "losses/entropy": "entropy",
           "eval/success_rate/step": "success", "eval/crash_rate/step": "crash",
           "eval/timeout_rate/step": "timeout", "eval/episode_length/step": "ep_len"}

def parse_run(rid, meta):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    txt = (L / f"{rid}.log").read_text(errors="ignore")
    ret = {int(e): float(v) for e, E, v in PAT.findall(txt)}
    ser = {}
    for f in sorted(glob.glob(RUNS + rid + "/summaries/events.out.tfevents*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            if tag in SCALARS:
                key = SCALARS[tag]
            elif tag.startswith("eval/flight/") and tag.endswith("/step"):
                key = tag[len("eval/flight/"):-len("/step")]
            else:
                continue
            ev = ea.Scalars(tag)
            for i, e in enumerate(ev):
                it = i + 1 if tag.startswith(("info/", "losses/")) else e.step // 131072 + 1
                ser.setdefault(it, {})[key] = e.value
    rows = []
    for it in range(1, meta["iters"] + 1):
        v = ser.get(it, {})
        ent = v.pop("entropy", None)
        rows.append(dict(meta, iter=it, ret=ret.get(it, np.nan),
                         noise=math.exp(ent / 4 - HALF_LOG_2PIE) if ent is not None else np.nan, **v))
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(L / "ledger.csv")
    led = led[led.exit == 0].drop_duplicates("run_id", keep="last")
    old = pd.read_csv(OUT / "series.csv") if (OUT / "series.csv").exists() else pd.DataFrame()
    have = set(old.run_id) if len(old) else set()
    new = []
    for _, r in led.iterrows():
        if r.run_id in have:
            continue
        meta = dict(run_id=r.run_id, stage=int(r.stage), task=r.task, init=r.init, mode=r["mode"],
                    seed=int(r.seed), iters=int(r.iters))
        try:
            new += parse_run(r.run_id, meta)
        except Exception as e:  # keep going; report
            print("failed", r.run_id, e)
    s = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if new else old
    s.to_csv(OUT / "series.csv", index=False)
    rows = []
    num = [c for c in s.columns if c not in ("run_id", "stage", "task", "init", "mode", "seed", "iters", "iter")]
    for (rid,), g in s.groupby(["run_id"]):
        g = g.set_index("iter")
        m = g.iloc[0]
        for K in (100, 300, 600):
            if K > int(m.iters):
                continue
            w = g[(g.index > 0.9 * K) & (g.index <= K)]
            rows.append(dict(run_id=rid, stage=m.stage, task=m.task, init=m.init, mode=m["mode"], seed=m.seed, K=K,
                             **{c: float(np.nanmean(w[c])) if w[c].notna().any() else np.nan for c in num}))
    pd.DataFrame(rows).to_csv(OUT / "budgets.csv", index=False)
    print(f"{s.run_id.nunique()} runs parsed ({len(new) // 600 if new else 0} new)")

if __name__ == "__main__":
    main()
