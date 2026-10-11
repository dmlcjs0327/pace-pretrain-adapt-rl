"""Mechanism analysis: per-iteration optimizer and loss logs of finished runs.

Reads the tensorboard summaries of every finished run of the given stages and writes one row per run and PPO epoch
(iteration) with: lr (info/last_lr), kl (info/kl), grad_norm (info/grad_norm, before clipping), clipped
(info/grad_clipped), reg (losses/reg: the PACE regularizer or auxiliary loss including lambda), a_loss, c_loss,
entropy, bounds (losses/bounds_loss). Output: logs/study2/analysis/mech_logs.csv (incremental, keyed by run_id).
Usage: python study2_mech_logs.py --stages 1 2 [--limit N]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import argparse
import glob
import os

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = _AG_ROOT + "/aerial_gym/rl_training/rl_games/runs/"
A = _EXP_ROOT + "/logs/study2/analysis/"
L = _EXP_ROOT + "/logs/study2/ledger.csv"
TAGS = {"info/last_lr": "lr", "info/kl": "kl", "info/grad_norm": "grad_norm", "info/grad_clipped": "clipped",
        "losses/reg": "reg", "losses/a_loss": "a_loss", "losses/c_loss": "c_loss", "losses/entropy": "entropy",
        "losses/bounds_loss": "bounds"}

def read(rid):
    rows = {}
    for f in sorted(glob.glob(RUNS + rid + "/summaries/events.out.tfevents*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        have = ea.Tags()["scalars"]
        for tag, col in TAGS.items():
            if tag not in have:
                continue
            for e in ea.Scalars(tag):
                rows.setdefault(int(e.step), {})[col] = e.value
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "step"
    return df.reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = A + "mech_logs.csv"
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    have = set(old.run_id) if len(old) else set()
    led = pd.read_csv(L)
    led = led[(led.exit == 0) & led.stage.isin(a.stages)].drop_duplicates("run_id", keep="last")
    todo = [r for _, r in led.iterrows() if r.run_id not in have]
    if a.limit:
        todo = todo[:a.limit]
    parts = []
    for i, r in enumerate(todo):
        df = read(r.run_id)
        if df.empty:
            continue
        df["it"] = df.groupby(lambda _: 0).cumcount() + 1 if "step" not in df else range(1, len(df) + 1)
        for c in ("run_id", "stage", "task", "init", "mode", "seed"):
            df[c] = r[c]
        parts.append(df)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(todo)}", flush=True)
            pd.concat([old] + parts, ignore_index=True).to_csv(out, index=False)
    if parts:
        pd.concat([old] + parts, ignore_index=True).to_csv(out, index=False)
    print("done", len(todo))

if __name__ == "__main__":
    main()
