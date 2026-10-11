"""Replace the RL wall times of cost_standalone.json by the times read from the run logs.

Before this correction, dispatch() of run_study2.py polled each run every 60 s, so the wall time it records is rounded up to
the next poll (up to 60 s too long; 240.2 and 300.2 s in the standalone measurement). The log of a run starts with its
launch time (one-second resolution), and its last write marks the end of the run, so log end - launch is the run time.
The former values stay in t_rl_poll; t_rl holds the log-based times, which every consumer of the file reads.
Usage: python cost_fix_rl_times.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import csv
import datetime
import json
import os
import re
from pathlib import Path

EXP = Path(_EXP_ROOT + "")
COST = EXP / "logs" / "study2" / "cost"
RES = EXP / "logs" / "study2" / "analysis" / "cost_standalone.json"

def log_seconds(run_id):
    log = COST / f"{run_id}.log"
    first = log.open().readline()
    t0 = datetime.datetime.fromisoformat(re.search(r"\[(.*?)\]", first).group(1)).timestamp()
    return round(os.stat(log).st_mtime - t0, 1)

def main():
    d = json.loads(RES.read_text())
    if "t_rl_poll" in d:
        print("already fixed")
        return
    new = {}
    for r in csv.DictReader((COST / "ledger.csv").open()):
        new[f"{r['init']}|{r['mode']}"] = log_seconds(r["run_id"])
    missing = set(d["t_rl"]) - set(new)
    assert not missing, missing
    d["t_rl_poll"] = d["t_rl"]
    d["t_rl"] = {k: new[k] for k in d["t_rl_poll"]}
    d["t_rl_note"] = "t_rl from the run logs (launch time to last write); t_rl_poll from the 60-s poll of dispatch()"
    RES.write_text(json.dumps(d, indent=1))
    for k in d["t_rl"]:
        print(f"{k:16s} poll {d['t_rl_poll'][k]:7.1f}  log {d['t_rl'][k]:7.1f}")

if __name__ == "__main__":
    main()
