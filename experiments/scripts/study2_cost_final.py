"""Costs in units of one standalone 600-iteration FullFT run, and the break-even points (pre-registration v2,
section 10 correction, section 13 S5, section 24, section 25 V12).

Inputs: cost_standalone.json (measure_cost.py, RL times fixed by cost_fix_rl_times.py), time_to_level.json (iterations
saved by FullFT in the confirmation, T1 and T2), v11_v12.json (V12 speed-ups, three seeds), and the per-epoch elapsed
times in the standalone pretraining logs (for the 1- and 5-epoch backbones of V12: start-up + elapsed time at that epoch).
Units (section 10 correction): c = (collection + pretraining) / FullFT; pretraining only = pretraining / FullFT;
r = mode / FullFT; Aux-SSL adds the head refit to c.
Break-even: reuses after which one backbone pays for itself = pretraining time / time saved per run, where time saved
per run = iterations saved x (FullFT time / 600); with and without the data collection.
Best of n: n restarts of training from scratch cost n FullFT runs; one pretrained run costs 1 + c.
Output: logs/study2/analysis/cost_final.json and a printed summary.
Usage: python study2_cost_final.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import re
from pathlib import Path

EXP = Path(_EXP_ROOT + "")
A = EXP / "logs" / "study2" / "analysis"
COST = EXP / "logs" / "study2" / "cost"
NAME = {"MAE": "MAE", "MAEs": "MAE-T", "CPC": "CPC", "SPR": "SPR", "FDM": "FDM", "IDM": "PIDM", "PIDM": "AP", "BC": "BC"}

def elapsed_at(obj, epoch):
    for ln in (COST / f"pretrain_{obj}.log").open(errors="ignore"):
        m = re.search(rf"epoch {epoch}/25 .*elapsed=([\d.]+)s", ln)
        if m:
            return float(m.group(1))
    return None

def main():
    d = json.loads((A / "cost_standalone.json").read_text())
    assert "t_rl_poll" in d, "run cost_fix_rl_times.py first"
    ff = d["t_rl"]["random|FullFT"]
    tc = d["t_collect"]
    out = {"FullFT_s": ff, "collect_s": tc, "collect_units": round(tc / ff, 3), "objectives": {}, "modes": {}}
    for o, t in d["t_pre"].items():
        out["objectives"][NAME[o]] = dict(pretrain_s=t, pretrain_units=round(t / ff, 3), c=round((tc + t) / ff, 3),
                                          refit_units=round(d["t_refit"][o] / ff, 3) if o in d["t_refit"] else None)
    for k, t in d["t_rl"].items():
        out["modes"][k] = round(t / ff, 3)
    t = json.loads((A / "time_to_level.json").read_text())
    be = {}
    for o in ("MAE", "PIDM"):
        tp = d["t_pre"][o]
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r = t.get(f"confirmation|{o}|q{q}")
            if not r or not r["saved"][0] or r["saved"][0] <= 0:
                continue
            sec = r["saved"][0] * ff / 600
            lo, hi = r["saved"][2] * ff / 600, r["saved"][1] * ff / 600  # more iterations saved -> fewer reuses
            be[f"{NAME[o]}|q{q}"] = dict(iterations_saved=r["saved"][0], seconds_saved=round(sec, 1),
                                         reuses=round(tp / sec, 2), reuses_ci=[round(tp / lo, 2), round(tp / hi, 2)],
                                         reuses_with_collection=round((tp + tc) / sec, 2),
                                         reuses_with_collection_ci=[round((tp + tc) / lo, 2), round((tp + tc) / hi, 2)])
    out["break_even"] = be
    v12 = json.loads((A / "v11_v12.json").read_text())
    ve = {}
    for o in ("MAE", "PIDM"):
        startup = d["t_pre"][o] - elapsed_at(o, 25)
        for ep, init in ((1, f"{o}e1"), (5, f"{o}e5"), (25, o)):
            pre = startup + elapsed_at(o, ep)
            for q in (0.5, 1.0):
                r = v12.get(f"V12|{init}|speedup|q{q}")
                if not r or not r["saved"][0] or r["saved"][0] <= 0:
                    continue
                sec = r["saved"][0] * ff / 600
                ve[f"{NAME[o]}|e{ep}|q{q}"] = dict(pretrain_s=round(pre, 1), iterations_saved=r["saved"][0],
                                                   reuses=round(pre / sec, 2), reuses_with_collection=round((pre + tc) / sec, 2))
    out["V12_break_even"] = ve
    out["best_of_n"] = {NAME[o]: dict(one_pretrained_run=round(1 + out["objectives"][NAME[o]]["c"], 3),
                                      share_of_ten_restarts=round((1 + out["objectives"][NAME[o]]["c"]) / 10, 3))
                        for o in ("MAE", "PIDM")}
    (A / "cost_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
