"""V11 (pre-registration v2, section 25, stage 34) and V12 (stage 35).

V11: decision D2 of the confirmation (AP with Aux-SSL against its random counterpart) with the original pair (random
     backbone with the MAE auxiliary loss, stage 2) and with the matched pair (random backbone with the AP auxiliary loss,
     stage 34, mode AuxSSLap), nine settings, ten seeds, 600 iterations (training return, Eq. gain, seed t interval).
V12: MAE and AP pretrained for 1 and 5 epochs (stage 35, init MAEe1, MAEe5, PIDMe1, PIDMe5) against the 25-epoch
     backbones (stage 1), FullFT, R2 and R3 x T1-T3, seeds 7, 42, 99: gain over training from scratch at 100 and 600
     iterations (seed t interval); speed-up to the final level of training from scratch in T1 and T2 (section 24 method);
     iterations saved at the median, and the number of reuses after which one backbone pays for itself, with the
     pretraining time proportional to the number of epochs (25 epochs = PRETRAIN_S seconds) and RUN_S seconds per
     600-iteration run, both from the standalone measurement (cost_standalone.json); study2_cost_final.py gives the
     reported break-even points with the logged epoch times.
Output: logs/study2/analysis/v11_v12.json and a printed summary.
Usage: python study2_v11_v12.py
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
import study2_review_stats as R  # noqa: E402
import study2_time_to_level as TL  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")
S10 = [7, 13, 21, 42, 99, 123, 256, 512, 777, 2024]
S3 = [7, 42, 99]
T12 = ["state_setpoint_x500_clean", "recovery_x500", "ss_lmf2_clean", "recovery_lmf2"]
_C = json.loads((A / "cost_standalone.json").read_text())  # standalone measurement (RL times from the run logs)
PRETRAIN_S = {"MAE": _C["t_pre"]["MAE"], "PIDM": _C["t_pre"]["PIDM"]}  # 25 epochs, seed 7
RUN_S = _C["t_rl"]["random|FullFT"]  # one 600-iteration FullFT run

def r4(r):
    return None if r is None else dict(mean=round(r["mean"], 4), t95=[round(x, 4) for x in r["t95"]], p=round(r["p"], 4))

def main():
    b = R.load_budgets()
    out = {}
    tasks9 = sorted(b[b.stage == 2].task.unique())
    out["V11|D2_original"] = r4(R.contrast(b, 2, 600, S10, tasks9, ("PIDM", "AuxSSL"), ("random", "AuxSSL")))
    v34 = b[b.stage == 34].assign(stage=2)
    comb = pd.concat([b[b.stage == 2], v34])
    out["V11|D2_matched"] = r4(R.contrast(comb, 2, 600, S10, tasks9, ("PIDM", "AuxSSL"), ("random", "AuxSSLap")))
    out["V11|randomAP_aux_vs_scratch"] = r4(R.contrast(comb, 2, 600, S10, tasks9, ("random", "AuxSSLap"), ("random", "FullFT")))
    tasks6 = sorted(b[b.stage == 35].task.unique())
    base = b[(b.stage == 1) & b.seed.isin(S3) & (b["mode"] == "FullFT") & b.init.isin(["random", "MAE", "PIDM"])]
    comb35 = pd.concat([base.assign(stage=900), b[b.stage == 35].assign(stage=900)])
    for init in ("MAEe1", "MAEe5", "MAE", "PIDMe1", "PIDMe5", "PIDM"):
        for K in (100, 600):
            out[f"V12|{init}|G|K{K}"] = r4(R.contrast(comb35, 900, K, S3, tasks6, (init, "FullFT"), ("random", "FullFT")))
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret"])
    d = s[((s.stage == 1) & s.seed.isin(S3) & s.init.isin(["random", "MAE", "PIDM"]) | (s.stage == 35))
          & s.task.isin(T12) & (s["mode"] == "FullFT")]
    combos = [(i, "FullFT") for i in ("MAEe1", "MAEe5", "MAE", "PIDMe1", "PIDMe5", "PIDM")]
    tab = TL.table(d, combos, [0.5, 1.0])
    for init, ep in (("MAEe1", 1), ("MAEe5", 5), ("MAE", 25), ("PIDMe1", 1), ("PIDMe5", 5), ("PIDM", 25)):
        for q in (0.5, 1.0):
            r = TL.boot(tab[tab.init == init], q)
            out[f"V12|{init}|speedup|q{q}"] = r
            if r and r.get("saved") and r["saved"][0] and r["saved"][0] > 0:
                pre = PRETRAIN_S["MAE" if init.startswith("MAE") else "PIDM"] * ep / 25  # proportional (study2_cost_final.py uses the logged epoch times)
                out[f"V12|{init}|breakeven_reuses|q{q}"] = round(pre / (r["saved"][0] * RUN_S / 600), 2)
    (A / "v11_v12.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
