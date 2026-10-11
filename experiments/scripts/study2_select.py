"""Stage-2 selection of Study 2 v2 (pre-registration v2, section 4, candidate sets extended in section 11).

Normalized score S_e(i, m) = terminal return (iterations 541-600) / seed-mean terminal return of random+FullFT in
setting e, with seed means per setting.
  objective: MAE always; among CPC, SPR, PIDM (AP), IDM, FDM, BC, MAEs the one with the largest mean over
             (common mode x 6 settings) of S_e(objective, m) - S_e(random, m). Common modes are FullFT, LateFT,
             EncFrozen, BackboneLoRA, DiscLR, L2SP, LPFT, AuxSSL; each objective is averaged over the common modes
             in which it exists (BC has no AuxSSL).
  mode:      FullFT always; among LateFT, EncFrozen, BackboneLoRA, DiscLR, L2SP, LPFT, AuxSSL the one with the
             largest mean over (pretrained objective x 6 settings) of S_e(objective, mode); AuxSSL over the
             backbones it applies to (all but BC). BCreg (BC only) is not a candidate.
  budget:    900 iterations if the return of random+FullFT (mean over settings and seeds) rises by more than 10%
             from iterations 481-540 to 541-600, otherwise 600
For the record only (not used for the decision) it also prints the rule of section 4 on the original grid
(4 objectives x 4 modes) and the mode scores over the seven backbones that have every mode.
Refuses to decide while any of the 1,296 screening runs is missing. Writes logs/study2/stage2_selection.json.
Usage: python study2_select.py [--write]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

A = Path(_EXP_ROOT + "/logs/study2/analysis")
PRE = ["MAE", "CPC", "SPR", "PIDM", "IDM", "FDM", "BC", "MAEs"]       # pretrained backbones
OBJ_CAND = [o for o in PRE if o != "MAE"]
COMMON = ["FullFT", "LateFT", "EncFrozen", "BackboneLoRA", "DiscLR", "L2SP", "LPFT", "AuxSSL"]
MODE_CAND = [m for m in COMMON if m != "FullFT"]
N_SCREEN = 1296

def scores(S, objs, modes, cand_obj, cand_mode):
    """Objective and mode scores of section 4 restricted to the given grid; cells that do not exist are skipped."""
    P = S.pivot_table(index=["task", "mode"], columns="init", values="S")
    P = P[P.index.get_level_values("mode").isin(modes)]
    obj = {}
    for o in cand_obj:
        if o in P and "random" in P:
            d = (P[o] - P["random"]).dropna()
            if len(d):
                obj[o] = float(d.mean())
    mode = {}
    for m in cand_mode:
        v = S[(S["mode"] == m) & (S.init.isin(objs))].S
        if len(v):
            mode[m] = float(v.mean())
    return obj, mode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.stage == 1) & (b.K == 600)]
    n = b.run_id.nunique()
    print(f"{n} of {N_SCREEN} screening runs parsed")
    cell = b.groupby(["task", "init", "mode"]).ret.mean()
    ref = cell.xs(("random", "FullFT"), level=("init", "mode"))
    S = (cell / cell.index.get_level_values("task").map(ref).values).rename("S").reset_index()
    obj_score, mode_score = scores(S, PRE, COMMON, OBJ_CAND, MODE_CAND)
    rec_obj, rec_mode = scores(S, ["MAE", "CPC", "SPR", "PIDM"], COMMON[:4], ["CPC", "SPR", "PIDM"], COMMON[1:4])
    _, same7 = scores(S, [o for o in PRE if o != "BC"], COMMON, [], MODE_CAND)
    # Record only: the same rule on T1 and T2 settings only, because the scratch
    # return of T3 is 50-100 times smaller and its normalized scores carry most of the variation
    ST = S[~S.task.str.startswith("agg")]
    noT3_obj, noT3_mode = scores(ST, PRE, COMMON, OBJ_CAND, MODE_CAND)
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "init", "mode", "iter", "ret"])
    s = s[(s.stage == 1) & (s.init == "random") & (s["mode"] == "FullFT")]
    w1 = s[(s.iter > 480) & (s.iter <= 540)].groupby("run_id").ret.mean().mean()
    w2 = s[(s.iter > 540) & (s.iter <= 600)].groupby("run_id").ret.mean().mean()
    rise = w2 / w1 - 1
    out = {"objective": max(obj_score, key=obj_score.get) if obj_score else None,
           "mode": max(mode_score, key=mode_score.get) if mode_score else None,
           "iters": 900 if rise > 0.10 else 600,
           "objective_scores": obj_score, "mode_scores": mode_score, "scratch_rise_481_540_to_541_600": rise,
           "record_only": {"section4_original_grid": {"objective_scores": rec_obj, "mode_scores": rec_mode},
                           "mode_scores_over_7_backbones_without_BC": same7,
                           "T1_T2_only": {"objective_scores": noT3_obj, "mode_scores": noT3_mode,
                                          "objective": max(noT3_obj, key=noT3_obj.get) if noT3_obj else None,
                                          "mode": max(noT3_mode, key=noT3_mode.get) if noT3_mode else None}},
           "rule": "pre-registration v2 section 4 with the candidate sets of section 11",
           "written": datetime.now().isoformat(timespec="seconds")}
    print(json.dumps(out, indent=1))
    if n < N_SCREEN:
        print("screening stage incomplete: no decision written")
        return
    if a.write:
        (A.parent / "stage2_selection.json").write_text(json.dumps(out, indent=1))
        print("written", A.parent / "stage2_selection.json")

if __name__ == "__main__":
    main()
