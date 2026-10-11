"""Section 23 (R1, R2) of pre-registration v2: the width contrasts under each training configuration.

Frames (all MAE pretrained at the width, seeds 7, 42, 99, six settings R2/R3, 600 iterations):
  w4 C3  stage 14 (the study configuration)     w4 C1  stage 21 (the R1 winner)
  w8 C3  stage 20                               w8 Cx  stage 23 (the R2 winner, if not C3)
For each frame: the contrasts of study2_size_d2.py at K = 100 and 600 (gain over scratch of the same frame, joint seed
bootstrap, 4,000 resamples), LoRA vs FullFT and Frozen vs FullFT of MAE, and the return of MAE Frozen over MAE FullFT
in T1/T2 (mean over seeds and settings). Frames whose runs are incomplete are skipped.
Output: logs/study2/analysis/r_width.json and a printed summary.
Usage: python study2_r_width.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
from pathlib import Path

import numpy as np
import pandas as pd

import study2_analyze as AN

A = Path(_EXP_ROOT + "/logs/study2/analysis")
AN.A = A
T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
FRAMES = {"w4|C3": 14, "w4|C1": 21, "w8|C3": 20, "w8|R2win": 23}
CONTRASTS = [("MAE|FullFT vs scratch", ("MAE", "FullFT"), ("random", "FullFT")),
             ("MAE|Frozen vs scratch", ("MAE", "EncFrozen"), ("random", "FullFT")),
             ("MAE|LoRA vs scratch", ("MAE", "BackboneLoRA"), ("random", "FullFT")),
             ("MAE|Frozen vs random|Frozen", ("MAE", "EncFrozen"), ("random", "EncFrozen")),
             ("MAE|LoRA vs random|LoRA", ("MAE", "BackboneLoRA"), ("random", "BackboneLoRA")),
             ("MAE|LoRA vs MAE|FullFT", ("MAE", "BackboneLoRA"), ("MAE", "FullFT")),
             ("MAE|Frozen vs MAE|FullFT", ("MAE", "EncFrozen"), ("MAE", "FullFT"))]

def frame(b, stage, width):
    f = b[b.stage == stage].copy()
    if stage == 14:
        f = f[f.run_id.str.contains(f"w{width:g}__")]
    return f.assign(stage=1)

def main():
    b = pd.read_csv(A / "budgets.csv")
    out = {}
    for name, st in FRAMES.items():
        w = 4.0 if name.startswith("w4") else 8.0
        f = frame(b, st, w)
        n = f[f.K == 600].run_id.nunique()
        out[f"{name}|runs"] = int(n)
        if n < 108:
            continue
        for cname, Ac, Bc in CONTRASTS:
            for K in (100, 600):
                v = AN.delta(f, 1, Ac, Bc, K)
                if v is not None:
                    out[f"{name}|{cname}|K{K}"] = v
        g = f[(f.K == 600) & (f.init == "MAE")].set_index(["mode", "task", "seed"]).ret
        out[f"{name}|MAE Frozen / FullFT (T1,T2)"] = round(float(np.mean(
            [np.mean([g[("EncFrozen", t, s)] / g[("FullFT", t, s)] for s in (7, 42, 99)]) for t in T12])), 3)
        sc = f[(f.K == 600) & (f.init == "random") & (f["mode"] == "FullFT")].groupby("task").ret.mean().round(1)
        out[f"{name}|scratch return 600"] = sc.to_dict()
    (A / "r_width.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
