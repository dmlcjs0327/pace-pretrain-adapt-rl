"""V2 (pre-registration v2, section 25): restrained modes at their best of three learning-rate settings.

For each backbone (random, MAE, AP = run name PIDM) and each restrained mode (EncFrozen = Frozen, BackboneLoRA = LoRA;
LateFT = PartialFT has only C1 and C3), the best setting among C1 (stage 24, constant 3e-4), C3 (stage 1, adaptive with
ceiling 1e-3, the study setting) and C5 (stage 25, adaptive with ceiling 3e-3) is the one with the highest mean terminal
return over the six screening settings and seeds 7, 42, 99. Its gain is compared with training from scratch under C3
(stage 1), with seed-level t intervals. Choosing the setting on the same seeds favors the restrained modes.
Output: logs/study2/analysis/v2_bestlr_stats.json and a printed report.
Usage: python study2_v2_bestlr.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import sys

sys.path.insert(0, _EXP_ROOT + "/scripts")
import study2_review_stats as R  # noqa: E402

b = R.load_budgets()
b = b[b.K == 600]
T6 = sorted(b[b.stage == 24].task.unique())
STAGE = {"C1": 24, "C3": 1, "C5": 25}
out = {}
for init in ("random", "MAE", "PIDM"):
    for mode in ("EncFrozen", "BackboneLoRA", "LateFT"):
        means = {}
        for cfg, st in STAGE.items():
            d = b[(b.stage == st) & (b.init == init) & (b["mode"] == mode) & b.seed.isin(R.S3) & b.task.isin(T6)]
            if len(d) == len(T6) * 3:
                means[cfg] = float(d.ret.mean())
        if not means:
            continue
        best = max(means, key=means.get)
        res = {"mean_terminal_return": means, "best": best}
        for cfg in means:
            rows = b[(b.stage == STAGE[cfg]) & (b.init == init) & (b["mode"] == mode)].copy()
            ref = b[(b.stage == 1) & (b.init == "random") & (b["mode"] == "FullFT")].copy()
            comb = R.pd.concat([rows.assign(stage=900), ref.assign(stage=900)])
            r = R.contrast(R.with_k(comb) if hasattr(R, "with_k") else comb, 900, 600, R.S3, T6, (init, mode), ("random", "FullFT"))
            res[cfg] = r
        out[f"{init}|{mode}"] = res
        rb = res[best]
        print(f"{init:6s} {mode:12s} best {best} (means {', '.join(f'{k} {v:.1f}' for k, v in means.items())})  "
              f"vs scratch C3: {R.fmt(rb)}")
json.dump(out, open(R.A_DIR / "v2_bestlr_stats.json", "w"), indent=1)
print("saved", R.A_DIR / "v2_bestlr_stats.json")
