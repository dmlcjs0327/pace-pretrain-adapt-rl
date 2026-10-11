"""S17 network width (stage 14, section 19) and S18 second domain (stage 15, section 20) of pre-registration v2.

Gains G use the definition of the paper (study2_analyze.delta): mean over settings of the difference of training
returns averaged over (0.9K, K], in units of the training return of training from scratch (random + FullFT) of the
same setting, with a joint seed bootstrap (4,000 resamples).
S17: per width w in {0.5, 1, 2, 4} (w = 1 from stage 1, seeds 7, 42, 99, R2 and R3), the contrasts below at K = 100
     and 600, and the slope of each contrast against log2(number of parameters) with a joint seed bootstrap.
S18: per setting of the second domain (Anymal full state D1, without base linear velocity D2), the same contrasts,
     the position P of the shuffled backbones (Eq. position), and the floor rule of section 20.
Output: logs/study2/analysis/size_d2.json and a printed summary.
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import study2_analyze as AN

A = Path(_EXP_ROOT + "/logs/study2/analysis")
AN.A = A
PARAMS = {0.5: 37065, 1.0: 145801, 2.0: 578313, 4.0: 2303497}  # policy parameters (actor, critic, log-std)
TASKS = ["state_setpoint_x500_clean", "recovery_x500", "agg_x500", "ss_lmf2_clean", "recovery_lmf2", "agg_lmf2"]
CONTRASTS = [("MAE|FullFT vs scratch", ("MAE", "FullFT"), ("random", "FullFT")),
             ("MAE|Frozen vs scratch", ("MAE", "EncFrozen"), ("random", "FullFT")),
             ("MAE|Frozen vs random|Frozen", ("MAE", "EncFrozen"), ("random", "EncFrozen")),
             ("MAE|LoRA vs random|LoRA", ("MAE", "BackboneLoRA"), ("random", "BackboneLoRA")),
             ("MAE|LoRA vs scratch", ("MAE", "BackboneLoRA"), ("random", "FullFT"))]
COLLECTOR_RETURN = 58.77  # mean episode return of the second-domain collector at its last iteration (section 20)
rng = np.random.default_rng(20)

def width_of(rid):
    m = re.search(r"__(?:random|MAE)w([0-9.]+)__", rid)
    return float(m.group(1)) if m else 1.0

def size_frames(b):
    b14 = b[b.stage == 14].copy()
    b14["width"] = b14.run_id.map(width_of)
    b1 = b[(b.stage == 1) & b.seed.isin([7, 42, 99]) & b.task.isin(TASKS) & b.init.isin(["random", "MAE"])
           & b["mode"].isin(["FullFT", "EncFrozen", "BackboneLoRA"])].copy()
    b1["width"] = 1.0
    out = {}
    for w, g in pd.concat([b1, b14]).groupby("width"):
        out[w] = g.assign(stage=1)
    return out

def size_analysis(b):
    fr = size_frames(b)
    res = {}
    for name, Ac, Bc in CONTRASTS:
        for K in (100, 600):
            for w, f in sorted(fr.items()):
                v = AN.delta(f, 1, Ac, Bc, K)
                if v is not None:
                    res[f"S17|w{w:g}|{name}|K{K}"] = v
            # slope over widths with a joint seed bootstrap (the same seeds index every width)
            ws = [w for w in sorted(fr) if AN.delta(fr[w], 1, Ac, Bc, K) is not None]
            if len(ws) >= 3:
                x = np.log2([PARAMS[w] for w in ws])

                def g_at(w, seeds):
                    return AN.delta(fr[w][fr[w].seed.isin(seeds)], 1, Ac, Bc, K)
                pts = [AN.delta(fr[w], 1, Ac, Bc, K)[0] for w in ws]
                slope = float(np.polyfit(x, pts, 1)[0])
                bs = []
                for _ in range(400):
                    ss = list(rng.choice([7, 42, 99], 3))
                    vals = []
                    for w in ws:
                        d = fr[w]
                        d = pd.concat([d[d.seed == s].assign(seed=1000 * i + s) for i, s in enumerate(ss)])
                        v = AN.delta(d, 1, Ac, Bc, K)
                        vals.append(v[0] if v else np.nan)
                    if not np.isnan(vals).any():
                        bs.append(float(np.polyfit(x, vals, 1)[0]))
                res[f"S17|slope_per_doubling_params|{name}|K{K}"] = [round(slope, 4), *np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist(), len(ws)]
    return res

def d2_analysis(b):
    d = b[(b.stage == 15)].copy()
    res = {}
    if d.empty:
        return res
    for task, g in d.groupby("task"):
        f = g.assign(stage=1)
        sc = f[(f.init == "random") & (f["mode"] == "FullFT") & (f.K == 600)].ret
        res[f"S18|{task}|scratch_ret600_mean"] = round(float(sc.mean()), 3) if len(sc) else None
        res[f"S18|{task}|floor"] = bool(len(sc) and sc.mean() < 0.1 * COLLECTOR_RETURN)
        for obj in ("MAE", "PIDM"):
            for name, (m_a, m_b) in (("FullFT vs scratch", ("FullFT", "FullFT")), ("Frozen vs scratch", ("EncFrozen", "FullFT")),
                                     ("Frozen vs random|Frozen", ("EncFrozen", "EncFrozen")), ("LoRA vs random|LoRA", ("BackboneLoRA", "BackboneLoRA"))):
                Bi = "random"
                for K in (100, 600):
                    v = AN.delta(f, 1, (obj, m_a), (Bi, m_b), K)
                    if v is not None:
                        res[f"S18|{task}|{obj}|{name}|K{K}"] = v
            for K in (100, 600):
                o = AN.delta(f, 1, (obj, "FullFT"), ("random", "FullFT"), K)
                s = AN.delta(f, 1, (obj + "shuf", "FullFT"), ("random", "FullFT"), K)
                if o and s and abs(o[0]) > 1e-9:
                    res[f"S18|{task}|{obj}shuf|P|K{K}"] = round(s[0] / o[0], 3)
    return res

def main():
    b = pd.read_csv(A / "budgets.csv")
    out = {"size": size_analysis(b), "d2": d2_analysis(b)}
    (A / "size_d2.json").write_text(json.dumps(out, indent=1))
    for part in out.values():
        for k, v in part.items():
            print(k, v)

if __name__ == "__main__":
    main()
