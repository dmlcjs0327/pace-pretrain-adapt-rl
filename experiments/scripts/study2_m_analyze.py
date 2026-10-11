"""Analysis of the follow-up mechanism controls M1-M4 (pre-registration v2, section 22). Parts whose inputs are
missing are skipped, so the script can be rerun as the side jobs and stage 14 finish.

M1 head width (supervised, frozen backbones): closed-loop return of Frozen students with heads of 64 (S19), 256, and
   1,024 units, divided by that of the FullFT student with a head of 64 (S19), per setting and seed, mean over seeds and
   then over the four T1/T2 settings. Also MAE FullFT 1,024 over FullFT 64 and MAE Frozen 1,024 over FullFT 1,024.
   Test R^2 of action and value per head width.
M2 non-IID minibatches and moving targets (supervised): closed-loop return of each condition divided by that of the
   IID student (S19) of the same backbone, mode, setting, and seed. Key contrast: ratio(Frozen) - ratio(FullFT)
   (negative = the frozen students lose more).
M3 critic first: explained variance of the critic at iterations 25, 50, 100 (study2_m3_critic_early.py), and the
   fixed-state evaluation return at the same iterations in units of the final (iterations >= 550) fixed-state return
   of training from scratch of the setting. Differences pretrained - random.
M4 width dose-response: per width (0.5, 1, 2, 4) and seed, the effective dimension and linear value R^2 of the frozen
   MAE features (q_width_features.csv) and the RL contrasts at 600 iterations (study2_analyze.delta, one seed at a
   time): MAE Frozen vs scratch, MAE Frozen vs random Frozen, MAE FullFT vs scratch. Spearman correlations over the
   12 width-seed points. Also reported: the return of MAE Frozen over that of MAE FullFT at the same width and seed
   (T1/T2 settings, 600 iterations), which does not depend on the scratch run.
Every interval is a joint seed bootstrap with 4,000 resamples (the same resampled seeds index all settings).
Output: logs/study2/analysis/m_mech.json and a printed summary.
Usage: python study2_m_analyze.py
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
from scipy.stats import spearmanr

import study2_analyze as AN

A = Path(_EXP_ROOT + "/logs/study2/analysis")
AN.A = A
T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
SEEDS = [7, 42, 99]
B = 4000
rng = np.random.default_rng(22)

def boot(fn):
    """Point estimate and 95% interval of fn(seeds) under a joint resampling of the seeds."""
    pt = fn(SEEDS)
    bs = [fn(list(rng.choice(SEEDS, len(SEEDS)))) for _ in range(B)]
    return [round(float(pt), 4), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 4).tolist()]

def students():
    """Closed-loop return of every student, keyed by (init, mode, head, regime, task, seed)."""
    frames = [pd.read_csv(A / f) for f in ("eval_s19_students.csv", "eval_m12_students.csv") if (A / f).exists()]
    e = pd.concat(frames).drop_duplicates("run_id", keep="last")
    p = e.run_id.str.split("__")
    e = e.assign(task=p.str[1], init=p.str[2], mode=p.str[3], seed=p.str[4].str.replace("seed", "").astype(int),
                 head=p.map(lambda x: int(x[5][1:]) if len(x) > 5 else 64),
                 regime=p.map(lambda x: x[6] if len(x) > 6 else "iid"))
    return e.set_index(["init", "mode", "head", "regime", "task", "seed"]).ret

def r2_table():
    frames = [pd.read_csv(A / f) for f in ("s19_students.csv", "m12_students.csv") if (A / f).exists()]
    r = pd.concat(frames)
    r["head_hidden"] = r.get("head_hidden", 64)
    r["head_hidden"] = r.head_hidden.fillna(64).astype(int)
    r["regime"] = r.get("regime", "iid")
    r["regime"] = r.regime.fillna("iid")
    return r.groupby(["init", "mode", "head_hidden", "regime"])[["r2_action", "r2_value"]].mean().round(3)

def ratio(g, num, den, tasks=T12):
    def f(ss):
        return np.mean([np.mean([g[num + (t, s)] / g[den + (t, s)] for s in ss]) for t in tasks])
    try:
        f(SEEDS)
    except KeyError:
        return None
    return boot(f)

def m1(g):
    out = {}
    for init in ("random", "MAE", "PIDM"):
        for h in (64, 256, 1024):
            v = ratio(g, (init, "Frozen", h, "iid"), (init, "FullFT", 64, "iid"))
            if v:
                out[f"M1|{init}|Frozen h{h} / FullFT h64"] = v
    for name, num, den in (("MAE|FullFT h1024 / FullFT h64", ("MAE", "FullFT", 1024, "iid"), ("MAE", "FullFT", 64, "iid")),
                           ("MAE|Frozen h1024 / FullFT h1024", ("MAE", "Frozen", 1024, "iid"), ("MAE", "FullFT", 1024, "iid"))):
        v = ratio(g, num, den)
        if v:
            out["M1|" + name] = v
    return out

def m2(g):
    out = {}
    for init in ("random", "MAE"):
        for reg in ("noniid", "nonstat"):
            r = {}
            for mode in ("Frozen", "FullFT"):
                v = ratio(g, (init, mode, 64, reg), (init, mode, 64, "iid"))
                if v:
                    out[f"M2|{init}|{mode}|{reg} / iid"] = r[mode] = v

            def diff(ss, init=init, reg=reg):
                return np.mean([np.mean([g[(init, "Frozen", 64, reg, t, s)] / g[(init, "Frozen", 64, "iid", t, s)]
                                         - g[(init, "FullFT", 64, reg, t, s)] / g[(init, "FullFT", 64, "iid", t, s)]
                                         for s in ss]) for t in T12])
            if len(r) == 2:
                out[f"M2|{init}|{reg}|Frozen ratio - FullFT ratio"] = boot(diff)
    return out

def m3():
    p = A / "m3_critic_early.csv"
    if not p.exists():
        return {}
    c = pd.read_csv(p)
    out = {}
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    ev = ev[(ev.disturb == 0) & ev.seed.isin(SEEDS) & ev.task.isin(T12)]
    scratch = ev[(ev.stage == 1) & (ev.init == "random") & (ev["mode"] == "FullFT") & (ev.iter >= 550)].groupby(["task", "seed"]).ret.mean()
    fixed = {}
    for init, st in (("random", 1), ("MAE", 1), ("PIDM", 1), ("MAEshuf", 5)):
        e = ev[(ev.stage == st) & (ev.init == init) & (ev["mode"] == "FullFT")]
        for (t, s, it), r in e.groupby(["task", "seed", "iter"]).ret.mean().items():
            fixed[(init, it, t, s)] = r / scratch[(t, s)]
    cv = c.set_index(["init", "iter", "task", "seed"]).ev_critic
    tasks = sorted(c.task.unique())
    for it in (25, 50, 100):
        for init in ("random", "MAE", "PIDM", "MAEshuf"):
            try:
                out[f"M3|EV|{init}|it{it}"] = boot(lambda ss: np.mean([np.mean([cv[(init, it, t, s)] for s in ss]) for t in tasks]))
                out[f"M3|return|{init}|it{it}"] = boot(lambda ss: np.mean([np.mean([fixed[(init, it, t, s)] for s in ss]) for t in tasks]))
            except KeyError:
                continue
        for init in ("MAE", "PIDM", "MAEshuf"):
            try:
                out[f"M3|EV diff|{init}-random|it{it}"] = boot(
                    lambda ss: np.mean([np.mean([cv[(init, it, t, s)] - cv[("random", it, t, s)] for s in ss]) for t in tasks]))
                out[f"M3|return diff|{init}-random|it{it}"] = boot(
                    lambda ss: np.mean([np.mean([fixed[(init, it, t, s)] - fixed[("random", it, t, s)] for s in ss]) for t in tasks]))
            except KeyError:
                continue
    return out

def m4():
    q = pd.read_csv(A / "q_width_features.csv")
    q = q[q.init == "MAE"].set_index(["width", "seed"])
    b = pd.read_csv(A / "budgets.csv")
    from study2_size_d2 import size_frames
    fr = size_frames(b)
    rows = []
    for w in (0.5, 1.0, 2.0, 4.0):
        if w not in fr:
            continue
        for s in SEEDS:
            r = dict(width=w, seed=s, d_eff=float(q.loc[(w, s), "part_ratio"]), r2_value=float(q.loc[(w, s), "r2_value_lin"]))
            for name, Ac, Bc in (("frozen_vs_scratch", ("MAE", "EncFrozen"), ("random", "FullFT")),
                                 ("frozen_vs_random_frozen", ("MAE", "EncFrozen"), ("random", "EncFrozen")),
                                 ("fullft_vs_scratch", ("MAE", "FullFT"), ("random", "FullFT"))):
                v = AN.delta(fr[w], 1, Ac, Bc, 600, seeds=[s])
                r[name] = v[0] if v else np.nan
            # also reported: Frozen over FullFT of MAE at the same width (T1/T2, return at 600), free of the scratch run
            f6 = fr[w][(fr[w].K == 600) & (fr[w].seed == s) & (fr[w].init == "MAE")].set_index(["mode", "task"]).ret
            r["frozen_over_fullft"] = float(np.mean([f6[("EncFrozen", t)] / f6[("FullFT", t)] for t in T12]))
            rows.append(r)
    d = pd.DataFrame(rows).assign(config="C3")
    # section 23 (R1): width x4 trained again with the recalibrated configuration C1 (stage 21), reported alongside
    b21 = b[b.stage == 21].assign(stage=1)
    if b21[b21.K == 600].run_id.nunique() >= 108:
        rows = []
        for s in SEEDS:
            r = dict(width=4.0, seed=s, d_eff=float(q.loc[(4.0, s), "part_ratio"]), r2_value=float(q.loc[(4.0, s), "r2_value_lin"]))
            for name, Ac, Bc in (("frozen_vs_scratch", ("MAE", "EncFrozen"), ("random", "FullFT")),
                                 ("frozen_vs_random_frozen", ("MAE", "EncFrozen"), ("random", "EncFrozen")),
                                 ("fullft_vs_scratch", ("MAE", "FullFT"), ("random", "FullFT"))):
                v = AN.delta(b21, 1, Ac, Bc, 600, seeds=[s])
                r[name] = v[0] if v else np.nan
            f6 = b21[(b21.K == 600) & (b21.seed == s) & (b21.init == "MAE")].set_index(["mode", "task"]).ret
            r["frozen_over_fullft"] = float(np.mean([f6[("EncFrozen", t)] / f6[("FullFT", t)] for t in T12]))
            rows.append(r)
        d = pd.concat([d, pd.DataFrame(rows).assign(config="C1")], ignore_index=True)
    d.to_csv(A / "m4_width_points.csv", index=False)
    full = d[d.config == "C3"]
    alt = pd.concat([d[(d.config == "C3") & (d.width != 4.0)], d[(d.width == 4.0) & (d.config == ("C1" if (d.config == "C1").any() else "C3"))]])
    out = {"M4|points": len(full), "M4|widths": sorted(full.width.unique().tolist())}
    for tag, dd0 in (("", full), ("recal|", alt)):
        for x in ("d_eff", "r2_value"):
            for y in ("frozen_vs_scratch", "frozen_vs_random_frozen", "fullft_vs_scratch", "frozen_over_fullft"):
                dd = dd0[[x, y]].dropna()
                if len(dd) >= 6:
                    rho, p = spearmanr(dd[x], dd[y])
                    out[f"M4|{tag}spearman|{x}|{y}"] = [round(float(rho), 3), round(float(p), 4), len(dd)]
    out["M4|per_width_mean"] = full.groupby("width").mean(numeric_only=True).drop(columns="seed").round(3).to_dict("index")
    out["M4|per_width_mean_recal"] = alt.groupby("width").mean(numeric_only=True).drop(columns="seed").round(3).to_dict("index")
    return out

def main():
    out = {}
    if (A / "eval_m12_students.csv").exists():
        g = students()
        out.update(m1(g))
        out.update(m2(g))
        out["R2"] = {"|".join(map(str, k)): v for k, v in r2_table().to_dict("index").items()}
    out.update(m3())
    out.update(m4())
    (A / "m_mech.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        print(k, v)

if __name__ == "__main__":
    main()
