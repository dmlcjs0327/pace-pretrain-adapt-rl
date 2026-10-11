"""Mechanism figures added in v8 (pre-registration v2, section 17: S10-S13), drawn from the analysis tables.

Fig. 10  What RL keeps of the pretrained representation: linear decodability of the pretraining targets over RL
         (mech_compat.csv), Aux-SSL minus FullFT per objective (screening), and the auxiliary-weight dose (S11).
Fig. 11  Where the hard-task gain comes from: goal-directedness of the flight data (data_goal_cos.npz), the scripted
         collector as a reference on T3 (expert_eval.csv), and target-swapped backbones (S10).
Fig. 12  Take-off timing in T3: training curves of the confirmation runs and the budget extension (S13).
Parts whose runs are not complete are drawn as red dashed placeholders (expected values) and are replaced
automatically when the runs exist; the console output says which parts are measured.
Usage: python make_v8_mech_figs.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import make_v6_figs as M
from make_v6_figs import A, COL, NAME, OUT, RED, W, despine, plt

sys.path.insert(0, _EXP_ROOT + "/scripts")
import study2_analyze as AN  # noqa: E402

SCR = ("random", "FullFT")
EXPECT = {  # placeholders (expected values, pre-registration section 17); replaced when the runs exist
    "S11": {"AuxSSL01": -0.35, "AuxSSL001": -0.10},          # Aux-SSL minus FullFT of MAE, screening settings
    "S10": {"MAEtswap": 0.15, "PIDMtswap": 0.10, "MAEag": 0.14, "PIDMag": 0.08},  # T3 reach (V3 added in v13)
}
STATUS = []

def boot_mean(x, B=4000):
    x = np.asarray(x, float)
    bs = [np.mean(np.random.default_rng(i).choice(x, len(x))) for i in range(200)]
    return np.mean(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def fig_rewrite(b):
    d = pd.read_csv(A / "mech_compat.csv").drop_duplicates(["run_id", "iter"])
    d = d.assign(cond=d.init + "|" + d["mode"])
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.25]))
    for ax, obj, title in ((axs[0], "MAE", "(a) MAE target kept"), (axs[1], "PIDM", "(b) AP target kept")):
        col = f"r2_lin_{obj}"
        for cond, lab, c, ls in ((f"{obj}|FullFT", f"{NAME[obj]} + FullFT", COL[obj], "-"),
                                 (f"{obj}|AuxSSL", f"{NAME[obj]} + Aux-SSL", COL[obj], "--"),
                                 ("random|FullFT", "random + FullFT", COL["scratch"], "-")):
            g = d[d.cond == cond].groupby("iter")[col].mean()
            ax.plot(g.index, g.values, color=c, ls=ls, marker="o", ms=2.2, label=lab)
        ax.set_xlabel("RL iteration")
        ax.set_ylabel("linearly decodable share")
        ax.set_ylim(0, 1.0 if obj == "MAE" else 0.6)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.0)
        ax.legend(frameon=False, fontsize=5.6, loc="lower left" if obj == "MAE" else "center right")
        despine(ax)
    STATUS.append("fig10a,b measured (mech_compat.csv)")
    # (c) Aux-SSL minus FullFT per objective (screening) and the auxiliary-weight dose of MAE (S11)
    ax = axs[2]
    objs = ["MAE", "MAEs", "CPC", "SPR", "FDM", "IDM", "PIDM"]
    vals = [AN.delta(b, 1, (o, "AuxSSL"), (o, "FullFT"), 600) for o in objs]
    y = np.arange(len(objs))[::-1]
    for yi, o, v in zip(y, objs, vals):
        ax.errorbar(v[0], yi, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=3, color=COL[o], lw=0.8, capsize=1.2)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[o] for o in objs])
    ax.set_xlabel("Aux-SSL minus FullFT")
    ax.set_title("(c) Keeping the pretraining loss", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    # inset: dose of the auxiliary weight for MAE
    ins = ax.inset_axes([0.74, 0.56, 0.25, 0.36])
    s9 = b[(b.stage == 9) & (b.K == 600)]
    pts = {0.0: 0.0, 1.0: vals[0][0]}
    measured = {}
    for mode, lam in (("AuxSSL01", 0.1), ("AuxSSL001", 0.01)):
        g = s9[s9["mode"] == mode]
        if g.run_id.nunique() >= 18:
            bb = pd.concat([b[b.stage == 1], g.assign(stage=1, mode="AuxSSLx")])
            measured[lam] = AN.delta(bb, 1, ("MAE", "AuxSSLx"), ("MAE", "FullFT"), 600)[0]
    xs = [0.0, 0.01, 0.1, 1.0]
    ins.plot([0.0, 1.0], [pts[0.0], pts[1.0]], "o", color=COL["MAE"], ms=2.5)
    for lam, key in ((0.01, "AuxSSL001"), (0.1, "AuxSSL01")):
        if lam in measured:
            ins.plot([lam], [measured[lam]], "o", color=COL["MAE"], ms=2.5)
            STATUS.append(f"fig10c S11 lambda={lam} measured")
        else:
            ins.plot([lam], [EXPECT["S11"][key]], "o", mfc="white", mec=RED, ms=2.8)
            STATUS.append(f"fig10c S11 lambda={lam} PLACEHOLDER")
    yy = [pts[0.0], measured.get(0.01, EXPECT["S11"]["AuxSSL001"]), measured.get(0.1, EXPECT["S11"]["AuxSSL01"]), pts[1.0]]
    ins.plot(xs, yy, color=RED if len(measured) < 2 else COL["MAE"], ls="--" if len(measured) < 2 else "-", lw=0.7)
    ins.set_xscale("symlog", linthresh=0.01)
    ins.set_xticks([0, 0.01, 0.1, 1])
    ins.set_xticklabels(["0", ".01", ".1", "1"], fontsize=5.2)
    ins.tick_params(labelsize=5.2)
    ins.set_ylim(-0.92, 0.12)
    ins.set_title("MAE, weight $\\lambda$", fontsize=5.6)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.22)
    fig.savefig(OUT / "fig10_rewrite.pdf")
    fig.savefig(OUT / "fig10_rewrite.png", dpi=200)
    plt.close(fig)

def fig_source(ev):
    z = np.load(A / "data_goal_cos.npz")
    ex = pd.read_csv(A / "expert_eval.csv").drop_duplicates(["task", "cruise"], keep="last")
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 1.25]))
    ax = axs[0]
    bins = np.linspace(-1, 1, 41)
    for key, lab, c, ls in (("cos", "collected", COL["scratch"], "-"), ("cos_swap", "target-swapped", "#999999", "--"),
                            ("ag_cos", "task-agnostic", "#D55E00", ":")):
        # v13: the task-agnostic data of V3 (keys R2ag_cos, R3ag_cos of study2_data_goal.py)
        k2, k3 = (f"R2{key[:2]}_cos", f"R3{key[:2]}_cos") if key == "ag_cos" else (f"R2_{key}", f"R3_{key}")
        h = np.concatenate([z[k2], z[k3]])
        ax.hist(h, bins=bins, density=True, histtype="step", color=c, ls=ls, lw=1.0, label=lab)
    ax.set_xlabel("cos(velocity, direction to target)")
    ax.set_ylabel("density")
    ax.set_title("(a) Flight data", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")
    despine(ax)
    STATUS.append("fig11a measured (data_goal_cos.npz)")
    ax = axs[1]
    for t, lab, c in (("agg_x500", "R2", "#1B9E77"), ("agg_lmf2", "R3", "#7570B3"), ("agg_baseQuad", "R1 (held out)", "#D95F02")):
        g = ex[ex.task == t].sort_values("cruise")
        ax.plot(g.cruise, 100 * g.reached_05, marker="o", ms=2.4, color=c, label=lab)
    ax.axvspan(0.5, 5.0, color="#EEEEEE", lw=0, zorder=0)
    ax.text(0.6, 93, "cruise speeds\nin the data", fontsize=5.4, color="#666666", va="top")
    ax.set_xlabel("cruise speed of the collector (m/s)")
    ax.set_ylabel("reach rate, T3 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Scripted collector on T3", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right")
    despine(ax)
    STATUS.append("fig11b measured (expert_eval.csv)")
    # (c) T3 reach of scratch, original, and target-swapped backbones (screening seeds, fixed-state evaluation)
    ax = axs[2]
    e = ev[ev.task.isin(M.T3) & (ev.disturb == 0) & (ev.iter >= 550) & ev.seed.isin(M.SEEDS)]
    rows = [("scratch", e[(e.stage == 1) & (e.init == "random") & (e["mode"] == "FullFT")], COL["scratch"]),
            ("MAE", e[(e.stage == 1) & (e.init == "MAE") & (e["mode"] == "FullFT")], COL["MAE"]),
            ("MAE swapped", e[(e.stage == 8) & (e.init == "MAEtswap")], COL["MAE"]),
            ("MAE agnostic", e[(e.stage == 26) & (e.init == "MAEag") & (e["mode"] == "FullFT")], COL["MAE"]),
            ("AP", e[(e.stage == 1) & (e.init == "PIDM") & (e["mode"] == "FullFT")], COL["PIDM"]),
            ("AP swapped", e[(e.stage == 8) & (e.init == "PIDMtswap")], COL["PIDM"]),
            ("AP agnostic", e[(e.stage == 26) & (e.init == "PIDMag") & (e["mode"] == "FullFT")], COL["PIDM"])]
    for i, (lab, g, c) in enumerate(rows):
        runs = g.groupby(["task", "seed"]).reached_05.mean()
        if len(runs) >= 6:
            ax.bar(i, 100 * runs.mean(), color=c, alpha=0.85 if ("swapped" not in lab and "agnostic" not in lab) else 0.45, width=0.7)
            ax.plot([i] * len(runs), 100 * runs.values, "o", color="#333333", ms=1.6)
            STATUS.append(f"fig11c {lab.replace(chr(10), ' ')} measured ({len(runs)} runs)")
        else:
            key = ("MAE" if lab.startswith("MAE") else "PIDM") + ("ag" if "agnostic" in lab else "tswap")
            ax.bar(i, 100 * EXPECT["S10"][key], facecolor="white", edgecolor=RED, ls="--", lw=0.9, width=0.7)
            STATUS.append(f"fig11c {lab.replace(chr(10), ' ')} PLACEHOLDER ({len(runs)} of 6 runs evaluated)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0].replace("MAE ", "").replace("AP ", "").replace("swapped", "swap.").replace("agnostic", "agn.")
                        for r in rows], fontsize=5.8, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("reach rate, T3 (%)")
    ax.set_title("(c) Backbones after FullFT", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.30)
    fig.savefig(OUT / "fig11_source.pdf")
    fig.savefig(OUT / "fig11_source.png", dpi=200)
    plt.close(fig)

def fig_takeoff(b):
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret"])
    s2 = AN.with_reused(s)
    s2 = s2[(s2.stage == 2) & s2.task.isin(M.T3)]
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    sc = s2[(s2.init == "random") & (s2["mode"] == "FullFT")]
    best_scratch = sc.groupby(["task", "seed"]).ret.apply(lambda r: r.rolling(30, min_periods=10).mean().max()).max()
    for (t, sd), g in sc.groupby(["task", "seed"]):
        r = g.sort_values("iter").ret.rolling(30, min_periods=10).mean()
        ax.plot(g.sort_values("iter").iter, r, color="#BBBBBB", lw=0.5)
    pm = s2[(s2.init == "MAE") & (s2["mode"] == "FullFT")]
    for (t, sd), g in pm.groupby(["task", "seed"]):
        g = g.sort_values("iter")
        r = g.ret.rolling(30, min_periods=10).mean()
        c = COL["MAE"] if r.max() > best_scratch else "#1F3A5F"
        ax.plot(g.iter, r, color=c, lw=0.7, alpha=0.9)
    ax.axhline(best_scratch, color="#666666", lw=0.6, ls=":")
    ax.text(10, best_scratch + 8, "best scratch run", fontsize=5.4, color="#666666")
    ax.set_xlabel("RL iteration")
    ax.set_ylabel("training return (30-it. mean)")
    ax.set_title("(a) MAE + FullFT, T3, ten seeds", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    STATUS.append("fig12a measured (series.csv)")
    ax = axs[1]
    s10 = s[(s.stage == 10)]
    failed = [("agg_x500", 21), ("agg_x500", 512), ("agg_lmf2", 13), ("agg_lmf2", 21), ("agg_lmf2", 99), ("agg_lmf2", 123)]
    ctrl = [("agg_x500", 99), ("agg_lmf2", 7)]
    for (t, sd) in failed + ctrl:
        g = s10[(s10.task == t) & (s10.seed == sd)].sort_values("iter")
        c = "#1F3A5F" if (t, sd) in failed else COL["MAE"]
        if len(g) >= 1800:
            ax.plot(g.iter, g.ret.rolling(30, min_periods=10).mean(), color=c, lw=0.8)
            STATUS.append(f"fig12b {t} seed {sd} measured")
        else:
            lvl = 75 if (t, sd) in failed else 260
            if len(g):
                ax.plot(g.iter, g.ret.rolling(30, min_periods=10).mean(), color=c, lw=0.8)
            ax.plot([len(g) if len(g) else 600, 1800], [lvl, lvl], color=RED, ls="--", lw=0.8)
            STATUS.append(f"fig12b {t} seed {sd} PLACEHOLDER after {len(g)} iterations")
    ax.axvline(600, color="#999999", lw=0.6, ls=":")
    ax.axhline(best_scratch, color="#666666", lw=0.6, ls=":")
    ax.set_xlabel("RL iteration")
    ax.set_ylabel("training return (30-it. mean)")
    ax.set_title("(b) Budget extended to 1,800 iterations", loc="left", fontweight="bold", fontsize=7.0)
    if any("PLACEHOLDER" in x for x in STATUS if x.startswith("fig12b")):
        M.placeholder_tag(ax, y=0.90)
    despine(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.21)
    fig.savefig(OUT / "fig12_takeoff.pdf")
    fig.savefig(OUT / "fig12_takeoff.png", dpi=200)
    plt.close(fig)

def main():
    b = AN.with_reused(pd.read_csv(A / "budgets.csv"))
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    fig_rewrite(b)
    fig_source(ev)
    fig_takeoff(b)
    print("\n".join(STATUS))

if __name__ == "__main__":
    main()
