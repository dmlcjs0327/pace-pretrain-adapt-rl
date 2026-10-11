"""Data figures of the v6 manuscript (Study 2 v2), drawn from the analysis tables of experiments/logs/study2/analysis.

Every figure uses the runs that exist when the script runs. Parts that depend on runs that have not finished are drawn
as red dashed placeholders with the label 'placeholder' (expected values, to be replaced), so that a draft can be
complete in form without hiding which parts are measured. Run names differ from the manuscript names: IDM is PIDM,
PIDM is AP, MAEs is MAE-T, LateFT is PartialFT, EncFrozen is Frozen, BackboneLoRA is LoRA, DiscLR is Small-LR.
Wide figures are drawn at the one-column text width of elsarticle (5.4 in) with text of at least 6.5 pt.
Usage: python make_v6_figs.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

A = Path(_EXP_ROOT + "/logs/study2/analysis")
CK = Path(_AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3")
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 7.2,
                     "axes.labelsize": 7.2, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
                     "axes.titlesize": 7.4, "axes.linewidth": 0.6, "pdf.fonttype": 42, "lines.linewidth": 1.3})
W = 5.4
RED = "#C00000"
NAME = {"random": "random", "MAE": "MAE", "MAEs": "MAE-T", "CPC": "CPC", "SPR": "SPR", "FDM": "FDM", "IDM": "PIDM",
        "PIDM": "AP", "BC": "BC", "MAEshuf": "MAE shuffled", "PIDMshuf": "AP shuffled"}
LABEL = {"FullFT": "FullFT", "LateFT": "PartialFT", "EncFrozen": "Frozen", "BackboneLoRA": "LoRA", "LPFT": "LP-FT",
         "DiscLR": "Small-LR", "L2SP": "L2-SP", "AuxSSL": "Aux-SSL", "BCreg": "BC-reg"}
COL = {"random": "#8C8C8C", "MAE": "#56B4E9", "MAEs": "#0072B2", "CPC": "#D55E00", "SPR": "#009E73", "FDM": "#E69F00",
       "IDM": "#CC79A7", "PIDM": "#882255", "BC": "#4A3AA7", "scratch": "#222222"}
OBJS = ["MAE", "MAEs", "CPC", "SPR", "FDM", "IDM", "PIDM", "BC"]
MODES = ["FullFT", "LateFT", "EncFrozen", "BackboneLoRA", "LPFT", "DiscLR", "L2SP", "AuxSSL", "BCreg"]
T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
T3 = ["agg_x500", "agg_lmf2"]
T6 = T12 + T3
SEEDS = [7, 42, 99]
DIV = LinearSegmentedColormap.from_list("div", ["#B2182B", "#F2F2F2", "#2166AC"])
rng = np.random.default_rng(7)

def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def placeholder_tag(ax, x=0.98, y=0.04, text="red dashed: placeholder (expected)"):
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom", color=RED, fontsize=6.3, style="italic")

def load():
    b = pd.read_csv(A / "budgets.csv")
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    s = pd.read_csv(A / "series.csv", usecols=["run_id", "stage", "task", "init", "mode", "seed", "iter", "ret", "noise"])
    return b, ev, s

def eval_runs(ev, stage, tasks, iters, disturb=0.0):
    e = ev[(ev.stage == stage) & ev.task.isin(tasks) & ev.iter.isin(iters) & (ev.disturb == disturb) & ev.seed.isin(SEEDS)]
    return e

def gain_table(tab, col, tasks, ref):
    """Mean over tasks of (mean_seeds X(A) - mean_seeds X(scratch, same seeds)) / ref[task]; seeds that A has."""
    out = {}
    for (i, m), g in tab.groupby(["init", "mode"]):
        vals = []
        for t in tasks:
            a = g[g.task == t].set_index("seed")[col]
            sc = tab[(tab.task == t) & (tab.init == "random") & (tab["mode"] == "FullFT")].set_index("seed")[col]
            seeds = sorted(set(a.index) & set(sc.index))
            if not seeds:
                vals = None
                break
            vals.append((a.loc[seeds].mean() - sc.loc[seeds].mean()) / ref[t])
        if vals:
            out[(i, m)] = (float(np.mean(vals)), int(g.seed.nunique()))
    return out

# --------------------------------------------------------------------------------------------------------------
def fig_concept(ev):
    """Fig. 1: schematic of the paired comparison and the hard-task result (noise-free reach rate over training)."""
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(W, 2.15))
    ax = fig.add_axes([0.0, 0.0, 0.50, 1.0])
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 22)
    ax.axis("off")

    def box(x, y, w, h, t, fc="white", ec="#222222", ls="-", fs=6.9, tc="#222222", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6", fc=fc, ec=ec, lw=0.7, ls=ls))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal")

    ax.text(0.5, 21.2, "(a) Paired comparison", fontsize=7.6, fontweight="bold", va="top")
    box(0.8, 14.2, 13.0, 3.6, "reward-free\nflight data", fc="#DCEAF7")
    ax.annotate("", xy=(16.2, 16.0), xytext=(13.9, 16.0), arrowprops=dict(arrowstyle="-|>", lw=0.7))
    box(16.3, 14.2, 7.6, 3.6, "Encoder", fc="#DCEAF7")
    box(24.6, 14.2, 6.4, 3.6, "GRU", fc="#DCEAF7")
    box(31.7, 14.2, 6.4, 3.6, "Heads")
    ax.text(25.6, 12.9, "pretrained backbone (8 objectives)", ha="center", fontsize=6.6)
    box(16.3, 6.4, 7.6, 3.6, "Encoder", fc="#EFEFEF", ec="#8C8C8C", ls="--", tc="#555555")
    box(24.6, 6.4, 6.4, 3.6, "GRU", fc="#EFEFEF", ec="#8C8C8C", ls="--", tc="#555555")
    box(31.7, 6.4, 6.4, 3.6, "Heads")
    ax.text(25.6, 5.1, "random backbone (counterpart)", ha="center", fontsize=6.6, color="#555555")
    box(40.0, 7.6, 9.4, 8.8, "same\nadaptation\nmode (9),\nsame PPO,\nsame budget", fs=6.6)
    ax.annotate("", xy=(39.9, 13.6), xytext=(38.2, 16.0), arrowprops=dict(arrowstyle="-|>", lw=0.7))
    ax.annotate("", xy=(39.9, 10.4), xytext=(38.2, 8.2), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#8C8C8C"))
    ax.text(25.0, 1.6, "Controls: shuffled weights, difficulty ladder,\nBC action noise, equal compute",
            ha="center", fontsize=6.6, style="italic")

    ax2 = fig.add_axes([0.585, 0.19, 0.39, 0.66])
    e = eval_runs(ev, 1, T3, [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 575, 600])
    for (i, m, lab, c, ls) in (("random", "FullFT", "scratch", COL["scratch"], "-"), ("MAE", "FullFT", "MAE + FullFT", COL["MAE"], "-"),
                               ("PIDM", "FullFT", "AP + FullFT", COL["PIDM"], "-")):
        g = e[(e.init == i) & (e["mode"] == m)]
        p = g.pivot_table(index="iter", columns=["task", "seed"], values="reached_05")
        mean = p.mean(1)
        lo, hi = p.min(1), p.max(1)
        ax2.plot(mean.index, 100 * mean.values, color=c, ls=ls, label=lab)
        ax2.fill_between(mean.index, 100 * lo.values, 100 * hi.values, color=c, alpha=0.12, lw=0)
    ax2.set_xlabel("PPO iteration")
    ax2.set_ylabel("episodes reaching the target (%)")
    ax2.set_title("(b) Hard task T3, mean action", loc="left", fontweight="bold")
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=False, loc="upper left")
    despine(ax2)
    fig.savefig(OUT / "fig1_concept.pdf")
    fig.savefig(OUT / "fig1_concept.png", dpi=200)
    plt.close(fig)

def fig_grid(b, ev):
    """Fig. 3: gain over training from scratch for every backbone and mode, early (eval at 100) and final (600)."""
    e = eval_runs(ev, 1, T6, [100, 550, 575, 600])
    e100 = e[e.iter == 100][["task", "init", "mode", "seed", "ret"]]
    term = e[e.iter >= 550].groupby(["task", "init", "mode", "seed"]).ret.mean().reset_index()
    sc_term = term[(term.init == "random") & (term["mode"] == "FullFT")].groupby("task").ret.mean()
    tr = b[(b.stage == 1) & (b.K == 600) & b.task.isin(T6) & b.seed.isin(SEEDS)][["task", "init", "mode", "seed", "ret"]]
    sc_tr = tr[(tr.init == "random") & (tr["mode"] == "FullFT")].groupby("task").ret.mean()
    panels = [("(a) Early: eval. at 100 it., T1+T2", gain_table(e100, "ret", T12, sc_term)),
              ("(b) Final: return at 600 it., T1+T2", gain_table(tr, "ret", T12, sc_tr)),
              ("(c) Final: return at 600 it., T3", gain_table(tr, "ret", T3, sc_tr))]
    rows = ["MAE", "MAEs", "CPC", "SPR", "FDM", "IDM", "PIDM", "BC", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.08))
    lims = [0.6, 0.6, 2.5]
    partial = False
    for ax, (title, gt), lim in zip(axes, panels, lims):
        M = np.full((len(rows), len(MODES)), np.nan)
        N = np.zeros_like(M)
        for r, i in enumerate(rows):
            for c, m in enumerate(MODES):
                if (i, m) in gt:
                    M[r, c], N[r, c] = gt[(i, m)]
                    partial |= N[r, c] < 3
        im = ax.imshow(M, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        for r in range(len(rows)):
            for c in range(len(MODES)):
                if not np.isnan(M[r, c]):
                    v = M[r, c]
                    txt = f"{v:+.2f}" if abs(v) < 9.995 else f"{v:+.0f}"
                    ax.text(c, r, txt.replace("+0.", "+.").replace("-0.", "-."), ha="center", va="center", fontsize=5.0,
                            color="white" if abs(v) > 0.62 * lim else "#222222",
                            fontstyle="italic" if N[r, c] < 3 else "normal")
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([LABEL[m] for m in MODES], rotation=90, fontsize=6.2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([NAME[i] for i in rows] if ax is axes[0] else [], fontsize=6.6)
        ax.set_title(title, fontsize=6.6, loc="left", fontweight="bold")
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.25, aspect=22)
        cb.ax.tick_params(labelsize=6)
        if lim < 1:
            cb.set_ticks([-0.5, 0, 0.5])
        cb.outline.set_linewidth(0.4)
    if partial:
        fig.text(0.01, 0.005, "Italic: fewer than three seeds so far.", color=RED, fontsize=6.2)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.12)
    fig.savefig(OUT / "fig3_grid.pdf")
    fig.savefig(OUT / "fig3_grid.png", dpi=200)
    plt.close(fig)

def fig_curves(ev):
    """Fig. 4: noise-free learning curves in T1 and T2 per adaptation mode, normalized by the scratch final evaluation."""
    its = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 575, 600]
    e = eval_runs(ev, 1, T12, its)
    ref = e[(e.init == "random") & (e["mode"] == "FullFT") & (e.iter >= 550)].groupby("task").ret.mean()
    e = e.assign(norm=e.ret / e.task.map(ref))
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.95), sharey=True, gridspec_kw=dict(wspace=0.10))
    for ax, m in zip(axes, ["FullFT", "LateFT", "EncFrozen", "BackboneLoRA"]):
        sc = e[(e.init == "random") & (e["mode"] == "FullFT")].groupby(["iter", "task"]).norm.mean().groupby("iter").mean()
        ax.plot(sc.index, sc.values, color=COL["scratch"], ls=":", lw=1.0, label="scratch")
        for i in ("random", "MAE", "PIDM"):
            g = e[(e.init == i) & (e["mode"] == m)]
            p = g.groupby(["iter", "task", "seed"]).norm.mean().reset_index()
            piv = p.pivot_table(index="iter", columns=["task", "seed"], values="norm")
            tasks = sorted(set(piv.columns.get_level_values(0)))
            curve = np.nanmean([piv[t].mean(1) for t in tasks], axis=0)
            seeds = sorted(set(piv.columns.get_level_values(1)))
            bs = [np.nanmean([piv[t].reindex(columns=list(rng.choice(seeds, len(seeds)))).mean(1) for t in tasks], axis=0)
                  for _ in range(200)]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            lab = {"random": "random backbone", "MAE": "MAE", "PIDM": "AP"}[i]
            ax.plot(piv.index, curve, color=COL[i], label=lab)
            ax.fill_between(piv.index, lo, hi, color=COL[i], alpha=0.15, lw=0)
        ax.set_title(LABEL[m], loc="left", fontweight="bold")
        ax.set_xlim(0, 615)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_xlabel("PPO iteration")
        ax.axhline(1.0, color="#BBBBBB", lw=0.5, zorder=0)
        despine(ax)
    axes[0].set_ylabel("eval. return / scratch final")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.55, 1.0), fontsize=6.6)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.22)
    fig.savefig(OUT / "fig4_curves.pdf")
    fig.savefig(OUT / "fig4_curves.png", dpi=200)
    plt.close(fig)

def fig_shuffle(b, ev):
    """Fig. 5: position of the shuffled backbones between the random (0) and the original pretrained backbone (1)."""
    e = ev[ev.task.isin(T6) & ev.stage.isin([1, 5]) & ev.seed.isin(SEEDS) & (ev.disturb == 0)]
    e100 = e[e.iter == 100]
    eterm = e[e.iter >= 550].groupby(["task", "init", "mode", "seed"]).ret.mean()
    tr = b[(b.K == 600) & b.task.isin(T6) & b.stage.isin([1, 5]) & b.seed.isin(SEEDS)]

    def position(tab, col, tasks, base, m, den):
        P = tab.pivot_table(index=["task", "seed"], columns=["init", "mode"], values=col)

        def stat(ss):
            o = s_ = 0.0
            for t in tasks:
                X = P.loc[t].reindex(ss)
                r = X[("random", m)].mean()
                d = den[t]
                o += (X[(base, m)].mean() - r) / d
                s_ += (X[(base + "shuf", m)].mean() - r) / d
            return s_ / o if abs(o) > 1e-9 else np.nan
        pt = stat(SEEDS)
        # v13 (review of v12.3): per-seed positions and their range instead of a three-seed percentile bootstrap
        singles = [stat([sd]) for sd in SEEDS]
        print(f"position {base}shuf {m}: {pt:.2f} per seed {[round(v, 2) for v in singles]}")
        return pt, np.nanmin(singles), np.nanmax(singles), singles

    den_e = eterm.reset_index().query("init=='random' and mode=='FullFT'").groupby("task").ret.mean()
    den_t = tr[(tr.init == "random") & (tr["mode"] == "FullFT")].groupby("task").ret.mean()
    reads = [("early, eval. at 100 it.,\nT1 and T2", e100, "ret", T12, den_e),
             ("final, return at 600 it.,\nT3", tr, "ret", T3, den_t)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.35))
    x0 = np.arange(len(reads))
    combos = [("MAE", "FullFT"), ("PIDM", "FullFT"), ("MAE", "BackboneLoRA"), ("PIDM", "BackboneLoRA")]
    wbar = 0.19
    for k, (base, m) in enumerate(combos):
        vals = [position(tab, col, tasks, base, m, den) for _, tab, col, tasks, den in reads]
        pts = [v[0] for v in vals]
        err = np.array([[v[0] - v[1], v[2] - v[0]] for v in vals]).T
        err = np.clip(err, 0, 2.5)
        hatch = "" if m == "FullFT" else "////"
        ax.bar(x0 + (k - 1.5) * wbar, pts, wbar * 0.92, color=COL[base], hatch=hatch, edgecolor="white", lw=0.3,
               label=f"{NAME[base]} + {LABEL[m]}")
        ax.errorbar(x0 + (k - 1.5) * wbar, pts, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5)
        for j, v in enumerate(vals):
            ax.scatter([x0[j] + (k - 1.5) * wbar] * len(v[3]), np.clip(v[3], -1.05, 1.7), s=4, color="#222222", zorder=3, lw=0)
    ax.axhline(0, color="#8C8C8C", lw=0.6)
    ax.axhline(1, color="#333333", lw=0.6, ls="--")
    ax.text(-0.48, 1.03, "original", fontsize=6.3, ha="left", va="bottom")
    ax.text(-0.48, -0.10, "random", fontsize=6.3, ha="left", va="top", color="#666666")
    ax.set_xticks(x0)
    ax.set_xticklabels([r[0] for r in reads], fontsize=6.6)
    ax.set_ylabel("share of the gain kept")
    ax.set_ylim(-1.1, 1.75)
    ax.set_title("(a) Shuffled weights keep the scale, not the structure", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="lower left", handlelength=1.2, columnspacing=0.8)
    despine(ax)
    # weight scale of the pretrained backbones relative to the default initialization of PyTorch
    names = ["proprio_mlp.0.weight", "proprio_mlp.2.weight", "fusion_mlp.0.weight", "fusion_mlp.2.weight",
             "gru.weight_ih_l0", "gru.weight_hh_l0"]
    labs = ["enc. 1", "enc. 2", "enc. 3", "enc. 4", "GRU in", "GRU rec."]
    for k, o in enumerate(["MAE", "PIDM"]):
        rat = []
        for s in SEEDS:
            sd = torch.load(CK / f"P_{o}__seed{s}__srcmixed__backbone.pt", map_location="cpu")["backbone_state_dict"]
            row = []
            for n in names:
                v = sd[n]
                d = 1 / math.sqrt(3 * (128 if n.startswith("gru") else v.shape[1]))
                row.append(float(v.std()) / d)
            rat.append(row)
        rat = np.array(rat)
        ax2.bar(np.arange(len(names)) + (k - 0.5) * 0.38, rat.mean(0), 0.36, color=COL[o], label=NAME[o],
                yerr=[rat.mean(0) - rat.min(0), rat.max(0) - rat.mean(0)], error_kw=dict(elinewidth=0.6, capsize=1.2))
    ax2.axhline(1, color="#333333", lw=0.6, ls="--")
    ax2.text(-0.45, 1.08, "default init.", fontsize=6.2, ha="left")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(labs, rotation=35, ha="right", fontsize=6.4)
    ax2.set_ylabel("std. / default std.")
    ax2.set_title("(b) Weight scale after pretraining", loc="left", fontweight="bold", fontsize=7.0)
    ax2.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.0, 0.97))
    despine(ax2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.25)
    fig.savefig(OUT / "fig5_shuffle.pdf")
    fig.savefig(OUT / "fig5_shuffle.png", dpi=200)
    plt.close(fig)

LAD = {"agg_d1_x500": ("d1", "R2"), "agg_d1_lmf2": ("d1", "R3"), "agg_d2_x500": ("d2", "R2"), "agg_d2_lmf2": ("d2", "R3"),
       "agg_x500": ("d3", "R2"), "agg_lmf2": ("d3", "R3")}

def fig_ladder(b, ev):
    """Fig. 6: difficulty ladder. (a-c) reach rate of training episodes at 100, 300, 600 it. per level; (d) final gain
    against the scratch reach rate (fixed-state evaluation)."""
    d = b[b.task.isin(LAD) & b.seed.isin(SEEDS) & b.init.isin(["random", "MAE", "PIDM"]) & b["mode"].isin(["FullFT", "BackboneLoRA"])]
    d = d[(d.stage == 6) | ((d.stage == 1) & d.task.isin(T3))]
    fig = plt.figure(figsize=(W, 2.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.42)
    lvls = [("d1", "d1: 1–2 m"), ("d2", "d2: 2–4 m"), ("d3", "d3: 4–8 m (T3)")]
    for k, (lv, txt) in enumerate(lvls):
        ax = fig.add_subplot(gs[0, k])
        tasks = [t for t, (l, _) in LAD.items() if l == lv]
        for i, m, lab, c in (("random", "FullFT", "scratch", COL["scratch"]), ("MAE", "FullFT", "MAE + FullFT", COL["MAE"]),
                             ("PIDM", "FullFT", "AP + FullFT", COL["PIDM"])):
            g = d[d.task.isin(tasks) & (d.init == i) & (d["mode"] == m)]
            y = g.groupby("K").reached_05.mean()
            ax.plot(y.index, 100 * y.values, marker="o", ms=2.6, color=c, label=lab)
        ax.set_title(f"({'abc'[k]}) {txt}", loc="left", fontweight="bold", fontsize=7.0)
        ax.set_xticks([100, 300, 600])
        ax.set_xlim(60, 640)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("PPO iteration")
        if k == 0:
            ax.set_ylabel("reach rate, training (%)")
            ax.legend(frameon=False, fontsize=5.9, loc="lower right", handlelength=1.2)
        else:
            ax.set_yticklabels([])
        despine(ax)
    ax = fig.add_subplot(gs[0, 3])
    e = ev[ev.task.isin(LAD) & ev.seed.isin(SEEDS) & (ev.disturb == 0) & ev.iter.isin([550, 575, 600])]
    e = e[(e.stage == 6) | ((e.stage == 1) & e.task.isin(T3))]
    r = e.groupby(["task", "init", "mode", "seed"]).reached_05.mean().reset_index()
    sc = r[(r.init == "random") & (r["mode"] == "FullFT")].groupby("task").reached_05.mean()
    for i, m, c, mk in (("MAE", "FullFT", COL["MAE"], "o"), ("PIDM", "FullFT", COL["PIDM"], "o"),
                        ("MAE", "BackboneLoRA", COL["MAE"], "D"), ("PIDM", "BackboneLoRA", COL["PIDM"], "D")):
        g = r[(r.init == i) & (r["mode"] == m)].groupby("task").reached_05.mean()
        xs = [100 * sc[t] for t in LAD if t in g.index]
        ys = [100 * (g[t] - sc[t]) for t in LAD if t in g.index]
        rho = spearmanr(xs, ys).correlation
        ax.scatter(xs, ys, s=11, marker=mk, facecolor=c if m == "FullFT" else "white", edgecolor=c, lw=0.8,
                   label=f"{NAME[i]} + {LABEL[m]} ($\\rho$ = {rho:+.2f})", zorder=3)
    ax.axhline(0, color="#8C8C8C", lw=0.6)
    ax.set_xlabel("scratch reach rate (%)")
    ax.set_ylabel("gain over scratch (pp)")
    ax.set_title("(d) Final gain vs difficulty", loc="left", fontweight="bold", fontsize=7.0)
    ax.set_xlim(-5, 105)
    despine(ax)
    h, l = ax.get_legend_handles_labels()  # the legend of (d) sits below the panels, where it covers no point
    fig.legend(h, l, frameon=False, fontsize=5.6, loc="lower center", ncol=4, bbox_to_anchor=(0.56, 0.0), handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.33)
    fig.savefig(OUT / "fig6_ladder.pdf")
    fig.savefig(OUT / "fig6_ladder.png", dpi=200)
    plt.close(fig)

def fig_decline():
    """Fig. 7: late decline (noise-free learning curves of the stage-1 runs): best evaluation minus the final evaluation
    (mean of 550-600), in units of the final evaluation of training from scratch in the same setting (S6)."""
    c = pd.read_csv(A / "cumulative_learning_runs.csv")
    modes = ["FullFT", "LateFT", "BackboneLoRA", "LPFT", "DiscLR", "L2SP", "AuxSSL", "EncFrozen"]
    fig, ax = plt.subplots(figsize=(W * 0.62, 1.85))
    x = np.arange(len(modes))
    for k, (grp, col, lab) in enumerate((("pre", "#4C72B0", "pretrained (pooled)"), ("random", COL["random"], "random backbone"))):
        pts, los, his = [], [], []
        for m in modes:
            g = c[(c["mode"] == m) & ((c.init != "random") if grp == "pre" else (c.init == "random")) & (c.init != "BC")]
            v = -g.final_minus_best.values
            if len(v) == 0:
                pts.append(np.nan); los.append(0); his.append(0)
                continue
            bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
            pts.append(v.mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            los.append(v.mean() - lo)
            his.append(hi - v.mean())
        ax.bar(x + (k - 0.5) * 0.38, 100 * np.array(pts), 0.36, color=col, label=lab,
               yerr=100 * np.array([los, his]), error_kw=dict(elinewidth=0.6, capsize=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in modes], rotation=30, ha="right", fontsize=6.6)
    ax.set_ylabel("best minus final\n(% of scratch final eval.)")
    ax.legend(frameon=False, fontsize=6.4)
    despine(ax)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.95, bottom=0.28)
    fig.savefig(OUT / "fig7_decline.pdf")
    fig.savefig(OUT / "fig7_decline.png", dpi=200)
    plt.close(fig)

def fig_bc(ev, s):
    """Fig. 8: (a) noise-free evaluation of BC in T1 against scratch and MAE, with the S9 placeholder; (b) evaluation of
    the mean action over the logged return of sampled actions while the actor is frozen (iterations 26-50)."""
    T1 = ["state_setpoint_x500_clean", "ss_lmf2_clean"]
    its = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 575, 600]
    e = eval_runs(ev, 1, T1, its)
    ref = e[(e.init == "random") & (e["mode"] == "FullFT") & (e.iter >= 550)].groupby("task").ret.mean()
    e = e.assign(norm=e.ret / e.task.map(ref))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W, 1.95), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.32))
    curves = {}
    for i, m, lab, c, ls in (("random", "FullFT", "scratch", COL["scratch"], ":"), ("MAE", "FullFT", "MAE + FullFT", COL["MAE"], "-"),
                             ("BC", "FullFT", "BC + FullFT", COL["BC"], "-"), ("BC", "BCreg", "BC + BC-reg", COL["BC"], "--")):
        g = e[(e.init == i) & (e["mode"] == m)].groupby(["iter", "task"]).norm.mean().groupby("iter").mean()
        curves[(i, m)] = g
        ax.plot(g.index, g.values, color=c, ls=ls, label=lab)
    # S9 (pre-registration section 16): BC with the initial noise scaled by 1/4, stage 7 of the evaluation table;
    # a red dashed placeholder is drawn only while its evaluation is missing
    e7 = eval_runs(ev, 7, T1, its)
    if len(e7) and e7.iter.nunique() >= 8:
        e7 = e7.assign(norm=e7.ret / e7.task.map(ref))
        g7 = e7[(e7.init == "BC") & (e7["mode"] == "FullFT")].groupby(["iter", "task"]).norm.mean().groupby("iter").mean()
        ax.plot(g7.index, g7.values, color="#E69F00", ls="-", lw=1.3, label="BC + FullFT, noise $\\times$1/4")
    else:
        bc = curves[("BC", "FullFT")]
        xs = bc.index.values
        ph = np.interp(xs, [25, 50, 100, 200, 300, 600], [bc.loc[25], bc.loc[50], 0.55 * bc.loc[25], 0.6 * bc.loc[25],
                                                         0.7 * bc.loc[25], 0.85 * bc.loc[25]])
        ax.plot(xs, ph, color=RED, ls="--", lw=1.0, label="BC + FullFT, noise $\\times$1/4 (placeholder)")
    ax.axvspan(0, 50, color="#EEEEEE", lw=0, zorder=0)
    ax.text(25, 0.70, "critic\nwarm-up", fontsize=6.0, ha="center", va="bottom", color="#666666")
    ax.set_xlim(0, 600)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("eval. return / scratch final")
    ax.set_title("(a) T1, mean-action evaluation", loc="left", fontweight="bold", fontsize=7.0)
    ax.set_ylim(-0.05, 2.05)
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=2, handlelength=1.6, columnspacing=0.8)
    despine(ax)
    # (b) ratio of the mean-action evaluation to the logged (sampled) return while the actor is frozen
    tr = s[(s.stage == 1) & s.task.isin(T12) & (s.iter > 25) & (s.iter <= 50)].groupby(["task", "init", "mode", "seed"]).ret.mean()
    ee = ev[(ev.stage == 1) & ev.task.isin(T12) & (ev.disturb == 0) & ev.iter.isin([25, 50])].groupby(["task", "init", "mode", "seed"]).ret.mean()
    j = pd.concat([tr.rename("train"), ee.rename("eval")], axis=1, join="inner").reset_index()
    j = j[j["mode"].isin(["FullFT", "EncFrozen"])]
    j["ratio"] = j["eval"] / j["train"]
    j["T"] = np.where(j.task.isin(T1), "T1", "T2")
    inits = ["BC", "MAE", "PIDM", "random"]
    x = np.arange(len(inits))
    for k, T in enumerate(["T1", "T2"]):
        v = [j[(j.init == i) & (j["T"] == T)].ratio for i in inits]
        ax2.bar(x + (k - 0.5) * 0.38, [vv.mean() for vv in v], 0.36, color=["#4A3AA7", "#56B4E9", "#882255", "#8C8C8C"],
                alpha=1.0 if T == "T1" else 0.45, edgecolor="white", lw=0.3,
                yerr=[[vv.mean() - vv.min() for vv in v], [vv.max() - vv.mean() for vv in v]],
                error_kw=dict(elinewidth=0.6, capsize=1.2), label=T)
    ax2.axhline(1, color="#333333", lw=0.6, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([NAME[i] for i in inits], fontsize=6.6)
    ax2.set_ylabel("mean action / sampled actions")
    ax2.set_title("(b) Actor frozen (it. 26-50)", loc="left", fontweight="bold", fontsize=7.0)
    ax2.text(0.98, 0.95, "dark: T1, light: T2", transform=ax2.transAxes, ha="right", va="top", fontsize=6.0)
    despine(ax2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.22)
    fig.savefig(OUT / "fig8_bc.pdf")
    fig.savefig(OUT / "fig8_bc.png", dpi=200)
    plt.close(fig)

def fig_compute(ev):
    """Fig. 9: cost of the gain in T3 (fixed-state evaluation without wind, reach rate, R2 and R3 averaged).
    (a) Training from scratch to 3,000 iterations (screening runs to 600, then the equal-compute runs of stage 3, seeds
    7, 42, 99) against MAE and AP with FullFT after 600 iterations (confirmation stage, ten seeds). Placeholder: the
    cost of collecting the data and pretraining, measured alone on the GPU (open circles, cost_final.json). (b) Oracle best of n scratch runs over the ten confirmation seeds,
    Eq. (bestofn), against one pretrained run (mean and range over the ten seeds)."""
    T3m = ev.task.isin(T3) & (ev.disturb == 0)
    s1 = ev[T3m & (ev.stage == 1) & ev.seed.isin(SEEDS) & (ev.init == "random") & (ev["mode"] == "FullFT")]
    lng = ev[T3m & (ev.stage == 3) & ev.run_id.str.contains("it3000")]
    # confirmation stage: stage-2 runs plus the screening runs it reuses (R2, R3, seeds 7, 42, 99)
    c2 = ev[T3m & ((ev.stage == 2) | ((ev.stage == 1) & ev.seed.isin(SEEDS) & ev["mode"].isin(["FullFT", "AuxSSL"])
                                      & ev.init.isin(["random", "MAE", "PIDM"])))]
    c2 = c2[c2.iter >= 550].drop_duplicates(["task", "init", "mode", "seed", "iter"], keep="first")
    term = c2.groupby(["task", "init", "mode", "seed"]).reached_05.mean().reset_index()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W, 1.95), gridspec_kw=dict(wspace=0.32))
    cur1 = s1[s1.iter <= 600].groupby(["iter", "seed"]).reached_05.mean().groupby("iter").mean()
    cur3 = lng[lng.iter > 600].groupby(["iter", "seed"]).reached_05.mean().groupby("iter").mean()
    curve = pd.concat([cur1, cur3]).sort_index()
    ax.plot(curve.index / 600, 100 * curve.values, color=COL["scratch"], label="scratch (3 seeds)")
    cf = json.loads((A / "cost_final.json").read_text())["objectives"]
    cost = {"MAE": cf["MAE"]["c"], "PIDM": cf["AP"]["c"]}  # data collection + pretraining, in standalone FullFT runs
    for o in ("MAE", "PIDM"):
        g = term[(term.init == o) & (term["mode"] == "FullFT")].groupby("seed").reached_05.mean()
        ax.errorbar([1], [100 * g.mean()], yerr=[[100 * (g.mean() - g.quantile(0.25))], [100 * (g.quantile(0.75) - g.mean())]],
                    fmt="o", ms=3.6, color=COL[o], lw=0.8, capsize=1.5, zorder=3, label=f"{NAME[o]} + FullFT, RL only (10 seeds)")
        ax.scatter([1 + cost[o]], [100 * g.mean()], facecolor="white", edgecolor=COL[o], s=14, zorder=3, lw=0.9)
        ax.annotate("", xy=(1 + cost[o], 100 * g.mean()), xytext=(1.08, 100 * g.mean()),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=COL[o], ls="--"))
    ax.text(1.3, 6, "open circles: + data and pretraining", color="#555555", fontsize=6.0)
    ax.set_xlabel("total compute (600-iteration RL runs)")
    ax.set_ylabel("reach rate, T3 (%)")
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-2, 50)
    ax.set_title("(a) Equal compute, T3", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    despine(ax)
    sc = term[(term.init == "random") & (term["mode"] == "FullFT")]
    vals = sc.groupby("seed").reached_05.mean().sort_values().values
    m = len(vals)

    def best_of(n):
        return sum(math.comb(k, n - 1) / math.comb(m, n) * vals[k] for k in range(n - 1, m))
    ns = list(range(1, m + 1))
    for o in ("MAE", "PIDM"):
        g = term[(term.init == o) & (term["mode"] == "FullFT")].groupby("seed").reached_05.mean()
        ax2.axhline(100 * g.mean(), color=COL[o], lw=1.0, label=f"one {NAME[o]} + FullFT run (mean)")
        ax2.axhspan(100 * g.quantile(0.25), 100 * g.quantile(0.75), color=COL[o], alpha=0.12, lw=0)
    ax2.plot(ns, [100 * best_of(n) for n in ns], marker="o", ms=2.6, color=COL["scratch"], label="scratch, best of n")
    ax2.set_xlabel("number of scratch runs n")
    ax2.set_ylabel("reach rate, T3 (%)")
    ax2.set_xticks([1, 2, 3, 5, 10])
    ax2.set_ylim(-2, 50)
    ax2.set_title("(b) Selecting the best of n scratch runs", loc="left", fontweight="bold", fontsize=7.0)
    ax2.legend(frameon=False, fontsize=5.6, loc="upper right")
    despine(ax2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.22)
    fig.savefig(OUT / "fig9_compute.pdf")
    fig.savefig(OUT / "fig9_compute.png", dpi=200)
    plt.close(fig)
    return dict(best_of={n: round(100 * best_of(n), 2) for n in ns}, scratch_curve=(100 * curve).round(2).to_dict())

def main():
    b, ev, s = load()
    fig_concept(ev)
    fig_grid(b, ev)
    fig_curves(ev)
    fig_shuffle(b, ev)
    fig_ladder(b, ev)
    fig_decline()
    fig_bc(ev, s)
    fig_compute(ev)
    print("figures written to", OUT)

if __name__ == "__main__":
    main()
