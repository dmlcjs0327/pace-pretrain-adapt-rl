"""Generality figures added in v9 (pre-registration v2, sections 19 and 20), drawn from size_d2.json
(scripts/study2_size_d2.py).

Fig. 13  Network width: gain over training from scratch against the number of policy parameters (37k to 2.3M,
         widths x0.5 to x4), early (100 iterations) and final (600 iterations), for MAE with FullFT, Frozen, and LoRA.
         (c, d) M4 of section 22: per width and seed, the effective dimension of the frozen MAE features against the
         loss of freezing (MAE Frozen vs scratch at 600 iterations, the pre-registered measure) and against the return
         of MAE Frozen over MAE FullFT at the same width (T1/T2), from m4_width_points.csv (study2_m_analyze.py).
Fig. 14  Second domain (quadruped Anymal, full state D1 and without base linear velocity D2): gains over training
         from scratch of MAE and AP with FullFT, Frozen, and LoRA, early and final, and the position of the shuffled
         backbones.
Values that are not yet measured are drawn in red with dashed outlines (expected values of the pre-registration) and
are replaced automatically when the runs exist; the console output says which parts are measured.
Usage: python make_v9_scale_figs.py
"""
import json

import pandas as pd
from scipy.stats import spearmanr

import numpy as np

import make_v6_figs as M
from make_v6_figs import A, COL, OUT, RED, W, despine, plt

PARAMS = {0.5: 37065, 1.0: 145801, 2.0: 578313, 4.0: 2303497}
CONTR = [("MAE|FullFT vs scratch", "MAE + FullFT", "#D55E00", "o"),
         ("MAE|LoRA vs scratch", "MAE + LoRA", "#0072B2", "s"),
         ("MAE|Frozen vs scratch", "MAE + Frozen", "#8C8C8C", "v")]
EXPECT_SIZE = {  # expected values (section 19: H_S1 smaller freezing loss at larger width, H_S2 width-independent head start)
    ("MAE|FullFT vs scratch", 100): {0.5: 0.75, 2.0: 0.80, 4.0: 0.80},
    ("MAE|FullFT vs scratch", 600): {0.5: 0.70, 2.0: 0.78, 4.0: 0.78},
    ("MAE|LoRA vs scratch", 100): {0.5: 0.25, 2.0: 0.40, 4.0: 0.45},
    ("MAE|LoRA vs scratch", 600): {0.5: 0.00, 2.0: 0.15, 4.0: 0.20},
    ("MAE|Frozen vs scratch", 100): {0.5: -0.35, 2.0: -0.20, 4.0: -0.15},
    ("MAE|Frozen vs scratch", 600): {0.5: -0.50, 2.0: -0.30, 4.0: -0.20},
}
D2_TASKS = [("ige_anymal_flat", "ANYmal-full"), ("ige_anymal_blind", "ANYmal-blind")]
D2_CONDS = [("MAE", "FullFT vs scratch", "MAE FullFT"), ("PIDM", "FullFT vs scratch", "AP FullFT"),
            ("MAE", "Frozen vs scratch", "MAE Frozen"), ("PIDM", "Frozen vs scratch", "AP Frozen"),
            ("MAE", "LoRA vs random|LoRA", "MAE LoRA*"), ("PIDM", "LoRA vs random|LoRA", "AP LoRA*")]
EXPECT_D2 = {  # expected values (section 20: the quadrotor findings recur)
    ("FullFT vs scratch", 100): 0.30, ("FullFT vs scratch", 600): 0.05, ("Frozen vs scratch", 100): -0.20,
    ("Frozen vs scratch", 600): -0.30, ("LoRA vs random|LoRA", 100): 0.20, ("LoRA vs random|LoRA", 600): 0.10,
    "P100": 0.5, "P600": 0.3}
STATUS = []

def load():
    p = A / "size_d2.json"
    return json.loads(p.read_text()) if p.exists() else {"size": {}, "d2": {}}

WCOL = {0.5: "#CBC9E2", 1.0: "#9E9AC8", 2.0: "#6A51A3", 4.0: "#3F007D"}  # one hue, light to dark with width

def fig_size(r):
    fig, axs = plt.subplots(1, 4, figsize=(W, 2.05), gridspec_kw=dict(wspace=0.68, width_ratios=[1.15, 1.15, 1.0, 1.0]))
    placeholder = False
    for ax, K, title in ((axs[0], 100, "(a) Early: 100 iterations"), (axs[1], 600, "(b) Final: 600 iterations")):
        for key, lab, col, mk in CONTR:
            xs, ys, lo, hi, meas = [], [], [], [], []
            for w in sorted(PARAMS):
                v = r["size"].get(f"S17|w{w:g}|{key}|K{K}")
                if v is not None:
                    xs.append(PARAMS[w]); ys.append(v[0]); lo.append(v[1]); hi.append(v[2]); meas.append(True)
                elif w in EXPECT_SIZE[(key, K)]:
                    xs.append(PARAMS[w]); ys.append(EXPECT_SIZE[(key, K)][w]); lo.append(np.nan); hi.append(np.nan); meas.append(False)
                    placeholder = True
            xs, ys = np.array(xs), np.array(ys)
            ax.plot(xs, ys, color=col, lw=1.0, zorder=2, label=lab)
            for x, y, l, h, m in zip(xs, ys, lo, hi, meas):
                if m:
                    ax.errorbar([x], [y], yerr=[[y - l], [h - y]], fmt=mk, color=col, ms=4, elinewidth=0.7, capsize=1.5, zorder=3)
                else:
                    ax.plot([x], [y], mk, mfc="white", mec=RED, ms=4.5, mew=0.9, zorder=3)
            STATUS.append(f"fig13 {key} K{K}: measured at {sum(meas)} of {len(meas)} widths")
            # section 23 (R1): width x4 trained again with the recalibrated configuration C1, drawn hollow beside 2.3M
            rw = A / "r_width.json"
            if rw.exists():
                v = json.loads(rw.read_text()).get(f"w4|C1|{key}|K{K}")
                if v is not None:
                    x = PARAMS[4.0] * 1.35
                    ax.errorbar([x], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, mfc="white", mec=col, color=col,
                                ms=4, mew=1.0, elinewidth=0.7, capsize=1.5, zorder=3)
        ax.axhline(0, color="#666666", lw=0.6)
        ax.axvline(PARAMS[1.0], color="#BBBBBB", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([PARAMS[w] for w in sorted(PARAMS)])
        ax.set_xticklabels(["37k", "146k\n(main)", "578k", "2.3M"], fontsize=6.0)
        ax.set_xlim(PARAMS[0.5] / 1.6, PARAMS[4.0] * 1.8)
        ax.set_xlabel("policy parameters")
        ax.set_ylabel("gain over scratch")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.0)
        despine(ax)
    pts = A / "m4_width_points.csv"
    for ax, y, ylab, title in ((axs[2], "frozen_vs_scratch", "Frozen gain over scratch", "(c) Freezing loss"),
                               (axs[3], "frozen_over_fullft", "Frozen / FullFT return", "(d) Within width")):
        if pts.exists():
            d0 = pd.read_csv(pts).dropna(subset=["d_eff", y])
            d = d0[d0.config == "C3"] if "config" in d0 else d0
            for w, g in d.groupby("width"):
                ax.scatter(g.d_eff, g[y], s=16, color=WCOL[w], edgecolor="white", linewidth=0.6, zorder=3,
                           label=f"width x{w:g}")
            if "config" in d0 and (d0.config == "C1").any():  # section 23: width x4 recalibrated (C1), hollow
                g = d0[d0.config == "C1"]
                ax.scatter(g.d_eff, g[y], s=16, facecolor="white", edgecolor=WCOL[4.0], linewidth=0.9, zorder=3,
                           label="width x4, recalibrated")
            rho = spearmanr(d.d_eff, d[y])[0]
            ax.text(0.04, 0.97, f"Spearman {rho:.2f}\nn={len(d)}", transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
            STATUS.append(f"fig13 ({title[1]}) measured, {len(d)} points")
        if y == "frozen_vs_scratch":
            ax.axhline(0, color="#666666", lw=0.6)
        ax.set_xlabel("effective dimensions\nof frozen MAE features")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.0)
        despine(ax)
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = axs[2].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, fontsize=5.6, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=0.9)
    if placeholder:
        fig.text(0.99, 0.01, "red hollow: expected (not yet measured)", ha="right", va="bottom", color=RED, fontsize=6.0, style="italic")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.235)
    fig.savefig(OUT / "fig13_size.pdf")
    fig.savefig(OUT / "fig13_size.png", dpi=200)
    plt.close(fig)

def fig_d2(r):
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.85], wspace=0.42))
    placeholder = False
    x = np.arange(len(D2_CONDS))
    for ax, K, title in ((axs[0], 100, "(a) Early: 100 iterations"), (axs[1], 600, "(b) Final: 600 iterations")):
        for j, (task, tl) in enumerate(D2_TASKS):
            off = (j - 0.5) * 0.38
            for i, (obj, name, _) in enumerate(D2_CONDS):
                v = r["d2"].get(f"S18|{task}|{obj}|{name}|K{K}")
                col = COL["MAE"] if obj == "MAE" else COL["PIDM"]
                if v is not None:
                    ax.bar(x[i] + off, v[0], 0.36, color=col, alpha=1.0 if j == 0 else 0.55, edgecolor="white", lw=0.3)
                    ax.errorbar(x[i] + off, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.2)
                else:
                    ax.bar(x[i] + off, EXPECT_D2[(name, K)], 0.36, facecolor="white", edgecolor=RED, ls="--", lw=0.8)
                    placeholder = True
                STATUS.append(f"fig14 {task} {obj} {name} K{K}: {'measured' if v is not None else 'PLACEHOLDER'}")
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in D2_CONDS], fontsize=6.0, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel("gain over scratch")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.0)
        despine(ax)
    ax = axs[2]
    for j, (task, tl) in enumerate(D2_TASKS):
        for i, (obj, K) in enumerate((("MAE", 100), ("MAE", 600), ("PIDM", 100), ("PIDM", 600))):
            v = r["d2"].get(f"S18|{task}|{obj}shuf|P|K{K}")
            xx = i + (j - 0.5) * 0.38
            col = COL["MAE"] if obj == "MAE" else COL["PIDM"]
            den = r["d2"].get(f"S18|{task}|{obj}|FullFT vs scratch|K{K}")
            if v is not None and den is not None and den[1] <= 0 <= den[2]:
                # position undefined: the gain of the original backbone does not differ from zero
                ax.text(xx, 0.05, "n/a", rotation=90, ha="center", va="bottom", fontsize=5.0, color="#555555")
            elif v is not None:
                ax.bar(xx, v, 0.36, color=col, alpha=1.0 if j == 0 else 0.55, edgecolor="white", lw=0.3)
            else:
                ax.bar(xx, EXPECT_D2[f"P{K}"], 0.36, facecolor="white", edgecolor=RED, ls="--", lw=0.8)
                placeholder = True
    ax.axhline(0, color="#666666", lw=0.6)
    ax.axhline(1, color="#333333", lw=0.6, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["MAE\n100", "MAE\n600", "AP\n100", "AP\n600"], fontsize=5.8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("share of the gain kept by the shuffled backbone")
    ax.set_title("(c) Scale vs. structure", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    fig.text(0.01, 0.01, "Dark bars: ANYmal-full. Light bars: ANYmal-blind. *: compared with the random backbone in LoRA. n/a: position not defined.",
             ha="left", va="bottom", fontsize=6.2)
    if placeholder:
        fig.text(0.99, 0.01, "red dashed: expected (not yet measured)", ha="right", va="bottom", color=RED, fontsize=5.6, style="italic")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.31)
    fig.savefig(OUT / "fig14_domain2.pdf")
    fig.savefig(OUT / "fig14_domain2.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    r = load()
    fig_size(r)
    fig_d2(r)
    for s in STATUS:
        print(s)
