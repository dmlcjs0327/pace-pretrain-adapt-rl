"""Schematics of the manuscript: PACE framework (Fig. 2; v6 draws Fig. 1 in make_v6_figs.py).

Wide figure for the elsarticle one-column text width (5.40 in). All text is at least 6.6 pt.
Colors follow the Okabe-Ito palette of the earlier figures: one color per adaptation mode,
grey for the random counterpart.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 7.4, "pdf.fonttype": 42, "hatch.linewidth": 0.4})
C = {"FullFT": "#0072B2", "PartialFT": "#CC79A7", "Frozen": "#009E73", "LoRA": "#E69F00",
     "grey": "#8C8C8C", "dark": "#222222", "light": "#E6E6E6", "obj": ["#56B4E9", "#D55E00", "#009E73", "#CC79A7"]}

def framework():
    fig = plt.figure(figsize=(5.40, 4.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis("off")

    def box(x, y, w, h, text="", fc="white", ec=C["dark"], lw=0.7, fs=7.2, hatch=None, tc=C["dark"], ha="center"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=lw, hatch=hatch))
        if text:
            tx = x + w / 2 if ha == "center" else x + 1.0
            ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc, linespacing=1.2,
                    bbox=dict(fc="white", ec="none", pad=0.5) if hatch else None)

    def arrow(x0, y0, x1, y1, color=C["dark"], lw=0.9):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=9))

    def title(x, y, t):
        ax.text(x, y, t, fontsize=8.0, weight="bold", ha="left", va="center")

    # (a) reward-free flight data -------------------------------------------------------
    title(0.8, 72.5, "(a) Reward-free flight data")
    box(0.8, 53.0, 29.0, 16.5, "Waypoint generator +\nscripted outer loop through\nthe attitude interface\n4,096 vehicles, R2 and R3\nstates and commands, no reward", fs=6.8)
    ax.text(0.8, 50.2, "One window = 33 steps of one vehicle", fontsize=6.9, ha="left", va="center", style="italic")
    for i in range(11):
        ax.add_patch(plt.Rectangle((0.8 + i * 2.6, 45.0), 2.3, 3.2, fc="#BFD7EA" if i < 11 else "white", ec=C["dark"], lw=0.4))
    ax.text(15.3, 42.6, "standardized, inside one episode", fontsize=6.7, ha="center", va="center", color="#444444")
    arrow(30.3, 61.2, 34.3, 61.2)

    # (b) pretraining --------------------------------------------------------------------
    title(34.6, 72.5, "(b) Pretraining of the backbone")
    box(34.6, 56.5, 12.0, 9.5, "Encoder\nMLP")
    box(50.0, 56.5, 10.0, 9.5, "GRU\n128")
    arrow(46.6, 61.25, 50.0, 61.25)
    ax.text(34.6, 53.2, "Backbone = encoder + GRU\n(128,960 weights)", fontsize=6.9, ha="left", va="center", style="italic", linespacing=1.1)
    objs = [("MAE", "hidden entries"), ("MAE-T", "hidden steps"), ("CPC", "own future latent"),
            ("SPR", "slow-target latents"), ("FDM", "forward dynamics"), ("PIDM", "inverse dynamics"),
            ("AP", "next commands"), ("BC", "command, actor head")]
    pal = ["#56B4E9", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#882255", "#4A3AA7"]
    for i, (o, d) in enumerate(objs):
        x = 63.5 + (i % 2) * 18.4
        y = 66.2 - (i // 2) * 6.0
        box(x, y, 17.6, 5.2, "", fc="white", ec=pal[i], lw=1.0)
        ax.text(x + 1.0, y + 3.6, o, fontsize=7.0, weight="bold", color=C["dark"], ha="left", va="center")
        ax.text(x + 1.0, y + 1.5, d, fontsize=6.0, color="#333333", ha="left", va="center", linespacing=0.95)
    arrow(60.0, 61.25, 64.0, 61.25)
    ax.text(63.5, 45.0, "Temporary heads (kept for BC and Aux-SSL)", fontsize=6.5, ha="left", va="center", color="#444444")
    ax.text(34.6, 44.6, "One recipe for all eight\n(Adam, 25 epochs, clip 1.0)", fontsize=6.6, ha="left", va="center", linespacing=1.0)

    # (c) adaptation modes -----------------------------------------------------------------
    title(0.8, 38.0, "(c) Adaptation modes")
    modes = [("FullFT", [1, 1, 1], "145,801"), ("PartialFT", [0, 1, 1], "115,913"),
             ("Frozen", [0, 0, 1], "16,841"), ("LoRA", [2, 2, 1], "30,257")]
    names = ["Enc.", "GRU", "Heads"]
    for i, (m, flags, cnt) in enumerate(modes):
        y = 30.5 - i * 5.6
        ax.text(0.8, y + 2.8, m, fontsize=7.4, va="center", weight="bold", color=C[m])
        ax.text(0.8, y + 0.8, cnt + " trained", fontsize=6.4, va="center", color="#333333")
        for j, f in enumerate(flags):
            x = 16.5 + j * 8.2
            fc, ls, tc = {1: (C["dark"], "-", "white"), 0: ("#EDEDED", "--", C["dark"]), 2: ("#EDEDED", "--", C["dark"])}[f]
            ax.add_patch(FancyBboxPatch((x, y), 7.4, 3.9, boxstyle="round,pad=0.02,rounding_size=0.5",
                                        fc=fc, ec=C["dark"], lw=0.6, ls=ls))
            ax.text(x + 3.7, y + 1.95, names[j], fontsize=6.6, ha="center", va="center", color=tc)
            if f == 2:
                ax.add_patch(FancyBboxPatch((x + 5.3, y + 2.4), 2.6, 2.0, boxstyle="round,pad=0.02,rounding_size=0.3",
                                            fc=C["LoRA"], ec=C["dark"], lw=0.5))
        ax.add_patch(FancyBboxPatch((42.3, y), 5.8, 3.9, boxstyle="round,pad=0.02,rounding_size=0.5",
                                    fc="white", ec=C["grey"], lw=0.6, ls="--"))
        ax.text(45.2, y + 1.95, "rand.", fontsize=6.4, ha="center", va="center", color=C["grey"])
    ax.text(0.8, 11.2, "Other modes that train every weight:", fontsize=6.6, ha="left", va="center", weight="bold")
    ax.text(0.8, 8.0, "LP-FT (heads first), Small-LR (backbone lr x0.01),\nL2-SP (pull to start), Aux-SSL (keep pretraining loss),\nBC-reg (BC only)", fontsize=6.3, ha="left", va="top", linespacing=1.05)
    arrow(49.0, 19.0, 52.2, 19.0)

    # (d) shared PPO training --------------------------------------------------------------
    title(52.2, 38.0, "(d) Shared pipeline")
    box(52.2, 12.0, 19.5, 22.0, "Aerial Gym, 100 Hz\n\nT1 regulation\nT2 recovery\nT3 aggressive\n\nR1, R2, R3 airframes\n600 iterations, learning\nrate tuned on training\nfrom scratch only", fs=6.6)
    arrow(72.0, 19.0, 75.2, 19.0)

    # (e) measurement ------------------------------------------------------------------------
    title(75.2, 38.0, "(e) Measurement")
    items = [("Return and evaluation", "logged return at 100-600 it.;\nmean action from fixed states"),
             ("Interventions", "shuffled weights, target-\nswapped, task-agnostic data"),
             ("Cost account", "standalone cost, equal\ncompute, best of n runs")]
    for i, (h, d) in enumerate(items):
        y = 27.4 - i * 8.3
        box(75.2, y, 24.0, 7.3)
        ax.text(76.2, y + 5.3, h, fontsize=7.2, weight="bold", ha="left", va="center")
        ax.text(76.2, y + 2.4, d, fontsize=6.6, ha="left", va="center", color="#333333", linespacing=1.05)

    # two stages -----------------------------------------------------------------------------
    ax.text(52.2, 7.4, "Screening on R2 and R3, all conditions, 3 seeds", fontsize=6.8, ha="left", va="center")
    ax.text(52.2, 2.6, "Confirmation on R1 to R3, selected conditions, 10 seeds", fontsize=6.8, ha="left", va="center")
    arrow(50.6, 7.4, 50.6, 2.9, color=C["grey"], lw=0.7)
    fig.savefig(OUT / "fig2_framework.pdf")
    fig.savefig(OUT / "fig2_framework.png", dpi=200)
    plt.close(fig)

def concept():
    """Fig. 1: motivation (top) and the idea of PACE (bottom), two-column text width."""
    import numpy as np
    fig = plt.figure(figsize=(7.0, 3.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 64)
    ax.axis("off")

    def rbox(x, y, w, h, fc="white", ec=C["dark"], lw=0.7, ls="-", r=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))

    def arrow(x0, y0, x1, y1, color=C["dark"], lw=0.9, style="-|>", conn="arc3"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=0,
                                                                      shrinkB=0, mutation_scale=9, connectionstyle=conn))

    def tag(x, y, t):
        ax.text(x, y, t, fontsize=7.0, style="italic", weight="bold", ha="center", va="center", color="#7A2E00",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFF1E0", ec="#D55E00", lw=0.6))

    def policy(x, y, fc_bb, ec_bb, ls="-", label=None, scale=1.0):
        w = [9 * scale, 7 * scale, 7 * scale]
        names = ["Encoder", "GRU", "Heads"]
        xx = x
        for k, (ww, n) in enumerate(zip(w, names)):
            bb = k < 2
            rbox(xx, y, ww, 6 * scale, fc=fc_bb if bb else "white", ec=ec_bb if bb else C["dark"], ls=ls if bb else "-")
            ax.text(xx + ww / 2, y + 3 * scale, n, fontsize=6.6, ha="center", va="center")
            if k < 2:
                arrow(xx + ww, y + 3 * scale, xx + ww + 1.2, y + 3 * scale, lw=0.6)
            xx += ww + 1.2
        if label:
            ax.text(x, y - 1.6, label, fontsize=6.7, ha="left", va="center", color="#333333")
        return xx

    # ---- top left: language and vision ----
    ax.text(1.5, 61.5, "Pretrain-then-adapt in language and vision", fontsize=8.0, weight="bold", ha="left", va="center")
    for i in range(9):
        rbox(4, 32.5 + i * 2.7, 24, 2.2, fc="#D9D9D9", ec="#7F7F7F", lw=0.5, r=0.3)
    ax.text(16, 30.3, "large pretrained network", fontsize=6.7, ha="center", va="center", color="#333333")
    for i in (2, 5):
        rbox(28.6, 32.5 + i * 2.7, 2.6, 2.2, fc=C["LoRA"], ec=C["dark"], lw=0.5, r=0.3)
    ax.text(32.2, 41.2, "low-rank\nupdate", fontsize=6.5, ha="left", va="center", color="#7A4A00", linespacing=1.0)
    rbox(10, 57.2, 12, 2.6, fc="white", ec=C["dark"], lw=0.7, r=0.3)
    ax.text(16, 58.5, "new task head", fontsize=6.5, ha="center", va="center")
    arrow(16, 56.8, 16, 57.2, lw=0.6)
    ax.text(40.5, 53.5, "billions of weights", fontsize=7.0, ha="left", va="center")
    ax.text(40.5, 49.3, "fixed supervised targets", fontsize=7.0, ha="left", va="center")
    ax.text(40.5, 45.1, "adaptation widely studied", fontsize=7.0, ha="left", va="center")
    ax.plot([68.5, 68.5], [29.5, 63], color="#BBBBBB", lw=0.6)

    # ---- top right: compact recurrent flight controller ----
    ax.text(71, 61.5, "A compact recurrent flight controller", fontsize=8.0, weight="bold", ha="left", va="center")
    for i in range(8):
        ax.add_patch(plt.Rectangle((71.5 + i * 2.3, 50.5), 2.0, 3.4, fc="#BFD7EA", ec=C["dark"], lw=0.4))
    ax.text(80.6, 48.0, "reward-free flight data", fontsize=6.6, ha="center", va="center", color="#333333")
    arrow(90.2, 52.2, 93.2, 52.2)
    end = policy(93.5, 49.2, "#CFE8F7", C["dark"])
    ax.text(end - 0.9, 46.9, "145,801 weights", fontsize=6.6, ha="right", va="center", color="#333333")
    # quadrotor icon
    cx, cy = 132.5, 52.2
    ax.plot([cx - 4, cx + 4], [cy - 2.2, cy + 2.2], color=C["dark"], lw=1.1)
    ax.plot([cx - 4, cx + 4], [cy + 2.2, cy - 2.2], color=C["dark"], lw=1.1)
    for dx, dy in ((-4, -2.2), (4, 2.2), (-4, 2.2), (4, -2.2)):
        ax.add_patch(plt.Circle((cx + dx, cy + dy), 1.5, fc="white", ec=C["dark"], lw=0.7))
    arrow(end + 0.3, 52.2, cx - 7.2, 52.2)
    ax.text((end + cx - 7.0) / 2, 54.0, "command", fontsize=6.3, ha="center", va="center", color="#333333")
    arrow(cx, cy - 4.2, 106, 44.6, conn="arc3,rad=-0.25", color=C["grey"])
    ax.text(121.5, 39.8, "on-policy RL: the policy\ncollects its own data", fontsize=6.6, ha="center", va="center", color="#444444", linespacing=1.0)
    tag(103.0, 58.4, "What to pretrain?")
    tag(89.0, 40.0, "How to adapt?")
    arrow(89.0, 41.6, 94.5, 48.9, color="#D55E00", lw=0.6)
    # learning curve with the budget tag
    xs = np.linspace(0, 1, 60)
    ax.plot(72 + 16 * xs, 31.5 + 6.5 * (1 - np.exp(-3.2 * xs)), color=C["dark"], lw=0.9)
    ax.plot([72, 72, 89], [38.8, 31.5, 31.5], color="#777777", lw=0.5)
    for k in (0.2, 0.5, 1.0):
        ax.plot([72 + 16 * k] * 2, [31.5, 38.5], color="#BBBBBB", lw=0.5, ls="--")
    tag(106.0, 33.5, "How long to train?")
    arrow(98.2, 33.5, 89.5, 35.5, color="#D55E00", lw=0.6)

    # ---- bottom strip: the PACE answer ----
    ax.add_patch(FancyBboxPatch((0.8, 0.8), 138.4, 26.2, boxstyle="round,pad=0.02,rounding_size=1.0", fc="#F7F7F7", ec="#BBBBBB", lw=0.6))
    ax.text(2.5, 24.6, "PACE: judge a pretrained backbone only against a random backbone that is adapted and trained the same way",
            fontsize=7.6, weight="bold", ha="left", va="center")
    e1 = policy(3.0, 13.2, "#CFE8F7", C["dark"], label=None)
    ax.text(3.0, 11.4, "pretrained backbone (MAE, CPC, SPR, or PIDM)", fontsize=6.6, ha="left", va="center")
    e2 = policy(3.0, 3.6, "#EFEFEF", C["grey"], ls="--")
    ax.text(3.0, 1.95, "random backbone", fontsize=6.6, ha="left", va="center", color="#555555")
    rbox(44.0, 5.2, 27.0, 14.2, fc="white", ec=C["dark"])
    ax.text(57.5, 15.4, "same adaptation mode", fontsize=7.0, ha="center", va="center", weight="bold")
    ax.text(57.5, 12.0, "FullFT, PartialFT, Frozen, LoRA", fontsize=6.5, ha="center", va="center")
    ax.text(57.5, 8.6, "same PPO, same budget", fontsize=7.0, ha="center", va="center", weight="bold")
    arrow(e1 + 0.4, 16.2, 44.0, 14.0)
    arrow(e2 + 0.4, 6.6, 44.0, 10.0, color=C["grey"])
    arrow(71.3, 12.3, 76.0, 12.3)
    # measurement panel with a small curve pair
    rbox(76.3, 3.0, 62.0, 19.0, fc="white", ec=C["dark"])
    xs = np.linspace(0, 1, 50)
    ax.plot(79 + 17 * xs, 5.5 + 12 * (1 - np.exp(-4.5 * xs)), color=C["obj"][0], lw=1.1)
    ax.plot(79 + 17 * xs, 5.5 + 11 * (1 - np.exp(-2.0 * xs)), color=C["grey"], lw=1.1, ls="--")
    ax.plot([79, 79, 96.5], [18.5, 5.5, 5.5], color="#777777", lw=0.5)
    ax.text(87.5, 20.0, "return over budgets", fontsize=6.5, ha="center", va="center")
    ax.text(100.0, 17.2, "improvement over the random counterpart", fontsize=6.8, ha="left", va="center")
    ax.text(100.0, 12.4, "16 flight metrics", fontsize=6.8, ha="left", va="center")
    ax.text(100.0, 7.6, "network state and mechanism probes", fontsize=6.8, ha="left", va="center")
    fig.savefig(OUT / "fig1_concept.pdf")
    fig.savefig(OUT / "fig1_concept.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    pass  # v6: Fig. 1 is drawn by make_v6_figs.py
    framework()
