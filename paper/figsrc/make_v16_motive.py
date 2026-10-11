"""Fig. 1 of v16: the motivation of the study at a glance.

(a) Large networks are pretrained once and adapted to many tasks, often frozen or with a low-rank update.
(b) Small controllers trained with RL start from random weights in every run.
(c) The question of this study, a reusable starting point, and the answer by adaptation mode.
(d) Evidence: reach rate in the hardest task T3 (fixed-state evaluation with the mean action), drawn from the same
    analysis table as before (stage 1, R2 and R3, seeds 7, 42, 99).
Usage: python make_v16_motive.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_v6_figs import COL, OUT, T3, W, despine, eval_runs, load

DARK, GREY, LIGHT = "#222222", "#8C8C8C", "#EFEFEF"
BLUE, GREEN, RED, ORANGE = "#0072B2", "#009E73", "#C0392B", "#E69F00"

def main():
    _, ev, _ = load()
    fig = plt.figure(figsize=(W, 2.45))
    ax = fig.add_axes([0.0, 0.0, 0.72, 1.0])
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 24.5)
    ax.axis("off")

    def box(x, y, w, h, t="", fc="white", ec=DARK, ls="-", fs=6.2, tc=DARK, bold=False, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.5", fc=fc, ec=ec, lw=lw, ls=ls))
        if t:
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc,
                    fontweight="bold" if bold else "normal", linespacing=1.05)

    def arrow(x0, y0, x1, y1, color=DARK, lw=0.7):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=7))

    def title(x, y, t):
        ax.text(x, y, t, fontsize=7.0, fontweight="bold", ha="left", va="top")

    def note(x, y, t, c=DARK, fs=6.0, st="normal"):
        ax.text(x, y, t, fontsize=fs, ha="left", va="center", color=c, style=st, linespacing=1.1)

    # (a) large networks: one pretrained backbone, frozen or low-rank, many tasks
    title(0.4, 24.2, "(a) Large networks")
    box(0.8, 15.0, 9.6, 6.4, "pretrained\nonce,\nfrozen", fc="#D9D9D9", ec="#7F7F7F", fs=6.0)
    for yy in (19.4, 16.6):
        box(10.7, yy, 1.4, 1.3, fc=ORANGE, ec=DARK, lw=0.4)
    for k, yy in enumerate((20.2, 17.6, 15.0)):
        arrow(12.4, 18.2, 15.4, yy + 0.7, lw=0.5)
        box(15.5, yy, 6.6, 1.6, f"task {k + 1}", fs=5.9)
    note(0.8, 13.3, "frozen, or with low-rank updates", c="#7A4A00", fs=5.8)
    note(0.8, 12.0, "(orange), for many tasks", c="#7A4A00", fs=5.8)

    # (b) small controllers trained with RL: random start in every run
    title(0.4, 10.4, "(b) Small RL controllers")
    for k in range(3):
        x0 = 0.8 + k * 7.6
        box(x0, 3.0, 6.9, 3.0, "random\nweights", fc=LIGHT, ec=GREY, ls="--", fs=5.8, tc="#555555")
        ax.text(x0 + 3.45, 2.1, ["task 1", "task 2", "task 3"][k], fontsize=5.8, ha="center", va="center")
    note(0.8, 8.4, "a new run from random weights for", fs=5.8)
    note(0.8, 7.2, "every task, airframe, and seed", fs=5.8)
    note(0.8, 0.8, "a hard task may never be learned", c=RED, fs=5.8)
    ax.plot([24.6, 24.6], [0.6, 24.0], color="#BBBBBB", lw=0.6)

    # (c) this study: one backbone as a reusable starting point, and what keeps it useful
    title(25.4, 24.2, "(c) This study")
    box(25.8, 18.0, 9.4, 3.4, "reward-free\nflight data", fc="#DCEAF7", fs=5.9)
    arrow(35.3, 19.7, 37.3, 19.7)
    box(37.4, 18.0, 10.6, 3.4, "pretrain once:\none backbone", fc="#DCEAF7", fs=5.9)
    note(49.0, 20.6, "a reusable starting point", st="italic", fs=6.0)
    note(49.0, 19.1, "for new tasks and airframes?", st="italic", fs=6.0)
    note(55.0, 16.6, "adapted two ways,", c="#444444", fs=5.7)
    note(55.0, 15.4, "each against a", c="#444444", fs=5.7)
    note(55.0, 14.2, "random backbone", c="#444444", fs=5.7)
    # outcome by mode
    box(25.8, 7.0, 26.8, 5.6, fc="#E8F4EC", ec=GREEN)
    ax.text(26.6, 11.4, "Fine-tune every weight", fontsize=6.3, fontweight="bold", color=GREEN, ha="left", va="center")
    note(26.6, 9.7, "a starting point that RL rewrites", fs=5.8)
    note(26.6, 8.2, "hard task learnable, easier tasks sooner", fs=5.8)
    box(25.8, 0.8, 26.8, 5.6, fc="#FBEAEA", ec=RED)
    ax.text(26.6, 5.2, "Freeze the backbone", fontsize=6.3, fontweight="bold", color=RED, ha="left", va="center")
    note(26.6, 3.5, "a fixed feature extractor", fs=5.8)
    note(26.6, 2.0, "worse than training from scratch", fs=5.8)
    arrow(42.7, 17.9, 42.7, 12.7, color=GREEN)
    ax.plot([46.4, 54.2, 54.2], [17.9, 17.9, 3.6], color=RED, lw=0.7)
    arrow(54.2, 3.6, 52.7, 3.6, color=RED)
    # (d) evidence: T3 reach rate over training
    ax2 = fig.add_axes([0.80, 0.17, 0.18, 0.66])
    e = eval_runs(ev, 1, T3, [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 575, 600])
    for (i, m, lab, c) in (("random", "FullFT", "scratch", COL["scratch"]), ("MAE", "FullFT", "MAE", COL["MAE"]),
                           ("PIDM", "FullFT", "AP", COL["PIDM"])):
        g = e[(e.init == i) & (e["mode"] == m)]
        p = g.pivot_table(index="iter", columns=["task", "seed"], values="reached_05")
        mean = p.mean(1)
        ax2.plot(mean.index, 100 * mean.values, color=c, label=lab, lw=1.2)
        ax2.fill_between(mean.index, 100 * p.min(1).values, 100 * p.max(1).values, color=c, alpha=0.12, lw=0)
    ax2.set_xlabel("PPO iteration", fontsize=6.4)
    ax2.set_ylabel("T3 reach rate (%)", fontsize=6.4)
    ax2.tick_params(labelsize=6.0)
    ax2.set_title("(d) Hardest task T3", loc="left", fontweight="bold", fontsize=6.6, x=-0.36)
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=False, loc="upper left", fontsize=6.0, handlelength=1.4)
    despine(ax2)
    fig.savefig(OUT / "fig1_concept.pdf")
    fig.savefig(OUT / "fig1_concept.png", dpi=200)
    plt.close(fig)
    print("fig1 (motivation) written")

if __name__ == "__main__":
    main()
