"""Figure added in v12 (pre-registration v2, section 24, exploratory): iterations to reach the final level of training
from scratch, from time_to_level.csv and time_to_level_screening.csv (scripts/study2_time_to_level.py).

(a) Confirmation stage, T1 and T2 on R1-R3, ten seeds: median iterations to reach q times the final (541-600)
    return of training from scratch, for scratch, MAE + FullFT, and AP + FullFT, with seed-bootstrap 95% intervals.
(b) Screening stage, T1 and T2 on R2 and R3, three seeds: share of runs that reach the final level of training from
    scratch within 600 iterations, per adaptation mode, for MAE, AP, and the random counterpart.
Usage: python make_v12_speed_fig.py
"""
import numpy as np
import pandas as pd

import make_v6_figs as M
from make_v6_figs import A, COL, OUT, W, despine, plt

QS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MODES = [("FullFT", "FullFT"), ("AuxSSL", "Aux-SSL"), ("BackboneLoRA", "LoRA"), ("LPFT", "LP-FT"),
         ("LateFT", "PartialFT"), ("DiscLR", "Small-LR"), ("L2SP", "L2-SP"), ("EncFrozen", "Frozen")]
rng = np.random.default_rng(12)

def med_ci(x, seeds, col):
    pt = np.median(x[col])
    bs = [np.median(pd.concat([x[x.seed == s] for s in rng.choice(seeds, len(seeds))])[col]) for _ in range(2000)]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def main():
    d = pd.read_csv(A / "time_to_level.csv")
    d1 = pd.read_csv(A / "time_to_level_screening.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.32, width_ratios=[1.0, 1.25]))
    ax = axs[0]
    seeds = sorted(d.seed.unique())
    sc = d[d.init == "MAE"]  # scratch iterations are the same rows for every backbone
    for lab, col, src, key in (("training from scratch", COL["scratch"], sc, "s"), ("MAE + FullFT", COL["MAE"], d[d.init == "MAE"], "p"),
                               ("AP + FullFT", COL["PIDM"], d[d.init == "PIDM"], "p")):
        ys, lo, hi = [], [], []
        for q in QS:
            x = src[src[f"{key}{q}"].notna()]
            pt, l, h = med_ci(x, seeds, f"{key}{q}")
            ys.append(pt); lo.append(l); hi.append(h)
        xq = np.array(QS) * 100
        ax.plot(xq, ys, color=col, lw=1.4, marker="o", ms=3, label=lab)
        ax.fill_between(xq, lo, hi, color=col, alpha=0.18, lw=0)
    ax.set_xlabel("level reached (% of the final return\nof training from scratch)")
    ax.set_ylabel("iterations to reach the level")
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax.set_title("(a) Confirmation, T1 and T2", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    ax = axs[1]
    x = np.arange(len(MODES))
    for j, (init, lab) in enumerate((("random", "random"), ("MAE", "MAE"), ("PIDM", "AP"))):
        vals = [d1[(d1.init == init) & (d1["mode"] == m)]["p1.0"].notna().mean() if m != "FullFT" or init != "random" else 1.0
                for m, _ in MODES]
        ax.bar(x + (j - 1) * 0.26, vals, 0.24, color=COL[init], label=lab, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in MODES], rotation=35, ha="right", fontsize=6.0)
    ax.set_ylabel("share of runs that reach\nthe final level of scratch")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.0, ncol=3, loc="upper right", handlelength=1.2, columnspacing=0.8)
    ax.set_title("(b) Screening, by adaptation mode", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.30)
    fig.savefig(OUT / "fig16_speed.pdf")
    fig.savefig(OUT / "fig16_speed.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
