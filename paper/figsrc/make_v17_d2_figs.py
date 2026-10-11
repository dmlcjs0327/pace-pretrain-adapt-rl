"""Fig. 14 after the redesign of the second domain (pre-registration v2, sections 29 and 30; stages 40 and 41).

(a) Gain over training from scratch at 600 iterations (training return, units of the scratch return of the same
    setting, seed bootstrap over five seeds) for AP and MAE with FullFT and Frozen, in the four quadruped settings:
    flat ground with and without the base velocity, and rough terrain without the height scan, with and without the
    terrain curriculum (v17_v18_d2.json).
(b, c) Training return against iterations on flat ground (with the base velocity) and on rough terrain with the
    terrain curriculum, mean over five seeds, for training from scratch, AP and MAE with FullFT, and AP with Frozen.
Usage: python make_v17_d2_figs.py
"""
import json

import numpy as np
import pandas as pd

from make_v6_figs import A, COL, OUT, W, despine, plt

SETTINGS = [("ige_terrain_flat", "flat", 40), ("ige_terrain_flat_blind", "flat, no velocity", 40),
            ("ige_terrain_rough_curr", "rough, curriculum", 41), ("ige_terrain_rough", "rough", 41)]
CONDS = [("PIDM", "FullFT", "AP, FullFT"), ("MAE", "FullFT", "MAE, FullFT"), ("PIDM", "EncFrozen", "AP, Frozen"),
         ("MAE", "EncFrozen", "MAE, Frozen")]
SHADES = [1.0, 0.75, 0.5, 0.3]

def main():
    r = json.loads((A / "v17_v18_d2.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw=dict(width_ratios=[1.5, 1.0, 1.0], wspace=0.45))
    ax = axs[0]
    x = np.arange(len(CONDS))
    for j, (task, lab, st) in enumerate(SETTINGS):
        part = r["stage40" if st == 40 else "stage41"]
        off = (j - 1.5) * 0.2
        for i, (obj, mode, _) in enumerate(CONDS):
            v = part.get(f"{task}|{obj}|{mode} vs scratch|K600")
            if v is None:
                continue
            col = COL["PIDM"] if obj == "PIDM" else COL["MAE"]
            ax.bar(x[i] + off, v[0], 0.19, color=col, alpha=SHADES[j], edgecolor="white", lw=0.3,
                   label=lab if i == 0 else None)
            ax.errorbar(x[i] + off, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor="#333333",
                        elinewidth=0.5, capsize=1.0)
    ax.axhline(0, color="#666666", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in CONDS], fontsize=6.0, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("gain over scratch, 600 it.")
    ax.set_title("(a) Final gain", loc="left", fontweight="bold", fontsize=7.0)
    leg = ax.legend(frameon=False, fontsize=5.2, loc="upper right", title="setting (dark to light)", title_fontsize=5.2)
    for h in leg.legend_handles:
        h.set_color("#555555")
    despine(ax)
    s = pd.read_csv(A / "series.csv", usecols=["stage", "task", "init", "mode", "seed", "iter", "ret"])
    lines = [("random", "FullFT", "scratch", COL["scratch"], "-"), ("PIDM", "FullFT", "AP, FullFT", COL["PIDM"], "-"),
             ("MAE", "FullFT", "MAE, FullFT", COL["MAE"], "-"), ("PIDM", "EncFrozen", "AP, Frozen", COL["PIDM"], ":")]
    for ax, (task, title, st) in ((axs[1], ("ige_terrain_flat", "(b) Flat ground", 40)),
                                  (axs[2], ("ige_terrain_rough_curr", "(c) Rough, curriculum", 41))):
        d = s[(s.stage == st) & (s.task == task)]
        for init, mode, lab, c, ls in lines:
            g = d[(d.init == init) & (d["mode"] == mode)]
            if g.empty:
                continue
            cur = g.groupby(["seed", "iter"]).ret.mean().groupby("iter").mean().rolling(10, min_periods=1).mean()
            ax.plot(cur.index, cur.values, color=c, ls=ls, lw=1.0, label=lab)
        ax.set_xlabel("RL iteration")
        ax.set_ylabel("training return")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.0)
        despine(ax)
    axs[2].legend(frameon=False, fontsize=5.4, loc="lower right")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.27)
    fig.savefig(OUT / "fig14_domain2.pdf")
    fig.savefig(OUT / "fig14_domain2.png", dpi=200)
    plt.close(fig)
    print("fig14 redrawn from stages 40 and 41")

if __name__ == "__main__":
    main()
