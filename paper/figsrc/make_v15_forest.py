"""Forest plot of the pre-registered confirmation decisions.

Reads the values of Table 'tab:decisions' from figures/tab_decisions_v13.tex (written by make_v13_tables.py), so the
figure and the table always show the same numbers. Each row is one comparison, each panel one set of settings, and
each bar a Student-t 95% interval over ten seeds. The FullFT rows of D4 equal the FullFT rows of D1 and D2 and are
drawn once.
Usage: python make_v15_forest.py
"""
import re

from make_v6_figs import COL, OUT, W, despine, plt

ROWS = open(OUT / "tab_decisions_v13.tex").read().strip().splitlines()
GCI = re.compile(r"\\gci\{([+-]?[\d.]+)\}\{(\$-\$)?([\d.]+)\}\{(\$-\$)?([\d.]+)\}")

def parse():
    out = []
    for line in ROWS:
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        name = cells[1].replace("vs.\\ ", "vs. ").replace("\\ ", " ")
        vals = []
        for c in [cells[2]] + cells[4:7]:
            m = GCI.search(c)
            lo = -float(m.group(3)) if m.group(2) else float(m.group(3))
            hi = -float(m.group(5)) if m.group(4) else float(m.group(5))
            vals.append((float(m.group(1)), lo, hi))
        out.append((name, vals))
    return out

rows = [r for r in parse() if r[0] not in ("MAE + FullFT vs. scratch", "AP + FullFT vs. scratch")]
labels = {"MAE vs. random, FullFT": "MAE vs. random, FullFT",
          "MAE vs. random, Aux-SSL": "MAE vs. random, Aux-SSL",
          "AP vs. random, FullFT": "AP vs. random, FullFT",
          "AP vs. random, Aux-SSL": "AP vs. random, Aux-SSL",
          "Aux-SSL vs. FullFT, MAE": "Aux-SSL vs. FullFT, MAE",
          "Aux-SSL vs. FullFT, AP": "Aux-SSL vs. FullFT, AP",
          "MAE + Aux-SSL vs. scratch": "MAE + Aux-SSL vs. scratch",
          "AP + Aux-SSL vs. scratch": "AP + Aux-SSL vs. scratch"}
titles = ["(a) Nine settings", "(b) T1 regulation", "(c) T2 recovery", "(d) T3 aggressive"]
fig, axs = plt.subplots(1, 4, figsize=(W, 2.05), sharey=True, gridspec_kw=dict(wspace=0.18))
y = list(range(len(rows)))[::-1]
for p, ax in enumerate(axs):
    for yi, (name, vals) in zip(y, rows):
        g, lo, hi = vals[p]
        col = COL["MAE"] if name.startswith(("MAE", "Aux-SSL vs. FullFT, MAE")) else COL["PIDM"]
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
        ax.plot([g], [yi], "o", color=col, ms=3.6, mec="white", mew=0.6)
    ax.axvline(0, color="#666666", lw=0.6)
    ax.set_title(titles[p], loc="left", fontweight="bold", fontsize=7.0)
    ax.set_xlabel("gain (scratch units)")
    ax.grid(axis="x", color="#E5E5E5", lw=0.5)
    ax.set_axisbelow(True)
    despine(ax)
    ax.spines["left"].set_visible(p == 0)
axs[0].set_yticks(y)
axs[0].set_yticklabels([labels.get(r[0], r[0]) for r in rows], fontsize=6.4)
for ax in axs[1:]:
    ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=0.25, right=0.99, top=0.88, bottom=0.2)
fig.savefig(OUT / "fig17_forest.pdf")
fig.savefig(OUT / "fig17_forest.png", dpi=200)
print("fig17 written from", len(rows), "comparisons")
