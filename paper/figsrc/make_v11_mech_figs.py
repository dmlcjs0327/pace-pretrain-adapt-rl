"""Mechanism figure added in v11 (pre-registration v2, section 21).

Fig. 15  (a) Frozen relative to FullFT (final fixed-state evaluation, T1 and T2) under RL (stage 1) and under supervised
             adaptation (S19 students in closed loop), per backbone.
         (b) Gain over training from scratch at 600 iterations of MAE with FullFT and Frozen, with the compact
             13-dimensional state (stage 1) and with the expanded 128-dimensional input (S20, stage 16).
         (c) Position P (Eq. position) of weight-shuffled (stage 5), target-swapped (stage 8), and time-shuffled
             (S21, stage 17) MAE, early (fixed-state evaluation at 100 iterations, T1 and T2) and final (training
             return at 600 iterations, T3), as in study2_s10.py.
         (d) Alignment (CKA) of the initial features with the teachers' values against the early gain of FullFT
             (S22, s22_align.csv).
Values that are not yet measured are drawn in red with dashed outlines (expected values) and are replaced
automatically when the runs exist; the console output says which parts are measured.
Usage: python make_v11_mech_figs.py
"""
import json

import numpy as np
import pandas as pd

from make_v6_figs import A, COL, OUT, RED, W, despine, plt

T12 = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
T3 = ["agg_x500", "agg_lmf2"]
SEEDS = [7, 42, 99]
B = 2000
rng = np.random.default_rng(15)
EXPECT = {"sup_ratio": {"random": 0.84, "MAE": 0.91, "PIDM": 0.93},
          "expand": {"FullFT": 0.60, "EncFrozen": -0.15},
          "tshuf": {"early": 0.62, "final": 0.30}}
STATUS = []

def boot(fn):
    """v13 (review of v12.3): the value over the three seeds and the range of its single-seed values (no
    three-seed percentile bootstrap)."""
    pt = fn(SEEDS)
    singles = [fn([sd]) for sd in SEEDS]
    return pt, np.nanmin(singles), np.nanmax(singles)

def ev_table():
    ev = pd.read_csv(A / "eval_fixed.csv").drop_duplicates(["run_id", "iter", "disturb"], keep="last")
    return ev[(ev.disturb == 0) & ev.seed.isin(SEEDS)]

def ratio_rl(ev):
    e = ev[(ev.stage == 1) & ev.task.isin(T12) & (ev.iter >= 550)]
    g = e.groupby(["init", "mode", "task", "seed"]).ret.mean()
    out = {}
    for init in ("random", "MAE", "PIDM"):
        def f(ss, init=init):
            return np.mean([np.mean([g[(init, "EncFrozen", t, s)] / g[(init, "FullFT", t, s)] for s in ss]) for t in T12])
        out[init] = boot(f)
    return out

def ratio_rl_v8(ev, variant):
    """V8 (stage 31): Frozen with whitened backbone output or with the observation at the heads, over FullFT of the
    same backbone (stage 1), final fixed-state evaluation in T1 and T2."""
    e1 = ev[(ev.stage == 1) & ev.task.isin(T12) & (ev.iter >= 550)]
    e31 = ev[(ev.stage == 31) & ev.task.isin(T12) & (ev.iter >= 550)]
    g1 = e1.groupby(["init", "mode", "task", "seed"]).ret.mean()
    g31 = e31.groupby(["init", "task", "seed"]).ret.mean()
    out = {}
    for init in ("random", "MAE", "PIDM"):
        try:
            def f(ss, init=init):
                return np.mean([np.mean([g31[(f"{init}_{variant}", t, s)] / g1[(init, "FullFT", t, s)] for s in ss]) for t in T12])
            out[init] = boot(f)
        except KeyError:
            return None
    return out

def ratio_sup():
    p = A / "eval_s19_students.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p)
    e["task"] = e.run_id.str.split("__").str[1]
    e["init"] = e.run_id.str.split("__").str[2]
    e["mode"] = e.run_id.str.split("__").str[3]
    e["seed"] = e.run_id.str.split("__").str[4].str.replace("seed", "").astype(int)
    g = e.set_index(["init", "mode", "task", "seed"]).ret
    out = {}
    for init in ("random", "MAE", "PIDM"):
        try:
            def f(ss, init=init):
                return np.mean([np.mean([g[(init, "Frozen", t, s)] / g[(init, "FullFT", t, s)] for s in ss]) for t in T12])
            out[init] = boot(f)
        except KeyError:
            return None
    return out

def gains_600(stage, frozen_init, scratch_init, tasks):
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.stage == stage) & (b.K == 600) & b.seed.isin(SEEDS) & b.task.isin(tasks)]
    if stage == 16:
        b = b[b.run_id.str.contains("x__")]
    tab = b.pivot_table(index=["task", "seed"], columns=["init", "mode"], values="ret")
    if (scratch_init, "FullFT") not in tab.columns:
        return None
    den = tab[(scratch_init, "FullFT")].groupby(level="task").mean()
    out = {}
    for mode in ("FullFT", "EncFrozen"):
        if (frozen_init, mode) not in tab.columns or tab[(frozen_init, mode)].isna().any():
            return None

        def f(ss, mode=mode):
            return np.mean([(tab.loc[t].reindex(ss)[(frozen_init, mode)].mean() -
                             tab.loc[t].reindex(ss)[(scratch_init, "FullFT")].mean()) / den[t] for t in tasks])
        out[mode] = boot(f)
    return out

def positions(ev, alt, stage):
    """P = G(alt, random) / G(MAE, random), early on the 100-iteration evaluation in T1-T2, final on the training
    return at 600 iterations in T3 (FullFT), as study2_s10.py."""
    e = ev[(ev["mode"] == "FullFT") & (((ev.stage == 1) & ev.init.isin(["random", "MAE"])) | ((ev.stage == stage) & (ev.init == alt)))]
    if alt not in set(e.init):
        return None
    e100 = e[e.iter == 100].pivot_table(index=["task", "seed"], columns="init", values="ret")
    eterm = e[e.iter >= 550].groupby(["task", "init", "seed"]).ret.mean().unstack("init")
    den_e = eterm["random"].groupby("task").mean()
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.K == 600) & b.seed.isin(SEEDS) & (b["mode"] == "FullFT") & b.task.isin(T3)]
    b = b[((b.stage == 1) & b.init.isin(["random", "MAE"])) | ((b.stage == stage) & (b.init == alt))]
    tr = b.pivot_table(index=["task", "seed"], columns="init", values="ret")
    den_t = tr["random"].groupby("task").mean()

    def G(tab, tasks, den, x, ss):
        return np.mean([(tab.loc[t].reindex(ss)[x].mean() - tab.loc[t].reindex(ss)["random"].mean()) / den[t] for t in tasks])
    out = {}
    for name, tab, tasks, den in (("early", e100, T12, den_e), ("final", tr, T3, den_t)):
        if alt not in tab.columns or tab[alt].reindex(pd.MultiIndex.from_product([tasks, SEEDS])).isna().any():
            return None
        out[name] = boot(lambda ss: G(tab, tasks, den, alt, ss) / G(tab, tasks, den, "MAE", ss))
    return out

def main():
    ev = ev_table()
    fig, axs = plt.subplots(1, 4, figsize=(W, 2.1), gridspec_kw=dict(wspace=0.75, width_ratios=[1.05, 1.0, 1.15, 1.0]))
    # (a) Frozen relative to FullFT: RL, RL with whitened features, RL with the observation at the heads (V8), and
    # supervised students
    ax = axs[0]
    rl, sup = ratio_rl(ev), ratio_sup()
    wh, sk = ratio_rl_v8(ev, "whiten"), ratio_rl_v8(ev, "skip")
    names = [("random", "random"), ("MAE", "MAE"), ("PIDM", "AP")]
    x = np.arange(3)
    bars = (("RL", rl, "#8C8C8C"), ("RL, whitened", wh, COL["MAE"]), ("RL, obs. to heads", sk, "#D8B365"),
            ("supervised", sup, "#5AAE61"))
    wd = 0.2
    for j, (lab, d, c) in enumerate(bars):
        if d is None:
            continue
        for i, (k, _) in enumerate(names):
            v = d[k]
            xi = i + (j - 1.5) * wd
            ax.bar(xi, v[0], wd * 0.92, color=c, label=lab if i == 0 else None)
            ax.errorbar(xi, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color="k", lw=0.6, capsize=1.0)
    STATUS.append("fig15a RL, V8 " + ("measured" if wh and sk else "MISSING") + ", supervised " + ("measured" if sup else "MISSING"))
    ax.axhline(1, color="#999999", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n[1] for n in names], fontsize=6.0, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Frozen / FullFT")
    ax.set_title("(a) Freezing", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.0, loc="upper left", ncol=2, handlelength=0.8, columnspacing=0.6)
    despine(ax)
    # (b) compact vs expanded input
    ax = axs[1]
    comp = gains_600(1, "MAE", "random", T3 + T12)
    expd = gains_600(16, "MAE", "random", T3 + T12)
    for j, (mode, lab) in enumerate((("FullFT", "FullFT"), ("EncFrozen", "Frozen"))):
        v = comp[mode]
        ax.bar(j - 0.18, v[0], 0.34, color="#8C8C8C", label="13 inputs" if j == 0 else None)
        ax.errorbar(j - 0.18, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color="k", lw=0.7, capsize=1.2)
        if expd:
            e2 = expd[mode]
            ax.bar(j + 0.18, e2[0], 0.34, color=COL["MAE"], label="128 inputs" if j == 0 else None)
            ax.errorbar(j + 0.18, e2[0], yerr=[[e2[0] - e2[1]], [e2[2] - e2[0]]], color="k", lw=0.7, capsize=1.2)
        else:
            ax.bar(j + 0.18, EXPECT["expand"][mode], 0.34, facecolor="white", edgecolor=RED, ls="--", lw=0.9,
                   label="128 inputs" if j == 0 else None)
    STATUS.append("fig15b compact measured; expanded " + ("measured" if expd else "PLACEHOLDER"))
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FullFT", "Frozen"])
    ax.set_ylabel("gain, 600 it.")
    ax.set_ylim(-0.55, 1.45)
    ax.set_title("(b) Input", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=1)
    despine(ax)
    # (c) positions
    ax = axs[2]
    ctr = [("MAEshuf", 5, "weights\nshuffled"), ("MAEtswap", 8, "target\nswapped"), ("MAEtshuf", 17, "time\nshuffled")]
    for i, (alt, st, lab) in enumerate(ctr):
        p = positions(ev, alt, st)
        for dx, key, col in ((-0.18, "early", COL["MAE"]), (0.18, "final", "#0072B2")):
            if p:
                v = p[key]
                ax.bar(i + dx, v[0], 0.34, color=col, label=(key if i == 0 else None))
                ax.errorbar(i + dx, v[0], yerr=[[max(v[0] - v[1], 0)], [max(v[2] - v[0], 0)]], color="k", lw=0.7, capsize=1.2)
            else:
                ax.bar(i + dx, EXPECT["tshuf"][key], 0.34, facecolor="white", edgecolor=RED, ls="--", lw=0.9,
                       label=(key if i == 0 else None))
        STATUS.append(f"fig15c {alt} " + ("measured " + str({k: round(v[0], 2) for k, v in p.items()}) if p else "PLACEHOLDER"))
    ax.axhline(1, color="#999999", lw=0.6, ls=":")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[2].replace("\n", " ") for c in ctr], fontsize=5.8, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("retained share $P$")
    ax.set_ylim(-0.2, 1.6)
    ax.set_title("(c) What transfers", loc="left", fontweight="bold", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    despine(ax)
    # (d) v13: held-out linear R^2 of the teacher's value from the frozen features against the gain of freezing over
    # training from scratch, width 1, nine backbones x three seeds (study2_rep_review.py)
    ax = axs[3]
    d = pd.read_csv(A / "rep_review.csv").dropna(subset=["g600_frozen"])
    grp = [("pretrained", ~d.init.isin(["random", "BC"]), COL["MAE"], "o"), ("random", d.init == "random", "#8C8C8C", "s"),
           ("BC", d.init == "BC", "#D55E00", "^")]
    for lab, m, col, mk in grp:
        ax.scatter(d[m].value_r2_ridge_out, d[m].g600_frozen, s=8, color=col, marker=mk, label=lab)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 0.88), handletextpad=0.1, borderaxespad=0.1)
    js = json.loads((A / "rep_review.json").read_text())
    r = js["value_r2_ridge_out|g600_frozen"]
    ax.text(0.03, 0.95, f"$\\rho$ = {r['rho']:.2f}", transform=ax.transAxes, fontsize=5.8, va="top")
    STATUS.append("fig15d measured (rep_review)")
    ax.set_xlabel("frozen value $R^2$")
    ax.set_ylabel("Frozen gain, 600 it.")
    ax.set_title("(d) Value in features", loc="left", fontweight="bold", fontsize=7.0)
    despine(ax)
    fig.subplots_adjust(left=0.095, right=0.955, top=0.85, bottom=0.32)
    fig.savefig(OUT / "fig15_mech.pdf")
    fig.savefig(OUT / "fig15_mech.png", dpi=200)
    plt.close(fig)
    for s in STATUS:
        print(s)

if __name__ == "__main__":
    main()
