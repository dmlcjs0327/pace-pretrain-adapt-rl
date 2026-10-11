"""Tables of v13 from the re-analysis of the reviews (pre-registration v2, section 25, analysis only).

tab_grid_v13.tex      original screening grid (MAE, CPC, SPR, AP x FullFT, PartialFT, Frozen, LoRA), gain at 600 iterations
                      against training from scratch and against the random counterpart, mean over six settings with the
                      Student-t 95% interval over the three seeds (two degrees of freedom).
tab_decisions_v13.tex confirmation decisions D1-D4, nine settings and per task family, ten seeds, t intervals, Holm p.
Uses study2_review_stats.contrast on budgets.csv. Usage: python make_v13_tables.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
import json
import sys
from pathlib import Path

sys.path.insert(0, _EXP_ROOT + "/scripts")
import study2_review_stats as R  # noqa: E402

OUT = str(Path(__file__).resolve().parent.parent / "figures") + "/"  # figures of this manuscript folder
NAME = {"MAE": "MAE", "CPC": "CPC", "SPR": "SPR", "PIDM": "AP"}
MODES = [("FullFT", "FullFT"), ("LateFT", "PartialFT"), ("EncFrozen", "Frozen"), ("BackboneLoRA", "LoRA")]

def g(r):
    m, lo, hi = r["mean"], r["t95"][0], r["t95"][1]
    f = lambda v: ("0.00" if abs(v) < 0.005 else f"{v:.2f}").replace("-", "$-$")
    return f"\\gci{{{m:+.2f}}}{{{f(lo)}}}{{{f(hi)}}}"

def main():
    b = R.load_budgets()
    T6 = sorted(b[(b.stage == 1) & (b.K == 600)].task.unique())
    rows = []
    for o in ("MAE", "CPC", "SPR", "PIDM"):
        cells = [g(R.contrast(b, 1, 600, R.S3, T6, (o, m), ("random", "FullFT"))) for m, _ in MODES]
        rows.append(f"{NAME[o]} & " + " & ".join(cells) + " \\\\")
    open(OUT + "tab_grid_scratch_v13.tex", "w").write("\n".join(rows) + "\n")
    rows = []
    for o in ("MAE", "CPC", "SPR", "PIDM"):
        cells = [g(R.contrast(b, 1, 600, R.S3, T6, (o, m), ("random", m))) for m, _ in MODES]
        rows.append(f"{NAME[o]} & " + " & ".join(cells) + " \\\\")
    open(OUT + "tab_grid_random_v13.tex", "w").write("\n".join(rows) + "\n")
    cells = ["0"] + [g(R.contrast(b, 1, 600, R.S3, T6, ("random", m), ("random", "FullFT"))) for m, _ in MODES[1:]]
    rows = ["Random vs.\\ scratch & " + " & ".join(cells) + " \\\\"]
    open(OUT + "tab_grid_last_v13.tex", "w").write("\n".join(rows) + "\n")

    st = json.load(open(R.A_DIR / "review_stats.json"))["A"]
    lab = [("D1|FullFT", "D1", "MAE vs.\\ random, FullFT"), ("D1|AuxSSL", "", "MAE vs.\\ random, Aux-SSL"),
           ("D2|FullFT", "D2", "AP vs.\\ random, FullFT"), ("D2|AuxSSL", "", "AP vs.\\ random, Aux-SSL"),
           ("D3|MAE", "D3", "Aux-SSL vs.\\ FullFT, MAE"), ("D3|PIDM", "", "Aux-SSL vs.\\ FullFT, AP"),
           ("D4|MAE|FullFT", "D4", "MAE + FullFT vs.\\ scratch"), ("D4|MAE|AuxSSL", "", "MAE + Aux-SSL vs.\\ scratch"),
           ("D4|PIDM|FullFT", "", "AP + FullFT vs.\\ scratch"), ("D4|PIDM|AuxSSL", "", "AP + Aux-SSL vs.\\ scratch")]
    rows = []
    for k, d, txt in lab:
        a = st["K600|10 seeds|9 settings"][k]
        ph = a["p_holm"]
        pstr = "$<$0.001" if ph < 0.001 else f"{ph:.3f}"
        fam = [g(st[f"K600|10 seeds|{f}"][k]) for f in ("T1", "T2", "T3")]
        rows.append(f"{d} & {txt} & {g(a)} & {pstr} & " + " & ".join(fam) + " \\\\")
    open(OUT + "tab_decisions_v13.tex", "w").write("\n".join(rows) + "\n")
    # robustness: fixed-state evaluation (ten and seven new seeds), training return without the floor setting
    T9 = sorted(b[(b.stage == 2) & (b.K == 600)].task.unique())
    T8 = [t for t in T9 if t != "agg_baseQuad"]
    rows = []
    for k, d, txt in lab:
        parts = k.split("|")
        if parts[0] in ("D1", "D2"):
            o = "MAE" if parts[0] == "D1" else "PIDM"
            A_, B_ = (o, parts[1]), ("random", parts[1])
        elif parts[0] == "D3":
            A_, B_ = (parts[1], "AuxSSL"), (parts[1], "FullFT")
        else:
            A_, B_ = (parts[1], parts[2]), ("random", "FullFT")
        cells = [R.contrast(b, 2, 600, R.S10, T9, A_, B_, col="eterm"),
                 R.contrast(b, 2, 600, R.NEW7, T9, A_, B_, col="eterm"),
                 R.contrast(b, 2, 600, R.S10, T8, A_, B_),
                 R.contrast(b, 2, 600, R.NEW7, T9, A_, B_)]
        rows.append(f"{d} & {txt} & " + " & ".join(g(c) for c in cells) + " \\\\")
    open(OUT + "tab_decisions_robust_v13.tex", "w").write("\n".join(rows) + "\n")
    # screening: every mode against training from scratch, six settings, t intervals and seed range (appendix)
    MODES8 = [("FullFT", "FullFT"), ("LateFT", "PartialFT"), ("EncFrozen", "Frozen"), ("BackboneLoRA", "LoRA"),
              ("LPFT", "LP-FT"), ("DiscLR", "Small-LR"), ("L2SP", "L2-SP"), ("AuxSSL", "Aux-SSL")]
    rows = []
    for m, lab in MODES8:
        cells = []
        for o in ("random", "MAE", "PIDM"):
            if (o, m) == ("random", "FullFT"):
                cells.append("0")
                continue
            r = R.contrast(b, 1, 600, R.S3, T6, (o, m), ("random", "FullFT"))
            f = lambda v: ("0.00" if abs(v) < 0.005 else f"{v:.2f}").replace("-", "$-$")
            cells.append(g(r) + f" \\newline {{\\scriptsize({f(r['range'][0])} to {f(r['range'][1])})}}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open(OUT + "tab_screen_modes_v13.tex", "w").write("\n".join(rows) + "\n")
    print(open(OUT + "tab_grid_scratch_v13.tex").read())
    print(open(OUT + "tab_decisions_v13.tex").read())
    print(open(OUT + "tab_decisions_robust_v13.tex").read())

if __name__ == "__main__":
    main()
