"""Representation analyses requested by the reviews of v12.3 (pre-registration v2, section 25, analysis only).

For every backbone at width 1 (random under seeds 7, 42, 99; the eight objectives; shuffled MAE and AP), features are
the GRU output at the last step of 33-step windows on two state distributions:
  teacher  the test episode of the S19 teacher data of the four T1/T2 settings (4,096 windows each, generator seed 5,
           as study2_s22_align and study2_width_features), with the teacher's value and mean action as targets;
  flight   4,096 windows of the v3 flight data of R2 and R3 (every 4th fixed-seed window of pace_regularizers), the
           distribution of the property table of the appendix.
Reported per backbone (mean over settings): participation ratio PR = (sum l)^2 / sum l^2, entropy effective rank
exp(H(l / sum l)) (Roy and Vetterli, 2007), k90; for the value and the action, the energy fractions e_i of the centered
target along the eigenvectors of the feature covariance (Proposition 1), the share outside the span (1 - sum e_i), the
weak-direction term W(a) = sum e_i (a / (l_i + a))^2 for a = c * mean(l), c in {1e-3, 1e-2, 1e-1, 1}, in-sample OLS
R^2, and held-out R^2 (70/30 split) of OLS and of ridge (RidgeCV). These are joined with the freezing cost of the
screening stage (stage 1, T1/T2 settings): Frozen minus scratch in scratch units (g600_frozen) and the ratio of the
terminal returns Frozen / FullFT of the same backbone. Spearman correlations over the 27 backbone-seed units, with a
cluster bootstrap that resamples backbones (nine clusters of three seeds).
Output: logs/study2/analysis/rep_review.csv and rep_review.json.
Usage: python study2_rep_review.py
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401  (must precede torch)
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV

import study2_drift as D
from study2_s22_align import CK, OBJ, TASKS, test_windows
from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone

A = D.L / "analysis"
STATS = _DATA_ROOT + "/v3/state_stats.pt"
CS = (1e-3, 1e-2, 1e-1, 1.0)

def spectrum(F):
    F = F - F.mean(0)
    C = (F.T @ F / F.shape[0]).double()
    lam, U = torch.linalg.eigh(C)
    lam, U = lam.clamp(min=0).flip(0), U.flip(1)
    return lam, U, F.double()

def geometry(lam):
    p = lam / lam.sum()
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    er = float(torch.exp(-(p[p > 0] * torch.log(p[p > 0])).sum()))
    k90 = int((torch.cumsum(p, 0) < 0.9).sum()) + 1
    return pr, er, k90

def energy(lam, U, F, y):
    """Energy fractions of the centered target y (n,) along the eigenvectors (columns of U) of the feature covariance."""
    yc = y - y.mean()
    # projections of the target on the whitened principal directions of the data: coefficient of y on F u_i
    Z = F @ U  # (n, k) principal scores, variance lam_i
    keep = lam > 1e-10 * lam.max()
    z = Z[:, keep] / torch.sqrt(lam[keep] * F.shape[0])
    e = (z.T @ yc) ** 2 / (yc @ yc)
    return lam[keep], e

def held_out(F, y, rng):
    n = F.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    out = {}
    for name, m in (("ols", LinearRegression()), ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 15)))):
        m.fit(F[tr], y[tr])
        r = 1 - ((m.predict(F[te]) - y[te]) ** 2).sum(0) / ((y[te] - y[te].mean(0)) ** 2).sum(0)
        out[name] = float(np.mean(r))
    return out

def backbones():
    out = []
    for o in OBJ + ["MAEshuf", "PIDMshuf"]:
        for s in (7, 42, 99):
            out.append((o, s, D.ssl_backbone(f"{CK}/P_{o}__seed{s}__srcmixed__backbone.pt")))
    for s in (7, 42, 99):
        torch.manual_seed(s)
        bb = ProprioGRUBackbone(proprio_dim=13, proprio_hidden=64, fusion_hidden=128, gru_hidden=128)
        out.append(("random", s, D.effective_backbone(bb.state_dict(), prefix="")))
    return out

def freezing_cost():
    b = pd.read_csv(A / "budgets.csv")
    b = b[(b.stage == 1) & (b.K == 600) & b.task.isin(TASKS) & b.seed.isin([7, 42, 99])]
    piv = b.pivot_table(index=["task", "seed"], columns=["init", "mode"], values="ret")
    den = piv[("random", "FullFT")].groupby(level="task").mean()
    rows = {}
    for init in OBJ + ["random"]:
        for s in (7, 42, 99):
            g, r = [], []
            for t in TASKS:
                fz, ff, sc = piv.loc[(t, s), (init, "EncFrozen")], piv.loc[(t, s), (init, "FullFT")], piv.loc[(t, s), ("random", "FullFT")]
                g.append((fz - sc) / den[t])
                r.append(fz / ff)
            rows[(init, s)] = dict(g600_frozen=float(np.mean(g)), frozen_over_fullft=float(np.mean(r)))
    return rows

def main():
    from aerial_gym.rl_training.rl_games.pace_regularizers import load_windows
    rng = np.random.default_rng(5)
    teach = {t: test_windows(t, rng) for t in TASKS}
    xs, _, _ = load_windows(torch.load(STATS, map_location="cpu"))
    flight = xs[::4].float().to(D.dev)
    rows = []
    rr = np.random.default_rng(70)
    with torch.no_grad():
        for init, s, W in backbones():
            r = dict(init=init, seed=s)
            lam, _, _ = spectrum(D.features(W, flight))
            r["pr_flight"], r["erank_flight"], r["k90_flight"] = geometry(lam)
            acc = {}
            for t in TASKS:
                x, ya, yv = teach[t]
                F = D.features(W, x)
                lam, U, Fd = spectrum(F)
                g = geometry(lam)
                for k, v in zip(("pr_teacher", "erank_teacher", "k90_teacher"), g):
                    acc.setdefault(k, []).append(v)
                for tgt, Y in (("value", np.asarray(yv)[:, None]), ("action", np.clip(np.asarray(ya), -1, 1))):
                    outside, insample, weak = [], [], {c: [] for c in CS}
                    for j in range(Y.shape[1]):
                        y = torch.tensor(Y[:, j], dtype=torch.float64, device=Fd.device)
                        lk, e = energy(lam, U, Fd, y)
                        outside.append(float(1 - e.sum()))
                        insample.append(float(e.sum()))
                        for c in CS:
                            a = c * float(lam.mean())
                            weak[c].append(float((e * (a / (lk + a)) ** 2).sum()))
                    acc.setdefault(f"{tgt}_outside_span", []).append(np.mean(outside))
                    acc.setdefault(f"{tgt}_r2_ols_in", []).append(np.mean(insample))
                    for c in CS:
                        acc.setdefault(f"{tgt}_weak_c{c:g}", []).append(np.mean(weak[c]))
                    ho = held_out(F.cpu().numpy(), Y, rr)
                    acc.setdefault(f"{tgt}_r2_ols_out", []).append(ho["ols"])
                    acc.setdefault(f"{tgt}_r2_ridge_out", []).append(ho["ridge"])
            r.update({k: float(np.mean(v)) for k, v in acc.items()})
            rows.append(r)
            print(init, s, {k: round(v, 3) for k, v in r.items() if k in ("pr_flight", "erank_flight", "pr_teacher",
                                                                          "erank_teacher", "value_outside_span",
                                                                          "value_weak_c0.01")}, flush=True)
    df = pd.DataFrame(rows)
    fc = freezing_cost()
    for k in ("g600_frozen", "frozen_over_fullft"):
        df[k] = [fc.get((i, s), {}).get(k, np.nan) for i, s in zip(df.init, df.seed)]
    df.to_csv(A / "rep_review.csv", index=False)
    out = {"per_backbone": df.groupby("init").mean(numeric_only=True).round(4).drop(columns="seed").to_dict("index")}
    d = df.dropna(subset=["g600_frozen"])
    inits = sorted(d.init.unique())
    brng = np.random.default_rng(9)
    xs_cols = ["pr_teacher", "erank_teacher", "pr_flight", "erank_flight", "value_outside_span", "value_r2_ols_out",
               "value_r2_ridge_out", "action_r2_ridge_out"] + [f"value_weak_c{c:g}" for c in CS]
    for x in xs_cols:
        for y in ("g600_frozen", "frozen_over_fullft"):
            rho = spearmanr(d[x], d[y])[0]
            bs = []
            for _ in range(2000):
                pick = brng.choice(inits, len(inits))
                dd = pd.concat([d[d.init == i] for i in pick])
                if dd[x].nunique() > 1 and dd[y].nunique() > 1:
                    bs.append(spearmanr(dd[x], dd[y])[0])
            out[f"{x}|{y}"] = dict(rho=round(float(rho), 3), cluster95=np.round(np.percentile(bs, [2.5, 97.5]), 3).tolist(),
                                   n=int(len(d)), clusters=len(inits))
    (A / "rep_review.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k != "per_backbone":
            print(k, v)
    print(pd.DataFrame(out["per_backbone"]).T[["pr_teacher", "erank_teacher", "pr_flight", "erank_flight",
                                                "value_outside_span", "value_weak_c0.01", "value_r2_ols_out",
                                                "value_r2_ridge_out", "g600_frozen", "frozen_over_fullft"]].round(3).to_string())

if __name__ == "__main__":
    main()
