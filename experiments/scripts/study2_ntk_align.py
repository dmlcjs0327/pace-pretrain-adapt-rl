"""Empirical check of Proposition 3 (time to a level under linearized training, pre-registration v2, section 24 note):
alignment of the full-network tangent kernel with the targets that RL must learn, at the start of FullFT.

For every backbone (random under seed s, or one of the eight objectives with seed s), a policy of the study shape is
built with new actor and critic heads drawn under the same seed s (so all backbones of a seed share the same heads).
On 192 test windows of the S19 teacher data per T1/T2 setting (generator seed 5), per-example gradients of the critic
output and of the four actor outputs at the last step are taken with respect to all parameters. With J the matrix of
these gradients, K = J J^T is the empirical tangent kernel at initialization. For the residual r = y - f_0 (y: the
teacher's standardized value, or the teacher's mean action), the relative speed of the residual is
    S = (r^T K r / r^T r) / (tr K / n),
the ratio of the initial decay rate of the loss along r to the mean eigenvalue of K. S > 1 means that the kernel moves
the network faster along the target than along an average direction. The centered versions (suffix _c) first remove the
mean over states of each output, a direction that every kernel learns fast, so they measure the state-dependent part. Reported per backbone: S for value and action
(mean over the four settings). Then Spearman correlations over backbone-seed units with the convergence speed-up of
FullFT (time_to_level_screening.csv, levels 0.8 and 1.0) and with the early gain g100_full (s22_align.csv).
Output: logs/study2/analysis/ntk_align.csv and ntk_align.json.
Section 25 (review of v12.3): --extra adds the shuffled MAE and AP backbones (scale-matched, structure-free) to the
table (rows appended; the correlations stay over the original backbones).
Usage: python study2_ntk_align.py [--extra]
"""
import argparse
import isaacgym  # noqa: F401  (must precede torch)
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import study2_drift as D
from study2_s19_student import Student
from study2_s22_align import CK, OBJ, TASKS, test_windows

A = D.L / "analysis"
N = 192

def per_example_kernel(model, x, out):
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(x.shape[0]):
        h, mu, v = model(x[i:i + 1])
        f = v[:, -1] if out == "value" else mu[:, -1, :]
        for k in range(f.numel()):
            g = torch.autograd.grad(f.reshape(-1)[k], params, retain_graph=k + 1 < f.numel(), allow_unused=True)
            rows.append(torch.cat([(t if t is not None else torch.zeros_like(q)).reshape(-1) for t, q in zip(g, params)]))
    J = torch.stack(rows)
    return J @ J.T

def speed(K, r, center=False):
    if center:  # remove the mean over states of each output, which every kernel learns fast (bias direction)
        r = r - r.mean(0, keepdim=True)
    r = r.reshape(-1, 1)
    return float((r.T @ K @ r).item() / (r.T @ r).item() / (torch.trace(K).item() / K.shape[0]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(5)
    data = {t: test_windows(t, rng, n=N) for t in TASKS}
    rows = []
    for init in (["MAEshuf", "PIDMshuf"] if args.extra else ["random"] + OBJ):
        for s in (7, 42, 99):
            torch.manual_seed(s)
            m = Student().to(D.dev)
            if init != "random":
                p = f"{CK}/P_{init}__seed{s}__srcmixed__backbone.pt"
                if not os.path.exists(p):
                    continue
                m.backbone.load_state_dict(torch.load(p, map_location="cpu")["backbone_state_dict"])
            m.train()
            sv, sa, svc, sac = [], [], [], []
            for t in TASKS:
                x, ya, yv = data[t]
                with torch.no_grad():
                    _, mu0, v0 = m(x)
                z = torch.tensor((yv - yv.mean()) / (yv.std() + 1e-8), dtype=torch.float32, device=D.dev)
                Kv = per_example_kernel(m, x, "value")
                sv.append(speed(Kv, z - v0[:, -1])); svc.append(speed(Kv, (z - v0[:, -1])[:, None], center=True))
                a = torch.tensor(np.clip(ya, -1, 1), dtype=torch.float32, device=D.dev)
                Ka = per_example_kernel(m, x, "action")
                sa.append(speed(Ka, a - mu0[:, -1, :])); sac.append(speed(Ka, a - mu0[:, -1, :], center=True))
            rows.append(dict(init=init, seed=s, speed_value=float(np.mean(sv)), speed_action=float(np.mean(sa)),
                             speed_value_c=float(np.mean(svc)), speed_action_c=float(np.mean(sac))))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    if args.extra:
        old = pd.read_csv(A / "ntk_align.csv")
        old = old[~old.init.isin(["MAEshuf", "PIDMshuf"])]
        pd.concat([old, d], ignore_index=True).to_csv(A / "ntk_align.csv", index=False)
        print(pd.concat([old, d]).groupby("init")[["speed_value_c", "speed_action_c"]].mean().round(3).to_string())
        return
    t = pd.read_csv(A / "time_to_level_screening.csv")
    t = t[t["mode"] == "FullFT"]
    for q in (0.8, 1.0):
        sp = {}
        for (init, s), g in t.groupby(["init", "seed"]):
            ok = g[g[f"s{q}"].notna() & g[f"p{q}"].notna()]
            if len(ok):
                sp[(init, s)] = float(np.median(ok[f"s{q}"] / ok[f"p{q}"]))
        d[f"speedup_q{q}"] = [sp.get((i, s), 1.0 if i == "random" else np.nan) for i, s in zip(d.init, d.seed)]
    al = pd.read_csv(A / "s22_align.csv")[["init", "seed", "g100_full"]]
    d = d.merge(al, on=["init", "seed"], how="left")
    d.to_csv(A / "ntk_align.csv", index=False)
    out = {}
    for x in ("speed_value", "speed_action", "speed_value_c", "speed_action_c"):
        for y in ("speedup_q0.8", "speedup_q1.0", "g100_full"):
            dd = d[[x, y]].dropna()
            rho, p = spearmanr(dd[x], dd[y])
            out[f"{x}|{y}"] = dict(rho=round(float(rho), 3), p=round(float(p), 4), n=int(len(dd)))
    out["per_backbone"] = d.groupby("init")[["speed_value_c", "speed_action_c", "speedup_q0.8", "g100_full"]].mean().round(3).to_dict("index")
    (A / "ntk_align.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
