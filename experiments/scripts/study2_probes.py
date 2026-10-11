"""Mechanism probes of Study 2 (exploratory, pre-registration section 6; method Table 'tab:probes').

For every finished run it reads the policies saved every 25 iterations and writes
  probes_dyn.csv   one row per run and checkpoint (iteration 0 = start of RL, then 25, 50, ..., 600)
                   erank, dead          effective rank and fraction of dead units of the final GRU output (P3)
                   dW_<block>           relative weight change of encoder, gru, actor, critic since the previous
                                        checkpoint, ||W_k - W_{k-25}|| / ||W_{k-25}|| (P4; heads from k = 50)
  probes_info.csv  one row per run and iteration in {0, 300, 600}
                   acc_airframe         test accuracy of a logistic regression that identifies the airframe (P1)
                   r2_<q>_<h>           test R^2 of a ridge regression of quantity q in {vel, tilt, rate}
                                        h in {5, 10} steps after the window (P1)
                   jac_norm             mean Frobenius norm of d(actor mean)/d(normalized observation) at the last
                                        step of the window, normalized as in RL (running statistics of the run,
                                        clipped to [-5, 5] as in rl-games) (P5)
                   jac_std              the same with respect to the observation standardized by the
                                        pretraining statistics, so runs share one unit
  probes_ref.json  the same information probes for two references without a network: the last state of the
                   window, and the whole window flattened
Features for P1 and P3 are the final GRU outputs of a fixed probe set of 4,096 windows of 33 steps (1,366 /
1,365 / 1,365 from the three airframes), under two input pipelines (column suffix):
  _pre  standardized with the pretraining statistics as in study2_drift.py (isolates the weights)
  _rl   normalized with the running statistics of the run at that checkpoint and clipped to [-5, 5], as the
        policy sees its input in RL (iteration 0 uses the statistics of iteration 25)
cka_rl in probes_dyn.csv is the CKA to the start of RL under the second pipeline, so it includes the drift of
the running normalization (study2_drift.py gives the first pipeline).
Windows end at least 10 steps before the end of their episode so that the future targets exist.
Low-rank updates are folded into effective weights. The start of RL is the pretrained backbone file, and for
random runs the backbone of the Frozen run with the same task and seed (as in study2_drift.py).
Usage: python study2_probes.py [--limit N]     (incremental)
"""
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import glob
import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import study2_drift as D

torch.set_num_threads(8)
dev = "cpu"
D.dev = dev
A = D.L / "analysis"
H = (5, 10)
EPS = 1e-5

def probe_set():
    """Windows (N, 33, 13) raw, airframe labels, and future targets (v3 data of every airframe in D.FILES)."""
    import h5py
    rng = np.random.default_rng(11)
    X, lab, fut = [], [], []
    n_air = len(D.FILES)
    sizes = [4096 // n_air + (1 if k < 4096 % n_air else 0) for k in range(n_air)]
    for a, ((fn, g), n) in enumerate(zip(D.FILES, sizes)):
        with h5py.File(D.DATA + fn, "r") as f:
            s = f[g]["states"][:2700 * 4096].reshape(2700, 4096, 13)
        ok = np.arange(2700)[(np.arange(2700) % 300) <= 300 - 33 - max(H)]
        ts, ns = rng.choice(ok, n), rng.integers(0, 4096, n)
        X.append(np.stack([s[t:t + 33, k] for t, k in zip(ts, ns)]))
        fut.append(np.stack([np.stack([s[t + 32 + h, k] for h in H]) for t, k in zip(ts, ns)]))
        lab.append(np.full(n, a))
    X, fut, lab = np.concatenate(X), np.concatenate(fut), np.concatenate(lab)
    q = fut[:, :, 3:7]  # quaternion x, y, z, w
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2), -1, 1)))
    T = {}
    for j, h in enumerate(H):
        T[f"vel_{h}"], T[f"tilt_{h}"], T[f"rate_{h}"] = fut[:, j, 7:10], tilt[:, j:j + 1], fut[:, j, 10:13]
    return X, lab, T

def split(n):
    idx = np.random.default_rng(3).permutation(n)
    return idx[: n // 2], idx[n // 2:]

def info_probe(Z, lab, T, tr, te):
    out = {}
    sc = StandardScaler().fit(Z[tr])
    sc.scale_ = np.maximum(sc.scale_, 1e-3)  # dead units stay near zero instead of exploding
    Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr, lab[tr])
    out["acc_airframe"] = float(clf.score(Zte, lab[te]))
    for k, y in T.items():
        r = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(Ztr, y[tr])
        out[f"r2_{k}"] = float(r2_score(y[te], r.predict(Zte), multioutput="uniform_average"))
    return out

def plasticity(Z):
    Zc = Z - Z.mean(0)
    sv = torch.linalg.svdvals(Zc)
    p = sv ** 2 / (sv ** 2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum())), float((Z.std(0) < 1e-3).float().mean())

def effective_all(sd, prefix="a2c_network.policy."):
    W = D.effective_backbone(sd, prefix)
    for blk in ("actor_trunk.0", "actor_mu", "critic_trunk.0", "critic_v"):
        W[f"{blk}.weight"], W[f"{blk}.bias"] = sd[f"{prefix}{blk}.weight"].float(), sd[f"{prefix}{blk}.bias"].float()
    return W

BLOCKS = {"encoder": [f"{b}.{i}.{p}" for b in ("proprio_mlp", "fusion_mlp") for i in ("0", "2") for p in ("weight", "bias")],
          "gru": ["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0"],
          "actor": [f"{b}.{p}" for b in ("actor_trunk.0", "actor_mu") for p in ("weight", "bias")],
          "critic": [f"{b}.{p}" for b in ("critic_trunk.0", "critic_v") for p in ("weight", "bias")]}

def actor_mean(W, x_norm_seq):
    """Actor mean at the last step for inputs already normalized as the network sees them, (N, 33, 13)."""
    z = D.features(W, x_norm_seq)
    h = torch.nn.functional.elu(z @ W["actor_trunk.0.weight"].T + W["actor_trunk.0.bias"])
    return h @ W["actor_mu.weight"].T + W["actor_mu.bias"]

def rl_norm(sd, xraw):
    if "running_mean_std.running_mean" not in sd:  # study 2 v2: fixed input normalization inside the network
        m, s = sd["a2c_network.fixed_obs_mean"].float(), sd["a2c_network.fixed_obs_std"].float()
        return torch.clamp((xraw - m) / s, -5.0, 5.0), m, s
    m = sd["running_mean_std.running_mean"].float()
    s = torch.sqrt(sd["running_mean_std.running_var"].float() + EPS)
    return torch.clamp((xraw - m) / s, -5.0, 5.0), m, s

def suffixed(d, suf):
    return {f"{k}_{suf}": v for k, v in d.items()}

def jacobian(W, sd, xraw, pre_std):
    """Mean Frobenius norm of d mu / d x_last, w.r.t. the RL-normalized input and w.r.t. pretraining units."""
    xn, m, s = rl_norm(sd, xraw)
    head, last = xn[:, :-1].detach(), xn[:, -1:].detach().clone().requires_grad_(True)
    mu = actor_mean(W, torch.cat([head, last], 1))
    J = torch.stack([torch.autograd.grad(mu[:, i].sum(), last, retain_graph=i < mu.shape[1] - 1)[0][:, 0] for i in range(mu.shape[1])], 1)
    inside = ((xraw[:, -1] - m) / s).abs() < 5.0
    J = J * inside[:, None, :]
    J_std = J * (pre_std / s)[None, None, :]
    return float(J.norm(dim=(1, 2)).mean()), float(J_std.norm(dim=(1, 2)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Xraw, lab, T = probe_set()
    ref = torch.load(D.CK + "P_MAE__seed7__srcmixed__backbone.pt", map_location="cpu")
    pre_m, pre_s = torch.tensor(ref["state_mean"], dtype=torch.float32), torch.tensor(ref["state_std"], dtype=torch.float32)
    xr = torch.tensor(Xraw, dtype=torch.float32)
    xstd = (xr - pre_m) / pre_s
    tr, te = split(len(lab))
    if not (A / "probes_ref.json").exists():
        from sklearn.preprocessing import PolynomialFeatures
        last = Xraw[:, -1]
        R = {"last_state": info_probe(last, lab, T, tr, te),
             "last_state_quadratic": info_probe(PolynomialFeatures(2, include_bias=False).fit_transform(last), lab, T, tr, te),
             "window": info_probe(Xraw.reshape(len(Xraw), -1), lab, T, tr, te)}
        json.dump(R, open(A / "probes_ref.json", "w"), indent=1)
        print("references:", json.dumps(R))
    paths = {k: A / f"probes_{k}.csv" for k in ("dyn", "info")}
    old = {k: pd.read_csv(p) if p.exists() else pd.DataFrame() for k, p in paths.items()}
    have = set(old["info"].run_id) if len(old["info"]) else set()
    led = pd.read_csv(D.L / "ledger.csv")
    led = led[led.exit == 0].drop_duplicates("run_id", keep="last")
    rows = {"dyn": [], "info": []}
    cache0 = {}
    done = 0
    for _, r in led.iterrows():
        if r.run_id in have:
            continue
        meta = dict(run_id=r.run_id, stage=r.stage, task=r.task, init=r.init, mode=r["mode"], seed=r.seed)
        if r.init == "random":
            fz = glob.glob(f"{D.RUNS}s2__st{r.stage}__{r.task}__random__EncFrozen__seed{r.seed}__it{r.iters}/nn/last_*_ep_25_rew_*.pth")
            if not fz:
                continue
            W0 = D.effective_backbone(torch.load(fz[0], map_location="cpu")["model"])
            key0 = ("random", r.task, r.seed)
        else:
            W0 = D.ssl_backbone(f"{D.CK}P_{r.init}__seed{r.seed}__srcmixed__backbone.pt")
            key0 = (r.init, r.seed)
        ck = {}
        for f in glob.glob(f"{D.RUNS}{r.run_id}/nn/last_*_ep_*_rew_*.pth"):
            m = re.search(r"_ep_(\d+)_rew_", f)
            if m and not f.endswith("_.pth"):
                ck[int(m.group(1))] = f
        if max(ck, default=0) < int(r.iters):
            continue
        first = torch.load(ck[25], map_location="cpu")["model"]
        x0_rl = rl_norm(first, xr)[0]
        with torch.no_grad():
            Z0, Z0_rl = D.features(W0, xstd), D.features(W0, x0_rl)
        er, dd = plasticity(Z0)
        er2, dd2 = plasticity(Z0_rl)
        rows["dyn"].append(dict(meta, iter=0, erank_pre=er, dead_pre=dd, erank_rl=er2, dead_rl=dd2, cka_rl=1.0))
        if key0 not in cache0:
            cache0[key0] = suffixed(info_probe(Z0.numpy(), lab, T, tr, te), "pre")
        W = effective_all(first)
        jn, js = jacobian(W, first, xr, pre_s)  # heads at iteration 0 are not saved; use iteration 25
        rows["info"].append(dict(meta, iter=0, **cache0[key0], **suffixed(info_probe(Z0_rl.numpy(), lab, T, tr, te), "rl"),
                                 jac_norm=jn, jac_std=js))
        prev = dict(W0)
        for k in sorted(ck):
            sd = torch.load(ck[k], map_location="cpu")["model"]
            W = effective_all(sd)
            with torch.no_grad():
                Z, Z_rl = D.features(W, xstd), D.features(W, rl_norm(sd, xr)[0])
            er, dd = plasticity(Z)
            er2, dd2 = plasticity(Z_rl)
            row = dict(meta, iter=k, erank_pre=er, dead_pre=dd, erank_rl=er2, dead_rl=dd2, cka_rl=D.cka(Z0_rl, Z_rl))
            for b, keys in BLOCKS.items():
                if all(x in prev for x in keys):
                    row[f"dW_{b}"] = D.rel(W, prev, keys)
            rows["dyn"].append(row)
            if k in (300, 600):
                jn, js = jacobian(W, sd, xr, pre_s)
                rows["info"].append(dict(meta, iter=k, **suffixed(info_probe(Z.numpy(), lab, T, tr, te), "pre"),
                                         **suffixed(info_probe(Z_rl.numpy(), lab, T, tr, te), "rl"), jac_norm=jn, jac_std=js))
            prev = W
        done += 1
        print(f"{done:4d} {r.run_id}", flush=True)
        if a.limit and done >= a.limit:
            break
    for k in paths:
        d = pd.concat([old[k], pd.DataFrame(rows[k])], ignore_index=True) if rows[k] else old[k]
        d.to_csv(paths[k], index=False)
    info = pd.concat([old["info"], pd.DataFrame(rows["info"])], ignore_index=True) if rows["info"] else old["info"]
    if len(info):
        cols = [f"{c}_{p}" for p in ("pre", "rl") for c in ("acc_airframe", "r2_vel_10", "r2_tilt_10")] + ["jac_norm", "jac_std"]
        print(info.groupby(["init", "mode", "iter"])[cols].mean().round(3).to_string())

if __name__ == "__main__":
    main()
