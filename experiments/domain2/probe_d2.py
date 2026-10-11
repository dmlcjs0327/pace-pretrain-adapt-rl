"""Reward-free probe that selects the domain-specific pretraining settings of the second domain (section 29, V17).

For each candidate backbone (frozen), windows of 33 steps of the held-out robots (every eighth robot of
domain2/data/terrain_plane.npz and terrain_rough.npz, never used for pretraining; no reset inside the window) are run
through the encoder and the GRU, and the state z_t at step t = 24 of the window is read. Linear ridge probes
(closed form, penalty 1e-3 times the number of fit rows) are fitted on the first half of the held-out robots and scored
by R^2 on the second half, for three target groups:
  future   the 48 standardized inputs at t + 1, t + 4, and t + 8 (one R^2 over all 144 targets);
  velocity the 3 standardized base linear velocities at t, read from z_t computed with the base linear velocity
           removed from the input (raw value 0, as the policy sees it in the setting without velocity);
  terrain  the mean and standard deviation of the 140 height samples at t, rough terrain only (the policy never sees
           the height scan).
R^2 of a group = 1 - sum of squared errors / sum of squares around the fit-set mean, over all targets of the group.
Score = mean of the three group R^2 values. Selection rule (fixed before any run of stages 40 and 41): for each
objective the candidate with the highest score is chosen; when the best two differ by less than 0.01, the candidate
nearest to the quadrotor recipe (entry masking 0.3; four predicted actions) among them is chosen.
Output: domain2/data/probe_selection.json. Usage: python probe_d2.py <backbone .pt> ... [--random]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import igenv  # noqa: E402,F401  (isaacgym before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WIN, T_READ, PER_PART = 33, 24, 12000
DEFAULT_NEAR = {"MAE": "e30", "PIDM": "h4"}

def windows(p, rng):
    """(t, n) starts of valid windows of the held-out robots, split into fit and test halves by robot."""
    hold = p["hold_idx"]
    resets = p["resets"][:, hold]
    T = resets.shape[0]
    c = np.concatenate([np.zeros((1, len(hold)), np.int64), np.cumsum(resets, 0)], 0)
    t0 = np.arange(T - WIN + 1)
    ok = np.argwhere(~((c[t0 + WIN - 1] - c[t0]) > 0))                 # (t, j) with j an index into hold
    half = len(hold) // 2
    out = {}
    for name, sel in (("fit", ok[:, 1] < half), ("test", ok[:, 1] >= half)):
        w = ok[sel]
        out[name] = w[rng.choice(len(w), size=min(PER_PART // 2, len(w)), replace=False)]
    return out

def gather(p, st, w):
    hold = p["hold_idx"]
    idx_t = w[:, 0][:, None] + np.arange(WIN)[None, :]
    x = p["states"][idx_t, hold[w[:, 1]][:, None]]                     # [B, 33, 48] raw
    xs = (x - st["sm"]) / st["ss"]
    xb = x.copy()
    xb[:, :, 0:3] = 0.0
    xbs = (xb - st["sm"]) / st["ss"]
    h = p["holdout_heights"][idx_t[:, T_READ], w[:, 1]].astype(np.float32)   # [B, 140]
    return xs.astype(np.float32), xbs.astype(np.float32), h

def r2(Zf, Yf, Zt, Yt):
    Zf1 = np.c_[Zf, np.ones(len(Zf))]
    Zt1 = np.c_[Zt, np.ones(len(Zt))]
    lam = 1e-3 * len(Zf)
    B = np.linalg.solve(Zf1.T @ Zf1 + lam * np.eye(Zf1.shape[1]), Zf1.T @ Yf)
    e = Yt - Zt1 @ B
    return float(1 - (e ** 2).sum() / ((Yt - Yf.mean(0)) ** 2).sum())

def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    rng = np.random.default_rng(29)
    stt = torch.load(os.path.join(HERE, "data", "terrain_stats.pt"))
    st = {"sm": np.asarray(stt["state_mean"], np.float32), "ss": np.asarray(stt["state_std"], np.float32)}
    data = {}
    for k in ("plane", "rough"):
        p = np.load(os.path.join(HERE, "data", f"terrain_{k}.npz"))
        w = windows(p, rng)
        data[k] = {s: gather(p, st, w[s]) for s in ("fit", "test")}
    cands = [(os.path.basename(q), q) for q in paths]
    if "--random" in sys.argv:
        cands.append(("random", None))
    res = {}
    for name, path in cands:
        torch.manual_seed(7)
        bb = ProprioGRUBackbone(proprio_dim=48).cuda().eval()
        if path:
            bb.load_state_dict(torch.load(path, map_location="cpu")["backbone_state_dict"])
        feats = {}
        with torch.no_grad():
            for k in data:
                for s in ("fit", "test"):
                    xs, xbs, h = data[k][s]
                    z = bb(torch.from_numpy(xs).cuda())[0][:, T_READ].cpu().numpy()
                    zb = bb(torch.from_numpy(xbs).cuda())[0][:, T_READ].cpu().numpy()
                    fut = np.concatenate([xs[:, T_READ + d] for d in (1, 4, 8)], 1)
                    vel = xs[:, T_READ, 0:3]
                    ter = np.stack([h.mean(1), h.std(1)], 1)
                    feats[(k, s)] = (z, zb, fut, vel, ter)
        def cat(s, i):
            return np.concatenate([feats[(k, s)][i] for k in data], 0)
        g_future = r2(cat("fit", 0), cat("fit", 2), cat("test", 0), cat("test", 2))
        g_vel = r2(cat("fit", 1), cat("fit", 3), cat("test", 1), cat("test", 3))
        g_ter = r2(feats[("rough", "fit")][0], feats[("rough", "fit")][4], feats[("rough", "test")][0], feats[("rough", "test")][4])
        res[name] = {"future": round(g_future, 4), "velocity": round(g_vel, 4), "terrain": round(g_ter, 4),
                     "score": round((g_future + g_vel + g_ter) / 3, 4)}
        print(name, res[name], flush=True)
    sel = {}
    for obj in ("MAE", "PIDM"):
        rows = sorted(((v["score"], n) for n, v in res.items() if n.startswith(f"P_{obj}") and "__d2terrain" in n), reverse=True)
        if not rows:
            continue
        best = rows[0]
        close = [n for sc, n in rows if best[0] - sc < 0.01]
        near = [n for n in close if f"{obj}{DEFAULT_NEAR[obj]}__" in n]
        sel[obj] = near[0] if near else best[1]
    out = {"scores": res, "selected": sel}
    with open(os.path.join(HERE, "data", "probe_selection.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("selected", sel)

if __name__ == "__main__":
    main()
