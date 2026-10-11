"""Pretraining of the recurrent backbone on the second domain (pre-registration v2, section 20), with the recipe of
scripts/pretrain_time_ordered_v3.py: windows of 33 consecutive steps of one robot that contain no reset, states and
actions standardized with domain2/data/stats.pt, Adam 3e-4, gradient-norm clip 1.0, 25 epochs x 1,000 x 2,048.
  MAE : hide each state entry with probability 0.3, reconstruct the hidden entries (MaskedStateLoss, 48 entries).
  PIDM (paper: AP): predict the next four actions from z_t (PIDMLoss, 12 actions).
  MAEs (paper: MAE-T; section 28, V16): hide whole time steps (all 48 entries), the ratio drawn per minibatch from
       {0.15, 0.3, 0.5, 0.75}, as the quadrotor MAEs of pretrain_time_ordered_v3.py (MaskDP).
--data broad (section 28, V16): domain2/data/anymal_flat_broad.npz (collect_data.py --broad), standardized with the
same fixed statistics domain2/data/stats.pt; the backbone name gets the suffix "br" (e.g. P_MAEsbr__seed7__d2anymal).
--data terrain (section 29, V17): domain2/data/terrain_plane.npz and terrain_rough.npz (collect_terrain.py) without
their held-out robots (every eighth robot, kept for the probe), standardized with domain2/data/terrain_stats.pt; backbones are named
P_<objective><tag>__seed<seed>__d2terrain__backbone.pt. The domain-specific settings of section 29 are given by
--mask-unit {entry, step}, --mask-ratio (-1 = drawn per minibatch from {0.15, 0.3, 0.5, 0.75}), and --ap-steps.
Usage: python pretrain_d2.py --objective MAE --seed 7 [--data broad|terrain] [--mask-unit step --mask-ratio -1 --tag t]
Output: ckpt_time_ordered_v3/P_<objective>__seed<seed>__d2anymal__backbone.pt (same format as the quadrotor backbones).
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import igenv  # noqa: E402,F401  (isaacgym before torch)
import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone  # noqa: E402
from aerial_gym.ssl_pretrain.losses import MaskedStateLoss, PIDMLoss  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "anymal_flat.npz")
STATS = os.path.join(HERE, "data", "stats.pt")
OUT = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3"
WIN, S_DIM, A_DIM = 33, 48, 12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["MAE", "PIDM", "MAEs"])
    ap.add_argument("--data", choices=["flat", "broad", "terrain"], default="flat")
    ap.add_argument("--mask-unit", choices=["entry", "step"], default="entry")
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--ap-steps", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    if a.data == "terrain":
        parts = [np.load(os.path.join(HERE, "data", f"terrain_{k}.npz")) for k in ("plane", "rough")]
        keep = [np.setdiff1d(np.arange(p["resets"].shape[1]), p["hold_idx"]) for p in parts]
        d = {k: np.concatenate([p[k][:, kp] for p, kp in zip(parts, keep)], 1) for k in ("states", "actions", "resets")}
        st = torch.load(os.path.join(HERE, "data", "terrain_stats.pt"))
    else:
        d = np.load(DATA if a.data == "flat" else DATA.replace("anymal_flat.npz", "anymal_flat_broad.npz"))
        st = torch.load(STATS)
    S = torch.from_numpy((d["states"] - np.asarray(st["state_mean"], np.float32)) / np.asarray(st["state_std"], np.float32))
    A = torch.from_numpy((d["actions"] - np.asarray(st["action_mean"], np.float32)) / np.asarray(st["action_std"], np.float32))
    resets = d["resets"]                                   # [T, N]; True where the step ended an episode
    T, N = resets.shape
    # a window [t, t + 33) of robot n is valid when no episode ends at steps t .. t + 31
    c = np.concatenate([np.zeros((1, N), np.int64), np.cumsum(resets, 0)], 0)
    t0 = np.arange(T - WIN + 1)
    bad = (c[t0 + WIN - 1] - c[t0]) > 0                    # [T - 32, N]
    valid = np.argwhere(~bad)                              # (t, n) pairs
    print(f"{a.objective} seed={a.seed} valid windows {len(valid)} of {bad.size}", flush=True)
    backbone = ProprioGRUBackbone(proprio_dim=S_DIM).to(dev)
    head = (MaskedStateLoss(128, S_DIM, mask_ratio=0.3) if a.objective in ("MAE", "MAEs")
            else PIDMLoss(128, A_DIM, predict_steps=a.ap_steps)).to(dev)
    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    ar = torch.arange(WIN)
    hist, tic = [], time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for _ in range(a.steps):
            pick = valid[rng.integers(0, len(valid), size=a.batch)]
            ts, ns = torch.from_numpy(pick[:, 0]), torch.from_numpy(pick[:, 1])
            idx_t, idx_n = ts[:, None] + ar[None, :], ns[:, None]
            x = S[idx_t, idx_n].to(dev, non_blocking=True)    # [B, 33, 48]
            u = A[idx_t, idx_n].to(dev, non_blocking=True)    # [B, 33, 12]
            if a.objective == "MAE":
                r = a.mask_ratio if a.mask_ratio >= 0 else float(rng.choice([0.15, 0.3, 0.5, 0.75]))
                if a.mask_unit == "entry":
                    mask = torch.rand(x.shape, device=dev) < r
                else:
                    mask = (torch.rand(x.shape[:2], device=dev) < r).unsqueeze(-1).expand_as(x)
                z, _ = backbone(x * (~mask).float())
                loss = head.forward_with_z_full(z, x, mask)
            elif a.objective == "MAEs":
                r = float(rng.choice([0.15, 0.3, 0.5, 0.75]))
                mask = (torch.rand(x.shape[:2], device=dev) < r).unsqueeze(-1).expand_as(x)
                z, _ = backbone(x * (~mask).float())
                loss = head.forward_with_z_full(z, x, mask)
            else:
                z, _ = backbone(x)
                loss = head.forward_with_z_full(z, u)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item()
        hist.append(tot / a.steps)
        print(f"[{a.objective}/seed{a.seed}/d2] epoch {ep + 1}/{a.epochs} loss={hist[-1]:.6f} elapsed={time.time() - tic:.1f}s", flush=True)
    tag = "br" if a.data == "broad" else ""
    src = "d2terrain" if a.data == "terrain" else "d2anymal"
    path = os.path.join(OUT, f"P_{a.objective}{tag}{a.tag}__seed{a.seed}__{src}__backbone.pt")
    torch.save({"backbone_state_dict": backbone.state_dict(), "objective": a.objective, "seed": a.seed,
                "loss_history": hist, "proprio_dim": S_DIM, "fusion_hidden": 128, "gru_hidden": 128,
                "domain": "anymal_terrain (IsaacGymEnvs), collector = shipped AnymalTerrain PPO" if a.data == "terrain"
                else "anymal_flat (IsaacGymEnvs), collector = shipped PPO", "window_rows": WIN, "data": a.data,
                "mask_unit": a.mask_unit, "mask_ratio": a.mask_ratio, "ap_steps": a.ap_steps,
                "head_state_dict": head.state_dict(), "state_mean": st["state_mean"], "state_std": st["state_std"],
                "action_mean": st["action_mean"], "action_std": st["action_std"]}, path)
    print("saved", path)

if __name__ == "__main__":
    main()
