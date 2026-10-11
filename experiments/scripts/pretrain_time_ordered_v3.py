"""Time-ordered, standardized pretraining of the recurrent backbone for four objectives
(MAE, CPC, SPR, PIDM) under one recipe — v3 data.

v3 changes: data from collect_offline_trajectory_v3.py (flown through the attitude interface of RL, correct
world-frame outer loop, 5 normal-start and 4 upset-start episodes per airframe); mixed source = X500 (R2) and
LMF2 (R3) only, all 2,700 time steps each (about 11 million states each), weighted equally; the base
quadrotor (R1) is held out; commands are the executed RL-interface commands; SPR uses a_t..a_{t+k-1}
(losses.py fix). The statistics are written to data/v3/state_stats.pt and reused as the fixed input
normalization of RL.

Data: the offline buffers store the 4,096 vehicles of one time step contiguously (row = t * 4096 + n).
A window is 33 consecutive time steps of ONE vehicle inside one 300-step collection episode.
States and collection commands are standardized with the per-entry mean and standard deviation
of the pretraining data. The objectives, heads, and losses are those of aerial_gym/ssl_pretrain
(losses.py) with the settings of trainer.py:
  MAE : hide each state entry with probability 0.3, reconstruct the hidden entries (MaskedStateLoss).
  CPC : InfoNCE between z_t and z_{t+k}, k in {1, 2, 4, 8} (ContrastivePredictiveLoss).
  SPR : predict projected target latents 1, 2, 3, 5 steps ahead from z_t and the commands, with a
        moving-average target backbone (decay 0.99) (SelfPredictiveLoss).
  PIDM: predict the next four commands from z_t (PIDMLoss).
Optimizer: Adam 3e-4, gradient-norm clip 1.0, 25 epochs x 1,000 minibatches x 2,048 windows.

Usage:
  python pretrain_time_ordered.py --objective SPR --seed 7
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401  (must precede torch in this code base)
import argparse
import copy
import os
import time

import h5py
import numpy as np
import torch

from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
from aerial_gym.ssl_pretrain.losses import (ContrastivePredictiveLoss, MaskedStateLoss, PIDMLoss,
                                            SelfPredictiveLoss)
from aerial_gym.ssl_pretrain.extra_losses import BehaviorCloningLoss, ForwardDynamicsLoss, InverseDynamicsLoss

DATA = _DATA_ROOT + "/"
FILES = {"x500": ("v3/ssl_v3_x500.h5", "x500"),
         "lmf2": ("v3/ssl_v3_lmf2.h5", "lmf2_proprio_clean")}
SOURCES = ("x500", "lmf2")
OUT = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3"
STATS = DATA + "v3/state_stats.pt"
N_ENVS, EP_LEN, WIN, MIXED_STEPS = 4096, 300, 33, 2700

def load(name, n_steps):
    fn, grp = FILES[name]
    with h5py.File(DATA + fn, "r") as f:
        s = f[grp]["states"][: n_steps * N_ENVS].reshape(n_steps, N_ENVS, 13)
        a = f[grp]["actions"][: n_steps * N_ENVS].reshape(n_steps, N_ENVS, 4)
    return s, a

def loss_module(obj, a_mean=None, a_std=None, z=128, d=13):
    return {"IDM": lambda: InverseDynamicsLoss(z, 4),
            "FDM": lambda: ForwardDynamicsLoss(z, 4, d),
            "BC": lambda: BehaviorCloningLoss(z, 64, 4, a_mean, a_std),
            "MAE": lambda: MaskedStateLoss(z, d, mask_ratio=0.3),
            "CPC": lambda: ContrastivePredictiveLoss(z, predict_steps=(1, 2, 4, 8), temperature=0.1),
            "SPR": lambda: SelfPredictiveLoss(z, 4, predict_steps=(1, 2, 3, 5)),
            "PIDM": lambda: PIDMLoss(z, 4, predict_steps=4)}[obj]()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["MAE", "CPC", "SPR", "PIDM", "IDM", "FDM", "BC"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    # MAE variants; the defaults reproduce the study backbones exactly
    ap.add_argument("--mask-unit", choices=["entry", "step"], default="entry",
                    help="entry: hide each state entry; step: hide whole time steps (all 13 entries)")
    ap.add_argument("--mask-ratio", type=float, default=0.3,
                    help="masking probability; a negative value draws it per minibatch from {0.15, 0.3, 0.5, 0.75} (MaskDP)")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--tag", default="")
    # head refit: load a finished backbone, keep it frozen, and train only the head, so
    # that the objective can be continued as an auxiliary loss during RL
    ap.add_argument("--init-backbone", default=None)
    ap.add_argument("--freeze-backbone", action="store_true")
    # S10: target-swapped data. In every window the goal-relative
    # position becomes dp_t - dp_1 + dp_1 of another window of the minibatch: the kinematics inside the window and
    # the distribution of the first distance stay, the goal-directedness of the collector is removed. Standardization
    # is affine per entry, so the same formula applies to the standardized entries 0:3.
    ap.add_argument("--target-swap", action="store_true")
    # V14: target-masked data. The standardized goal-relative entries 0:3
    # are set to 0 (the data mean) in every window, so the backbone never sees the goal. The input weights of these
    # entries then get no gradient and keep their initial values. Other entries and the commands are unchanged.
    ap.add_argument("--target-mask", action="store_true")
    ap.add_argument("--width", type=float, default=1.0,
                    help="S17 (pre-registration v2, section 19): every hidden width x this factor; 1 = the study default")
    # S20 (pre-registration v2, section 21): the fixed sensor model of ssl_pretrain/expand.py (13 -> 128 inputs)
    ap.add_argument("--expand", action="store_true")
    # S21 (pre-registration v2, section 21): the time steps of every window are permuted at random (one permutation per
    # window, the 13 entries of a step move together), so the set of states and the within-step structure stay and
    # the temporal dynamics are removed
    ap.add_argument("--time-shuffle", action="store_true")
    # V3 (pre-registration v2, section 25): task-agnostic data (collect_offline_trajectory_v3ag.py), standardized with
    # the v3 statistics like every other backbone
    ap.add_argument("--data", choices=["v3", "v3ag"], default="v3")
    a = ap.parse_args()
    if a.expand and a.objective != "MAE":
        raise SystemExit("--expand is defined for MAE only (section 21)")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"

    if a.data == "v3ag":
        FILES.update({"x500": ("v3ag/ssl_v3ag_x500.h5", "x500"), "lmf2": ("v3ag/ssl_v3ag_lmf2.h5", "lmf2_proprio_clean")})
    arrays = [load(n, MIXED_STEPS) for n in SOURCES]
    st = torch.load(STATS)  # written once by make_v3_data.py from the same arrays
    s_mean, s_std = np.asarray(st["state_mean"]), np.asarray(st["state_std"])
    a_mean, a_std = np.asarray(st["action_mean"]), np.asarray(st["action_std"])
    states = [torch.from_numpy((s - s_mean) / s_std).float() for s, _ in arrays]
    acts = [torch.from_numpy((x - a_mean) / a_std).float() for _, x in arrays]
    t = np.arange(MIXED_STEPS - WIN + 1)
    starts = t[(t % EP_LEN) <= EP_LEN - WIN]            # window stays inside one episode
    print(f"{a.objective} seed={a.seed} windows per airframe={len(starts)} x {N_ENVS} vehicles", flush=True)

    PH, FH, GH = int(round(64 * a.width)), int(round(128 * a.width)), int(round(128 * a.width))
    IN = 13
    if a.expand:
        from aerial_gym.ssl_pretrain.expand import expansion_params, expand, OUT_DIM
        EA, Eb = expansion_params(dev)
        IN = OUT_DIM
    backbone = ProprioGRUBackbone(proprio_dim=IN, proprio_hidden=PH, fusion_hidden=FH, gru_hidden=GH).to(dev)
    if a.init_backbone:
        backbone.load_state_dict(torch.load(a.init_backbone, map_location="cpu")["backbone_state_dict"])
    if a.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    head = loss_module(a.objective, a_mean, a_std, z=GH, d=IN).to(dev)
    target = None
    if a.objective == "SPR":
        target = copy.deepcopy(backbone).to(dev)
        for p in target.parameters():
            p.requires_grad = False
        target.eval()
    params = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    ar = torch.arange(WIN)
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for _ in range(a.steps):
            which = rng.integers(0, len(SOURCES), size=a.batch)      # airframes weighted equally
            xs, us = [], []
            for k in range(len(SOURCES)):
                m = int((which == k).sum())
                ts = torch.from_numpy(rng.choice(starts, size=m))
                ns = torch.from_numpy(rng.integers(0, N_ENVS, size=m))
                idx_t, idx_n = ts[:, None] + ar[None, :], ns[:, None]
                xs.append(states[k][idx_t, idx_n])
                us.append(acts[k][idx_t, idx_n])
            x = torch.cat(xs, 0).to(dev, non_blocking=True)      # [B, 33, 13]
            if a.target_swap:
                perm = torch.randperm(x.shape[0], device=dev)
                x = x.clone()
                x[:, :, 0:3] = x[:, :, 0:3] - x[:, :1, 0:3] + x[perm, :1, 0:3]
            if a.target_mask:
                x = x.clone()
                x[:, :, 0:3] = 0.0
            u = torch.cat(us, 0).to(dev, non_blocking=True)      # [B, 33, 4]
            if a.time_shuffle:
                perm = torch.argsort(torch.rand(x.shape[0], WIN, device=dev), dim=1)
                x = torch.gather(x, 1, perm.unsqueeze(-1).expand_as(x))
                u = torch.gather(u, 1, perm.unsqueeze(-1).expand_as(u))
            if a.expand:
                x = expand(x, EA, Eb)
            if a.objective == "MAE":
                r = a.mask_ratio if a.mask_ratio >= 0 else float(rng.choice([0.15, 0.3, 0.5, 0.75]))
                if a.mask_unit == "entry":
                    mask = torch.rand(x.shape, device=dev) < r
                else:
                    mask = (torch.rand(x.shape[:2], device=dev) < r).unsqueeze(-1).expand_as(x)
                z, _ = backbone(x * (~mask).float())
                loss = head.forward_with_z_full(z, x, mask)
            elif a.objective == "SPR":
                z, _ = backbone(x)
                with torch.no_grad():
                    zt, _ = target(x)
                loss = head.forward_with_z_full(z, zt, u)
            elif a.objective in ("PIDM", "IDM", "BC"):
                z, _ = backbone(x)
                loss = head.forward_with_z_full(z, u)
            elif a.objective == "FDM":
                z, _ = backbone(x)
                loss = head.forward_with_z_full(z, x, u)
            else:
                z, _ = backbone(x)
                loss = head.forward_with_z_full(z)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if target is not None and not a.freeze_backbone:
                with torch.no_grad():
                    for pt, po in zip(target.parameters(), backbone.parameters()):
                        pt.mul_(0.99).add_(po, alpha=0.01)
            tot += loss.item()
        hist.append(tot / a.steps)
        print(f"[{a.objective}/seed{a.seed}] epoch {ep + 1}/{a.epochs} loss={hist[-1]:.6f} "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    wtag = "" if a.width == 1.0 else f"__w{a.width:g}"
    path = os.path.join(a.out_dir, f"P_{a.objective}{a.tag}__seed{a.seed}__srcmixed{wtag}__backbone.pt")
    torch.save({"backbone_state_dict": backbone.state_dict(), "objective": a.objective, "seed": a.seed, "data": a.data,
                "loss_history": hist, "proprio_dim": IN, "expand": a.expand, "time_shuffle": a.time_shuffle, "proprio_hidden": PH, "fusion_hidden": FH, "gru_hidden": GH, "width": a.width,
                "data_order": "time_ordered_single_vehicle_within_episode", "window_rows": WIN,
                "source": "v3: x500 and lmf2, all 2700 time steps each", "head_state_dict": head.state_dict(),
                "mask_unit": a.mask_unit, "mask_ratio": a.mask_ratio, "init_backbone": a.init_backbone,
                "freeze_backbone": a.freeze_backbone, "target_swap": a.target_swap, "target_mask": a.target_mask,
                "state_mean": s_mean.tolist(), "state_std": s_std.tolist(),
                "action_mean": a_mean.tolist(), "action_std": a_std.tolist()}, path)
    print(f"Saved {path}", flush=True)

if __name__ == "__main__":
    main()
