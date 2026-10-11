"""S19 (pre-registration v2, section 21): supervised adaptation of the same backbones and modes as RL.

A student has the policy architecture of the study (backbone + new actor and critic heads). It starts from a random
backbone (default initialization under the seed) or a pretrained one (MAE or AP = run name PIDM, screening seeds) and
is trained by supervised learning on the teacher data of study2_s19_teacher.py, in the mode
  Frozen    : backbone frozen, heads trained
  PartialFT : encoder frozen, GRU and heads trained
  FullFT    : everything trained.
Loss: squared error of the actor mean to the teacher's mean action plus squared error of the standardized critic
output to the standardized teacher value, on steps 9-33 of 33-step windows (the GRU starts from zero; the first 8 steps
are a warm-up). Adam 3e-4, gradient-norm clip 1.0, 10 epochs x 500 minibatches x 1,024 windows, equal for every mode.
Windows never cross an episode end. Training data: all but the last episode; test data: the last episode.
Reported (one row in logs/study2/analysis/s19_students.csv): test R^2 of the action (mean over the 4 dimensions) and
of the value, and the linear CKA between the backbone features (GRU output at the last step) before and after
training on the test windows. The student is saved in the checkpoint format of the RL runs (fixed input normalization,
value normalization from the teacher data) for the closed-loop evaluation (study2_eval_batch.py --files).
Usage: python study2_s19_student.py --task state_setpoint_x500_clean --init MAE --seed 7 --mode Frozen
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401  (must precede torch in this code base)
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
import study2_drift as D

CKPT = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3"
STATS = _DATA_ROOT + "/v3/state_stats.pt"
EPISODES = {"state_setpoint_x500_clean": 2, "ss_lmf2_clean": 2, "recovery_x500": 4, "recovery_lmf2": 4}
WIN, WARM = 33, 8
OUT_CSV = D.L / "analysis" / "s19_students.csv"
OUT_DIR = D.L / "s19" / "students"

def windows(done, t_lo, t_hi):
    """Start indices (t, n) of 33-step windows inside [t_lo, t_hi) that contain no episode end before their last step."""
    T, N = done.shape
    ends = np.zeros((T + 1, N), dtype=np.int64)
    ends[1:] = np.cumsum(done, 0)
    t0 = np.arange(t_lo, t_hi - WIN + 1)
    ok = (ends[t0 + WIN - 1] - ends[t0]) == 0  # no end at steps t0 .. t0+31
    tt, nn_ = np.nonzero(ok)
    return np.stack([t0[tt], nn_], 1)

def r2(pred, y):
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((pred - y) ** 2).sum(0) / np.maximum(ss, 1e-12)

def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))

class Student(nn.Module):
    def __init__(self, gh=128, hh=64, feat="none"):
        super().__init__()
        self.backbone = ProprioGRUBackbone(proprio_dim=13, proprio_hidden=64, fusion_hidden=128, gru_hidden=gh)
        self.feat = feat  # V8 (section 25): "whiten" (fixed map, set by fit_whitening) or "skip" (observation to heads)
        hin = gh + (13 if feat == "skip" else 0)
        self.register_buffer("feat_mu", torch.zeros(gh))
        self.register_buffer("feat_W", torch.eye(gh))
        if hh > 0:
            self.actor_trunk = nn.Sequential(nn.Linear(hin, hh), nn.ELU())
            self.critic_trunk = nn.Sequential(nn.Linear(hin, hh), nn.ELU())
        else:  # V8 (section 25): linear heads, the readout of Proposition 1
            self.actor_trunk, self.critic_trunk, hh = nn.Identity(), nn.Identity(), hin
        self.actor_mu = nn.Linear(hh, 4)
        self.critic_v = nn.Linear(hh, 1)

    def forward(self, x):
        h, _ = self.backbone(x)
        hin = (h - self.feat_mu) @ self.feat_W.T if self.feat == "whiten" else h
        if self.feat == "skip":
            hin = torch.cat([hin, x], dim=-1)
        return h, self.actor_mu(self.actor_trunk(hin)), self.critic_v(self.critic_trunk(hin)).squeeze(-1)

    @torch.no_grad()
    def fit_whitening(self, stats_path):
        """The map of the RL runs of V8a: backbone output on 4,096 windows of the v3 data (every 4th fixed-seed window
        of pace_regularizers), first 8 steps dropped, W = (S + eps I)^(-1/2), eps = 0.01 tr(S) / n."""
        from aerial_gym.rl_training.rl_games.pace_regularizers import load_windows
        xs, _, _ = load_windows(torch.load(stats_path, map_location="cpu"))
        dev = next(self.parameters()).device
        h = self.backbone(xs[::4].float().to(dev))[0][:, 8:, :].reshape(-1, self.feat_mu.numel()).double().cpu()
        mu = h.mean(0)
        S = torch.cov((h - mu).T)
        eps = 0.01 * torch.trace(S) / S.shape[0]
        lam, V = torch.linalg.eigh(S)
        self.feat_mu.copy_(mu.float().to(dev))
        self.feat_W.copy_((V @ torch.diag((lam.clamp(min=0) + eps) ** -0.5) @ V.T).float().to(dev))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPISODES))
    ap.add_argument("--init", required=True, choices=["random", "MAE", "PIDM"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["Frozen", "PartialFT", "FullFT"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1024)
    # M1 and M2 (pre-registration v2, section 22): head width, and the data regime of the supervised training
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--regime", choices=["iid", "noniid", "nonstat"], default="iid")
    # V8 (pre-registration v2, section 25): whitened features or the observation at the heads; --head-hidden 0 = linear
    ap.add_argument("--feat", choices=["none", "whiten", "skip"], default="none")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    z = np.load(D.L / "s19" / f"teacher_{a.task}.npz")
    obs, act, val, done = z["obs"], z["act"], z["value"], z["done"]
    T = obs.shape[0]
    t_split = T - T // EPISODES[a.task]  # the last episode is the test set
    tr, te = windows(done, 0, t_split), windows(done, t_split, T)
    vm, vs = float(val[:t_split].mean()), float(val[:t_split].std() + 1e-6)
    obs_t, act_t = torch.from_numpy(obs).to(dev), torch.from_numpy(act).to(dev)
    val_t = torch.from_numpy((val - vm) / vs).to(dev)
    phase_labels = None
    if a.regime == "nonstat":  # M2: moving targets, labels of the checkpoints 100, ..., 600 in six phases
        rz = np.load(D.L / "s19" / f"relabel_{a.task}.npz")
        phase_labels = []
        for k in (100, 200, 300, 400, 500, 600):
            v_k = rz[f"value_{k}"]
            m_k, s_k = float(v_k[:t_split].mean()), float(v_k[:t_split].std() + 1e-6)
            phase_labels.append((torch.from_numpy(rz[f"act_{k}"]).to(dev), torch.from_numpy((v_k - m_k) / s_k).to(dev)))
    ok32 = None
    if a.regime == "noniid":  # M2: minibatches of 32 vehicles x 32 consecutive, overlapping window starts
        Tw = t_split - WIN + 1
        valid = np.zeros((Tw, obs.shape[1]), bool)
        valid[tr[:, 0], tr[:, 1]] = True
        c = np.concatenate([np.zeros((1, valid.shape[1]), int), np.cumsum(valid, 0)])
        ok32 = (c[32:] - c[:-32]) == 32  # [Tw - 31, N]: starts t0 with 32 valid consecutive windows
    ar = torch.arange(WIN, device=dev)

    def batch(idx, labels=None):
        t = torch.from_numpy(idx[:, 0]).to(dev)[:, None] + ar[None]
        n = torch.from_numpy(idx[:, 1]).to(dev)[:, None]
        A_, V_ = (act_t, val_t) if labels is None else labels
        return obs_t[t, n], A_[t, n], V_[t, n]

    def sample():
        if ok32 is None:
            return tr[rng.integers(0, len(tr), size=a.batch)]
        tt, nn_ = np.nonzero(ok32)
        pick = rng.choice(len(tt), size=a.batch // 32, replace=False)
        starts = (tt[pick][:, None] + np.arange(32)[None]).reshape(-1)
        return np.stack([starts, np.repeat(nn_[pick], 32)], 1)

    net = Student(hh=a.head_hidden, feat=a.feat).to(dev)
    if a.init != "random":
        ck = torch.load(f"{CKPT}/P_{a.init}__seed{a.seed}__srcmixed__backbone.pt", map_location="cpu")
        net.backbone.load_state_dict(ck["backbone_state_dict"])
    if a.feat == "whiten":
        net.fit_whitening(STATS)
    frozen = {"Frozen": ("proprio_mlp", "fusion_mlp", "gru"), "PartialFT": ("proprio_mlp", "fusion_mlp"),
              "FullFT": ()}[a.mode]
    for n_, p in net.backbone.named_parameters():
        p.requires_grad = not n_.startswith(frozen)
    te_sel = te[rng.choice(len(te), size=min(8192, len(te)), replace=False)]

    def feats_and_eval():
        net.eval()
        F, P, V, Y, U = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(te_sel), 2048):
                x, y, u = batch(te_sel[i:i + 2048])
                h, mu, v = net(x)
                F.append(h[:, -1].cpu().numpy())
                P.append(mu[:, WARM:].reshape(-1, 4).cpu().numpy())
                Y.append(y[:, WARM:].reshape(-1, 4).cpu().numpy())
                V.append(v[:, WARM:].reshape(-1).cpu().numpy())
                U.append(u[:, WARM:].reshape(-1).cpu().numpy())
        net.train()
        return np.concatenate(F), np.concatenate(P), np.concatenate(Y), np.concatenate(V), np.concatenate(U)

    f0 = feats_and_eval()[0]
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4)
    t0, hist = time.time(), []
    total = a.epochs * a.steps
    step = 0
    for ep in range(a.epochs):
        tot = 0.0
        for _ in range(a.steps):
            lab = None if phase_labels is None else phase_labels[min(5, step * 6 // total)]
            x, y, u = batch(sample(), lab)
            step += 1
            _, mu, v = net(x)
            loss = ((mu[:, WARM:] - y[:, WARM:]) ** 2).mean() + ((v[:, WARM:] - u[:, WARM:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item()
        hist.append(tot / a.steps)
    f1, P, Y, V, U = feats_and_eval()
    row = dict(task=a.task, init=a.init, seed=a.seed, mode=a.mode, head_hidden=a.head_hidden, regime=a.regime, feat=a.feat,
               n_train_windows=len(tr), n_test_windows=len(te_sel),
               r2_action=float(np.mean(r2(P, Y))), r2_value=float(r2(V[:, None], U[:, None])[0]),
               cka_drift=cka(f0, f1), final_train_loss=hist[-1], seconds=round(time.time() - t0, 1),
               loss_history=json.dumps([round(x, 5) for x in hist]))
    # checkpoint in the RL format for the closed-loop evaluation
    st = torch.load(STATS)
    sd = {"a2c_network.fixed_obs_mean": torch.as_tensor(st["state_mean"], dtype=torch.float32),
          "a2c_network.fixed_obs_std": torch.as_tensor(st["state_std"], dtype=torch.float32),
          "a2c_network.policy.log_std": torch.zeros(4),
          "value_mean_std.running_mean": torch.tensor([vm]), "value_mean_std.running_var": torch.tensor([vs ** 2]),
          "value_mean_std.count": torch.tensor(1.0)}
    for k, v in net.backbone.state_dict().items():
        sd[f"a2c_network.policy.{k}"] = v.detach().cpu()
    for mod in ("actor_trunk", "actor_mu", "critic_trunk", "critic_v"):
        for k, v in getattr(net, mod).state_dict().items():
            sd[f"a2c_network.policy.{mod}.{k}"] = v.detach().cpu()
    name = f"s19__{a.task}__{a.init}__{a.mode}__seed{a.seed}"
    m12 = a.head_hidden != 64 or a.regime != "iid"
    v8 = a.feat != "none" or a.head_hidden == 0
    if m12 or v8:  # section 22 students go to their own folder and table; section 25 (V8) students to theirs
        name += f"__h{a.head_hidden}__{a.regime}" + (f"__{a.feat}" if v8 else "")
    out_dir = OUT_DIR.parent / ("v8" if v8 else "m12") if (m12 or v8) else OUT_DIR
    out_csv = OUT_CSV.parent / ("v8_students.csv" if v8 else "m12_students.csv") if (m12 or v8) else OUT_CSV
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": sd, "s19": row}, out_dir / f"{name}.pth")
    row["student"] = name
    old = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(out_csv, index=False)
    print(f"{name}: R2 action {row['r2_action']:.3f} value {row['r2_value']:.3f} CKA {row['cka_drift']:.3f} "
          f"({row['seconds']} s)", flush=True)

if __name__ == "__main__":
    main()
