"""Representation drift of the backbone during adaptation (Study 2, RQ4).

For every finished run it loads the policies saved every 25 iterations and computes, against the backbone at
the start of RL,
  cka      linear CKA between the final GRU features on a fixed probe set of 4,096 windows of 33 steps
  dW_enc   relative Frobenius change of the encoder weights, ||W_k - W_0|| / ||W_0||
  dW_gru   the same for the GRU weights
The probe windows are standardized with the pretraining statistics, which in study 2 v2 are also the fixed input
normalization of RL, so the measure reflects the input the policy receives. Low-rank updates are folded into effective weights W0 + BA.
Start of RL: the pretrained backbone file for pretrained runs, and the backbone of the Frozen run with the same
task and seed for random runs (the random initialization is identical across modes, checked on T1xR2 seed 7).
Output: logs/study2/analysis/drift.csv (incremental).
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401
import glob
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

L = Path(_EXP_ROOT + "/logs/study2")
RUNS = _AG_ROOT + "/aerial_gym/rl_training/rl_games/runs/"
CK = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3/"
DATA = _DATA_ROOT + "/"
# v3 flight data (study 2 v2): R2 and R3, plus R1 (not used for pretraining) for the probes if collected.
FILES = [("v3/ssl_v3_x500.h5", "x500"), ("v3/ssl_v3_lmf2.h5", "lmf2_proprio_clean"), ("v3/ssl_v3_baseQuad.h5", "base_quadrotor")]
import os as _os
FILES = [f for f in FILES if _os.path.exists(DATA + f[0])]
dev = "cuda" if torch.cuda.is_available() else "cpu"

def probe():
    rng = np.random.default_rng(1)
    X = []
    for fn, g in FILES:
        with h5py.File(DATA + fn, "r") as f:
            s = f[g]["states"][:2700 * 4096].reshape(2700, 4096, 13)
        ok = np.arange(2700)[(np.arange(2700) % 300) <= 267]
        n = 4096 // len(FILES) + 1
        ts, ns = rng.choice(ok, n), rng.integers(0, 4096, n)
        X.append(np.stack([s[t:t + 33, n] for t, n in zip(ts, ns)]))
    X = np.concatenate(X)[:4096]
    ref = torch.load(CK + "P_MAE__seed7__srcmixed__backbone.pt", map_location="cpu")
    return torch.tensor((X - np.array(ref["state_mean"])) / np.array(ref["state_std"]), dtype=torch.float32, device=dev)

def effective_backbone(sd, prefix="a2c_network.policy."):
    """Returns {name: tensor} for proprio_mlp.{0,2}, fusion_mlp.{0,2} weights/biases and gru weights/biases."""
    out = {}
    for blk in ("proprio_mlp", "fusion_mlp"):
        for li in ("0", "2"):
            base = f"{prefix}{blk}.{li}"
            if f"{base}.weight" in sd:
                out[f"{blk}.{li}.weight"], out[f"{blk}.{li}.bias"] = sd[f"{base}.weight"], sd[f"{base}.bias"]
            else:  # LoRALinear
                W0, b = sd[f"{base}.base.weight"], sd[f"{base}.base.bias"]
                out[f"{blk}.{li}.weight"] = W0 + sd[f"{base}.B.weight"] @ sd[f"{base}.A.weight"]
                out[f"{blk}.{li}.bias"] = b
    for n in ("weight_ih_l0", "weight_hh_l0"):
        k = f"{prefix}gru.{n}"
        if k in sd:
            out[f"gru.{n}"] = sd[k]
        else:
            p = f"{prefix}gru.parametrizations.{n}"
            out[f"gru.{n}"] = sd[f"{p}.original"] + sd[f"{p}.0.B"] @ sd[f"{p}.0.A"]
    for n in ("bias_ih_l0", "bias_hh_l0"):
        out[f"gru.{n}"] = sd[f"{prefix}gru.{n}"]
    return {k: v.float().to(dev) for k, v in out.items()}

def ssl_backbone(path):
    sd = torch.load(path, map_location="cpu")["backbone_state_dict"]
    # ssl ProprioGRUBackbone -> same names as the policy (proprio_mlp, fusion_mlp, gru) after load_ssl_backbone
    m = {}
    for k, v in sd.items():
        m["a2c_network.policy." + k] = v
    return effective_backbone(m)

def features(W, x):
    h = torch.nn.functional.elu(x @ W["proprio_mlp.0.weight"].T + W["proprio_mlp.0.bias"])
    h = torch.nn.functional.elu(h @ W["proprio_mlp.2.weight"].T + W["proprio_mlp.2.bias"])
    h = torch.nn.functional.elu(h @ W["fusion_mlp.0.weight"].T + W["fusion_mlp.0.bias"])
    h = torch.nn.functional.elu(h @ W["fusion_mlp.2.weight"].T + W["fusion_mlp.2.bias"])
    out, _ = torch._VF.gru(h, torch.zeros(1, h.shape[0], 128, device=dev),
                           [W["gru.weight_ih_l0"], W["gru.weight_hh_l0"], W["gru.bias_ih_l0"], W["gru.bias_hh_l0"]],
                           True, 1, 0.0, False, False, True)
    return out[:, -1]

def cka(A, Bm):
    A = A - A.mean(0)
    Bm = Bm - Bm.mean(0)
    return float((A.T @ Bm).norm() ** 2 / ((A.T @ A).norm() * (Bm.T @ Bm).norm()))

def rel(W, W0, keys):
    num = sum(((W[k] - W0[k]) ** 2).sum() for k in keys)
    den = sum((W0[k] ** 2).sum() for k in keys)
    return float(torch.sqrt(num / den))

def main():
    A = L / "analysis"
    out_path = A / "drift.csv"
    old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    have = set(old.run_id) if len(old) else set()
    led = pd.read_csv(L / "ledger.csv")
    led = led[led.exit == 0].drop_duplicates("run_id", keep="last")
    x = probe()
    enc = [f"{b}.{i}.{p}" for b in ("proprio_mlp", "fusion_mlp") for i in ("0", "2") for p in ("weight", "bias")]
    gru = ["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0"]
    rows = []
    for _, r in led.iterrows():
        if r.run_id in have:
            continue
        if r.init == "random":
            fz = glob.glob(f"{RUNS}s2__st{r.stage}__{r.task}__random__EncFrozen__seed{r.seed}__it{r.iters}/nn/last_*_ep_25_rew_*.pth")
            if not fz:
                continue
            W0 = effective_backbone(torch.load(fz[0], map_location="cpu")["model"])
        else:
            W0 = ssl_backbone(f"{CK}P_{r.init}__seed{r.seed}__srcmixed__backbone.pt")
        Z0 = features(W0, x)
        for f in sorted(glob.glob(f"{RUNS}{r.run_id}/nn/last_*_ep_*_rew_*.pth")):
            m = re.search(r"_ep_(\d+)_rew_", f)
            if not m or f.endswith("_.pth"):
                continue
            W = effective_backbone(torch.load(f, map_location="cpu")["model"])
            rows.append(dict(run_id=r.run_id, stage=r.stage, task=r.task, init=r.init, mode=r["mode"], seed=r.seed,
                             iter=int(m.group(1)), cka=cka(Z0, features(W, x)), dW_enc=rel(W, W0, enc), dW_gru=rel(W, W0, gru)))
    d = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    d.to_csv(out_path, index=False)
    print(f"{d.run_id.nunique() if len(d) else 0} runs in drift.csv")
    if len(d):
        print(d[d.iter == d.iter.max()].groupby(["init", "mode"])[["cka", "dW_enc", "dW_gru"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
