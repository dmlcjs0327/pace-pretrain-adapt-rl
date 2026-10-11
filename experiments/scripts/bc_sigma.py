"""Initial action noise of the behavior-cloning backbone.

The BC policy is a Gaussian with the BC actor as its mean. Its maximum-likelihood standard deviation per command
dimension is the root mean squared error of the BC actor against the recorded commands. This script computes it on
the 16,384 offline windows of the auxiliary losses (pace_regularizers.load_windows, R2 and R3, all 33 steps of each
window, as in the BC pretraining loss) and writes it next to the checkpoint as
P_BC__seed<s>__srcmixed__bc_sigma.json; network_pretrain_adapt sets log_std = log(sigma) when it loads the BC
actor head, and refuses to load the head without this file.
Usage: python bc_sigma.py --seeds 7 42 99
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import json
from pathlib import Path

import torch

from aerial_gym.rl_training.rl_games.network_pretrain_adapt import PretrainAdaptBuilder

CK = Path(_AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3")
ST = _DATA_ROOT + "/v3/state_stats.pt"

def build(ckpt):
    cfg = dict(proprio_dim=13, fusion_hidden=128, gru_hidden=128, use_vae_branch=False, lora_rank=8, fixed_input_norm=ST,
               adapt_mode="FullFT", ssl_backbone_ckpt=str(ckpt), ssl_actor_head=True, reg_mode="bc_reg", reg_lambda=1.0,
               bc_sigma_skip=True)
    b = PretrainAdaptBuilder()
    b.load({"pretrain_adapt": cfg})
    return b.build("bc", actions_num=4, input_shape=(13,), num_seqs=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    for s in a.seeds:
        ck = CK / f"P_BC__seed{s}__srcmixed__backbone.pt"
        net = build(ck)
        pol, reg = net.policy, net.pace_reg
        mods = {k: getattr(pol, k) for k in ("proprio_mlp", "fusion_mlp", "gru")}
        se, n = torch.zeros(4), 0
        with torch.no_grad():
            for i in range(0, reg.xs.shape[0], 1024):
                z = reg._feats(mods, reg.xs[i:i + 1024])
                r = pol.actor_mu(pol.actor_trunk(z)) - reg.ar[i:i + 1024]
                se += (r ** 2).sum(dim=(0, 1))
                n += r.shape[0] * r.shape[1]
        sigma = (se / n).sqrt()
        out = {"sigma": [round(float(v), 6) for v in sigma], "mse_mean": round(float((se / n).mean()), 6),
               "n_windows": int(reg.xs.shape[0]), "steps_per_window": int(reg.xs.shape[1]),
               "definition": "RMSE of the BC actor mean against the recorded commands per command dimension "
                             "(maximum-likelihood std of the Gaussian BC policy)",
               "checkpoint": ck.name}
        p = CK / f"P_BC__seed{s}__srcmixed__bc_sigma.json"
        p.write_text(json.dumps(out, indent=1))
        print(s, out["sigma"], "mean MSE", out["mse_mean"], "->", p.name)

if __name__ == "__main__":
    main()
