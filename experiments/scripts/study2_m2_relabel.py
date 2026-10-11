"""M2 (pre-registration v2, section 22): labels of the moving-target condition.

The states of the S19 teacher data (teacher_<task>.npz, stored as the network sees them) are labelled again by the
checkpoints of the same teacher run at iterations 100, 200, 300, 400, 500, and 600: mean action (clipped to [-1, 1]) and
value (value normalization of that checkpoint undone). The recurrent state is reset where an episode ended, as during
the collection. Output: logs/study2/s19/relabel_<task>.npz with act_<k> [T, N, 4] and value_<k> [T, N].
Usage: python study2_m2_relabel.py --task state_setpoint_x500_clean
"""
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import glob

import numpy as np
import torch

import study2_drift as D
from study2_probes import effective_all

ITERS = (100, 200, 300, 400, 500, 600)

def labels(sd, x, done, dev):
    W = {k: v.to(dev) for k, v in effective_all(sd).items()}
    vm = float(sd["value_mean_std.running_mean"].reshape(-1)[0])
    vs = float(torch.sqrt(sd["value_mean_std.running_var"].reshape(-1)[0] + 1e-5))
    T, N, _ = x.shape
    h = torch.zeros(N, W["gru.weight_hh_l0"].shape[1], device=dev)
    A = np.zeros((T, N, 4), np.float32)
    V = np.zeros((T, N), np.float32)
    with torch.no_grad():
        for t in range(T):
            z = torch.from_numpy(x[t]).to(dev)
            for blk in ("proprio_mlp", "fusion_mlp"):
                for li in ("0", "2"):
                    z = torch.nn.functional.elu(z @ W[f"{blk}.{li}.weight"].T + W[f"{blk}.{li}.bias"])
            h = torch._VF.gru_cell(z, h, W["gru.weight_ih_l0"], W["gru.weight_hh_l0"], W["gru.bias_ih_l0"], W["gru.bias_hh_l0"])
            a = torch.nn.functional.elu(h @ W["actor_trunk.0.weight"].T + W["actor_trunk.0.bias"])
            A[t] = torch.clamp(a @ W["actor_mu.weight"].T + W["actor_mu.bias"], -1, 1).cpu().numpy()
            c = torch.nn.functional.elu(h @ W["critic_trunk.0.weight"].T + W["critic_trunk.0.bias"])
            V[t] = ((c @ W["critic_v.weight"].T + W["critic_v.bias"]).squeeze(-1) * vs + vm).cpu().numpy()
            h = h * (1.0 - torch.from_numpy(done[t]).to(dev)).unsqueeze(1)  # reset after an episode end
    return A, V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    z = np.load(D.L / "s19" / f"teacher_{a.task}.npz")
    rid = str(z["teacher"])
    out = {}
    for k in ITERS:
        f = [p for p in glob.glob(f"{D.RUNS}{rid}/nn/last_*_ep_{k}_rew_*.pth") if "rew__" not in p]
        sd = torch.load(f[0], map_location="cpu")["model"]
        out[f"act_{k}"], out[f"value_{k}"] = labels(sd, z["obs"], z["done"], D.dev)
    # consistency check: the final checkpoint reproduces the stored labels of the collection
    print(a.task, "max |act_600 - stored act|", float(np.abs(out["act_600"] - z["act"]).max()),
          "mean |value_600 - stored value|", float(np.abs(out["value_600"] - z["value"]).mean()), flush=True)
    np.savez_compressed(D.L / "s19" / f"relabel_{a.task}.npz", **out)

if __name__ == "__main__":
    main()
