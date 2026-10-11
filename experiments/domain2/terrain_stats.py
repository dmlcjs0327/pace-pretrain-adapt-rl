"""Fixed input normalization of the second domain after section 29 (V17): the mean and standard deviation of the 48
policy inputs and the 12 actions over the pretraining part of domain2/data/terrain_plane.npz and terrain_rough.npz
(held-out robots excluded). Every policy of stages 40 and 41, random ones included, scales its input with these values.
Output: domain2/data/terrain_stats.pt (same keys as domain2/data/stats.pt). Usage: python terrain_stats.py
"""
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    s_sum = s_sq = a_sum = a_sq = None
    n = 0
    for k in ("plane", "rough"):
        p = np.load(os.path.join(HERE, "data", f"terrain_{k}.npz"))
        keep = np.setdiff1d(np.arange(p["resets"].shape[1]), p["hold_idx"])
        S = p["states"][:, keep].reshape(-1, 48).astype(np.float64)
        A = p["actions"][:, keep].reshape(-1, 12).astype(np.float64)
        s_sum = S.sum(0) if s_sum is None else s_sum + S.sum(0)
        s_sq = (S ** 2).sum(0) if s_sq is None else s_sq + (S ** 2).sum(0)
        a_sum = A.sum(0) if a_sum is None else a_sum + A.sum(0)
        a_sq = (A ** 2).sum(0) if a_sq is None else a_sq + (A ** 2).sum(0)
        n += len(S)
    sm, am = s_sum / n, a_sum / n
    ss, as_ = np.sqrt(np.maximum(s_sq / n - sm ** 2, 0)) + 1e-6, np.sqrt(np.maximum(a_sq / n - am ** 2, 0)) + 1e-6
    torch.save({"state_mean": sm.tolist(), "state_std": ss.tolist(), "action_mean": am.tolist(), "action_std": as_.tolist()},
               os.path.join(HERE, "data", "terrain_stats.pt"))
    print(f"terrain stats over {n} states; input std min {ss.min():.4f} max {ss.max():.4f}")

if __name__ == "__main__":
    main()
