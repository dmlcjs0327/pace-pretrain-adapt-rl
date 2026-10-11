"""S12: can the scripted controller that collected the pretraining data
fly the tasks? The outer loop of collect_offline_trajectory_v3.py (desired velocity towards the target with a cruise
speed, K_V = 2, thrust and tilt for the attitude interface, yaw rate 0, no noise) acts on the task observation
[target - position, quaternion xyzw, body linear velocity, body angular velocity] through the same
lee_attitude_control interface as RL (commands clipped to [-1, 1]). One block of 256 vehicles per cruise speed,
with the fixed initial states and targets of the batched evaluator (study2_eval_batch.py). No learning.
Output: logs/study2/analysis/expert_eval.csv (one row per task and cruise speed; flight metrics of the first episode).
Usage: python study2_expert_t3.py --task agg_x500 [--speeds 2 3 4 5 6 8]
"""
import isaacgym  # noqa: F401  (must precede torch)
import argparse

import pandas as pd
import torch

from aerial_gym.registry.task_registry import task_registry
import study2_eval_batch as EB
from study2_eval import SEED

G, K_V = 9.81, 2.0

def rotate(q, v):
    """Body-to-world rotation of v by the quaternion q (xyzw)."""
    x, y, z, w = q.unbind(-1)
    r0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    r1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1)
    r2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return torch.stack([(r0 * v).sum(-1), (r1 * v).sum(-1), (r2 * v).sum(-1)], -1)

class Expert:
    """outer_loop of collect_offline_trajectory_v3.py with a fixed cruise speed and zero yaw rate."""

    def __init__(self, cruise):
        self.cruise = float(cruise)
        self.h = None

    def act(self, obs, dones=None):
        diff, q, vb = obs[:, 0:3], obs[:, 3:7], obs[:, 7:10]
        vel_world = rotate(q, vb)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=0.1)
        v_des = diff / dist * torch.clamp(dist, max=self.cruise)
        f = K_V * (v_des - vel_world)
        f[:, 2] += G
        fn = f.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        a0 = fn[:, 0] / G - 1.0
        x, y, z, w = q.unbind(-1)
        psi = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = torch.cos(psi), torch.sin(psi)
        b = f / fn
        bx, by = c * b[:, 0] + s * b[:, 1], -s * b[:, 0] + c * b[:, 1]
        roll = -torch.asin(by.clamp(-1, 1))
        pitch = torch.atan2(bx, b[:, 2])
        cmd = torch.stack([a0, roll, pitch, torch.zeros_like(a0)], dim=-1)
        return torch.clamp(cmd, -1.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--speeds", type=float, nargs="+", default=[2, 3, 4, 5, 6, 8])
    a = ap.parse_args()
    K = len(a.speeds)
    task = task_registry.make_task(a.task, seed=SEED, num_envs=EB.N * K, headless=True, use_warp=False)
    pols = [Expert(v) for v in a.speeds]
    rows = EB.evaluate_batch(task, pols, K)
    out = []
    for v, row in zip(a.speeds, rows):
        out.append(dict(task=a.task, cruise=v, **row))
        print(f"{a.task} cruise {v}: reached_05 {row.get('reached_05', float('nan')):.3f} "
              f"min_dist {row.get('min_dist_m', float('nan')):.2f} crash {row.get('collided', float('nan')):.2f} "
              f"diverged {row.get('diverged', float('nan')):.2f} ret {row['ret']:.1f}", flush=True)
    path = EB.A / "expert_eval.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(out)], ignore_index=True).to_csv(path, index=False)

if __name__ == "__main__":
    main()
