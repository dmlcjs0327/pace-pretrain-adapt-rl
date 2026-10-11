"""M3 (pre-registration v2, section 22): does the critic benefit from pretraining first?

For FullFT runs of stage 1 (random, MAE, AP = PIDM) and stage 5 (shuffled MAE), T1 and T2 on R2 and R3, seeds 7, 42,
99, the checkpoints of iterations 25, 50, and 100 fly 256 vehicles for one episode from the fixed initial states with
sampled actions, as in study2_s19c_critic.py. Reported per checkpoint: the explained variance EV = 1 - Var(G - V) /
Var(G) of the critic against the realized discounted return G (first half of the episode, gamma 0.99, scaled reward),
and the mean episode return. The prediction (direction only): pretrained critics have a higher EV than random ones
at 25 iterations, before the gap in return opens at 50 iterations.
Output: logs/study2/analysis/m3_critic_early.csv (rows of one task are replaced on a rerun).
Usage: python study2_m3_critic_early.py --task state_setpoint_x500_clean   (run from the rl_games folder)
"""
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import glob

import pandas as pd
import torch

from aerial_gym.registry.task_registry import task_registry
import study2_drift as D
from study2_eval import SEED
from study2_s19c_critic import N, rollout

OUT = D.L / "analysis" / "m3_critic_early.csv"
RUNS = [("random", 1), ("MAE", 1), ("PIDM", 1), ("MAEshuf", 5)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    task = task_registry.make_task(a.task, seed=SEED, num_envs=N, headless=True, use_warp=False)
    rows = []
    for init, st in RUNS:
        for seed in (7, 42, 99):
            rid = f"s2__st{st}__{a.task}__{init}__FullFT__seed{seed}__it600"
            for it in (25, 50, 100):
                f = [p for p in glob.glob(f"{D.RUNS}{rid}/nn/last_*_ep_{it}_rew_*.pth") if "rew__" not in p]
                if not f:
                    print("missing", rid, it)
                    continue
                sd = torch.load(f[0], map_location="cpu")["model"]
                h, v, g, veh = rollout(task, sd)
                ev = float(1 - (g - v).var() / g.var())
                rows.append(dict(task=a.task, init=init, seed=seed, iter=it, ev_critic=ev, mean_G=float(g.mean()),
                                 mean_V=float(v.mean())))
                print(rows[-1], flush=True)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(old):
        old = old[old.task != a.task]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
