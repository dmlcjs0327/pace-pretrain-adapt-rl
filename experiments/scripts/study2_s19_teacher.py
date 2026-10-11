"""S19 (pre-registration v2, section 21): teacher data for supervised adaptation.

The teacher is the confirmation-stage scratch run (random + FullFT, seed 13, stage 2, checkpoint 600) of one T1 or T2
setting. It flies 2,048 vehicles from the training initial-state distribution (fixed seed): two 300-step episodes in
T1, four 150-step episodes in T2. The executed action is the teacher's mean plus Gaussian noise with the teacher's own
standard deviation (DART-style noise injection, so the data cover states around the teacher's path); the recorded
labels are the teacher's mean action (clipped to [-1, 1]) and its value (the critic output, value normalization undone).
Observations are stored as the network sees them (fixed pretraining statistics, clipped to [-5, 5]).
Output: logs/study2/s19/teacher_<task>.npz with obs [T, N, 13], act [T, N, 4], value [T, N], rew [T, N], done [T, N]
(done[t] = 1 when the step t ended an episode), and the teacher run id.
Usage: python study2_s19_teacher.py --task state_setpoint_x500_clean
"""
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import glob
import os

import numpy as np
import torch

from aerial_gym.registry.task_registry import task_registry
import study2_drift as D
from study2_eval import Policy

SEED = 20261006
N_VEH = 2048
EPISODES = {"state_setpoint_x500_clean": 2, "ss_lmf2_clean": 2, "recovery_x500": 4, "recovery_lmf2": 4}
OUT = D.L / "s19"

class TeacherPolicy(Policy):
    """The evaluation policy of study2_eval.py plus the critic output and the policy noise."""

    def __init__(self, sd, dev):
        super().__init__(sd, dev)
        self.vm = float(sd["value_mean_std.running_mean"].reshape(-1)[0])
        self.vs = float(torch.sqrt(sd["value_mean_std.running_var"].reshape(-1)[0] + 1e-5))

    def step(self, obs, dones):
        W = self.W
        x = torch.clamp((obs - self.m) / self.s, -5.0, 5.0)
        mu = self.act(obs, dones)  # updates self.h; mean action clipped to [-1, 1]
        c = torch.nn.functional.elu(self.h @ W["critic_trunk.0.weight"].T + W["critic_trunk.0.bias"])
        v = (c @ W["critic_v.weight"].T + W["critic_v.bias"]).squeeze(-1) * self.vs + self.vm
        a_exec = torch.clamp(mu + self.std * torch.randn_like(mu), -1.0, 1.0)
        return x, mu, v, a_exec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EPISODES))
    a = ap.parse_args()
    rid = f"s2__st2__{a.task}__random__FullFT__seed13__it600"
    f = glob.glob(f"{D.RUNS}{rid}/nn/last_*_ep_600_rew_*.pth")
    if not f:
        raise SystemExit(f"teacher checkpoint missing: {rid}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    task = task_registry.make_task(a.task, seed=SEED, num_envs=N_VEH, headless=True, use_warp=False)
    pol = TeacherPolicy(torch.load(f[0], map_location="cpu")["model"], task.device)
    T = int(task.task_config.episode_len_steps) * EPISODES[a.task]
    obs = task.reset()[0]["observations"].clone()
    buf = {k: [] for k in ("obs", "act", "value", "rew", "done")}
    dones = None
    for _ in range(T):
        with torch.no_grad():
            x, mu, v, a_exec = pol.step(obs, dones)
        out, rew, term, trunc, _ = task.step(a_exec)
        dones = (term > 0) | (trunc > 0)
        for k, val in (("obs", x), ("act", mu), ("value", v), ("rew", rew), ("done", dones.float())):
            buf[k].append(val.detach().float().cpu().numpy())
        obs = out["observations"].clone()
    os.makedirs(OUT, exist_ok=True)
    arr = {k: np.stack(v).astype(np.float32) for k, v in buf.items()}
    np.savez_compressed(OUT / f"teacher_{a.task}.npz", teacher=rid, **arr)
    print(f"{a.task}: {T} steps x {N_VEH} vehicles, episodes ended {int(arr['done'].sum())}, "
          f"mean reward {arr['rew'].mean():.3f}, value mean {arr['value'].mean():.2f}", flush=True)

if __name__ == "__main__":
    main()
