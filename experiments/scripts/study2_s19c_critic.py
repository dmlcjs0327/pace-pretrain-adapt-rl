"""S19c (pre-registration v2, section 21, exploratory): value error of the RL critic against the best fit from the same
features, for frozen and fully fine-tuned backbones.

Checkpoints: stage 1, iteration 600, init random and MAE, modes FullFT and EncFrozen (Frozen), T1 and T2 on R2 and R3,
seeds 7, 42, 99 (48 policies). Each policy flies 256 vehicles for one episode from the fixed initial states of the
evaluation, with sampled actions as in training (its own action noise). Recorded per step: the GRU output h_t, the
critic value V_t (value normalization undone, in the units of the scaled reward of PPO), and the reward (scaled by the
reward scale 0.1 of PPO). For steps in the first half of the episode, G_t is the discounted return to the end of the
episode (gamma 0.99).
  EV  = 1 - Var(G - V) / Var(G)                               explained variance of the RL critic
  R2  = test R^2 of a head of the critic's shape (128-64-1, ELU) trained on (h_t, G_t) of half of the vehicles,
        tested on the other half (Adam 3e-3, 2,000 full-batch steps)
  rho = (1 - EV) / (1 - R2)                                    error of the RL critic relative to the best fit
Reference: TD(lambda) on fixed linear features can multiply the best approximation error by up to
(1 - lambda gamma) / (1 - gamma) = 5.95 for gamma 0.99, lambda 0.95 (Tsitsiklis and Van Roy, 1997).
Output: logs/study2/analysis/s19c_critic.csv.
Usage: python study2_s19c_critic.py --task state_setpoint_x500_clean   (run from the rl_games folder)
"""
import isaacgym  # noqa: F401  (must precede torch)
import glob

import numpy as np
import pandas as pd
import torch

from aerial_gym.registry.task_registry import task_registry
import study2_drift as D
from study2_eval import Policy, SEED

N = 256
GAMMA, RSCALE = 0.99, 0.1
TASKS = ["state_setpoint_x500_clean", "ss_lmf2_clean", "recovery_x500", "recovery_lmf2"]
OUT = D.L / "analysis" / "s19c_critic.csv"

def rollout(task, sd):
    pol = Policy(sd, task.device)
    pol.sample = True
    W = pol.W
    vm = float(sd["value_mean_std.running_mean"].reshape(-1)[0])
    vs = float(torch.sqrt(sd["value_mean_std.running_var"].reshape(-1)[0] + 1e-5))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    obs = task.reset()[0]["observations"].clone()
    H, V, R, Dn = [], [], [], []
    dones = None
    T = int(task.task_config.episode_len_steps)
    for _ in range(T):
        with torch.no_grad():
            a = pol.act(obs, dones)
            c = torch.nn.functional.elu(pol.h @ W["critic_trunk.0.weight"].T + W["critic_trunk.0.bias"])
            v = (c @ W["critic_v.weight"].T + W["critic_v.bias"]).squeeze(-1) * vs + vm
        H.append(pol.h.clone())
        V.append(v)
        out, rew, term, trunc, _ = task.step(a)
        dones = (term > 0) | (trunc > 0)
        R.append(rew * RSCALE)
        Dn.append(dones.clone())
        obs = out["observations"].clone()
    H, V, R, Dn = torch.stack(H), torch.stack(V), torch.stack(R), torch.stack(Dn)  # [T, N, ...]
    alive = torch.cumprod(torch.cat([torch.ones_like(Dn[:1]), (~Dn[:-1])]).float(), 0)  # first episode only
    G = torch.zeros_like(R)
    acc = torch.zeros(N, device=R.device)
    for t in reversed(range(T)):
        acc = R[t] + GAMMA * acc * (~Dn[t]).float()
        G[t] = acc
    half = T // 2
    m = alive[:half] > 0
    return H[:half][m], V[:half][m], G[:half][m], torch.arange(N, device=R.device).expand(half, N)[m]

def fit_head(h, g, veh):
    tr, te = veh % 2 == 0, veh % 2 == 1
    mu, sd = g[tr].mean(), g[tr].std() + 1e-6
    head = torch.nn.Sequential(torch.nn.Linear(h.shape[1], 64), torch.nn.ELU(), torch.nn.Linear(64, 1)).to(h.device)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    y = (g - mu) / sd
    for _ in range(2000):
        loss = ((head(h[tr]).squeeze(-1) - y[tr]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(h[te]).squeeze(-1)
    return float(1 - ((pred - y[te]) ** 2).mean() / y[te].var())

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    a = ap.parse_args()
    rows = []
    for t in [a.task]:  # one simulation per process
        task = task_registry.make_task(t, seed=SEED, num_envs=N, headless=True, use_warp=False)
        for init in ("random", "MAE"):
            for mode in ("FullFT", "EncFrozen"):
                for seed in (7, 42, 99):
                    rid = f"s2__st1__{t}__{init}__{mode}__seed{seed}__it600"
                    f = glob.glob(f"{D.RUNS}{rid}/nn/last_*_ep_600_rew_*.pth")
                    if not f:
                        print("missing", rid)
                        continue
                    sd = torch.load(f[0], map_location="cpu")["model"]
                    h, v, g, veh = rollout(task, sd)
                    ev = float(1 - (g - v).var() / g.var())
                    torch.manual_seed(seed)
                    r2 = fit_head(h, g, veh)
                    rows.append(dict(task=t, init=init, mode=mode, seed=seed, n=int(len(g)), ev_critic=ev, r2_fit=r2,
                                     rho=(1 - ev) / max(1 - r2, 1e-6), mean_G=float(g.mean()), mean_V=float(v.mean())))
                    print(rows[-1], flush=True)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(old):
        old = old[old.task != a.task]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
