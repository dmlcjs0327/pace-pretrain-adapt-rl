"""Fixed-state evaluation of the saved Study 2 policies with their mean action (method, measurement suite).

For one setting (task and airframe), it builds the environment once with 256 vehicles and, for every finished
run of that setting, resets all vehicles from the same random seed (so every policy starts from the same 256
initial states and targets), runs the policy of the chosen iteration deterministically (mean action, clipped to
[-1, 1] as rl-games does), and records the flight metrics of the first episode of every vehicle.
The policy is evaluated from its saved weights with low-rank updates folded in and with the input normalization of
the checkpoint (the fixed statistics in study 2 v2, the running statistics before), clipped to [-5, 5].
The checkpoints of iterations 550, 575, and 600 are evaluated, and the analysis averages them, in line with the
terminal return that averages iterations 541 to 600: a single checkpoint is a noisy snapshot of runs whose return
oscillates (checked on T1 x R2 seed 7, where the evaluation follows the logged return of each checkpoint).
Robustness (pre-registration v2, section 8): --disturb L applies a steady wind, a horizontal world-frame force of
L x 0.2 x m g on each vehicle, with a random direction per vehicle drawn from the fixed seed and held over the
episode; L in {1, 2}. L = 0 is the nominal evaluation. (The one-step impulse disturbance of Aerial Gym moved the
vehicle by only about 0.04 m/s per impulse and was not used.)
Output: logs/study2/analysis/eval_fixed.csv, one row per run, iteration, and disturbance level (means over 256
vehicles).
Usage: python study2_eval.py --task state_setpoint_x500_clean [--iters 550 575 600] [--stage 1]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import glob

import numpy as np
import pandas as pd
import torch

from aerial_gym.registry.task_registry import task_registry
import study2_drift as D
from study2_probes import effective_all

N_ENVS = 256
SEED = 20261001
A = D.L / "analysis"

class Policy:
    def __init__(self, sd, dev):
        self.W = {k: v.to(dev) for k, v in effective_all(sd).items()}
        if "running_mean_std.running_mean" in sd:
            self.m = sd["running_mean_std.running_mean"].float().to(dev)
            self.s = torch.sqrt(sd["running_mean_std.running_var"].float() + 1e-5).to(dev)
        else:  # study 2 v2: fixed input normalization inside the network
            self.m = sd["a2c_network.fixed_obs_mean"].float().to(dev)
            self.s = sd["a2c_network.fixed_obs_std"].float().to(dev)
        self.h = None
        # V8 (pre-registration v2, section 25): whitened backbone output and/or the observation at the heads' input
        fmu = sd.get("a2c_network.policy.feat_mu")
        self.fmu = None if fmu is None else fmu.float().to(dev)
        self.fW = None if fmu is None else sd["a2c_network.policy.feat_W"].float().to(dev)
        self.skip = self.W["actor_trunk.0.weight"].shape[1] > self.W["gru.weight_hh_l0"].shape[1]
        self.std = torch.exp(sd["a2c_network.policy.log_std"].float()).to(dev)
        self.sample = False
        # S20 (pre-registration v2, section 21): expanded input; the environment wrapper of RL standardizes the
        # 13-dimensional state and expands it, and the network uses identity statistics
        self.expand = self.m.numel() == 128
        if self.expand:
            from aerial_gym.ssl_pretrain.expand import expansion_params
            st = torch.load(_DATA_ROOT + "/v3/state_stats.pt")
            self.m13 = torch.as_tensor(st["state_mean"], dtype=torch.float32, device=dev)
            self.s13 = torch.as_tensor(st["state_std"], dtype=torch.float32, device=dev)
            self.EA, self.Eb = expansion_params(dev)

    def act(self, obs, dones=None):
        W = self.W
        if self.expand:
            from aerial_gym.ssl_pretrain.expand import expand
            obs = expand(torch.clamp((obs - self.m13) / self.s13, -5.0, 5.0), self.EA, self.Eb)
        x = torch.clamp((obs - self.m) / self.s, -5.0, 5.0)
        x0 = x
        for blk in ("proprio_mlp", "fusion_mlp"):
            for li in ("0", "2"):
                x = torch.nn.functional.elu(x @ W[f"{blk}.{li}.weight"].T + W[f"{blk}.{li}.bias"])
        if self.h is None:
            self.h = torch.zeros(obs.shape[0], W["gru.weight_hh_l0"].shape[1], device=obs.device)  # GRU width (S17: widths x0.5-x4)
        if dones is not None:
            self.h = self.h * (1.0 - dones.float()).unsqueeze(1)
        self.h = torch._VF.gru_cell(x, self.h, W["gru.weight_ih_l0"], W["gru.weight_hh_l0"], W["gru.bias_ih_l0"], W["gru.bias_hh_l0"])
        hin = self.h if self.fmu is None else (self.h - self.fmu) @ self.fW.T
        if self.skip:
            hin = torch.cat([hin, x0], dim=-1)
        a = torch.nn.functional.elu(hin @ W["actor_trunk.0.weight"].T + W["actor_trunk.0.bias"])
        mu = a @ W["actor_mu.weight"].T + W["actor_mu.bias"]
        if self.sample:
            mu = mu + self.std * torch.randn_like(mu)
        return torch.clamp(mu, -1.0, 1.0)

def evaluate(task, pol):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    obs = task.reset()[0]["observations"].clone()
    task._flight_new_episode = torch.ones(N_ENVS, dtype=torch.bool, device=task.device)
    if hasattr(task, "_flight"):
        task._flight.finish(torch.ones(N_ENVS, dtype=torch.bool, device=task.device), torch.zeros(N_ENVS, device=task.device),
                            torch.zeros(N_ENVS, device=task.device), torch.zeros(N_ENVS, device=task.device))
    first = torch.zeros(N_ENVS, dtype=torch.bool, device=task.device)
    rec, ret = {}, torch.zeros(N_ENVS, device=task.device)
    dones = None
    for _ in range(int(task.task_config.episode_len_steps) + 10):
        with torch.no_grad():
            a = pol.act(obs, dones)
        out, rew, term, trunc, infos = task.step(a)
        obs = out["observations"].clone()
        dones = (term > 0) | (trunc > 0)
        new = dones & ~first
        ret += rew * (~first).float()
        for k, v in infos.items():
            if k.startswith("flight/"):
                rec.setdefault(k[7:], torch.full((N_ENVS,), float("nan"), device=task.device))[new] = v[new].float()
        first |= dones
        if bool(first.all()):
            break
    row = {k: float(torch.nanmean(v)) for k, v in rec.items()}
    row["ret"], row["n_done"] = float(ret.mean()), int(first.sum())
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--iters", type=int, nargs="+", default=[550, 575, 600])
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--sample", action="store_true", help="sampled actions (check against the training return)")
    ap.add_argument("--disturb", type=float, default=0.0, help="robustness level L (0 nominal, 1, 2)")
    ap.add_argument("--runs", nargs="*", help="only these run ids (with --sample: printed, not stored)")
    a = ap.parse_args()
    out_path = A / "eval_fixed.csv"
    old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    if len(old) and "disturb" not in old:
        old["disturb"] = 0.0
    have = set(zip(old.run_id, old.iter, old.disturb)) if len(old) else set()
    led = pd.read_csv(D.L / "ledger.csv")
    led = led[(led.exit == 0) & (led.task == a.task) & (led.stage == a.stage)].drop_duplicates("run_id", keep="last")
    if a.runs:
        led = led[led.run_id.isin(a.runs)]
    todo = [(r, k) for _, r in led.iterrows() for k in a.iters if a.sample or (r.run_id, k, a.disturb) not in have]
    if not todo:
        print("nothing to evaluate")
        return
    task = task_registry.make_task(a.task, seed=SEED, num_envs=N_ENVS, headless=True, use_warp=False)
    if a.disturb > 0:
        from aerial_gym.utils.math import quat_rotate_inverse
        rob = task.sim_env.robot_manager.robot
        mass = float(task.sim_env.robot_manager.robot_mass)
        gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
        ang = torch.rand(N_ENVS, generator=gen) * 2 * np.pi
        wind = torch.zeros(N_ENVS, 3)
        wind[:, 0], wind[:, 1] = torch.cos(ang), torch.sin(ang)
        wind = (wind * a.disturb * 0.2 * mass * 9.81).to(task.device)

        def steady_wind():
            q = task.sim_env.global_tensor_dict["robot_orientation"]
            rob.robot_force_tensors[:, 0, 0:3] += quat_rotate_inverse(q, wind)
        rob.apply_disturbance = steady_wind
        print(f"steady wind level {a.disturb}: mass {mass:.3f} kg, force {wind.norm(dim=1)[0].item():.2f} N")
    rows = []
    for r, k in todo:
        f = glob.glob(f"{D.RUNS}{r.run_id}/nn/last_*_ep_{k}_rew_*.pth")
        if not f:
            continue
        pol = Policy(torch.load(f[0], map_location="cpu")["model"], task.device)
        pol.sample = a.sample
        row = dict(run_id=r.run_id, stage=r.stage, task=r.task, init=r.init, mode=r["mode"], seed=r.seed, iter=k, disturb=a.disturb, **evaluate(task, pol))
        rows.append(row)
        print(f"{r.run_id} it{k} ret={row['ret']:.1f} final_dist={row.get('final_dist_m', float('nan')):.2f} done={row['n_done']}", flush=True)
    if not a.sample:
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
