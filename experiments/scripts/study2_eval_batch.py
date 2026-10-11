"""Batched fixed-state evaluation of the saved Study 2 policies (same measurement as study2_eval.py, faster).

study2_eval.py evaluates one policy at a time on 256 vehicles; most of its time is the fixed cost of a simulation
step, so a full pass over all runs would take more than 50 hours. This script builds the environment with K x 256
vehicles and evaluates K policies at once, policy k flying vehicles [256k, 256k + 256). After every reset (fixed seed),
the floating-point state of the first 256 vehicles (robot root states, targets, controller gains, and the task and
robot buffers) is copied to the other blocks and written to the simulator, so every policy starts from the same 256
initial states and targets, as in study2_eval.py. A steady wind (--disturb L) is drawn for 256 vehicles from the same
fixed seed and tiled over the blocks. The robots of the study have no built-in random disturbance.
Policies run with their mean action (clipped to [-1, 1]); metrics are those of the first episode of every vehicle.
Output: logs/study2/analysis/eval_fixed.csv (same columns as study2_eval.py, plus the evaluator name).
Usage: python study2_eval_batch.py --task state_setpoint_x500_clean [--iters 25 100] [--stage 1] [--disturb 0] [--k 16]
       python study2_eval_batch.py --task state_setpoint_x500_clean --check   (same policy in every block)
"""
import isaacgym  # noqa: F401  (must precede torch)
import argparse
import glob

import numpy as np
import pandas as pd
import torch

from aerial_gym.registry.task_registry import task_registry
import study2_drift as D
from study2_eval import Policy, SEED

N = 256
A = D.L / "analysis"

def tile_(t, K):
    """Copy rows [0, N) of a per-vehicle float tensor to the other K - 1 blocks, in place."""
    if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dim() >= 1 and t.shape[0] == N * K:
        if 0 in t.stride():  # broadcast view of a constant (expand): identical for every vehicle already
            return
        t[N:] = t[:N].repeat(K - 1, *([1] * (t.dim() - 1)))

def _walk(obj, K, seen, depth=0):
    """Tile every per-vehicle float tensor reachable from the Aerial Gym objects of the task (robot, controller,
    control allocator and motor model with their randomized time constants and initial thrusts, task buffers)."""
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    items = obj.items() if isinstance(obj, dict) else (vars(obj).items() if hasattr(obj, "__dict__") else [])
    for _, v in list(items):
        if isinstance(v, torch.Tensor):
            tile_(v, K)
        elif isinstance(v, dict) or (hasattr(v, "__dict__") and not isinstance(v, type)
                                     and type(v).__module__.startswith("aerial_gym")):
            _walk(v, K, seen, depth + 1)

def replicate(task, K):
    _walk(task, K, set())
    task.sim_env.IGE_env.write_to_sim()

def reset_all(task, K):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    task.reset()
    replicate(task, K)
    return task.get_return_tuple()[0]["observations"].clone()

TILT_BINS = (0.0, 20.0, 40.0, 60.0)  # V19 (section 31): initial tilt bins in degrees, --tilt-bins only

def evaluate_batch(task, pols, K, tilt_bins=False):
    NE = N * K
    obs = reset_all(task, K)
    if tilt_bins:  # initial tilt of every vehicle (quaternion x, y, z, w): angle between body z and world z
        q0 = task.sim_env.global_tensor_dict["robot_orientation"].clone()
        tilt0 = torch.rad2deg(torch.acos(torch.clamp(1 - 2 * (q0[:, 0] ** 2 + q0[:, 1] ** 2), -1.0, 1.0)))
    task._flight_new_episode = torch.ones(NE, dtype=torch.bool, device=task.device)
    if hasattr(task, "_flight"):
        task._flight.finish(torch.ones(NE, dtype=torch.bool, device=task.device), torch.zeros(NE, device=task.device),
                            torch.zeros(NE, device=task.device), torch.zeros(NE, device=task.device))
    first = torch.zeros(NE, dtype=torch.bool, device=task.device)
    rec, ret = {}, torch.zeros(NE, device=task.device)
    dones = None
    for p in pols:
        p.h = None
    for _ in range(int(task.task_config.episode_len_steps) + 10):
        with torch.no_grad():
            a = torch.cat([p.act(obs[k * N:(k + 1) * N], None if dones is None else dones[k * N:(k + 1) * N])
                           for k, p in enumerate(pols)])
        out, rew, term, trunc, infos = task.step(a)
        obs = out["observations"].clone()
        dones = (term > 0) | (trunc > 0)
        new = dones & ~first
        ret += rew * (~first).float()
        for key, v in infos.items():
            if key.startswith("flight/"):
                rec.setdefault(key[7:], torch.full((NE,), float("nan"), device=task.device))[new] = v[new].float()
        first |= dones
        if bool(first.all()):
            break
    rows = []
    for k in range(K):
        s = slice(k * N, (k + 1) * N)
        row = {key: float(torch.nanmean(v[s])) for key, v in rec.items()}
        row["ret"], row["n_done"] = float(ret[s].mean()), int(first[s].sum())
        if tilt_bins and "reached_05" in rec:
            for lo, hi in zip(TILT_BINS[:-1], TILT_BINS[1:]):
                m = (tilt0[s] >= lo) & (tilt0[s] < hi if hi < TILT_BINS[-1] else tilt0[s] <= hi + 1e-3)
                row[f"reached_05_tilt{int(lo)}_{int(hi)}"] = float(torch.nanmean(rec["reached_05"][s][m])) if bool(m.any()) else float("nan")
                row[f"n_tilt{int(lo)}_{int(hi)}"] = int(m.sum())
        rows.append(row)
    return rows

def gpu_gate(need=3500):
    """Operations: take the launch lock of the RL workers and wait for free GPU memory before the
    simulation allocates, as the workers do (an evaluation launched without it starved a starting run). The caller
    releases the lock after its first batch has allocated."""
    import fcntl
    import subprocess
    import time
    lockf = open(D.L / "gpu_launch.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    while int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      text=True).split()[0]) < need:
        time.sleep(30)
    return lockf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--iters", type=int, nargs="+", default=[550, 575, 600])
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--disturb", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--check", action="store_true", help="evaluate one policy in every block and print the spread")
    # S19 (pre-registration v2, section 21): evaluate checkpoint files directly (supervised students), written to --out
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--out", default=None)
    # V19 (pre-registration v2, section 31): reach rate by initial tilt bin (0-20, 20-40, 40-60 deg), written to --out
    ap.add_argument("--tilt-bins", action="store_true")
    a = ap.parse_args()
    K = a.k
    out_path = A / (a.out or "eval_fixed.csv")
    old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    have = set(zip(old.run_id, old.iter, old.disturb)) if len(old) else set()
    led = pd.read_csv(D.L / "ledger.csv")
    led = led[(led.exit == 0) & (led.task == a.task) & (led.stage == a.stage)].drop_duplicates("run_id", keep="last")
    if a.runs:
        led = led[led.run_id.isin(a.runs)]
    todo = []
    for _, r in led.iterrows():
        for it in a.iters:
            if a.check or (r.run_id, it, a.disturb) not in have:
                f = glob.glob(f"{D.RUNS}{r.run_id}/nn/last_*_ep_{it}_rew_*.pth")
                if f:
                    todo.append((r, it, f[0]))
    if a.files:
        import os
        done_ids = set(old.run_id) if len(old) else set()
        todo = [(pd.Series(dict(run_id=os.path.basename(f)[:-4], stage=19, task=a.task, init="", mode="", seed=-1)), 0, f)
                for f in a.files if os.path.basename(f)[:-4] not in done_ids]
    if a.check:
        todo = todo[:1]
    if not todo:
        print("nothing to evaluate")
        return
    import fcntl
    lockf = gpu_gate()
    task = task_registry.make_task(a.task, seed=SEED, num_envs=N * K, headless=True, use_warp=False)
    if a.disturb > 0:
        from aerial_gym.utils.math import quat_rotate_inverse
        rob = task.sim_env.robot_manager.robot
        mass = float(task.sim_env.robot_manager.robot_mass)
        gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
        ang = torch.rand(N, generator=gen) * 2 * np.pi
        wind = torch.zeros(N, 3)
        wind[:, 0], wind[:, 1] = torch.cos(ang), torch.sin(ang)
        wind = (wind * a.disturb * 0.2 * mass * 9.81).repeat(K, 1).to(task.device)

        def steady_wind():
            q = task.sim_env.global_tensor_dict["robot_orientation"]
            rob.robot_force_tensors[:, 0, 0:3] += quat_rotate_inverse(q, wind)
        rob.apply_disturbance = steady_wind
    if a.check:
        r, it, f = todo[0]
        pols = [Policy(torch.load(f, map_location="cpu")["model"], task.device) for _ in range(K)]
        rows = evaluate_batch(task, pols, K)
        rets = np.array([x["ret"] for x in rows])
        print(f"check {r.run_id} it{it}: block returns min {rets.min():.3f} max {rets.max():.3f} spread {rets.max() - rets.min():.2e}")
        return
    rows = []
    for i in range(0, len(todo), K):
        chunk = todo[i:i + K]
        pols = [Policy(torch.load(f, map_location="cpu")["model"], task.device) for _, _, f in chunk]
        pols += [pols[-1]] * (K - len(pols))  # unused blocks
        res = evaluate_batch(task, pols, K, tilt_bins=a.tilt_bins)
        if lockf is not None:  # the simulation and the first batch have allocated their memory
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()
            lockf = None
        for (r, it, _), row in zip(chunk, res):
            rows.append(dict(run_id=r.run_id, stage=r.stage, task=r.task, init=r.init, mode=r["mode"], seed=r.seed, iter=it,
                             disturb=a.disturb, evaluator="batch16", **row))
        print(f"{a.task} L{a.disturb}: {min(i + K, len(todo))}/{len(todo)}", flush=True)
    old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
