"""Task-agnostic reward-free flight data v3ag (pre-registration v2, section 25, V3).

Same as collect_offline_trajectory_v3.py (attitude interface of RL, world-frame outer loop with K_V = 2, OU noise on
the command with sigma 0.1 and theta 0.15, 300-step episodes with all vehicles reset together, the same state layout),
except that the flight is not directed at any goal:
  - The desired velocity is an Ornstein-Uhlenbeck process in the world frame, per axis
        v <- v - THETA_V v + SIGMA_V sqrt(2 THETA_V) N(0, 1),
    THETA_V = 0.01 per step (a time scale of 1 s), stationary standard deviation 2.0 m/s horizontally and 0.7 m/s
    vertically, speed limited to 5 m/s (the largest cruise speed of v3). It starts from its stationary distribution.
  - An altitude guard adds 2 (1 - z) m/s below 1 m and 2 (4.5 - z) m/s above 4.5 m, so the vehicles stay within the
    height range of v3 (0.5-5 m).
  - The goal-relative entries of the state (0:3) hold a point drawn per episode and vehicle from the ranges of v3
    (x, y in [-5, 5] m, z in [0.5, 5] m). The vehicle never flies to it; it only keeps the input layout of RL.
  - No upset starts: every episode starts from the normal initial states.
  - The yaw rate is drawn from [-1, 1] rad/s and redrawn every 50-200 steps, as in v3.
Usage: python collect_offline_trajectory_v3ag.py --robot x500 --episodes 9 --out <h5> --seed 0
"""

import isaacgym  # must be before torch

import argparse
import os
import time

import h5py
import numpy as np
import torch

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("collect_offline_trajectory_v3ag")

WAYPOINT_HORIZON_MIN, WAYPOINT_HORIZON_MAX = 50, 200
POS_RANGE, HEIGHT_RANGE = (-5.0, 5.0), (0.5, 5.0)
YAWRATE_RANGE = (-1.0, 1.0)
OU_SIGMA, OU_THETA = 0.1, 0.15
K_V = 2.0
EPISODE_LEN = 300
G = 9.81
THETA_V = 0.01
SIGMA_V = (2.0, 2.0, 0.7)
VMAX = 5.0
Z_LOW, Z_HIGH, K_Z = 1.0, 4.5, 2.0

def yaw_of(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def attitude_command(v_des, vel_world, quat, yawrate):
    """Command [a0, roll, pitch, yaw rate] of lee_attitude_control that tracks the desired velocity (as in v3)."""
    acc = K_V * (v_des - vel_world)
    f = acc.clone()
    f[:, 2] += G
    fn = f.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    a0 = fn[:, 0] / G - 1.0
    psi = yaw_of(quat)
    c, s = torch.cos(psi), torch.sin(psi)
    b = f / fn
    bx, by = c * b[:, 0] + s * b[:, 1], -s * b[:, 0] + c * b[:, 1]
    roll = -torch.asin(by.clamp(-1, 1))
    pitch = torch.atan2(bx, b[:, 2])
    return torch.stack([a0, roll, pitch, yawrate[:, 0]], dim=-1)

def collect(robot, episodes, num_envs, device, headless):
    env = SimBuilder().build_env(sim_name="base_sim", env_name="empty_env", robot_name=robot,
                                 controller_name="lee_attitude_control", args=None, device=device,
                                 num_envs=num_envs, headless=headless, use_warp=False)
    steps = episodes * EPISODE_LEN
    S = np.zeros((steps * num_envs, 13), dtype=np.float32)
    A = np.zeros((steps * num_envs, 4), dtype=np.float32)
    Dn = np.zeros((steps * num_envs,), dtype=bool)
    diag = {"cos_vel_goal": [], "tilt_deg": [], "sat": [], "low": [], "speed": [], "height": []}
    sig = torch.tensor(SIGMA_V, device=device)
    t0 = time.time()
    for ep in range(episodes):
        env.reset()
        point = torch.empty(num_envs, 3, device=device)
        point[:, 0:2] = torch.empty(num_envs, 2, device=device).uniform_(*POS_RANGE)
        point[:, 2] = torch.empty(num_envs, device=device).uniform_(*HEIGHT_RANGE)
        v_ou = sig * torch.randn(num_envs, 3, device=device)
        next_resample = torch.randint(WAYPOINT_HORIZON_MIN, WAYPOINT_HORIZON_MAX, (num_envs,), device=device)
        yawrate = torch.empty(num_envs, 1, device=device).uniform_(*YAWRATE_RANGE)
        ou = torch.zeros(num_envs, 4, device=device)
        for k in range(EPISODE_LEN):
            gtd = env.global_tensor_dict
            expired = k >= next_resample
            if expired.any():
                yawrate = torch.where(expired[:, None], torch.empty(num_envs, 1, device=device).uniform_(*YAWRATE_RANGE), yawrate)
                next_resample = torch.where(expired, k + torch.randint(WAYPOINT_HORIZON_MIN, WAYPOINT_HORIZON_MAX, (num_envs,), device=device), next_resample)
            v_ou = v_ou - THETA_V * v_ou + sig * np.sqrt(2 * THETA_V) * torch.randn_like(v_ou)
            pos, quat = gtd["robot_position"], gtd["robot_orientation"]
            v_des = v_ou.clone()
            z = pos[:, 2]
            v_des[:, 2] += K_Z * torch.clamp(Z_LOW - z, min=0.0) - K_Z * torch.clamp(z - Z_HIGH, min=0.0)
            sp = v_des.norm(dim=-1, keepdim=True)
            v_des = torch.where(sp > VMAX, v_des / sp * VMAX, v_des)
            cmd = attitude_command(v_des, gtd["robot_linvel"], quat, yawrate)
            ou = ou * (1 - OU_THETA) + OU_SIGMA * torch.randn_like(ou)
            act = torch.clamp(cmd + ou, -1.0, 1.0)
            state = torch.cat([point - pos, quat, gtd["robot_body_linvel"], gtd["robot_body_angvel"]], dim=-1)
            row = (ep * EPISODE_LEN + k) * num_envs
            S[row:row + num_envs] = state.cpu().numpy()
            A[row:row + num_envs] = act.cpu().numpy()
            Dn[row:row + num_envs] = k == EPISODE_LEN - 1
            if k % 10 == 0:
                g = point - pos
                v = gtd["robot_linvel"]
                cosv = (g * v).sum(-1) / (g.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-6)
                upz = 1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
                diag["cos_vel_goal"].append(float(cosv[v.norm(dim=-1) > 0.3].mean()))
                diag["tilt_deg"].append(float(torch.rad2deg(torch.acos(upz.clamp(-1, 1))).mean()))
                diag["sat"].append(float((act.abs() >= 0.999).float().mean()))
                diag["low"].append(float((pos[:, 2] < 0.2).float().mean()))
                diag["speed"].append(float(v.norm(dim=-1).mean()))
                diag["height"].append(float(pos[:, 2].mean()))
            env.step(actions=act)
        logger.info(f"[{robot}] episode {ep + 1}/{episodes} done, {time.time() - t0:.0f}s")
    summary = {k: float(np.mean(v)) for k, v in diag.items()}
    logger.info(f"[{robot}] diagnostics {summary}")
    return S, A, Dn, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=["x500", "lmf2_proprio_clean"])
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    S, A, Dn, summary = collect(a.robot, a.episodes, a.num_envs, a.device, True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        g = f.create_group(a.robot)
        g.create_dataset("states", data=S)
        g.create_dataset("actions", data=A)
        g.create_dataset("dones", data=Dn)
        f.attrs.update(dict(total_steps=S.shape[0], num_envs=a.num_envs, time_steps=a.episodes * EPISODE_LEN,
                            episode_len=EPISODE_LEN, robot_meta=a.robot, upset_start=False, seed=a.seed,
                            layout="v3ag (section 25): task-agnostic OU velocity commands, no goal, no upset starts",
                            state_layout="0:3=fixed unpursued point - pos (world), 3:7=quat xyzw, 7:10=body_linvel, 10:13=body_angvel",
                            action_layout="lee_attitude_control command [a0 (thrust=(a0+1)mg), roll rad, pitch rad, yaw_rate rad/s], clipped to [-1,1]",
                            **{f"diag_{k}": v for k, v in summary.items()}))
    print("wrote", a.out, S.shape, summary)

if __name__ == "__main__":
    main()
