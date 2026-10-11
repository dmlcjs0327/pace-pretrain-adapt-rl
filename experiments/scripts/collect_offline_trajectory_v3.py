"""Reward-free flight data v3 for Study 2.

Changes from v2 (review document experiments/configs/STUDY2_REVIEW_20261001_ko.md, section 1.3-1.6):
  1) The vehicle is flown through the SAME interface as RL: lee_attitude_control, whose command is
     [a0, roll, pitch, yaw rate] with thrust (a0 + 1) m g, roll and pitch in rad, yaw rate in rad/s, and every
     entry clipped to [-1, 1] as rl-games clips the RL action. The recorded command is the executed command.
  2) A scripted outer loop computes the command correctly in the world frame: desired velocity towards the
     waypoint (v2 patterns and cruise speeds), desired acceleration a = K_V (v_des - v_world), thrust vector
     m (a + g e_z), and roll and pitch that point the body z-axis along it for the current yaw
     (R = Rz(yaw) Ry(pitch) Rx(roll), as quat_from_euler_xyz in the attitude controller).
     v2 passed world-frame velocity commands to lee_velocity_control, which reads them in the yaw frame.
  3) Part of the episodes start from the upset states of the recovery task (robot '<name>_recovery': tilt up to
     60 deg, body rates up to 4 rad/s, velocity up to 2 m/s), so the data cover the RL state range better.
  4) Ornstein-Uhlenbeck noise is added in command units (sigma 0.1 per entry, theta 0.15) before clipping.
The state layout is unchanged: 0:3 goal_rel_pos (target - position, world frame), 3:7 quaternion xyzw,
7:10 body linear velocity, 10:13 body angular velocity. Episodes last 300 steps (3 s at 100 Hz) and all
vehicles reset together, so windows never cross an episode.
Usage: python collect_offline_trajectory_v3.py --robot x500 --episodes 5 --out <h5> [--upset]
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

logger = CustomLogger("collect_offline_trajectory_v3")

WAYPOINT_HORIZON_MIN, WAYPOINT_HORIZON_MAX = 50, 200
POS_RANGE, HEIGHT_RANGE, SPEED_RANGE = (-5.0, 5.0), (0.5, 5.0), (0.5, 5.0)
YAWRATE_RANGE = (-1.0, 1.0)
OU_SIGMA, OU_THETA = 0.1, 0.15
K_V = 2.0
EPISODE_LEN = 300
PATTERN_PROBS = {"random": 0.40, "sinusoidal": 0.20, "spiral": 0.20, "step": 0.20}
G = 9.81

def assign_patterns(num_envs, device):
    names = list(PATTERN_PROBS)
    p = np.array([PATTERN_PROBS[n] for n in names])
    return torch.from_numpy(np.random.choice(len(names), size=num_envs, p=p / p.sum())).to(device)

def sample_speed(n, device):
    return torch.empty(n, 1, device=device).uniform_(*SPEED_RANGE)

def sample_target(n, t, device, pattern_ids, phase):
    """The four waypoint patterns of v2 (sample_target_for_pattern), unchanged."""
    tgt = torch.empty(n, 3, device=device)
    tgt[:, 0:2] = torch.empty(n, 2, device=device).uniform_(*POS_RANGE)
    tgt[:, 2] = torch.empty(n, device=device).uniform_(*HEIGHT_RANGE)
    tt = t * 0.01
    m = pattern_ids == 1
    if m.any():
        tgt[m, 0] = 3.0 * torch.sin(0.5 * tt + phase[m])
        tgt[m, 1] = 3.0 * torch.cos(0.5 * tt + phase[m])
        tgt[m, 2] = 2.0 + 1.0 * torch.sin(0.3 * tt + phase[m])
    m = pattern_ids == 2
    if m.any():
        radius = 1.0 + 2.0 * (torch.sin(0.1 * tt + phase[m]) ** 2)
        ang = 0.7 * tt + phase[m]
        tgt[m, 0] = radius * torch.cos(ang)
        tgt[m, 1] = radius * torch.sin(ang)
        tgt[m, 2] = 1.5 + 1.5 * torch.sin(0.2 * tt + phase[m])
    return tgt

def yaw_of(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def outer_loop(target, pos, vel_world, quat, cruise, yawrate):
    """Command [a0, roll, pitch, yaw rate] of lee_attitude_control that tracks the desired velocity."""
    diff = target - pos
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=0.1)
    v_des = diff / dist * torch.minimum(dist, cruise)
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
    diag = {"cos_vel_goal": [], "tilt_deg": [], "sat": [], "low": []}
    t0 = time.time()
    for ep in range(episodes):
        env.reset()
        pattern_ids = assign_patterns(num_envs, device)
        phase = torch.empty(num_envs, device=device).uniform_(0, 2 * np.pi)
        cruise = sample_speed(num_envs, device)
        next_resample = torch.randint(WAYPOINT_HORIZON_MIN, WAYPOINT_HORIZON_MAX, (num_envs,), device=device)
        target = sample_target(num_envs, 0, device, pattern_ids, phase)
        yawrate = torch.empty(num_envs, 1, device=device).uniform_(*YAWRATE_RANGE)
        ou = torch.zeros(num_envs, 4, device=device)
        for k in range(EPISODE_LEN):
            gtd = env.global_tensor_dict
            expired = k >= next_resample
            if expired.any():
                target = torch.where(expired[:, None], sample_target(num_envs, k, device, pattern_ids, phase), target)
                cruise = torch.where(expired[:, None], sample_speed(num_envs, device), cruise)
                yawrate = torch.where(expired[:, None], torch.empty(num_envs, 1, device=device).uniform_(*YAWRATE_RANGE), yawrate)
                next_resample = torch.where(expired, k + torch.randint(WAYPOINT_HORIZON_MIN, WAYPOINT_HORIZON_MAX, (num_envs,), device=device), next_resample)
            tv = (pattern_ids == 1) | (pattern_ids == 2)
            if tv.any():
                target = torch.where(tv[:, None], sample_target(num_envs, k, device, pattern_ids, phase), target)
            pos, quat = gtd["robot_position"], gtd["robot_orientation"]
            cmd = outer_loop(target, pos, gtd["robot_linvel"], quat, cruise, yawrate)
            ou = ou * (1 - OU_THETA) + OU_SIGMA * torch.randn_like(ou)
            act = torch.clamp(cmd + ou, -1.0, 1.0)
            state = torch.cat([target - pos, quat, gtd["robot_body_linvel"], gtd["robot_body_angvel"]], dim=-1)
            row = (ep * EPISODE_LEN + k) * num_envs
            S[row:row + num_envs] = state.cpu().numpy()
            A[row:row + num_envs] = act.cpu().numpy()
            Dn[row:row + num_envs] = k == EPISODE_LEN - 1
            if k % 10 == 0:
                g = target - pos
                v = gtd["robot_linvel"]
                cosv = (g * v).sum(-1) / (g.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-6)
                upz = 1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
                diag["cos_vel_goal"].append(float(cosv[v.norm(dim=-1) > 0.3].mean()))
                diag["tilt_deg"].append(float(torch.rad2deg(torch.acos(upz.clamp(-1, 1))).mean()))
                diag["sat"].append(float((act.abs() >= 0.999).float().mean()))
                diag["low"].append(float((pos[:, 2] < 0.2).float().mean()))
            env.step(actions=act)
        logger.info(f"[{robot}] episode {ep + 1}/{episodes} done, {time.time() - t0:.0f}s")
    summary = {k: float(np.mean(v)) for k, v in diag.items()}
    logger.info(f"[{robot}] diagnostics {summary}")
    return S, A, Dn, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=["base_quadrotor", "x500", "lmf2_proprio_clean"])
    ap.add_argument("--upset", action="store_true", help="start from the upset states of the recovery task")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    robot = a.robot + ("_recovery" if a.upset else "")
    S, A, Dn, summary = collect(robot, a.episodes, a.num_envs, a.device, True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        g = f.create_group(a.robot)
        g.create_dataset("states", data=S)
        g.create_dataset("actions", data=A)
        g.create_dataset("dones", data=Dn)
        f.attrs.update(dict(total_steps=S.shape[0], num_envs=a.num_envs, state_dim=13, action_dim=4, robot_meta=robot,
                            state_layout="0:3=goal_rel_pos (target-pos, world), 3:7=quat xyzw, 7:10=body_linvel, 10:13=body_angvel",
                            action_layout="lee_attitude_control command [a0 (thrust=(a0+1)mg), roll rad, pitch rad, yaw_rate rad/s], clipped to [-1,1]",
                            episode_len=EPISODE_LEN, upset_start=bool(a.upset), seed=a.seed, **{f"diag_{k}": v for k, v in summary.items()}))
    print("wrote", a.out, S.shape, summary)

if __name__ == "__main__":
    main()
