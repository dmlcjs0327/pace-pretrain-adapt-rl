"""Merge the v3 flight data (normal-start episodes first, then upset-start episodes) into one file per airframe
and write the standardization statistics shared by pretraining and the fixed input normalization of RL.

Output: data/v3/ssl_v3_{x500,lmf2}.h5 (group = robot name; states, actions, dones; row = t * 4096 + vehicle;
2,700 time steps = 9 episodes of 300 steps) and data/v3/state_stats.pt (state/action mean and std over both
airframes, all rows, float64 accumulation, std + 1e-6).
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import h5py
import numpy as np
import torch

D = _DATA_ROOT + "/v3/"
PARTS = {"x500": ("x500", ["x500_normal.h5", "x500_upset.h5"]),
         "lmf2": ("lmf2_proprio_clean", ["lmf2_normal.h5", "lmf2_upset.h5"])}
allS, allA = [], []
for name, (grp, files) in PARTS.items():
    S, A, Dn = [], [], []
    for fn in files:
        with h5py.File(D + fn, "r") as f:
            S.append(f[grp]["states"][:]); A.append(f[grp]["actions"][:]); Dn.append(f[grp]["dones"][:])
    S, A, Dn = np.concatenate(S), np.concatenate(A), np.concatenate(Dn)
    assert S.shape[0] == 2700 * 4096 and Dn.reshape(2700, 4096)[299::300].all() and Dn.sum() == 9 * 4096
    with h5py.File(D + f"ssl_v3_{name}.h5", "w") as f:
        g = f.create_group(grp)
        g.create_dataset("states", data=S); g.create_dataset("actions", data=A); g.create_dataset("dones", data=Dn)
        f.attrs.update(dict(total_steps=S.shape[0], num_envs=4096, time_steps=2700, episode_len=300,
                            layout="episodes 1-5 normal start, 6-9 upset start (recovery-task initial states)",
                            state_layout="0:3=goal_rel_pos (target-pos, world), 3:7=quat xyzw, 7:10=body_linvel, 10:13=body_angvel",
                            action_layout="lee_attitude_control command [a0, roll, pitch, yaw_rate], clipped to [-1,1]"))
    allS.append(S.astype(np.float64)); allA.append(A.astype(np.float64))
    print(name, S.shape, "written")
S, A = np.concatenate(allS), np.concatenate(allA)
st = {"state_mean": S.mean(0).tolist(), "state_std": (S.std(0) + 1e-6).tolist(),
      "action_mean": A.mean(0).tolist(), "action_std": (A.std(0) + 1e-6).tolist(), "n_rows": int(S.shape[0]),
      "sources": ["x500", "lmf2_proprio_clean"]}
torch.save(st, D + "state_stats.pt")
np.set_printoptions(precision=3, suppress=True)
print("state mean", np.array(st["state_mean"])); print("state std ", np.array(st["state_std"]))
print("action mean", np.array(st["action_mean"])); print("action std ", np.array(st["action_std"]))
