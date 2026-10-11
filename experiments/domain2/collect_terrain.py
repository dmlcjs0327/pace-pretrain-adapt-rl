"""Reward-free data of the second domain after section 29.

The collector is the shipped AnymalTerrain PPO (separate MLPs 512-256-128, its own input normalization, 188 inputs with
the height scan), trained on the shipped rough-terrain task with its terrain curriculum (side_jobs_s29_collector.sh).
It runs with its stochastic policy and the two perturbations of the quadrotor data v3 (pre-registration section 1):
  (1) Ornstein-Uhlenbeck noise in action units (sigma 0.1, theta 0.15) on every robot before clipping;
  (2) for the first 4/9 of the robots, the push of the shipped task (base xy linear velocity set to U(-1, 1) m/s)
      every 100 steps (2 s), on top of the shipped push every 15 s.
One run covers one terrain: "plane" (flat ground) or "rough" (the shipped trimesh terrain without the curriculum, so
robots stay on levels drawn uniformly from all ten levels). Each run has 2,048 robots and 2,700 steps (50 Hz); two runs
give 11.06 million states, as the quadrotor data. Commands are drawn by the task at every reset (shipped behavior).
Stored: the 48 policy inputs (the height scan removed: inputs 0:36 and 176:188), the 12 actions sent, the resets; the
reward is not stored. Every eighth robot (256 robots, pushed and unpushed alike) is held out from pretraining and used
only by the probe of section 29; for these robots the 140 height samples (float16) are also stored.
Output: domain2/data/terrain_<plane|rough>.npz. Usage: python collect_terrain.py <collector .pth> <plane|rough>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import igenv  # noqa: E402  (imports isaacgym first)
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N, T, HOLD = 2048, 2700, 256
OU_SIGMA, OU_THETA = 0.1, 0.15
PUSH_SHARE, PUSH_EVERY = 4 / 9, 100
TERRAIN = {"plane": {"env.terrain.terrainType": "plane", "env.terrain.curriculum": False},
           "rough": {"env.terrain.curriculum": False}}

class Collector(torch.nn.Module):
    """Actor of an rl-games continuous_a2c_logstd model (separate or shared MLP) with its running input normalization."""

    def __init__(self, sd):
        super().__init__()
        self.m = sd["running_mean_std.running_mean"].float()
        self.v = sd["running_mean_std.running_var"].float()
        layers, i = [], 0
        while f"a2c_network.actor_mlp.{i}.weight" in sd:
            lin = torch.nn.Linear(*sd[f"a2c_network.actor_mlp.{i}.weight"].shape[::-1])
            lin.weight.data, lin.bias.data = sd[f"a2c_network.actor_mlp.{i}.weight"], sd[f"a2c_network.actor_mlp.{i}.bias"]
            layers += [lin, torch.nn.ELU()]
            i += 2
        self.mlp = torch.nn.Sequential(*layers)
        self.mu = torch.nn.Linear(*sd["a2c_network.mu.weight"].shape[::-1])
        self.mu.weight.data, self.mu.bias.data = sd["a2c_network.mu.weight"], sd["a2c_network.mu.bias"]
        self.log_std = sd["a2c_network.sigma"].float()

    def forward(self, obs):
        x = torch.clamp((obs - self.m.to(obs.device)) / torch.sqrt(self.v.to(obs.device) + 1e-5), -5.0, 5.0)
        mu = self.mu(self.mlp(x))
        return mu + torch.exp(self.log_std.to(obs.device)) * torch.randn_like(mu)

def push(env, ids):
    """The push of the shipped AnymalTerrain task (push_robots), for the robots ids only."""
    from isaacgym import gymtorch
    env.root_states[ids, 7:9] = torch.empty(len(ids), 2, device=env.device).uniform_(-1.0, 1.0)
    i32 = ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_states),
                                                gymtorch.unwrap_tensor(i32), len(i32))

def main():
    ckpt, terrain = sys.argv[1], sys.argv[2]
    sd = {k.replace("_orig_mod.", ""): v for k, v in torch.load(ckpt, map_location="cpu")["model"].items()}
    pol = Collector(sd).cuda()
    env = igenv.make("AnymalTerrain", N, seed=17 if terrain == "plane" else 19, overrides=TERRAIN[terrain])
    obs = env.reset()["obs"]
    pushed = torch.arange(int(round(N * PUSH_SHARE)), device=env.device)
    hold_idx = torch.arange(0, N, N // HOLD, device=env.device)
    ou = torch.zeros(N, 12, device=env.device)
    S = np.zeros((T, N, 48), np.float32)
    A = np.zeros((T, N, 12), np.float32)
    R = np.zeros((T, N), bool)
    H = np.zeros((T, HOLD, 140), np.float16)
    rew, err, falls = 0.0, 0.0, 0
    with torch.no_grad():
        for t in range(T):
            ou = ou * (1 - OU_THETA) + OU_SIGMA * torch.randn_like(ou)
            a = torch.clamp(pol(obs) + ou, -1.0, 1.0)
            if t % PUSH_EVERY == PUSH_EVERY // 2:
                push(env, pushed)
            S[t] = torch.cat([obs[:, :36], obs[:, 176:188]], 1).cpu().numpy()
            A[t] = a.cpu().numpy()
            H[t] = obs[hold_idx, 36:176].cpu().numpy().astype(np.float16)
            o, r, d, info = env.step(a)
            rew += r.mean().item()
            err += torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1).mean().item()
            falls += int(d.sum().item())
            R[t] = d.cpu().numpy()
            obs = o["obs"]
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"terrain_{terrain}.npz"), states=S, actions=A, resets=R, holdout_heights=H,
             terrain_levels=env.terrain_levels.cpu().numpy(), hold_idx=hold_idx.cpu().numpy())
    print(f"collector on {terrain}: mean reward per step {rew / T:.4f}, xy velocity tracking error {err / T:.3f} m/s, "
          f"resets {falls} over {N} robots x {T} steps, mean terrain level {env.terrain_levels.float().mean().item():.2f}",
          flush=True)

if __name__ == "__main__":
    main()
