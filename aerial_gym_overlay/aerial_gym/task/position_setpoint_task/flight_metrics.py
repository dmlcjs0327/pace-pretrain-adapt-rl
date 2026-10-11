"""Per-episode flight metrics for the setpoint tasks.

The accumulator never changes observations, rewards, or terminations. At every step it
receives the pre-reset state of all vehicles, and when an episode ends it reports, for the
finished vehicles, the metrics of that episode through `infos["flight/<name>"]`. The
training observer averages each metric over the most recent finished episodes.

Metrics of one episode (T = number of steps of the episode, control rate 100 Hz):
  final_dist_m      distance to the target at the last step
  mean_dist_m       mean distance over the episode
  min_dist_m        smallest distance reached
  frac_within_05    fraction of steps within 0.5 m of the target
  reached_05        1 if the vehicle was ever within 0.5 m, else 0
  reach_time_s      time until it was first within 0.5 m (NaN if never)
  max_tilt_deg      largest angle between the body z-axis and the vertical
  mean_tilt_deg     mean of that angle
  mean_angvel       mean norm of the body angular velocity (rad/s)
  mean_speed        mean speed (m/s)
  action_rate       mean norm of the change of the executed command between steps
  action_sat        fraction of executed command components at the bound (|a| >= 0.999)
  diverged          1 if the episode ended because the distance exceeded 8 m
  collided          1 if it ended by a collision reported by the simulator
  timed_out         1 if it reached the episode length
  ep_len_s          episode duration (s)
"""

import torch

DT = 0.01

class FlightMetrics:
    NAMES = ("final_dist_m", "mean_dist_m", "min_dist_m", "frac_within_05", "reached_05", "reach_time_s",
             "max_tilt_deg", "mean_tilt_deg", "mean_angvel", "mean_speed", "action_rate", "action_sat",
             "diverged", "collided", "timed_out", "ep_len_s")

    def __init__(self, num_envs, device):
        z = lambda: torch.zeros(num_envs, device=device)
        self.n = z()
        self.sum_dist = z()
        self.min_dist = torch.full((num_envs,), float("inf"), device=device)
        self.within = z()
        self.first_reach = torch.full((num_envs,), float("nan"), device=device)
        self.max_tilt = z()
        self.sum_tilt = z()
        self.sum_angvel = z()
        self.sum_speed = z()
        self.sum_arate = z()
        self.sum_sat = z()
        self.n_arate = z()

    @torch.no_grad()
    def step(self, dist, up_z, angvel, linvel, action, prev_action, new_episode):
        """Accumulate one step. `new_episode` marks vehicles whose previous step ended an episode."""
        tilt = torch.rad2deg(torch.acos(torch.clamp(up_z, -1.0, 1.0)))
        self.n += 1
        self.sum_dist += dist
        self.min_dist = torch.minimum(self.min_dist, dist)
        inside = (dist < 0.5).float()
        self.within += inside
        hit = torch.isnan(self.first_reach) & (inside > 0)
        self.first_reach = torch.where(hit, self.n * DT, self.first_reach)
        self.max_tilt = torch.maximum(self.max_tilt, tilt)
        self.sum_tilt += tilt
        self.sum_angvel += torch.norm(angvel, dim=1)
        self.sum_speed += torch.norm(linvel, dim=1)
        self.sum_sat += (action.abs() >= 0.999).float().mean(dim=1)
        valid = (~new_episode).float()  # the change of command is defined from the second step on
        self.sum_arate += torch.norm(action - prev_action, dim=1) * valid
        self.n_arate += valid

    @torch.no_grad()
    def finish(self, done, final_dist, collided_or_diverged, truncated):
        """Return the metrics of all vehicles (meaningful where `done`) and reset those vehicles."""
        n = torch.clamp(self.n, min=1.0)
        diverged = (final_dist > 8.0).float()
        out = {
            "final_dist_m": final_dist.clone(),
            "mean_dist_m": self.sum_dist / n,
            "min_dist_m": self.min_dist.clone(),
            "frac_within_05": self.within / n,
            "reached_05": (~torch.isnan(self.first_reach)).float(),
            "reach_time_s": self.first_reach.clone(),
            "max_tilt_deg": self.max_tilt.clone(),
            "mean_tilt_deg": self.sum_tilt / n,
            "mean_angvel": self.sum_angvel / n,
            "mean_speed": self.sum_speed / n,
            "action_rate": self.sum_arate / torch.clamp(self.n_arate, min=1.0),
            "action_sat": self.sum_sat / n,
            "diverged": diverged,
            "collided": collided_or_diverged * (1.0 - diverged),
            "timed_out": truncated * (1.0 - collided_or_diverged),
            "ep_len_s": self.n * DT,
        }
        d = done
        self.n[d] = 0
        self.sum_dist[d] = 0
        self.min_dist[d] = float("inf")
        self.within[d] = 0
        self.first_reach[d] = float("nan")
        self.max_tilt[d] = 0
        self.sum_tilt[d] = 0
        self.sum_angvel[d] = 0
        self.sum_speed[d] = 0
        self.sum_arate[d] = 0
        self.sum_sat[d] = 0
        self.n_arate[d] = 0
        return {"flight/" + k: v for k, v in out.items()}
