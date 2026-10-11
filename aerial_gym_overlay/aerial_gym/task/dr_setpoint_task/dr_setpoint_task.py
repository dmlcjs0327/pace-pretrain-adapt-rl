"""Setpoint tasks with unobserved dynamics variation.

Each episode draws, per vehicle and hidden from the policy:
  steady wind     horizontal world-frame force of magnitude U(0, WIND_MAX) x m g, uniform direction
  thrust gain     the executed collective thrust is k x the commanded one, k ~ U(1 - GAIN, 1 + GAIN)
  motor response  motor time constants scaled by U(0.5, 1.5) (robot configs '<name>_dr', drawn by the motor
                  model at every reset)
The policy observes only the usual 13-dimensional state, so it must infer these conditions from the flight
history. Wind and thrust gain are applied as extra body forces after the controller, in the same place as the
disturbances of Aerial Gym (BaseMultirotor.apply_disturbance).
"""
import math

import torch

from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
from aerial_gym.utils.math import quat_rotate_inverse

WIND_MAX = 0.3
GAIN = 0.2

class DRSetpointTask(PositionSetpointTask):
    def __init__(self, task_config, **kwargs):
        super().__init__(task_config, **kwargs)
        n, dev = self.sim_env.num_envs, self.device
        self.dr_wind = torch.zeros(n, 3, device=dev)
        self.dr_gain = torch.ones(n, device=dev)
        self._dr_mass = float(self.sim_env.robot_manager.robot_mass)
        rob = self.sim_env.robot_manager.robot
        orig = rob.apply_disturbance

        def apply_dr():
            orig()
            q = self.sim_env.global_tensor_dict["robot_orientation"]
            f = quat_rotate_inverse(q, self.dr_wind)
            f[:, 2] += (self.dr_gain - 1.0) * rob.controller.wrench_command[:, 2]
            rob.robot_force_tensors[:, 0, 0:3] += f

        rob.apply_disturbance = apply_dr
        self._dr_resample(torch.arange(n, device=dev))

    def _dr_resample(self, env_ids):
        if isinstance(env_ids, torch.Tensor) and len(env_ids) == 0:
            return
        k = len(env_ids)
        dev = self.device
        mag = torch.rand(k, device=dev) * WIND_MAX * self._dr_mass * 9.81
        ang = torch.rand(k, device=dev) * 2 * math.pi
        self.dr_wind[env_ids, 0] = mag * torch.cos(ang)
        self.dr_wind[env_ids, 1] = mag * torch.sin(ang)
        self.dr_wind[env_ids, 2] = 0.0
        self.dr_gain[env_ids] = 1.0 + GAIN * (2 * torch.rand(k, device=dev) - 1)

    def reset(self):
        out = super().reset()
        self._dr_resample(torch.arange(self.sim_env.num_envs, device=self.device))
        return out

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._dr_resample(env_ids)
