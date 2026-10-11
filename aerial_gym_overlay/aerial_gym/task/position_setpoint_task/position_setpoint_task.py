import os
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.nn.depth_cnn import DepthCNN
from aerial_gym.task.position_setpoint_task.flight_metrics import FlightMetrics

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task")

def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)

class PositionSetpointTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Get the dictionary once from the environment and use it to get the observations later.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.use_depth_obs = getattr(self.task_config, "use_depth_obs", False)
        self.depth_cnn_latent_dim = getattr(self.task_config, "depth_cnn_latent_dim", 64)
        if self.use_depth_obs and "depth_range_pixels" in self.obs_dict:
            self.task_config.observation_space_dim = 13 + self.depth_cnn_latent_dim
            depth_shape = self.obs_dict["depth_range_pixels"].shape
            # (num_envs, num_sensors, H, W) -> H, W for CNN
            h, w = depth_shape[2], depth_shape[3]
            self.depth_cnn = DepthCNN(
                in_channels=1,
                latent_dim=self.depth_cnn_latent_dim,
                input_height=h,
                input_width=w,
            ).to(self.device)
            self.depth_cnn.eval()
            for p in self.depth_cnn.parameters():
                p.requires_grad = False
            logger.info("Depth observation enabled: CNN latent_dim=%s, obs_dim=%s", self.depth_cnn_latent_dim, self.task_config.observation_space_dim)
        else:
            if self.use_depth_obs and "depth_range_pixels" not in self.obs_dict:
                logger.warning("use_depth_obs=True but depth_range_pixels not in obs (use robot_name='base_quadrotor_with_camera'). Disabling depth obs.")
            self.use_depth_obs = False
            self.depth_cnn = None

        obs_dim = self.task_config.observation_space_dim
        self.observation_space = Dict(
            {
                "observations": Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "observation": Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        self.counter = 0

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "observation": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "achieved_goal": torch.zeros(
                (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
            ),
            "desired_goal": torch.zeros(
                (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self._t2_curriculum()
        self.target_position[:, 0:3] = 0.0  # torch.rand_like(self.target_position) * 10.0
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()

    def _t2_curriculum(self):
        """S23b (pre-registration v2, section 21): AG_T2_CURRICULUM = "100:0.25,200:0.5" scales the initial roll and
        pitch range and the linear and angular velocity ranges of the robot by 0.25 up to RL iteration 100, by 0.5 up
        to 200, and leaves them unchanged afterwards (yaw and position unchanged). Training only; the runner sets it."""
        spec = os.environ.get("AG_T2_CURRICULUM", "")
        if not spec:
            return
        rob = self.sim_env.robot_manager.robot
        if not hasattr(self, "_t2_orig"):
            self._t2_orig = (rob.min_init_state.clone(), rob.max_init_state.clone())
            self._t2_spec = sorted((int(p.split(":")[0]), float(p.split(":")[1])) for p in spec.split(","))
            self._t2_horizon = int(os.environ.get("AG_HORIZON", "32"))
        it = self.counter // self._t2_horizon
        scale = 1.0
        for end, sc in self._t2_spec:
            if it < end:
                scale = sc
                break
        lo, hi = self._t2_orig
        idx = [3, 4, 7, 8, 9, 10, 11, 12]  # roll, pitch, vx, vy, vz, wx, wy, wz (symmetric ranges around 0)
        rob.min_init_state[..., idx] = lo[..., idx] * scale
        rob.max_init_state[..., idx] = hi[..., idx] * scale

    def reset_idx(self, env_ids):
        # Only the vehicles that are reset get a new target. The earlier code set the target of
        # every vehicle to the origin, which was harmless for fixed-origin tasks but reset the targets of all
        # in-flight vehicles in the aggressive setpoint task.
        self.target_position[env_ids, 0:3] = 0.0
        self._t2_curriculum()
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self._t2_curriculum()  # S23b: the environment manager also resets vehicles inside sim_env.step
        self.prev_actions[:] = self.actions
        self.actions = actions

        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.
        self.sim_env.step(actions=self.actions)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        dones = torch.logical_or(self.terminations > 0, self.truncations > 0)
        # Flight metrics (logging only; observations, rewards, and terminations are unchanged).
        if not hasattr(self, "_flight"):
            self._flight = FlightMetrics(self.sim_env.num_envs, self.device)
            self._flight_new_episode = torch.ones(self.sim_env.num_envs, dtype=torch.bool, device=self.device)
        fm_dist = torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1)
        self._flight.step(fm_dist, quat_axis(self.obs_dict["robot_orientation"], 2)[..., 2],
                          self.obs_dict["robot_body_angvel"], self.obs_dict["robot_body_linvel"],
                          self.actions, self.prev_actions, self._flight_new_episode)
        flight_infos = self._flight.finish(dones, fm_dist, (self.terminations > 0).float(),
                                           (self.truncations > 0).float())
        self._flight_new_episode = dones.clone()
        terminal_obs = self.build_observation_tensor()
        terminal_achieved_goal = self.obs_dict["robot_position"].clone()
        terminal_desired_goal = self.target_position.clone()
        success_threshold = float(getattr(self.task_config, "success_threshold", 0.5))
        success = (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1)
            < success_threshold
        ).float()
        self.infos = {
            "success": success,
            "crashes": self.terminations.float().clone(),
            "timeouts": self.truncations.float().clone(),
            "terminal_observation": terminal_obs,
            "terminal_achieved_goal": terminal_achieved_goal,
            "terminal_desired_goal": terminal_desired_goal,
            "achieved_goal": terminal_achieved_goal,
            "desired_goal": terminal_desired_goal,
        }
        self.infos.update(flight_infos)

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        return self.get_return_tuple()

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        self.task_obs["observations"][:] = self.build_observation_tensor()
        self.task_obs["observation"][:] = self.task_obs["observations"]
        self.task_obs["achieved_goal"][:] = self.obs_dict["robot_position"]
        self.task_obs["desired_goal"][:] = self.target_position
        if self.use_depth_obs and self.depth_cnn is not None and "depth_range_pixels" in self.obs_dict:
            depth_img = self.obs_dict["depth_range_pixels"].squeeze(1)
            if depth_img.dim() == 3:
                depth_img = depth_img.unsqueeze(1)
            with torch.no_grad():
                latent = self.depth_cnn(depth_img)
            self.task_obs["observations"][:, 13 : 13 + self.depth_cnn_latent_dim] = latent
            self.task_obs["observation"][:, 13 : 13 + self.depth_cnn_latent_dim] = latent
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def build_observation_tensor(self):
        obs = torch.zeros(
            (self.sim_env.num_envs, self.task_config.observation_space_dim),
            device=self.device,
            requires_grad=False,
        )
        obs[:, 0:3] = self.target_position - self.obs_dict["robot_position"]
        obs[:, 3:7] = self.obs_dict["robot_orientation"]
        obs[:, 7:10] = self.obs_dict["robot_body_linvel"]
        obs[:, 10:13] = self.obs_dict["robot_body_angvel"]
        return obs

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_linvel = obs_dict["robot_linvel"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            root_quats,
            angular_velocity,
            obs_dict["crashes"],
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )

    def recompute_reward(
        self,
        relabeled_next_obs,
        achieved_goal,
        desired_goal,
        crashes,
        current_achieved_goal=None,
        relabeled_obs=None,
    ):
        _ = current_achieved_goal, relabeled_obs
        robot_quats = relabeled_next_obs[:, 3:7]
        robot_angvels = relabeled_next_obs[:, 10:13]
        lin_vels = relabeled_next_obs[:, 7:10]
        pos_error = quat_apply_inverse(robot_quats, desired_goal - achieved_goal)
        rewards, _ = compute_reward(
            pos_error,
            lin_vels,
            robot_quats,
            robot_angvels,
            crashes.clone(),
            1.0,
            torch.zeros(
                (relabeled_next_obs.shape[0], self.task_config.action_space_dim),
                device=self.device,
            ),
            torch.zeros(
                (relabeled_next_obs.shape[0], self.task_config.action_space_dim),
                device=self.device,
            ),
            self.task_config.reward_parameters,
        )
        return rewards

@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)

@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)

@torch.jit.script
def compute_reward(
    pos_error,
    lin_vels,
    robot_quats,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    dist = torch.norm(pos_error, dim=1)

    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)

    dist_reward = (20 - dist) / 40.0

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

    total_reward = (
        pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    )
    total_reward[:] = curriculum_level_multiplier * total_reward

    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes
