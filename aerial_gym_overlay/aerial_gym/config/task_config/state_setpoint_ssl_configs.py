"""Task configurations of the study on R1 (base quadrotor), R2 (X500), and R3 (LMF2): T1 regulation (state_setpoint_*, ss_*), T2 recovery from an upset (recovery_*), T3 aggressive setpoint (agg_*), the difficulty ladder of T3 (agg_d1_*, agg_d2_*), and T3 from upset initial states (agg_upset_*). The observation holds 13 state values and the action four commands."""
import torch

class _StateSetpointBaseCfg:
    """Common base: the 13-value state observation of PositionSetpointTask, no camera, the empty environment."""
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_attitude_control"
    args = {}
    num_envs = 4096
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    use_depth_obs = False
    depth_cnn_latent_dim = 0
    action_space_dim = 4
    episode_len_steps = 300
    return_state_before_reset = False
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

class state_setpoint_baseQuad_clean_config(_StateSetpointBaseCfg):
    """D1 — baseQuad, clean (in-distribution baseline)."""
    robot_name = "base_quadrotor"

class state_setpoint_baseQuad_disturb_config(_StateSetpointBaseCfg):
    """D2 — baseQuad with force impulse disturbance (`base_quadrotor_disturb`)."""
    robot_name = "base_quadrotor_disturb"

class state_setpoint_x500_clean_config(_StateSetpointBaseCfg):
    """T1 on R2 (X500): motor time constants 0.0125 and 0.025 s against 0.04 s of the base quadrotor."""
    robot_name = "x500"

class state_setpoint_lmf2_disturb_config(_StateSetpointBaseCfg):
    """LMF2 with the camera disabled and disturbances enabled (legacy, not used in the study)."""
    robot_name = "lmf2_proprio"

# ============================================================================
# The grid of three airframes and three tasks.
# ============================================================================

class ss_lmf2_clean_config(_StateSetpointBaseCfg):
    """T1×R3 — LMF2 morphing arm, no disturbance."""
    robot_name = "lmf2_proprio_clean"

class _PositionSetpointSmoothCfg(_StateSetpointBaseCfg):
    """T2 base: dense pos error + amplified action_diff_penalty (×5) +
    """
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [5.0, 5.0, 5.0],          # ×5 (was 1.0)
        "absolute_action_reward_gain": [5.0, 5.0, 5.0],       # ×2.5 (was 2.0)
        "crash_penalty": -100,
    }

class psm_baseQuad_config(_PositionSetpointSmoothCfg):
    """T2×R1 — baseQuad position_setpoint with strong smoothing penalty."""
    robot_name = "base_quadrotor"

class psm_x500_config(_PositionSetpointSmoothCfg):
    """T2×R2 — x500 position_setpoint with strong smoothing penalty."""
    robot_name = "x500"

class psm_lmf2_config(_PositionSetpointSmoothCfg):
    """T2×R3 — lmf2 position_setpoint with strong smoothing penalty."""
    robot_name = "lmf2_proprio_clean"

# T3: position_setpoint_gc (sparse goal-conditioned)
class _PositionSetpointGCBaseCfg(_StateSetpointBaseCfg):
    """T3 base: sparse goal-conditioned reward — success threshold reach reward.
    """
    reward_parameters = {
        "success_threshold": 0.5,
        "success_reward": 10.0,
        "collision_penalty": -10.0,
        "step_penalty": -0.01,
        "max_dist": 15.0,
        "crash_penalty": -100,
    }

class gc_baseQuad_config(_PositionSetpointGCBaseCfg):
    """T3×R1 — baseQuad sparse goal-conditioned."""
    robot_name = "base_quadrotor"

class gc_x500_config(_PositionSetpointGCBaseCfg):
    """T3×R2 — x500 sparse goal-conditioned."""
    robot_name = "x500"

class gc_lmf2_config(_PositionSetpointGCBaseCfg):
    """T3×R3 — lmf2 sparse goal-conditioned."""
    robot_name = "lmf2_proprio_clean"

# ============================================================================
# 9D grid v2: Flight task spectrum (regulation/tracking/aggressive)
#   T1 = ss (static setpoint)        — R1, R2, R3 _StateSetpointBaseCfg subclass
#   T2 = traj (trajectory tracking)  — TrajectoryTrackingTask, 6s episode
#   T3 = agg (aggressive setpoint)   — AggressiveSetpointTask, 1s episode
# ============================================================================

class _TrajectoryTrackingBaseCfg(_StateSetpointBaseCfg):
    """
    """
    episode_len_steps = 300
    sim_dt = 0.02              # for trajectory time computation
    trajectory_dt = 0.02
    reward_parameters = {
        "pos_error_gain1": [3.0, 3.0, 3.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [3.0, 3.0, 3.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

class traj_baseQuad_config(_TrajectoryTrackingBaseCfg):
    """T2×R1 — baseQuad trajectory tracking."""
    robot_name = "base_quadrotor"

class traj_x500_config(_TrajectoryTrackingBaseCfg):
    """T2×R2 — x500 trajectory tracking."""
    robot_name = "x500"

class traj_lmf2_config(_TrajectoryTrackingBaseCfg):
    """T2×R3 — lmf2 trajectory tracking."""
    robot_name = "lmf2_proprio_clean"

# T3: Aggressive setpoint — large step + tight time bound
class _AggressiveSetpointBaseCfg(_StateSetpointBaseCfg):
    """T3 base: aggressive setpoint reaching. The episode lasts 200 steps (2 s at 100 Hz): with the attitude interface
    (thrust up to 2 mg, tilt about 60 deg) the horizontal acceleration is at most about 17 m/s^2, so the shortest
    time to cover 8 m is about 1.37 s, and a 1-s episode would leave most targets out of reach.
    """
    episode_len_steps = 200
    aggressive_dist_min = 4.0
    aggressive_dist_max = 8.0
    reward_parameters = {
        "pos_error_gain1": [4.0, 4.0, 4.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [4.0, 4.0, 4.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 10.0,
        "max_dist": 20.0,
        "action_diff_penalty_gain": [0.3, 0.3, 0.3],
        "absolute_action_reward_gain": [1.0, 1.0, 1.0],
        "crash_penalty": -200,
    }

class agg_baseQuad_config(_AggressiveSetpointBaseCfg):
    """T3×R1 — baseQuad aggressive setpoint."""
    robot_name = "base_quadrotor"

class agg_x500_config(_AggressiveSetpointBaseCfg):
    """T3×R2 — x500 aggressive setpoint (faster motor advantage)."""
    robot_name = "x500"

class agg_lmf2_config(_AggressiveSetpointBaseCfg):
    """T3×R3 — lmf2 aggressive setpoint (morphing arm dynamics)."""
    robot_name = "lmf2_proprio_clean"

# Difficulty ladder of T3: same task, reward, and 2 s episode, with the
# target distance drawn from [1, 2] m (d1) or [2, 4] m (d2); the original T3 uses [4, 8] m.
class agg_d1_x500_config(agg_x500_config):
    aggressive_dist_min = 1.0
    aggressive_dist_max = 2.0

class agg_d1_lmf2_config(agg_lmf2_config):
    aggressive_dist_min = 1.0
    aggressive_dist_max = 2.0

class agg_d2_x500_config(agg_x500_config):
    aggressive_dist_min = 2.0
    aggressive_dist_max = 4.0

class agg_d2_lmf2_config(agg_lmf2_config):
    aggressive_dist_min = 2.0
    aggressive_dist_max = 4.0

# ============================================================================
# ----------------------------------------------------------------------------
#
# ============================================================================
class _RecoveryBaseCfg(_StateSetpointBaseCfg):
    """T2 base: recovery from upset. target=origin static.
    """
    episode_len_steps = 150
    success_threshold = 0.3

class recovery_baseQuad_config(_RecoveryBaseCfg):
    """T2×R1 — baseQuad recovery from upset."""
    robot_name = "base_quadrotor_recovery"

class recovery_x500_config(_RecoveryBaseCfg):
    """T2×R2 — x500 recovery from upset (faster motor advantage)."""
    robot_name = "x500_recovery"

class recovery_lmf2_config(_RecoveryBaseCfg):
    """T2×R3 — lmf2 recovery from upset (morphing arm dynamics)."""
    robot_name = "lmf2_proprio_clean_recovery"

# V19: T3 with diverse initial states (T3 positions, T2 upset attitudes
# and velocities); task, reward, targets, and episode are those of T3.
class agg_upset_x500_config(agg_x500_config):
    robot_name = "x500_agg_upset"

class agg_upset_lmf2_config(agg_lmf2_config):
    robot_name = "lmf2_proprio_clean_agg_upset"
