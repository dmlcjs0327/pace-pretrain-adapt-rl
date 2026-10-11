import torch

class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "forest_env"
    robot_name = "x500_with_camera_lowres"
    controller_name = "lee_attitude_control"
    args = {}
    num_envs = 4096
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    # Depth image as observation. Use _camera_lowres (64×64, faster) or _camera (135×240).
    use_depth_obs = True
    depth_cnn_latent_dim = 64
    action_space_dim = 4
    episode_len_steps = 300  # real physics time for simulation is this value multiplied by sim.dt
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
