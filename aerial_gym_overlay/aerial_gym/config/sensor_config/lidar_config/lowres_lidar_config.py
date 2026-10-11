from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig

class LowResLidarConfig(BaseLidarConfig):
    """Lightweight 360-degree lidar for RL obstacle awareness."""

    height = 16
    width = 128
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = -20
    vertical_fov_deg_max = 20
    max_range = 12.0
    min_range = 0.2

    return_pointcloud = False
    segmentation_camera = False
    randomize_placement = False
    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]
