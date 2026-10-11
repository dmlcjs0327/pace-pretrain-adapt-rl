"""
Low-resolution depth camera config for faster image-based RL training.
Use with robot_name="base_quadrotor_with_camera_lowres" when use_depth_obs=True.
"""
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)

class DepthCameraLowResConfig(BaseDepthCameraConfig):
    """64×64 depth for faster Warp rendering and CNN; segmentation off to save compute."""

    height = 64
    width = 64
    segmentation_camera = False
