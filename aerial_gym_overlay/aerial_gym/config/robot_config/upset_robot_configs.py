"""Robots for V19: T3 with diverse initial states.

The initial position is drawn as in T3 (ratio 0-1 of the environment bounds, the parent configuration), and the initial
attitude and velocities are drawn from the upset ranges of T2 (X500RecoveryCfg): roll and pitch within +-60 deg, yaw
free, linear velocity within +-2 m/s and angular velocity within +-4 rad/s per axis. Everything else is the parent's.
"""
import numpy as np

from aerial_gym.config.robot_config.x500_config import X500Cfg
from aerial_gym.config.robot_config.lmf2_config import LMF2ProprioCleanCfg

_UPSET_MIN = [-np.pi / 3, -np.pi / 3, -np.pi, 1.0, -2.0, -2.0, -2.0, -4.0, -4.0, -4.0]
_UPSET_MAX = [np.pi / 3, np.pi / 3, np.pi, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0]

class X500AggUpsetCfg(X500Cfg):
    class init_config:
        min_init_state = list(X500Cfg.init_config.min_init_state[:3]) + _UPSET_MIN
        max_init_state = list(X500Cfg.init_config.max_init_state[:3]) + _UPSET_MAX

class LMF2AggUpsetCfg(LMF2ProprioCleanCfg):
    class init_config:
        min_init_state = list(LMF2ProprioCleanCfg.init_config.min_init_state[:3]) + _UPSET_MIN
        max_init_state = list(LMF2ProprioCleanCfg.init_config.max_init_state[:3]) + _UPSET_MAX
