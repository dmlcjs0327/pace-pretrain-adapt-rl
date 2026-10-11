"""Task configs of the dynamics-varied setpoint tasks: T1 regulation and T2 recovery on R2 and R3,
with the robots of dr_robot_configs.py and the hidden wind and thrust gain of DRSetpointTask."""
from aerial_gym.config.task_config.state_setpoint_ssl_configs import (
    state_setpoint_x500_clean_config, ss_lmf2_clean_config, recovery_x500_config, recovery_lmf2_config)

class ss_dr_x500_config(state_setpoint_x500_clean_config):
    robot_name = "x500_dr"

class ss_dr_lmf2_config(ss_lmf2_clean_config):
    robot_name = "lmf2_proprio_clean_dr"

class recovery_dr_x500_config(recovery_x500_config):
    robot_name = "x500_recovery_dr"

class recovery_dr_lmf2_config(recovery_lmf2_config):
    robot_name = "lmf2_proprio_clean_recovery_dr"
