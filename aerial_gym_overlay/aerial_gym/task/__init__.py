from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
    PositionSetpointTask,
)

from aerial_gym.task.position_setpoint_task_sim2real.position_setpoint_task_sim2real import (
    PositionSetpointTaskSim2Real,
)

from aerial_gym.task.position_setpoint_task_sim2real_end_to_end.position_setpoint_task_sim2real_end_to_end import (
    PositionSetpointTaskSim2RealEndToEnd,
)

from aerial_gym.task.position_setpoint_task_acceleration_sim2real.position_setpoint_task_acceleration_sim2real import (
    PositionSetpointTaskAccelerationSim2Real,
)

from aerial_gym.task.navigation_task.navigation_task import NavigationTask

from aerial_gym.config.task_config.position_setpoint_task_config import (
    task_config as position_setpoint_task_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_config import (
    task_config as position_setpoint_task_sim2real_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_end_to_end_config import (
    task_config as position_setpoint_task_sim2real_end_to_end_config,
)

from aerial_gym.config.task_config.position_setpoint_task_acceleration_sim2real_config import (
    task_config as position_setpoint_task_acceleration_sim2real_config,
)

from aerial_gym.config.task_config.navigation_task_config import (
    task_config as navigation_task_config,
)

from aerial_gym.registry.task_registry import task_registry


task_registry.register_task(
    "position_setpoint_task", PositionSetpointTask, position_setpoint_task_config
)
task_registry.register_task(
    "position_setpoint_task_sim2real",
    PositionSetpointTaskSim2Real,
    position_setpoint_task_sim2real_config,
)

task_registry.register_task(
    "position_setpoint_task_sim2real_end_to_end",
    PositionSetpointTaskSim2RealEndToEnd,
    position_setpoint_task_sim2real_end_to_end_config,
)

task_registry.register_task(
    "position_setpoint_task_acceleration_sim2real",
    PositionSetpointTaskAccelerationSim2Real,
    position_setpoint_task_acceleration_sim2real_config,
)

task_registry.register_task("navigation_task", NavigationTask, navigation_task_config)


from aerial_gym.task.position_setpoint_task_reconfigurable.position_setpoint_task_reconfigurable import (
    PositionSetpointTaskReconfigurable,
)

from aerial_gym.config.task_config.position_setpoint_task_config_reconfigurable import (
    task_config as position_setpoint_task_config_reconfigurable,
)

from aerial_gym.task.position_setpoint_task_morphy.position_setpoint_task_morphy import (
    PositionSetpointTaskMorphy,
)

from aerial_gym.config.task_config.position_setpoint_task_morphy_config import (
    task_config as position_setpoint_task_config_morphy,
)


task_registry.register_task(
    "position_setpoint_task_reconfigurable",
    PositionSetpointTaskReconfigurable,
    position_setpoint_task_config_reconfigurable,
)

task_registry.register_task(
    "position_setpoint_task_morphy",
    PositionSetpointTaskMorphy,
    position_setpoint_task_config_morphy,
)

# Tasks of the pretrain-adapt study (three quadrotor tasks, their variants, and the dynamics-varied settings).
from aerial_gym.task.aggressive_setpoint_task.aggressive_setpoint_task import AggressiveSetpointTask
from aerial_gym.task.dr_setpoint_task.dr_setpoint_task import DRSetpointTask
from aerial_gym.config.task_config.state_setpoint_ssl_configs import (
    state_setpoint_baseQuad_clean_config,
    state_setpoint_x500_clean_config,
    ss_lmf2_clean_config,
    recovery_baseQuad_config,
    recovery_x500_config,
    recovery_lmf2_config,
    agg_baseQuad_config,
    agg_x500_config,
    agg_lmf2_config,
    agg_d1_x500_config,
    agg_d1_lmf2_config,
    agg_d2_x500_config,
    agg_d2_lmf2_config,
    agg_upset_x500_config,
    agg_upset_lmf2_config,
)
from aerial_gym.config.task_config.dr_setpoint_configs import (
    ss_dr_x500_config,
    ss_dr_lmf2_config,
    recovery_dr_x500_config,
    recovery_dr_lmf2_config,
)
task_registry.register_task("state_setpoint_baseQuad_clean", PositionSetpointTask, state_setpoint_baseQuad_clean_config)
task_registry.register_task("state_setpoint_x500_clean", PositionSetpointTask, state_setpoint_x500_clean_config)
task_registry.register_task("ss_lmf2_clean", PositionSetpointTask, ss_lmf2_clean_config)
task_registry.register_task("recovery_baseQuad", PositionSetpointTask, recovery_baseQuad_config)
task_registry.register_task("recovery_x500", PositionSetpointTask, recovery_x500_config)
task_registry.register_task("recovery_lmf2", PositionSetpointTask, recovery_lmf2_config)
task_registry.register_task("agg_baseQuad", AggressiveSetpointTask, agg_baseQuad_config)
task_registry.register_task("agg_x500", AggressiveSetpointTask, agg_x500_config)
task_registry.register_task("agg_lmf2", AggressiveSetpointTask, agg_lmf2_config)
task_registry.register_task("agg_d1_x500", AggressiveSetpointTask, agg_d1_x500_config)
task_registry.register_task("agg_d1_lmf2", AggressiveSetpointTask, agg_d1_lmf2_config)
task_registry.register_task("agg_d2_x500", AggressiveSetpointTask, agg_d2_x500_config)
task_registry.register_task("agg_d2_lmf2", AggressiveSetpointTask, agg_d2_lmf2_config)
task_registry.register_task("agg_upset_x500", AggressiveSetpointTask, agg_upset_x500_config)
task_registry.register_task("agg_upset_lmf2", AggressiveSetpointTask, agg_upset_lmf2_config)
task_registry.register_task("ss_dr_x500", DRSetpointTask, ss_dr_x500_config)
task_registry.register_task("ss_dr_lmf2", DRSetpointTask, ss_dr_lmf2_config)
task_registry.register_task("recovery_dr_x500", DRSetpointTask, recovery_dr_x500_config)
task_registry.register_task("recovery_dr_lmf2", DRSetpointTask, recovery_dr_lmf2_config)
