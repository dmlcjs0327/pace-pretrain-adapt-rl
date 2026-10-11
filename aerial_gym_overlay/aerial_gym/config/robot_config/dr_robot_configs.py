"""Robot configs for the dynamics-varied setpoint tasks: motor time constants scaled by U(0.5, 1.5)
around the nominal values, drawn per vehicle at every reset by the motor model."""
from aerial_gym.config.robot_config.x500_config import X500Cfg, X500RecoveryCfg
from aerial_gym.config.robot_config.lmf2_config import LMF2ProprioCleanCfg, LMF2ProprioCleanRecoveryCfg

def _scaled(base):
    mm = base.control_allocator_config.motor_model_config

    class motor_model_config(mm):
        motor_time_constant_increasing_min = 0.5 * mm.motor_time_constant_increasing_min
        motor_time_constant_increasing_max = 1.5 * mm.motor_time_constant_increasing_max
        motor_time_constant_decreasing_min = 0.5 * mm.motor_time_constant_decreasing_min
        motor_time_constant_decreasing_max = 1.5 * mm.motor_time_constant_decreasing_max

    class control_allocator_config(base.control_allocator_config):
        pass

    control_allocator_config.motor_model_config = motor_model_config
    return type(base.__name__ + "DR", (base,), {"control_allocator_config": control_allocator_config})

X500DRCfg = _scaled(X500Cfg)
X500RecoveryDRCfg = _scaled(X500RecoveryCfg)
LMF2ProprioCleanDRCfg = _scaled(LMF2ProprioCleanCfg)
LMF2ProprioCleanRecoveryDRCfg = _scaled(LMF2ProprioCleanRecoveryCfg)
