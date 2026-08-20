"""ArmHack Stand transition task with contact-selected ordered stepping."""

from __future__ import annotations

from isaaclab.utils import configclass

from .g1_stand_foot_recovery_env_cfg import (
    G1StandFootRecoveryEnvCfg,
    G1StandFootRecoveryEnvCfg_PLAY,
)


def _configure_adaptive_switch(cfg) -> None:
    """Enable a gentle two-step sequence while retaining robust Stand terms."""
    cfg.events.reset_robot_joints.params["phase_one_probability"] = 0.0
    cfg.events.reset_robot_joints.params["phase_two_probability"] = 0.0
    cfg.events.reset_robot_joints.params["asymmetric_support_probability"] = 0.65
    cfg.rewards.sequential_foot_step_progress.weight = 10.0
    cfg.rewards.sequential_foot_step_target_exp.weight = 18.0
    cfg.rewards.sequential_foot_step_clearance_exp.weight = 10.0
    cfg.rewards.sequential_active_foot_contact.weight = -4.0
    cfg.rewards.sequential_active_foot_clearance_l2.weight = -220.0
    cfg.rewards.sequential_active_foot_upward_velocity.weight = 3.0
    cfg.rewards.sequential_active_foot_velocity_l2.weight = -2.0
    cfg.rewards.sequential_active_foot_single_support.weight = 6.0
    cfg.rewards.sequential_foot_step_landing_exp.weight = 30.0
    cfg.rewards.sequential_foot_step_completion.weight = 80.0
    cfg.rewards.sequential_foot_step_lift.weight = 35.0
    cfg.rewards.sequential_foot_step_order_violation.weight = -25.0
    cfg.rewards.sequential_foot_final_target_l2.weight = -80.0
    cfg.rewards.sequential_final_ankle_distance_exp.weight = 25.0
    cfg.rewards.sequential_support_foot_drift_l2.weight = -180.0
    cfg.rewards.torso_xy_position_l2.weight = -12.0
    cfg.rewards.torso_yaw_l2.weight = -8.0
    cfg.rewards.torso_xy_position_near_stance_l2.weight = -28.0
    cfg.rewards.torso_yaw_near_stance_l2.weight = -16.0
    cfg.rewards.ankle_distance_l1.weight = -16.0
    cfg.rewards.ankle_distance_exp.weight = 10.0
    cfg.rewards.ankle_distance_success.weight = 12.0
    cfg.rewards.ankle_torques_l2.weight = -1.2e-3
    cfg.rewards.feet_slide.weight = -0.10


@configclass
class G1StandAdaptiveSwitchEnvCfg(G1StandFootRecoveryEnvCfg):
    """Train Walk->Stand recovery with lower-contact-force foot first."""

    def __post_init__(self):
        super().__post_init__()
        _configure_adaptive_switch(self)


@configclass
class G1StandAdaptiveSwitchEnvCfg_PLAY(G1StandFootRecoveryEnvCfg_PLAY):
    """Small smoke/evaluation variant."""

    def __post_init__(self):
        super().__post_init__()
        _configure_adaptive_switch(self)
