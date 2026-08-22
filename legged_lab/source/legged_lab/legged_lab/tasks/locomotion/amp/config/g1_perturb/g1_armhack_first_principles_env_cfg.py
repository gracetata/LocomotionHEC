"""First-principles ArmHack Stand/Walk tasks and shared physical objectives."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs.g1_perturb_env import G1_LOWER_BODY_JOINT_NAMES
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1_FOOT_BODY_NAMES,
    G1_LOCOMOTION_JOINT_NAMES,
)

from .g1_stand_adaptive_hold_env_cfg import G1StandAdaptiveHoldEnvCfg
from .g1_walk_ankle_spacing_env_cfg import G1WalkAnkleSpacingLateralEnvCfg
from .g1_walk_precision_switch_env_cfg import G1WalkPrecisionSwitchEnvCfg
from .g1_walk_yaw_force_robust_env_cfg import G1WalkYawForceRobustEnvCfg


ANKLE_DISTANCE_TARGET_M = 0.30
ANKLE_DISTANCE_TOLERANCE_M = 0.03
FOOT_TARGET_OFFSET_M = 0.15
SE2_XY_ACCEPTANCE_M = 0.05
SE2_YAW_ACCEPTANCE_RAD = 0.10


def _configure_stand_first_principles(cfg: G1StandAdaptiveHoldEnvCfg) -> None:
    reset = cfg.events.reset_robot_joints.params
    reset["phase_one_probability"] = 0.20
    reset["phase_two_probability"] = 0.25
    reset["asymmetric_support_probability"] = 0.65
    reset["final_distance"] = ANKLE_DISTANCE_TARGET_M

    cfg.upper_body_perturbation.producer_state_probability = 0.35
    cfg.upper_body_perturbation.producer_state_from = "primary"
    cfg.upper_body_perturbation.producer_state_to = "secondary"
    cfg.upper_body_perturbation.producer_joint_position_noise = 0.015
    cfg.upper_body_perturbation.producer_joint_velocity_noise = 0.05

    sequential_names = (
        "sequential_foot_step_progress",
        "sequential_foot_step_target_exp",
        "sequential_foot_step_clearance_exp",
        "sequential_active_foot_contact",
        "sequential_active_foot_clearance_l2",
        "sequential_active_foot_upward_velocity",
        "sequential_active_foot_velocity_l2",
        "sequential_active_foot_single_support",
        "sequential_foot_step_landing_exp",
        "sequential_foot_step_completion",
        "sequential_foot_step_lift",
        "sequential_foot_step_order_violation",
        "sequential_foot_final_target_l2",
        "sequential_final_ankle_distance_exp",
        "sequential_support_foot_drift_l2",
    )
    for name in sequential_names:
        getattr(cfg.rewards, name).params["lateral_target_offset_m"] = FOOT_TARGET_OFFSET_M
    cfg.rewards.sequential_final_ankle_distance_exp.params["target_distance_m"] = ANKLE_DISTANCE_TARGET_M
    for name in (
        "ankle_distance_l1",
        "ankle_distance_exp",
        "ankle_distance_success",
        "ankle_torques_l2",
        "torso_xy_position_near_stance_l2",
        "torso_yaw_near_stance_l2",
    ):
        getattr(cfg.rewards, name).params["target_distance"] = ANKLE_DISTANCE_TARGET_M

    cfg.rewards.sequential_foot_step_progress.weight = 12.0
    cfg.rewards.sequential_foot_step_target_exp.weight = 24.0
    cfg.rewards.sequential_foot_step_completion.weight = 100.0
    cfg.rewards.sequential_foot_step_lift.weight = 30.0
    cfg.rewards.sequential_foot_step_order_violation.weight = -60.0
    cfg.rewards.sequential_foot_final_target_l2.weight = -160.0
    cfg.rewards.sequential_final_ankle_distance_exp.weight = 40.0
    cfg.rewards.sequential_support_foot_drift_l2.weight = -220.0
    cfg.rewards.torso_xy_position_l2.weight = -35.0
    cfg.rewards.torso_yaw_l2.weight = -25.0
    cfg.rewards.torso_xy_position_near_stance_l2.weight = -120.0
    cfg.rewards.torso_yaw_near_stance_l2.weight = -80.0
    cfg.rewards.ankle_distance_l1.weight = -20.0
    cfg.rewards.ankle_distance_exp.weight = 14.0
    cfg.rewards.ankle_distance_success.weight = 18.0
    cfg.rewards.ankle_torques_l2.weight = -1.5e-3
    cfg.rewards.feet_slide.weight = -0.60
    cfg.rewards.double_support.weight = 0.50

    post_params = dict(cfg.rewards.post_complete_foot_velocity_l2.params)
    cfg.rewards.post_complete_foot_velocity_l2.weight = -200.0
    cfg.rewards.post_complete_contact_loss.weight = -60.0
    cfg.rewards.post_complete_target_l2.weight = -240.0
    cfg.rewards.post_complete_foot_angular_velocity_l2 = RewTerm(
        func=mdp.sequential_post_complete_foot_angular_velocity_l2,
        weight=-2.0,
        params=post_params,
    )
    cfg.rewards.post_complete_contact_force_balance_l2 = RewTerm(
        func=mdp.sequential_post_complete_contact_force_balance_l2,
        weight=-10.0,
        params=post_params,
    )
    cfg.rewards.post_complete_foot_yaw_l2 = RewTerm(
        func=mdp.sequential_post_complete_foot_yaw_l2,
        weight=-20.0,
        params=post_params,
    )
    lower_action_indices = [
        G1_LOCOMOTION_JOINT_NAMES.index(name) for name in G1_LOWER_BODY_JOINT_NAMES
    ]
    cfg.rewards.post_complete_action_rate_l2 = RewTerm(
        func=mdp.sequential_post_complete_action_rate_l2,
        weight=-2.0,
        params={**post_params, "action_indices": lower_action_indices},
    )
    cfg.rewards.post_complete_joint_velocity_l2 = RewTerm(
        func=mdp.sequential_post_complete_joint_velocity_l2,
        weight=-0.50,
        params={
            **post_params,
            "lower_body_cfg": SceneEntityCfg(
                "robot", joint_names=G1_LOWER_BODY_JOINT_NAMES, preserve_order=True
            ),
        },
    )


def _configure_walk_first_principles(cfg, *, spacing_weight: float) -> None:
    cfg.upper_body_perturbation.producer_state_probability = 0.35
    cfg.upper_body_perturbation.producer_state_from = "secondary"
    cfg.upper_body_perturbation.producer_state_to = "primary"
    cfg.upper_body_perturbation.producer_joint_position_noise = 0.015
    cfg.upper_body_perturbation.producer_joint_velocity_noise = 0.05
    cfg.commands.base_velocity.reset_command_to_zero = True

    if getattr(cfg.rewards, "ankle_distance_30cm_kernel", None) is None:
        cfg.rewards.ankle_distance_30cm_kernel = RewTerm(
            func=mdp.ankle_distance_target_kernel,
            weight=spacing_weight,
            params={
                "target_distance": ANKLE_DISTANCE_TARGET_M,
                "std": 0.05,
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
                ),
            },
        )
    else:
        cfg.rewards.ankle_distance_30cm_kernel.weight = spacing_weight
        cfg.rewards.ankle_distance_30cm_kernel.params["target_distance"] = ANKLE_DISTANCE_TARGET_M
        cfg.rewards.ankle_distance_30cm_kernel.params["std"] = 0.05

    cfg.rewards.torso_roll_pitch_l2.weight = -3.0
    cfg.rewards.feet_slide.weight = -0.60
    if getattr(cfg.rewards, "action_rate_l2", None) is not None:
        cfg.rewards.action_rate_l2.weight = -0.02
    if getattr(cfg.rewards, "strict_zero_body_motion_l2", None) is not None:
        cfg.rewards.strict_zero_body_motion_l2.weight = -10.0
        cfg.rewards.strict_zero_feet_motion_l2.weight = -6.0
        cfg.rewards.strict_zero_joint_vel_l2.weight = -0.08
        cfg.rewards.strict_zero_double_support.weight = 4.0
    if getattr(cfg.rewards, "feet_swing_clearance_band_l2", None) is not None:
        cfg.rewards.feet_swing_clearance_band_l2.params["target_height"] = 0.070
        cfg.rewards.feet_swing_clearance_band_l2.weight = -1.5
    if getattr(cfg.rewards, "pure_yaw_planar_drift_l2", None) is not None:
        cfg.rewards.pure_yaw_planar_drift_l2.weight = -50.0
        cfg.rewards.pure_yaw_planar_drift_l2.params["max_penalty"] = 100.0
    if getattr(cfg.rewards, "pure_yaw_root_rate_error_l2", None) is not None:
        cfg.rewards.pure_yaw_root_rate_error_l2.weight = -8.0

    for side in ("left", "right"):
        setattr(
            cfg.events,
            f"randomize_{side}_wrist_wrench",
            EventTerm(
                func=mdp.apply_external_force_torque,
                mode="interval",
                interval_range_s=(2.5, 5.0),
                is_global_time=False,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=f"{side}_wrist_yaw_link"),
                    "force_range": (-20.0, 20.0),
                    "torque_range": (-1.5, 1.5),
                },
            ),
        )


@configclass
class G1StandFirstPrinciplesEnvCfg(G1StandAdaptiveHoldEnvCfg):
    """One actor learns contact-selected two-step recovery and stable phase-two hold."""

    def __post_init__(self):
        super().__post_init__()
        _configure_stand_first_principles(self)


@configclass
class G1WalkFirstPrinciplesBaseEnvCfg(G1WalkPrecisionSwitchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_walk_first_principles(self, spacing_weight=180.0)


@configclass
class G1WalkFirstPrinciplesLateralEnvCfg(G1WalkAnkleSpacingLateralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_walk_first_principles(self, spacing_weight=220.0)


@configclass
class G1WalkFirstPrinciplesYawEnvCfg(G1WalkYawForceRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_walk_first_principles(self, spacing_weight=220.0)
