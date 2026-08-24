"""First-principles ArmHack Stand and Walk tasks, one actor per task."""

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.envs.g1_perturb_env import G1_LOWER_BODY_JOINT_NAMES
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1_FOOT_BODY_NAMES,
    G1_LOCOMOTION_JOINT_NAMES,
)

from .g1_stand_foot_recovery_env_cfg import G1StandFootRecoveryEnvCfg
from .g1_walk_behavior_env_cfg import G1WalkBehaviorFinetuneEnvCfg
from .reference_data import STAND_RANDOM_POSE_BANK_RELATIVE_PATH


G1_WALK_RESPONSE_MODE_CONFIG_PATH = os.path.join(
    LEGGED_LAB_ROOT_DIR,
    "data",
    "MotionData",
    "g1_29dof",
    "amp",
    "armhack_walk_response_50hz",
    "task_sampling_config.json",
)


def _step_params(pelvis_cfg, ankle_cfg, contact_cfg) -> dict:
    return {
        "pelvis_cfg": pelvis_cfg,
        "foot_cfg": ankle_cfg,
        "sensor_cfg": contact_cfg,
        "lateral_target_offset_m": 0.15,
        "min_clearance_m": 0.035,
        "landing_tolerance_m": 0.10,
        "min_step_duration_s": 0.20,
    }


@configclass
class G1ArmHackStandFirstPrinciplesSingleEnvCfg(G1StandFootRecoveryEnvCfg):
    """One policy learns both ordered adjustments and the final quiet hold."""

    def __post_init__(self):
        super().__post_init__()

        # One actor sees phase through the existing 96-D observation. There is
        # no expert router, second actor, or runtime policy hierarchy.
        self.episode_length_s = 12.0
        self.upper_body_perturbation.source = "random_pose_trajectory"
        self.upper_body_perturbation.random_pose_bank_path = (
            STAND_RANDOM_POSE_BANK_RELATIVE_PATH.as_posix()
        )
        self.upper_body_perturbation.random_curriculum_enabled = False
        self.upper_body_perturbation.random_curriculum_motion_scale = 1.0
        self.upper_body_perturbation.random_transition_duration_range_s = (1.5, 5.0)
        # The inherited recovery schedule rewrites reward weights every step.
        # This task is a fresh objective, so its explicit weights below are the
        # sole source of truth and randomization is full-strength from reset.
        self.curriculum.stance_recovery = None

        pelvis_cfg = SceneEntityCfg("robot", body_names="pelvis")
        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
        ankle_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )
        ankle_joint_cfg = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
        )
        leg_names = [name for name in G1_LOWER_BODY_JOINT_NAMES if not name.startswith("waist_")]
        leg_cfg = SceneEntityCfg("robot", joint_names=leg_names, preserve_order=True)
        leg_action_indices = [G1_LOCOMOTION_JOINT_NAMES.index(name) for name in leg_names]
        step = _step_params(pelvis_cfg, ankle_cfg, contact_cfg)
        # Skill-acquisition stage: a coarse touchdown gate lets the single
        # actor experience phase transitions. Later 2000-iteration stages
        # tighten this to the final precision target.
        sequential_terms = (
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
        for term_name in sequential_terms:
            getattr(self.rewards, term_name).params["landing_tolerance_m"] = 0.10
            getattr(self.rewards, term_name).params["min_step_duration_s"] = 0.20

        # Broad initial stance and velocity distribution. Phase-zero dominates;
        # phase-one/final samples keep the second step and planted hold trainable.
        self.events.reset_robot_joints.params.update(
            {
                "distance_range": (0.08, 0.46),
                "close_distance_range": (0.08, 0.16),
                "close_stance_probability": 0.45,
                "nominal_distance_range": (0.27, 0.33),
                "nominal_stance_probability": 0.15,
                "asymmetric_support_probability": 0.50,
                "phase_one_probability": 0.40,
                "phase_two_probability": 0.10,
                "support_distance_range": (0.22, 0.34),
                "position_scale_range": (0.92, 1.08),
                "velocity_range": (-0.20, 0.20),
            }
        )
        self.events.handoff_state_reset = EventTerm(
            func=mdp.reset_from_handoff_state_library,
            mode="reset",
            params={"state_library_path": "", "probability": 0.0},
        )
        self.events.random_end_effector_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                    preserve_order=True,
                ),
                "force_range": (-5.0, 5.0),
                "torque_range": (-0.75, 0.75),
            },
        )
        self.events.random_torso_external_wrench.params["force_range"] = (-5.0, 5.0)
        self.events.random_torso_external_wrench.params["torque_range"] = (-0.75, 0.75)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "yaw": (-0.12, 0.12),
        }

        # Two real single-foot swings, accurate landing, fixed support foot.
        self.rewards.sequential_foot_step_progress.weight = 12.0
        # Dense target/single-support rewards are intentionally small: large
        # per-step rewards make hovering more profitable than touchdown.
        self.rewards.sequential_foot_step_target_exp.weight = 1.0
        self.rewards.sequential_foot_step_clearance_exp.weight = 18.0
        self.rewards.sequential_active_foot_clearance_l2.weight = -1500.0
        self.rewards.sequential_active_foot_upward_velocity.weight = 8.0
        self.rewards.sequential_active_foot_velocity_l2.weight = -0.2
        self.rewards.sequential_active_foot_single_support.weight = 0.5
        self.rewards.sequential_active_foot_contact.weight = -50.0
        self.rewards.sequential_foot_step_landing_exp.weight = 80.0
        self.rewards.sequential_foot_step_completion.weight = 150.0
        self.rewards.sequential_foot_step_lift.weight = 150.0
        self.rewards.sequential_foot_step_order_violation.weight = -30.0
        self.rewards.sequential_foot_final_target_l2.weight = -60.0
        self.rewards.sequential_final_ankle_distance_exp.weight = 45.0
        self.rewards.sequential_support_foot_drift_l2.weight = -80.0
        self.rewards.sequential_active_foot_air_time_excess_l2 = RewTerm(
            func=mdp.sequential_active_foot_air_time_excess_l2,
            weight=-15.0,
            params={**step, "max_step_duration_s": 1.0},
        )
        self.rewards.sequential_active_foot_descent_exp = RewTerm(
            func=mdp.sequential_active_foot_descent_exp,
            weight=8.0,
            params={
                **step,
                "target_downward_velocity_mps": 0.15,
                "velocity_std_mps": 0.10,
                "target_gate_std_m": 0.08,
            },
        )

        # SE(2) is reset-relative and remains active through the whole episode.
        self.rewards.torso_xy_position_l2.weight = -8.0
        self.rewards.torso_yaw_l2.weight = -4.0
        self.rewards.torso_xy_position_near_stance_l2.weight = -20.0
        self.rewards.torso_yaw_near_stance_l2.weight = -10.0
        self.rewards.root_xy_position_l2.weight = 0.0
        self.terminations.sequential_pelvis_xy_out_of_bounds.params["max_displacement_m"] = 0.60

        # Final-phase objectives directly target the reported failure: repeated
        # stepping, foot velocity, load oscillation, action chatter and ankle effort.
        self.rewards.post_completion_airborne = RewTerm(
            func=mdp.sequential_post_completion_airborne, weight=-25.0, params=step
        )
        self.rewards.post_completion_foot_motion_l2 = RewTerm(
            func=mdp.sequential_post_completion_foot_motion_l2,
            weight=-2.0,
            params={**step, "velocity_scale": 0.10},
        )
        self.rewards.post_completion_joint_vel_l2 = RewTerm(
            func=mdp.sequential_post_completion_joint_vel_l2,
            weight=-0.02,
            params={**step, "asset_cfg": leg_cfg},
        )
        self.rewards.post_completion_action_rate_l2 = RewTerm(
            func=mdp.sequential_post_completion_action_rate_l2,
            weight=-0.08,
            params={**step, "action_indices": leg_action_indices},
        )
        self.rewards.post_completion_contact_imbalance_l2 = RewTerm(
            func=mdp.sequential_post_completion_contact_imbalance_l2,
            weight=-2.0,
            params={**step, "force_scale_n": 300.0},
        )
        self.rewards.post_completion_ankle_torque_l2 = RewTerm(
            func=mdp.sequential_post_completion_ankle_torque_l2,
            weight=-2.0e-4,
            params={**step, "ankle_cfg": ankle_joint_cfg},
        )
        # Do not pay the global distance objective before the ordered task is
        # complete: otherwise both feet can slide apart without taking steps.
        self.rewards.feet_planar_separation_l2.weight = 0.0
        self.rewards.ankle_distance_l1.weight = 0.0
        self.rewards.ankle_distance_exp.weight = 0.0
        self.rewards.ankle_distance_success.weight = 0.0
        self.rewards.ankle_distance_success.params["tolerance"] = 0.02
        self.rewards.termination_penalty.weight = -500.0


@configclass
class G1ArmHackStandFirstPrinciplesSingleEnvCfg_PLAY(
    G1ArmHackStandFirstPrinciplesSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1ArmHackStandFirstPrinciplesStrictSingleEnvCfg(
    G1ArmHackStandFirstPrinciplesSingleEnvCfg
):
    """Tighten touchdown, SE(2), quiet hold and perturbations in the same actor."""

    def __post_init__(self):
        super().__post_init__()
        sequential_terms = (
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
            "sequential_active_foot_air_time_excess_l2",
            "sequential_active_foot_descent_exp",
            "post_completion_airborne",
            "post_completion_foot_motion_l2",
            "post_completion_joint_vel_l2",
            "post_completion_action_rate_l2",
            "post_completion_contact_imbalance_l2",
            "post_completion_ankle_torque_l2",
        )
        for term_name in sequential_terms:
            term = getattr(self.rewards, term_name)
            if "landing_tolerance_m" in term.params:
                term.params["landing_tolerance_m"] = 0.04
            if "min_step_duration_s" in term.params:
                term.params["min_step_duration_s"] = 0.40

        self.events.reset_robot_joints.params.update(
            {
                "phase_one_probability": 0.25,
                "phase_two_probability": 0.40,
                "asymmetric_support_probability": 0.45,
            }
        )
        # Precision/hold stage remains mildly perturbed. Full-force robustness
        # is a later 2000-iteration continuation after this skill is retained.
        self.events.random_end_effector_wrench.params["force_range"] = (-5.0, 5.0)
        self.events.random_end_effector_wrench.params["torque_range"] = (-0.75, 0.75)
        self.events.random_torso_external_wrench.params["force_range"] = (-5.0, 5.0)
        self.events.random_torso_external_wrench.params["torque_range"] = (-0.75, 0.75)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "yaw": (-0.12, 0.12),
        }

        self.rewards.sequential_active_foot_contact.weight = -20.0
        self.rewards.sequential_active_foot_clearance_l2.weight = -800.0
        self.rewards.sequential_active_foot_upward_velocity.weight = 5.0
        self.rewards.sequential_foot_step_landing_exp.weight = 120.0
        self.rewards.sequential_foot_step_completion.weight = 200.0
        self.rewards.sequential_foot_step_lift.weight = 100.0
        self.rewards.sequential_foot_final_target_l2.weight = -150.0
        self.rewards.sequential_support_foot_drift_l2.weight = -180.0
        self.rewards.torso_xy_position_l2.weight = -12.0
        self.rewards.torso_yaw_l2.weight = -6.0
        self.rewards.torso_xy_position_near_stance_l2.weight = -35.0
        self.rewards.torso_yaw_near_stance_l2.weight = -18.0
        self.terminations.sequential_pelvis_xy_out_of_bounds.params["max_displacement_m"] = 0.40

        self.rewards.post_completion_airborne.weight = -40.0
        self.rewards.post_completion_foot_motion_l2.weight = -2.0
        self.rewards.post_completion_joint_vel_l2.weight = -0.03
        self.rewards.post_completion_action_rate_l2.weight = -0.10
        self.rewards.post_completion_contact_imbalance_l2.weight = -2.0
        self.rewards.post_completion_ankle_torque_l2.weight = -2.0e-4


@configclass
class G1ArmHackStandFirstPrinciplesStrictSingleEnvCfg_PLAY(
    G1ArmHackStandFirstPrinciplesStrictSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1ArmHackStandFirstPrinciplesOneStepSingleEnvCfg(
    G1ArmHackStandFirstPrinciplesStrictSingleEnvCfg
):
    """Enforce exactly one left step, one right step, then reset-relative hold."""

    def __post_init__(self):
        super().__post_init__()
        pelvis_cfg = SceneEntityCfg("robot", body_names="pelvis")
        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
        ankle_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )
        step = _step_params(pelvis_cfg, ankle_cfg, contact_cfg)
        step["landing_tolerance_m"] = 0.035
        step["min_step_duration_s"] = 0.35
        self.events.reset_robot_joints.params.update(
            {
                "phase_one_probability": 0.30,
                "phase_two_probability": 0.20,
                "asymmetric_support_probability": 0.50,
            }
        )
        sequential_terms = (
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
            "sequential_active_foot_air_time_excess_l2",
            "sequential_active_foot_descent_exp",
            "post_completion_airborne",
            "post_completion_foot_motion_l2",
            "post_completion_joint_vel_l2",
            "post_completion_action_rate_l2",
            "post_completion_contact_imbalance_l2",
            "post_completion_ankle_torque_l2",
        )
        for term_name in sequential_terms:
            term = getattr(self.rewards, term_name)
            if "landing_tolerance_m" in term.params:
                term.params["landing_tolerance_m"] = 0.035
            if "min_step_duration_s" in term.params:
                term.params["min_step_duration_s"] = 0.35

        # Completion is the primary objective.  Dense unfinished-phase costs
        # prevent the no-step local optimum, while one-shot repeat penalties
        # still make exactly one successful attempt per foot preferable.
        self.rewards.sequential_foot_step_progress.weight = 30.0
        self.rewards.sequential_foot_step_target_exp.weight = 3.0
        self.rewards.sequential_active_foot_contact.weight = -20.0
        self.rewards.sequential_active_foot_clearance_l2.weight = -600.0
        self.rewards.sequential_foot_step_lift.weight = 80.0
        self.rewards.sequential_foot_step_landing_exp.weight = 250.0
        self.rewards.sequential_foot_step_completion.weight = 600.0
        self.rewards.sequential_foot_final_target_l2.weight = -250.0
        self.rewards.sequential_final_ankle_distance_exp.weight = 100.0
        self.rewards.sequential_support_foot_drift_l2.weight = -180.0
        self.rewards.sequential_repeated_lift_event = RewTerm(
            func=mdp.sequential_repeated_lift_event, weight=-120.0, params=step
        )
        self.rewards.sequential_step_count_excess = RewTerm(
            func=mdp.sequential_step_count_excess, weight=-1.0, params=step
        )
        self.rewards.sequential_exact_step_budget_success = RewTerm(
            func=mdp.sequential_exact_step_budget_success, weight=60.0, params=step
        )
        self.rewards.sequential_incomplete_step_penalty = RewTerm(
            func=mdp.sequential_incomplete_step_penalty, weight=-25.0, params=step
        )
        self.rewards.sequential_phase_time_excess_l2 = RewTerm(
            func=mdp.sequential_phase_time_excess_l2,
            weight=-2.0,
            params={**step, "grace_s": 1.5, "scale_s": 2.0},
        )
        self.rewards.sequential_active_contact_slide_l2 = RewTerm(
            func=mdp.sequential_active_contact_slide_l2,
            weight=-3.0,
            params={**step, "velocity_scale_mps": 0.05},
        )
        self.rewards.sequential_active_path_excess_l2 = RewTerm(
            func=mdp.sequential_active_path_excess_l2,
            weight=-1.0,
            params={**step, "path_ratio": 1.20, "margin_m": 0.02},
        )
        self.rewards.post_completion_torso_xy_l2 = RewTerm(
            func=mdp.sequential_post_completion_torso_xy_l2,
            weight=-120.0,
            params={**step, "torso_cfg": torso_cfg},
        )
        self.rewards.post_completion_torso_yaw_l2 = RewTerm(
            func=mdp.sequential_post_completion_torso_yaw_l2,
            weight=-60.0,
            params={**step, "torso_cfg": torso_cfg},
        )
        self.rewards.post_completion_airborne.weight = -100.0
        self.rewards.post_completion_foot_motion_l2.weight = -5.0
        self.rewards.post_completion_joint_vel_l2.weight = -0.05
        self.rewards.post_completion_action_rate_l2.weight = -0.20
        self.rewards.post_completion_contact_imbalance_l2.weight = -5.0
        self.terminations.sequential_pelvis_xy_out_of_bounds.params["max_displacement_m"] = 0.30


@configclass
class G1ArmHackStandFirstPrinciplesOneStepSingleEnvCfg_PLAY(
    G1ArmHackStandFirstPrinciplesOneStepSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1ArmHackWalkFirstPrinciplesSingleEnvCfg(G1WalkBehaviorFinetuneEnvCfg):
    """Acquisition stage: retain the gait while arms begin to move."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.upper_body_perturbation.source = "random_pose_trajectory"
        self.upper_body_perturbation.random_pose_bank_path = (
            STAND_RANDOM_POSE_BANK_RELATIVE_PATH.as_posix()
        )
        self.upper_body_perturbation.random_initialize_joint_state_on_reset = True
        self.upper_body_perturbation.random_curriculum_enabled = True
        self.upper_body_perturbation.random_curriculum_static_steps = 24_000
        self.upper_body_perturbation.random_curriculum_ramp_steps = 24_000
        self.upper_body_perturbation.random_curriculum_motion_scale = 1.0
        self.upper_body_perturbation.random_transition_duration_range_s = (1.5, 5.0)

        ankle_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        foot_cfg = SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)
        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )
        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")

        # Command sampler starts at exact zero, then transitions through the
        # useful <=0.4 band before full commands. No command-specific actors.
        self.commands.base_velocity.mode_probability = 0.70
        self.commands.base_velocity.hard_zero_stand = True
        self.commands.base_velocity.mode_command_clip_min = (-0.40, -0.40, -0.40)
        self.commands.base_velocity.mode_command_clip_max = (0.40, 0.40, 0.40)
        self.commands.base_velocity.reset_command_to_zero = True
        self.commands.base_velocity.smoothing_time_constant = 0.35
        self.commands.base_velocity.max_linear_accel = 0.45
        self.commands.base_velocity.max_yaw_accel = 0.60

        self.events.handoff_state_reset = EventTerm(
            func=mdp.reset_from_handoff_state_library,
            mode="reset",
            params={"state_library_path": "", "probability": 0.0},
        )
        self.events.random_end_effector_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                    preserve_order=True,
                ),
                "force_range": (-3.0, 3.0),
                "torque_range": (-0.5, 0.5),
            },
        )

        self.rewards.ankle_distance_30cm = RewTerm(
            func=mdp.ankle_distance_target_kernel,
            weight=8.0,
            params={"target_distance": 0.30, "std": 0.045, "asset_cfg": ankle_cfg},
        )
        self.rewards.useful_low_speed_tracking_l2 = RewTerm(
            func=mdp.useful_low_speed_tracking_l2,
            weight=-0.5,
            params={
                "command_name": "base_velocity",
                "deadband": 0.03,
                "max_command": 0.40,
                "linear_error_scale": 0.10,
                "yaw_error_scale": 0.15,
            },
        )
        self.rewards.pure_yaw_planar_drift_l2 = RewTerm(
            func=mdp.pure_yaw_planar_drift_l2,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.05,
                "velocity_scale": 0.08,
                "max_penalty": 4.0,
            },
        )
        self.rewards.pure_yaw_rate_error_l2 = RewTerm(
            func=mdp.pure_yaw_root_rate_error_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.05,
                "error_scale": 0.12,
                "max_penalty": 16.0,
            },
        )
        self.rewards.pure_yaw_torso_pitch_l2 = RewTerm(
            func=mdp.pure_yaw_torso_pitch_l2,
            weight=-0.5,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.05,
                "max_translation_command": 0.01,
                "pitch_scale": 0.08,
                "asset_cfg": torso_cfg,
            },
        )
        self.rewards.feet_swing_clearance_band_l2 = RewTerm(
            func=mdp.feet_swing_clearance_band_l2,
            weight=-0.75,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "target_height": 0.065,
                "std": 0.025,
                "air_time_threshold": 0.04,
                "command_threshold": 0.03,
                "max_height": 0.14,
            },
        )
        self.rewards.strict_zero_body_motion_l2.weight = -0.2
        self.rewards.strict_zero_feet_motion_l2.weight = -0.1
        self.rewards.strict_zero_joint_vel_l2.weight = -0.003
        self.rewards.strict_zero_double_support.weight = 0.2
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.8
        self.rewards.track_torso_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_torso_yaw_rate_exp.weight = 0.7
        self.rewards.termination_penalty.weight = -300.0


@configclass
class G1ArmHackWalkFirstPrinciplesSingleEnvCfg_PLAY(
    G1ArmHackWalkFirstPrinciplesSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1ArmHackWalkFirstPrinciplesStrictSingleEnvCfg(
    G1ArmHackWalkFirstPrinciplesSingleEnvCfg
):
    """Full stop/low-speed/pure-yaw/force objectives in the same Walk actor."""

    def __post_init__(self):
        super().__post_init__()
        self.upper_body_perturbation.random_curriculum_enabled = False
        self.upper_body_perturbation.random_curriculum_motion_scale = 1.0
        # Precision stage increases stop/yaw pressure without introducing the
        # full force range in one discontinuous jump. A later robust stage
        # reaches the final disturbance envelope.
        self.commands.base_velocity.mode_probability = 0.85
        self.events.random_end_effector_wrench.params["force_range"] = (-5.0, 5.0)
        self.events.random_end_effector_wrench.params["torque_range"] = (-0.75, 0.75)
        self.rewards.strict_zero_body_motion_l2.weight = -1.0
        self.rewards.strict_zero_feet_motion_l2.weight = -0.5
        self.rewards.strict_zero_joint_vel_l2.weight = -0.01
        self.rewards.strict_zero_double_support.weight = 0.8
        self.rewards.useful_low_speed_tracking_l2.weight = -1.0
        self.rewards.pure_yaw_planar_drift_l2.weight = -4.0
        self.rewards.pure_yaw_rate_error_l2.weight = -2.0
        self.rewards.pure_yaw_torso_pitch_l2.weight = -1.0
        self.rewards.feet_swing_clearance_band_l2.weight = -1.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_torso_lin_vel_xy_exp.weight = 1.2
        self.rewards.track_torso_yaw_rate_exp.weight = 0.8


@configclass
class G1ArmHackWalkFirstPrinciplesStrictSingleEnvCfg_PLAY(
    G1ArmHackWalkFirstPrinciplesStrictSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1ArmHackWalkFirstPrinciplesRobustSingleEnvCfg(
    G1ArmHackWalkFirstPrinciplesStrictSingleEnvCfg
):
    """Force-ramp stage before the final full disturbance envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.mode_probability = 0.88
        self.events.random_end_effector_wrench.params["force_range"] = (-8.0, 8.0)
        self.events.random_end_effector_wrench.params["torque_range"] = (-1.0, 1.0)
        self.events.random_torso_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "force_range": (-5.0, 5.0),
                "torque_range": (-0.75, 0.75),
            },
        )


@configclass
class G1ArmHackWalkFirstPrinciplesRobustSingleEnvCfg_PLAY(
    G1ArmHackWalkFirstPrinciplesRobustSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1ArmHackWalkFirstPrinciplesResponseSingleEnvCfg(
    G1ArmHackWalkFirstPrinciplesRobustSingleEnvCfg
):
    """Recover signed backward/lateral/yaw response without adding experts."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.mode_sampling_config_path = G1_WALK_RESPONSE_MODE_CONFIG_PATH
        self.commands.base_velocity.mode_probability = 0.97
        self.commands.base_velocity.mode_command_clip_min = (-0.45, -0.50, -0.90)
        self.commands.base_velocity.mode_command_clip_max = (0.80, 0.50, 0.90)
        self.rewards.command_response_shortfall_l1.weight = -4.0
        self.rewards.nonzero_single_stance.weight = 2.0
        self.rewards.useful_low_speed_tracking_l2.weight = -2.0
        self.rewards.pure_yaw_planar_drift_l2.weight = -6.0
        self.rewards.pure_yaw_rate_error_l2.weight = -5.0
        self.rewards.pure_yaw_torso_pitch_l2.weight = -2.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_torso_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_torso_yaw_rate_exp.weight = 1.5
        self.rewards.relative_command_response_shortfall_l1 = RewTerm(
            func=mdp.relative_command_response_shortfall_reward_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "epsilon": 0.03,
                "min_speed_fraction": 0.70,
                "min_lin_normalizer": 0.03,
                "min_yaw_normalizer": 0.08,
            },
        )


@configclass
class G1ArmHackWalkFirstPrinciplesResponseSingleEnvCfg_PLAY(
    G1ArmHackWalkFirstPrinciplesResponseSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
