"""First-principles ArmHack Stand and Walk tasks, one actor per task."""

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

from .g1_stand_foot_recovery_env_cfg import G1StandFootRecoveryEnvCfg
from .g1_walk_behavior_env_cfg import G1WalkBehaviorFinetuneEnvCfg
from .reference_data import STAND_RANDOM_POSE_BANK_RELATIVE_PATH


def _step_params(pelvis_cfg, ankle_cfg, contact_cfg) -> dict:
    return {
        "pelvis_cfg": pelvis_cfg,
        "foot_cfg": ankle_cfg,
        "sensor_cfg": contact_cfg,
        "lateral_target_offset_m": 0.15,
        "min_clearance_m": 0.035,
        "landing_tolerance_m": 0.025,
        "min_step_duration_s": 0.40,
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

        # Broad initial stance and velocity distribution. Phase-zero dominates;
        # phase-one/final samples keep the second step and planted hold trainable.
        self.events.reset_robot_joints.params.update(
            {
                "distance_range": (0.08, 0.46),
                "close_distance_range": (0.08, 0.16),
                "close_stance_probability": 0.45,
                "nominal_distance_range": (0.27, 0.33),
                "nominal_stance_probability": 0.15,
                "asymmetric_support_probability": 0.20,
                "phase_one_probability": 0.20,
                "phase_two_probability": 0.20,
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
                "force_range": (-15.0, 15.0),
                "torque_range": (-2.0, 2.0),
            },
        )

        # Two real single-foot swings, accurate landing, fixed support foot.
        self.rewards.sequential_foot_step_progress.weight = 12.0
        self.rewards.sequential_foot_step_target_exp.weight = 14.0
        self.rewards.sequential_foot_step_clearance_exp.weight = 10.0
        self.rewards.sequential_active_foot_clearance_l2.weight = -450.0
        self.rewards.sequential_active_foot_upward_velocity.weight = 3.0
        self.rewards.sequential_active_foot_velocity_l2.weight = -12.0
        self.rewards.sequential_active_foot_single_support.weight = 6.0
        self.rewards.sequential_active_foot_contact.weight = -5.0
        self.rewards.sequential_foot_step_landing_exp.weight = 35.0
        self.rewards.sequential_foot_step_completion.weight = 80.0
        self.rewards.sequential_foot_step_lift.weight = 45.0
        self.rewards.sequential_foot_step_order_violation.weight = -30.0
        self.rewards.sequential_foot_final_target_l2.weight = -120.0
        self.rewards.sequential_final_ankle_distance_exp.weight = 45.0
        self.rewards.sequential_support_foot_drift_l2.weight = -220.0

        # SE(2) is reset-relative and remains active through the whole episode.
        self.rewards.torso_xy_position_l2.weight = -35.0
        self.rewards.torso_yaw_l2.weight = -18.0
        self.rewards.torso_xy_position_near_stance_l2.weight = -80.0
        self.rewards.torso_yaw_near_stance_l2.weight = -35.0
        self.rewards.root_xy_position_l2.weight = 0.0
        self.terminations.sequential_pelvis_xy_out_of_bounds.params["max_displacement_m"] = 0.25

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
        self.rewards.ankle_distance_l1.weight = -4.0
        self.rewards.ankle_distance_exp.weight = 4.0
        self.rewards.ankle_distance_success.weight = 6.0
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
class G1ArmHackWalkFirstPrinciplesSingleEnvCfg(G1WalkBehaviorFinetuneEnvCfg):
    """One actor covers stop, low speed, general walking and pure yaw."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.upper_body_perturbation.source = "random_pose_trajectory"
        self.upper_body_perturbation.random_pose_bank_path = (
            STAND_RANDOM_POSE_BANK_RELATIVE_PATH.as_posix()
        )
        self.upper_body_perturbation.random_initialize_joint_state_on_reset = True
        self.upper_body_perturbation.random_curriculum_enabled = False
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
        self.commands.base_velocity.mode_probability = 0.90
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
                "force_range": (-15.0, 15.0),
                "torque_range": (-2.0, 2.0),
            },
        )

        self.rewards.ankle_distance_30cm = RewTerm(
            func=mdp.ankle_distance_target_kernel,
            weight=8.0,
            params={"target_distance": 0.30, "std": 0.045, "asset_cfg": ankle_cfg},
        )
        self.rewards.useful_low_speed_tracking_l2 = RewTerm(
            func=mdp.useful_low_speed_tracking_l2,
            weight=-2.5,
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
            weight=-8.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.05,
                "velocity_scale": 0.08,
                "max_penalty": 4.0,
            },
        )
        self.rewards.pure_yaw_rate_error_l2 = RewTerm(
            func=mdp.pure_yaw_root_rate_error_l2,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.05,
                "error_scale": 0.12,
                "max_penalty": 16.0,
            },
        )
        self.rewards.pure_yaw_torso_pitch_l2 = RewTerm(
            func=mdp.pure_yaw_torso_pitch_l2,
            weight=-3.0,
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
            weight=-1.5,
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
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.track_torso_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_torso_yaw_rate_exp.weight = 1.0
        self.rewards.termination_penalty.weight = -300.0


@configclass
class G1ArmHackWalkFirstPrinciplesSingleEnvCfg_PLAY(
    G1ArmHackWalkFirstPrinciplesSingleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
