"""ArmHack Stand foot-spacing recovery and push-robustness continuation."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs.g1_perturb_env import G1_LOWER_BODY_JOINT_NAMES
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1_FOOT_BODY_NAMES,
    G1_LOCOMOTION_JOINT_NAMES,
)

from .g1_stand_randomized_payload_env_cfg import G1StandRandomizedPayloadEnvCfg


@configclass
class G1StandFootRecoveryEnvCfg(G1StandRandomizedPayloadEnvCfg):
    """Recover a 30 cm stance from randomized initial spacing under pushes."""

    def __post_init__(self):
        super().__post_init__()

        perturbation = self.upper_body_perturbation
        perturbation.random_curriculum_enabled = False
        perturbation.random_curriculum_static_steps = 0
        perturbation.random_curriculum_ramp_steps = 0
        perturbation.random_curriculum_motion_scale = 1.0
        perturbation.random_transition_duration_range_s = (2.0, 6.0)
        self.sim.physx.enable_external_forces_every_iteration = True

        # Keep the policy input at 96 dimensions, but repurpose the existing
        # 3-D command observation as reset-relative [dx_b, dy_b, dyaw].  This
        # gives the actor the missing information needed to return after a push.
        self.commands.base_velocity = mdp.RelativePose2dCommandCfg(
            asset_name="robot",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=False,
            ranges=mdp.RelativePose2dCommandCfg.Ranges(radius=(0.0, 0.0), heading=(0.0, 0.0)),
            command_gain_xy=2.0,
            command_gain_yaw=1.5,
            command_clip_xy=0.50,
            command_clip_yaw=0.60,
        )
        self.rewards.track_torso_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_torso_yaw_rate_exp.weight = 1.5
        sequential_phase_obs_params = {"phase_action_index": 27, "lifted_action_index": 28}
        self.observations.policy.actions = ObsTerm(
            func=mdp.sequential_phase_augmented_last_action,
            params=sequential_phase_obs_params,
        )
        self.observations.critic.actions = ObsTerm(
            func=mdp.sequential_phase_augmented_last_action,
            params=sequential_phase_obs_params,
        )

        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
        pelvis_cfg = SceneEntityCfg("robot", body_names="pelvis")
        ankle_body_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        ankle_joint_cfg = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
        )
        leg_joint_names = [name for name in G1_LOWER_BODY_JOINT_NAMES if not name.startswith("waist_")]
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=leg_joint_names, preserve_order=True)
        leg_action_indices = [G1_LOCOMOTION_JOINT_NAMES.index(name) for name in leg_joint_names]
        foot_sensor_cfg = SceneEntityCfg(
            "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_with_random_stance,
            mode="reset",
            params={
                "distance_range": (0.08, 0.48),
                "close_distance_range": (0.08, 0.14),
                "close_stance_probability": 0.60,
                "nominal_distance_range": (0.28, 0.32),
                "nominal_stance_probability": 0.10,
                "asymmetric_support_probability": 0.0,
                "phase_one_probability": 0.0,
                "phase_two_probability": 0.0,
                "support_distance_range": (0.20, 0.32),
                "kinematic_nominal_distance": 0.237,
                "kinematic_distance_per_rad": 1.22,
                "position_scale_range": (0.95, 1.05),
                "velocity_range": (0.0, 0.0),
            },
        )
        # Phase sampling is performed atomically inside reset_robot_joints;
        # keeping a later independent event would resample an incompatible
        # phase after the asymmetric joint state had already been written.
        self.events.sequential_phase_training_reset = None
        self.events.random_torso_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": torso_cfg,
                "force_range": (-20.0, 20.0),
                "torque_range": (-3.0, 3.0),
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 7.0),
            is_global_time=False,
            params={
                "velocity_range": {
                    "x": (-0.35, 0.35),
                    "y": (-0.35, 0.35),
                    "yaw": (-0.45, 0.45),
                }
            },
        )

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_link_mass = None
        self.events.scale_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
                "stiffness_distribution_params": (0.90, 1.10),
                "damping_distribution_params": (0.90, 1.10),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.scale_joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
                "friction_distribution_params": (0.80, 1.20),
                "armature_distribution_params": (0.90, 1.10),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.terminations.sequential_pelvis_xy_out_of_bounds = DoneTerm(
            func=mdp.sequential_pelvis_xy_out_of_bounds,
            params={"max_displacement_m": 0.20, "pelvis_cfg": pelvis_cfg},
        )

        self.rewards.ankle_distance_l1 = RewTerm(
            func=mdp.ankle_distance_l1,
            weight=-12.0,
            params={"target_distance": 0.30, "asset_cfg": ankle_body_cfg},
        )
        self.rewards.ankle_distance_exp = RewTerm(
            func=mdp.ankle_distance_exp,
            weight=8.0,
            params={"target_distance": 0.30, "std": 0.06, "asset_cfg": ankle_body_cfg},
        )
        self.rewards.ankle_distance_success = RewTerm(
            func=mdp.ankle_distance_success,
            weight=8.0,
            params={
                "target_distance": 0.30,
                "tolerance": 0.03,
                "asset_cfg": ankle_body_cfg,
            },
        )
        self.rewards.ankle_longitudinal_alignment_l2 = RewTerm(
            func=mdp.ankle_longitudinal_alignment_l2,
            weight=-0.5,
            params={"asset_cfg": ankle_body_cfg},
        )
        self.rewards.ankle_torques_l2 = RewTerm(
            func=mdp.ankle_torques_l2_near_target,
            weight=-1.0e-4,
            params={
                "target_distance": 0.30,
                "gate_std": 0.06,
                "ankle_body_cfg": ankle_body_cfg,
                "ankle_joint_cfg": ankle_joint_cfg,
            },
        )
        self.rewards.ankle_transition_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=0.0,
            params={"asset_cfg": ankle_joint_cfg},
        )
        self.rewards.lower_body_joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=0.0,
            params={"asset_cfg": leg_joint_cfg},
        )
        self.rewards.lower_body_joint_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=0.0,
            params={"asset_cfg": leg_joint_cfg},
        )
        self.rewards.lower_body_action_rate_l2 = RewTerm(
            func=mdp.action_rate_l2_selected,
            weight=0.0,
            params={"action_indices": leg_action_indices},
        )
        self.rewards.lower_body_position_target_error_l2 = RewTerm(
            func=mdp.joint_position_target_error_l2_selected,
            weight=0.0,
            params={
                "action_indices": leg_action_indices,
                "action_scale": 0.25,
                "asset_cfg": leg_joint_cfg,
            },
        )
        self.rewards.ankle_separation_speed_l2 = RewTerm(
            func=mdp.ankle_separation_speed_l2,
            weight=0.0,
            params={"asset_cfg": ankle_body_cfg},
        )
        self.rewards.foot_contact_force_excess_l2 = RewTerm(
            func=mdp.foot_contact_force_excess_l2,
            weight=0.0,
            params={"sensor_cfg": foot_sensor_cfg, "threshold_n": 260.0, "force_scale_n": 100.0},
        )
        sequential_step_params = {
            "pelvis_cfg": pelvis_cfg,
            "foot_cfg": ankle_body_cfg,
            "sensor_cfg": foot_sensor_cfg,
            "lateral_target_offset_m": 0.15,
            "min_clearance_m": 0.035,
            "landing_tolerance_m": 0.055,
            "min_step_duration_s": 0.45,
        }
        self.rewards.sequential_foot_step_progress = RewTerm(
            func=mdp.sequential_foot_step_progress,
            weight=0.0,
            params={**sequential_step_params, "progress_scale_m": 0.02},
        )
        self.rewards.sequential_foot_step_target_exp = RewTerm(
            func=mdp.sequential_foot_step_target_exp,
            weight=0.0,
            params={**sequential_step_params, "std": 0.04},
        )
        self.rewards.sequential_foot_step_clearance_exp = RewTerm(
            func=mdp.sequential_foot_step_clearance_exp,
            weight=0.0,
            params={**sequential_step_params, "target_clearance_m": 0.055, "std": 0.025},
        )
        self.rewards.sequential_active_foot_contact = RewTerm(
            func=mdp.sequential_active_foot_contact,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.sequential_active_foot_clearance_l2 = RewTerm(
            func=mdp.sequential_active_foot_clearance_l2,
            weight=0.0,
            params={**sequential_step_params, "target_clearance_m": 0.055, "max_clearance_m": 0.11},
        )
        self.rewards.sequential_active_foot_upward_velocity = RewTerm(
            func=mdp.sequential_active_foot_upward_velocity,
            weight=0.0,
            params={
                **sequential_step_params,
                "target_upward_velocity_mps": 0.30,
                "pelvis_gate_std_m": 0.06,
                "target_gate_std_m": 0.12,
            },
        )
        self.rewards.sequential_active_foot_velocity_l2 = RewTerm(
            func=mdp.sequential_active_foot_velocity_l2,
            weight=0.0,
            params={
                **sequential_step_params,
                "max_horizontal_speed_mps": 0.35,
                "max_vertical_speed_mps": 0.30,
            },
        )
        self.rewards.sequential_active_foot_single_support = RewTerm(
            func=mdp.sequential_active_foot_single_support,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.sequential_foot_step_landing_exp = RewTerm(
            func=mdp.sequential_foot_step_landing_exp,
            weight=0.0,
            params={**sequential_step_params, "target_std": 0.03, "height_std": 0.025},
        )
        self.rewards.sequential_foot_step_completion = RewTerm(
            func=mdp.sequential_foot_step_completion,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.sequential_foot_step_lift = RewTerm(
            func=mdp.sequential_foot_step_lift,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.sequential_foot_step_order_violation = RewTerm(
            func=mdp.sequential_foot_step_order_violation,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.sequential_foot_final_target_l2 = RewTerm(
            func=mdp.sequential_foot_final_target_l2,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.sequential_final_ankle_distance_exp = RewTerm(
            func=mdp.sequential_final_ankle_distance_exp,
            weight=0.0,
            params={**sequential_step_params, "target_distance_m": 0.30, "std": 0.015},
        )
        self.rewards.sequential_support_foot_drift_l2 = RewTerm(
            func=mdp.sequential_support_foot_drift_l2,
            weight=0.0,
            params=sequential_step_params,
        )
        self.rewards.torso_xy_position_l2 = RewTerm(
            func=mdp.torso_xy_position_l2,
            weight=-0.5,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_yaw_l2 = RewTerm(
            func=mdp.torso_yaw_l2,
            weight=-0.5,
            params={"target_yaw": 0.0, "asset_cfg": torso_cfg},
        )
        self.rewards.torso_xy_position_near_stance_l2 = RewTerm(
            func=mdp.torso_xy_position_near_stance_l2,
            weight=-12.0,
            params={
                "target_distance": 0.30,
                "gate_std": 0.06,
                "torso_cfg": torso_cfg,
                "ankle_cfg": ankle_body_cfg,
            },
        )
        self.rewards.torso_yaw_near_stance_l2 = RewTerm(
            func=mdp.torso_yaw_near_stance_l2,
            weight=-6.0,
            params={
                "target_distance": 0.30,
                "gate_std": 0.06,
                "torso_cfg": torso_cfg,
                "ankle_cfg": ankle_body_cfg,
            },
        )
        self.rewards.root_xy_position_l2.weight = -0.25
        self.rewards.double_support.weight = 0.02
        self.rewards.feet_slide.weight = -0.02
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

        self.curriculum.stance_recovery = CurrTerm(
            func=mdp.stance_recovery_curriculum,
            params={
                "reward_weight_schedules": {
                    "ankle_distance_l1": ((0, -12.0), (12000, -12.0), (32000, -8.0)),
                    "ankle_distance_exp": ((0, 8.0), (12000, 8.0), (32000, 6.0)),
                    "ankle_distance_success": ((0, 8.0), (12000, 8.0), (32000, 6.0)),
                    "torso_xy_position_l2": ((0, -2.0), (4000, -3.0), (12000, -6.0)),
                    "torso_yaw_l2": ((0, -1.0), (4000, -1.5), (12000, -3.0)),
                    "root_xy_position_l2": ((0, -1.0), (4000, -1.5), (12000, -3.0)),
                    "feet_slide": ((0, -0.04), (4000, -0.04), (12000, -0.12)),
                    "double_support": ((0, 0.04), (4000, 0.04), (12000, 0.10)),
                    "ankle_torques_l2": ((0, -1.0e-3), (12000, -1.0e-3)),
                },
                "wrench_force_abs_schedule": ((0, 5.0), (4000, 10.0), (12000, 20.0)),
                "wrench_torque_abs_schedule": ((0, 0.75), (4000, 1.5), (12000, 3.0)),
                "push_xy_abs_schedule": ((0, 0.10), (4000, 0.15), (12000, 0.35)),
                "push_yaw_abs_schedule": ((0, 0.12), (4000, 0.20), (12000, 0.45)),
                "step_offset": 0,
            },
        )


@configclass
class G1StandFootRecoveryEnvCfg_PLAY(G1StandFootRecoveryEnvCfg):
    """Small evaluation variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
