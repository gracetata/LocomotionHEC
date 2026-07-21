"""Full-body G1 recovery-and-hold Stand task.

This module is intentionally independent from ArmHack.  The actor observes the
standard 96-D G1 AMP policy vector and directly controls all 29 policy joints;
there is no scripted arm target and no action overwrite after inference.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg,
    G1_FOOT_BODY_NAMES,
    G1_LOCOMOTION_JOINT_NAMES,
)

from . import rewards as recovery_rewards


G1_LEG_JOINT_NAMES = [
    name
    for name in G1_LOCOMOTION_JOINT_NAMES
    if any(token in name for token in ("hip", "knee", "ankle"))
]
G1_WAIST_JOINT_NAMES = [name for name in G1_LOCOMOTION_JOINT_NAMES if name.startswith("waist_")]
G1_ARM_JOINT_NAMES = [
    name
    for name in G1_LOCOMOTION_JOINT_NAMES
    if any(token in name for token in ("shoulder", "elbow", "wrist"))
]

PERTURBED_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_knee_link",
    "right_knee_link",
]

DEFAULT_CARTESIAN_KEY_BODY_NAMES = [
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


@configclass
class G1ExtremeStandRecoveryEnvCfg(G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg):
    """Recover from randomized initial state and repeated multi-body disturbances."""

    def __post_init__(self):
        super().__post_init__()

        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=G1_LEG_JOINT_NAMES, preserve_order=True)
        waist_joint_cfg = SceneEntityCfg("robot", joint_names=G1_WAIST_JOINT_NAMES, preserve_order=True)
        arm_joint_cfg = SceneEntityCfg("robot", joint_names=G1_ARM_JOINT_NAMES, preserve_order=True)
        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
        foot_asset_cfg = SceneEntityCfg(
            "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )
        foot_sensor_cfg = SceneEntityCfg(
            "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )
        cartesian_key_body_cfg = SceneEntityCfg(
            "robot", body_names=DEFAULT_CARTESIAN_KEY_BODY_NAMES, preserve_order=True
        )

        # Keep the original deployment contract explicit at the task boundary.
        self.actions.joint_pos.joint_names = G1_LOCOMOTION_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params = {"asset_cfg": full_joint_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": full_joint_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": full_joint_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": full_joint_cfg}
        self.observations.disc.joint_pos.params = {"asset_cfg": full_joint_cfg}
        self.observations.disc.joint_vel.params = {"asset_cfg": full_joint_cfg}

        # This is a zero-command recovery-and-hold task, not a walking task.
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            rel_standing_envs=1.0,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(0.0, 0.0),
            ),
        )
        self.events.reset_from_ref = None

        # Cache the unperturbed asset-default Cartesian pose before any reset
        # noise is sampled.  The foot reference is separate so its exact
        # default planar distance is available without a hard-coded number.
        self.events.cache_default_cartesian_pose = EventTerm(
            func=mdp.cache_default_key_body_offsets,
            mode="startup",
            params={
                "asset_cfg": cartesian_key_body_cfg,
                "reference_attr": recovery_rewards.DEFAULT_CARTESIAN_REFERENCE_ATTR,
            },
        )
        self.events.cache_default_feet_pose = EventTerm(
            func=mdp.cache_default_key_body_offsets,
            mode="startup",
            params={
                "asset_cfg": foot_asset_cfg,
                "reference_attr": recovery_rewards.DEFAULT_FEET_REFERENCE_ATTR,
            },
        )

        # Additive joint offsets are required: multiplicative noise would leave
        # every zero-valued default joint unchanged.  The three disjoint reset
        # terms cover all 29 controlled joints with group-appropriate ranges.
        self.events.reset_robot_joints = None
        self.events.reset_leg_joints_with_noise = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": leg_joint_cfg,
                "position_range": (-0.25, 0.25),
                "velocity_range": (-1.0, 1.0),
            },
        )
        self.events.reset_waist_joints_with_noise = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": waist_joint_cfg,
                "position_range": (-0.35, 0.35),
                "velocity_range": (-1.25, 1.25),
            },
        )
        self.events.reset_arm_joints_with_noise = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": arm_joint_cfg,
                "position_range": (-0.60, 0.60),
                "velocity_range": (-1.50, 1.50),
            },
        )
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.08, 0.08),
            "roll": (-0.25, 0.25),
            "pitch": (-0.25, 0.25),
            "yaw": (-0.30, 0.30),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.50, 0.50),
            "y": (-0.50, 0.50),
            "z": (-0.35, 0.35),
            "roll": (-0.80, 0.80),
            "pitch": (-0.80, 0.80),
            "yaw": (-0.60, 0.60),
        }

        # Random dynamics are sampled independently for each environment.
        self.events.physics_material.params["static_friction_range"] = (0.50, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.40, 1.00)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="torso_link"
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["pelvis", "torso_link"], preserve_order=True
        )
        self.events.randomize_rigid_body_com.params["com_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
            "z": (-0.04, 0.04),
        }
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.75, 1.25)
        self.events.scale_actuator_gains.params.update(
            {
                "asset_cfg": full_joint_cfg,
                "stiffness_distribution_params": (0.75, 1.25),
                "damping_distribution_params": (0.75, 1.25),
                "distribution": "uniform",
            }
        )
        self.events.scale_joint_parameters.params.update(
            {
                "asset_cfg": full_joint_cfg,
                "friction_distribution_params": (0.50, 1.50),
                "armature_distribution_params": (0.70, 1.30),
                "distribution": "uniform",
            }
        )

        # Clear every persistent wrench on reset, then use independent clocks
        # and magnitudes for the torso, pelvis, arms and legs.  The actor is not
        # told which body will be disturbed next.
        self.sim.physx.enable_external_forces_every_iteration = True
        self.events.base_external_force_torque.params.update(
            {
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=PERTURBED_BODY_NAMES, preserve_order=True
                ),
                "force_range": (0.0, 0.0),
                "torque_range": (0.0, 0.0),
            }
        )
        self.events.random_torso_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": torso_cfg,
                "force_range": (-35.0, 35.0),
                "torque_range": (-5.0, 5.0),
            },
        )
        self.events.random_pelvis_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.5, 5.5),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "force_range": (-30.0, 30.0),
                "torque_range": (-4.0, 4.0),
            },
        )
        self.events.random_arm_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(1.5, 4.5),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_shoulder_pitch_link",
                        "right_shoulder_pitch_link",
                        "left_elbow_link",
                        "right_elbow_link",
                    ],
                    preserve_order=True,
                ),
                "force_range": (-12.0, 12.0),
                "torque_range": (-2.0, 2.0),
            },
        )
        self.events.random_leg_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_hip_pitch_link",
                        "right_hip_pitch_link",
                        "left_knee_link",
                        "right_knee_link",
                    ],
                    preserve_order=True,
                ),
                "force_range": (-12.0, 12.0),
                "torque_range": (-2.0, 2.0),
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            is_global_time=False,
            params={
                "velocity_range": {
                    "x": (-0.60, 0.60),
                    "y": (-0.60, 0.60),
                    "z": (-0.25, 0.25),
                    "roll": (-1.0, 1.0),
                    "pitch": (-1.0, 1.0),
                    "yaw": (-0.70, 0.70),
                }
            },
        )

        # Replace walking objectives with recovery, stillness, uprightness and
        # default-pose objectives.  Small root-XY cost allows a recovery step
        # but discourages indefinite drift after the disturbance is gone.
        for name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "torso_lateral_vel_cmd_l2",
            "feet_air_time",
            "feet_swing_clearance_band_l2",
            "gait_timing_symmetry_l1",
            "directional_speed_floor_l1",
            "directional_double_air_l1",
            "backward_single_stance",
            "backward_double_stance_l1",
            "directional_double_stance_support",
            "directional_velocity_leak_l1",
            "arm_style_prior",
        ):
            setattr(self.rewards, name, None)

        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)
        self.rewards.default_joint_pose_exp = RewTerm(
            func=recovery_rewards.default_joint_pose_exp,
            weight=5.0,
            params={"std": 0.25, "asset_cfg": full_joint_cfg},
        )
        self.rewards.default_leg_joint_pose_exp = RewTerm(
            func=recovery_rewards.default_joint_pose_exp,
            weight=3.0,
            params={"std": 0.18, "asset_cfg": leg_joint_cfg},
        )
        self.rewards.default_key_body_pose_exp = RewTerm(
            func=recovery_rewards.default_key_body_pose_exp,
            weight=2.5,
            params={
                "std": 0.12,
                "asset_cfg": cartesian_key_body_cfg,
                "reference_attr": recovery_rewards.DEFAULT_CARTESIAN_REFERENCE_ATTR,
            },
        )
        self.rewards.default_feet_distance_l2 = RewTerm(
            func=recovery_rewards.default_feet_distance_l2,
            weight=-8.0,
            params={
                "asset_cfg": foot_asset_cfg,
                "reference_attr": recovery_rewards.DEFAULT_FEET_REFERENCE_ATTR,
            },
        )
        self.rewards.track_torso_lin_vel_xy_exp = RewTerm(
            func=mdp.track_torso_lin_vel_xy_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.20, "asset_cfg": torso_cfg},
        )
        self.rewards.track_torso_yaw_rate_exp = RewTerm(
            func=mdp.track_torso_yaw_rate_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": 0.20, "asset_cfg": torso_cfg},
        )
        self.rewards.double_support = RewTerm(
            func=mdp.double_support,
            weight=0.30,
            params={"sensor_cfg": foot_sensor_cfg},
        )
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
        self.rewards.torso_roll_pitch_l2 = RewTerm(
            func=mdp.torso_roll_pitch_l2,
            weight=-4.0,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_ang_vel_xy_l2 = RewTerm(
            func=mdp.torso_ang_vel_xy_l2,
            weight=-0.25,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_vertical_velocity_l2 = RewTerm(
            func=mdp.torso_vertical_velocity_l2,
            weight=-0.35,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_height_band_l2 = RewTerm(
            func=mdp.torso_height_band_l2,
            weight=-0.80,
            params={
                "target_height": 0.84,
                "lower_deadband": -0.03,
                "upper_deadband": 0.05,
                "std": 0.05,
                "asset_cfg": torso_cfg,
            },
        )
        self.rewards.torso_specific_force_xy_l2 = RewTerm(
            func=mdp.torso_specific_force_xy_l2,
            weight=-0.01,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.root_xy_position_l2 = RewTerm(func=mdp.root_xy_position_l2, weight=-0.35)
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.30)
        self.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
        self.rewards.dof_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-2.0e-6,
            params={"asset_cfg": full_joint_cfg},
        )
        self.rewards.dof_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-1.0e-7,
            params={"asset_cfg": full_joint_cfg},
        )
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_waist = None
        self.rewards.joint_deviation_arms = None
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.20,
            params={"sensor_cfg": foot_sensor_cfg, "asset_cfg": foot_asset_cfg},
        )
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)


@configclass
class G1ExtremeStandRecoveryEnvCfg_PLAY(G1ExtremeStandRecoveryEnvCfg):
    """Small play/evaluation variant with the same policy and disturbance contract."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
