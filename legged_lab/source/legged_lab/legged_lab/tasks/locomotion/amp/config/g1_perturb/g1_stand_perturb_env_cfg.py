"""G1 AMP stand perturbation environment configs."""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs.g1_perturb_env import G1_UPPER_BODY_JOINT_NAMES, UpperBodyPerturbationCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg,
    G1_FOOT_BODY_NAMES,
)
from .g1_walk_perturb_env_cfg import _configure_perturbation_common
from .reference_data import STAND_ARM_MOTION_RELATIVE_PATH


STAND_PERTURB_CSV_PATH = STAND_ARM_MOTION_RELATIVE_PATH.as_posix()


@configclass
class G1StandPerturbEnvCfg(G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg):
    """Strict zero-command balance task with scripted upper-body disturbances."""

    upper_body_perturbation: UpperBodyPerturbationCfg = UpperBodyPerturbationCfg(
        joint_names=G1_UPPER_BODY_JOINT_NAMES,
        source="csv",
        csv_path=STAND_PERTURB_CSV_PATH,
        csv_time_column="time_s",
        csv_loop=False,
        csv_use_g1_action_order_q_columns=False,
        csv_q_column_joint_order="sdk",
        csv_randomize_start_on_reset=True,
        csv_end_margin_s=0.25,
        csv_interpolate=True,
        csv_initialize_joint_state_on_reset=True,
        csv_curriculum_enabled=True,
        # G1 AMP collects 24 control steps per PPO iteration. The first 500
        # iterations use random static poses; the next 1000 ramp continuously
        # to quarter-speed motion.
        csv_curriculum_static_steps=12_000,
        csv_curriculum_ramp_steps=24_000,
        csv_curriculum_motion_scale=0.25,
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_perturbation_common(self)

        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=1.0,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(-math.pi, math.pi),
            ),
        )

        self.events.reset_from_ref = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
        foot_asset_cfg = SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)
        foot_sensor_cfg = SceneEntityCfg(
            "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
        )
        ankle_limit_cfg = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
        )
        hip_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])
        waist_cfg = SceneEntityCfg("robot", joint_names="waist_.*_joint")

        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)
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
            weight=0.25,
            params={"sensor_cfg": foot_sensor_cfg},
        )
        self.rewards.torso_lateral_vel_cmd_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.feet_swing_clearance_band_l2 = None
        self.rewards.gait_timing_symmetry_l1 = None
        self.rewards.directional_speed_floor_l1 = None
        self.rewards.directional_double_air_l1 = None
        self.rewards.backward_single_stance = None
        self.rewards.backward_double_stance_l1 = None
        self.rewards.directional_double_stance_support = None
        self.rewards.directional_velocity_leak_l1 = None

        # Recreate every Stand reward explicitly. The walking parent deliberately
        # sets several of these terms to None, so changing weights conditionally
        # would silently leave them disabled.
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        self.rewards.torso_roll_pitch_l2 = RewTerm(
            func=mdp.torso_roll_pitch_l2,
            weight=-3.0,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_ang_vel_xy_l2 = RewTerm(
            func=mdp.torso_ang_vel_xy_l2,
            weight=-0.15,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_vertical_velocity_l2 = RewTerm(
            func=mdp.torso_vertical_velocity_l2,
            weight=-0.30,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.torso_height_band_l2 = RewTerm(
            func=mdp.torso_height_band_l2,
            weight=-0.60,
            params={
                "target_height": 0.84,
                "lower_deadband": -0.02,
                "upper_deadband": 0.04,
                "std": 0.04,
                "asset_cfg": torso_cfg,
            },
        )
        self.rewards.torso_specific_force_xy_l2 = RewTerm(
            func=mdp.torso_specific_force_xy_l2,
            weight=-0.01,
            params={"asset_cfg": torso_cfg},
        )
        self.rewards.root_xy_position_l2 = RewTerm(func=mdp.root_xy_position_l2, weight=-1.0)
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.30)
        self.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": ankle_limit_cfg},
        )
        self.rewards.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.10,
            params={"asset_cfg": hip_cfg},
        )
        self.rewards.joint_deviation_waist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.12,
            params={"asset_cfg": waist_cfg},
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={"sensor_cfg": foot_sensor_cfg, "asset_cfg": foot_asset_cfg},
        )
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1StandPerturbEnvCfg_PLAY(G1StandPerturbEnvCfg):
    """Reduced-env stand perturbation play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
