"""G1 AMP stand perturbation environment configs."""

from __future__ import annotations

import math

from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs.g1_perturb_env import G1_UPPER_BODY_JOINT_NAMES, UpperBodyPerturbationCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg,
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

        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.track_torso_lin_vel_xy_exp = None
        self.rewards.track_torso_yaw_rate_exp = None
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

        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.torso_roll_pitch_l2.weight = -4.0
        if self.rewards.torso_ang_vel_xy_l2 is not None:
            self.rewards.torso_ang_vel_xy_l2.weight = -0.20
        if self.rewards.torso_vertical_velocity_l2 is not None:
            self.rewards.torso_vertical_velocity_l2.weight = -0.60
        self.rewards.torso_height_band_l2.weight = -1.20
        if self.rewards.torso_specific_force_xy_l2 is not None:
            self.rewards.torso_specific_force_xy_l2.weight = -0.03
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -1.20
        if self.rewards.ang_vel_xy_l2 is not None:
            self.rewards.ang_vel_xy_l2.weight = -0.20
        if self.rewards.feet_slide is not None:
            self.rewards.feet_slide.weight = -0.20
        if self.rewards.joint_deviation_hip is not None:
            self.rewards.joint_deviation_hip.weight = -0.16
        if self.rewards.joint_deviation_waist is not None:
            self.rewards.joint_deviation_waist.weight = -0.18
        if self.rewards.termination_penalty is not None:
            self.rewards.termination_penalty.weight = -250.0


@configclass
class G1StandPerturbEnvCfg_PLAY(G1StandPerturbEnvCfg):
    """Reduced-env stand perturbation play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
