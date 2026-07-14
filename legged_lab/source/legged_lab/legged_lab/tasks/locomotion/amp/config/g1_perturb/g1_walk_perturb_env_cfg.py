"""G1 AMP walk perturbation environment configs."""

from __future__ import annotations

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
import legged_lab.tasks.locomotion.velocity.mdp as velocity_mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.envs.g1_perturb_env import (
    G1_LOWER_BODY_JOINT_NAMES,
    G1_UPPER_BODY_JOINT_NAMES,
    UpperBodyPerturbationCfg,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg,
    G1_LOCOMOTION_JOINT_NAMES,
    G1_NAV2_AUGMENTED_CMD_DATA_PATH,
)


G1_LOWER_BODY_JOINT_IDS = [G1_LOCOMOTION_JOINT_NAMES.index(name) for name in G1_LOWER_BODY_JOINT_NAMES]
G1_NAV2_RAW_SUCCESS_CMD_DATA_PATH = os.path.abspath(
    os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "..",
        "..",
        "..",
        "nav2_loopback_actual",
        "actual_raw_success",
        "all_cmd_vel_success.csv",
    )
)


def _resolve_walk_perturb_cmd_data_path() -> str:
    if os.path.isfile(G1_NAV2_AUGMENTED_CMD_DATA_PATH):
        return G1_NAV2_AUGMENTED_CMD_DATA_PATH
    return G1_NAV2_RAW_SUCCESS_CMD_DATA_PATH


G1_WALK_PERTURB_CMD_DATA_PATH = _resolve_walk_perturb_cmd_data_path()


def _paired_arm_pose(left: list[float], right: list[float]) -> list[float]:
    if len(left) != 7 or len(right) != 7:
        raise ValueError("Each arm pose must have exactly 7 joint values.")
    return [
        left[0],
        right[0],
        left[1],
        right[1],
        left[2],
        right[2],
        left[3],
        right[3],
        left[4],
        right[4],
        left[5],
        right[5],
        left[6],
        right[6],
    ]


G1_WALK_PERTURB_POSE_SET = [
    _paired_arm_pose(
        [0.91, 0.52, 0.11, 0.01, -0.12, -1.03, 0.01],
        [0.91, -0.52, -0.11, 0.01, 0.12, -1.03, -0.01],
    ),
    _paired_arm_pose(
        [0.2504, 0.2650, -0.0919, 0.8356, 0.0031, 0.0104, -0.0102],
        [0.2504, -0.2650, 0.0919, 0.8356, -0.0031, 0.0104, 0.0102],
    ),
    _paired_arm_pose(
        [0.27, 0.79, -0.22, -0.49, 0.85, 0.40, 0.05],
        [0.27, -0.79, 0.22, -0.49, -0.85, 0.40, -0.05],
    ),
]


def _configure_perturbation_common(cfg) -> None:
    lower_body_joint_cfg = SceneEntityCfg("robot", joint_names=G1_LOWER_BODY_JOINT_NAMES, preserve_order=True)

    cfg.actions.joint_pos.joint_names = G1_LOCOMOTION_JOINT_NAMES
    cfg.actions.joint_pos.preserve_order = True

    cfg.observations.disc.joint_pos.params = {"asset_cfg": lower_body_joint_cfg}
    cfg.observations.disc.joint_vel.params = {"asset_cfg": lower_body_joint_cfg}
    cfg.observations.disc_demo.ref_joint_pos.params["joint_ids"] = G1_LOWER_BODY_JOINT_IDS
    cfg.observations.disc_demo.ref_joint_vel.params["joint_ids"] = G1_LOWER_BODY_JOINT_IDS

    cfg.rewards.dof_torques_l2.params = {"asset_cfg": lower_body_joint_cfg}
    cfg.rewards.dof_acc_l2.params = {"asset_cfg": lower_body_joint_cfg}
    cfg.rewards.action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-0.005,
        params={"action_indices": G1_LOWER_BODY_JOINT_IDS},
    )
    cfg.rewards.joint_deviation_arms = None
    cfg.rewards.arm_style_prior = None


@configclass
class G1WalkPerturbFinetuneEnvCfg(G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg):
    """Walk task that treats upper-body motion as an internal perturbation."""

    upper_body_perturbation: UpperBodyPerturbationCfg = UpperBodyPerturbationCfg(
        joint_names=G1_UPPER_BODY_JOINT_NAMES,
        source="pose_set",
        pose_set=G1_WALK_PERTURB_POSE_SET,
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_perturbation_common(self)

        self.commands.base_velocity = mdp.Nav2RecordedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 2.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            data_path=G1_WALK_PERTURB_CMD_DATA_PATH,
            augmentation_filter="none,mirror_lr",
            dataset_sample_dt=0.05,
            window_duration_s=2.0,
            command_scale=(0.70, 0.55, 0.55),
            command_clip_min=(-0.20, -0.30, -0.60),
            command_clip_max=(0.60, 0.30, 0.60),
            smoothing_time_constant=0.30,
            max_linear_accel=0.60,
            max_yaw_accel=0.80,
            reset_command_to_zero=True,
            sample_groups_uniformly=True,
            scenario_family_weights={
                "sample500_random": 1.0,
                "sharp_turn": 2.4,
                "hard_turn": 2.0,
                "complex_turn": 1.6,
                "moving_obstacle": 1.3,
                "random_obstacle": 1.0,
                "baseline": 0.7,
                "large_success": 0.8,
            },
            controller_weights={"mppi": 1.15, "dwb": 1.0},
            augmentation_weights={"none": 1.0, "mirror_lr": 1.0},
            ranges=mdp.Nav2RecordedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.10, 0.30),
                lin_vel_y=(-0.12, 0.12),
                ang_vel_z=(-0.30, 0.30),
                heading=None,
            ),
        )

        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=velocity_mdp.lin_vel_cmd_levels,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "lin_vel_x_limit": [-0.20, 0.60],
                "lin_vel_y_limit": [-0.30, 0.30],
            },
        )
        self.curriculum.ang_vel_cmd_levels = CurrTerm(
            func=velocity_mdp.ang_vel_cmd_levels,
            params={
                "reward_term_name": "track_ang_vel_z_exp",
                "ang_vel_z_limit": [-0.60, 0.60],
            },
        )

        self.rewards.torso_roll_pitch_l2.weight = -1.8
        if self.rewards.torso_ang_vel_xy_l2 is not None:
            self.rewards.torso_ang_vel_xy_l2.weight = -0.10
        if self.rewards.torso_vertical_velocity_l2 is not None:
            self.rewards.torso_vertical_velocity_l2.weight = -0.38
        self.rewards.torso_height_band_l2.weight = -0.48
        if self.rewards.torso_specific_force_xy_l2 is not None:
            self.rewards.torso_specific_force_xy_l2.weight = -0.015
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -0.30
        if self.rewards.ang_vel_xy_l2 is not None:
            self.rewards.ang_vel_xy_l2.weight = -0.08
        if self.rewards.feet_slide is not None:
            self.rewards.feet_slide.weight = -0.20


@configclass
class G1WalkPerturbFinetuneEnvCfg_PLAY(G1WalkPerturbFinetuneEnvCfg):
    """Reduced-env walk perturbation play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
