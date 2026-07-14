"""G1 AMP walk perturbation environment configs."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
import legged_lab.tasks.locomotion.velocity.mdp as velocity_mdp
from legged_lab.envs.g1_perturb_env import (
    G1_LOWER_BODY_JOINT_NAMES,
    G1_UPPER_BODY_JOINT_NAMES,
    UpperBodyPerturbationCfg,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg,
    G1_LOCOMOTION_JOINT_NAMES,
)
from .reference_data import WALK_ARM_POSE_SET_RELATIVE_PATH, load_walk_arm_pose_set


G1_LOWER_BODY_JOINT_IDS = [G1_LOCOMOTION_JOINT_NAMES.index(name) for name in G1_LOWER_BODY_JOINT_NAMES]
G1_WALK_PERTURB_POSE_SET_PATH = WALK_ARM_POSE_SET_RELATIVE_PATH.as_posix()
G1_WALK_PERTURB_POSE_SET = load_walk_arm_pose_set(G1_WALK_PERTURB_POSE_SET_PATH)


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

        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 2.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
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
