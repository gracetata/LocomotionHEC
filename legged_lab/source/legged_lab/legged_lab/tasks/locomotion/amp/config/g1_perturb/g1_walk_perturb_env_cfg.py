"""G1 ArmHack Walk configuration built directly on the Nav2 task contract."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs.g1_perturb_env import G1_LOWER_BODY_JOINT_NAMES, G1_UPPER_BODY_JOINT_NAMES
from legged_lab.envs.g1_walk_perturb_env import WalkUpperBodyPerturbationCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpNav2FinetuneEnvCfg,
    G1_LOCOMOTION_JOINT_NAMES,
)

from .reference_data import (
    WALK_ARM_POSE_SET_RELATIVE_PATH,
    WALK_NAV2_COMMAND_RELATIVE_PATH,
    load_walk_arm_pose_entries,
)


G1_LOWER_BODY_JOINT_IDS = [
    G1_LOCOMOTION_JOINT_NAMES.index(name) for name in G1_LOWER_BODY_JOINT_NAMES
]
G1_WALK_PERTURB_POSE_SET_PATH = WALK_ARM_POSE_SET_RELATIVE_PATH.as_posix()
G1_WALK_PERTURB_POSE_ENTRIES = load_walk_arm_pose_entries(G1_WALK_PERTURB_POSE_SET_PATH)
G1_WALK_PERTURB_POSE_NAMES = [name for name, _ in G1_WALK_PERTURB_POSE_ENTRIES]
G1_WALK_PERTURB_POSE_SET = [values for _, values in G1_WALK_PERTURB_POSE_ENTRIES]
G1_WALK_PERTURB_NAV2_COMMAND_PATH = WALK_NAV2_COMMAND_RELATIVE_PATH.as_posix()


def _configure_perturbation_common(cfg) -> None:
    """Mask scripted arms from AMP/regularizers while preserving a 29-D policy interface.

    Stand imports this helper. Keep its defaults stable; Walk applies its Nav2
    Stage-4 action-rate weight after this function returns.
    """

    lower_body_joint_cfg = SceneEntityCfg(
        "robot", joint_names=G1_LOWER_BODY_JOINT_NAMES, preserve_order=True
    )

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
class G1WalkPerturbFinetuneEnvCfg(G1AmpNav2FinetuneEnvCfg):
    """Nav2 Stage-4 walking with one fixed, named arm pose per episode."""

    upper_body_perturbation: WalkUpperBodyPerturbationCfg = WalkUpperBodyPerturbationCfg(
        joint_names=G1_UPPER_BODY_JOINT_NAMES,
        source="pose_set",
        pose_names=G1_WALK_PERTURB_POSE_NAMES,
        pose_set=G1_WALK_PERTURB_POSE_SET,
        pose_name="pos2_down",
        initialize_joint_state_on_reset=True,
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_perturbation_common(self)

        # Freeze the successful Nav2 Stage-4 command contract. The local CSV is
        # raw-only, so mirror_lr is synthesized deterministically at load time.
        self.commands.base_velocity = mdp.Nav2RecordedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            data_path=G1_WALK_PERTURB_NAV2_COMMAND_PATH,
            augmentation_filter="none,mirror_lr",
            synthesize_mirror_lr=True,
            scenario_family_filter="complex_turn",
            dataset_sample_dt=0.05,
            window_duration_s=4.0,
            command_scale=(0.85, 0.75, 0.75),
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
            controller_weights={"mppi": 1.5, "dwb": 1.0},
            augmentation_weights={"none": 1.0, "mirror_lr": 1.0},
            ranges=mdp.Nav2RecordedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.20, 0.60),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.60, 0.60),
                heading=None,
            ),
        )

        # Restore the saved Nav2 Stage-4 reward contract. The only intentional
        # differences are the lower-body masks above and no reward on scripted arms.
        self.rewards.track_lin_vel_xy_exp.weight = 1.8
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.30
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.35
        self.rewards.track_torso_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_torso_lin_vel_xy_exp.params["std"] = 0.30
        self.rewards.track_torso_yaw_rate_exp.weight = 0.65
        self.rewards.track_torso_yaw_rate_exp.params["std"] = 0.35
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.torso_roll_pitch_l2.weight = -0.04
        self.rewards.torso_ang_vel_xy_l2.weight = 0.0
        self.rewards.torso_vertical_velocity_l2.weight = -0.01
        self.rewards.torso_height_band_l2 = None
        self.rewards.torso_lateral_vel_cmd_l2.weight = 0.0
        self.rewards.torso_specific_force_xy_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -0.20
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.006
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_deviation_hip.weight = -0.10
        self.rewards.joint_deviation_waist.weight = -0.10
        self.rewards.feet_air_time.weight = 0.50
        self.rewards.feet_slide.weight = -0.10
        self.rewards.feet_swing_clearance_band_l2 = None
        self.rewards.gait_timing_symmetry_l1 = None
        self.rewards.directional_speed_floor_l1 = None
        self.rewards.directional_double_air_l1 = None
        self.rewards.backward_single_stance = None
        self.rewards.backward_double_stance_l1 = None
        self.rewards.directional_double_stance_support = None
        self.rewards.directional_velocity_leak_l1 = None
        self.rewards.termination_penalty.weight = -200.0


@configclass
class G1WalkPerturbFinetuneEnvCfg_PLAY(G1WalkPerturbFinetuneEnvCfg):
    """Reduced-env Walk configuration for named-pose evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
