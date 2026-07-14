"""Unitree G1 29-DoF AMP flat-walking environment configuration.

Core classes:
    G1AmpRewards defines task rewards for command tracking plus conservative
    regularizers and zero-weight torso/IMU fine-tune targets. G1AmpEnvCfg wires
    the Unitree G1 asset, ACCAD/GMR walking motion dataset, AMP discriminator
    demo observations, and Reference State Initialization (RSI). G1AmpEnvCfg_PLAY
    reduces environment count for policy replay. G1AmpSegmentedYawFinetuneEnvCfg
    swaps only the command distribution for speed-segmented yaw fine-tuning from
    an existing 96-D baseline policy checkpoint. G1AmpCmuWalkCoreAdaptiveEnvCfg
    uses the CMU walk core demos with adaptive command tracking.
    G1AmpCmuWalkFullAdaptiveEnvCfg uses the full CMU walk set with the same
    adaptive task setup. G1AmpCmuWalkWashedAdaptiveEnvCfg uses the washed CMU
    walk set with a stronger style/yaw-tracking balance.
    G1AmpCmuWalkWashedAdaptiveStrictEnvCfg adds strict upright, height,
    foot-clearance, and gait-symmetry rewards. G1AmpCmuWalkWashedAdaptiveStrictArmPriorEnvCfg
    adds a supervised arm-style prior on top of strict rewards. G1AmpToTargetEnvCfg
    trains reset-relative short-range SE(2) positioning and stop-hold. G1AmpNav2FinetuneEnvCfg
    uses recorded Nav2 loopback cmd_vel windows as the command distribution.

Inputs/outputs:
    The motion loader consumes Lab-format pickle files with root_pos/root_rot,
    dof_pos, key_body_pos, fps, and loop_mode. The policy observes 96 values:
    base angular velocity, projected gravity, velocity command, 29 joint
    positions, 29 joint velocities, and 29 previous actions.

Usage:
    python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 3000
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-Play-v0 --checkpoint logs/rsl_rl/g1_amp/<run>/model_*.pt
"""

import math
import json
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import (
    UNITREE_G1_29DOF_CFG,
    UNITREE_S3_G1_29DOF_CFG,
    UNITREE_S3_G1_29DOF_MJCF_CFG,
)

G1_AMP_ROBOT_ASSET_ENV = "LEGGED_LAB_G1_AMP_ROBOT_ASSET"
# Policy/deployment order. This must match scripts/tools/retarget/config/g1_29dof.yaml
# lab_dof_names and the exported baseline policy metadata, not the Unitree SDK order.
G1_LOCOMOTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
G1_LOCOMOTION_JOINT_IDS = list(range(len(G1_LOCOMOTION_JOINT_NAMES)))

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml.
KEY_BODY_NAMES = [
    "left_ankle_roll_link", 
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
] # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
G1_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
TO_TARGET_SUCCESS_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4
DEFAULT_RSI_RATIO = 0.5
G1_ACCAD_G1USED_50HZ_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "accad_g1used_50hz"
)
G1_CMU_WALK_CORE_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "cmu_walk_50hz_task_core"
)
G1_CMU_WALK_FULL_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "cmu_walk_50hz"
)
G1_CMU_WALK_WASHED_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "cmu_walk_washed_50hz"
)
G1_COMMAND_BALANCED_DIRECTIONAL_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "command_balanced_directional_50hz"
)
G1_COMMAND_BALANCED_TASK_SAMPLING_CONFIG_PATH = os.path.join(
    G1_COMMAND_BALANCED_DIRECTIONAL_MOTION_DATA_DIR, "task_sampling_config.json"
)
DEFAULT_G1_ARM_STYLE_PRIOR_PATH = "logs/arm_style_prior/g1_cmu_walk_washed_arm_prior.pt"
G1_PROJECT_ROOT_DIR = os.path.abspath(os.path.join(LEGGED_LAB_ROOT_DIR, "..", "..", "..", ".."))
G1_NAV2_AUGMENTED_CMD_DATA_PATH = os.path.join(
    G1_PROJECT_ROOT_DIR, "nav2_loopback_actual", "actual_augmented", "all_cmd_vel_augmented.csv"
)
G1_ACCAD_G1USED_MOTION_WEIGHTS = {
    "B10_-__Walk_turn_left_45_stageii": 1.0,
    "B11_-__Walk_turn_left_135_stageii": 1.0,
    "B13_-__Walk_turn_right_90_stageii": 1.0,
    "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
    "B15_-__Walk_turn_around_stageii": 1.0,
    "B22_-__side_step_left_stageii": 1.0,
    "B23_-__side_step_right_stageii": 1.0,
    "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
    "B9_-__Walk_turn_left_90_stageii": 1.0,
    "C11_-_run_turn_left_90_stageii": 1.0,
    "C12_-_run_turn_left_45_stageii": 1.0,
    "C13_-_run_turn_left_135_stageii": 1.0,
    "C14_-_run_turn_right_90_stageii": 1.0,
    "C15_-_run_turn_right_45_stageii": 1.0,
    "C16_-_run_turn_right_135_stageii": 1.0,
    "C17_-_run_change_direction_stageii": 1.0,
    "C1_-_stand_to_run_stageii": 1.0,
    "C3_-_run_stageii": 1.0,
    "C4_-_run_to_walk_a_stageii": 1.0,
    "C5_-_walk_to_run_stageii": 1.0,
    "C6_-_stand_to_run_backwards_stageii": 1.0,
    "C8_-_run_backwards_to_stand_stageii": 1.0,
    "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
    "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
    "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
    "Walk_B15_-_Walk_turn_around_stageii": 1.0,
    "Walk_B16_-_Walk_turn_change_stageii": 1.0,
    "Walk_B22_-_Side_step_left_stageii": 1.0,
    "Walk_B23_-_Side_step_right_stageii": 1.0,
    "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
}


def _motion_weights_from_dir(motion_data_dir: str) -> dict[str, float]:
    if not os.path.isdir(motion_data_dir):
        raise FileNotFoundError(f"G1 motion directory does not exist: {motion_data_dir}")
    motion_names = [os.path.splitext(name)[0] for name in sorted(os.listdir(motion_data_dir)) if name.endswith(".pkl")]
    if not motion_names:
        raise ValueError(f"G1 motion directory contains no .pkl files: {motion_data_dir}")
    return {name: 1.0 for name in motion_names}


def _motion_weights_from_json(motion_data_dir: str, filename: str = "motion_weights.json") -> dict[str, float]:
    weights_path = os.path.join(motion_data_dir, filename)
    if not os.path.isfile(weights_path):
        return _motion_weights_from_dir(motion_data_dir)
    with open(weights_path, encoding="utf-8") as weights_file:
        raw_weights = json.load(weights_file)
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(f"G1 motion weight file is empty or invalid: {weights_path}")

    available_motion_names = {
        os.path.splitext(name)[0]
        for name in os.listdir(motion_data_dir)
        if name.endswith(".pkl")
    }
    missing_motion_names = sorted(set(raw_weights).difference(available_motion_names))
    if missing_motion_names:
        raise ValueError(
            f"G1 motion weight file {weights_path} references missing motions: {missing_motion_names[:10]}"
        )
    return {str(name): float(weight) for name, weight in raw_weights.items()}


def _to_target_motion_weights_from_manifest(motion_data_dir: str) -> dict[str, float]:
    """Bias AMP demos toward slow core clips while keeping a small walking prior."""
    weights = _motion_weights_from_dir(motion_data_dir)
    manifest_path = os.path.join(motion_data_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return weights

    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    for item in manifest:
        motion_name = os.path.splitext(str(item.get("file", "")))[0]
        if motion_name not in weights:
            continue
        mean_speed = float(item.get("mean_speed_xy", 0.0))
        random_ratio = float(item.get("random_frame_ratio", 0.0))
        if mean_speed <= 0.15:
            weight = 3.0
        elif mean_speed <= 0.30:
            weight = 2.0
        elif mean_speed <= 0.45:
            weight = 1.2
        else:
            weight = 0.45
        if random_ratio >= 0.80:
            weight *= 1.25
        weights[motion_name] = weight
    return weights


def _selected_g1_robot_asset():
    robot_asset = os.environ.get(G1_AMP_ROBOT_ASSET_ENV, "s3_g1_29dof").strip().lower()
    if robot_asset in (
        "",
        "config",
        "s3",
        "s3_g1_29dof",
        "s3_g1_29dof_mjcf",
        "s3_mjcf",
        "g1_29dof_mjcf",
        "g1_mjcf",
    ):
        return robot_asset
    if robot_asset in ("g1", "g1_29dof", "original_g1"):
        return robot_asset
    raise ValueError(
        f"Unsupported {G1_AMP_ROBOT_ASSET_ENV}={robot_asset!r}. "
        "Valid values: config, s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof, g1_29dof_mjcf."
    )


def _select_g1_robot_cfg():
    robot_asset = _selected_g1_robot_asset()
    if robot_asset in ("g1", "g1_29dof", "original_g1"):
        return UNITREE_G1_29DOF_CFG
    if robot_asset in ("s3_g1_29dof_mjcf", "s3_mjcf", "g1_29dof_mjcf", "g1_mjcf"):
        return UNITREE_S3_G1_29DOF_MJCF_CFG
    return UNITREE_S3_G1_29DOF_CFG


def _select_contact_sensor_prim_path():
    robot_asset = _selected_g1_robot_asset()
    if robot_asset in ("g1", "g1_29dof", "original_g1"):
        return "{ENV_REGEX_NS}/Robot/.*"
    return "{ENV_REGEX_NS}/Robot/pelvis/.*"


def _configure_cmu_walk_core_adaptive_rewards(rewards: "G1AmpRewards") -> None:
    rewards.track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_adaptive_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "ema_decay": 0.995,
            "min_sigma": 0.04,
        },
    )
    rewards.track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_adaptive_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "ema_decay": 0.995,
            "min_sigma": 0.01,
        },
    )
    rewards.track_torso_lin_vel_xy_exp = None
    rewards.track_torso_yaw_rate_exp = None
    rewards.flat_orientation_l2 = None
    rewards.torso_roll_pitch_l2.weight = -0.05
    rewards.torso_ang_vel_xy_l2 = None
    rewards.torso_vertical_velocity_l2 = None
    rewards.torso_lateral_vel_cmd_l2 = None
    rewards.torso_specific_force_xy_l2 = None
    rewards.lin_vel_z_l2 = None
    rewards.ang_vel_xy_l2 = None
    rewards.dof_pos_limits = None
    rewards.joint_deviation_hip = None
    rewards.joint_deviation_arms = None
    rewards.joint_deviation_waist = None
    rewards.feet_air_time = None
    rewards.feet_slide = None
    rewards.termination_penalty = None


def _configure_cmu_walk_washed_adaptive_rewards(rewards: "G1AmpRewards") -> None:
    rewards.track_ang_vel_z_exp.weight = 1.25
    rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.7)
    rewards.track_ang_vel_z_exp.params["ema_decay"] = 0.99
    rewards.track_ang_vel_z_exp.params["min_sigma"] = 0.04


def _configure_cmu_walk_washed_adaptive_strict_rewards(rewards: "G1AmpRewards") -> None:
    torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
    foot_asset_cfg = SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)

    rewards.track_torso_lin_vel_xy_exp = RewTerm(
        func=mdp.track_torso_lin_vel_xy_exp,
        weight=0.40,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": torso_cfg},
    )
    rewards.track_torso_yaw_rate_exp = RewTerm(
        func=mdp.track_torso_yaw_rate_exp,
        weight=0.30,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": torso_cfg},
    )
    rewards.torso_roll_pitch_l2 = RewTerm(
        func=mdp.torso_roll_pitch_l2,
        weight=-1.50,
        params={"asset_cfg": torso_cfg},
    )
    rewards.torso_ang_vel_xy_l2 = RewTerm(
        func=mdp.torso_ang_vel_xy_l2,
        weight=-0.08,
        params={"asset_cfg": torso_cfg},
    )
    rewards.torso_vertical_velocity_l2 = RewTerm(
        func=mdp.torso_vertical_velocity_l2,
        weight=-0.30,
        params={"asset_cfg": torso_cfg},
    )
    rewards.torso_specific_force_xy_l2 = RewTerm(
        func=mdp.torso_specific_force_xy_l2,
        weight=-0.01,
        params={"asset_cfg": torso_cfg},
    )
    rewards.torso_height_band_l2 = RewTerm(
        func=mdp.torso_height_band_l2,
        weight=-0.35,
        params={
            "target_height": 0.84,
            "lower_deadband": -0.02,
            "upper_deadband": 0.04,
            "std": 0.04,
            "asset_cfg": torso_cfg,
        },
    )
    rewards.feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.40,
        params={"command_name": "base_velocity", "sensor_cfg": foot_sensor_cfg, "threshold": 0.28},
    )
    rewards.feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.12,
        params={"sensor_cfg": foot_sensor_cfg, "asset_cfg": foot_asset_cfg},
    )
    rewards.feet_swing_clearance_band_l2 = RewTerm(
        func=mdp.feet_swing_clearance_band_l2,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "asset_cfg": foot_asset_cfg,
            "target_height": 0.055,
            "std": 0.03,
            "air_time_threshold": 0.06,
            "command_threshold": 0.10,
            "max_height": 0.13,
        },
    )
    rewards.gait_timing_symmetry_l1 = RewTerm(
        func=mdp.gait_timing_symmetry_l1,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "deadband": 0.03,
            "std": 0.08,
            "min_air_time": 0.06,
            "command_threshold": 0.10,
        },
    )


def _configure_command_balanced_directional_v2_rewards(rewards: "G1AmpRewards") -> None:
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)

    rewards.track_lin_vel_xy_exp.weight = 1.25
    rewards.track_lin_vel_xy_exp.params["min_sigma"] = 0.025
    rewards.track_torso_lin_vel_xy_exp.weight = 0.45
    rewards.feet_slide.weight = -0.16
    rewards.feet_swing_clearance_band_l2.weight = -0.30
    rewards.feet_swing_clearance_band_l2.params["target_height"] = 0.060
    rewards.gait_timing_symmetry_l1.weight = -0.18

    rewards.directional_speed_floor_l1 = RewTerm(
        func=mdp.directional_speed_floor_l1,
        weight=-0.30,
        params={
            "command_name": "base_velocity",
            "min_speed_fraction": 0.85,
            "std": 0.20,
            "command_threshold": 0.15,
            "backward_threshold": 0.15,
            "lateral_threshold": 0.15,
        },
    )
    rewards.directional_double_air_l1 = RewTerm(
        func=mdp.directional_double_air_l1,
        weight=-0.55,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "command_threshold": 0.15,
            "backward_threshold": 0.15,
            "lateral_threshold": 0.15,
        },
    )
    rewards.backward_single_stance = RewTerm(
        func=mdp.backward_single_stance,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "backward_threshold": 0.15,
            "min_mode_time": 0.02,
        },
    )
    rewards.backward_double_stance_l1 = RewTerm(
        func=mdp.backward_double_stance_l1,
        weight=-0.10,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "backward_threshold": 0.15,
        },
    )


def _configure_command_balanced_directional_v3_rewards(rewards: "G1AmpRewards") -> None:
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)

    _configure_command_balanced_directional_v2_rewards(rewards)

    # V2 solved backward double-air hopping, but over-corrected backward/lateral
    # into too much single support. V3 keeps the anti-hop terms and replaces the
    # hard backward single-stance push with mode-specific support shaping.
    rewards.backward_single_stance = None
    rewards.backward_double_stance_l1 = None
    rewards.directional_speed_floor_l1.weight = -0.28
    rewards.directional_double_air_l1.weight = -0.60

    rewards.feet_air_time.weight = 0.32
    rewards.feet_slide.weight = -0.18
    rewards.feet_swing_clearance_band_l2.weight = -0.28
    rewards.gait_timing_symmetry_l1.weight = -0.14
    rewards.torso_vertical_velocity_l2.weight = -0.34
    rewards.torso_height_band_l2.weight = -0.42

    rewards.directional_double_stance_support = RewTerm(
        func=mdp.directional_double_stance_support,
        weight=0.16,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "backward_threshold": 0.15,
            "lateral_threshold": 0.15,
            "backward_scale": 0.55,
            "lateral_scale": 1.0,
            "min_contact_time": 0.02,
        },
    )
    rewards.directional_velocity_leak_l1 = RewTerm(
        func=mdp.directional_velocity_leak_l1,
        weight=-0.12,
        params={
            "command_name": "base_velocity",
            "lateral_threshold": 0.15,
            "backward_threshold": 0.15,
            "lateral_x_deadband": 0.04,
            "lateral_yaw_deadband": 0.04,
            "sagittal_y_deadband": 0.04,
            "sagittal_yaw_deadband": 0.04,
            "lin_std": 0.20,
            "yaw_std": 0.35,
        },
    )


@configclass
class G1AmpRewards():
    """Reward terms for the MDP."""
    # -- optional standing rewards
    alive = None
    double_support = None
    root_xy_position_l2 = None

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_torso_lin_vel_xy_exp = RewTerm(
        func=mdp.track_torso_lin_vel_xy_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    track_torso_yaw_rate_exp = RewTerm(
        func=mdp.track_torso_yaw_rate_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    torso_roll_pitch_l2 = RewTerm(
        func=mdp.torso_roll_pitch_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    torso_ang_vel_xy_l2 = RewTerm(
        func=mdp.torso_ang_vel_xy_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    torso_vertical_velocity_l2 = RewTerm(
        func=mdp.torso_vertical_velocity_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    torso_height_band_l2 = RewTerm(
        func=mdp.torso_height_band_l2,
        weight=0.0,
        params={
            "target_height": 0.84,
            "lower_deadband": -0.02,
            "upper_deadband": 0.04,
            "std": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    torso_lateral_vel_cmd_l2 = RewTerm(
        func=mdp.torso_lateral_vel_cmd_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    torso_specific_force_xy_l2 = RewTerm(
        func=mdp.torso_specific_force_xy_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_swing_clearance_band_l2 = RewTerm(
        func=mdp.feet_swing_clearance_band_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True),
            "target_height": 0.055,
            "std": 0.03,
            "air_time_threshold": 0.06,
            "command_threshold": 0.10,
            "max_height": 0.13,
        },
    )
    gait_timing_symmetry_l1 = RewTerm(
        func=mdp.gait_timing_symmetry_l1,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True),
            "deadband": 0.03,
            "std": 0.08,
            "min_air_time": 0.06,
            "command_threshold": 0.10,
        },
    )
    directional_speed_floor_l1 = None
    directional_double_air_l1 = None
    backward_single_stance = None
    backward_double_stance_l1 = None
    directional_double_stance_support = None
    directional_velocity_leak_l1 = None
    arm_style_prior = None
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1AmpToTargetRewards(G1AmpRewards):
    """Reward terms for reset-relative precise SE(2) positioning."""

    target_position_exp = RewTerm(
        func=mdp.target_position_exp,
        weight=2.2,
        params={"command_name": "base_velocity", "std": 0.42},
    )
    target_heading_exp = RewTerm(
        func=mdp.target_heading_exp,
        weight=0.9,
        params={"command_name": "base_velocity", "std": 0.75},
    )
    target_approach_velocity = RewTerm(
        func=mdp.target_approach_velocity,
        weight=0.55,
        params={
            "command_name": "base_velocity",
            "speed_scale": 0.35,
            "stop_distance": 0.10,
            "stop_latch_gate": True,
        },
    )
    target_pose_stillness_exp = RewTerm(
        func=mdp.target_pose_stillness_exp,
        weight=1.6,
        params={
            "command_name": "base_velocity",
            "position_std": 0.13,
            "heading_std": 0.22,
            "lin_vel_std": 0.08,
            "yaw_vel_std": 0.12,
            "use_stop_latch": True,
        },
    )
    near_target_velocity_l2 = RewTerm(
        func=mdp.near_target_velocity_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "position_std": 0.18,
            "heading_std": 0.30,
            "yaw_weight": 0.5,
            "use_stop_latch": True,
        },
    )
    near_target_joint_deviation_l1 = RewTerm(
        func=mdp.near_target_joint_deviation_l1,
        weight=-0.8,
        params={
            "command_name": "base_velocity",
            "position_std": 0.16,
            "heading_std": 0.25,
            "use_stop_latch": True,
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True),
        },
    )
    target_stop_latch_drift_l2 = RewTerm(
        func=mdp.target_stop_latch_drift_l2,
        weight=-2.5,
        params={
            "command_name": "base_velocity",
            "position_tolerance": 0.08,
            "heading_tolerance": 0.18,
            "position_std": 0.05,
            "heading_std": 0.12,
        },
    )
    target_success_bonus = RewTerm(
        func=mdp.target_success_bonus,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "position_threshold": 0.06,
            "heading_threshold": 0.12,
            "lin_vel_threshold": 0.06,
            "yaw_vel_threshold": 0.10,
            "mean_joint_threshold": 0.10,
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True),
        },
    )


@configclass
class G1AmpToTargetV2Rewards(G1AmpToTargetRewards):
    """Denser reward set for learnable short-range SE(2) target reaching."""

    target_position_exp = RewTerm(
        func=mdp.target_position_exp,
        weight=4.5,
        params={"command_name": "base_velocity", "std": 0.45},
    )
    target_heading_exp = RewTerm(
        func=mdp.target_heading_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.75},
    )
    target_lin_vel_tracking_adaptive_exp = RewTerm(
        func=mdp.target_lin_vel_tracking_adaptive_exp,
        weight=0.65,
        params={
            "command_name": "base_velocity",
            "std": 0.60,
            "ema_decay": 0.995,
            "min_sigma": 0.03,
            "position_gain": 0.70,
            "max_lin_speed": 0.32,
            "stop_distance": 0.10,
            "stop_heading": 0.22,
            "warmup_steps": 2400,
            "ramp_steps": 12000,
        },
    )
    target_yaw_rate_tracking_adaptive_exp = RewTerm(
        func=mdp.target_yaw_rate_tracking_adaptive_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": 0.95,
            "ema_decay": 0.995,
            "min_sigma": 0.04,
            "heading_gain": 0.65,
            "max_yaw_rate": 0.45,
            "stop_distance": 0.10,
            "stop_heading": 0.22,
            "warmup_steps": 2400,
            "ramp_steps": 12000,
        },
    )
    target_approach_velocity = RewTerm(
        func=mdp.target_approach_velocity,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "speed_scale": 0.35,
            "stop_distance": 0.10,
            "stop_latch_gate": True,
            "warmup_steps": 4800,
            "ramp_steps": 12000,
        },
    )
    target_pose_stillness_exp = RewTerm(
        func=mdp.target_pose_stillness_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "position_std": 0.16,
            "heading_std": 0.28,
            "lin_vel_std": 0.10,
            "yaw_vel_std": 0.16,
            "use_stop_latch": True,
        },
    )
    near_target_velocity_l2 = RewTerm(
        func=mdp.near_target_velocity_l2,
        weight=-1.2,
        params={
            "command_name": "base_velocity",
            "position_std": 0.18,
            "heading_std": 0.30,
            "yaw_weight": 1.0,
            "use_stop_latch": True,
        },
    )
    near_target_joint_deviation_l1 = RewTerm(
        func=mdp.near_target_joint_deviation_l1,
        weight=-1.2,
        params={
            "command_name": "base_velocity",
            "position_std": 0.16,
            "heading_std": 0.25,
            "use_stop_latch": True,
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True),
        },
    )
    near_target_key_body_deviation_l2 = RewTerm(
        func=mdp.near_target_key_body_deviation_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "position_std": 0.16,
            "heading_std": 0.25,
            "use_stop_latch": True,
            "asset_cfg": SceneEntityCfg("robot", body_names=TO_TARGET_SUCCESS_KEY_BODY_NAMES, preserve_order=True),
            "reference_attr": mdp.DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
        },
    )
    target_stop_latch_drift_l2 = RewTerm(
        func=mdp.target_stop_latch_drift_l2,
        weight=-2.5,
        params={
            "command_name": "base_velocity",
            "position_tolerance": 0.08,
            "heading_tolerance": 0.18,
            "position_std": 0.05,
            "heading_std": 0.12,
        },
    )
    target_success_bonus = RewTerm(
        func=mdp.target_success_bonus,
        weight=14.0,
        params={
            "command_name": "base_velocity",
            "position_threshold": 0.08,
            "heading_threshold": 0.18,
            "lin_vel_threshold": 0.08,
            "yaw_vel_threshold": 0.12,
            "mean_joint_threshold": 0.16,
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True),
            "key_body_mean_threshold": 0.08,
            "key_body_asset_cfg": SceneEntityCfg(
                "robot", body_names=TO_TARGET_SUCCESS_KEY_BODY_NAMES, preserve_order=True
            ),
            "key_body_reference_attr": mdp.DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
        },
    )


@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP environment."""
    
    rewards: G1AmpRewards = G1AmpRewards()
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = _select_g1_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = _select_contact_sensor_prim_path()
        locomotion_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )

        # Keep the policy interface pinned to the original G1 29-DoF layout even
        # if the underlying asset grows passive payload/gripper joints.
        self.actions.joint_pos.joint_names = G1_LOCOMOTION_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = G1_ACCAD_G1USED_50HZ_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = G1_ACCAD_G1USED_MOTION_WEIGHTS

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        
        # policy / critic / discriminator key_body_pos_b terms are not used in current AMP base config.
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.policy.joint_pos.params = {"asset_cfg": locomotion_joint_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": locomotion_joint_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": locomotion_joint_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": locomotion_joint_cfg}
        self.observations.disc.joint_pos.params = {"asset_cfg": locomotion_joint_cfg}
        self.observations.disc.joint_vel.params = {"asset_cfg": locomotion_joint_cfg}
        
        # discriminator demostration observations
        
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_lin_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["joint_ids"] = G1_LOCOMOTION_JOINT_IDS
        self.observations.disc_demo.ref_joint_vel.params["joint_ids"] = G1_LOCOMOTION_JOINT_IDS

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = G1_LOCOMOTION_JOINT_NAMES
        self.events.scale_actuator_gains.params["asset_cfg"].preserve_order = True
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = G1_LOCOMOTION_JOINT_NAMES
        self.events.scale_joint_parameters.params["asset_cfg"].preserve_order = True
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_from_ref = EventTerm(
            func=mdp.ref_state_init_subset,
            mode="reset",
            params={
                "asset_cfg": locomotion_joint_cfg,
                "motion_dataset": "motion_dataset",
                "rsi_ratio": DEFAULT_RSI_RATIO,
                "pos_rsi": False,
                "height_offset": 0.05,
            },
        )
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        self.rewards.dof_torques_l2.params = {"asset_cfg": locomotion_joint_cfg}
        self.rewards.dof_acc_l2.params = {"asset_cfg": locomotion_joint_cfg}
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        
        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None


@configclass
class G1AmpEnvCfg_PLAY(G1AmpEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 48 
        self.scene.env_spacing = 2.5
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class G1AmpSegmentedYawFinetuneEnvCfg(G1AmpEnvCfg):
    """G1 AMP fine-tune task with speed-segmented yaw command sampling."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.CurvatureVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            low_speed_threshold=0.40,
            high_speed_ang_vel_z_std=0.10,
            ranges=mdp.CurvatureVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.40, 1.00),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.30, 0.30),
                heading=None,
                low_speed_ang_vel_z=(-0.50, 0.50),
            ),
        )


@configclass
class G1AmpCurvatureFinetuneEnvCfg(G1AmpSegmentedYawFinetuneEnvCfg):
    """Backward-compatible alias for the segmented-yaw fine-tune task."""
    pass


@configclass
class G1AmpSegmentedYawFinetuneEnvCfg_PLAY(G1AmpSegmentedYawFinetuneEnvCfg):
    """Reduced-env segmented-yaw command task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCurvatureFinetuneEnvCfg_PLAY(G1AmpSegmentedYawFinetuneEnvCfg_PLAY):
    """Backward-compatible play-task alias for segmented-yaw fine-tuning."""
    pass


@configclass
class G1AmpCmuWalkCoreAdaptiveEnvCfg(G1AmpEnvCfg):
    """G1 AMP task using CMU walk core demos and adaptive velocity tracking."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = G1_CMU_WALK_CORE_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_dir(G1_CMU_WALK_CORE_MOTION_DATA_DIR)
        self.motion_data.motion_dataset.root_rot_order = "xyzw"
        self.motion_data.motion_dataset.target_dof_names = G1_LOCOMOTION_JOINT_NAMES
        self.motion_data.motion_dataset.strict_dof_names = True

        self.commands.base_velocity = mdp.CurvatureVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            low_speed_threshold=0.40,
            high_speed_ang_vel_z_std=0.10,
            ranges=mdp.CurvatureVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.40, 1.00),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.30, 0.30),
                heading=None,
                low_speed_ang_vel_z=(-0.50, 0.50),
            ),
        )

        _configure_cmu_walk_core_adaptive_rewards(self.rewards)


@configclass
class G1AmpCmuWalkCoreAdaptiveEnvCfg_PLAY(G1AmpCmuWalkCoreAdaptiveEnvCfg):
    """Reduced-env CMU walk core adaptive task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCmuWalkFullAdaptiveEnvCfg(G1AmpCmuWalkCoreAdaptiveEnvCfg):
    """G1 AMP task using the full CMU walk demos with adaptive velocity tracking."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = G1_CMU_WALK_FULL_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_dir(G1_CMU_WALK_FULL_MOTION_DATA_DIR)


@configclass
class G1AmpCmuWalkFullAdaptiveEnvCfg_PLAY(G1AmpCmuWalkFullAdaptiveEnvCfg):
    """Reduced-env full CMU walk adaptive task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCmuWalkWashedAdaptiveEnvCfg(G1AmpCmuWalkCoreAdaptiveEnvCfg):
    """G1 AMP task using washed CMU walk demos with adaptive velocity tracking."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = G1_CMU_WALK_WASHED_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_dir(G1_CMU_WALK_WASHED_MOTION_DATA_DIR)
        _configure_cmu_walk_washed_adaptive_rewards(self.rewards)


@configclass
class G1AmpCmuWalkWashedAdaptiveEnvCfg_PLAY(G1AmpCmuWalkWashedAdaptiveEnvCfg):
    """Reduced-env washed CMU walk adaptive task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCmuWalkWashedAdaptiveStrictEnvCfg(G1AmpCmuWalkWashedAdaptiveEnvCfg):
    """Washed CMU walk adaptive task with strict upright gait shaping rewards."""

    def __post_init__(self):
        super().__post_init__()

        _configure_cmu_walk_washed_adaptive_strict_rewards(self.rewards)


@configclass
class G1AmpCmuWalkWashedAdaptiveStrictEnvCfg_PLAY(G1AmpCmuWalkWashedAdaptiveStrictEnvCfg):
    """Reduced-env strict washed CMU walk adaptive task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCmuWalkWashedAdaptiveStrictArmPriorEnvCfg(G1AmpCmuWalkWashedAdaptiveStrictEnvCfg):
    """Strict washed CMU walk task with a lightweight supervised G1 arm-style prior."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.arm_style_prior = RewTerm(
            func=mdp.arm_style_prior_exp,
            weight=0.35,
            params={
                "checkpoint_path": DEFAULT_G1_ARM_STYLE_PRIOR_PATH,
                "std": 0.25,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class G1AmpCmuWalkWashedAdaptiveStrictArmPriorEnvCfg_PLAY(G1AmpCmuWalkWashedAdaptiveStrictArmPriorEnvCfg):
    """Reduced-env strict arm-prior task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCommandBalancedDirectionalStrictArmPriorEnvCfg(G1AmpCmuWalkWashedAdaptiveStrictArmPriorEnvCfg):
    """Strict arm-prior G1 AMP task using command-balanced directional demos."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = G1_COMMAND_BALANCED_DIRECTIONAL_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_json(
            G1_COMMAND_BALANCED_DIRECTIONAL_MOTION_DATA_DIR
        )
        self.motion_data.motion_dataset.root_rot_order = "xyzw"
        self.motion_data.motion_dataset.target_dof_names = G1_LOCOMOTION_JOINT_NAMES
        self.motion_data.motion_dataset.strict_dof_names = True

        self.commands.base_velocity = mdp.ModeBalancedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(8.0, 8.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            sampling_config_path=G1_COMMAND_BALANCED_TASK_SAMPLING_CONFIG_PATH,
            ranges=mdp.ModeBalancedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.65, 1.05),
                lin_vel_y=(-0.45, 0.45),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        )


@configclass
class G1AmpCommandBalancedDirectionalStrictArmPriorEnvCfg_PLAY(
    G1AmpCommandBalancedDirectionalStrictArmPriorEnvCfg
):
    """Reduced-env command-balanced strict arm-prior task for replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCommandBalancedDirectionalStrictArmPriorV2EnvCfg(
    G1AmpCommandBalancedDirectionalStrictArmPriorEnvCfg
):
    """Command-balanced task with extra backward/lateral anti-hop shaping."""

    def __post_init__(self):
        super().__post_init__()

        _configure_command_balanced_directional_v2_rewards(self.rewards)


@configclass
class G1AmpCommandBalancedDirectionalStrictArmPriorV2EnvCfg_PLAY(
    G1AmpCommandBalancedDirectionalStrictArmPriorV2EnvCfg
):
    """Reduced-env V2 command-balanced task for replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg(
    G1AmpCommandBalancedDirectionalStrictArmPriorEnvCfg
):
    """V3 command-balanced task tuned from MuJoCo mask/validation lessons."""

    def __post_init__(self):
        super().__post_init__()

        _configure_command_balanced_directional_v3_rewards(self.rewards)


@configclass
class G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg_PLAY(
    G1AmpCommandBalancedDirectionalStrictArmPriorV3EnvCfg
):
    """Reduced-env V3 command-balanced task for replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1AmpToTargetEnvCfg(G1AmpEnvCfg):
    """G1 AMP task for reset-relative short-range SE(2) positioning and stop-hold."""

    rewards: G1AmpToTargetRewards = G1AmpToTargetRewards()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0

        self.motion_data.motion_dataset.motion_data_dir = G1_CMU_WALK_CORE_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _to_target_motion_weights_from_manifest(
            G1_CMU_WALK_CORE_MOTION_DATA_DIR
        )
        self.motion_data.motion_dataset.root_rot_order = "xyzw"
        self.motion_data.motion_dataset.target_dof_names = G1_LOCOMOTION_JOINT_NAMES
        self.motion_data.motion_dataset.strict_dof_names = True

        self.commands.base_velocity = mdp.RelativePose2dCommandCfg(
            asset_name="robot",
            resampling_time_range=(8.0, 8.0),
            debug_vis=False,
            ranges=mdp.RelativePose2dCommandCfg.Ranges(
                radius=(0.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
            success_position_threshold=0.06,
            success_heading_threshold=0.12,
            success_lin_vel_threshold=0.06,
            success_yaw_vel_threshold=0.10,
            stop_latch_position_threshold=0.08,
            stop_latch_heading_threshold=0.18,
            stop_exit_position_threshold=0.12,
            stop_exit_heading_threshold=0.28,
        )

        _configure_cmu_walk_washed_adaptive_strict_rewards(self.rewards)
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.track_torso_lin_vel_xy_exp = None
        self.rewards.track_torso_yaw_rate_exp = None
        self.rewards.torso_lateral_vel_cmd_l2 = None
        self.rewards.torso_specific_force_xy_l2.weight = -0.004
        self.rewards.torso_vertical_velocity_l2.weight = -0.08
        self.rewards.action_rate_l2.weight = -0.008
        self.rewards.dof_torques_l2.weight = -2.5e-6
        self.rewards.dof_acc_l2.weight = -1.2e-7
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.22
        self.rewards.feet_slide.weight = -0.14
        self.rewards.feet_swing_clearance_band_l2.weight = -0.15
        self.rewards.gait_timing_symmetry_l1.weight = -0.08
        self.rewards.termination_penalty.weight = -150.0
        self.rewards.arm_style_prior = RewTerm(
            func=mdp.arm_style_prior_exp,
            weight=0.20,
            params={
                "checkpoint_path": DEFAULT_G1_ARM_STYLE_PRIOR_PATH,
                "std": 0.28,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        if self.events.reset_from_ref is not None:
            self.events.reset_from_ref.params["rsi_ratio"] = 0.20


@configclass
class G1AmpToTargetCommandBalancedEnvCfg(G1AmpToTargetEnvCfg):
    """ToTarget task using command-balanced directional AMP demos."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = G1_COMMAND_BALANCED_DIRECTIONAL_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_json(
            G1_COMMAND_BALANCED_DIRECTIONAL_MOTION_DATA_DIR
        )
        self.motion_data.motion_dataset.root_rot_order = "xyzw"
        self.motion_data.motion_dataset.target_dof_names = G1_LOCOMOTION_JOINT_NAMES
        self.motion_data.motion_dataset.strict_dof_names = True


@configclass
class G1AmpToTargetV2EnvCfg(G1AmpToTargetCommandBalancedEnvCfg):
    """Learnable ToTarget task with balanced AMP demos, dense tracking, and command curriculum."""

    rewards: G1AmpToTargetV2Rewards = G1AmpToTargetV2Rewards()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.radius = (0.0, 0.12)
        self.commands.base_velocity.ranges.heading = (-0.20, 0.20)
        self.commands.base_velocity.success_position_threshold = 0.08
        self.commands.base_velocity.success_heading_threshold = 0.18
        self.commands.base_velocity.success_lin_vel_threshold = 0.08
        self.commands.base_velocity.success_yaw_vel_threshold = 0.12

        self.events.cache_default_key_body_offsets = EventTerm(
            func=mdp.cache_default_key_body_offsets,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=TO_TARGET_SUCCESS_KEY_BODY_NAMES, preserve_order=True
                ),
                "reference_attr": mdp.DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
            },
        )
        self.terminations.target_pose_success_hold = DoneTerm(
            func=mdp.target_pose_success_hold,
            params={
                "command_name": "base_velocity",
                "hold_steps": 8,
                "position_threshold": 0.08,
                "heading_threshold": 0.18,
                "lin_vel_threshold": 0.08,
                "yaw_vel_threshold": 0.12,
                "mean_joint_threshold": 0.16,
                "key_body_mean_threshold": 0.08,
                "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True),
                "key_body_asset_cfg": SceneEntityCfg(
                    "robot", body_names=TO_TARGET_SUCCESS_KEY_BODY_NAMES, preserve_order=True
                ),
                "key_body_reference_attr": mdp.DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
            },
        )

        self.curriculum.to_target_command = CurrTerm(
            func=mdp.to_target_command_curriculum,
            params={
                "command_name": "base_velocity",
                "radius_max_schedule": ((0, 0.12), (24000, 0.25), (72000, 0.60), (144000, 1.00)),
                "heading_abs_schedule": ((0, 0.20), (36000, 0.60), (96000, 1.60), (168000, math.pi)),
                "position_std_schedule": ((0, 0.45), (60000, 0.35), (144000, 0.25)),
                "heading_std_schedule": ((0, 0.75), (96000, 0.55), (168000, 0.40)),
            },
        )

        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waist = None
        self.rewards.termination_penalty.weight = -150.0

        self.rewards.torso_roll_pitch_l2.weight = -2.5
        self.rewards.torso_height_band_l2.weight = -0.8
        self.rewards.torso_ang_vel_xy_l2.weight = -0.14
        self.rewards.torso_vertical_velocity_l2.weight = -0.25
        self.rewards.torso_specific_force_xy_l2.weight = -0.012
        self.rewards.action_rate_l2.weight = -0.014
        self.rewards.dof_torques_l2.weight = -3.0e-6
        self.rewards.dof_acc_l2.weight = -1.5e-7
        self.rewards.feet_air_time.weight = 0.20
        self.rewards.feet_air_time.params["threshold"] = 0.28
        self.rewards.feet_slide.weight = -0.18
        self.rewards.feet_swing_clearance_band_l2.weight = -0.25
        self.rewards.gait_timing_symmetry_l1.weight = -0.12
        self.rewards.arm_style_prior = RewTerm(
            func=mdp.arm_style_prior_exp,
            weight=0.35,
            params={
                "checkpoint_path": DEFAULT_G1_ARM_STYLE_PRIOR_PATH,
                "std": 0.25,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class G1AmpToTargetV2EnvCfg_PLAY(G1AmpToTargetV2EnvCfg):
    """Reduced-env ToTarget v2 task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.radius = (0.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.curriculum.to_target_command = None
        self.events.reset_from_ref = None
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


@configclass
class G1AmpToTargetEnvCfg_PLAY(G1AmpToTargetEnvCfg):
    """Reduced-env ToTarget task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.debug_vis = True
        self.events.reset_from_ref = None
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


@configclass
class G1AmpNav2FinetuneEnvCfg(G1AmpEnvCfg):
    """G1 AMP fine-tune task using recorded Nav2 cmd_vel windows."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.Nav2RecordedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 2.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            data_path=G1_NAV2_AUGMENTED_CMD_DATA_PATH,
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
                lin_vel_x=(-0.20, 0.60),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.60, 0.60),
                heading=None,
            ),
        )

        self.rewards.track_lin_vel_xy_exp.weight = 1.1
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.45
        self.rewards.track_ang_vel_z_exp.weight = 0.9
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.45
        self.rewards.track_torso_lin_vel_xy_exp.weight = 0.35
        self.rewards.track_torso_lin_vel_xy_exp.params["std"] = 0.45
        self.rewards.track_torso_yaw_rate_exp.weight = 0.25
        self.rewards.track_torso_yaw_rate_exp.params["std"] = 0.45
        self.rewards.torso_roll_pitch_l2.weight = -0.04
        self.rewards.torso_ang_vel_xy_l2.weight = 0.0
        self.rewards.torso_vertical_velocity_l2.weight = -0.01
        self.rewards.torso_lateral_vel_cmd_l2.weight = 0.0
        self.rewards.torso_specific_force_xy_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.006


@configclass
class G1AmpNav2FinetuneEnvCfg_PLAY(G1AmpNav2FinetuneEnvCfg):
    """Reduced-env Nav2 command task for policy replay and diagnostics."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
