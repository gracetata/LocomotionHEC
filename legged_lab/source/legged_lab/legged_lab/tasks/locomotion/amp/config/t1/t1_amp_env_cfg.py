import os
import math
import json
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
from legged_lab.assets.roboparty import T1_29DOF_CFG

# T1 body names used for AMP discriminator key-body positions (currently unused / commented out).
# Ordered in pairs (left, right) consistent with G1 convention:
#   left_foot_link   / right_foot_link    ← equivalent to G1's ankle_roll_link
#   left_hand_link   / right_hand_link    ← equivalent to G1's wrist_yaw_link
#   AL3              / AR3                ← body after shoulder-roll (equiv. G1's shoulder_roll_link)
KEY_BODY_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
    "AL3",
    "AR3",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4
DEFAULT_RSI_RATIO = 0.5
NO_HEAD_JOINT_NAMES = [
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]
NO_HEAD_JOINT_IDS = list(range(2, 29))
FORWARD_BACK_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "t1_29dof_forward_back_filtered_50hz"
)
CMU_WALK_CORE_MOTION_DATA_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "T1_Lab", "cmu_walk_50hz_core"
)
DEFAULT_ARM_STYLE_PRIOR_PATH = "logs/arm_style_prior/t1_cmu_walk_core_arm_prior.pt"


def _motion_weights_from_dir(motion_data_dir: str) -> dict[str, float]:
    if not os.path.isdir(motion_data_dir):
        raise FileNotFoundError(
            f"Forward/back motion directory {motion_data_dir} does not exist. "
            "Run scripts/tools/build_t1_forward_back_motion_subset.py first."
        )
    motion_names = [os.path.splitext(name)[0] for name in sorted(os.listdir(motion_data_dir)) if name.endswith(".pkl")]
    if not motion_names:
        raise ValueError(
            f"Forward/back motion directory {motion_data_dir} contains no .pkl files. "
            "Run scripts/tools/build_t1_forward_back_motion_subset.py first."
        )
    return {name: 1.0 for name in motion_names}


def _read_motion_metadata(motion_data_dir: str) -> dict:
    metadata_path = os.path.join(motion_data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_forward_back_metadata() -> dict:
    return _read_motion_metadata(FORWARD_BACK_MOTION_DATA_DIR)


def _keep_only_command_tracking_rewards(rewards: "T1AmpRewards") -> None:
    """Disable non-command environment rewards while leaving PPOAMP style reward untouched."""

    rewards.flat_orientation_l2 = None
    rewards.lin_vel_z_l2 = None
    rewards.ang_vel_xy_l2 = None
    rewards.dof_torques_l2 = None
    rewards.dof_acc_l2 = None
    rewards.action_rate_l2 = None
    rewards.dof_pos_limits = None
    rewards.shoulder_pitch_pos_limits = None
    rewards.shoulder_pitch_action_l2 = None
    rewards.joint_deviation_hip = None
    rewards.joint_deviation_arms = None
    rewards.joint_deviation_waist = None
    rewards.joint_deviation_head = None
    rewards.feet_air_time = None
    rewards.feet_slide = None
    rewards.termination_penalty = None


def _metadata_range(metadata: dict, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
    values = metadata.get("command_ranges", {}).get(key, fallback)
    lo, hi = float(values[0]), float(values[1])
    if abs(hi - lo) < 1.0e-3:
        lo -= 0.05
        hi += 0.05
    return lo, hi


@configclass
class T1AmpRewards:
    """Reward terms for T1 AMP MDP.

    Mirrors G1AmpRewards exactly, but adapts body/joint name patterns
    to the T1 robot's naming convention.
    """

    # -- task tracking (identical weights to G1)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- penalties (identical weights to G1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # ankle joint limit penalty (equiv. G1's ankle_pitch/roll)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Roll"],
            )
        },
    )

    shoulder_pitch_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Shoulder_Pitch", "Right_Shoulder_Pitch"],
            )
        },
    )

    shoulder_pitch_action_l2 = RewTerm(
        func=mdp.action_l2_selected,
        weight=-0.01,
        params={"action_indices": [0, 7]},
    )

    # keep hip roll/yaw near zero (equiv. G1)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Hip_Yaw", "Right_Hip_Yaw", "Left_Hip_Roll", "Right_Hip_Roll"],
            )
        },
    )

    # Let AMP style shape arm motion; only keep explicit guard rails for shoulder-pitch limit hacks.
    joint_deviation_arms = None

    arm_style_prior = None

    # keep waist near zero (T1 has single Waist joint vs G1's 3 waist joints)
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist"])},
    )

    # keep head joints near zero (T1-specific; G1 has no head joints)
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )

    feet_air_time = None

    # feet slide penalty (T1 feet)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class T1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the T1 AMP environment.

    Aligned with G1AmpEnvCfg as closely as possible:
    - Same reward weights
    - Same command ranges (x-only, -0.2 to 1.5 m/s)
    - Same curriculum (off)
    - Same action scale (0.25, inherited from base)
    - Adapted body/joint names for T1 naming convention
    """

    rewards: T1AmpRewards = T1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # Motion data (retargeted to T1 29-DoF @ 50 Hz)
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "t1_29dof_accad_g1used_50hz_amp_official"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
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

        # ------------------------------------------------------
        # Animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Action/obs DOF alignment (exclude head 2-DoF)
        # ------------------------------------------------------
        self.actions.joint_pos.joint_names = NO_HEAD_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True

        no_head_joint_cfg = SceneEntityCfg("robot", joint_names=NO_HEAD_JOINT_NAMES, preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": no_head_joint_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": no_head_joint_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": no_head_joint_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": no_head_joint_cfg}
        self.observations.disc.joint_pos.params = {"asset_cfg": no_head_joint_cfg}
        self.observations.disc.joint_vel.params = {"asset_cfg": no_head_joint_cfg}
        self.observations.disc_demo.ref_joint_pos.params["joint_ids"] = NO_HEAD_JOINT_IDS
        self.observations.disc_demo.ref_joint_vel.params["joint_ids"] = NO_HEAD_JOINT_IDS

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        self.observations.disc.history_length = AMP_NUM_STEPS

        # discriminator demonstration observations
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_lin_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events: adapt body names to T1 convention
        # ------------------------------------------------------
        # T1 main trunk body is "Trunk" (G1: "torso_link")
        self.events.add_base_mass.params["asset_cfg"].body_names = "Trunk"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]

        # Scale mass of limb bodies; T1 uses CamelCase / non-standard names → use ".*" for all
        self.events.scale_link_mass.params["asset_cfg"].body_names = [".*"]

        # Randomize all T1 joint gains (T1 joints have no "_joint" suffix)
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_from_ref = EventTerm(
            func=mdp.ref_state_init_subset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "motion_dataset": "motion_dataset",
                "rsi_ratio": DEFAULT_RSI_RATIO,
                "pos_rsi": False,
                "height_offset": 0.05,
            },
        )

        # ------------------------------------------------------
        # Commands (x-only, -0.2 to 1.5 m/s, same as G1)
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # ------------------------------------------------------
        # Curriculum: disabled (same as current G1 training)
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # ------------------------------------------------------
        # Terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None


@configclass
class T1AmpEnvCfg_PLAY(T1AmpEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class T1AmpForwardBackEnvCfg(T1AmpEnvCfg):
    """T1 AMP task restricted to the measured forward/back demonstration space."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = FORWARD_BACK_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_dir(FORWARD_BACK_MOTION_DATA_DIR)

        metadata = _read_forward_back_metadata()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = _metadata_range(metadata, "lin_vel_x", (-0.6, 0.9))
        self.commands.base_velocity.ranges.lin_vel_y = _metadata_range(metadata, "lin_vel_y", (-0.12, 0.13))
        self.commands.base_velocity.ranges.ang_vel_z = _metadata_range(metadata, "ang_vel_z", (-0.58, 0.58))
        self.commands.base_velocity.ranges.heading = None


@configclass
class T1AmpForwardBackEnvCfg_PLAY(T1AmpForwardBackEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class T1AmpCmuWalkCoreEnvCfg(T1AmpEnvCfg):
    """T1 AMP task using the Lab-converted CMU walk core demo set."""

    def __post_init__(self):
        super().__post_init__()

        self.motion_data.motion_dataset.motion_data_dir = CMU_WALK_CORE_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights_from_dir(CMU_WALK_CORE_MOTION_DATA_DIR)

        metadata = _read_motion_metadata(CMU_WALK_CORE_MOTION_DATA_DIR)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = _metadata_range(metadata, "lin_vel_x", (0.5, 0.7))
        self.commands.base_velocity.ranges.lin_vel_y = _metadata_range(metadata, "lin_vel_y", (-0.06, 0.06))
        self.commands.base_velocity.ranges.ang_vel_z = _metadata_range(metadata, "ang_vel_z", (-0.45, 0.45))
        self.commands.base_velocity.ranges.heading = None


@configclass
class T1AmpCmuWalkCoreCommandOnlyEnvCfg(T1AmpCmuWalkCoreEnvCfg):
    """CMU walk core task with only command-tracking environment rewards.

    PPOAMP still adds AMP style reward outside the IsaacLab RewardManager; this
    variant removes shaping penalties that can pull upper-body motion away from
    the demonstration style during late training.
    """

    def __post_init__(self):
        super().__post_init__()

        _keep_only_command_tracking_rewards(self.rewards)


@configclass
class T1AmpCmuWalkCoreCommandOnlyArmPriorEnvCfg(T1AmpCmuWalkCoreCommandOnlyEnvCfg):
    """Command-only CMU walk core task with a supervised arm-style prior reward."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.arm_style_prior = RewTerm(
            func=mdp.arm_style_prior_exp,
            weight=1.0,
            params={
                "checkpoint_path": DEFAULT_ARM_STYLE_PRIOR_PATH,
                "std": 0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class T1AmpCmuWalkCoreEnvCfg_PLAY(T1AmpCmuWalkCoreEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
