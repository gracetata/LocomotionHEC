"""Walk-only behavior refinement for strict stop, micro-speed and safe foot placement."""

from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.envs.g1_perturb_env import G1_LOWER_BODY_JOINT_NAMES
import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1_FOOT_BODY_NAMES

from .g1_walk_robust_env_cfg import (
    G1WalkRobustFinetuneEnvCfg,
    G1WalkRobustFinetuneEnvCfg_PLAY,
)


G1_WALK_BEHAVIOR_MODE_CONFIG_PATH = os.path.join(
    LEGGED_LAB_ROOT_DIR,
    "data",
    "MotionData",
    "g1_29dof",
    "amp",
    "armhack_walk_behavior_50hz",
    "task_sampling_config.json",
)


def _configure_walk_behavior(cfg) -> None:
    """Apply behavior terms without mutating a Stand or legacy Walk config."""
    foot_sensor_cfg = SceneEntityCfg(
        "contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
    )
    foot_asset_cfg = SceneEntityCfg(
        "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
    )
    lower_body_cfg = SceneEntityCfg(
        "robot", joint_names=G1_LOWER_BODY_JOINT_NAMES, preserve_order=True
    )

    # This distribution explicitly separates exact zero from every nonzero
    # micro-speed and contains pure-yaw, lateral and diagonal modes.  The policy
    # still observes only [vx, vy, wz], never the sampled mode or future target.
    cfg.commands.base_velocity.mode_sampling_config_path = G1_WALK_BEHAVIOR_MODE_CONFIG_PATH
    cfg.commands.base_velocity.mode_probability = 0.85
    cfg.commands.base_velocity.hard_zero_stand = True
    cfg.commands.base_velocity.mode_command_scale = (1.0, 1.0, 1.0)
    cfg.commands.base_velocity.mode_command_clip_min = (-0.65, -0.45, -1.0)
    cfg.commands.base_velocity.mode_command_clip_max = (1.05, 0.45, 1.0)

    # The inherited translational >0.1 m/s air-time gate creates exactly the
    # unwanted micro-speed and pure-turn dead zones, so it is disabled here.
    cfg.rewards.feet_air_time = None
    cfg.rewards.strict_zero_body_motion_l2 = RewTerm(
        func=mdp.strict_zero_command_body_motion_l2,
        weight=-4.0,
        params={"command_name": "base_velocity", "epsilon": 1.0e-6},
    )
    cfg.rewards.strict_zero_feet_motion_l2 = RewTerm(
        func=mdp.strict_zero_command_feet_motion_l2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": foot_asset_cfg,
            "epsilon": 1.0e-6,
        },
    )
    cfg.rewards.strict_zero_joint_vel_l2 = RewTerm(
        func=mdp.strict_zero_command_joint_vel_l2,
        weight=-0.03,
        params={
            "command_name": "base_velocity",
            "asset_cfg": lower_body_cfg,
            "epsilon": 1.0e-6,
        },
    )
    cfg.rewards.strict_zero_double_support = RewTerm(
        func=mdp.strict_zero_command_double_support,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "epsilon": 1.0e-6,
        },
    )
    cfg.rewards.nonzero_single_stance = RewTerm(
        func=mdp.nonzero_command_single_stance,
        weight=1.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "epsilon": 1.0e-6,
            "min_mode_time": 0.02,
        },
    )
    cfg.rewards.command_response_shortfall_l1 = RewTerm(
        func=mdp.command_response_shortfall_l1,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "epsilon": 1.0e-6,
            "min_speed_fraction": 0.50,
        },
    )
    cfg.rewards.rapid_footstep_l1 = RewTerm(
        func=mdp.rapid_footstep_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
            "epsilon": 1.0e-6,
            "min_air_time": 0.16,
        },
    )
    cfg.rewards.oriented_footprint_proximity_l2 = RewTerm(
        func=mdp.oriented_footprint_proximity_l2,
        weight=-8.0,
        params={
            "asset_cfg": foot_asset_cfg,
            "center_offset_x": 0.035,
            "half_length": 0.090,
            "half_width": 0.035,
            "min_clearance": 0.025,
            "std": 0.020,
            "max_penalty": 25.0,
        },
    )


@configclass
class G1WalkBehaviorFinetuneEnvCfg(G1WalkRobustFinetuneEnvCfg):
    """Continue the completed robust Walk model on behavior-specific failures."""

    def __post_init__(self):
        super().__post_init__()
        _configure_walk_behavior(self)


@configclass
class G1WalkBehaviorFinetuneEnvCfg_PLAY(G1WalkRobustFinetuneEnvCfg_PLAY):
    """Deterministic behavior-refinement evaluation task."""

    def __post_init__(self):
        super().__post_init__()
        _configure_walk_behavior(self)
