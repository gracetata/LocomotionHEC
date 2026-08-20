"""ArmHack Walk refinement for precise low-speed tracking and clean swing."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1_FOOT_BODY_NAMES

from .g1_walk_ankle_spacing_env_cfg import G1WalkAnkleSpacingBaseEnvCfg


@configclass
class G1WalkPrecisionSwitchEnvCfg(G1WalkAnkleSpacingBaseEnvCfg):
    """Refine useful commands in [-0.4, 0.4] after an explicit zero start."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.reset_command_to_zero = True
        command.command_clip_min = (-0.40, -0.40, -0.40)
        command.command_clip_max = (0.40, 0.40, 0.40)
        command.mode_command_clip_min = (-0.40, -0.40, -0.40)
        command.mode_command_clip_max = (0.40, 0.40, 0.40)
        command.ranges.lin_vel_x = (-0.40, 0.40)
        command.ranges.lin_vel_y = (-0.40, 0.40)
        command.ranges.ang_vel_z = (-0.40, 0.40)

        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")
        foot_cfg = SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)
        # The inherited value 500 is suitable for discovering a 30-cm gait,
        # but overwhelms the much smaller velocity signal during precision
        # continuation.  Keep a strong nonzero spacing anchor without letting
        # it dominate every useful-command sample.
        self.rewards.ankle_distance_30cm_kernel.weight = 80.0
        self.rewards.precision_torso_velocity_tracking = RewTerm(
            func=mdp.precision_torso_velocity_tracking_exp,
            weight=24.0,
            params={
                "command_name": "base_velocity",
                "min_command": 0.04,
                "max_command": 0.40,
                "lin_std": 0.055,
                "yaw_std": 0.085,
                "asset_cfg": torso_cfg,
            },
        )
        self.rewards.precision_torso_velocity_error = RewTerm(
            func=mdp.precision_torso_velocity_error_l2,
            weight=-16.0,
            params={
                "command_name": "base_velocity",
                "min_command": 0.04,
                "max_command": 0.40,
                "yaw_scale": 0.35,
                "asset_cfg": torso_cfg,
            },
        )
        self.rewards.feet_swing_clearance_band_l2 = RewTerm(
            func=mdp.feet_swing_clearance_band_l2,
            weight=-1.50,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": sensor_cfg,
                "asset_cfg": foot_cfg,
                "target_height": 0.065,
                "std": 0.025,
                "air_time_threshold": 0.04,
                "command_threshold": 0.035,
                "max_height": 0.13,
            },
        )
        self.rewards.feet_slide.weight = -0.40


@configclass
class G1WalkPrecisionSwitchEnvCfg_PLAY(G1WalkPrecisionSwitchEnvCfg):
    """Small deterministic smoke/evaluation variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
