"""AMP locomotion command generators.

Core exports:
    CurvatureVelocityCommandCfg and CurvatureVelocityCommand keep the historical
    class name but define a mixed low-speed uniform-yaw / high-speed Gaussian-yaw
    velocity command for G1 AMP fine-tuning. Nav2RecordedVelocityCommandCfg and
    Nav2RecordedVelocityCommand replay successful Nav2 cmd_vel windows while
    preserving vx/vy/wz correlations. RelativePose2dCommandCfg and
    RelativePose2dCommand emit reset-relative short-range SE(2) target errors.

Inputs/outputs:
    The command term outputs the same three values consumed by the existing
    policy observation: body-frame lin_vel_x, lin_vel_y, and yaw_rate. It does
    not change observation or action dimensions.

Usage:
    from legged_lab.tasks.locomotion.amp.mdp.commands import CurvatureVelocityCommandCfg
    cfg.commands.base_velocity = CurvatureVelocityCommandCfg(asset_name="robot", ...)
    from legged_lab.tasks.locomotion.amp.mdp.commands import Nav2RecordedVelocityCommandCfg
    cfg.commands.base_velocity = Nav2RecordedVelocityCommandCfg(asset_name="robot", data_path="...csv")
"""

from .curvature_velocity_command import CurvatureVelocityCommand, CurvatureVelocityCommandCfg
from .hybrid_nav2_mode_velocity_command import (
    HybridNav2ModeVelocityCommand,
    HybridNav2ModeVelocityCommandCfg,
)
from .mode_balanced_velocity_command import ModeBalancedVelocityCommand, ModeBalancedVelocityCommandCfg
from .nav2_recorded_velocity_command import Nav2RecordedVelocityCommand, Nav2RecordedVelocityCommandCfg
from .relative_pose_command import RelativePose2dCommand, RelativePose2dCommandCfg

__all__ = [
    "CurvatureVelocityCommand",
    "CurvatureVelocityCommandCfg",
    "HybridNav2ModeVelocityCommand",
    "HybridNav2ModeVelocityCommandCfg",
    "ModeBalancedVelocityCommand",
    "ModeBalancedVelocityCommandCfg",
    "Nav2RecordedVelocityCommand",
    "Nav2RecordedVelocityCommandCfg",
    "RelativePose2dCommand",
    "RelativePose2dCommandCfg",
]
