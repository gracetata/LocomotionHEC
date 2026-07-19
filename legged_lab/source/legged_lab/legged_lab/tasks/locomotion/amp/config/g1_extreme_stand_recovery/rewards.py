"""Reward helpers used only by the G1 extreme Stand recovery task."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def default_joint_pose_exp(
    env,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward recovery of every selected joint to its asset default pose.

    The mean-square reduction keeps the scale independent of the number of
    selected joints.  The target is the current asset's randomized default
    joint pose; no disturbance target or future information enters the policy.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    mean_square_error = torch.mean(torch.square(error), dim=1)
    return torch.exp(-mean_square_error / (std * std))
