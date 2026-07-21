"""Reward helpers used only by the G1 extreme Stand recovery task."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


DEFAULT_CARTESIAN_REFERENCE_ATTR = "_extreme_stand_default_key_body_offsets_b"
DEFAULT_FEET_REFERENCE_ATTR = "_extreme_stand_default_foot_offsets_b"


def _key_body_offsets_yaw_frame(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected Cartesian body positions relative to root in the root-yaw frame."""
    body_ids = list(asset_cfg.body_ids)
    if not body_ids:
        raise ValueError("Cartesian default-pose rewards require at least one body id.")
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    root_yaw_quat_w = math_utils.yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(body_ids), -1)
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    return math_utils.quat_apply_inverse(root_yaw_quat_w, body_pos_w - root_pos_w)


def _cached_reference(env, current: torch.Tensor, reference_attr: str) -> torch.Tensor:
    reference = getattr(env, reference_attr, None)
    if reference is None:
        raise RuntimeError(
            f"Missing cached default Cartesian reference: {reference_attr}. "
            "Configure cache_default_key_body_offsets as a startup event."
        )
    if reference.shape != current.shape:
        raise RuntimeError(
            f"Cached Cartesian reference shape mismatch for {reference_attr}: "
            f"reference={tuple(reference.shape)}, current={tuple(current.shape)}"
        )
    return reference.to(device=current.device, dtype=current.dtype)


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


def default_key_body_pose_exp(
    env,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = DEFAULT_CARTESIAN_REFERENCE_ATTR,
) -> torch.Tensor:
    """Reward Cartesian key-body recovery to the cached asset-default pose.

    Positions are expressed relative to the root in its yaw frame.  Therefore
    global translation and heading do not change the posture target, while
    joint-induced limb displacement and root roll/pitch still contribute.
    """
    if std <= 0.0:
        raise ValueError(f"default_key_body_pose_exp requires std > 0, got {std}")
    asset: Articulation = env.scene[asset_cfg.name]
    current = _key_body_offsets_yaw_frame(asset, asset_cfg)
    reference = _cached_reference(env, current, reference_attr)
    mean_square_distance = torch.mean(torch.sum(torch.square(current - reference), dim=-1), dim=1)
    return torch.exp(-mean_square_distance / (std * std))


def default_feet_distance_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = DEFAULT_FEET_REFERENCE_ATTR,
) -> torch.Tensor:
    """Squared error from the cached default planar distance between both feet.

    This is symmetric: both an overly narrow and an overly wide stance are
    penalized.  The target comes from the asset default pose rather than a
    hard-coded distance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current = _key_body_offsets_yaw_frame(asset, asset_cfg)
    if current.shape[1] != 2:
        raise ValueError("default_feet_distance_l2 expects exactly two foot bodies.")
    reference = _cached_reference(env, current, reference_attr)
    current_distance = torch.linalg.vector_norm(current[:, 0, :2] - current[:, 1, :2], dim=1)
    reference_distance = torch.linalg.vector_norm(reference[:, 0, :2] - reference[:, 1, :2], dim=1)
    return torch.square(current_distance - reference_distance)
