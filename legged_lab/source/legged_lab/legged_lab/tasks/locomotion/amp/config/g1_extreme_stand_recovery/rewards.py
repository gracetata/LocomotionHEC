"""Reward helpers used only by the G1 extreme Stand recovery task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils import math as math_utils


DEFAULT_CARTESIAN_REFERENCE_ATTR = "_extreme_stand_default_key_body_offsets_b"
DEFAULT_FEET_REFERENCE_ATTR = "_extreme_stand_default_foot_offsets_b"


def _topk_mean(values: torch.Tensor, topk: int) -> torch.Tensor:
    """Return a batch-wise Top-K mean without diluting isolated joint peaks."""
    if values.ndim != 2:
        raise ValueError(f"Top-K reward input must be rank 2, got shape={tuple(values.shape)}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    return torch.topk(values, k=min(int(topk), values.shape[1]), dim=1).values.mean(dim=1)


def _selected_joint_names(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[str]:
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        return list(asset.joint_names[ids])
    return [asset.joint_names[index] for index in ids]


def _joint_group_scale(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    *,
    leg_scale: float,
    waist_scale: float,
    arm_scale: float,
) -> torch.Tensor:
    if min(leg_scale, waist_scale, arm_scale) <= 0.0:
        raise ValueError("Joint normalization scales must all be positive.")
    values = []
    for name in _selected_joint_names(asset, asset_cfg):
        if name.startswith("waist_"):
            values.append(waist_scale)
        elif any(token in name for token in ("shoulder", "elbow", "wrist")):
            values.append(arm_scale)
        else:
            values.append(leg_scale)
    return torch.tensor(values, device=asset.device, dtype=asset.data.joint_pos.dtype).unsqueeze(0)


def _near_default_gate(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    variance: float,
) -> torch.Tensor:
    if variance <= 0.0:
        raise ValueError(f"near-default variance must be positive, got {variance}")
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.exp(-torch.mean(torch.square(error), dim=1) / variance)


def _near_default_rational_score(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    pose_scale: float,
) -> torch.Tensor:
    """Return a non-vanishing default-pose score for corrective rewards.

    A narrow Gaussian is useful as a final accuracy bonus, but it can
    underflow to an effectively zero signal after a large reset or push.  This
    rational score stays differentiable away from the target while remaining
    bounded in ``(0, 1]``.
    """
    if pose_scale <= 0.0:
        raise ValueError(f"near-default pose_scale must be positive, got {pose_scale}")
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    mean_square_error = torch.mean(torch.square(error), dim=1)
    return torch.reciprocal(1.0 + mean_square_error / pose_scale)


def _joint_effort_limits(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1.0e-6)


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


def default_key_body_pose_gaussian(
    env,
    variance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = DEFAULT_CARTESIAN_REFERENCE_ATTR,
) -> torch.Tensor:
    """Give a narrow Gaussian reward only near the default Cartesian pose.

    ``variance`` is expressed in square metres and is intentionally explicit:

    ``reward = exp(-0.5 * mean_keypoint_squared_distance / variance)``.

    A small variance creates a steep peak around the cached asset-default
    posture.  The broader generalized-coordinate rewards still provide a
    recovery gradient when the robot starts far from this peak.
    """
    if variance <= 0.0:
        raise ValueError(
            f"default_key_body_pose_gaussian requires variance > 0, got {variance}"
        )
    asset: Articulation = env.scene[asset_cfg.name]
    current = _key_body_offsets_yaw_frame(asset, asset_cfg)
    reference = _cached_reference(env, current, reference_attr)
    mean_square_distance = torch.mean(
        torch.sum(torch.square(current - reference), dim=-1), dim=1
    )
    return torch.exp(-0.5 * mean_square_distance / variance)


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


def default_feet_distance_gaussian(
    env,
    variance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = DEFAULT_FEET_REFERENCE_ATTR,
) -> torch.Tensor:
    """Give a narrow Gaussian reward around the default planar foot distance."""
    if variance <= 0.0:
        raise ValueError(
            f"default_feet_distance_gaussian requires variance > 0, got {variance}"
        )
    asset: Articulation = env.scene[asset_cfg.name]
    current = _key_body_offsets_yaw_frame(asset, asset_cfg)
    if current.shape[1] != 2:
        raise ValueError("default_feet_distance_gaussian expects exactly two foot bodies.")
    reference = _cached_reference(env, current, reference_attr)
    current_distance = torch.linalg.vector_norm(
        current[:, 0, :2] - current[:, 1, :2], dim=1
    )
    reference_distance = torch.linalg.vector_norm(
        reference[:, 0, :2] - reference[:, 1, :2], dim=1
    )
    square_error = torch.square(current_distance - reference_distance)
    return torch.exp(-0.5 * square_error / variance)


class joint_jerk_l2(ManagerTermBase):
    """Penalize the mean squared time derivative of joint acceleration.

    Acceleration is sampled once per policy/control step, so the finite
    difference is divided by ``env.step_dt`` rather than the physics sub-step.
    The first reward sample after every reset is forced to zero; otherwise the
    reset discontinuity would be incorrectly charged to the policy.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.joint_acc[:, asset_cfg.joint_ids]
        self._previous_joint_acc = torch.zeros_like(current)
        self._history_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_acc[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if env.step_dt <= 0.0:
            raise ValueError(f"joint_jerk_l2 requires env.step_dt > 0, got {env.step_dt}")
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.joint_acc[:, asset_cfg.joint_ids]
        jerk = (current - self._previous_joint_acc) / env.step_dt
        penalty = torch.mean(torch.square(jerk), dim=1)
        penalty = torch.where(self._history_valid, penalty, torch.zeros_like(penalty))
        self._previous_joint_acc.copy_(current)
        self._history_valid.fill_(True)
        return penalty


class action_second_difference_l2(ManagerTermBase):
    """Penalize alternating/high-frequency policy actions.

    The ordinary action-rate term only sees ``a_t - a_(t-1)``.  A policy can
    still produce a two-frame oscillation whose first difference repeatedly
    changes sign.  This term stores the previous action difference and
    penalizes the discrete curvature

    ``(a_t - a_(t-1)) - (a_(t-1) - a_(t-2))``.

    No division by ``step_dt**2`` is used: actions are dimensionless policy
    outputs and the deployment contract fixes their update rate at 50 Hz.
    Keeping this as a per-control-step quantity gives a stable, interpretable
    reward scale.  The first sample after reset is zero so reset discontinuities
    are never attributed to the policy.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._previous_action_delta = torch.zeros_like(env.action_manager.action)
        self._history_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_action_delta[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(self, env) -> torch.Tensor:
        action_delta = env.action_manager.action - env.action_manager.prev_action
        second_difference = action_delta - self._previous_action_delta
        penalty = torch.sum(torch.square(second_difference), dim=1)
        penalty = torch.where(self._history_valid, penalty, torch.zeros_like(penalty))
        self._previous_action_delta.copy_(action_delta)
        self._history_valid.fill_(True)
        return penalty


class joint_torque_rate_l2(ManagerTermBase):
    """Penalize the mean squared time derivative of applied joint torque.

    Unlike the existing torque-magnitude cost, this term directly discourages
    rapid sign changes and large step-to-step torque jumps.  It uses the
    actuator torque actually applied by Isaac rather than an inferred target.
    The first sample after reset is zero to exclude simulator state jumps.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.applied_torque[:, asset_cfg.joint_ids]
        self._previous_joint_torque = torch.zeros_like(current)
        self._history_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_torque[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if env.step_dt <= 0.0:
            raise ValueError(
                f"joint_torque_rate_l2 requires env.step_dt > 0, got {env.step_dt}"
            )
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.applied_torque[:, asset_cfg.joint_ids]
        torque_rate = (current - self._previous_joint_torque) / env.step_dt
        penalty = torch.mean(torch.square(torque_rate), dim=1)
        penalty = torch.where(self._history_valid, penalty, torch.zeros_like(penalty))
        self._previous_joint_torque.copy_(current)
        self._history_valid.fill_(True)
        return penalty


def target_q_default_error_l2(
    env,
    action_term_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the physical PD target's offset from the default joint pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_q = env.action_manager.get_term(action_term_name).processed_actions
    default_q = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(target_q - default_q), dim=1)


class target_q_velocity_l2(ManagerTermBase):
    """Penalize control-step velocity of the physical joint-position target."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        action_term_name = cfg.params.get("action_term_name", "joint_pos")
        current = env.action_manager.get_term(action_term_name).processed_actions
        self._previous_target_q = torch.zeros_like(current)
        self._history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._previous_target_q[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(self, env, action_term_name: str = "joint_pos") -> torch.Tensor:
        if env.step_dt <= 0.0:
            raise ValueError("target_q_velocity_l2 requires a positive control step.")
        current = env.action_manager.get_term(action_term_name).processed_actions
        velocity = (current - self._previous_target_q) / env.step_dt
        penalty = torch.mean(torch.square(velocity), dim=1)
        penalty = torch.where(self._history_valid, penalty, torch.zeros_like(penalty))
        self._previous_target_q.copy_(current)
        self._history_valid.fill_(True)
        return penalty


class target_q_acceleration_l2(ManagerTermBase):
    """Penalize control-step acceleration of the physical joint-position target."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        action_term_name = cfg.params.get("action_term_name", "joint_pos")
        current = env.action_manager.get_term(action_term_name).processed_actions
        self._previous_target_q = torch.zeros_like(current)
        self._previous_target_velocity = torch.zeros_like(current)
        self._history_count = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._previous_target_q[env_ids] = 0.0
        self._previous_target_velocity[env_ids] = 0.0
        self._history_count[env_ids] = 0

    def __call__(self, env, action_term_name: str = "joint_pos") -> torch.Tensor:
        if env.step_dt <= 0.0:
            raise ValueError("target_q_acceleration_l2 requires a positive control step.")
        current = env.action_manager.get_term(action_term_name).processed_actions
        target_velocity = (current - self._previous_target_q) / env.step_dt
        target_acceleration = (target_velocity - self._previous_target_velocity) / env.step_dt
        penalty = torch.mean(torch.square(target_acceleration), dim=1)
        penalty = torch.where(self._history_count >= 2, penalty, torch.zeros_like(penalty))
        self._previous_target_q.copy_(current)
        self._previous_target_velocity.copy_(target_velocity)
        self._history_count.clamp_max_(2).add_(1).clamp_max_(2)
        return penalty


class normalized_joint_jerk_topk_l2(ManagerTermBase):
    """Penalize the worst normalized joint jerks instead of averaging 29 joints."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.joint_acc[:, asset_cfg.joint_ids]
        self._previous_joint_acc = torch.zeros_like(current)
        self._history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._scale = _joint_group_scale(
            asset,
            asset_cfg,
            leg_scale=cfg.params.get("leg_scale", 5000.0),
            waist_scale=cfg.params.get("waist_scale", 3000.0),
            arm_scale=cfg.params.get("arm_scale", 3000.0),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._previous_joint_acc[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(
        self,
        env,
        topk: int = 4,
        leg_scale: float = 5000.0,
        waist_scale: float = 3000.0,
        arm_scale: float = 3000.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del leg_scale, waist_scale, arm_scale
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.joint_acc[:, asset_cfg.joint_ids]
        normalized_jerk = (current - self._previous_joint_acc) / env.step_dt / self._scale
        penalty = _topk_mean(torch.square(normalized_jerk), topk)
        penalty = torch.where(self._history_valid, penalty, torch.zeros_like(penalty))
        self._previous_joint_acc.copy_(current)
        self._history_valid.fill_(True)
        return penalty


def normalized_joint_torque_topk_l2(
    env,
    topk: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the worst applied torques relative to each joint's effort limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    relative = asset.data.applied_torque[:, asset_cfg.joint_ids] / _joint_effort_limits(asset, asset_cfg)
    return _topk_mean(torch.square(relative), topk)


class normalized_joint_torque_rate_topk_l2(ManagerTermBase):
    """Penalize the worst physical torque rates normalized by effort limits."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.applied_torque[:, asset_cfg.joint_ids]
        self._previous_torque = torch.zeros_like(current)
        self._history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._previous_torque[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(
        self,
        env,
        topk: int = 4,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        current = asset.data.applied_torque[:, asset_cfg.joint_ids]
        relative_rate = (current - self._previous_torque) / env.step_dt / _joint_effort_limits(asset, asset_cfg)
        penalty = _topk_mean(torch.square(relative_rate), topk)
        penalty = torch.where(self._history_valid, penalty, torch.zeros_like(penalty))
        self._previous_torque.copy_(current)
        self._history_valid.fill_(True)
        return penalty


def soft_peak_joint_torque_topk_l2(
    env,
    soft_ratio: float = 0.60,
    topk: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize torque above a soft fraction of each joint's effort limit."""
    if not 0.0 < soft_ratio < 1.0:
        raise ValueError(f"soft_ratio must be in (0, 1), got {soft_ratio}")
    asset: Articulation = env.scene[asset_cfg.name]
    relative = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / _joint_effort_limits(asset, asset_cfg)
    return _topk_mean(torch.square(torch.relu(relative - soft_ratio)), topk)


def mechanical_power_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mean squared mechanical joint power ``(torque * velocity)^2``."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(power), dim=1)


class near_default_settle_penalty(ManagerTermBase):
    """Apply strong braking only after the robot is near the default pose.

    Large pose errors leave the recovery policy free to move quickly.  As the
    error shrinks, the Gaussian gate progressively suppresses joint velocity,
    policy action, normalized torque jumps and the worst normalized jerks.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self._previous_acc = torch.zeros_like(asset.data.joint_acc[:, asset_cfg.joint_ids])
        self._previous_torque = torch.zeros_like(asset.data.applied_torque[:, asset_cfg.joint_ids])
        self._history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._jerk_scale = _joint_group_scale(
            asset,
            asset_cfg,
            leg_scale=cfg.params.get("leg_jerk_scale", 5000.0),
            waist_scale=cfg.params.get("waist_jerk_scale", 3000.0),
            arm_scale=cfg.params.get("arm_jerk_scale", 3000.0),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._previous_acc[env_ids] = 0.0
        self._previous_torque[env_ids] = 0.0
        self._history_valid[env_ids] = False

    def __call__(
        self,
        env,
        variance: float = 0.01,
        joint_velocity_weight: float = 5.0,
        action_weight: float = 5.0,
        torque_rate_weight: float = 3.0,
        jerk_weight: float = 3.0,
        topk: int = 4,
        leg_jerk_scale: float = 5000.0,
        waist_jerk_scale: float = 3000.0,
        arm_jerk_scale: float = 3000.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del leg_jerk_scale, waist_jerk_scale, arm_jerk_scale
        asset: Articulation = env.scene[asset_cfg.name]
        joint_velocity = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
        action = torch.mean(torch.square(env.action_manager.action), dim=1)
        jerk = (asset.data.joint_acc[:, asset_cfg.joint_ids] - self._previous_acc) / env.step_dt
        jerk_penalty = _topk_mean(torch.square(jerk / self._jerk_scale), topk)
        # Per-control-step normalized torque jump keeps this composite term's
        # scale interpretable; the separate physical torque-rate term uses /dt.
        torque_jump = (
            asset.data.applied_torque[:, asset_cfg.joint_ids] - self._previous_torque
        ) / _joint_effort_limits(asset, asset_cfg)
        torque_penalty = _topk_mean(torch.square(torque_jump), topk)
        valid = self._history_valid
        jerk_penalty = torch.where(valid, jerk_penalty, torch.zeros_like(jerk_penalty))
        torque_penalty = torch.where(valid, torque_penalty, torch.zeros_like(torque_penalty))
        gate = _near_default_gate(asset, asset_cfg, variance)
        penalty = gate * (
            joint_velocity_weight * joint_velocity
            + action_weight * action
            + torque_rate_weight * torque_penalty
            + jerk_weight * jerk_penalty
        )
        self._previous_acc.copy_(asset.data.joint_acc[:, asset_cfg.joint_ids])
        self._previous_torque.copy_(asset.data.applied_torque[:, asset_cfg.joint_ids])
        self._history_valid.fill_(True)
        return penalty


def _single_disturbance_term(env, event_name: str):
    term = env.event_manager.get_term_cfg(event_name).func
    if not hasattr(term, "has_disturbed") or not hasattr(term, "active_time_left"):
        raise RuntimeError(f"Event term {event_name!r} does not expose disturbance state.")
    return term


def post_disturbance_pose_recovery(
    env,
    event_name: str = "single_disturbance",
    variance: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward fast return to default pose during the guaranteed quiet window."""
    asset: Articulation = env.scene[asset_cfg.name]
    event = _single_disturbance_term(env, event_name)
    quiet = event.has_disturbed & (event.active_time_left <= 0.0)
    return quiet.to(asset.data.joint_pos.dtype) * _near_default_gate(asset, asset_cfg, variance)


def post_disturbance_pose_recovery_rational(
    env,
    event_name: str = "single_disturbance",
    pose_scale: float = 0.0225,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward recovery throughout the quiet window without Gaussian underflow."""
    asset: Articulation = env.scene[asset_cfg.name]
    event = _single_disturbance_term(env, event_name)
    quiet = event.has_disturbed & (event.active_time_left <= 0.0)
    score = _near_default_rational_score(asset, asset_cfg, pose_scale)
    return quiet.to(score.dtype) * score


def post_disturbance_stillness(
    env,
    event_name: str = "single_disturbance",
    pose_variance: float = 0.01,
    velocity_variance: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward near-zero joint speed after recovery, not during the impulse."""
    if velocity_variance <= 0.0:
        raise ValueError("velocity_variance must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    event = _single_disturbance_term(env, event_name)
    quiet = event.has_disturbed & (event.active_time_left <= 0.0)
    speed_error = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-speed_error / velocity_variance)
    return quiet.to(stillness.dtype) * _near_default_gate(asset, asset_cfg, pose_variance) * stillness


def post_disturbance_stillness_rational(
    env,
    event_name: str = "single_disturbance",
    pose_scale: float = 0.0225,
    velocity_scale: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward quiet-window settling with usable gradients after large pushes.

    V5 multiplied two narrow exponentials.  In practice that term was exactly
    zero in TensorBoard for the entire run.  Mean-square velocity and pose are
    instead mapped through rational kernels, so a moving or displaced robot
    still receives a directionally useful signal and the maximum remains one.
    """
    if velocity_scale <= 0.0:
        raise ValueError(f"velocity_scale must be positive, got {velocity_scale}")
    asset: Articulation = env.scene[asset_cfg.name]
    event = _single_disturbance_term(env, event_name)
    quiet = event.has_disturbed & (event.active_time_left <= 0.0)
    velocity_mse = torch.mean(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    stillness_score = torch.reciprocal(1.0 + velocity_mse / velocity_scale)
    pose_score = _near_default_rational_score(asset, asset_cfg, pose_scale)
    return quiet.to(stillness_score.dtype) * pose_score * stillness_score


def near_default_target_lock_penalty(
    env,
    pose_scale: float = 0.0225,
    target_weight: float = 30.0,
    action_weight: float = 5.0,
    joint_velocity_weight: float = 5.0,
    action_term_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Lock the physical PD target to default only after recovery is close.

    Far from the default posture the policy remains free to make decisive
    recovery motions.  As physical joint error shrinks, the bounded rational
    gate strongly suppresses target drift, non-zero policy action and residual
    joint speed.  This directly closes the V5 failure mode where the body was
    nearly upright but the PD target stayed about 0.19 rad from default.
    """
    if min(target_weight, action_weight, joint_velocity_weight) < 0.0:
        raise ValueError("near-default target-lock weights must be non-negative.")
    asset: Articulation = env.scene[asset_cfg.name]
    target_q = env.action_manager.get_term(action_term_name).processed_actions
    default_q = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    target_error = torch.mean(torch.square(target_q - default_q), dim=1)
    action_error = torch.mean(torch.square(env.action_manager.action), dim=1)
    velocity_error = torch.mean(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    pose_score = _near_default_rational_score(asset, asset_cfg, pose_scale)
    # Squaring the score concentrates the lock near the recovered basin while
    # preserving a small corrective gradient at moderate pose error.
    gate = torch.square(pose_score)
    return gate * (
        target_weight * target_error
        + action_weight * action_error
        + joint_velocity_weight * velocity_error
    )
