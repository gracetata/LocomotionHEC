"""Reward helpers for AMP locomotion tasks.

Core utilities:
    This file defines task tracking, regularization, contact, style-prior, and
    torso-mounted IMU stability reward terms used by AMP locomotion configs.

Inputs/outputs:
    Each reward function receives a ManagerBasedRLEnv and optional
    SceneEntityCfg selectors, then returns one scalar tensor per environment.
    The torso helpers read body_pos/body_quat/body velocity/acceleration for
    torso_link and compare them against command-perfect or upright values.

Usage:
    RewTerm(func=mdp.track_torso_lin_vel_xy_exp, weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5,
                    "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils

from .reward_math import (
    allowed_footstep_cadence_hz,
    convex_footprint_signed_clearance_xy,
    nonzero_single_stance_command_scale,
    relative_command_response_shortfall_l1,
    swept_convex_footprint_signed_clearance_xy,
    touchdown_pose_progress,
    two_goal_command_masks,
    two_goal_dense_root_pose_progress,
    two_goal_response_shortfall_l2,
    update_footstep_cadence_state,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ARM_STYLE_PRIOR_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, dict[str, torch.Tensor | list[str]]]] = {}
GRAVITY_ACCELERATION_M_PER_S2 = 9.80665
DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR = "_to_target_default_key_body_offsets_b"


class _ArmStylePrior(torch.nn.Module):
    """MLP architecture matching scripts/tools/train_t1_arm_style_prior.py."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _load_arm_style_prior(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, torch.Tensor | list[str]]]:
    resolved_path = Path(checkpoint_path)
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    cache_key = (str(resolved_path), str(device))
    if cache_key in _ARM_STYLE_PRIOR_CACHE:
        return _ARM_STYLE_PRIOR_CACHE[cache_key]
    if not resolved_path.exists():
        raise FileNotFoundError(f"Arm style prior checkpoint not found: {resolved_path}")

    payload = torch.load(resolved_path, map_location=device, weights_only=False)
    input_mean = payload["input_mean"].to(device=device, dtype=torch.float32)
    input_std = payload["input_std"].to(device=device, dtype=torch.float32)
    target_mean = payload["target_mean"].to(device=device, dtype=torch.float32)
    target_std = payload["target_std"].to(device=device, dtype=torch.float32)
    model = _ArmStylePrior(input_mean.numel(), target_mean.numel(), list(payload["hidden_dims"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata: dict[str, torch.Tensor | list[str]] = {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "input_joint_names": list(payload["input_joint_names"]),
        "target_joint_names": list(payload["target_joint_names"]),
    }
    _ARM_STYLE_PRIOR_CACHE[cache_key] = (model, metadata)
    return model, metadata


def _joint_ids_from_names(asset: Articulation, joint_names: list[str]) -> list[int]:
    name_to_id = {name: index for index, name in enumerate(asset.joint_names)}
    missing = [name for name in joint_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Arm style prior joint names are missing from asset: {missing}")
    return [name_to_id[name] for name in joint_names]


def _single_body_id(asset_cfg: SceneEntityCfg) -> int:
    body_ids = list(asset_cfg.body_ids)
    if not body_ids:
        raise ValueError(f"Reward term for asset '{asset_cfg.name}' requires exactly one body id.")
    return int(body_ids[0])


def _body_lin_vel_yaw_frame(asset: RigidObject, body_id: int) -> torch.Tensor:
    body_quat_w = asset.data.body_quat_w[:, body_id, :]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_id, :]
    return math_utils.quat_apply_inverse(math_utils.yaw_quat(body_quat_w), body_lin_vel_w)


def _body_ang_vel_body_frame(asset: RigidObject, body_id: int) -> torch.Tensor:
    body_quat_w = asset.data.body_quat_w[:, body_id, :]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_id, :]
    return math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)


def _body_upright_scale(asset: RigidObject, body_id: int) -> torch.Tensor:
    body_quat_w = asset.data.body_quat_w[:, body_id, :]
    projected_gravity = math_utils.quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)
    return torch.clamp(-projected_gravity[:, 2], 0.0, 0.7) / 0.7


def _upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _adaptive_tracking_sigma(
    env: ManagerBasedRLEnv,
    key: str,
    error: torch.Tensor,
    std: float,
    ema_decay: float,
    min_sigma: float,
) -> torch.Tensor:
    initial_sigma = max(float(std) ** 2, float(min_sigma))
    state = getattr(env, "_adaptive_tracking_reward_state", None)
    if state is None:
        state = {}
        setattr(env, "_adaptive_tracking_reward_state", state)

    if key not in state:
        state[key] = {
            "ema_error": torch.tensor(initial_sigma, dtype=error.dtype, device=error.device),
            "sigma": torch.tensor(initial_sigma, dtype=error.dtype, device=error.device),
        }

    term_state = state[key]
    mean_error = torch.mean(error.detach())
    decay = min(max(float(ema_decay), 0.0), 1.0)
    ema_error = term_state["ema_error"].to(dtype=error.dtype, device=error.device)
    sigma = term_state["sigma"].to(dtype=error.dtype, device=error.device)
    ema_error = ema_error * decay + mean_error * (1.0 - decay)
    sigma = torch.clamp(torch.minimum(sigma, ema_error), min=float(min_sigma))
    term_state["ema_error"] = ema_error.detach()
    term_state["sigma"] = sigma.detach()

    extras = getattr(env, "extras", None)
    if extras is not None:
        log_extras = extras.setdefault("log", {})
        log_extras[f"Adaptive Tracking/{key}_sigma"] = sigma.detach()
        log_extras[f"Adaptive Tracking/{key}_ema_error"] = ema_error.detach()
        log_extras[f"Adaptive Tracking/{key}_mean_error"] = mean_error.detach()

    return sigma
    
    
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # return torch.exp(-lin_vel_error / std**2)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # return torch.exp(-ang_vel_error / std**2)
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_adaptive_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    ema_decay: float = 0.995,
    min_sigma: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward XY command tracking with a monotonically tightening adaptive sigma."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    sigma = _adaptive_tracking_sigma(env, "lin_vel_xy", lin_vel_error, std, ema_decay, min_sigma)
    return torch.exp(-lin_vel_error / sigma) * _upright_scale(env)


def track_ang_vel_z_adaptive_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    ema_decay: float = 0.995,
    min_sigma: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate command tracking with a monotonically tightening adaptive sigma."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    sigma = _adaptive_tracking_sigma(env, "ang_vel_z", ang_vel_error, std, ema_decay, min_sigma)
    return torch.exp(-ang_vel_error / sigma) * _upright_scale(env)


def _target_pose_errors(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "_update_command"):
        command_term._update_command()
    command = command_term.command
    position_error = torch.norm(command[:, :2], dim=1)
    heading_error = torch.abs(math_utils.wrap_to_pi(command[:, 2]))
    return command, position_error, heading_error


def _target_stop_latched(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    stop_latched = getattr(command_term, "stop_latched", None)
    if stop_latched is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return stop_latched


def _key_body_offsets_yaw_frame(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    body_ids = list(asset_cfg.body_ids)
    if not body_ids:
        raise ValueError("Key-body offset rewards require at least one body id.")
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    root_yaw_quat_w = math_utils.yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(body_ids), -1)
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    return math_utils.quat_apply_inverse(root_yaw_quat_w, body_pos_w - root_pos_w)


def _target_key_body_reference(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    asset_cfg: SceneEntityCfg,
    reference_attr: str = DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
) -> torch.Tensor:
    current_offsets = _key_body_offsets_yaw_frame(asset, asset_cfg)
    reference = getattr(env, reference_attr, None)
    if (
        reference is None
        or reference.shape != current_offsets.shape
        or reference.device != current_offsets.device
        or reference.dtype != current_offsets.dtype
    ):
        reference = current_offsets.detach().clone()
        setattr(env, reference_attr, reference)
    return reference


def target_key_body_mean_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
) -> torch.Tensor:
    """Mean root-yaw-frame distance from cached default key-body offsets."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_offsets = _key_body_offsets_yaw_frame(asset, asset_cfg)
    reference_offsets = _target_key_body_reference(env, asset, asset_cfg, reference_attr)
    return torch.mean(torch.norm(current_offsets - reference_offsets, dim=-1), dim=1)


def target_pose_success_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float = 0.08,
    heading_threshold: float = 0.18,
    lin_vel_threshold: float = 0.08,
    yaw_vel_threshold: float = 0.12,
    mean_joint_threshold: float = 0.16,
    key_body_mean_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    key_body_asset_cfg: SceneEntityCfg | None = None,
    key_body_reference_attr: str = DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
) -> torch.Tensor:
    """Return success when the target pose, residual motion, and rest posture all match thresholds."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, position_error, heading_error = _target_pose_errors(env, command_name)
    lin_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    mean_joint_error = torch.mean(joint_error, dim=1)
    success = (
        (position_error <= float(position_threshold))
        & (heading_error <= float(heading_threshold))
        & (lin_speed <= float(lin_vel_threshold))
        & (yaw_rate <= float(yaw_vel_threshold))
        & (mean_joint_error <= float(mean_joint_threshold))
    )
    if key_body_asset_cfg is not None and key_body_mean_threshold is not None:
        key_body_error = target_key_body_mean_error(env, key_body_asset_cfg, key_body_reference_attr)
        success = success & (key_body_error <= float(key_body_mean_threshold))
    return success


def _interpolate_schedule(step: int, schedule: Sequence[tuple[int, float]]) -> float:
    """Piecewise-linear scalar schedule used by ToTarget curriculum terms."""
    if not schedule:
        raise ValueError("Schedule must contain at least one milestone.")
    milestones = sorted((int(item[0]), float(item[1])) for item in schedule)
    if step <= milestones[0][0]:
        return milestones[0][1]
    for (left_step, left_value), (right_step, right_value) in zip(milestones[:-1], milestones[1:]):
        if step <= right_step:
            span = max(right_step - left_step, 1)
            alpha = min(max((step - left_step) / span, 0.0), 1.0)
            return left_value + alpha * (right_value - left_value)
    return milestones[-1][1]


def _step_ramp(env: ManagerBasedRLEnv, warmup_steps: int = 0, ramp_steps: int = 0) -> torch.Tensor | float:
    """Return a scalar ramp in [0, 1] based on the environment step counter."""
    step = int(env.common_step_counter)
    warmup_steps = max(int(warmup_steps), 0)
    ramp_steps = max(int(ramp_steps), 0)
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(max((step - warmup_steps) / ramp_steps, 0.0), 1.0)


def to_target_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    radius_max_schedule: Sequence[tuple[int, float]],
    heading_abs_schedule: Sequence[tuple[int, float]],
    position_std_schedule: Sequence[tuple[int, float]] | None = None,
    heading_std_schedule: Sequence[tuple[int, float]] | None = None,
) -> dict[str, float]:
    """Expand ToTarget command ranges and optionally tighten dense pose reward sigmas."""
    del env_ids
    step = int(env.common_step_counter)
    radius_max = _interpolate_schedule(step, radius_max_schedule)
    heading_abs = _interpolate_schedule(step, heading_abs_schedule)

    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.radius = (0.0, max(float(radius_max), 0.0))
    command_term.cfg.ranges.heading = (-max(float(heading_abs), 0.0), max(float(heading_abs), 0.0))

    state = {
        "radius_max": float(command_term.cfg.ranges.radius[1]),
        "heading_abs": float(command_term.cfg.ranges.heading[1]),
    }

    if position_std_schedule is not None and "target_position_exp" in env.reward_manager.active_terms:
        std = _interpolate_schedule(step, position_std_schedule)
        term_cfg = env.reward_manager.get_term_cfg("target_position_exp")
        term_cfg.params["std"] = float(std)
        env.reward_manager.set_term_cfg("target_position_exp", term_cfg)
        state["position_std"] = float(std)

    if heading_std_schedule is not None and "target_heading_exp" in env.reward_manager.active_terms:
        std = _interpolate_schedule(step, heading_std_schedule)
        term_cfg = env.reward_manager.get_term_cfg("target_heading_exp")
        term_cfg.params["std"] = float(std)
        env.reward_manager.set_term_cfg("target_heading_exp", term_cfg)
        state["heading_std"] = float(std)

    return state


def _target_desired_velocity(
    command: torch.Tensor,
    position_error: torch.Tensor,
    heading_error: torch.Tensor,
    position_gain: float,
    max_lin_speed: float,
    heading_gain: float,
    max_yaw_rate: float,
    stop_distance: float,
    stop_heading: float,
    stop_latched: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    desired_lin = command[:, :2] * float(position_gain)
    desired_lin_norm = torch.norm(desired_lin, dim=1, keepdim=True)
    desired_lin = desired_lin * torch.clamp(float(max_lin_speed) / torch.clamp(desired_lin_norm, min=1.0e-6), max=1.0)

    desired_yaw = torch.clamp(command[:, 2] * float(heading_gain), -float(max_yaw_rate), float(max_yaw_rate))
    should_stop = stop_latched | (
        (position_error < float(stop_distance)) & (heading_error < float(stop_heading))
    )
    desired_lin = desired_lin * (~should_stop).unsqueeze(-1).float()
    desired_yaw = desired_yaw * (~should_stop).float()
    return desired_lin, desired_yaw, should_stop


def target_lin_vel_tracking_adaptive_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.45,
    ema_decay: float = 0.995,
    min_sigma: float = 0.03,
    position_gain: float = 1.25,
    max_lin_speed: float = 0.65,
    stop_distance: float = 0.08,
    stop_heading: float = 0.20,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a dense velocity target induced by the remaining SE(2) pose error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command, position_error, heading_error = _target_pose_errors(env, command_name)
    stop_latched = _target_stop_latched(env, command_name)
    desired_lin, _, _ = _target_desired_velocity(
        command,
        position_error,
        heading_error,
        position_gain=position_gain,
        max_lin_speed=max_lin_speed,
        heading_gain=1.0,
        max_yaw_rate=1.0,
        stop_distance=stop_distance,
        stop_heading=stop_heading,
        stop_latched=stop_latched,
    )
    lin_vel_error = torch.sum(torch.square(desired_lin - asset.data.root_lin_vel_b[:, :2]), dim=1)
    sigma = _adaptive_tracking_sigma(env, "target_lin_vel_xy", lin_vel_error, std, ema_decay, min_sigma)
    return torch.exp(-lin_vel_error / sigma) * _upright_scale(env) * _step_ramp(env, warmup_steps, ramp_steps)


def target_yaw_rate_tracking_adaptive_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.75,
    ema_decay: float = 0.995,
    min_sigma: float = 0.04,
    heading_gain: float = 1.50,
    max_yaw_rate: float = 1.20,
    stop_distance: float = 0.08,
    stop_heading: float = 0.20,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a dense yaw-rate target induced by the remaining heading error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command, position_error, heading_error = _target_pose_errors(env, command_name)
    stop_latched = _target_stop_latched(env, command_name)
    _, desired_yaw, _ = _target_desired_velocity(
        command,
        position_error,
        heading_error,
        position_gain=1.0,
        max_lin_speed=1.0,
        heading_gain=heading_gain,
        max_yaw_rate=max_yaw_rate,
        stop_distance=stop_distance,
        stop_heading=stop_heading,
        stop_latched=stop_latched,
    )
    yaw_rate_error = torch.square(desired_yaw - asset.data.root_ang_vel_b[:, 2])
    sigma = _adaptive_tracking_sigma(env, "target_yaw_rate", yaw_rate_error, std, ema_decay, min_sigma)
    return torch.exp(-yaw_rate_error / sigma) * _upright_scale(env) * _step_ramp(env, warmup_steps, ramp_steps)


def target_position_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    stop_latch_gate: bool = False,
) -> torch.Tensor:
    """Reward current XY distance to a reset-relative SE(2) target."""
    _, position_error, _ = _target_pose_errors(env, command_name)
    sigma = max(float(std) ** 2, 1.0e-6)
    reward = torch.exp(-torch.square(position_error) / sigma) * _upright_scale(env)
    if stop_latch_gate:
        reward = reward * (~_target_stop_latched(env, command_name)).float()
    return reward


def target_heading_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    stop_latch_gate: bool = False,
) -> torch.Tensor:
    """Reward current yaw error to a reset-relative SE(2) target."""
    _, _, heading_error = _target_pose_errors(env, command_name)
    sigma = max(float(std) ** 2, 1.0e-6)
    reward = torch.exp(-torch.square(heading_error) / sigma) * _upright_scale(env)
    if stop_latch_gate:
        reward = reward * (~_target_stop_latched(env, command_name)).float()
    return reward


def target_approach_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_scale: float = 0.35,
    stop_distance: float = 0.10,
    stop_latch_gate: bool = True,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity projected toward the remaining target vector while far from the target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command, position_error, _ = _target_pose_errors(env, command_name)
    direction = command[:, :2] / torch.clamp(position_error.unsqueeze(-1), min=1.0e-6)
    projected_velocity = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    reward = torch.tanh(projected_velocity / max(float(speed_scale), 1.0e-6))
    active = position_error > float(stop_distance)
    if stop_latch_gate:
        active = active & (~_target_stop_latched(env, command_name))
    return reward * active.float() * _upright_scale(env) * _step_ramp(env, warmup_steps, ramp_steps)


def target_pose_stillness_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float = 0.12,
    heading_std: float = 0.20,
    lin_vel_std: float = 0.08,
    yaw_vel_std: float = 0.12,
    use_stop_latch: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward being stopped only when already close to the SE(2) target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    _, position_error, heading_error = _target_pose_errors(env, command_name)
    pose_gate = torch.exp(
        -torch.square(position_error) / max(float(position_std) ** 2, 1.0e-6)
        - torch.square(heading_error) / max(float(heading_std) ** 2, 1.0e-6)
    )
    lin_speed_sq = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    yaw_rate_sq = torch.square(asset.data.root_ang_vel_b[:, 2])
    stillness = torch.exp(
        -lin_speed_sq / max(float(lin_vel_std) ** 2, 1.0e-6)
        - yaw_rate_sq / max(float(yaw_vel_std) ** 2, 1.0e-6)
    )
    reward = pose_gate * stillness * _upright_scale(env)
    if use_stop_latch:
        reward = reward * _target_stop_latched(env, command_name).float()
    return reward


def near_target_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float = 0.18,
    heading_std: float = 0.30,
    yaw_weight: float = 0.5,
    use_stop_latch: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual root motion with a smooth gate around the target pose."""
    asset: RigidObject = env.scene[asset_cfg.name]
    _, position_error, heading_error = _target_pose_errors(env, command_name)
    pose_gate = torch.exp(
        -torch.square(position_error) / max(float(position_std) ** 2, 1.0e-6)
        - torch.square(heading_error) / max(float(heading_std) ** 2, 1.0e-6)
    )
    if use_stop_latch:
        pose_gate = torch.maximum(pose_gate, _target_stop_latched(env, command_name).float())
    velocity_l2 = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    velocity_l2 += float(yaw_weight) * torch.square(asset.data.root_ang_vel_b[:, 2])
    return pose_gate * velocity_l2


def near_target_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float = 0.16,
    heading_std: float = 0.25,
    use_stop_latch: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mean joint deviation from the default pose only near the target."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, position_error, heading_error = _target_pose_errors(env, command_name)
    pose_gate = torch.exp(
        -torch.square(position_error) / max(float(position_std) ** 2, 1.0e-6)
        - torch.square(heading_error) / max(float(heading_std) ** 2, 1.0e-6)
    )
    if use_stop_latch:
        pose_gate = torch.maximum(pose_gate, _target_stop_latched(env, command_name).float())
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    return torch.mean(joint_error, dim=1) * pose_gate


def near_target_key_body_deviation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float = 0.16,
    heading_std: float = 0.25,
    use_stop_latch: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
) -> torch.Tensor:
    """Penalize key-body offsets from the cached default pose only near the target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    _, position_error, heading_error = _target_pose_errors(env, command_name)
    pose_gate = torch.exp(
        -torch.square(position_error) / max(float(position_std) ** 2, 1.0e-6)
        - torch.square(heading_error) / max(float(heading_std) ** 2, 1.0e-6)
    )
    if use_stop_latch:
        pose_gate = torch.maximum(pose_gate, _target_stop_latched(env, command_name).float())
    current_offsets = _key_body_offsets_yaw_frame(asset, asset_cfg)
    reference_offsets = _target_key_body_reference(env, asset, asset_cfg, reference_attr)
    key_body_error_l2 = torch.mean(torch.sum(torch.square(current_offsets - reference_offsets), dim=-1), dim=1)
    return key_body_error_l2 * pose_gate


def target_stop_latch_drift_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_tolerance: float = 0.08,
    heading_tolerance: float = 0.18,
    position_std: float = 0.05,
    heading_std: float = 0.12,
) -> torch.Tensor:
    """Penalize leaving the target pose after sticky stop mode has been entered."""
    _, position_error, heading_error = _target_pose_errors(env, command_name)
    stop_latched = _target_stop_latched(env, command_name).float()
    position_violation = torch.clamp(position_error - float(position_tolerance), min=0.0)
    heading_violation = torch.clamp(heading_error - float(heading_tolerance), min=0.0)
    drift = torch.square(position_violation / max(float(position_std), 1.0e-6))
    drift += torch.square(heading_violation / max(float(heading_std), 1.0e-6))
    return drift * stop_latched


def target_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float = 0.06,
    heading_threshold: float = 0.12,
    lin_vel_threshold: float = 0.06,
    yaw_vel_threshold: float = 0.10,
    mean_joint_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    key_body_mean_threshold: float | None = None,
    key_body_asset_cfg: SceneEntityCfg | None = None,
    key_body_reference_attr: str = DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
) -> torch.Tensor:
    """Sparse bonus for reaching, stopping, and returning close to the default pose."""
    success = target_pose_success_mask(
        env,
        command_name=command_name,
        position_threshold=position_threshold,
        heading_threshold=heading_threshold,
        lin_vel_threshold=lin_vel_threshold,
        yaw_vel_threshold=yaw_vel_threshold,
        mean_joint_threshold=mean_joint_threshold,
        asset_cfg=asset_cfg,
        key_body_mean_threshold=key_body_mean_threshold,
        key_body_asset_cfg=key_body_asset_cfg,
        key_body_reference_attr=key_body_reference_attr,
    )
    return success.float() * _upright_scale(env)


def track_torso_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Reward torso yaw-frame XY velocity tracking against the command in m/s."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    torso_lin_vel_yaw_b = _body_lin_vel_yaw_frame(asset, body_id)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - torso_lin_vel_yaw_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2) * _body_upright_scale(asset, body_id)


def track_torso_yaw_rate_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Reward torso yaw-rate tracking against the command in rad/s."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    torso_ang_vel_b = _body_ang_vel_body_frame(asset, body_id)
    yaw_rate_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - torso_ang_vel_b[:, 2])
    return torch.exp(-yaw_rate_error / std**2) * _body_upright_scale(asset, body_id)


def torso_roll_pitch_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    """Penalize torso roll/pitch errors from the upright target in rad^2."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    body_quat_w = asset.data.body_quat_w[:, body_id, :]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(body_quat_w)
    return torch.square(math_utils.wrap_to_pi(roll)) + torch.square(math_utils.wrap_to_pi(pitch))


def torso_ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    """Penalize torso roll/pitch angular velocity in (rad/s)^2."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    torso_ang_vel_b = _body_ang_vel_body_frame(asset, body_id)
    return torch.sum(torch.square(torso_ang_vel_b[:, :2]), dim=1)


def torso_vertical_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    """Penalize torso vertical velocity error from 0 m/s."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    return torch.square(asset.data.body_lin_vel_w[:, body_id, 2])


def torso_height_band_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    lower_deadband: float,
    upper_deadband: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Penalize torso height only outside an asymmetric target band."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    height_error = asset.data.body_pos_w[:, body_id, 2] - target_height
    lower_violation = torch.clamp(lower_deadband - height_error, min=0.0)
    upper_violation = torch.clamp(height_error - upper_deadband, min=0.0)
    band_violation = lower_violation + upper_violation
    return torch.square(band_violation / max(float(std), 1.0e-6))


def torso_lateral_vel_cmd_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Penalize torso lateral velocity error against command_y in (m/s)^2."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    torso_lin_vel_yaw_b = _body_lin_vel_yaw_frame(asset, body_id)
    return torch.square(torso_lin_vel_yaw_b[:, 1] - env.command_manager.get_command(command_name)[:, 1])


def torso_specific_force_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    """Penalize horizontal accelerometer specific-force error in (m/s^2)^2."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    body_quat_w = asset.data.body_quat_w[:, body_id, :]
    body_lin_acc_w = asset.data.body_lin_acc_w[:, body_id, :]
    gravity_acc_w = torch.tensor(
        [0.0, 0.0, -GRAVITY_ACCELERATION_M_PER_S2], dtype=body_lin_acc_w.dtype, device=body_lin_acc_w.device
    )
    specific_force_b = math_utils.quat_apply_inverse(body_quat_w, body_lin_acc_w - gravity_acc_w)
    return torch.sum(torch.square(specific_force_b[:, :2]), dim=1)


def arm_style_prior_exp(
    env: ManagerBasedRLEnv,
    checkpoint_path: str,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward arm joint positions that match a supervised demo arm-style prior.

    The prior predicts arm joint positions from root body velocity plus non-arm
    joint positions/velocities. The returned exponential reward remains a normal
    differentiable tensor with respect to current arm joint positions.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    model, metadata = _load_arm_style_prior(checkpoint_path, asset.data.joint_pos.device)
    input_joint_names = metadata["input_joint_names"]
    target_joint_names = metadata["target_joint_names"]
    assert isinstance(input_joint_names, list)
    assert isinstance(target_joint_names, list)
    input_joint_ids = _joint_ids_from_names(asset, input_joint_names)
    target_joint_ids = _joint_ids_from_names(asset, target_joint_names)
    features = torch.cat(
        [
            asset.data.root_lin_vel_b,
            asset.data.root_ang_vel_b,
            asset.data.joint_pos[:, input_joint_ids],
            asset.data.joint_vel[:, input_joint_ids],
        ],
        dim=-1,
    )
    input_mean = metadata["input_mean"]
    input_std = metadata["input_std"]
    target_mean = metadata["target_mean"]
    target_std = metadata["target_std"]
    assert isinstance(input_mean, torch.Tensor)
    assert isinstance(input_std, torch.Tensor)
    assert isinstance(target_mean, torch.Tensor)
    assert isinstance(target_std, torch.Tensor)
    predicted_arm_pos = model((features - input_mean) / input_std) * target_std + target_mean
    current_arm_pos = asset.data.joint_pos[:, target_joint_ids]
    error = torch.mean(torch.square(current_arm_pos - predicted_arm_pos), dim=-1)
    return torch.exp(-error / (std * std))


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def root_xy_position_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize horizontal root displacement from the center of each environment."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_xy_from_origin = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.sum(torch.square(root_xy_from_origin), dim=1)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_rate_l2_selected(env: ManagerBasedRLEnv, action_indices: list[int]) -> torch.Tensor:
    """Penalize the rate of change of selected raw action dimensions."""
    action_delta = env.action_manager.action[:, action_indices] - env.action_manager.prev_action[:, action_indices]
    return torch.sum(torch.square(action_delta), dim=1)


def action_l2_selected(env: ManagerBasedRLEnv, action_indices: list[int]) -> torch.Tensor:
    """Penalize selected raw action dimensions using an L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, action_indices]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def feet_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_planar_separation_l2(
    env: ManagerBasedRLEnv,
    target_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared error from a target planar foot-center separation.

    The XY norm is invariant to robot yaw and keeps the Stand posture target
    explicit.  It does not reward widening without bound and does not constrain
    either foot to a fixed world coordinate.
    """
    if target_distance <= 0.0:
        raise ValueError(f"target_distance must be positive, got {target_distance}")
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_planar_separation_l2 expects exactly two foot bodies.")
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    distance = torch.linalg.vector_norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=1)
    return torch.square(distance - target_distance)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 只对超过 threshold 的空中时间给予正奖励（防止负值惩罚）
    positive_air = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(positive_air * first_contact.float(), dim=1)
    # no reward for zero command
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def double_support(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward simultaneous contact of both selected feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("double_support expects exactly two foot bodies.")
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return torch.all(in_contact, dim=1).float()


def _strict_zero_command_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    epsilon: float,
) -> torch.Tensor:
    """Select only an actual three-axis zero command, without a low-speed deadband."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command, dim=1) <= float(epsilon)


def strict_zero_command_body_motion_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    epsilon: float = 1.0e-6,
    lin_scale: float = 0.08,
    yaw_scale: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar drift and yaw motion only when all command components are zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_motion = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    yaw_motion = torch.square(asset.data.root_ang_vel_b[:, 2])
    normalized = lin_motion / max(float(lin_scale) ** 2, 1.0e-8)
    normalized += yaw_motion / max(float(yaw_scale) ** 2, 1.0e-8)
    return normalized * _strict_zero_command_mask(env, command_name, epsilon).float()


def strict_zero_command_feet_motion_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    epsilon: float = 1.0e-6,
    velocity_scale: float = 0.12,
) -> torch.Tensor:
    """Penalize either foot moving after the policy receives an exact stop command."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    motion = torch.sum(torch.square(foot_velocity), dim=(1, 2))
    motion /= max(float(velocity_scale) ** 2, 1.0e-8)
    return motion * _strict_zero_command_mask(env, command_name, epsilon).float()


def strict_zero_command_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    epsilon: float = 1.0e-6,
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Suppress lower-body limit cycles at exact zero without constraining nonzero micro-speeds."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_motion = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    joint_motion /= max(float(velocity_scale) ** 2, 1.0e-8)
    return joint_motion * _strict_zero_command_mask(env, command_name, epsilon).float()


def strict_zero_command_double_support(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Reward both feet supporting the robot for an exact stop command."""
    return double_support(env, sensor_cfg) * _strict_zero_command_mask(
        env, command_name, epsilon
    ).float()


def nonzero_command_single_stance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    epsilon: float = 1.0e-6,
    min_mode_time: float = 0.02,
    micro_linear_speed_max: float = 0.15,
    pure_yaw_translation_max: float = 0.005,
    pure_yaw_min_command: float = 0.05,
    micro_speed_bonus: float = 0.50,
    pure_yaw_bonus: float = 0.75,
) -> torch.Tensor:
    """Reward real stepping, with stronger micro-speed and pure-yaw incentives."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    mode_time = torch.where(
        in_contact,
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids],
        contact_sensor.data.current_air_time[:, sensor_cfg.body_ids],
    )
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    stable_mode = torch.min(mode_time, dim=1).values >= float(min_mode_time)
    command = env.command_manager.get_command(command_name)
    command_scale = nonzero_single_stance_command_scale(
        command,
        epsilon=epsilon,
        micro_linear_speed_max=micro_linear_speed_max,
        pure_yaw_translation_max=pure_yaw_translation_max,
        pure_yaw_min_command=pure_yaw_min_command,
        micro_speed_bonus=micro_speed_bonus,
        pure_yaw_bonus=pure_yaw_bonus,
    )
    return single_stance.float() * stable_mode.float() * command_scale * _upright_scale(env)


def command_response_shortfall_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    epsilon: float = 1.0e-6,
    min_speed_fraction: float = 0.50,
    lin_scale: float = 0.12,
    yaw_scale: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lean-without-motion solutions for translation and pure in-place turning."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_speed.unsqueeze(1), min=1.0e-8)
    projected_speed = torch.sum(asset.data.root_lin_vel_b[:, :2] * command_direction, dim=1)
    lin_shortfall = torch.clamp(
        float(min_speed_fraction) * command_speed - projected_speed, min=0.0
    )
    lin_shortfall *= (command_speed > float(epsilon)).float()

    desired_yaw = torch.abs(command[:, 2]) * float(min_speed_fraction)
    signed_yaw_response = torch.sign(command[:, 2]) * asset.data.root_ang_vel_b[:, 2]
    yaw_shortfall = torch.clamp(desired_yaw - signed_yaw_response, min=0.0)
    yaw_shortfall *= (torch.abs(command[:, 2]) > float(epsilon)).float()
    return (
        lin_shortfall / max(float(lin_scale), 1.0e-6)
        + yaw_shortfall / max(float(yaw_scale), 1.0e-6)
    ) * _upright_scale(env)


def relative_command_response_shortfall_reward_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    epsilon: float = 1.0e-6,
    min_speed_fraction: float = 0.50,
    min_lin_normalizer: float = 0.01,
    min_yaw_normalizer: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize fractional response shortfall without upright attenuation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return relative_command_response_shortfall_l1(
        command=command,
        actual_lin_vel_xy_b=asset.data.root_lin_vel_b[:, :2],
        actual_yaw_rate_b=asset.data.root_ang_vel_b[:, 2],
        epsilon=epsilon,
        min_speed_fraction=min_speed_fraction,
        min_lin_normalizer=min_lin_normalizer,
        min_yaw_normalizer=min_yaw_normalizer,
    )


def pure_yaw_torso_roll_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_translation_command: float = 0.005,
    min_yaw_command: float = 0.05,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Penalize torso side-lean only for an in-place yaw command."""
    if std <= 0.0:
        raise ValueError("pure_yaw_torso_roll_l2 requires std > 0.")
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = _single_body_id(asset_cfg)
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.body_quat_w[:, body_id, :])
    command = env.command_manager.get_command(command_name)
    pure_yaw = (
        torch.linalg.vector_norm(command[:, :2], dim=1) <= float(max_translation_command)
    ) & (torch.abs(command[:, 2]) >= float(min_yaw_command))
    return (
        torch.square(math_utils.wrap_to_pi(roll)) / float(std) ** 2
    ) * pure_yaw.float()


class command_conditioned_footstep_cadence_l1(ManagerTermBase):
    """Penalize command-relative excess cadence using an EMA of touchdown intervals."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._elapsed_since_touchdown = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._cadence_ema_hz = torch.zeros_like(self._elapsed_since_touchdown)
        self._has_touchdown = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._has_interval = torch.zeros_like(self._has_touchdown)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._elapsed_since_touchdown[env_ids] = 0.0
        self._cadence_ema_hz[env_ids] = 0.0
        self._has_touchdown[env_ids] = False
        self._has_interval[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        epsilon: float = 1.0e-6,
        ema_alpha: float = 0.5,
        base_hz: float = 1.0,
        linear_gain: float = 3.33,
        yaw_gain: float = 1.5,
        maximum_hz: float = 3.0,
        max_penalty: float = 3.0,
    ) -> torch.Tensor:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if env.step_dt <= 0.0:
            raise ValueError("Cadence reward requires env.step_dt > 0.")
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        touchdown = torch.any(
            contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids],
            dim=1,
        )
        (
            self._elapsed_since_touchdown,
            self._cadence_ema_hz,
            self._has_touchdown,
            self._has_interval,
        ) = update_footstep_cadence_state(
            self._elapsed_since_touchdown,
            self._cadence_ema_hz,
            self._has_touchdown,
            self._has_interval,
            touchdown,
            step_dt=float(env.step_dt),
            ema_alpha=ema_alpha,
        )

        command = env.command_manager.get_command(command_name)
        allowed_hz = allowed_footstep_cadence_hz(
            command,
            base_hz=base_hz,
            linear_gain=linear_gain,
            yaw_gain=yaw_gain,
            maximum_hz=maximum_hz,
        )
        active = torch.linalg.vector_norm(command, dim=1) > float(epsilon)
        penalty = torch.clamp(
            (self._cadence_ema_hz - allowed_hz) / allowed_hz,
            min=0.0,
            max=float(max_penalty),
        )
        return penalty * active.float() * self._has_interval.float()


def rapid_footstep_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    epsilon: float = 1.0e-6,
    min_air_time: float = 0.16,
) -> torch.Tensor:
    """Penalize touchdown after an implausibly short swing to reduce excessive cadence."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    shortfall = torch.clamp(float(min_air_time) - last_air_time, min=0.0)
    command = env.command_manager.get_command(command_name)
    active = torch.linalg.vector_norm(command, dim=1) > float(epsilon)
    return (
        torch.sum(shortfall * first_contact.float(), dim=1)
        / max(float(min_air_time), 1.0e-6)
        * active.float()
    )


def oriented_footprint_proximity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    center_offset_x: float = 0.035,
    half_length: float = 0.090,
    half_width: float = 0.035,
    min_clearance: float = 0.025,
    std: float = 0.020,
    max_penalty: float = 25.0,
) -> torch.Tensor:
    """Penalize close/overlapping oriented sole rectangles using a 2-D SAT clearance.

    S3 G1 contact points use the conservative envelope x=[-0.055, 0.125] and
    y=[-0.035, 0.035] around each ankle-roll link. The four projected corners
    therefore reflect sole length, width and yaw instead of using ankle-center
    distance, which misses toe/heel and crossing-foot intersections.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    corners_xy = _oriented_footprint_corners_xy(
        asset,
        asset_cfg,
        center_offset_x=center_offset_x,
        half_length=half_length,
        half_width=half_width,
    )

    # The torch-only helper evaluates SAT axis_separation on every edge normal.
    signed_clearance = convex_footprint_signed_clearance_xy(
        corners_xy[:, 0], corners_xy[:, 1]
    )
    violation = torch.clamp(float(min_clearance) - signed_clearance, min=0.0)
    return torch.clamp(
        torch.square(violation / max(float(std), 1.0e-6)), max=float(max_penalty)
    )


def _oriented_footprint_corners_xy(
    asset: RigidObject,
    asset_cfg: SceneEntityCfg,
    *,
    center_offset_x: float,
    half_length: float,
    half_width: float,
) -> torch.Tensor:
    """Transform two conservative sole rectangles into world-frame XY corners."""
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("Oriented footprint rewards expect exactly two foot bodies.")

    local_corners = torch.tensor(
        [
            [center_offset_x - half_length, -half_width, 0.0],
            [center_offset_x + half_length, -half_width, 0.0],
            [center_offset_x + half_length, half_width, 0.0],
            [center_offset_x - half_length, half_width, 0.0],
        ],
        dtype=asset.data.body_pos_w.dtype,
        device=asset.data.body_pos_w.device,
    )
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_envs = foot_pos.shape[0]
    quaternions = foot_quat.unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4)
    corner_vectors = local_corners.view(1, 1, 4, 3).expand(num_envs, 2, -1, -1)
    world_corners = math_utils.quat_apply(quaternions, corner_vectors.reshape(-1, 3))
    world_corners = world_corners.view(num_envs, 2, 4, 3) + foot_pos.unsqueeze(2)
    return world_corners[..., :2]


class _safe_alternating_touchdown_pose_progress(ManagerTermBase):
    """Track root-pose changes between consecutive safe alternating touchdowns."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._last_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._last_mode = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)
        self._has_anchor = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._anchor_root_xy_w = torch.zeros(env.num_envs, 2, device=env.device)
        self._anchor_heading_w = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_foot[env_ids] = -1
        self._last_mode[env_ids] = 0
        self._has_anchor[env_ids] = False
        self._anchor_root_xy_w[env_ids] = 0.0
        self._anchor_heading_w[env_ids] = 0.0

    def _measure(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        *,
        min_lateral_command: float,
        min_yaw_command: float,
        min_clearance: float,
        center_offset_x: float,
        half_length: float,
        half_width: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        lateral, pure_yaw = two_goal_command_masks(
            command,
            lateral_min_command=min_lateral_command,
            pure_yaw_min_command=min_yaw_command,
        )
        mode = lateral.to(torch.int8) + 2 * pure_yaw.to(torch.int8)
        mode_changed = mode != self._last_mode
        self._last_foot[mode_changed] = -1
        self._has_anchor[mode_changed] = False
        self._last_mode.copy_(mode)

        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        exactly_one_touchdown = torch.sum(first_contact.int(), dim=1) == 1
        touchdown_foot = torch.argmax(first_contact.int(), dim=1)
        alternating = (self._last_foot >= 0) & (touchdown_foot != self._last_foot)
        active_touchdown = exactly_one_touchdown & (mode > 0)

        corners = _oriented_footprint_corners_xy(
            asset,
            asset_cfg,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        clearance = convex_footprint_signed_clearance_xy(corners[:, 0], corners[:, 1])
        safe = clearance >= float(min_clearance)
        event = active_touchdown & alternating & self._has_anchor & safe

        root_xy_w = asset.data.root_pos_w[:, :2]
        heading_w = asset.data.heading_w
        forward_delta, lateral_delta, yaw_delta = touchdown_pose_progress(
            self._anchor_root_xy_w,
            self._anchor_heading_w,
            root_xy_w,
            heading_w,
        )

        # Always advance the anchor on a task touchdown. An unsafe touchdown is
        # therefore never allowed to accumulate delayed progress credit.
        self._last_foot[active_touchdown] = touchdown_foot[active_touchdown]
        self._anchor_root_xy_w[active_touchdown] = root_xy_w[active_touchdown]
        self._anchor_heading_w[active_touchdown] = heading_w[active_touchdown]
        self._has_anchor[active_touchdown] = True
        return command, lateral, pure_yaw, event, forward_delta, lateral_delta, yaw_delta


class lateral_command_progress(_safe_alternating_touchdown_pose_progress):
    """Reward real lateral root displacement between safe alternating touchdowns."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        min_command: float = 0.10,
        target_displacement: float = 0.060,
        forward_leak_scale: float = 0.050,
        yaw_leak_scale: float = 0.100,
        min_clearance: float = 0.025,
        center_offset_x: float = 0.035,
        half_length: float = 0.090,
        half_width: float = 0.035,
    ) -> torch.Tensor:
        if target_displacement <= 0.0 or forward_leak_scale <= 0.0 or yaw_leak_scale <= 0.0:
            raise ValueError("Lateral touchdown progress scales must be positive.")
        command, lateral, _, event, forward_delta, lateral_delta, yaw_delta = self._measure(
            env,
            command_name,
            sensor_cfg,
            asset_cfg,
            min_lateral_command=min_command,
            min_yaw_command=min_command,
            min_clearance=min_clearance,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        signed_progress = torch.sign(command[:, 1]) * lateral_delta / float(target_displacement)
        quality = torch.exp(
            -torch.square(forward_delta / float(forward_leak_scale))
            -torch.square(yaw_delta / float(yaw_leak_scale))
        )
        return (
            torch.clamp(signed_progress, min=-1.0, max=1.25)
            * quality
            * event.float()
            * lateral.float()
            * _upright_scale(env)
        )


class pure_yaw_command_progress(_safe_alternating_touchdown_pose_progress):
    """Reward real wrapped root-heading change between safe alternating touchdowns."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        min_command: float = 0.10,
        target_yaw_delta: float = 0.100,
        planar_drift_scale: float = 0.050,
        min_clearance: float = 0.025,
        center_offset_x: float = 0.035,
        half_length: float = 0.090,
        half_width: float = 0.035,
    ) -> torch.Tensor:
        if target_yaw_delta <= 0.0 or planar_drift_scale <= 0.0:
            raise ValueError("Pure-yaw touchdown progress scales must be positive.")
        command, _, pure_yaw, event, forward_delta, lateral_delta, yaw_delta = self._measure(
            env,
            command_name,
            sensor_cfg,
            asset_cfg,
            min_lateral_command=min_command,
            min_yaw_command=min_command,
            min_clearance=min_clearance,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        signed_progress = torch.sign(command[:, 2]) * yaw_delta / float(target_yaw_delta)
        planar_drift = torch.sqrt(torch.square(forward_delta) + torch.square(lateral_delta))
        quality = torch.exp(-torch.square(planar_drift / float(planar_drift_scale)))
        return (
            torch.clamp(signed_progress, min=-1.0, max=1.25)
            * quality
            * event.float()
            * pure_yaw.float()
            * _upright_scale(env)
        )


class dense_root_pose_command_progress(ManagerTermBase):
    """Reward command-directed root-pose progress on every control step.

    This is a dense learning signal for policies that do not yet produce a
    touchdown. It uses finite differences of the root pose rather than torso
    velocity, so a lateral sway or yaw oscillation earns positive credit on one
    half-cycle and an equal negative credit on the other. Safe alternating
    touchdown terms remain the authoritative sparse completion signal.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._previous_root_xy_w = torch.zeros(env.num_envs, 2, device=env.device)
        self._previous_heading_w = torch.zeros(env.num_envs, device=env.device)
        self._last_mode = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_root_xy_w[env_ids] = 0.0
        self._previous_heading_w[env_ids] = 0.0
        self._last_mode[env_ids] = 0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        min_lateral_command: float = 0.10,
        min_yaw_command: float = 0.10,
        forward_leak_scale: float = 0.20,
        yaw_leak_scale: float = 0.30,
        planar_drift_scale: float = 0.15,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if min_lateral_command <= 0.0 or min_yaw_command <= 0.0:
            raise ValueError("Dense root-pose progress command thresholds must be positive.")
        if forward_leak_scale <= 0.0 or yaw_leak_scale <= 0.0 or planar_drift_scale <= 0.0:
            raise ValueError("Dense root-pose progress quality scales must be positive.")

        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        lateral, pure_yaw = two_goal_command_masks(
            command,
            lateral_min_command=min_lateral_command,
            pure_yaw_min_command=min_yaw_command,
        )
        mode = lateral.to(torch.int8) + 2 * pure_yaw.to(torch.int8)
        mode_changed = mode != self._last_mode
        active = mode > 0
        valid = self._initialized & ~mode_changed & active

        root_xy_w = asset.data.root_pos_w[:, :2]
        heading_w = asset.data.heading_w
        forward_delta, lateral_delta, yaw_delta = touchdown_pose_progress(
            self._previous_root_xy_w,
            self._previous_heading_w,
            root_xy_w,
            heading_w,
        )
        step_dt = float(env.step_dt)
        if step_dt <= 0.0:
            raise ValueError("Environment step_dt must be positive.")
        progress = two_goal_dense_root_pose_progress(
            command,
            forward_delta,
            lateral_delta,
            yaw_delta,
            step_dt=step_dt,
            lateral_min_command=min_lateral_command,
            pure_yaw_min_command=min_yaw_command,
            forward_leak_scale=forward_leak_scale,
            yaw_leak_scale=yaw_leak_scale,
            planar_drift_scale=planar_drift_scale,
        )

        self._previous_root_xy_w.copy_(root_xy_w)
        self._previous_heading_w.copy_(heading_w)
        self._last_mode.copy_(mode)
        self._initialized.fill_(True)
        return progress * valid.float() * _upright_scale(env)


def two_goal_response_shortfall(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_fraction: float = 0.50,
    max_penalty: float = 1.0,
    min_lateral_command: float = 0.10,
    min_yaw_command: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize bounded lateral/yaw response shortfall without an upright gate."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return two_goal_response_shortfall_l2(
        command,
        asset.data.root_lin_vel_b[:, :2],
        asset.data.root_ang_vel_b[:, 2],
        target_fraction=target_fraction,
        max_penalty=max_penalty,
        lateral_min_command=min_lateral_command,
        pure_yaw_min_command=min_yaw_command,
    )


def two_goal_signed_root_response(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lateral_command: float = 0.10,
    min_yaw_command: float = 0.10,
    minimum_ratio: float = -1.0,
    maximum_ratio: float = 1.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the signed root response without mixing in quality constraints.

    Lateral velocity and root yaw rate are the two task objectives.  Forward
    leakage, planar drift and sole clearance are independent constraints and
    are deliberately scored by separate terms.  Keeping objective and
    constraints separate avoids the vanishing learning signal produced by an
    exponential quality gate when the initial policy is far from feasible.
    """
    if minimum_ratio >= maximum_ratio:
        raise ValueError("minimum_ratio must be smaller than maximum_ratio.")
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lateral, pure_yaw = two_goal_command_masks(
        command,
        lateral_min_command=min_lateral_command,
        pure_yaw_min_command=min_yaw_command,
    )
    lateral_ratio = (
        torch.sign(command[:, 1]) * asset.data.root_lin_vel_b[:, 1]
        / torch.clamp(torch.abs(command[:, 1]), min=float(min_lateral_command))
    )
    yaw_ratio = (
        torch.sign(command[:, 2]) * asset.data.root_ang_vel_b[:, 2]
        / torch.clamp(torch.abs(command[:, 2]), min=float(min_yaw_command))
    )
    response = lateral_ratio * lateral.float() + yaw_ratio * pure_yaw.float()
    return torch.clamp(response, min=float(minimum_ratio), max=float(maximum_ratio))


def pure_yaw_planar_drift_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_yaw_command: float = 0.10,
    velocity_scale: float = 0.08,
    max_penalty: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize translational drift only during an in-place turn command."""
    if velocity_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("velocity_scale and max_penalty must be positive.")
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, pure_yaw = two_goal_command_masks(command, pure_yaw_min_command=min_yaw_command)
    drift = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.clamp(drift / float(velocity_scale) ** 2, max=float(max_penalty)) * pure_yaw.float()


def lateral_command_leak_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lateral_command: float = 0.10,
    forward_velocity_scale: float = 0.10,
    yaw_rate_scale: float = 0.15,
    max_penalty: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize forward and yaw leakage during a pure lateral command."""
    if forward_velocity_scale <= 0.0 or yaw_rate_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("Lateral leakage scales and max_penalty must be positive.")
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lateral, _ = two_goal_command_masks(command, lateral_min_command=min_lateral_command)
    leakage = torch.square(asset.data.root_lin_vel_b[:, 0]) / float(forward_velocity_scale) ** 2
    leakage += torch.square(asset.data.root_ang_vel_b[:, 2]) / float(yaw_rate_scale) ** 2
    return torch.clamp(leakage, max=float(max_penalty)) * lateral.float()


class safe_alternating_touchdown_progress(_safe_alternating_touchdown_pose_progress):
    """Reward safe alternating touchdowns only when they produce real progress."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        min_lateral_command: float = 0.10,
        min_yaw_command: float = 0.10,
        lateral_progress_per_step: float = 0.060,
        yaw_progress_per_step: float = 0.100,
        minimum_productive_fraction: float = 0.20,
        min_clearance: float = 0.025,
        center_offset_x: float = 0.035,
        half_length: float = 0.090,
        half_width: float = 0.035,
    ) -> torch.Tensor:
        if (
            lateral_progress_per_step <= 0.0
            or yaw_progress_per_step <= 0.0
            or minimum_productive_fraction < 0.0
        ):
            raise ValueError("Touchdown progress targets must be positive.")
        command, lateral, pure_yaw, event, _, lateral_delta, yaw_delta = self._measure(
            env,
            command_name,
            sensor_cfg,
            asset_cfg,
            min_lateral_command=min_lateral_command,
            min_yaw_command=min_yaw_command,
            min_clearance=min_clearance,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        signed_lateral_progress = torch.sign(command[:, 1]) * lateral_delta
        signed_yaw_progress = torch.sign(command[:, 2]) * yaw_delta
        normalized_progress = torch.where(
            lateral,
            signed_lateral_progress / float(lateral_progress_per_step),
            signed_yaw_progress / float(yaw_progress_per_step),
        )
        productive = normalized_progress >= float(minimum_productive_fraction)
        return (
            torch.clamp(normalized_progress, min=0.0, max=1.0)
            * productive.float()
            * event.float()
            * (lateral | pure_yaw).float()
            * _upright_scale(env)
        )


class safe_step_initiation(ManagerTermBase):
    """Give one small reward when a safe specialization step starts."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._swing_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._swing_active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        min_air_time: float = 0.040,
        min_clearance: float = 0.025,
        center_offset_x: float = 0.035,
        half_length: float = 0.090,
        half_width: float = 0.035,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        lateral, pure_yaw = two_goal_command_masks(command)
        active = lateral | pure_yaw
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        exactly_one_swing = torch.sum((air_time >= float(min_air_time)).int(), dim=1) == 1
        started = exactly_one_swing & ~self._swing_active & active
        corners = _oriented_footprint_corners_xy(
            asset,
            asset_cfg,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        clearance = convex_footprint_signed_clearance_xy(corners[:, 0], corners[:, 1])
        safe = clearance >= float(min_clearance)
        self._swing_active.copy_(exactly_one_swing & active)
        return started.float() * safe.float() * _upright_scale(env)


class safe_alternating_swing_progress(ManagerTermBase):
    """Reward a bounded single-foot swing only when swing feet alternate.

    Holding one foot up stops earning credit after ``max_air_time``. Repeating
    the same foot earns no credit until the opposite foot starts a safe swing,
    which removes the stationary single-support exploit of a plain air-time
    reward while providing a dense initiation signal before touchdown.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._last_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._last_mode = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)
        self._swing_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._current_swing_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_foot[env_ids] = -1
        self._last_mode[env_ids] = 0
        self._swing_active[env_ids] = False
        self._current_swing_valid[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        min_lateral_command: float = 0.10,
        min_yaw_command: float = 0.10,
        min_air_time: float = 0.040,
        max_air_time: float = 0.35,
        min_clearance: float = 0.025,
        center_offset_x: float = 0.035,
        half_length: float = 0.090,
        half_width: float = 0.035,
    ) -> torch.Tensor:
        if min_air_time <= 0.0 or max_air_time <= min_air_time:
            raise ValueError("Safe swing air-time bounds are invalid.")
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        lateral, pure_yaw = two_goal_command_masks(
            command,
            lateral_min_command=min_lateral_command,
            pure_yaw_min_command=min_yaw_command,
        )
        mode = lateral.to(torch.int8) + 2 * pure_yaw.to(torch.int8)
        mode_changed = mode != self._last_mode
        self._last_foot[mode_changed] = -1
        self._swing_active[mode_changed] = False
        self._current_swing_valid[mode_changed] = False
        self._last_mode.copy_(mode)

        active = mode > 0
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        swing_mask = air_time >= float(min_air_time)
        exactly_one_swing = torch.sum(swing_mask.int(), dim=1) == 1
        swing_foot = torch.argmax(swing_mask.int(), dim=1)
        started = exactly_one_swing & ~self._swing_active & active
        alternating = (self._last_foot < 0) | (swing_foot != self._last_foot)
        accepted = started & alternating
        self._current_swing_valid[started] = accepted[started]
        self._last_foot[accepted] = swing_foot[accepted]
        self._current_swing_valid[~exactly_one_swing] = False
        self._swing_active.copy_(exactly_one_swing & active)

        active_air_time = torch.max(air_time, dim=1).values
        bounded_duration = active_air_time <= float(max_air_time)
        corners = _oriented_footprint_corners_xy(
            asset,
            asset_cfg,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        clearance = convex_footprint_signed_clearance_xy(corners[:, 0], corners[:, 1])
        safe = clearance >= float(min_clearance)
        return (
            exactly_one_swing.float()
            * self._current_swing_valid.float()
            * bounded_duration.float()
            * safe.float()
            * active.float()
            * _upright_scale(env)
        )


class swept_oriented_footprint_proximity_l2(ManagerTermBase):
    """Shape-aware soft barrier with hard close/overlap costs over a swept step."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._previous_corners = torch.zeros(
            env.num_envs, 2, 4, 2, dtype=torch.float32, device=env.device
        )
        self._has_previous = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_corners[env_ids] = 0.0
        self._has_previous[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        center_offset_x: float = 0.035,
        half_length: float = 0.090,
        half_width: float = 0.035,
        soft_clearance: float = 0.040,
        hard_clearance: float = 0.025,
        hard_scale: float = 3.0,
        overlap_scale: float = 8.0,
        component: str = "combined",
        interpolation_steps: int = 4,
    ) -> torch.Tensor:
        if not 0.0 <= hard_clearance < soft_clearance:
            raise ValueError("Footprint clearances require 0 <= hard < soft.")
        if component not in {"soft", "hard", "combined"}:
            raise ValueError("Footprint barrier component must be soft, hard, or combined.")
        asset: RigidObject = env.scene[asset_cfg.name]
        current = _oriented_footprint_corners_xy(
            asset,
            asset_cfg,
            center_offset_x=center_offset_x,
            half_length=half_length,
            half_width=half_width,
        )
        previous = torch.where(
            self._has_previous.view(-1, 1, 1, 1), self._previous_corners, current
        )
        clearance = swept_convex_footprint_signed_clearance_xy(
            previous[:, 0],
            previous[:, 1],
            current[:, 0],
            current[:, 1],
            interpolation_steps=interpolation_steps,
        )
        self._previous_corners.copy_(current)
        self._has_previous.fill_(True)

        command = env.command_manager.get_command(command_name)
        lateral, pure_yaw = two_goal_command_masks(command)
        active = lateral | pure_yaw
        soft = torch.clamp(
            (float(soft_clearance) - clearance)
            / (float(soft_clearance) - float(hard_clearance)),
            min=0.0,
            max=1.0,
        )
        soft = torch.square(soft)
        hard = (clearance < float(hard_clearance)).float() * float(hard_scale)
        overlap = (clearance < 0.0).float() * float(overlap_scale)
        if component == "soft":
            penalty = soft
        elif component == "hard":
            penalty = hard + overlap
        else:
            penalty = soft + hard + overlap
        return penalty * active.float()


def feet_swing_clearance_band_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.055,
    std: float = 0.03,
    air_time_threshold: float = 0.06,
    command_threshold: float = 0.1,
    max_height: float = 0.13,
    high_std: float = 0.04,
    high_weight: float = 0.25,
) -> torch.Tensor:
    """Penalize low swing-foot clearance, with a light guard against over-lifting."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    lowest_foot_z = torch.min(foot_z, dim=1, keepdim=True).values
    clearance = foot_z - lowest_foot_z
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    swing_mask = air_time > air_time_threshold

    low_error = torch.clamp(target_height - clearance, min=0.0)
    low_penalty = torch.square(low_error / max(float(std), 1.0e-6))
    high_error = torch.clamp(clearance - max_height, min=0.0)
    high_penalty = float(high_weight) * torch.square(high_error / max(float(high_std), 1.0e-6))
    per_foot_penalty = (low_penalty + high_penalty) * swing_mask.float()
    penalty = torch.sum(per_foot_penalty, dim=1) / torch.clamp(torch.sum(swing_mask.float(), dim=1), min=1.0)

    command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty * command_active.float()


def gait_timing_symmetry_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    deadband: float = 0.03,
    std: float = 0.08,
    min_air_time: float = 0.06,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize left-right mismatch in the most recent completed air durations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("gait_timing_symmetry_l1 expects exactly two foot bodies.")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    timing_error = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
    violation = torch.clamp(timing_error - deadband, min=0.0)
    both_feet_have_history = torch.all(last_air_time > min_air_time, dim=1)
    command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return (violation / max(float(std), 1.0e-6)) * both_feet_have_history.float() * command_active.float()


def _directional_command_gate(
    command: torch.Tensor,
    backward_threshold: float = 0.15,
    lateral_threshold: float = 0.15,
    include_backward: bool = True,
    include_lateral: bool = True,
) -> torch.Tensor:
    backward = command[:, 0] < -abs(float(backward_threshold))
    lateral = torch.abs(command[:, 1]) > abs(float(lateral_threshold))
    gate = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
    if include_backward:
        gate = gate | backward
    if include_lateral:
        gate = gate | lateral
    return gate.float()


def directional_speed_floor_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_speed_fraction: float = 0.85,
    std: float = 0.20,
    command_threshold: float = 0.15,
    backward_threshold: float = 0.15,
    lateral_threshold: float = 0.15,
    include_backward: bool = True,
    include_lateral: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize directional commands that are satisfied with too little projected speed."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.norm(command_xy, dim=1)
    command_dir = command_xy / torch.clamp(command_speed.unsqueeze(-1), min=1.0e-6)
    projected_speed = torch.sum(asset.data.root_lin_vel_b[:, :2] * command_dir, dim=1)
    target_floor = float(min_speed_fraction) * command_speed
    violation = torch.clamp(target_floor - projected_speed, min=0.0)
    active = (command_speed > float(command_threshold)).float()
    gate = _directional_command_gate(
        command,
        backward_threshold=backward_threshold,
        lateral_threshold=lateral_threshold,
        include_backward=include_backward,
        include_lateral=include_lateral,
    )
    return (violation / max(float(std), 1.0e-6)) * active * gate * _upright_scale(env)


def directional_double_air_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.15,
    backward_threshold: float = 0.15,
    lateral_threshold: float = 0.15,
    include_backward: bool = True,
    include_lateral: bool = True,
) -> torch.Tensor:
    """Penalize both feet being airborne for backward/lateral locomotion commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    double_air = torch.sum(in_contact.int(), dim=1) == 0
    command_active = (torch.norm(command[:, :2], dim=1) > float(command_threshold)).float()
    gate = _directional_command_gate(
        command,
        backward_threshold=backward_threshold,
        lateral_threshold=lateral_threshold,
        include_backward=include_backward,
        include_lateral=include_lateral,
    )
    return double_air.float() * command_active * gate


def backward_single_stance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    backward_threshold: float = 0.15,
    min_mode_time: float = 0.02,
) -> torch.Tensor:
    """Reward alternating one-foot support during backward walking commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    mode_time = torch.where(in_contact, contact_time, air_time)
    stable_mode = torch.min(mode_time, dim=1).values > float(min_mode_time)
    backward = command[:, 0] < -abs(float(backward_threshold))
    return single_stance.float() * stable_mode.float() * backward.float() * _upright_scale(env)


def backward_double_stance_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    backward_threshold: float = 0.15,
) -> torch.Tensor:
    """Lightly discourage the backward policy from solving the task as a shuffle."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    double_stance = torch.sum(in_contact.int(), dim=1) == 2
    backward = command[:, 0] < -abs(float(backward_threshold))
    return double_stance.float() * backward.float()


def directional_double_stance_support(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    backward_threshold: float = 0.15,
    lateral_threshold: float = 0.15,
    backward_scale: float = 0.55,
    lateral_scale: float = 1.0,
    min_contact_time: float = 0.02,
) -> torch.Tensor:
    """Reward stable double-stance support for modes whose demos are not pure alternation.

    V2 fixed backward hopping by rewarding single stance directly, but the policy
    over-corrected into almost continuous single support for backward/lateral
    commands. This term softly gives backward and especially lateral commands a
    reason to keep human side-step style double support without reintroducing
    double-air hopping.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    stable_double_stance = torch.all(contact_time > float(min_contact_time), dim=1)

    backward = command[:, 0] < -abs(float(backward_threshold))
    lateral = torch.abs(command[:, 1]) > abs(float(lateral_threshold))
    scale = torch.zeros(command.shape[0], dtype=command.dtype, device=command.device)
    scale = torch.where(backward, torch.full_like(scale, float(backward_scale)), scale)
    scale = torch.where(lateral, torch.full_like(scale, float(lateral_scale)), scale)
    return stable_double_stance.float() * (torch.sum(in_contact.int(), dim=1) == 2).float() * scale * _upright_scale(env)


def directional_velocity_leak_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    lateral_threshold: float = 0.15,
    backward_threshold: float = 0.15,
    lateral_x_deadband: float = 0.04,
    lateral_yaw_deadband: float = 0.04,
    sagittal_y_deadband: float = 0.04,
    sagittal_yaw_deadband: float = 0.04,
    lin_std: float = 0.20,
    yaw_std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize uncommanded velocity leakage in directional locomotion modes."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_lin_vel_b = asset.data.root_lin_vel_b
    root_yaw_rate = asset.data.root_ang_vel_b[:, 2]

    lateral = torch.abs(command[:, 1]) > abs(float(lateral_threshold))
    backward = command[:, 0] < -abs(float(backward_threshold))

    lateral_x = torch.clamp(torch.abs(root_lin_vel_b[:, 0] - command[:, 0]) - float(lateral_x_deadband), min=0.0)
    lateral_yaw = torch.clamp(torch.abs(root_yaw_rate - command[:, 2]) - float(lateral_yaw_deadband), min=0.0)
    sagittal_y = torch.clamp(torch.abs(root_lin_vel_b[:, 1] - command[:, 1]) - float(sagittal_y_deadband), min=0.0)
    sagittal_yaw = torch.clamp(torch.abs(root_yaw_rate - command[:, 2]) - float(sagittal_yaw_deadband), min=0.0)

    lateral_penalty = lateral_x / max(float(lin_std), 1.0e-6) + lateral_yaw / max(float(yaw_std), 1.0e-6)
    backward_penalty = sagittal_y / max(float(lin_std), 1.0e-6) + sagittal_yaw / max(float(yaw_std), 1.0e-6)
    return (lateral_penalty * lateral.float() + backward_penalty * backward.float()) * _upright_scale(env)


def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    return torch.sum(diff, dim=1)


def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    每只脚单独计算：
    脚接触地面时，z 方向加速度大 → 惩罚
    """

    asset = env.scene["robot"]

    # 1️⃣ 取所有 body 的线加速度 (world)
    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # 2️⃣ 取“脚”的 z 方向线加速度
    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    # 3️⃣ 取脚的接触状态
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    # 4️⃣ 每只脚：加速度平方 × 接触状态
    acc_penalty = (foot_acc_z ** 2) * in_contact.float()

    # 防止数值爆炸（非常重要）
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    # 5️⃣ 所有脚加起来
    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    # 仅当速度命令较小（小于 1.5）时才启用该奖励
    cmd = env.command_manager.get_command(command_name)
    
    # 使用 xy 分量的速度范数作为速度大小判断
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def low_speed_sway_penalty(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear and angular velocities when command velocity is below threshold.
    
    This function penalizes the robot for moving (both linear and angular) when the command
    speed is very small, encouraging the robot to remain still during low-speed commands.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get command velocity
    command = env.command_manager.get_command(command_name)
    command_speed = torch.norm(command[:, :2], dim=1)
    
    # Penalize linear velocity in xy plane
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    
    # Penalize angular velocity
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    
    # Total velocity penalty
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    
    # Apply penalty only when command speed is below threshold
    return vel_penalty * (command_speed < command_threshold).float()


def staged_navigation_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    heading_threshold: float = 0.78,  # 45°，朝向误差阈值
    distance_threshold: float = 0.5,   #  距离目标的阈值
    near_goal_threshold: float = 2.0,  # 接近目标的距离阈值
    obstacle_threshold: float = 0.8,  # 前方障碍物的距离阈值
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ray_caster: RayCaster = env.scene[sensor_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :2] # 目标位置
    des_heading = command[:, 2] # 目标朝向
    distance = torch.norm(des_pos, dim=1) # 机器人到目标位置的距离
    
    vx = asset.data.root_lin_vel_b[:, 0] # 机器人在base frame下的前向速度
    vy = asset.data.root_lin_vel_b[:, 1] # 机器人在base frame下的侧向速度
    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) # 机器人在水平面的速度大小 
    ang_speed = torch.abs(asset.data.root_ang_vel_b[:, 2]) # 机器人绕垂直轴的角速度

    # 当前移动方向与期望朝向误差（规范化到 [-pi, pi]）
    move_dir_angle = torch.atan2(vy, vx)
    raw_diff = move_dir_angle - des_heading
    diff_wrapped = torch.remainder(raw_diff + torch.pi, 2 * torch.pi) - torch.pi
    move_heading_error = diff_wrapped.abs()
    
    # 雷达前方最近障碍物距离（已在外部被 clamp）
    origin = ray_caster.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    hits = ray_caster.data.ray_hits_w  # [num_envs, num_rays, 3]
    distances = torch.norm(hits - origin, dim=-1).clamp(min=0.2, max=5.0)  # [num_envs, num_rays]
    front_min_dist = torch.min(distances, dim=1).values  # [num_envs]
    
    # 1) 朝向匹配奖励：误差越小奖励越高
    heading_reward = 1.0 / (1.0 + (move_heading_error / (heading_threshold + 1e-6))**2)

    # 2) 沿期望朝向的速度（越朝向目标前进越好），只奖励正向分量
    proj_vel = vx * torch.cos(des_heading) + vy * torch.sin(des_heading)
    progress_reward = torch.tanh(2.0 * proj_vel.clamp(min=0.0, max=1.0))  # 正向速度越大奖励越高，最大值接近1.0

    # 3) 障碍物清除奖励：鼓励与障碍物保持距离
    # 当 front_min_dist < obstacle_threshold 时，距离越大奖励越高（在 safe_min..obstacle_threshold 区间归一化到 0..1）
    safe_min = 0.5
    denom = max(obstacle_threshold - safe_min, 1e-6)
    obs_clearance = torch.clamp(front_min_dist - safe_min, min=0.0, max=obstacle_threshold - safe_min) / denom  # 0..1 when front_min_dist within [safe_min, obstacle_threshold]
    # 保持变量名兼容下游使用
    obs_approach_raw = obs_clearance
    
    # 对于 front_min_dist < safe_min 给出负向惩罚（碰撞/过近）
    collision_penalty = torch.clamp(safe_min - front_min_dist, min=0.0) / safe_min  # 0..1

    # 距离目标的奖励：距离越小越好，使用 near_goal_threshold 归一化尺度
    dist_reward = 1.0 / (1.0 + (distance / (near_goal_threshold + 1e-6))**2)

    # 分阶段加权：远离目标优先前进与保持清除，接近目标优先朝向精确并靠近目标
    is_far = distance > near_goal_threshold
    is_near = torch.logical_and(distance <= near_goal_threshold, distance > distance_threshold)
    is_at_goal = distance <= distance_threshold

    # 权重已调整并规范化（每阶段权重之和约为1）：
    # - far: 优先前进/清除障碍
    # - near: 优先朝向准确
    # - goal: 优先朝向与靠近目标（含站姿保持项）
    far_reward = 0.50 * progress_reward + 0.15 * heading_reward + 0.25 * obs_approach_raw + 0.10 * dist_reward
    near_reward = 0.20 * progress_reward + 0.45 * heading_reward + 0.15 * obs_approach_raw + 0.20 * dist_reward
    goal_reward = 0.05 * progress_reward + 0.60 * heading_reward + 0.15 * torch.exp(-torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)) + 0.20 * dist_reward

    reward = torch.zeros_like(distance)
    reward = torch.where(is_far, far_reward, reward)
    reward = torch.where(is_near, near_reward, reward)
    reward = torch.where(is_at_goal, goal_reward, reward)

    # 减去碰撞/过近惩罚及不良行为惩罚（横向速度、角速度）
    lateral_speed = torch.abs(vy)
    reward = reward - collision_penalty #- 0.08 * lateral_speed - 0.05 * ang_speed

    # 限幅防止数值爆炸
    reward = torch.clamp(reward, min=-1.0, max=2.0)

    return reward
