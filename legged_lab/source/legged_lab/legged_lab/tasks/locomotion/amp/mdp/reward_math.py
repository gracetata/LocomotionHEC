"""Torch-only numeric helpers shared by G1 AMP reward terms and unit tests."""

from __future__ import annotations

import torch


def relative_command_response_shortfall_l1(
    command: torch.Tensor,
    actual_lin_vel_xy_b: torch.Tensor,
    actual_yaw_rate_b: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
    min_speed_fraction: float = 0.50,
    min_lin_normalizer: float = 0.01,
    min_yaw_normalizer: float = 0.05,
) -> torch.Tensor:
    """Return normalized translation/yaw response shortfall."""
    if min_lin_normalizer <= 0.0 or min_yaw_normalizer <= 0.0:
        raise ValueError("Command-response normalizers must be positive.")
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_speed.unsqueeze(1), min=1.0e-8)
    projected_speed = torch.sum(actual_lin_vel_xy_b * command_direction, dim=1)
    lin_shortfall = torch.clamp(
        float(min_speed_fraction) * command_speed - projected_speed, min=0.0
    )
    lin_penalty = lin_shortfall / torch.clamp(
        command_speed, min=float(min_lin_normalizer)
    )
    lin_penalty *= (command_speed > float(epsilon)).float()

    yaw_magnitude = torch.abs(command[:, 2])
    desired_yaw = yaw_magnitude * float(min_speed_fraction)
    signed_yaw_response = torch.sign(command[:, 2]) * actual_yaw_rate_b
    yaw_shortfall = torch.clamp(desired_yaw - signed_yaw_response, min=0.0)
    yaw_penalty = yaw_shortfall / torch.clamp(
        yaw_magnitude, min=float(min_yaw_normalizer)
    )
    yaw_penalty *= (yaw_magnitude > float(epsilon)).float()
    return lin_penalty + yaw_penalty


def allowed_footstep_cadence_hz(
    command: torch.Tensor,
    base_hz: float = 1.6,
    linear_gain: float = 2.33,
    yaw_gain: float = 1.5,
    maximum_hz: float = 3.0,
) -> torch.Tensor:
    """Map a velocity command to the maximum desired total touchdown cadence."""
    if base_hz <= 0.0 or maximum_hz < base_hz:
        raise ValueError("Cadence bounds must satisfy 0 < base_hz <= maximum_hz.")
    cadence = (
        float(base_hz)
        + float(linear_gain) * torch.linalg.vector_norm(command[:, :2], dim=1)
        + float(yaw_gain) * torch.abs(command[:, 2])
    )
    return torch.clamp(cadence, min=float(base_hz), max=float(maximum_hz))


def nonzero_single_stance_command_scale(
    command: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
    micro_linear_speed_max: float = 0.15,
    pure_yaw_translation_max: float = 0.005,
    pure_yaw_min_command: float = 0.05,
    micro_speed_bonus: float = 0.50,
    pure_yaw_bonus: float = 0.75,
) -> torch.Tensor:
    """Scale stepping reward up for micro-translation and in-place yaw commands.

    The bonuses use a maximum rather than a sum, so pure yaw receives the
    intended pure-yaw scale without double-counting the zero translation as a
    micro-speed command. Exact zero remains inactive.
    """
    if micro_linear_speed_max <= epsilon:
        raise ValueError("micro_linear_speed_max must be greater than epsilon.")
    if pure_yaw_translation_max < 0.0 or pure_yaw_min_command <= epsilon:
        raise ValueError("Pure-yaw command thresholds must be non-negative and nonzero.")
    if micro_speed_bonus < 0.0 or pure_yaw_bonus < 0.0:
        raise ValueError("Single-stance command bonuses must be non-negative.")

    linear_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    yaw_magnitude = torch.abs(command[:, 2])
    active = torch.linalg.vector_norm(command, dim=1) > float(epsilon)
    micro_translation = (linear_speed > float(epsilon)) & (
        linear_speed <= float(micro_linear_speed_max)
    )
    pure_yaw = (linear_speed <= float(pure_yaw_translation_max)) & (
        yaw_magnitude >= float(pure_yaw_min_command)
    )
    bonus = torch.maximum(
        micro_translation.float() * float(micro_speed_bonus),
        pure_yaw.float() * float(pure_yaw_bonus),
    )
    return active.float() * (1.0 + bonus)


def update_footstep_cadence_state(
    elapsed_since_touchdown: torch.Tensor,
    cadence_ema_hz: torch.Tensor,
    has_touchdown: torch.Tensor,
    has_interval: torch.Tensor,
    touchdown: torch.Tensor,
    *,
    step_dt: float,
    ema_alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance resettable touchdown cadence state by one simulation step."""
    if step_dt <= 0.0:
        raise ValueError("Cadence state requires step_dt > 0.")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1].")
    elapsed = elapsed_since_touchdown + float(step_dt)
    cadence_ema = cadence_ema_hz.clone()
    seen_touchdown = has_touchdown.clone()
    seen_interval = has_interval.clone()

    valid_interval = touchdown & seen_touchdown
    instantaneous_hz = torch.reciprocal(torch.clamp(elapsed, min=float(step_dt)))
    first_interval = valid_interval & ~seen_interval
    repeated_interval = valid_interval & seen_interval
    cadence_ema[first_interval] = instantaneous_hz[first_interval]
    cadence_ema[repeated_interval] = (
        float(ema_alpha) * instantaneous_hz[repeated_interval]
        + (1.0 - float(ema_alpha)) * cadence_ema[repeated_interval]
    )
    seen_interval |= valid_interval
    seen_touchdown |= touchdown
    elapsed[touchdown] = 0.0
    return elapsed, cadence_ema, seen_touchdown, seen_interval


def convex_footprint_signed_clearance_xy(
    left_corners_xy: torch.Tensor,
    right_corners_xy: torch.Tensor,
) -> torch.Tensor:
    """Compute conservative signed SAT clearance for two batched convex footprints."""
    if left_corners_xy.ndim != 3 or right_corners_xy.ndim != 3:
        raise ValueError("Footprint tensors must have shape [num_envs, vertices, 2].")
    if left_corners_xy.shape != right_corners_xy.shape:
        raise ValueError("Left and right footprint tensors must have identical shapes.")
    if left_corners_xy.shape[-1] != 2 or left_corners_xy.shape[1] < 3:
        raise ValueError("Each footprint requires at least three 2-D vertices.")

    polygons = torch.stack([left_corners_xy, right_corners_xy], dim=1)
    edges = torch.roll(polygons, shifts=-1, dims=2) - polygons
    edge_normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
    axes = edge_normals.reshape(left_corners_xy.shape[0], -1, 2)
    axes = axes / torch.clamp(
        torch.linalg.vector_norm(axes, dim=2, keepdim=True), min=1.0e-8
    )
    left_projection = torch.sum(
        left_corners_xy.unsqueeze(1) * axes.unsqueeze(2), dim=3
    )
    right_projection = torch.sum(
        right_corners_xy.unsqueeze(1) * axes.unsqueeze(2), dim=3
    )
    left_min = torch.min(left_projection, dim=2).values
    left_max = torch.max(left_projection, dim=2).values
    right_min = torch.min(right_projection, dim=2).values
    right_max = torch.max(right_projection, dim=2).values
    axis_separation = torch.maximum(right_min - left_max, left_min - right_max)
    return torch.max(axis_separation, dim=1).values


def two_goal_command_masks(
    command: torch.Tensor,
    *,
    lateral_min_command: float = 0.10,
    lateral_max_forward_command: float = 0.25,
    lateral_max_yaw_command: float = 0.05,
    pure_yaw_min_command: float = 0.10,
    pure_yaw_max_translation_command: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify the two explicit refinement modes without a privileged mode id."""
    if command.ndim != 2 or command.shape[1] != 3:
        raise ValueError("Velocity commands must have shape [num_envs, 3].")
    if lateral_min_command <= 0.0 or pure_yaw_min_command <= 0.0:
        raise ValueError("Two-goal command thresholds must be positive.")

    lateral = (
        (torch.abs(command[:, 1]) >= float(lateral_min_command))
        & (torch.abs(command[:, 0]) <= float(lateral_max_forward_command))
        & (torch.abs(command[:, 2]) <= float(lateral_max_yaw_command))
    )
    pure_yaw = (
        (torch.linalg.vector_norm(command[:, :2], dim=1) <= float(pure_yaw_max_translation_command))
        & (torch.abs(command[:, 2]) >= float(pure_yaw_min_command))
    )
    return lateral, pure_yaw


def signed_command_progress_ratio(
    command_value: torch.Tensor,
    actual_value: torch.Tensor,
    *,
    min_command: float,
    minimum_ratio: float = -1.0,
    maximum_ratio: float = 1.25,
) -> torch.Tensor:
    """Return bounded signed response divided by command magnitude.

    A stationary policy receives exactly zero, motion in the wrong direction is
    negative, and modest overshoot cannot dominate the task return.
    """
    if command_value.shape != actual_value.shape:
        raise ValueError("Command and response tensors must have identical shapes.")
    if min_command <= 0.0 or maximum_ratio <= minimum_ratio:
        raise ValueError("Invalid command-progress normalization bounds.")
    active = torch.abs(command_value) >= float(min_command)
    ratio = torch.sign(command_value) * actual_value / torch.clamp(
        torch.abs(command_value), min=float(min_command)
    )
    return torch.clamp(ratio, min=float(minimum_ratio), max=float(maximum_ratio)) * active.float()


def two_goal_dense_root_pose_progress(
    command: torch.Tensor,
    forward_delta: torch.Tensor,
    lateral_delta: torch.Tensor,
    yaw_delta: torch.Tensor,
    *,
    step_dt: float,
    lateral_min_command: float = 0.10,
    pure_yaw_min_command: float = 0.10,
    forward_leak_scale: float = 0.20,
    yaw_leak_scale: float = 0.30,
    planar_drift_scale: float = 0.15,
) -> torch.Tensor:
    """Score finite-difference root-pose progress for the two target modes."""
    expected_shape = command[:, 0].shape
    if any(value.shape != expected_shape for value in (forward_delta, lateral_delta, yaw_delta)):
        raise ValueError("Root-pose deltas must have shape [num_envs].")
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive.")
    if forward_leak_scale <= 0.0 or yaw_leak_scale <= 0.0 or planar_drift_scale <= 0.0:
        raise ValueError("Dense root-pose progress quality scales must be positive.")

    lateral, pure_yaw = two_goal_command_masks(
        command,
        lateral_min_command=lateral_min_command,
        pure_yaw_min_command=pure_yaw_min_command,
    )
    forward_rate = forward_delta / float(step_dt)
    lateral_rate = lateral_delta / float(step_dt)
    yaw_rate = yaw_delta / float(step_dt)
    lateral_ratio = signed_command_progress_ratio(
        command[:, 1],
        lateral_rate,
        min_command=lateral_min_command,
        minimum_ratio=-1.25,
        maximum_ratio=1.25,
    )
    yaw_ratio = signed_command_progress_ratio(
        command[:, 2],
        yaw_rate,
        min_command=pure_yaw_min_command,
        minimum_ratio=-1.25,
        maximum_ratio=1.25,
    )
    planar_rate = torch.sqrt(torch.square(forward_rate) + torch.square(lateral_rate))
    lateral_quality = torch.exp(
        -torch.square(forward_rate / float(forward_leak_scale))
        -torch.square(yaw_rate / float(yaw_leak_scale))
    )
    yaw_quality = torch.exp(-torch.square(planar_rate / float(planar_drift_scale)))
    progress = torch.where(lateral, lateral_ratio, yaw_ratio)
    quality = torch.where(lateral, lateral_quality, yaw_quality)
    # Do not allow leakage to suppress punishment for wrong-direction motion.
    progress = torch.where(progress > 0.0, progress * quality, progress)
    return progress * (lateral | pure_yaw).float()


def two_goal_response_shortfall_l2(
    command: torch.Tensor,
    actual_lin_vel_xy_b: torch.Tensor,
    actual_yaw_rate_b: torch.Tensor,
    *,
    target_fraction: float = 0.50,
    max_penalty: float = 1.0,
    lateral_min_command: float = 0.10,
    pure_yaw_min_command: float = 0.10,
) -> torch.Tensor:
    """Return a bounded response-shortfall penalty for the two specialization modes.

    A stationary response has a penalty of ``target_fraction**2``. Motion in the
    commanded direction removes the penalty once it reaches ``target_fraction``
    of the requested speed, while reverse motion is penalized more strongly.
    """
    if target_fraction <= 0.0 or max_penalty <= 0.0:
        raise ValueError("Response target_fraction and max_penalty must be positive.")
    if actual_lin_vel_xy_b.shape != command[:, :2].shape:
        raise ValueError("Planar velocity response must have shape [num_envs, 2].")
    if actual_yaw_rate_b.shape != command[:, 2].shape:
        raise ValueError("Yaw-rate response must have shape [num_envs].")

    lateral, pure_yaw = two_goal_command_masks(
        command,
        lateral_min_command=lateral_min_command,
        pure_yaw_min_command=pure_yaw_min_command,
    )
    lateral_ratio = (
        torch.sign(command[:, 1]) * actual_lin_vel_xy_b[:, 1]
        / torch.clamp(torch.abs(command[:, 1]), min=float(lateral_min_command))
    )
    yaw_ratio = (
        torch.sign(command[:, 2]) * actual_yaw_rate_b
        / torch.clamp(torch.abs(command[:, 2]), min=float(pure_yaw_min_command))
    )
    response_ratio = torch.where(lateral, lateral_ratio, yaw_ratio)
    shortfall = torch.square(torch.clamp(float(target_fraction) - response_ratio, min=0.0))
    active = lateral | pure_yaw
    return torch.clamp(shortfall, max=float(max_penalty)) * active.float()


def touchdown_pose_progress(
    previous_root_xy_w: torch.Tensor,
    previous_heading_w: torch.Tensor,
    current_root_xy_w: torch.Tensor,
    current_heading_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure actual forward/lateral displacement and wrapped yaw change.

    Translation is expressed in the root-yaw frame saved at the preceding
    touchdown. This prevents torso sway or world-frame drift from being counted
    as command-directed progress.
    """
    if previous_root_xy_w.shape != current_root_xy_w.shape:
        raise ValueError("Previous and current root positions must have identical shapes.")
    if previous_root_xy_w.ndim != 2 or previous_root_xy_w.shape[1] != 2:
        raise ValueError("Root XY positions must have shape [num_envs, 2].")
    if previous_heading_w.shape != current_heading_w.shape:
        raise ValueError("Previous and current headings must have identical shapes.")
    if previous_heading_w.ndim != 1 or previous_heading_w.shape[0] != previous_root_xy_w.shape[0]:
        raise ValueError("Root headings must have shape [num_envs].")

    delta_w = current_root_xy_w - previous_root_xy_w
    cos_heading = torch.cos(previous_heading_w)
    sin_heading = torch.sin(previous_heading_w)
    forward_delta = cos_heading * delta_w[:, 0] + sin_heading * delta_w[:, 1]
    lateral_delta = -sin_heading * delta_w[:, 0] + cos_heading * delta_w[:, 1]
    yaw_delta = torch.remainder(current_heading_w - previous_heading_w + torch.pi, 2.0 * torch.pi) - torch.pi
    return forward_delta, lateral_delta, yaw_delta


def swept_convex_footprint_signed_clearance_xy(
    previous_left_corners_xy: torch.Tensor,
    previous_right_corners_xy: torch.Tensor,
    current_left_corners_xy: torch.Tensor,
    current_right_corners_xy: torch.Tensor,
    *,
    interpolation_steps: int = 4,
) -> torch.Tensor:
    """Conservatively evaluate footprint clearance along one simulation step."""
    if interpolation_steps < 1:
        raise ValueError("interpolation_steps must be at least one.")
    tensors = (
        previous_left_corners_xy,
        previous_right_corners_xy,
        current_left_corners_xy,
        current_right_corners_xy,
    )
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("All swept-footprint tensors must have identical shapes.")

    dtype = current_left_corners_xy.dtype
    device = current_left_corners_xy.device
    alpha = torch.linspace(
        1.0 / interpolation_steps,
        1.0,
        interpolation_steps,
        dtype=dtype,
        device=device,
    ).view(1, interpolation_steps, 1, 1)
    previous_left = previous_left_corners_xy.unsqueeze(1)
    previous_right = previous_right_corners_xy.unsqueeze(1)
    left = previous_left + alpha * (current_left_corners_xy.unsqueeze(1) - previous_left)
    right = previous_right + alpha * (current_right_corners_xy.unsqueeze(1) - previous_right)
    batch, steps, vertices, _ = left.shape
    clearance = convex_footprint_signed_clearance_xy(
        left.reshape(batch * steps, vertices, 2),
        right.reshape(batch * steps, vertices, 2),
    ).view(batch, steps)
    return torch.min(clearance, dim=1).values
