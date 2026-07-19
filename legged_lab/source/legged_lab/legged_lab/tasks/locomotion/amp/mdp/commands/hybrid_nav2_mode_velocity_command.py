"""Hybrid Nav2-window and mode-balanced velocity commands for Walk ArmHack.

The policy-facing command remains ``[vx, vy, wz]``. The source selector and
mode id are deliberately internal, so the policy receives neither a future command
nor a privileged mode label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

from .mode_balanced_velocity_command import _load_mode_config, _mode_range
from .nav2_recorded_velocity_command import (
    Nav2RecordedVelocityCommand,
    Nav2RecordedVelocityCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HybridNav2ModeVelocityCommand(Nav2RecordedVelocityCommand):
    """Sample each command window from Nav2 data or an eight-mode envelope."""

    cfg: HybridNav2ModeVelocityCommandCfg

    def __init__(self, cfg: HybridNav2ModeVelocityCommandCfg, env: ManagerBasedEnv):
        if not 0.0 <= float(cfg.mode_probability) <= 1.0:
            raise ValueError(f"mode_probability must be in [0, 1], got {cfg.mode_probability}")
        super().__init__(cfg, env)

        mode_config = _load_mode_config(cfg.mode_sampling_config_path)
        raw_modes: dict[str, dict[str, Any]] = mode_config["modes"]
        raw_weights: dict[str, float] = mode_config["mode_weights"]
        mode_names: list[str] = []
        weights: list[float] = []
        ranges: list[list[tuple[float, float]]] = []
        for mode_name, mode_cfg in raw_modes.items():
            weight = float(raw_weights.get(mode_name, 0.0))
            if weight <= 0.0:
                continue
            mode_names.append(str(mode_name))
            weights.append(weight)
            ranges.append(
                [
                    _mode_range(mode_cfg, "lin_vel_x"),
                    _mode_range(mode_cfg, "lin_vel_y"),
                    _mode_range(mode_cfg, "ang_vel_z"),
                ]
            )
        if not mode_names:
            raise ValueError(f"No positive-weight modes in {cfg.mode_sampling_config_path}")
        if cfg.forced_mode and cfg.forced_mode not in mode_names:
            raise ValueError(
                f"forced_mode={cfg.forced_mode!r} is not one of {', '.join(mode_names)}"
            )

        self.mode_names = tuple(mode_names)
        mode_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.mode_weights = mode_weights / torch.sum(mode_weights)
        range_tensor = torch.tensor(ranges, dtype=torch.float32, device=self.device)
        self.mode_range_min = range_tensor[:, :, 0]
        self.mode_range_max = range_tensor[:, :, 1]
        self.mode_command_scale = torch.tensor(
            cfg.mode_command_scale, dtype=torch.float32, device=self.device
        ).view(1, 3)
        self.mode_command_min = torch.tensor(
            cfg.mode_command_clip_min, dtype=torch.float32, device=self.device
        ).view(1, 3)
        self.mode_command_max = torch.tensor(
            cfg.mode_command_clip_max, dtype=torch.float32, device=self.device
        ).view(1, 3)

        self.is_mode_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_mode_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.mode_target_command_b = torch.zeros_like(self.vel_command_b)
        self.metrics["source_nav2_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["source_mode_ratio"] = torch.zeros(self.num_envs, device=self.device)
        for mode_name in self.mode_names:
            self.metrics[f"mode_{mode_name}_ratio"] = torch.zeros(
                self.num_envs, device=self.device
            )

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tMode probability: {float(self.cfg.mode_probability):.3f}"
        msg += f"\n\tModes: {', '.join(self.mode_names)}"
        if self.cfg.forced_mode:
            msg += f"\n\tForced mode: {self.cfg.forced_mode}"
        return msg

    def _sample_mode_ids(self, count: int) -> torch.Tensor:
        if self.cfg.forced_mode:
            mode_id = self.mode_names.index(self.cfg.forced_mode)
            return torch.full((count,), mode_id, dtype=torch.long, device=self.device)
        return torch.multinomial(self.mode_weights, count, replacement=True)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._resolve_env_ids(env_ids)
        count = int(env_ids_tensor.numel())
        if count == 0:
            return

        # Always establish a valid Nav2 window. Mode environments overwrite
        # only the policy-facing target; no command buffer is uninitialized.
        super()._resample_command(env_ids_tensor)
        if self.cfg.forced_mode:
            use_mode = torch.ones(count, dtype=torch.bool, device=self.device)
        else:
            use_mode = torch.rand(count, device=self.device) < float(self.cfg.mode_probability)
        self.is_mode_env[env_ids_tensor] = use_mode
        self.current_mode_ids[env_ids_tensor] = -1
        self.mode_target_command_b[env_ids_tensor] = 0.0

        mode_env_ids = env_ids_tensor[use_mode]
        if mode_env_ids.numel() == 0:
            return
        mode_ids = self._sample_mode_ids(int(mode_env_ids.numel()))
        low = self.mode_range_min[mode_ids]
        high = self.mode_range_max[mode_ids]
        target = low + torch.rand_like(low) * (high - low)
        target = target * self.mode_command_scale
        target = torch.maximum(torch.minimum(target, self.mode_command_max), self.mode_command_min)
        self.mode_target_command_b[mode_env_ids] = target
        self.current_mode_ids[mode_env_ids] = mode_ids

        # The explicit zero mode owns standing for mode-sampled environments;
        # rel_standing_envs remains the source of standing samples for Nav2.
        zero_mode = torch.all(torch.isclose(target, torch.zeros_like(target)), dim=1)
        self.is_standing_env[mode_env_ids] = zero_mode

    def _slew_limited_mode_command(
        self, previous: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        dt = float(self._env.step_dt)
        tau = float(self.cfg.smoothing_time_constant)
        if tau > 0.0:
            desired_delta = (target - previous) * min(max(dt / tau, 0.0), 1.0)
        else:
            desired_delta = target - previous

        max_linear_delta = max(float(self.cfg.max_linear_accel) * dt, 0.0)
        max_yaw_delta = max(float(self.cfg.max_yaw_accel) * dt, 0.0)
        if max_linear_delta > 0.0:
            desired_delta[:, :2] = torch.clamp(
                desired_delta[:, :2], -max_linear_delta, max_linear_delta
            )
        if max_yaw_delta > 0.0:
            desired_delta[:, 2] = torch.clamp(
                desired_delta[:, 2], -max_yaw_delta, max_yaw_delta
            )
        return previous + desired_delta

    def _update_command(self):
        mode_env_ids = self.is_mode_env.nonzero(as_tuple=False).flatten()
        previous_mode_command = self.vel_command_b[mode_env_ids].clone()
        super()._update_command()
        if mode_env_ids.numel() == 0:
            return

        target = self.mode_target_command_b[mode_env_ids].clone()
        standing = self.is_standing_env[mode_env_ids]
        target[standing] = 0.0
        self.target_vel_command_b[mode_env_ids] = target
        updated = self._slew_limited_mode_command(previous_mode_command, target)
        # A zero target is a semantic stop command, not merely the low end of a
        # continuous speed interval.  Behavior-refinement tasks opt into an
        # immediate policy-facing zero so the policy cannot hide behind the
        # command slew limiter after a stop request.  Existing Walk tasks keep
        # their historical smoothed behavior because the default is False.
        if self.cfg.hard_zero_stand:
            updated[standing] = 0.0
        self.vel_command_b[mode_env_ids] = updated

    def _update_metrics(self):
        super()._update_metrics()
        max_command_steps = max(
            float(self.cfg.resampling_time_range[1]) / float(self._env.step_dt), 1.0
        )
        increment = 1.0 / max_command_steps
        self.metrics["source_mode_ratio"] += self.is_mode_env.float() * increment
        self.metrics["source_nav2_ratio"] += (~self.is_mode_env).float() * increment
        for mode_id, mode_name in enumerate(self.mode_names):
            self.metrics[f"mode_{mode_name}_ratio"] += (
                (self.current_mode_ids == mode_id).float() * increment
            )


@configclass
class HybridNav2ModeVelocityCommandCfg(Nav2RecordedVelocityCommandCfg):
    """Configuration for a Nav2/mode-balanced hybrid command distribution."""

    class_type: type = HybridNav2ModeVelocityCommand

    mode_sampling_config_path: str = MISSING
    """JSON containing the named directional command modes and their weights."""

    mode_probability: float = 0.0
    """Probability that a window uses a named mode instead of Nav2."""

    forced_mode: str = ""
    """Optional evaluation-only mode name; non-empty forces all environments to it."""

    hard_zero_stand: bool = False
    """Immediately expose exactly zero for a sampled stand mode instead of slewing to it."""

    mode_command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Element-wise curriculum scale for mode-balanced commands."""

    mode_command_clip_min: tuple[float, float, float] = (-0.65, -0.45, -1.0)
    """Lower clip applied to mode-balanced commands."""

    mode_command_clip_max: tuple[float, float, float] = (1.05, 0.45, 1.0)
    """Upper clip applied to mode-balanced commands."""
