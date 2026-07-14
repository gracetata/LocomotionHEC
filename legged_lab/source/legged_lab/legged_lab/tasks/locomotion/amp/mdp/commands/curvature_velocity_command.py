"""Segmented-yaw velocity commands for G1 AMP fine-tuning.

Core classes:
    CurvatureVelocityCommandCfg configures a velocity command distribution that
    preserves the same 3-D velocity-command interface as IsaacLab's
    UniformVelocityCommand while changing only yaw-rate sampling. Low-speed
    commands receive broad uniform yaw; higher-speed commands receive Gaussian
    yaw around zero clipped by ranges.ang_vel_z.

Inputs/outputs:
    Inputs are per-environment resampling timers and scalar range parameters for
    velocity and yaw-rate sampling. Output is a tensor shaped [num_envs, 3]
    containing body-frame lin_vel_x, lin_vel_y, and yaw_rate, compatible with
    the existing 96-D G1 AMP policy observation.

Usage:
    cfg.commands.base_velocity = CurvatureVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=CurvatureVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 1.0), lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3), low_speed_ang_vel_z=(-0.5, 0.5),
        ),
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CurvatureVelocityCommand(UniformVelocityCommand):
    """Velocity command that splits commands by speed_xy.

    Low-speed commands (speed_xy <= low_speed_threshold) use independent
    x/y sampling with uniform yaw in low_speed_ang_vel_z. High-speed commands
    use Gaussian yaw sampling around zero, clipped within ranges.ang_vel_z.
    Between resamples, commands are held constant just like UniformVelocityCommand.
    """

    cfg: CurvatureVelocityCommandCfg

    def __init__(self, cfg: CurvatureVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.is_low_speed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "CurvatureVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tLow-speed threshold: {self.cfg.low_speed_threshold}\n"
        msg += f"\tHigh-speed yaw std: {self.cfg.high_speed_ang_vel_z_std}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _uniform(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(float(value_range[0]), float(value_range[1]))

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._resolve_env_ids(env_ids)
        count = int(env_ids_tensor.numel())
        if count == 0:
            return

        target = torch.zeros(count, 3, device=self.device)
        target[:, 0] = self._uniform(count, self.cfg.ranges.lin_vel_x)
        target[:, 1] = self._uniform(count, self.cfg.ranges.lin_vel_y)

        speed_xy = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2)
        low_speed_mask = speed_xy <= float(self.cfg.low_speed_threshold)
        self.is_low_speed_env[env_ids_tensor] = low_speed_mask

        if torch.any(low_speed_mask):
            low_count = int(torch.count_nonzero(low_speed_mask).item())
            target[low_speed_mask, 2] = self._uniform(low_count, self.cfg.ranges.low_speed_ang_vel_z)

        high_speed_mask = ~low_speed_mask
        if torch.any(high_speed_mask):
            high_count = int(torch.count_nonzero(high_speed_mask).item())
            yaw_target = torch.randn(high_count, device=self.device) * float(self.cfg.high_speed_ang_vel_z_std)
            yaw_target += float(self.cfg.high_speed_ang_vel_z_mean)
            yaw_target = torch.clamp(
                yaw_target,
                float(self.cfg.ranges.ang_vel_z[0]),
                float(self.cfg.ranges.ang_vel_z[1]),
            )
            target[high_speed_mask, 2] = yaw_target

        self.vel_command_b[env_ids_tensor, :] = target
        random_values = torch.empty(count, device=self.device)
        self.is_standing_env[env_ids_tensor] = random_values.uniform_(0.0, 1.0) <= float(self.cfg.rel_standing_envs)

    def _update_command(self):
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class CurvatureVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for speed-segmented yaw-rate velocity commands."""

    class_type: type = CurvatureVelocityCommand

    low_speed_threshold: float = 0.40
    """Speed below which independent yaw/lateral commands remain valid."""

    high_speed_ang_vel_z_mean: float = 0.0
    """Mean angular velocity for high-speed Gaussian yaw sampling."""

    high_speed_ang_vel_z_std: float = 0.10
    """Standard deviation for high-speed Gaussian yaw sampling."""

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Distribution ranges for segmented-yaw velocity commands."""

        low_speed_ang_vel_z: tuple[float, float] = (-0.60, 0.60)
        """Low-speed independent yaw-rate range in rad/s."""

    ranges: Ranges = MISSING
