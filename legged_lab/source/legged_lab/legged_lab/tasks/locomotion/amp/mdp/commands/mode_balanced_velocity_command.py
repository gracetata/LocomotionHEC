"""Mode-balanced velocity commands for directional G1 AMP training."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_MODE_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def _load_mode_config(path: str) -> dict[str, Any]:
    resolved_path = os.path.abspath(os.path.expanduser(path))
    if resolved_path in _MODE_CONFIG_CACHE:
        return _MODE_CONFIG_CACHE[resolved_path]
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Mode-balanced command config not found: {resolved_path}")
    with open(resolved_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    modes = config.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ValueError(f"Mode-balanced command config has no modes: {resolved_path}")
    mode_weights = config.get("mode_weights")
    if not isinstance(mode_weights, dict) or not mode_weights:
        raise ValueError(f"Mode-balanced command config has no mode_weights: {resolved_path}")
    _MODE_CONFIG_CACHE[resolved_path] = config
    return config


def _mode_range(mode_cfg: dict[str, Any], key: str) -> tuple[float, float]:
    if key not in mode_cfg:
        raise ValueError(f"Mode config missing command range {key!r}: {mode_cfg}")
    values = mode_cfg[key]
    if not isinstance(values, list | tuple) or len(values) != 2:
        raise ValueError(f"Mode command range {key!r} must have two values: {values!r}")
    low, high = float(values[0]), float(values[1])
    if high < low:
        raise ValueError(f"Mode command range {key!r} has high < low: {values!r}")
    return low, high


class ModeBalancedVelocityCommand(UniformVelocityCommand):
    """Velocity command sampler that first samples a mode, then vx/vy/wz."""

    cfg: ModeBalancedVelocityCommandCfg

    def __init__(self, cfg: ModeBalancedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        config = _load_mode_config(cfg.sampling_config_path)
        modes = config["modes"]
        raw_weights = config["mode_weights"]

        mode_names: list[str] = []
        weights: list[float] = []
        ranges: list[list[tuple[float, float]]] = []
        for mode_name, mode_cfg in modes.items():
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
            raise ValueError(f"No positive-weight modes in {cfg.sampling_config_path}")
        self.mode_names = tuple(mode_names)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.mode_weights = weight_tensor / torch.sum(weight_tensor)
        ranges_tensor = torch.tensor(ranges, dtype=torch.float32, device=self.device)
        self.mode_range_min = ranges_tensor[:, :, 0]
        self.mode_range_max = ranges_tensor[:, :, 1]
        self.command_scale = torch.tensor(cfg.command_scale, dtype=torch.float32, device=self.device).view(1, 3)
        self.command_min = torch.tensor(cfg.command_clip_min, dtype=torch.float32, device=self.device).view(1, 3)
        self.command_max = torch.tensor(cfg.command_clip_max, dtype=torch.float32, device=self.device).view(1, 3)
        self.current_mode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def __str__(self) -> str:
        msg = "ModeBalancedVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tModes: {', '.join(self.mode_names)}\n"
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

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._resolve_env_ids(env_ids)
        count = int(env_ids_tensor.numel())
        if count == 0:
            return

        mode_ids = torch.multinomial(self.mode_weights, count, replacement=True)
        low = self.mode_range_min[mode_ids]
        high = self.mode_range_max[mode_ids]
        random_values = torch.rand(count, 3, device=self.device)
        target = low + random_values * (high - low)
        target = target * self.command_scale
        target = torch.maximum(torch.minimum(target, self.command_max), self.command_min)
        self.vel_command_b[env_ids_tensor, :] = target
        self.current_mode_ids[env_ids_tensor] = mode_ids

        standing_random_values = torch.empty(count, device=self.device).uniform_(0.0, 1.0)
        self.is_standing_env[env_ids_tensor] = standing_random_values <= float(self.cfg.rel_standing_envs)

    def _update_command(self):
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class ModeBalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for mode-balanced velocity command sampling."""

    class_type: type = ModeBalancedVelocityCommand

    sampling_config_path: str = MISSING
    """Path to `task_sampling_config.json` from the command-balanced dataset."""

    command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Element-wise scale applied after mode sampling."""

    command_clip_min: tuple[float, float, float] = (-1.2, -0.6, -1.2)
    """Element-wise minimum after scaling."""

    command_clip_max: tuple[float, float, float] = (1.2, 0.6, 1.2)
    """Element-wise maximum after scaling."""
