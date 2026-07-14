"""Nav2-recorded velocity commands for G1 AMP fine-tuning.

Core classes:
    Nav2RecordedVelocityCommandCfg configures a data-driven command
    distribution backed by ``nav2_loopback_actual`` CSV files. The command term
    samples continuous successful Nav2 ``cmd_vel`` windows grouped by
    planner/controller/scenario/goal/augmentation and emits body-frame
    ``[vx, vy, wz]`` commands without changing the existing 96-D policy input.

Inputs/outputs:
    Inputs are CSV rows with ``vx``, ``vy``, ``wz``, ``t`` and source labels.
    The output is a tensor shaped ``[num_envs, 3]`` containing the current
    velocity command. Commands can be scaled for curriculum, stratified by
    source labels, and slew-limited before being shown to the policy.

Usage:
    cfg.commands.base_velocity = Nav2RecordedVelocityCommandCfg(
        asset_name="robot",
        data_path="/path/to/nav2_loopback_actual/actual_augmented/all_cmd_vel_augmented.csv",
        augmentation_filter="none,mirror_lr",
        command_scale=(0.7, 0.55, 0.55),
    )
"""

from __future__ import annotations

import csv
import math
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


_NAV2_DATASET_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _parse_filter(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        text = values.strip()
        if not text or text == "*":
            return None
        return {part.strip() for part in text.split(",") if part.strip()}
    parsed = {str(value).strip() for value in values if str(value).strip()}
    return parsed or None


def _weighted_value(weights: dict[str, float] | None, key: str) -> float:
    if not weights:
        return 1.0
    return float(weights.get(key, weights.get("*", 1.0)))


def _estimate_sample_dt(group_times: list[list[float]], fallback_dt: float) -> float:
    deltas: list[float] = []
    for times in group_times:
        previous = None
        for value in times:
            if previous is not None:
                delta = value - previous
                if 1.0e-4 <= delta <= 0.25:
                    deltas.append(delta)
                    if len(deltas) >= 20000:
                        break
            previous = value
        if len(deltas) >= 20000:
            break
    if not deltas:
        return float(fallback_dt)
    deltas.sort()
    return float(deltas[len(deltas) // 2])


def _load_nav2_dataset(
    data_path: str,
    augmentation_filter: str | Sequence[str] | None,
    scenario_family_filter: str | Sequence[str] | None,
    combo_filter: str | Sequence[str] | None,
    controller_filter: str | Sequence[str] | None,
    planner_filter: str | Sequence[str] | None,
    fallback_sample_dt: float,
) -> dict[str, Any]:
    resolved_path = os.path.abspath(os.path.expanduser(data_path))
    cache_key = (
        resolved_path,
        tuple(sorted(_parse_filter(augmentation_filter) or [])),
        tuple(sorted(_parse_filter(scenario_family_filter) or [])),
        tuple(sorted(_parse_filter(combo_filter) or [])),
        tuple(sorted(_parse_filter(controller_filter) or [])),
        tuple(sorted(_parse_filter(planner_filter) or [])),
        float(fallback_sample_dt),
    )
    if cache_key in _NAV2_DATASET_CACHE:
        return _NAV2_DATASET_CACHE[cache_key]
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Nav2 command dataset not found: {resolved_path}")

    augmentation_set = _parse_filter(augmentation_filter)
    family_set = _parse_filter(scenario_family_filter)
    combo_set = _parse_filter(combo_filter)
    controller_set = _parse_filter(controller_filter)
    planner_set = _parse_filter(planner_filter)

    grouped_rows: dict[tuple[str, str, str, str, str, str, str], list[tuple[float, tuple[float, float, float]]]] = {}
    group_metadata_by_key: dict[tuple[str, str, str, str, str, str, str], dict[str, str]] = {}
    raw_rows = 0
    kept_rows = 0

    with open(resolved_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"vx", "vy", "wz", "combo", "planner", "controller", "scenario", "goal_id"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Nav2 command dataset missing columns: {sorted(missing_columns)}")

        for row in reader:
            raw_rows += 1
            augmentation = row.get("augmentation", "none") or "none"
            family = row.get("scenario_family", "unknown") or "unknown"
            combo = row.get("combo", "unknown") or "unknown"
            controller = row.get("controller", "unknown") or "unknown"
            planner = row.get("planner", "unknown") or "unknown"
            if augmentation_set is not None and augmentation not in augmentation_set:
                continue
            if family_set is not None and family not in family_set:
                continue
            if combo_set is not None and combo not in combo_set:
                continue
            if controller_set is not None and controller not in controller_set:
                continue
            if planner_set is not None and planner not in planner_set:
                continue

            key = (combo, planner, controller, row.get("scenario", "unknown"), row.get("goal_id", "unknown"), augmentation, family)
            group = grouped_rows.setdefault(key, [])
            if key not in group_metadata_by_key:
                group_metadata_by_key[key] = {
                    "combo": combo,
                    "planner": planner,
                    "controller": controller,
                    "scenario_family": family,
                    "augmentation": augmentation,
                }
            kept_rows += 1
            time_text = row.get("t", "")
            if time_text:
                time_value = float(time_text)
            else:
                time_value = float(len(group)) * float(fallback_sample_dt)
            group.append((time_value, (float(row["vx"]), float(row["vy"]), float(row["wz"]))))

    commands: list[tuple[float, float, float]] = []
    group_starts: list[int] = []
    group_lengths: list[int] = []
    group_metadata: list[dict[str, str]] = []
    group_times: list[list[float]] = []
    for key in sorted(grouped_rows):
        rows = sorted(grouped_rows[key], key=lambda item: item[0])
        if not rows:
            continue
        group_starts.append(len(commands))
        group_lengths.append(len(rows))
        group_metadata.append(group_metadata_by_key[key])
        group_times.append([float(item[0]) for item in rows])
        commands.extend(item[1] for item in rows)

    if not commands or not group_starts:
        raise ValueError(f"Nav2 command dataset has no rows after filtering: {resolved_path}")

    dataset = {
        "path": resolved_path,
        "raw_rows": raw_rows,
        "kept_rows": kept_rows,
        "commands": torch.tensor(commands, dtype=torch.float32),
        "group_starts": torch.tensor(group_starts, dtype=torch.long),
        "group_lengths": torch.tensor(group_lengths, dtype=torch.long),
        "group_metadata": group_metadata,
        "sample_dt": _estimate_sample_dt(group_times, fallback_sample_dt),
    }
    _NAV2_DATASET_CACHE[cache_key] = dataset
    return dataset


class Nav2RecordedVelocityCommand(UniformVelocityCommand):
    """Velocity command that replays successful Nav2 cmd_vel windows."""

    cfg: Nav2RecordedVelocityCommandCfg

    def __init__(self, cfg: Nav2RecordedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        dataset = _load_nav2_dataset(
            cfg.data_path,
            cfg.augmentation_filter,
            cfg.scenario_family_filter,
            cfg.combo_filter,
            cfg.controller_filter,
            cfg.planner_filter,
            cfg.dataset_sample_dt,
        )
        self.dataset_path = str(dataset["path"])
        self.dataset_raw_rows = int(dataset["raw_rows"])
        self.dataset_kept_rows = int(dataset["kept_rows"])
        self.dataset_sample_dt = float(dataset["sample_dt"])
        self.dataset_commands = dataset["commands"].to(self.device)
        self.group_starts = dataset["group_starts"].to(self.device)
        self.group_lengths = dataset["group_lengths"].to(self.device)
        self.group_weights = self._build_group_weights(dataset["group_metadata"], dataset["group_lengths"])
        self.command_scale = torch.tensor(cfg.command_scale, dtype=torch.float32, device=self.device).view(1, 3)
        self.command_min = torch.tensor(cfg.command_clip_min, dtype=torch.float32, device=self.device).view(1, 3)
        self.command_max = torch.tensor(cfg.command_clip_max, dtype=torch.float32, device=self.device).view(1, 3)
        self.target_vel_command_b = torch.zeros_like(self.vel_command_b)
        self.row_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.row_end_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.row_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.window_rows = max(1, int(math.ceil(float(cfg.window_duration_s) / self.dataset_sample_dt)))

    def __str__(self) -> str:
        msg = "Nav2RecordedVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tDataset rows: {self.dataset_kept_rows}/{self.dataset_raw_rows}\n"
        msg += f"\tDataset dt: {self.dataset_sample_dt:.4f}s\n"
        msg += f"\tGroups: {int(self.group_starts.numel())}\n"
        msg += f"\tCommand scale: {tuple(float(v) for v in self.command_scale.flatten())}"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        resolved_env_ids = self._resolve_env_ids(env_ids)
        extras = super().reset(resolved_env_ids)
        if self.cfg.reset_command_to_zero:
            self.vel_command_b[resolved_env_ids, :] = 0.0
        return extras

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _build_group_weights(self, metadata: list[dict[str, str]], lengths: torch.Tensor) -> torch.Tensor:
        weights: list[float] = []
        for index, item in enumerate(metadata):
            weight = 1.0
            weight *= _weighted_value(self.cfg.scenario_family_weights, item["scenario_family"])
            weight *= _weighted_value(self.cfg.combo_weights, item["combo"])
            weight *= _weighted_value(self.cfg.controller_weights, item["controller"])
            weight *= _weighted_value(self.cfg.planner_weights, item["planner"])
            weight *= _weighted_value(self.cfg.augmentation_weights, item["augmentation"])
            if not self.cfg.sample_groups_uniformly:
                weight *= float(lengths[index].item())
            weights.append(max(weight, 0.0))
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if torch.sum(weight_tensor) <= 0.0:
            raise ValueError("Nav2 command group weights sum to zero.")
        return weight_tensor / torch.sum(weight_tensor)

    def _scaled_command(self, row_indices: torch.Tensor) -> torch.Tensor:
        command = self.dataset_commands[row_indices] * self.command_scale
        return torch.maximum(torch.minimum(command, self.command_max), self.command_min)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._resolve_env_ids(env_ids)
        count = int(env_ids_tensor.numel())
        if count == 0:
            return

        group_ids = torch.multinomial(self.group_weights, count, replacement=True)
        starts = self.group_starts[group_ids]
        lengths = self.group_lengths[group_ids]
        max_offsets = torch.clamp(lengths - self.window_rows + 1, min=1)
        offsets = torch.floor(torch.rand(count, device=self.device) * max_offsets.to(dtype=torch.float32)).to(torch.long)
        row_indices = starts + offsets
        row_ends = torch.minimum(starts + lengths, row_indices + self.window_rows)

        self.row_indices[env_ids_tensor] = row_indices
        self.row_end_indices[env_ids_tensor] = row_ends
        self.row_elapsed[env_ids_tensor] = 0.0
        self.target_vel_command_b[env_ids_tensor, :] = self._scaled_command(row_indices)
        random_values = torch.empty(count, device=self.device).uniform_(0.0, 1.0)
        self.is_standing_env[env_ids_tensor] = random_values <= float(self.cfg.rel_standing_envs)

    def _update_command(self):
        target = self._scaled_command(self.row_indices)
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        target[standing_env_ids, :] = 0.0
        self.target_vel_command_b[:] = target

        dt = float(self._env.step_dt)
        tau = float(self.cfg.smoothing_time_constant)
        if tau > 0.0:
            alpha = min(max(dt / tau, 0.0), 1.0)
            desired_delta = (target - self.vel_command_b) * alpha
        else:
            desired_delta = target - self.vel_command_b

        max_linear_delta = max(float(self.cfg.max_linear_accel) * dt, 0.0)
        max_yaw_delta = max(float(self.cfg.max_yaw_accel) * dt, 0.0)
        if max_linear_delta > 0.0:
            desired_delta[:, :2] = torch.clamp(desired_delta[:, :2], -max_linear_delta, max_linear_delta)
        if max_yaw_delta > 0.0:
            desired_delta[:, 2] = torch.clamp(desired_delta[:, 2], -max_yaw_delta, max_yaw_delta)
        self.vel_command_b[:] = self.vel_command_b + desired_delta

        self.row_elapsed += dt
        advance = torch.floor(self.row_elapsed / self.dataset_sample_dt).to(dtype=torch.long)
        advance_env_ids = (advance > 0).nonzero(as_tuple=False).flatten()
        if advance_env_ids.numel() > 0:
            self.row_indices[advance_env_ids] = torch.minimum(
                self.row_indices[advance_env_ids] + advance[advance_env_ids],
                self.row_end_indices[advance_env_ids] - 1,
            )
            self.row_elapsed[advance_env_ids] -= advance[advance_env_ids].to(dtype=self.row_elapsed.dtype) * self.dataset_sample_dt


@configclass
class Nav2RecordedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for Nav2 recorded-window velocity commands."""

    class_type: type = Nav2RecordedVelocityCommand

    data_path: str = MISSING
    """CSV file containing Nav2 cmd_vel rows."""

    augmentation_filter: str | Sequence[str] | None = "none"
    """Comma-separated augmentation labels to keep; use ``*`` or empty for all."""

    scenario_family_filter: str | Sequence[str] | None = ""
    """Optional comma-separated scenario_family labels to keep."""

    combo_filter: str | Sequence[str] | None = ""
    """Optional comma-separated planner/controller combos to keep."""

    controller_filter: str | Sequence[str] | None = ""
    """Optional comma-separated controllers to keep."""

    planner_filter: str | Sequence[str] | None = ""
    """Optional comma-separated planners to keep."""

    dataset_sample_dt: float = 0.05
    """Fallback dataset sample period in seconds when timestamps are unavailable."""

    window_duration_s: float = 2.0
    """Duration of the sampled continuous Nav2 command window."""

    command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Element-wise curriculum scale applied to ``vx``, ``vy`` and ``wz``."""

    command_clip_min: tuple[float, float, float] = (-0.6, -0.3, -0.6)
    """Element-wise minimum after scaling."""

    command_clip_max: tuple[float, float, float] = (0.6, 0.3, 0.6)
    """Element-wise maximum after scaling."""

    smoothing_time_constant: float = 0.30
    """First-order smoothing time constant for command changes in seconds."""

    max_linear_accel: float = 0.60
    """Maximum command slew rate for linear velocity components in m/s^2."""

    max_yaw_accel: float = 0.80
    """Maximum command slew rate for yaw-rate command in rad/s^2."""

    reset_command_to_zero: bool = True
    """Whether reset envs should ramp command from zero after a new episode starts."""

    sample_groups_uniformly: bool = True
    """Sample goal windows uniformly by group instead of by raw row count."""

    scenario_family_weights: dict[str, float] | None = None
    """Optional weights for scenario_family labels."""

    combo_weights: dict[str, float] | None = None
    """Optional weights for planner/controller combo labels."""

    controller_weights: dict[str, float] | None = None
    """Optional weights for controller labels."""

    planner_weights: dict[str, float] | None = None
    """Optional weights for planner labels."""

    augmentation_weights: dict[str, float] | None = None
    """Optional weights for augmentation labels."""