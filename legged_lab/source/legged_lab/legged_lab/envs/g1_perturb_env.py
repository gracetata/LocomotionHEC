"""AMP environment that injects scripted upper-body perturbations."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

LEGGED_LAB_ROOT_DIR = Path(__file__).resolve().parents[1]

from .manager_based_amp_env import ManagerBasedAmpEnv


G1_UPPER_BODY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_FULL_BODY_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_FULL_BODY_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_LOWER_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


@configclass
class UpperBodyPerturbationCfg:
    """Configuration for scripted upper-body perturbations."""

    enabled: bool = True
    joint_names: list[str] = G1_UPPER_BODY_JOINT_NAMES
    source: Literal["sine", "csv", "pose_set"] = "sine"
    csv_path: str = ""
    csv_time_column: str = "time_s"
    csv_loop: bool = True
    csv_joint_column_map: dict[str, str] = {}
    csv_use_g1_action_order_q_columns: bool = False
    csv_q_column_joint_order: Literal["lab", "sdk"] = "lab"
    csv_randomize_start_on_reset: bool = False
    csv_end_margin_s: float = 0.0
    csv_interpolate: bool = True
    csv_initialize_joint_state_on_reset: bool = False
    csv_curriculum_enabled: bool = False
    csv_curriculum_static_steps: int = 0
    csv_curriculum_ramp_steps: int = 0
    csv_curriculum_motion_scale: float = 1.0
    amplitude: float | dict[str, float] = 0.35
    frequency_hz: float = 0.5
    phase_offset: float = 0.0
    joint_phase_offsets: dict[str, float] = {}
    joint_position_offsets: dict[str, float] = {}
    use_default_center: bool = True
    clip_to_joint_limits: bool = True
    pose_set: list[list[float]] = []
    pose_probabilities: list[float] = []


class G1PerturbAmpEnv(ManagerBasedAmpEnv):
    """AMP env that replaces upper-body raw actions with scripted targets."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._perturbation_cfg: UpperBodyPerturbationCfg | None = None
        self._joint_pos_action: JointPositionAction | None = None
        self._upper_action_indices: torch.Tensor | None = None
        self._upper_asset_joint_ids: torch.Tensor | None = None
        self._upper_amplitudes: torch.Tensor | None = None
        self._upper_phase_offsets: torch.Tensor | None = None
        self._upper_joint_position_offsets: torch.Tensor | None = None
        self._csv_times: torch.Tensor | None = None
        self._csv_targets: torch.Tensor | None = None
        self._csv_start_time_offsets: torch.Tensor | None = None
        self._csv_sample_times: torch.Tensor | None = None
        self._pose_set_targets: torch.Tensor | None = None
        self._pose_probabilities: torch.Tensor | None = None
        self._active_pose_targets: torch.Tensor | None = None

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._upper_action_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self._upper_asset_joint_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._upper_amplitudes = torch.empty(0, device=self.device)
        self._upper_phase_offsets = torch.empty(0, device=self.device)
        self._upper_joint_position_offsets = torch.empty(0, device=self.device)
        self._csv_times = torch.empty(0, dtype=torch.float32, device=self.device)
        self._csv_targets = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        self._csv_start_time_offsets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._csv_sample_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._pose_set_targets = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        self._pose_probabilities = torch.empty(0, dtype=torch.float32, device=self.device)
        self._active_pose_targets = torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)

        self._perturbation_cfg = getattr(self.cfg, "upper_body_perturbation", None)
        if self._perturbation_cfg is not None and self._perturbation_cfg.enabled:
            self._initialize_upper_body_perturbation()
        else:
            self._perturbation_cfg = None

    def step(self, action: torch.Tensor):
        csv_motion_scale: float | None = None
        if self._perturbation_cfg is not None:
            action = self._compose_perturbed_action(action.to(self.device))
            if self._perturbation_cfg.source == "csv":
                csv_motion_scale = self._csv_curriculum_motion_scale()
                self._advance_csv_sample_times(csv_motion_scale)

        step_result = super().step(action)
        if csv_motion_scale is not None:
            log_extras = self.extras.setdefault("log", {})
            log_extras["ArmHack/csv_motion_scale"] = torch.tensor(csv_motion_scale, device=self.device)
            target_scale = max(float(self._perturbation_cfg.csv_curriculum_motion_scale), 0.0)
            if csv_motion_scale <= 0.0:
                stage = 0.0
            elif csv_motion_scale + 1.0e-8 < target_scale:
                stage = 1.0
            else:
                stage = 2.0
            log_extras["ArmHack/curriculum_stage"] = torch.tensor(stage, device=self.device)
            log_extras["ArmHack/csv_start_time_mean_s"] = torch.mean(self._csv_start_time_offsets)
            log_extras["ArmHack/csv_start_time_std_s"] = torch.std(
                self._csv_start_time_offsets, unbiased=False
            )
        return step_result

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor):
        super()._reset_idx(env_ids)

        cfg = getattr(self, "_perturbation_cfg", None)
        if cfg is None:
            return

        env_ids_tensor = self._as_env_ids_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        if cfg.source == "csv":
            self._reset_csv_start_offsets(env_ids_tensor)
            if cfg.csv_initialize_joint_state_on_reset:
                self._initialize_csv_arm_joint_state(env_ids_tensor)
        elif cfg.source == "pose_set":
            self._sample_pose_targets(env_ids_tensor)

    def _initialize_upper_body_perturbation(self) -> None:
        action_term = self.action_manager.get_term("joint_pos")
        if not isinstance(action_term, JointPositionAction):
            raise TypeError(
                "G1PerturbAmpEnv expects the 'joint_pos' action term to be a JointPositionAction, "
                f"but got {type(action_term).__name__}."
            )

        cfg = self._perturbation_cfg
        assert cfg is not None
        if not cfg.joint_names:
            raise ValueError("upper_body_perturbation.joint_names must not be empty.")

        self._joint_pos_action = action_term
        action_joint_names = list(action_term._joint_names)
        action_index_by_name = {name: index for index, name in enumerate(action_joint_names)}
        missing_joint_names = sorted(set(cfg.joint_names).difference(action_index_by_name))
        if missing_joint_names:
            raise ValueError(
                "Upper-body perturbation joints are missing from the joint_pos action term: "
                f"{missing_joint_names}"
            )

        upper_action_indices = [action_index_by_name[name] for name in cfg.joint_names]
        resolved_joint_ids = (
            list(range(action_term.action_dim))
            if isinstance(action_term._joint_ids, slice)
            else list(action_term._joint_ids)
        )

        self._upper_action_indices = torch.tensor(upper_action_indices, dtype=torch.long, device=self.device)
        self._upper_asset_joint_ids = torch.tensor(
            [resolved_joint_ids[index] for index in upper_action_indices], dtype=torch.long, device=self.device
        )
        self._upper_amplitudes = self._resolve_per_joint_parameter(cfg.amplitude, default_value=0.0)
        self._upper_phase_offsets = self._resolve_per_joint_parameter(cfg.joint_phase_offsets, default_value=0.0)
        self._upper_joint_position_offsets = self._resolve_per_joint_parameter(
            cfg.joint_position_offsets, default_value=0.0
        )

        if cfg.source == "csv":
            self._validate_csv_curriculum_cfg()
            self._csv_times, self._csv_targets = self._load_csv_trajectory()
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._reset_csv_start_offsets(all_env_ids)
            if cfg.csv_initialize_joint_state_on_reset:
                self._initialize_csv_arm_joint_state(all_env_ids)
        elif cfg.source == "pose_set":
            self._initialize_pose_set()

    def _compose_perturbed_action(self, raw_policy_action: torch.Tensor) -> torch.Tensor:
        cfg = self._perturbation_cfg
        assert cfg is not None

        perturbed_action = raw_policy_action.clone()
        upper_position_targets = self._compute_upper_body_position_targets()
        raw_upper_action = self._joint_targets_to_raw_actions(upper_position_targets)
        perturbed_action[:, self._upper_action_indices] = raw_upper_action
        return perturbed_action

    def _compute_upper_body_position_targets(self) -> torch.Tensor:
        cfg = self._perturbation_cfg
        assert cfg is not None

        if cfg.source == "csv":
            targets = self._sample_csv_targets()
        elif cfg.source == "pose_set":
            targets = self._active_pose_targets
        else:
            episode_time_s = self.episode_length_buf.to(dtype=torch.float32).unsqueeze(-1) * self.step_dt
            phase = (
                2.0 * math.pi * float(cfg.frequency_hz) * episode_time_s
                + float(cfg.phase_offset)
                + self._upper_phase_offsets.unsqueeze(0)
            )
            center = self._upper_position_center()
            targets = center + self._upper_joint_position_offsets.unsqueeze(0)
            targets = targets + self._upper_amplitudes.unsqueeze(0) * torch.sin(phase)

        return self._clip_upper_position_targets(targets)

    def _clip_upper_position_targets(
        self, targets: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if not cfg.clip_to_joint_limits:
            return targets

        robot = self.scene["robot"]
        joint_limits = robot.data.soft_joint_pos_limits[:, self._upper_asset_joint_ids, :]
        if env_ids is not None:
            joint_limits = joint_limits[env_ids]
        return torch.clamp(targets, min=joint_limits[..., 0], max=joint_limits[..., 1])

    def _upper_position_center(self) -> torch.Tensor:
        action_term = self._joint_pos_action
        cfg = self._perturbation_cfg
        assert action_term is not None
        assert cfg is not None

        if cfg.use_default_center:
            offset = action_term._offset
            if isinstance(offset, torch.Tensor):
                return offset[:, self._upper_action_indices]
            return torch.full(
                (self.num_envs, self._upper_action_indices.numel()),
                float(offset),
                dtype=torch.float32,
                device=self.device,
            )
        return torch.zeros(
            (self.num_envs, self._upper_action_indices.numel()),
            dtype=torch.float32,
            device=self.device,
        )

    def _joint_targets_to_raw_actions(self, joint_position_targets: torch.Tensor) -> torch.Tensor:
        action_term = self._joint_pos_action
        assert action_term is not None

        offset = action_term._offset
        scale = action_term._scale

        if isinstance(offset, torch.Tensor):
            upper_offset = offset[:, self._upper_action_indices]
        else:
            upper_offset = torch.full_like(joint_position_targets, float(offset))

        if isinstance(scale, torch.Tensor):
            upper_scale = scale[:, self._upper_action_indices]
        else:
            upper_scale = torch.full_like(joint_position_targets, float(scale))

        raw_actions = (joint_position_targets - upper_offset) / torch.clamp(upper_scale, min=1.0e-6)
        if action_term.cfg.clip is not None:
            raw_actions = torch.clamp(
                raw_actions,
                min=action_term._clip[:, self._upper_action_indices, 0],
                max=action_term._clip[:, self._upper_action_indices, 1],
            )
        return raw_actions

    def _resolve_per_joint_parameter(
        self, value: float | dict[str, float], default_value: float
    ) -> torch.Tensor:
        cfg = self._perturbation_cfg
        assert cfg is not None

        if isinstance(value, dict):
            resolved_values = [float(value.get(name, default_value)) for name in cfg.joint_names]
        else:
            resolved_values = [float(value)] * len(cfg.joint_names)
        return torch.tensor(resolved_values, dtype=torch.float32, device=self.device)

    def _initialize_pose_set(self) -> None:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if not cfg.pose_set:
            raise ValueError("upper_body_perturbation.pose_set must not be empty when source='pose_set'.")

        pose_set_targets = torch.tensor(cfg.pose_set, dtype=torch.float32, device=self.device)
        expected_num_joints = self._upper_action_indices.numel()
        if pose_set_targets.ndim != 2 or pose_set_targets.shape[1] != expected_num_joints:
            raise ValueError(
                "Each upper-body pose must have exactly "
                f"{expected_num_joints} entries, but got shape {tuple(pose_set_targets.shape)}."
            )

        self._pose_set_targets = pose_set_targets
        if cfg.pose_probabilities:
            if len(cfg.pose_probabilities) != pose_set_targets.shape[0]:
                raise ValueError(
                    "upper_body_perturbation.pose_probabilities must have the same length as pose_set."
                )
            probabilities = torch.tensor(cfg.pose_probabilities, dtype=torch.float32, device=self.device)
            if torch.any(probabilities < 0.0) or float(torch.sum(probabilities).item()) <= 0.0:
                raise ValueError("upper_body_perturbation.pose_probabilities must be non-negative and sum to > 0.")
            self._pose_probabilities = probabilities / torch.sum(probabilities)
        else:
            self._pose_probabilities = torch.empty(0, dtype=torch.float32, device=self.device)

        self._active_pose_targets = torch.zeros(
            (self.num_envs, expected_num_joints), dtype=torch.float32, device=self.device
        )
        self._sample_pose_targets(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _load_csv_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if not cfg.csv_path:
            raise ValueError("upper_body_perturbation.csv_path must be set when source='csv'.")

        resolved_path = self._resolve_csv_path(cfg.csv_path)
        with resolved_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader, None)
            if fieldnames is None:
                raise ValueError(f"CSV perturbation file is missing a header: {resolved_path}")

            has_time_column = cfg.csv_time_column in fieldnames
            time_column_index = fieldnames.index(cfg.csv_time_column) if has_time_column else None
            csv_times: list[float] = []
            csv_targets: list[list[float]] = []

            q_column_source_order = self._resolve_csv_q_column_source_order(fieldnames)
            if q_column_source_order is not None:
                q_column_indices = self._resolve_csv_q_target_indices(q_column_source_order, fieldnames)
                q_field_indices = [fieldnames.index(f"q{joint_index}") for joint_index in range(len(q_column_source_order))]
                for row_index, row in enumerate(reader):
                    if has_time_column:
                        csv_times.append(float(row[time_column_index]))
                    else:
                        csv_times.append(row_index * self.step_dt)
                    if len(row) == len(fieldnames):
                        raw_q_values = [float(row[column_index]) for column_index in q_field_indices]
                    else:
                        # Some source rows contain an extra natural-language column before q0.
                        # In that ragged case the final 29 columns remain the G1 q-values.
                        raw_q_values = [float(value) for value in row[-len(q_column_source_order):]]
                    csv_targets.append([raw_q_values[index] for index in q_column_indices])
            else:
                joint_column_indices = self._resolve_csv_named_joint_column_indices(fieldnames)
                for row_index, row in enumerate(reader):
                    if has_time_column:
                        csv_times.append(float(row[time_column_index]))
                    else:
                        csv_times.append(row_index * self.step_dt)
                    csv_targets.append([float(row[column_index]) for column_index in joint_column_indices])

        if not csv_targets:
            raise ValueError(f"CSV perturbation file contains no trajectory rows: {resolved_path}")
        if any(next_time < current_time for current_time, next_time in zip(csv_times, csv_times[1:])):
            raise ValueError(f"CSV perturbation times must be monotonically non-decreasing: {resolved_path}")
        if csv_times:
            initial_time = csv_times[0]
            csv_times = [time_value - initial_time for time_value in csv_times]

        return (
            torch.tensor(csv_times, dtype=torch.float32, device=self.device),
            torch.tensor(csv_targets, dtype=torch.float32, device=self.device),
        )

    def _resolve_csv_q_column_source_order(self, fieldnames: Sequence[str]) -> list[str] | None:
        cfg = self._perturbation_cfg
        assert cfg is not None

        if cfg.csv_joint_column_map:
            return None

        missing_named_columns = sorted(set(cfg.joint_names).difference(fieldnames))
        if not missing_named_columns:
            return None

        if not cfg.csv_use_g1_action_order_q_columns:
            return None

        if cfg.csv_q_column_joint_order == "lab":
            source_order = G1_FULL_BODY_ACTION_JOINT_NAMES
        elif cfg.csv_q_column_joint_order == "sdk":
            source_order = G1_FULL_BODY_SDK_JOINT_NAMES
        else:
            raise ValueError(
                "upper_body_perturbation.csv_q_column_joint_order must be 'lab' or 'sdk', "
                f"got {cfg.csv_q_column_joint_order!r}."
            )

        missing_q_columns = [f"q{joint_index}" for joint_index in range(len(source_order)) if f"q{joint_index}" not in fieldnames]
        if missing_q_columns:
            raise ValueError(f"CSV perturbation file is missing required q-columns: {missing_q_columns}")
        return source_order

    def _resolve_csv_q_target_indices(self, source_order: Sequence[str], fieldnames: Sequence[str]) -> list[int]:
        cfg = self._perturbation_cfg
        assert cfg is not None

        index_by_joint_name = {joint_name: index for index, joint_name in enumerate(source_order)}
        missing_joint_names = sorted(set(cfg.joint_names).difference(index_by_joint_name))
        if missing_joint_names:
            raise ValueError(f"CSV q-column source order is missing joints: {missing_joint_names}")
        return [index_by_joint_name[joint_name] for joint_name in cfg.joint_names]

    def _resolve_csv_named_joint_column_indices(self, fieldnames: Sequence[str]) -> list[int]:
        cfg = self._perturbation_cfg
        assert cfg is not None

        if cfg.csv_joint_column_map:
            missing_joint_mappings = sorted(set(cfg.joint_names).difference(cfg.csv_joint_column_map))
            if missing_joint_mappings:
                raise ValueError(
                    "upper_body_perturbation.csv_joint_column_map is missing joint mappings for: "
                    f"{missing_joint_mappings}"
                )
            resolved_columns = [cfg.csv_joint_column_map[joint_name] for joint_name in cfg.joint_names]
        else:
            missing_named_columns = sorted(set(cfg.joint_names).difference(fieldnames))
            if missing_named_columns:
                raise ValueError(
                    "CSV perturbation file does not contain named upper-body columns and "
                    "csv_use_g1_action_order_q_columns is disabled."
                )
            resolved_columns = list(cfg.joint_names)

        missing_columns = sorted(set(resolved_columns).difference(fieldnames))
        if missing_columns:
            raise ValueError(f"CSV perturbation file is missing mapped columns: {missing_columns}")
        column_index_by_name = {name: index for index, name in enumerate(fieldnames)}
        return [column_index_by_name[column_name] for column_name in resolved_columns]

    def _resolve_csv_path(self, csv_path: str) -> Path:
        path = Path(csv_path).expanduser()
        if path.is_absolute() and path.is_file():
            return path

        candidate_roots = [
            Path.cwd(),
            Path(LEGGED_LAB_ROOT_DIR),
            Path(LEGGED_LAB_ROOT_DIR).parent,
            Path(LEGGED_LAB_ROOT_DIR).parents[2],
        ]
        for root in candidate_roots:
            candidate = (root / path).resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Upper-body perturbation CSV not found: {csv_path}")

    def _validate_csv_curriculum_cfg(self) -> None:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if cfg.csv_curriculum_static_steps < 0:
            raise ValueError("csv_curriculum_static_steps must be non-negative.")
        if cfg.csv_curriculum_ramp_steps < 0:
            raise ValueError("csv_curriculum_ramp_steps must be non-negative.")
        if cfg.csv_curriculum_motion_scale < 0.0:
            raise ValueError("csv_curriculum_motion_scale must be non-negative.")

    def _csv_curriculum_motion_scale(self) -> float:
        cfg = self._perturbation_cfg
        assert cfg is not None
        target_scale = max(float(cfg.csv_curriculum_motion_scale), 0.0)
        if not cfg.csv_curriculum_enabled:
            return target_scale

        step = max(int(self.common_step_counter), 0)
        static_steps = max(int(cfg.csv_curriculum_static_steps), 0)
        ramp_steps = max(int(cfg.csv_curriculum_ramp_steps), 0)
        if step < static_steps:
            return 0.0
        if ramp_steps == 0:
            return target_scale

        ramp_fraction = min(max((step - static_steps) / ramp_steps, 0.0), 1.0)
        return target_scale * ramp_fraction

    def _sample_csv_targets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if self._csv_times.numel() == 0:
            raise RuntimeError("CSV perturbation trajectory is not initialized.")

        sample_time_s = self._csv_sample_times if env_ids is None else self._csv_sample_times[env_ids]
        final_time = float(self._csv_times[-1].item())
        if cfg.csv_loop and final_time > 0.0:
            sample_time_s = torch.remainder(sample_time_s, final_time)
        else:
            sample_time_s = torch.clamp(sample_time_s, min=0.0, max=final_time)

        upper_indices = torch.searchsorted(self._csv_times, sample_time_s, right=False)
        upper_indices = torch.clamp(upper_indices, max=self._csv_targets.shape[0] - 1)
        if not cfg.csv_interpolate or self._csv_targets.shape[0] == 1:
            return self._csv_targets[upper_indices]

        lower_indices = torch.clamp(upper_indices - 1, min=0)
        lower_times = self._csv_times[lower_indices]
        upper_times = self._csv_times[upper_indices]
        interval = upper_times - lower_times
        alpha = torch.where(
            interval > 1.0e-8,
            (sample_time_s - lower_times) / torch.clamp(interval, min=1.0e-8),
            torch.zeros_like(sample_time_s),
        )
        lower_targets = self._csv_targets[lower_indices]
        upper_targets = self._csv_targets[upper_indices]
        return torch.lerp(lower_targets, upper_targets, alpha.unsqueeze(-1))

    def _advance_csv_sample_times(self, motion_scale: float) -> None:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if motion_scale <= 0.0 or self._csv_sample_times.numel() == 0:
            return

        self._csv_sample_times.add_(float(motion_scale) * float(self.step_dt))
        final_time = float(self._csv_times[-1].item())
        if cfg.csv_loop and final_time > 0.0:
            self._csv_sample_times.remainder_(final_time)
        else:
            self._csv_sample_times.clamp_(min=0.0, max=final_time)

    def _initialize_csv_arm_joint_state(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        targets = self._sample_csv_targets(env_ids)
        targets = self._clip_upper_position_targets(targets, env_ids=env_ids)
        target_velocities = torch.zeros_like(targets)
        robot = self.scene["robot"]
        robot.write_joint_state_to_sim(
            targets,
            target_velocities,
            joint_ids=self._upper_asset_joint_ids,
            env_ids=env_ids,
        )

    def _reset_csv_start_offsets(self, env_ids: torch.Tensor) -> None:
        cfg = self._perturbation_cfg
        assert cfg is not None
        if env_ids.numel() == 0 or self._csv_times.numel() == 0:
            return

        final_time = float(self._csv_times[-1].item())
        if not cfg.csv_randomize_start_on_reset or final_time <= 0.0:
            start_offsets = torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device)
        elif cfg.csv_loop:
            start_offsets = torch.rand(env_ids.numel(), device=self.device) * final_time
        else:
            episode_horizon_s = float(self.max_episode_length) * float(self.step_dt)
            maximum_motion_scale = (
                max(float(cfg.csv_curriculum_motion_scale), 0.0)
                if cfg.csv_curriculum_enabled
                else 1.0
            )
            required_motion_horizon_s = episode_horizon_s * maximum_motion_scale
            max_start_time = max(
                final_time - required_motion_horizon_s - float(cfg.csv_end_margin_s),
                0.0,
            )
            start_offsets = torch.rand(env_ids.numel(), device=self.device) * max_start_time

        self._csv_start_time_offsets[env_ids] = start_offsets
        self._csv_sample_times[env_ids] = start_offsets

    def _sample_pose_targets(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._pose_set_targets.numel() == 0:
            raise RuntimeError("Upper-body pose-set perturbations are not initialized.")

        if self._pose_probabilities.numel() > 0:
            pose_indices = torch.multinomial(self._pose_probabilities, env_ids.numel(), replacement=True)
        else:
            pose_indices = torch.randint(self._pose_set_targets.shape[0], (env_ids.numel(),), device=self.device)
        self._active_pose_targets[env_ids] = self._pose_set_targets[pose_indices]

    def _as_env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        if env_ids_tensor.ndim == 0:
            env_ids_tensor = env_ids_tensor.unsqueeze(0)
        return env_ids_tensor
