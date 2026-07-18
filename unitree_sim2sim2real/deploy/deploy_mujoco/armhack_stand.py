"""Deterministic ArmHack Stand replay and reporting for the G1 MuJoCo runner.

The exported actor contains only the 96 -> 29 neural policy.  IsaacLab applies
the ArmHack arm targets outside the actor, then stores the composed 29-D raw
action as ``last_action``.  This module reproduces that deployment contract in
MuJoCo without changing the generic locomotion path when the feature is off.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARM_JOINT_NAMES = [
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

PAYLOAD_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def _quat_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _component_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": float(np.max(np.abs(values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def _norm_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
    }


class ArmHackStandReplay:
    """Apply deterministic arm targets and write MuJoCo Stand evaluation artifacts."""

    def __init__(self, config: dict[str, Any], policy_joint_names: list[str], default_angles: np.ndarray):
        self.config = config
        self.policy_joint_names = list(policy_joint_names)
        self.default_angles = np.asarray(default_angles, dtype=np.float64)
        self.action_scale = float(config["action_scale"])
        if self.action_scale <= 0.0:
            raise ValueError("ArmHack Stand requires action_scale > 0.")

        self.csv_path = Path(str(config["armhack_stand_csv_path"])).expanduser().resolve()
        self.manifest_path = Path(str(config["armhack_stand_manifest_path"])).expanduser().resolve()
        self.checkpoint_path = Path(str(config["armhack_stand_checkpoint_path"])).expanduser().resolve()
        self.report_path = Path(str(config["armhack_stand_report_path"])).expanduser().resolve()
        self.plot_path = self.report_path.with_name(f"{self.report_path.stem}__torso_world_6d.png")
        self.trace_path = self.report_path.with_name(f"{self.report_path.stem}__trace.csv")
        self.test_id = str(config.get("armhack_stand_test_id", "all"))
        self.payload_kg = float(config.get("armhack_stand_payload_kg", 0.0))
        if not 0.0 <= self.payload_kg <= 3.0:
            raise ValueError("ArmHack Stand payload must be within [0, 3] kg per wrist.")

        for path, label in (
            (self.csv_path, "test CSV"),
            (self.manifest_path, "manifest"),
            (self.checkpoint_path, "checkpoint"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"ArmHack Stand {label} does not exist: {path}")

        missing_policy_joints = sorted(set(ARM_JOINT_NAMES).difference(self.policy_joint_names))
        if missing_policy_joints:
            raise ValueError(f"ArmHack arm joints are absent from policy_joint_names: {missing_policy_joints}")
        self.arm_policy_indices = np.asarray(
            [self.policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64
        )
        self.balance_joint_names = [name for name in self.policy_joint_names if name not in set(ARM_JOINT_NAMES)]

        self.csv_times, self.csv_targets = self._load_csv()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != 5:
            raise ValueError("ArmHack Stand MuJoCo evaluation requires manifest schema_version=5.")
        if self.manifest.get("data_scope") != "arm_only_14_dof" or self.manifest.get("contains_full_body_state") is not False:
            raise ValueError("ArmHack Stand manifest must describe arm-only 14-DoF data.")
        self.timeline = self._load_timeline(float(self.csv_times[-1]))

        self.last_target = self.csv_targets[0].copy()
        self.last_target_time = 0.0
        self.torso_reference: np.ndarray | None = None
        self.sample_times: list[float] = []
        self.joint_samples: list[np.ndarray] = []
        self.arm_target_samples: list[np.ndarray] = []
        self.torso_delta_samples: list[np.ndarray] = []
        self.payload_report: dict[str, dict[str, float]] = {}

    @property
    def csv_duration_s(self) -> float:
        return float(self.csv_times[-1])

    def _load_csv(self) -> tuple[np.ndarray, np.ndarray]:
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "time_s" not in reader.fieldnames:
                raise ValueError(f"ArmHack Stand CSV has no time_s column: {self.csv_path}")
            missing = sorted(set(ARM_JOINT_NAMES).difference(reader.fieldnames))
            if missing:
                raise ValueError(f"ArmHack Stand CSV is missing arm joints: {missing}")
            unexpected_joint_columns = [
                name for name in reader.fieldnames if name != "time_s" and name not in ARM_JOINT_NAMES
            ]
            if unexpected_joint_columns:
                raise ValueError(
                    "ArmHack Stand MuJoCo CSV must contain only time_s and 14 arm joints; "
                    f"unexpected columns: {unexpected_joint_columns}"
                )
            times: list[float] = []
            targets: list[list[float]] = []
            for row in reader:
                times.append(float(row["time_s"]))
                targets.append([float(row[name]) for name in ARM_JOINT_NAMES])
        if not targets:
            raise ValueError(f"ArmHack Stand CSV contains no rows: {self.csv_path}")
        times_array = np.asarray(times, dtype=np.float64)
        targets_array = np.asarray(targets, dtype=np.float64)
        if not np.all(np.isfinite(times_array)) or not np.all(np.isfinite(targets_array)):
            raise ValueError("ArmHack Stand CSV contains non-finite values.")
        if np.any(np.diff(times_array) < 0.0):
            raise ValueError("ArmHack Stand CSV time_s must be monotonically non-decreasing.")
        times_array = times_array - times_array[0]
        return times_array, targets_array

    def _load_timeline(self, duration_s: float) -> list[dict[str, float | str]]:
        files = self.manifest.get("files", {})
        timeline: list[dict[str, Any]] = []
        collection_modes = {
            "all",
            "representative_poses",
            "synthesized_poses",
            "randomized_poses",
            "representative_trajectories",
            "synthesized_trajectories",
            "randomized_trajectories",
            "down_to_horizontal",
        }
        if self.test_id in collection_modes:
            metadata = files.get(self.test_id, {})
            timeline = list(metadata.get("detailed_timeline") or metadata.get("timeline") or [])
        elif "_item" in self.test_id:
            mode, item_text = self.test_id.rsplit("_item", 1)
            item_index = int(item_text) - 1
            mapping = {
                "representative_pose": ("representative_poses", "pose_id"),
                "synthesized_pose": ("synthesized_poses", "pose_id"),
                "randomized_pose": ("randomized_poses", "pose_id"),
                "representative_trajectory": ("representative_trajectories", "trajectory_id"),
                "synthesized_trajectory": ("synthesized_trajectories", "trajectory_id"),
                "randomized_trajectory": ("randomized_trajectories", "trajectory_id"),
            }
            if mode not in mapping:
                raise ValueError(f"Unsupported ArmHack Stand test id: {self.test_id}")
            collection, label_key = mapping[mode]
            items = self.manifest.get(collection, [])
            if item_index < 0 or item_index >= len(items):
                raise ValueError(f"ArmHack Stand test item is out of range: {self.test_id}")
            item = items[item_index]
            timeline = [
                {
                    "kind": "static_hold" if "pose" in mode else "trajectory",
                    "label": str(item[label_key]),
                    "start_s": 0.0,
                    "end_s": duration_s,
                }
            ]
        cleaned: list[dict[str, float | str]] = []
        for stage in timeline:
            start_s = max(float(stage.get("start_s", 0.0)), 0.0)
            end_s = min(float(stage.get("end_s", duration_s)), duration_s)
            if end_s > start_s:
                cleaned.append(
                    {
                        "kind": str(stage.get("kind", "stage")),
                        "label": str(stage.get("label", self.test_id)),
                        "start_s": start_s,
                        "end_s": end_s,
                    }
                )
        if not cleaned:
            cleaned = [{"kind": "stage", "label": self.test_id, "start_s": 0.0, "end_s": duration_s}]
        return cleaned

    def initialize_model_and_state(
        self,
        mujoco_module,
        model,
        data,
        qpos_addresses: dict[str, int],
        torso_body_id: int,
    ) -> None:
        for name, target in zip(ARM_JOINT_NAMES, self.csv_targets[0], strict=True):
            data.qpos[qpos_addresses[name]] = float(target)

        for body_name in PAYLOAD_BODY_NAMES:
            body_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"MuJoCo model has no ArmHack payload body: {body_name}")
            original_mass = float(model.body_mass[body_id])
            if original_mass <= 0.0:
                raise ValueError(f"MuJoCo payload body has non-positive mass: {body_name}={original_mass}")
            final_mass = original_mass + self.payload_kg
            inertia_scale = final_mass / original_mass
            model.body_mass[body_id] = final_mass
            model.body_inertia[body_id] *= inertia_scale
            self.payload_report[body_name] = {
                "original_mass_kg": original_mass,
                "added_mass_kg": self.payload_kg,
                "final_mass_kg": final_mass,
                "inertia_scale": inertia_scale,
            }

        mujoco_module.mj_forward(model, data)
        self.torso_reference = self._torso_pose(data, torso_body_id)

    @staticmethod
    def _torso_pose(data, torso_body_id: int) -> np.ndarray:
        position = np.asarray(data.xpos[torso_body_id], dtype=np.float64).copy()
        rpy = _quat_wxyz_to_rpy(np.asarray(data.xquat[torso_body_id], dtype=np.float64))
        return np.concatenate((position, rpy))

    def sample_target(self, time_s: float) -> np.ndarray:
        sample_time = float(np.clip(time_s, 0.0, self.csv_times[-1]))
        upper_index = int(np.searchsorted(self.csv_times, sample_time, side="left"))
        upper_index = min(upper_index, len(self.csv_times) - 1)
        lower_index = max(upper_index - 1, 0)
        lower_time = float(self.csv_times[lower_index])
        upper_time = float(self.csv_times[upper_index])
        if upper_time - lower_time <= 1.0e-10:
            return self.csv_targets[upper_index].copy()
        alpha = (sample_time - lower_time) / (upper_time - lower_time)
        return (1.0 - alpha) * self.csv_targets[lower_index] + alpha * self.csv_targets[upper_index]

    def compose_action(self, policy_action: np.ndarray, time_s: float) -> np.ndarray:
        composed = np.asarray(policy_action, dtype=np.float32).copy()
        target = self.sample_target(time_s)
        raw_arm_action = (target - self.default_angles[self.arm_policy_indices]) / self.action_scale
        composed[self.arm_policy_indices] = raw_arm_action.astype(np.float32)
        self.last_target = target
        self.last_target_time = float(time_s)
        return composed

    def record_control_sample(
        self,
        data,
        qpos_addresses: dict[str, int],
        torso_body_id: int,
        time_s: float,
    ) -> None:
        if self.torso_reference is None:
            raise RuntimeError("ArmHack Stand torso reference was not initialized.")
        joint_position = np.asarray(
            [data.qpos[qpos_addresses[name]] for name in self.policy_joint_names], dtype=np.float64
        )
        torso_delta = self._torso_pose(data, torso_body_id) - self.torso_reference
        torso_delta[3:] = _wrap_to_pi(torso_delta[3:])
        self.sample_times.append(float(time_s))
        self.joint_samples.append(joint_position)
        self.arm_target_samples.append(self.last_target.copy())
        self.torso_delta_samples.append(torso_delta)

    @staticmethod
    def _stage_color(stage: dict[str, float | str]) -> str:
        kind = str(stage["kind"])
        label = str(stage["label"])
        if "transition" in kind or "bridge" in kind:
            return "#B0BEC5"
        if label == "arms_down_hold":
            return "#2F4B7C"
        if label == "arms_forward_horizontal_hold":
            return "#00A087"
        if label.startswith("representative_pose"):
            return "#4E79A7"
        if label.startswith("synth_pose"):
            return "#B07AA1"
        if label.startswith("randomized_pose"):
            return "#76B7B2"
        if label.startswith("representative_trajectory"):
            return "#59A14F"
        if label.startswith("synth_trajectory"):
            return "#F28E2B"
        if label.startswith("randomized_trajectory"):
            return "#EDC948"
        return "#BAB0AC"

    @staticmethod
    def _short_stage_label(stage: dict[str, float | str]) -> str:
        kind = str(stage["kind"])
        label = str(stage["label"])
        if "transition" in kind:
            return "D→H" if label == "arms_down_to_forward_horizontal" else "T"
        if "bridge" in kind:
            return "B"
        if label == "arms_down_hold":
            return "AD"
        if label == "arms_forward_horizontal_hold":
            return "AH"
        replacements = (
            ("representative_pose_", "RP"),
            ("synth_pose_", "SP"),
            ("randomized_pose_", "GP"),
            ("representative_trajectory_", "RT"),
            ("synth_trajectory_", "ST"),
            ("randomized_trajectory_", "GT"),
        )
        for prefix, short_prefix in replacements:
            if label.startswith(prefix):
                return short_prefix + label.removeprefix(prefix)
        return label[:8]

    def _stage_at(self, time_s: float) -> tuple[str, str]:
        for stage in self.timeline:
            if float(stage["start_s"]) <= time_s < float(stage["end_s"]) + 1.0e-9:
                return str(stage["kind"]), str(stage["label"])
        stage = self.timeline[-1]
        return str(stage["kind"]), str(stage["label"])

    def _write_trace(
        self,
        joint_samples: np.ndarray,
        target_samples: np.ndarray,
        torso_delta: np.ndarray,
    ) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["time_s", "stage_kind", "stage_label"]
            + [f"actual_{name}" for name in self.policy_joint_names]
            + [f"target_{name}" for name in ARM_JOINT_NAMES]
            + ["delta_x_w", "delta_y_w", "delta_z_w", "delta_roll_w", "delta_pitch_w", "delta_yaw_w"]
        )
        with self.trace_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            for sample_index, time_s in enumerate(self.sample_times):
                stage_kind, stage_label = self._stage_at(time_s)
                writer.writerow(
                    [f"{time_s:.8f}", stage_kind, stage_label]
                    + [f"{value:.9f}" for value in joint_samples[sample_index]]
                    + [f"{value:.9f}" for value in target_samples[sample_index]]
                    + [f"{value:.9f}" for value in torso_delta[sample_index]]
                )

    def _write_plot(self, torso_delta: np.ndarray) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        times = np.asarray(self.sample_times, dtype=np.float64)
        figure = plt.figure(figsize=(18.0, 10.0), layout="constrained")
        grid = figure.add_gridspec(3, 1, height_ratios=(3.0, 3.0, 1.15), hspace=0.16)
        position_axis = figure.add_subplot(grid[0, 0])
        rotation_axis = figure.add_subplot(grid[1, 0], sharex=position_axis)
        stage_axis = figure.add_subplot(grid[2, 0], sharex=position_axis)
        for index, (label, color) in enumerate(zip(("dx_w", "dy_w", "dz_w"), ("#1F77B4", "#D62728", "#2CA02C"), strict=True)):
            position_axis.plot(times, torso_delta[:, index], label=label, color=color, linewidth=1.15)
        for index, (label, color) in enumerate(zip(("droll_w", "dpitch_w", "dyaw_w"), ("#9467BD", "#FF7F0E", "#17BECF"), strict=True), start=3):
            rotation_axis.plot(times, torso_delta[:, index], label=label, color=color, linewidth=1.15)
        displayed_end = float(times[-1]) if len(times) else self.csv_duration_s
        for stage in self.timeline:
            start_s = float(stage["start_s"])
            end_s = min(float(stage["end_s"]), displayed_end)
            if end_s <= start_s:
                continue
            color = self._stage_color(stage)
            position_axis.axvspan(start_s, end_s, color=color, alpha=0.055, linewidth=0.0)
            rotation_axis.axvspan(start_s, end_s, color=color, alpha=0.055, linewidth=0.0)
            stage_axis.axvspan(
                start_s,
                end_s,
                facecolor=color,
                alpha=0.88,
                linewidth=0.4,
                edgecolor="white",
            )
            stage_axis.text(
                0.5 * (start_s + end_s),
                0.5,
                self._short_stage_label(stage),
                ha="center",
                va="center",
                rotation=90 if end_s - start_s < 2.5 else 0,
                fontsize=6.5,
                color="#111111",
                clip_on=True,
            )
        for axis in (position_axis, rotation_axis):
            axis.axhline(0.0, color="#666666", linewidth=0.6)
            axis.grid(True, alpha=0.22)
            axis.legend(loc="upper right", ncol=3)
        position_axis.set_ylabel("World translation displacement (m)")
        rotation_axis.set_ylabel("World RPY displacement (rad)")
        rotation_axis.set_xlabel("Test time (s)")
        stage_axis.set_ylim(0.0, 1.0)
        stage_axis.set_yticks([])
        stage_axis.set_xlabel("Test stage timeline (s)")
        stage_axis.set_xlim(0.0, max(displayed_end, 1.0e-6))
        figure.suptitle(f"ArmHack Stand MuJoCo torso world-frame 6D displacement — {self.test_id}", fontsize=16, fontweight="bold")
        legend = [
            Patch(facecolor="#4E79A7", label="RP"),
            Patch(facecolor="#B07AA1", label="SP"),
            Patch(facecolor="#76B7B2", label="GP"),
            Patch(facecolor="#59A14F", label="RT"),
            Patch(facecolor="#F28E2B", label="ST"),
            Patch(facecolor="#EDC948", label="GT"),
            Patch(facecolor="#B0BEC5", label="transition / bridge"),
            Patch(facecolor="#2F4B7C", label="AD"),
            Patch(facecolor="#00A087", label="AH"),
        ]
        figure.legend(handles=legend, loc="lower center", ncol=9, frameon=False)
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

    def finalize(self, generic_report: dict[str, Any], sim_time: float, control_dt: float) -> dict[str, Any]:
        if not self.joint_samples or not self.torso_delta_samples:
            raise RuntimeError("ArmHack Stand MuJoCo report has no control samples.")
        joint_samples = np.stack(self.joint_samples)
        target_samples = np.stack(self.arm_target_samples)
        torso_delta = np.stack(self.torso_delta_samples)
        actual_arm = joint_samples[:, self.arm_policy_indices]
        arm_error = actual_arm - target_samples
        abs_step_delta = np.mean(np.abs(np.diff(joint_samples, axis=0)), axis=0) if len(joint_samples) > 1 else np.zeros(joint_samples.shape[1])

        joint_statistics: dict[str, dict[str, float | str]] = {}
        for index, name in enumerate(self.policy_joint_names):
            stats = _component_stats(joint_samples[:, index])
            stats["mean_abs_step_delta_rad"] = float(abs_step_delta[index])
            stats["group"] = "arm_input_joint" if name in ARM_JOINT_NAMES else "balance_policy_joint"
            joint_statistics[name] = stats

        arm_tracking: dict[str, dict[str, float]] = {}
        for index, name in enumerate(ARM_JOINT_NAMES):
            error = arm_error[:, index]
            arm_tracking[name] = {
                "mean_abs_error_rad": float(np.mean(np.abs(error))),
                "rms_error_rad": float(np.sqrt(np.mean(np.square(error)))),
                "max_abs_error_rad": float(np.max(np.abs(error))),
            }

        component_names = ("delta_x_w", "delta_y_w", "delta_z_w", "delta_roll_w", "delta_pitch_w", "delta_yaw_w")
        torso_statistics = {name: _component_stats(torso_delta[:, index]) for index, name in enumerate(component_names)}
        torso_norms = {
            "horizontal_translation_norm": np.linalg.norm(torso_delta[:, :2], axis=1),
            "translation_3d_norm": np.linalg.norm(torso_delta[:, :3], axis=1),
            "rpy_displacement_norm": np.linalg.norm(torso_delta[:, 3:], axis=1),
        }
        torso_norm_statistics = {name: _norm_stats(values) for name, values in torso_norms.items()}
        complete = float(self.sample_times[-1]) >= self.csv_duration_s - 0.5 * float(control_dt)
        healthy = bool(generic_report.get("health", {}).get("healthy", False))
        checkpoint_sha = _sha256(self.checkpoint_path)
        policy_path = Path(str(self.config["policy_path"])).expanduser().resolve()

        result: dict[str, Any] = {
            "simulator": "MuJoCo",
            "test_id": self.test_id,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "policy_path": str(policy_path),
            "policy_sha256": _sha256(policy_path),
            "test_csv": str(self.csv_path),
            "test_csv_sha256": _sha256(self.csv_path),
            "manifest": str(self.manifest_path),
            "manifest_sha256": _sha256(self.manifest_path),
            "manifest_schema_version": int(self.manifest["schema_version"]),
            "control_dt_s": float(control_dt),
            "control_samples": len(self.sample_times),
            "sim_time_s": float(sim_time),
            "csv_duration_s": self.csv_duration_s,
            "complete_csv_playback": complete,
            "healthy": healthy,
            "health_failure_count": 0 if healthy else 1,
            "payload_kg_per_wrist": self.payload_kg,
            "payload_bodies": self.payload_report,
            "joint_statistics": joint_statistics,
            "arm_tracking": arm_tracking,
            "arm_tracking_overall": {
                "mean_abs_error_rad": float(np.mean(np.abs(arm_error))),
                "rms_error_rad": float(np.sqrt(np.mean(np.square(arm_error)))),
                "max_abs_error_rad": float(np.max(np.abs(arm_error))),
            },
            "torso_world_6d": torso_statistics,
            "torso_world_norms": torso_norm_statistics,
            "timeline": self.timeline,
            "report_path": str(self.report_path),
            "plot_path": str(self.plot_path),
            "trace_path": str(self.trace_path),
        }

        self._write_trace(joint_samples, target_samples, torso_delta)
        self._write_plot(torso_delta)
        self._write_markdown(result, generic_report)
        return result

    def _write_markdown(self, result: dict[str, Any], generic_report: dict[str, Any]) -> None:
        health = generic_report.get("health", {})
        important = generic_report.get("important_metrics", {})
        torso = result["torso_world_6d"]
        norms = result["torso_world_norms"]
        lines = [
            "# ArmHack Stand MuJoCo sim2sim 测试报告",
            "",
            "## 测试身份",
            "",
            f"- simulator：`MuJoCo`",
            f"- checkpoint：`{result['checkpoint']}`",
            f"- checkpoint SHA-256：`{result['checkpoint_sha256']}`",
            f"- exported TorchScript：`{result['policy_path']}`",
            f"- exported policy SHA-256：`{result['policy_sha256']}`",
            f"- 测试项：`{result['test_id']}`",
            f"- 测试 CSV：`{result['test_csv']}`",
            f"- 测试 CSV SHA-256：`{result['test_csv_sha256']}`",
            f"- manifest schema：`v{result['manifest_schema_version']}`，SHA-256 `{result['manifest_sha256']}`",
            f"- 控制样本：`{result['control_samples']}`，控制周期 `{result['control_dt_s']:.6f} s`",
            f"- CSV 时长：`{result['csv_duration_s']:.3f} s`，完整播放：`{result['complete_csv_playback']}`",
            f"- 左/右 wrist-yaw 末端附加质量：各 `{result['payload_kg_per_wrist']:.3f} kg`",
            f"- MuJoCo health：`{result['healthy']}`，health failure count：`{result['health_failure_count']}`，fall time：`{health.get('fall_time')}`",
            "- 输入范围：CSV 只覆盖 14 个双臂关节；15 个腰腿关节仍由 policy 控制。覆盖后的 29 维 raw action 会写回下一帧 `last_action`。",
            "",
            "## 结论",
            "",
            f"- 完整稳定通过：`{bool(result['complete_csv_playback'] and result['healthy'])}`。判据为完整播放 schema v5 CSV 且 MuJoCo health 全程有效。",
            f"- 最低 root 高度：`{float(health.get('min_root_height', 0.0)):.6f} m`；最大绝对 roll/pitch：`{float(health.get('max_abs_roll', 0.0)):.6f} / {float(health.get('max_abs_pitch', 0.0)):.6f} rad`。",
            f"- torso 水平位移 RMS/最大值：`{norms['horizontal_translation_norm']['rms']:.6f} / {norms['horizontal_translation_norm']['max']:.6f} m`。",
            f"- torso pitch 位移 RMS/最大绝对值：`{torso['delta_pitch_w']['rms']:.6f} / {torso['delta_pitch_w']['max_abs']:.6f} rad`。",
            f"- 双臂实际跟踪总体 MAE/RMS/最大误差：`{result['arm_tracking_overall']['mean_abs_error_rad']:.6f} / {result['arm_tracking_overall']['rms_error_rad']:.6f} / {result['arm_tracking_overall']['max_abs_error_rad']:.6f} rad`。",
            "",
            "## 每关节实际波动",
            "",
            "| 关节 | 分组 | 平均逐步波动 rad/step | 实际角均值 rad | 标准差 rad | 极差 rad |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name in self.policy_joint_names:
            stats = result["joint_statistics"][name]
            group = "双臂输入关节" if stats["group"] == "arm_input_joint" else "平衡策略关节"
            lines.append(
                f"| `{name}` | {group} | {stats['mean_abs_step_delta_rad']:.8f} | "
                f"{stats['mean']:.8f} | {stats['std']:.8f} | {stats['range']:.8f} |"
            )
        lines += [
            "",
            "## 双臂目标跟踪误差",
            "",
            "| 关节 | MAE rad | RMS rad | 最大绝对误差 rad |",
            "|---|---:|---:|---:|",
        ]
        for name in ARM_JOINT_NAMES:
            stats = result["arm_tracking"][name]
            lines.append(
                f"| `{name}` | {stats['mean_abs_error_rad']:.8f} | {stats['rms_error_rad']:.8f} | {stats['max_abs_error_rad']:.8f} |"
            )
        lines += [
            "",
            "## 躯干世界坐标系 6D 位移",
            "",
            "| 分量 | 单位 | 有符号均值 | 绝对值均值 | 标准差 | RMS | 最大绝对值 | 极差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, unit in zip(("delta_x_w", "delta_y_w", "delta_z_w", "delta_roll_w", "delta_pitch_w", "delta_yaw_w"), ("m", "m", "m", "rad", "rad", "rad"), strict=True):
            stats = torso[name]
            lines.append(
                f"| `{name}` | {unit} | {stats['mean']:.8f} | {stats['mean_abs']:.8f} | "
                f"{stats['std']:.8f} | {stats['rms']:.8f} | {stats['max_abs']:.8f} | {stats['range']:.8f} |"
            )
        lines += [
            "",
            "| 综合位移 | 单位 | 均值 | 标准差 | RMS | 最大值 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, unit in (("horizontal_translation_norm", "m"), ("translation_3d_norm", "m"), ("rpy_displacement_norm", "rad")):
            stats = norms[name]
            lines.append(f"| `{name}` | {unit} | {stats['mean']:.8f} | {stats['std']:.8f} | {stats['rms']:.8f} | {stats['max']:.8f} |")
        lines += [
            "",
            "### 6D 曲线与阶段",
            "",
            f"![ArmHack Stand MuJoCo torso world-frame 6D displacement]({self.plot_path.name})",
            "",
            "| 开始 s | 结束 s | 类型 | 姿态或轨迹阶段 |",
            "|---:|---:|---|---|",
        ]
        for stage in self.timeline:
            lines.append(
                f"| {float(stage['start_s']):.3f} | {float(stage['end_s']):.3f} | "
                f"`{stage['kind']}` | `{stage['label']}` |"
            )
        lines += ["", "## MuJoCo Important Metrics", "", "| 指标 | 均值 |", "|---|---:|"]
        for name in sorted(important):
            lines.append(f"| `{name}` | {float(important[name]):.8f} |")
        lines += [
            "",
            "## 输出文件",
            "",
            f"- JSON：`{self.config.get('metrics_path', '')}`",
            f"- 逐帧 CSV：`{self.trace_path}`",
            f"- 6D PNG：`{self.plot_path}`",
            "",
            "## 结论边界",
            "",
            "该报告验证的是当前 MuJoCo XML、PD 参数与固定 schema v5 输入下的 sim2sim 行为。它不替代 IsaacLab 报告，也不代表真实机器人性能。若 `complete_csv_playback=False` 或 `healthy=False`，不能判定该测试项完整稳定通过。",
            "",
        ]
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")
