#!/usr/bin/env python3
"""Build a command-balanced directional G1 AMP motion dataset.

The builder consumes Lab/GMR-format 50 Hz G1 motion pickles, estimates
body-frame velocity commands from root motion, cuts short homogeneous segments,
filters obvious quality failures, and writes a new dataset with per-frame task
labels plus sampling metadata.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
import shutil
import sys
import types
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def _install_numpy_pickle_shim() -> None:
    """Allow numpy>=2 pickles to load in numpy<2 runtimes."""

    if hasattr(np, "_core"):
        return
    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        full_name = f"numpy._core.{submodule}"
        if full_name not in sys.modules:
            try:
                sys.modules[full_name] = importlib.import_module(f"numpy.core.{submodule}")
            except ImportError:
                pass


_install_numpy_pickle_shim()


REPO_ROOT = Path(__file__).resolve().parents[2]
AMP_ROOT = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp"
DEFAULT_SOURCE_DIRS = [
    AMP_ROOT / "cmu_walk_washed_50hz",
    AMP_ROOT / "cmu_walk_50hz_task_core",
    AMP_ROOT / "accad_g1used_50hz",
]
DEFAULT_AUDIT_DIR = AMP_ROOT / "accad_walk_50hz"
DEFAULT_OUTPUT_DIR = AMP_ROOT / "command_balanced_directional_50hz"

MODE_ID_BY_NAME = {
    "forward_slow": 0,
    "forward_normal": 1,
    "backward": 2,
    "lateral_left": 3,
    "lateral_right": 4,
    "turn_left": 5,
    "turn_right": 6,
    "stand": 7,
}
MODE_NAME_BY_ID = {value: key for key, value in MODE_ID_BY_NAME.items()}
TARGET_MODE_WEIGHTS = {
    "forward_slow": 0.16,
    "forward_normal": 0.22,
    "backward": 0.18,
    "lateral_left": 0.11,
    "lateral_right": 0.11,
    "turn_left": 0.08,
    "turn_right": 0.08,
    "stand": 0.06,
}
COMMAND_RANGES = {
    "forward_slow": {"lin_vel_x": [0.20, 0.55], "lin_vel_y": [-0.15, 0.15], "ang_vel_z": [-0.35, 0.35]},
    "forward_normal": {"lin_vel_x": [0.55, 1.05], "lin_vel_y": [-0.20, 0.20], "ang_vel_z": [-0.35, 0.35]},
    "backward": {"lin_vel_x": [-0.65, -0.15], "lin_vel_y": [-0.20, 0.20], "ang_vel_z": [-0.35, 0.35]},
    "lateral_left": {"lin_vel_x": [-0.30, 0.30], "lin_vel_y": [0.15, 0.45], "ang_vel_z": [-0.35, 0.35]},
    "lateral_right": {"lin_vel_x": [-0.30, 0.30], "lin_vel_y": [-0.45, -0.15], "ang_vel_z": [-0.35, 0.35]},
    "turn_left": {"lin_vel_x": [-0.20, 0.90], "lin_vel_y": [-0.25, 0.25], "ang_vel_z": [0.25, 1.00]},
    "turn_right": {"lin_vel_x": [-0.20, 0.90], "lin_vel_y": [-0.25, 0.25], "ang_vel_z": [-1.00, -0.25]},
    "stand": {"lin_vel_x": [0.0, 0.0], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
}

ARM_JOINT_NAMES = [
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
REQUIRED_KEYS = {
    "fps",
    "root_pos",
    "root_rot",
    "dof_pos",
    "dof_names",
    "local_body_pos",
    "link_body_list",
}
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
WRIST_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
TORSO_BODY_NAME = "torso_link"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_motion(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        motion = pickle.load(file)
    if not isinstance(motion, dict):
        raise ValueError(f"{path} did not contain a dictionary.")
    return motion


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    if values.ndim == 1:
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")
        return smoothed[: values.shape[0]]
    return np.stack([_moving_average(values[:, dim], window) for dim in range(values.shape[1])], axis=1)


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-8)


def _quat_yaw_xyzw(root_rot: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(root_rot)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.unwrap(yaw)


def _quat_pitch_xyzw(root_rot: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(root_rot)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinp = 2.0 * (w * y - z * x)
    return np.arcsin(np.clip(sinp, -1.0, 1.0))


def _quat_apply_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + q_w * t + np.cross(q_xyz, t)


def _estimate_commands(motion: dict[str, Any], smooth_seconds: float) -> dict[str, np.ndarray]:
    fps = float(motion["fps"])
    dt = 1.0 / fps
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    smooth_window = max(1, int(round(smooth_seconds * fps)))

    global_velocity_xy = _moving_average(np.gradient(root_pos[:, :2], dt, axis=0), smooth_window)
    yaw = _quat_yaw_xyzw(root_rot)
    yaw_rate = _moving_average(np.gradient(yaw, dt), smooth_window)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_vx = cos_yaw * global_velocity_xy[:, 0] + sin_yaw * global_velocity_xy[:, 1]
    local_vy = -sin_yaw * global_velocity_xy[:, 0] + cos_yaw * global_velocity_xy[:, 1]
    local_velocity_xy = np.stack([local_vx, local_vy], axis=1)
    command = np.column_stack([local_velocity_xy, yaw_rate])
    return {
        "time": np.arange(root_pos.shape[0], dtype=np.float32) / fps,
        "command": command.astype(np.float32),
        "global_velocity_xy": global_velocity_xy.astype(np.float32),
        "local_velocity_xy": local_velocity_xy.astype(np.float32),
        "speed_xy": np.linalg.norm(local_velocity_xy, axis=1).astype(np.float32),
        "yaw_rate": yaw_rate.astype(np.float32),
        "yaw": yaw.astype(np.float32),
    }


def _classify_mode(mean_command: np.ndarray, mean_speed: float) -> str | None:
    vx, vy, wz = (float(value) for value in mean_command)
    abs_wz = abs(wz)
    if mean_speed < 0.12 and abs_wz < 0.15:
        return "stand"
    if 0.15 <= mean_speed <= 0.90 and 0.25 <= abs_wz <= 1.00:
        return "turn_left" if wz > 0.0 else "turn_right"
    if 0.20 <= vx < 0.55 and abs(vy) < 0.15 and abs_wz < 0.35:
        return "forward_slow"
    if 0.55 <= vx <= 1.05 and abs(vy) < 0.20 and abs_wz < 0.35:
        return "forward_normal"
    if -0.65 <= vx <= -0.15 and abs(vy) < 0.20 and abs_wz < 0.35:
        return "backward"
    if abs(vx) < 0.30 and abs_wz < 0.35:
        if vy > 0.15:
            return "lateral_left"
        if vy < -0.15:
            return "lateral_right"
    return None


def _validate_motion_format(motion: dict[str, Any], path: Path) -> None:
    missing = sorted(REQUIRED_KEYS.difference(motion))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    fps = float(motion["fps"])
    if abs(fps - 50.0) > 1.0e-6:
        raise ValueError(f"{path} has fps={fps}, expected 50.")
    root_pos = np.asarray(motion["root_pos"])
    root_rot = np.asarray(motion["root_rot"])
    dof_pos = np.asarray(motion["dof_pos"])
    local_body_pos = np.asarray(motion["local_body_pos"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path} root_pos shape is invalid: {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(f"{path} root_rot shape is invalid: {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[0] != root_pos.shape[0] or dof_pos.shape[1] != 29:
        raise ValueError(f"{path} dof_pos shape is invalid: {dof_pos.shape}")
    if local_body_pos.ndim != 3 or local_body_pos.shape[0] != root_pos.shape[0] or local_body_pos.shape[2] != 3:
        raise ValueError(f"{path} local_body_pos shape is invalid: {local_body_pos.shape}")
    dof_names = list(motion["dof_names"])
    link_body_list = list(motion["link_body_list"])
    if len(dof_names) != dof_pos.shape[1]:
        raise ValueError(f"{path} dof_names length does not match dof_pos width.")
    if len(link_body_list) != local_body_pos.shape[1]:
        raise ValueError(f"{path} link_body_list length does not match local_body_pos width.")
    for name in FOOT_BODY_NAMES + WRIST_BODY_NAMES + [TORSO_BODY_NAME]:
        if name not in link_body_list:
            raise ValueError(f"{path} link_body_list does not contain {name}.")
    for name in ARM_JOINT_NAMES:
        if name not in dof_names:
            raise ValueError(f"{path} dof_names does not contain {name}.")
    for key in ("root_pos", "root_rot", "dof_pos", "local_body_pos"):
        if not np.all(np.isfinite(np.asarray(motion[key]))):
            raise ValueError(f"{path} contains non-finite values in {key}.")
    quat_norm = np.linalg.norm(root_rot.astype(np.float64), axis=1)
    if float(np.max(np.abs(quat_norm - 1.0))) > 5.0e-3:
        raise ValueError(f"{path} root_rot is not unit-normalized.")


def _world_body_positions(motion: dict[str, Any]) -> np.ndarray:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    local_body_pos = np.asarray(motion["local_body_pos"], dtype=np.float64)
    return root_pos[:, None, :] + _quat_apply_xyzw(root_rot[:, None, :], local_body_pos)


def _foot_metrics(motion: dict[str, Any], start: int, end: int) -> dict[str, float]:
    fps = float(motion["fps"])
    dt = 1.0 / fps
    links = list(motion["link_body_list"])
    foot_ids = [links.index(name) for name in FOOT_BODY_NAMES]
    foot_pos = _world_body_positions(motion)[:, foot_ids, :]
    foot_vel = np.gradient(foot_pos, dt, axis=0)

    segment_pos = foot_pos[start:end]
    segment_vel = foot_vel[start:end]
    ground_z = np.percentile(foot_pos[:, :, 2], 5.0, axis=0)
    contact = (segment_pos[:, :, 2] <= ground_z[None, :] + 0.045) & (np.abs(segment_vel[:, :, 2]) < 0.70)
    contact_count = np.sum(contact, axis=1)
    contact_speed = np.linalg.norm(segment_vel[:, :, :2], axis=2)
    contact_speeds = contact_speed[contact]
    if contact_speeds.size == 0:
        contact_slide_mean = 0.0
        contact_slide_p95 = 0.0
    else:
        contact_slide_mean = float(np.mean(contact_speeds))
        contact_slide_p95 = float(np.percentile(contact_speeds, 95.0))
    return {
        "single_stance_fraction": float(np.mean(contact_count == 1)),
        "double_stance_fraction": float(np.mean(contact_count == 2)),
        "double_air_fraction": float(np.mean(contact_count == 0)),
        "left_contact_fraction": float(np.mean(contact[:, 0])),
        "right_contact_fraction": float(np.mean(contact[:, 1])),
        "contact_slide_mean": contact_slide_mean,
        "contact_slide_p95": contact_slide_p95,
    }


def _arm_metrics(motion: dict[str, Any], start: int, end: int) -> dict[str, float | str]:
    fps = float(motion["fps"])
    dof_names = list(motion["dof_names"])
    links = list(motion["link_body_list"])
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)[start:end]
    local_body_pos = np.asarray(motion["local_body_pos"], dtype=np.float64)[start:end]

    left_shoulder_pitch = dof_pos[:, dof_names.index("left_shoulder_pitch_joint")]
    right_shoulder_pitch = dof_pos[:, dof_names.index("right_shoulder_pitch_joint")]
    left_amp = float(np.percentile(left_shoulder_pitch, 95.0) - np.percentile(left_shoulder_pitch, 5.0))
    right_amp = float(np.percentile(right_shoulder_pitch, 95.0) - np.percentile(right_shoulder_pitch, 5.0))
    shoulder_pitch_amp = 0.5 * (left_amp + right_amp)

    torso_id = links.index(TORSO_BODY_NAME)
    wrist_ids = [links.index(name) for name in WRIST_BODY_NAMES]
    left_wrist_x = local_body_pos[:, wrist_ids[0], 0] - local_body_pos[:, torso_id, 0]
    right_wrist_x = local_body_pos[:, wrist_ids[1], 0] - local_body_pos[:, torso_id, 0]
    wrist_rel_x_amp = 0.5 * (
        float(np.percentile(left_wrist_x, 95.0) - np.percentile(left_wrist_x, 5.0))
        + float(np.percentile(right_wrist_x, 95.0) - np.percentile(right_wrist_x, 5.0))
    )

    arm_ids = [dof_names.index(name) for name in ARM_JOINT_NAMES]
    if end - start > 1:
        arm_vel = np.diff(dof_pos[:, arm_ids], axis=0) * fps
        arm_vel_p99 = float(np.percentile(np.abs(arm_vel), 99.0))
    else:
        arm_vel_p99 = 0.0
    if shoulder_pitch_amp < 0.06 and wrist_rel_x_amp < 0.02:
        arm_quality = "dead"
    elif shoulder_pitch_amp < 0.12 or wrist_rel_x_amp < 0.05:
        arm_quality = "low"
    else:
        arm_quality = "good"
    return {
        "shoulder_pitch_amp": shoulder_pitch_amp,
        "left_shoulder_pitch_amp": left_amp,
        "right_shoulder_pitch_amp": right_amp,
        "wrist_rel_x_amp": wrist_rel_x_amp,
        "arm_joint_vel_p99": arm_vel_p99,
        "arm_quality": arm_quality,
    }


def _root_metrics(motion: dict[str, Any], start: int, end: int) -> dict[str, float]:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)[start:end]
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)[start:end]
    pitch = _quat_pitch_xyzw(root_rot)
    return {
        "root_height_mean": float(np.mean(root_pos[:, 2])),
        "root_height_range": float(np.ptp(root_pos[:, 2])),
        "root_pitch_abs_p95": float(np.percentile(np.abs(pitch), 95.0)),
    }


def _quality_and_reject_reason(motion: dict[str, Any], start: int, end: int, mode_name: str) -> tuple[dict[str, Any], str | None]:
    quality: dict[str, Any] = {}
    quality.update(_foot_metrics(motion, start, end))
    quality.update(_arm_metrics(motion, start, end))
    quality.update(_root_metrics(motion, start, end))

    locomotion_mode = mode_name != "stand"
    if quality["double_air_fraction"] > 0.30:
        return quality, "double_air_fraction_gt_0p30"
    if mode_name in ("backward", "lateral_left", "lateral_right") and quality["double_air_fraction"] > 0.20:
        return quality, "directional_double_air_fraction_gt_0p20"
    if locomotion_mode and quality["contact_slide_p95"] > 2.50:
        return quality, "contact_slide_p95_gt_2p50"
    if quality["root_height_range"] > 0.35:
        return quality, "root_height_range_gt_0p35"
    if quality["root_pitch_abs_p95"] > 0.75:
        return quality, "root_pitch_abs_p95_gt_0p75"
    if mode_name not in ("stand", "lateral_left", "lateral_right") and quality["arm_quality"] == "dead":
        return quality, "dead_arm_swing"
    if quality["arm_joint_vel_p99"] > 20.0:
        return quality, "arm_joint_vel_p99_gt_20"
    return quality, None


def _segment_starts(num_frames: int, window_frames: int, stride_frames: int, min_frames: int) -> list[int]:
    if num_frames < min_frames:
        return []
    if num_frames <= window_frames:
        return [0]
    starts = list(range(0, max(num_frames - window_frames + 1, 1), stride_frames))
    last_start = num_frames - window_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def _slice_motion(motion: dict[str, Any], start: int, end: int, task: dict[str, Any], cleaning: dict[str, Any]) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    frame_count = end - start
    for key, value in motion.items():
        if key in ("task", "task_scope", "cleaning"):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (np.asarray(motion["root_pos"]).shape[0],):
            sliced[key] = value[start:end].copy()
        else:
            sliced[key] = deepcopy(value)
    root_pos = np.asarray(sliced["root_pos"], dtype=np.float32).copy()
    root_pos[:, :2] -= root_pos[0:1, :2]
    sliced["root_pos"] = root_pos
    sliced["fps"] = int(motion["fps"])
    sliced["task"] = task
    sliced["task_scope"] = {
        "command_frame": "robot_base_local",
        "mode_name": task["mode_name"],
        "mode_id": int(task["mode_id"][0]) if frame_count > 0 else -1,
        "window_seconds": frame_count / float(motion["fps"]),
        "label_version": "command_balanced_directional_v1",
    }
    sliced["cleaning"] = cleaning
    return sliced


def _build_segments_for_motion(
    source_name: str,
    motion_path: Path,
    motion: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    fps = float(motion["fps"])
    command_data = _estimate_commands(motion, args.smooth_seconds)
    num_frames = int(np.asarray(motion["root_pos"]).shape[0])
    min_frames = int(math.ceil(args.min_window_seconds * fps))
    window_frames = min(int(round(args.window_seconds * fps)), int(round(args.max_window_seconds * fps)))
    stride_frames = max(1, int(round(args.stride_seconds * fps)))

    accepted: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    starts = _segment_starts(num_frames, window_frames, stride_frames, min_frames)
    for segment_index, start in enumerate(starts):
        end = min(start + window_frames, num_frames)
        if end - start < min_frames:
            continue
        command = command_data["command"][start:end]
        mean_command = np.mean(command, axis=0)
        mean_speed = float(np.mean(command_data["speed_xy"][start:end]))
        mode_name = _classify_mode(mean_command, mean_speed)
        if mode_name is None:
            rejected.append(
                {
                    "source_dataset": source_name,
                    "source_file": motion_path.name,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "reason": "no_matching_mode",
                    "mean_vx": float(mean_command[0]),
                    "mean_vy": float(mean_command[1]),
                    "mean_wz": float(mean_command[2]),
                    "mean_speed_xy": mean_speed,
                }
            )
            continue

        quality, reject_reason = _quality_and_reject_reason(motion, start, end, mode_name)
        if reject_reason is not None:
            rejected.append(
                {
                    "source_dataset": source_name,
                    "source_file": motion_path.name,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "mode": mode_name,
                    "reason": reject_reason,
                    "quality": quality,
                }
            )
            continue

        mode_id = MODE_ID_BY_NAME[mode_name]
        frame_count = end - start
        output_stem = f"{source_name}_{motion_path.stem}_{segment_index:04d}_{mode_name}"
        task = {
            "time": np.arange(frame_count, dtype=np.float32) / fps,
            "command": command.astype(np.float32),
            "global_velocity_xy": command_data["global_velocity_xy"][start:end].astype(np.float32),
            "local_velocity_xy": command_data["local_velocity_xy"][start:end].astype(np.float32),
            "speed_xy": command_data["speed_xy"][start:end].astype(np.float32),
            "yaw_rate": command_data["yaw_rate"][start:end].astype(np.float32),
            "mode_id": np.full(frame_count, mode_id, dtype=np.int16),
            "mode_name": mode_name,
            "mode_name_by_id": MODE_NAME_BY_ID,
            "source_dataset": source_name,
            "source_file": motion_path.name,
            "source_frame_range": [int(start), int(end)],
            "quality": quality,
        }
        cleaning = {
            "passed": True,
            "reject_reason": "",
            "label_version": "command_balanced_directional_v1",
            "source_dataset": source_name,
            "source_file": motion_path.name,
        }
        output_motion = _slice_motion(motion, start, end, task, cleaning)
        manifest_item = {
            "file": f"{output_stem}.pkl",
            "source_dataset": source_name,
            "source_file": motion_path.name,
            "source_start_frame": int(start),
            "source_end_frame_exclusive": int(end),
            "frames": int(frame_count),
            "duration_s": float((frame_count - 1) / fps),
            "mode": mode_name,
            "mode_id": int(mode_id),
            "mean_vx": float(mean_command[0]),
            "mean_vy": float(mean_command[1]),
            "mean_wz": float(mean_command[2]),
            "mean_speed_xy": mean_speed,
            "quality": quality,
        }
        accepted.append((output_stem, output_motion, manifest_item))
    return accepted, rejected


def _source_name(path: Path) -> str:
    return path.name.replace("-", "_")


def _audit_sources(source_dirs: list[Path], audit_dir: Path | None) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "source_dirs": [str(path) for path in source_dirs],
        "audit_dir": str(audit_dir) if audit_dir is not None else "",
        "default_training_sources": [path.name for path in source_dirs],
    }
    if audit_dir is None or not audit_dir.is_dir():
        return audit
    source_files = set()
    for source_dir in source_dirs:
        source_files.update(path.name for path in source_dir.glob("*.pkl"))
    audit_files = set(path.name for path in audit_dir.glob("*.pkl"))
    audit["audit_file_count"] = len(audit_files)
    audit["overlap_with_training_sources"] = sorted(audit_files.intersection(source_files))
    audit["audit_only_files"] = sorted(audit_files.difference(source_files))
    return audit


def _write_dataset_format(output_dir: Path, manifest: dict[str, Any]) -> None:
    mode_lines = "\n".join(
        f"- `{name}`: id `{MODE_ID_BY_NAME[name]}`, command ranges `{COMMAND_RANGES[name]}`"
        for name in MODE_ID_BY_NAME
    )
    text = f"""# G1 Command-Balanced Directional 50Hz Dataset

Generated by `scripts/tools/build_g1_command_balanced_dataset.py`.

## Summary

- Source datasets: {', '.join(manifest['summary']['source_datasets'])}
- Segment count: {manifest['summary']['segment_count']}
- Total duration: {manifest['summary']['total_duration_s']:.2f} s
- Quaternion order: `xyzw`
- Command frame: robot base local `[vx, vy, wz]`

## Pickle Fields

Each `.pkl` keeps the Lab/GMR fields `fps`, `root_pos`, `root_rot`, `dof_pos`,
`dof_names`, `local_body_pos`, `link_body_list`, `robot`, and `robot_xml`.
Segments add:

- `task.command`: `[T, 3]` body-frame command in m/s and rad/s.
- `task.mode_id`: `[T]` integer mode label.
- `task.mode_name_by_id`: mapping from ids to names.
- `task.source_dataset`, `task.source_file`, `task.source_frame_range`.
- `task.quality`: foot-contact, root, and arm-swing quality metrics.
- `task_scope`: command frame and segment-level label summary.
- `cleaning`: pass/reject metadata for accepted segments.

## Modes

{mode_lines}

## Sidecar Files

- `manifest.json`: summary, per-segment metadata, source audit, and rejects.
- `motion_weights.json`: motion-name weights balanced by target mode weight.
- `task_sampling_config.json`: mode-balanced command sampler configuration.
"""
    (output_dir / "DATASET_FORMAT.md").write_text(text, encoding="utf-8")


def _write_sidecars(
    output_dir: Path,
    source_dirs: list[Path],
    audit_dir: Path | None,
    manifest_items: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    durations_by_mode: dict[str, float] = defaultdict(float)
    counts_by_mode: Counter[str] = Counter()
    for item in manifest_items:
        durations_by_mode[str(item["mode"])] += float(item["duration_s"])
        counts_by_mode[str(item["mode"])] += 1

    active_target_weight_sum = sum(TARGET_MODE_WEIGHTS[mode] for mode in durations_by_mode)
    motion_weights: dict[str, float] = {}
    for item in manifest_items:
        mode = str(item["mode"])
        duration = float(item["duration_s"])
        if durations_by_mode[mode] <= 0.0:
            continue
        normalized_target = TARGET_MODE_WEIGHTS[mode] / active_target_weight_sum
        weight = normalized_target * duration / durations_by_mode[mode]
        motion_weights[Path(str(item["file"])).stem] = float(weight)

    summary = {
        "source_datasets": [path.name for path in source_dirs],
        "segment_count": len(manifest_items),
        "total_duration_s": float(sum(float(item["duration_s"]) for item in manifest_items)),
        "counts_by_mode": dict(sorted(counts_by_mode.items())),
        "durations_by_mode_s": {key: float(value) for key, value in sorted(durations_by_mode.items())},
        "target_mode_weights": TARGET_MODE_WEIGHTS,
        "active_target_weight_sum": float(active_target_weight_sum),
    }
    rejected_counter = Counter(str(item.get("reason", "unknown")) for item in rejected)
    manifest = {
        "summary": summary,
        "source_audit": _audit_sources(source_dirs, audit_dir),
        "segments": manifest_items,
        "rejected_summary": dict(sorted(rejected_counter.items())),
        "rejected": rejected[:2000],
    }

    task_sampling_config = {
        "command_frame": "robot_base_local",
        "fps": 50,
        "mode_weights": TARGET_MODE_WEIGHTS,
        "mode_name_by_id": MODE_NAME_BY_ID,
        "modes": COMMAND_RANGES,
        "source_dataset": output_dir.name,
    }

    (output_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    (output_dir / "motion_weights.json").write_text(
        json.dumps(_jsonable(motion_weights), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "task_sampling_config.json").write_text(
        json.dumps(_jsonable(task_sampling_config), indent=2),
        encoding="utf-8",
    )
    _write_dataset_format(output_dir, manifest)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to rebuild.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_dirs = [_resolve(path) for path in args.source_dir]
    audit_dir = _resolve(args.audit_dir) if args.audit_dir else None
    output_dir = _resolve(args.output)
    _prepare_output_dir(output_dir, args.overwrite)

    manifest_items: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        source_name = _source_name(source_dir)
        for motion_path in sorted(source_dir.glob("*.pkl")):
            motion = _load_motion(motion_path)
            _validate_motion_format(motion, motion_path)
            accepted_for_motion, rejected_for_motion = _build_segments_for_motion(source_name, motion_path, motion, args)
            rejected.extend(rejected_for_motion)
            for output_stem, output_motion, manifest_item in accepted_for_motion:
                output_path = output_dir / f"{output_stem}.pkl"
                with output_path.open("wb") as file:
                    pickle.dump(output_motion, file, protocol=pickle.HIGHEST_PROTOCOL)
                manifest_items.append(manifest_item)

    if not manifest_items:
        raise RuntimeError("No segments were accepted; inspect reject thresholds or source data.")
    _write_sidecars(output_dir, source_dirs, audit_dir, manifest_items, rejected)
    counts = Counter(str(item["mode"]) for item in manifest_items)
    durations = defaultdict(float)
    for item in manifest_items:
        durations[str(item["mode"])] += float(item["duration_s"])
    return {
        "output_dir": str(output_dir),
        "segments": len(manifest_items),
        "counts_by_mode": dict(sorted(counts.items())),
        "durations_by_mode_s": {key: float(value) for key, value in sorted(durations.items())},
        "rejected": len(rejected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        action="append",
        default=None,
        help="Source Lab/GMR-format directory. May be repeated.",
    )
    parser.add_argument(
        "--audit_dir",
        default=str(DEFAULT_AUDIT_DIR.relative_to(REPO_ROOT)),
        help="Optional source used for overlap audit but not training output.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Output dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild the output directory.")
    parser.add_argument("--smooth_seconds", type=float, default=0.14, help="Moving-average smoothing for commands.")
    parser.add_argument("--min_window_seconds", type=float, default=1.20)
    parser.add_argument("--window_seconds", type=float, default=2.40)
    parser.add_argument("--max_window_seconds", type=float, default=3.00)
    parser.add_argument("--stride_seconds", type=float, default=0.60)
    args = parser.parse_args()
    if args.source_dir is None:
        args.source_dir = [str(path.relative_to(REPO_ROOT)) for path in DEFAULT_SOURCE_DIRS]
    return args


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
