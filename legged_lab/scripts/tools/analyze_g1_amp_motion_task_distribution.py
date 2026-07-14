#!/usr/bin/env python3
"""Print command/task distribution summaries for G1 AMP motion datasets."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import pickle
import sys
import types

import numpy as np


def _install_numpy_pickle_shim() -> None:
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
DEFAULT_DIRS = [
    "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz_task_core",
    "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz",
]


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64)
    if values.ndim == 1:
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")
        return smoothed[: values.shape[0]]
    return np.stack([_moving_average(values[:, dim], window) for dim in range(values.shape[1])], axis=1)


def _quat_to_yaw(root_rot: np.ndarray, order: str) -> np.ndarray:
    if order == "xyzw":
        x, y, z, w = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
    elif order == "wxyz":
        w, x, y, z = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
    else:
        raise ValueError(f"Unsupported root_rot_order={order!r}")
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.unwrap(yaw)


def _metrics(
    motion: dict, root_rot_order: str, smooth_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fps = int(motion["fps"])
    dt = 1.0 / fps
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    window = max(1, int(round(smooth_seconds * fps)))

    velocity_xy = _moving_average(np.gradient(root_pos[:, :2], dt, axis=0), window)
    yaw = _quat_to_yaw(root_rot, root_rot_order)
    wz = _moving_average(np.gradient(yaw, dt), window)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_vx = cos_yaw * velocity_xy[:, 0] + sin_yaw * velocity_xy[:, 1]
    local_vy = -sin_yaw * velocity_xy[:, 0] + cos_yaw * velocity_xy[:, 1]
    local_velocity_xy = np.stack([local_vx, local_vy], axis=1)
    speed_xy = np.linalg.norm(local_velocity_xy, axis=1)
    return local_velocity_xy, speed_xy, wz, velocity_xy, yaw


def _path_stats(motion: dict) -> dict[str, float]:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    displacement = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
    path_length = float(np.sum(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1)))
    return {
        "displacement": displacement,
        "path_length": path_length,
        "path_turn_ratio": path_length / max(displacement, 1.0e-4),
        "x_range": float(np.ptp(root_pos[:, 0])),
        "y_range": float(np.ptp(root_pos[:, 1])),
    }


def _heading_yaw_abs_error(velocity_xy: np.ndarray, yaw: np.ndarray, moving_speed_threshold: float) -> np.ndarray:
    speed_xy = np.linalg.norm(velocity_xy, axis=1)
    moving = speed_xy > moving_speed_threshold
    if not np.any(moving):
        return np.empty((0,), dtype=np.float64)
    heading = np.arctan2(velocity_xy[:, 1], velocity_xy[:, 0])
    error = (heading - yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(error[moving])


def _alignment_summary(abs_error: np.ndarray) -> str:
    if abs_error.size == 0:
        return "moving_frames=0"
    return (
        f"moving_frames={abs_error.size} "
        f"mean={np.mean(abs_error):.4f} p50={np.quantile(abs_error, 0.50):.4f} "
        f"p95={np.quantile(abs_error, 0.95):.4f} "
        f"aligned<30deg={np.mean(abs_error < np.deg2rad(30.0)):.4f} "
        f"sideways>60deg={np.mean(abs_error > np.deg2rad(60.0)):.4f} "
        f"backward>120deg={np.mean(abs_error > np.deg2rad(120.0)):.4f}"
    )


def _task_mask(local_velocity_xy: np.ndarray, speed_xy: np.ndarray, wz: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    vx = local_velocity_xy[:, 0]
    vy = local_velocity_xy[:, 1]
    velocity_ok = (vx > args.vx_min) & (vx < args.vx_max) & (vy > args.vy_min) & (vy < args.vy_max)
    low_speed = speed_xy < args.speed_split
    low_yaw_ok = (wz > args.low_speed_wz_min) & (wz < args.low_speed_wz_max)
    high_yaw_ok = (wz > args.high_speed_wz_min) & (wz < args.high_speed_wz_max)
    return velocity_ok & ((low_speed & low_yaw_ok) | (~low_speed & high_yaw_ok))


def _stats(values: np.ndarray) -> str:
    if values.size == 0:
        return "empty"
    return (
        f"mean={np.mean(values):+.4f} std={np.std(values):.4f} "
        f"min={np.min(values):+.4f} p05={np.quantile(values, 0.05):+.4f} "
        f"p50={np.quantile(values, 0.50):+.4f} p95={np.quantile(values, 0.95):+.4f} max={np.max(values):+.4f}"
    )


def _print_distribution(path: Path, args: argparse.Namespace) -> None:
    motion_paths = sorted(path.glob("*.pkl"))
    if not motion_paths:
        raise FileNotFoundError(f"No .pkl files found in {path}")

    all_vxy = []
    all_speed = []
    all_wz = []
    all_heading_yaw_abs_error = []
    all_embedded_task_abs_delta = []
    per_clip = []
    embedded_task_count = 0
    for motion_path in motion_paths:
        with motion_path.open("rb") as f:
            motion = pickle.load(f)
        if "task" in motion:
            embedded_task_count += 1
        local_velocity_xy, speed_xy, wz, velocity_xy, yaw = _metrics(
            motion, args.root_rot_order, args.smooth_seconds
        )
        mask = _task_mask(local_velocity_xy, speed_xy, wz, args)
        path_stats = _path_stats(motion)
        heading_yaw_abs_error = _heading_yaw_abs_error(velocity_xy, yaw, args.moving_speed_threshold)
        if heading_yaw_abs_error.size:
            all_heading_yaw_abs_error.append(heading_yaw_abs_error)
        if isinstance(motion.get("task"), dict) and "command" in motion["task"]:
            embedded_command = np.asarray(motion["task"]["command"], dtype=np.float64)
            computed_command = np.column_stack([local_velocity_xy, wz])
            count = min(embedded_command.shape[0], computed_command.shape[0])
            if count:
                all_embedded_task_abs_delta.append(np.abs(embedded_command[:count, :3] - computed_command[:count]))
        all_vxy.append(local_velocity_xy)
        all_speed.append(speed_xy)
        all_wz.append(wz)
        per_clip.append(
            {
                "name": motion_path.name,
                "frames": len(speed_xy),
                "in_scope": float(np.mean(mask)),
                "speed_mean": float(np.mean(speed_xy)),
                "abs_wz_mean": float(np.mean(np.abs(wz))),
                "abs_wz_max": float(np.max(np.abs(wz))),
                "straight_ratio": float(np.mean(np.abs(wz) < 0.05)),
                "lateral_ratio": float(np.mean(np.abs(local_velocity_xy[:, 1]) > 0.10)),
                "path_turn_ratio": path_stats["path_turn_ratio"],
                "displacement": path_stats["displacement"],
                "path_length": path_stats["path_length"],
                "x_range": path_stats["x_range"],
                "y_range": path_stats["y_range"],
                "heading_yaw_abs_error_mean": float(np.mean(heading_yaw_abs_error)) if heading_yaw_abs_error.size else 0.0,
                "sideways_ratio_60deg": float(np.mean(heading_yaw_abs_error > np.deg2rad(60.0)))
                if heading_yaw_abs_error.size
                else 0.0,
                "backward_ratio_120deg": float(np.mean(heading_yaw_abs_error > np.deg2rad(120.0)))
                if heading_yaw_abs_error.size
                else 0.0,
            }
        )

    vxy = np.concatenate(all_vxy, axis=0)
    speed = np.concatenate(all_speed, axis=0)
    wz = np.concatenate(all_wz, axis=0)
    heading_yaw_abs_error = (
        np.concatenate(all_heading_yaw_abs_error, axis=0)
        if all_heading_yaw_abs_error
        else np.empty((0,), dtype=np.float64)
    )
    embedded_task_abs_delta = (
        np.concatenate(all_embedded_task_abs_delta, axis=0)
        if all_embedded_task_abs_delta
        else np.empty((0, 3), dtype=np.float64)
    )
    task_scope = _task_mask(vxy, speed, wz, args)
    duration_s = len(speed) / float(args.fps)

    print(f"\n=== {path.name} ===")
    print(f"dir: {path}")
    print(f"files={len(motion_paths)} frames={len(speed)} duration_s={duration_s:.2f} embedded_task_labels={embedded_task_count}")
    print(f"current_task_scope_ratio={np.mean(task_scope):.4f}")
    print(f"straight |wz|<0.05 ratio={np.mean(np.abs(wz) < 0.05):.4f}")
    print(f"mild yaw 0.05<=|wz|<0.15 ratio={np.mean((np.abs(wz) >= 0.05) & (np.abs(wz) < 0.15)):.4f}")
    print(f"turn yaw 0.15<=|wz|<0.30 ratio={np.mean((np.abs(wz) >= 0.15) & (np.abs(wz) < 0.30)):.4f}")
    print(f"strong yaw |wz|>=0.30 ratio={np.mean(np.abs(wz) >= 0.30):.4f}")
    print(f"lateral |vy|>0.10 ratio={np.mean(np.abs(vxy[:, 1]) > 0.10):.4f}")
    print(f"backward vx<-0.05 ratio={np.mean(vxy[:, 0] < -0.05):.4f}")
    print(f"slow speed<{args.speed_split:g} ratio={np.mean(speed < args.speed_split):.4f}")
    print(f"walk speed>={args.speed_split:g} ratio={np.mean(speed >= args.speed_split):.4f}")
    print(f"heading_vs_root_yaw_abs_error: {_alignment_summary(heading_yaw_abs_error)}")
    if embedded_task_abs_delta.size:
        print(
            "embedded_task_command_abs_delta: "
            f"vx={np.mean(embedded_task_abs_delta[:, 0]):.6f} "
            f"vy={np.mean(embedded_task_abs_delta[:, 1]):.6f} "
            f"wz={np.mean(embedded_task_abs_delta[:, 2]):.6f}"
        )
    print(f"vx: {_stats(vxy[:, 0])}")
    print(f"vy: {_stats(vxy[:, 1])}")
    print(f"speed_xy: {_stats(speed)}")
    print(f"wz: {_stats(wz)}")
    print(f"top {args.top_k} clips by abs_wz_mean:")
    for row in sorted(per_clip, key=lambda item: item["abs_wz_mean"], reverse=True)[: args.top_k]:
        print(
            f"  {row['name']} frames={row['frames']} in_scope={row['in_scope']:.3f} "
            f"speed_mean={row['speed_mean']:.3f} abs_wz_mean={row['abs_wz_mean']:.3f} "
            f"abs_wz_max={row['abs_wz_max']:.3f} straight={row['straight_ratio']:.3f} "
            f"lateral={row['lateral_ratio']:.3f}"
        )
    print(f"top {args.top_k} clips by path_turn_ratio:")
    for row in sorted(per_clip, key=lambda item: item["path_turn_ratio"], reverse=True)[: args.top_k]:
        print(
            f"  {row['name']} frames={row['frames']} path_turn={row['path_turn_ratio']:.2f} "
            f"disp={row['displacement']:.3f} path={row['path_length']:.3f} "
            f"x_rng={row['x_range']:.3f} y_rng={row['y_range']:.3f} "
            f"abs_wz_mean={row['abs_wz_mean']:.3f} align_err={row['heading_yaw_abs_error_mean']:.3f}"
        )
    print(f"top {args.top_k} clips by heading/root-yaw misalignment:")
    for row in sorted(per_clip, key=lambda item: item["heading_yaw_abs_error_mean"], reverse=True)[: args.top_k]:
        print(
            f"  {row['name']} frames={row['frames']} align_err={row['heading_yaw_abs_error_mean']:.3f} "
            f"sideways={row['sideways_ratio_60deg']:.3f} backward={row['backward_ratio_120deg']:.3f} "
            f"path_turn={row['path_turn_ratio']:.2f} abs_wz_mean={row['abs_wz_mean']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_dirs", nargs="+", default=DEFAULT_DIRS, help="Motion directories to compare.")
    parser.add_argument("--root_rot_order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--smooth_seconds", type=float, default=0.20)
    parser.add_argument("--moving_speed_threshold", type=float, default=0.15)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--vx_min", type=float, default=-0.4)
    parser.add_argument("--vx_max", type=float, default=1.0)
    parser.add_argument("--vy_min", type=float, default=-0.3)
    parser.add_argument("--vy_max", type=float, default=0.3)
    parser.add_argument("--speed_split", type=float, default=0.4)
    parser.add_argument("--low_speed_wz_min", type=float, default=-0.5)
    parser.add_argument("--low_speed_wz_max", type=float, default=0.5)
    parser.add_argument("--high_speed_wz_min", type=float, default=-0.3)
    parser.add_argument("--high_speed_wz_max", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    print("Current task scope:")
    print(
        f"  vx=({args.vx_min}, {args.vx_max}), vy=({args.vy_min}, {args.vy_max}), "
        f"speed_split={args.speed_split}, low_speed_wz=({args.low_speed_wz_min}, {args.low_speed_wz_max}), "
        f"high_speed_wz=({args.high_speed_wz_min}, {args.high_speed_wz_max})"
    )
    for motion_dir in args.motion_dirs:
        _print_distribution(_resolve(motion_dir), args)


if __name__ == "__main__":
    main()
