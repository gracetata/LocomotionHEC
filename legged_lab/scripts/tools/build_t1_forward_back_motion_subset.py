#!/usr/bin/env python3
"""Build a forward/back-only T1 AMP motion subset by velocity-based clipping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "source" / "legged_lab" / "legged_lab" / "data" / "MotionData" / "t1_29dof_accad_g1used_50hz_amp_official"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "source" / "legged_lab" / "legged_lab" / "data" / "MotionData" / "t1_29dof_forward_back_filtered_50hz"


def wxyz_to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)
    return Rotation.from_quat(quat_xyzw)


def compute_body_velocities(root_pos: np.ndarray, root_rot: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fps
    root_vel_w = np.empty_like(root_pos)
    root_vel_w[:-1] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel_w[-1] = root_vel_w[-2]

    rotation = wxyz_to_rotation(root_rot)
    root_vel_b = rotation.inv().apply(root_vel_w)
    rotvec = (rotation[:-1].inv() * rotation[1:]).as_rotvec() / dt
    root_ang_vel_b = np.empty_like(root_pos)
    root_ang_vel_b[:-1] = rotvec
    root_ang_vel_b[-1] = root_ang_vel_b[-2]
    return root_vel_b, root_ang_vel_b


def contiguous_true_runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_frames:
                runs.append((start, index))
            start = None
    return runs


def crop_motion(motion: dict, start: int, end: int) -> dict:
    cropped = {}
    first_xy = np.asarray(motion["root_pos"])[start, :2].copy()
    for key, value in motion.items():
        if key in {"root_pos", "root_rot", "dof_pos", "key_body_pos"}:
            cropped_value = np.asarray(value)[start:end].copy()
            if key == "root_pos":
                cropped_value[:, :2] -= first_xy
            elif key == "key_body_pos":
                cropped_value[:, :, :2] -= first_xy[None, None, :]
            cropped[key] = cropped_value
        else:
            cropped[key] = value
    cropped["loop_mode"] = 0
    return cropped


def estimate_single_stance_durations(segment_files: list[Path]) -> np.ndarray:
    durations: list[float] = []
    for path in segment_files:
        motion = joblib.load(path)
        fps = float(motion["fps"])
        key_body_pos = np.asarray(motion["key_body_pos"])
        foot_z = key_body_pos[:, :2, 2]
        contact_z = np.percentile(foot_z, 20, axis=0) + 0.03
        in_air = foot_z > contact_z[None, :]
        single_stance = np.sum(in_air, axis=1) == 1
        start = None
        for index, value in enumerate(single_stance.tolist() + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                durations.append((index - start) / fps)
                start = None
    return np.asarray(durations, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_abs_lin_y", type=float, default=0.20)
    parser.add_argument("--max_abs_ang_z", type=float, default=0.50)
    parser.add_argument("--min_abs_lin_x", type=float, default=0.15)
    parser.add_argument("--max_abs_lin_x", type=float, default=2.40)
    parser.add_argument("--min_frames", type=int, default=70)
    parser.add_argument("--clear", action="store_true", help="Remove existing generated pkl/metadata files first.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for path in args.output_dir.glob("*.pkl"):
            path.unlink()
        metadata_path = args.output_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

    all_lin_vel_b: list[np.ndarray] = []
    all_ang_vel_b: list[np.ndarray] = []
    written_files: list[Path] = []
    source_files = sorted(args.source_dir.glob("*.pkl"))
    if not source_files:
        raise FileNotFoundError(f"No source pkl files found in {args.source_dir}")

    for source_path in source_files:
        motion = joblib.load(source_path)
        fps = float(motion["fps"])
        root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
        root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
        root_vel_b, root_ang_vel_b = compute_body_velocities(root_pos, root_rot, fps)
        mask = (
            (np.abs(root_vel_b[:, 1]) <= args.max_abs_lin_y)
            & (np.abs(root_ang_vel_b[:, 2]) <= args.max_abs_ang_z)
            & (np.abs(root_vel_b[:, 0]) >= args.min_abs_lin_x)
            & (np.abs(root_vel_b[:, 0]) <= args.max_abs_lin_x)
        )
        for segment_index, (start, end) in enumerate(contiguous_true_runs(mask, args.min_frames)):
            cropped = crop_motion(motion, start, end)
            out_name = f"{source_path.stem}__fb_{segment_index:02d}_{start:04d}_{end:04d}.pkl"
            out_path = args.output_dir / out_name
            joblib.dump(cropped, out_path)
            written_files.append(out_path)
            all_lin_vel_b.append(root_vel_b[start:end])
            all_ang_vel_b.append(root_ang_vel_b[start:end])

    if not written_files:
        raise RuntimeError("No forward/back segments matched the requested thresholds.")

    lin_vel_b = np.concatenate(all_lin_vel_b, axis=0)
    ang_vel_b = np.concatenate(all_ang_vel_b, axis=0)
    stance_durations = estimate_single_stance_durations(written_files)
    feet_air_threshold = 0.22
    if stance_durations.size:
        feet_air_threshold = float(np.clip(np.percentile(stance_durations, 75), 0.16, 0.28))

    metadata = {
        "source_dir": str(args.source_dir),
        "num_source_files": len(source_files),
        "num_segments": len(written_files),
        "num_frames": int(sum(joblib.load(path)["root_pos"].shape[0] for path in written_files)),
        "filters": {
            "max_abs_lin_y": args.max_abs_lin_y,
            "max_abs_ang_z": args.max_abs_ang_z,
            "min_abs_lin_x": args.min_abs_lin_x,
            "max_abs_lin_x": args.max_abs_lin_x,
            "min_frames": args.min_frames,
        },
        "command_ranges": {
            "lin_vel_x": [float(np.percentile(lin_vel_b[:, 0], 5)), float(np.percentile(lin_vel_b[:, 0], 95))],
            "lin_vel_y": [float(np.percentile(lin_vel_b[:, 1], 5)), float(np.percentile(lin_vel_b[:, 1], 95))],
            "ang_vel_z": [float(np.percentile(ang_vel_b[:, 2], 5)), float(np.percentile(ang_vel_b[:, 2], 95))],
        },
        "feet_air_time_threshold": feet_air_threshold,
        "single_stance_duration_s": {
            "p50": float(np.percentile(stance_durations, 50)) if stance_durations.size else None,
            "p75": float(np.percentile(stance_durations, 75)) if stance_durations.size else None,
            "p90": float(np.percentile(stance_durations, 90)) if stance_durations.size else None,
        },
        "segments": [path.stem for path in written_files],
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps({key: metadata[key] for key in ["num_segments", "num_frames", "command_ranges", "feet_air_time_threshold", "single_stance_duration_s"]}, indent=2))


if __name__ == "__main__":
    main()