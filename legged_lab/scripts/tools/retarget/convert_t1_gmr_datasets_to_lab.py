#!/usr/bin/env python3
"""Batch-convert T1_GMR datasets into Lab/AMP motion format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from convert_t1_pkl_to_amp_format import convert_pkl


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_ROOT = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/T1_GMR"
DEFAULT_DST_ROOT = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/T1_Lab"
SIDECAR_NAMES = {
    "DATASET_FORMAT.md",
    "CLEANING_REPORT.md",
    "manifest.json",
    "summary.csv",
    "style_cluster_summary.csv",
    "window_diagnostics.csv",
}


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)


def quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[:, :3]
    q_w = quat[:, 3:4]
    t = 2.0 * np.cross(q_xyz, vec, axis=-1)
    return vec + q_w * t + np.cross(q_xyz, t, axis=-1)


def quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[:, :3] *= -1.0
    return out


def quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = lhs[:, 0], lhs[:, 1], lhs[:, 2], lhs[:, 3]
    x2, y2, z2, w2 = rhs[:, 0], rhs[:, 1], rhs[:, 2], rhs[:, 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=1,
    )


def quat_to_rotvec_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    flip = quat[:, 3] < 0.0
    quat[flip] *= -1.0
    xyz = quat[:, :3]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    norm_xyz = np.linalg.norm(xyz, axis=1)
    angle = 2.0 * np.arctan2(norm_xyz, w)
    scale = np.zeros_like(angle)
    valid = norm_xyz > 1.0e-8
    scale[valid] = angle[valid] / norm_xyz[valid]
    return xyz * scale[:, None]


def compute_body_velocities(root_pos: np.ndarray, root_rot_wxyz: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fps
    root_vel_w = np.empty_like(root_pos, dtype=np.float64)
    root_vel_w[:-1] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel_w[-1] = root_vel_w[-2] if root_vel_w.shape[0] > 1 else 0.0

    root_rot_xyzw = wxyz_to_xyzw(root_rot_wxyz.astype(np.float64))
    root_vel_b = quat_rotate_xyzw(quat_conjugate_xyzw(root_rot_xyzw), root_vel_w)

    root_ang_vel_b = np.zeros_like(root_pos, dtype=np.float64)
    if root_rot_xyzw.shape[0] > 1:
        delta = quat_multiply_xyzw(quat_conjugate_xyzw(root_rot_xyzw[:-1]), root_rot_xyzw[1:])
        root_ang_vel_b[:-1] = quat_to_rotvec_xyzw(delta) / dt
        root_ang_vel_b[-1] = root_ang_vel_b[-2]
    return root_vel_b, root_ang_vel_b


def command_samples(motion: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = motion.get("task")
    if isinstance(task, dict) and "local_velocity_xy" in task and "yaw_rate" in task:
        local_vel = np.asarray(task["local_velocity_xy"], dtype=np.float64)
        yaw_rate = np.asarray(task["yaw_rate"], dtype=np.float64)
        return local_vel[:, 0], local_vel[:, 1], yaw_rate
    root_vel_b, root_ang_vel_b = compute_body_velocities(
        np.asarray(motion["root_pos"], dtype=np.float64),
        np.asarray(motion["root_rot"], dtype=np.float64),
        float(motion["fps"]),
    )
    return root_vel_b[:, 0], root_vel_b[:, 1], root_ang_vel_b[:, 2]


def estimate_feet_air_time_threshold(motions: list[dict]) -> tuple[float, dict[str, float | None]]:
    durations: list[float] = []
    for motion in motions:
        key_body_pos = np.asarray(motion.get("key_body_pos", np.zeros((0, 0, 3))), dtype=np.float64)
        if key_body_pos.ndim != 3 or key_body_pos.shape[1] < 2:
            continue
        fps = float(motion["fps"])
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
    values = np.asarray(durations, dtype=np.float64)
    if values.size == 0:
        return 0.25, {"p50": None, "p75": None, "p90": None}
    return float(np.clip(np.percentile(values, 75), 0.16, 0.28)), {
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def convert_dataset(src_dir: Path, dst_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(src_dir.glob("*.pkl"))
    if not source_files:
        return {"dataset": src_dir.name, "num_files": 0, "num_frames": 0}

    converted_motions: list[dict] = []
    lin_x: list[np.ndarray] = []
    lin_y: list[np.ndarray] = []
    yaw_z: list[np.ndarray] = []
    for src_path in source_files:
        dst_path = dst_dir / src_path.name
        convert_pkl(str(src_path), str(dst_path))
        motion = joblib.load(dst_path)
        converted_motions.append(motion)
        x, y, yaw = command_samples(motion)
        lin_x.append(x)
        lin_y.append(y)
        yaw_z.append(yaw)

    for sidecar in SIDECAR_NAMES:
        src_sidecar = src_dir / sidecar
        if src_sidecar.exists():
            shutil.copy2(src_sidecar, dst_dir / sidecar)

    lin_x_all = np.concatenate(lin_x)
    lin_y_all = np.concatenate(lin_y)
    yaw_z_all = np.concatenate(yaw_z)
    feet_air_time_threshold, single_stance = estimate_feet_air_time_threshold(converted_motions)
    metadata = {
        "source_dir": str(src_dir),
        "num_source_files": len(source_files),
        "num_files": len(source_files),
        "num_frames": int(sum(np.asarray(motion["root_pos"]).shape[0] for motion in converted_motions)),
        "fps_values": sorted({int(motion["fps"]) for motion in converted_motions}),
        "format": {
            "root_rot": "wxyz",
            "dof_pos": "29 DoF Lab order with head zeros; joint angles preserve GMR XML zero convention",
            "key_body_pos": "world positions for left/right feet, hands, AL3, AR3",
        },
        "command_ranges": {
            "lin_vel_x": [float(np.percentile(lin_x_all, 5)), float(np.percentile(lin_x_all, 95))],
            "lin_vel_y": [float(np.percentile(lin_y_all, 5)), float(np.percentile(lin_y_all, 95))],
            "ang_vel_z": [float(np.percentile(yaw_z_all, 5)), float(np.percentile(yaw_z_all, 95))],
        },
        "command_means": {
            "lin_vel_x": float(np.mean(lin_x_all)),
            "lin_vel_y": float(np.mean(lin_y_all)),
            "ang_vel_z": float(np.mean(yaw_z_all)),
        },
        "feet_air_time_threshold": feet_air_time_threshold,
        "single_stance_duration_s": single_stance,
        "motions": [path.stem for path in source_files],
    }
    with (dst_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset subdirectory to convert. Repeatable.")
    parser.add_argument("--clear", action="store_true", help="Remove destination dataset directories before writing.")
    args = parser.parse_args()

    datasets = args.dataset or [path.name for path in sorted(args.src_root.iterdir()) if path.is_dir()]
    summaries = []
    for dataset_name in datasets:
        src_dir = args.src_root / dataset_name
        dst_dir = args.dst_root / dataset_name
        if not src_dir.is_dir():
            raise FileNotFoundError(src_dir)
        if args.clear and dst_dir.exists():
            shutil.rmtree(dst_dir)
        summaries.append(convert_dataset(src_dir, dst_dir))

    args.dst_root.mkdir(parents=True, exist_ok=True)
    with (args.dst_root / "conversion_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()