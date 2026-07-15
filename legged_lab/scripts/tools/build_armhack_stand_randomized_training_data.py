#!/usr/bin/env python3
"""Build the reproducible ArmHack Stand random-pose training bank.

The bank stays inside the measured, training-reachable 14-DoF arm-pose
envelope.  It contains measured coverage anchors plus new convex combinations
of measured poses.  Convex synthesis preserves cross-joint correlations better
than sampling every joint independently while still producing unseen poses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "Reference Data" / "ArmHack" / "StandPerturb" / "g1_arm_trajectory_named_50hz.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "Reference Data"
    / "ArmHack"
    / "StandPerturb"
    / "RandomizedTraining"
    / "random_arm_pose_bank_seed20260715.json"
)
DEFAULT_SEED = 20260715
DEFAULT_BANK_SIZE = 512
DEFAULT_SOURCE_ANCHOR_COUNT = 64
DEFAULT_END_MARGIN_S = 0.25
MIN_SAFE_VELOCITY_RAD_S = 0.20
G1_ENV_ARM_JOINT_ORDER = [
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "time_s" not in reader.fieldnames:
            raise ValueError(f"Arm source must contain time_s: {path}")
        joint_names = [name for name in reader.fieldnames if name != "time_s"]
        rows = list(reader)
    if len(joint_names) != 14 or not rows:
        raise ValueError(f"Expected 14 named arm joints and non-empty data: {path}")
    if any(
        not name.startswith(("left_", "right_"))
        or not any(token in name for token in ("shoulder", "elbow", "wrist"))
        for name in joint_names
    ):
        raise ValueError(f"Source is not arm-only: {joint_names}")

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    positions = np.asarray([[float(row[name]) for name in joint_names] for row in rows], dtype=np.float64)
    times -= times[0]
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("Arm source contains non-finite values.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Arm source time must be strictly increasing.")
    return joint_names, times, positions


def _farthest_point_indices(positions: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or count > len(positions):
        raise ValueError(f"Invalid source anchor count {count} for {len(positions)} poses.")
    center = np.median(positions, axis=0)
    scale = np.maximum(np.percentile(positions, 95.0, axis=0) - np.percentile(positions, 5.0, axis=0), 0.05)
    normalized = np.clip((positions - center) / scale, -4.0, 4.0)
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    min_distance = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    while len(selected) < count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(normalized - normalized[next_index], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def _joint_statistics(
    joint_names: list[str],
    times: np.ndarray,
    positions: np.ndarray,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    velocities = np.diff(positions, axis=0) / np.diff(times)[:, None]
    statistics: dict[str, dict[str, float]] = {}
    velocity_limits: list[float] = []
    for joint_index, joint_name in enumerate(joint_names):
        values = positions[:, joint_index]
        abs_velocity = np.abs(velocities[:, joint_index])
        velocity_p99 = float(np.percentile(abs_velocity, 99.0))
        safe_velocity = max(velocity_p99, MIN_SAFE_VELOCITY_RAD_S)
        velocity_limits.append(safe_velocity)
        statistics[joint_name] = {
            "min_rad": float(np.min(values)),
            "p01_rad": float(np.percentile(values, 1.0)),
            "p05_rad": float(np.percentile(values, 5.0)),
            "median_rad": float(np.median(values)),
            "p95_rad": float(np.percentile(values, 95.0)),
            "p99_rad": float(np.percentile(values, 99.0)),
            "max_rad": float(np.max(values)),
            "abs_velocity_p95_rad_s": float(np.percentile(abs_velocity, 95.0)),
            "abs_velocity_p99_rad_s": velocity_p99,
            "abs_velocity_max_rad_s": float(np.max(abs_velocity)),
            "interpolation_velocity_limit_rad_s": safe_velocity,
        }
    return statistics, velocity_limits


def build_pose_bank(args: argparse.Namespace) -> dict[str, object]:
    joint_names, all_times, all_positions = _load_source(args.source)
    if set(joint_names) != set(G1_ENV_ARM_JOINT_ORDER):
        raise ValueError("Stand arm CSV joints differ from the expected G1 environment arm joints.")
    environment_order = [joint_names.index(name) for name in G1_ENV_ARM_JOINT_ORDER]
    joint_names = list(G1_ENV_ARM_JOINT_ORDER)
    all_positions = all_positions[:, environment_order]
    if args.end_margin_s < 0.0:
        raise ValueError("end_margin_s must be non-negative.")
    maximum_source_time_s = float(all_times[-1]) - args.end_margin_s
    reachable_mask = all_times <= maximum_source_time_s + 1.0e-9
    times = all_times[reachable_mask]
    positions = all_positions[reachable_mask]
    if len(positions) < args.source_anchor_count:
        raise ValueError("Training-reachable source contains too few samples for the requested anchors.")
    if args.bank_size < args.source_anchor_count:
        raise ValueError("bank_size must be at least source_anchor_count.")

    statistics, velocity_limits = _joint_statistics(joint_names, times, positions)
    candidate_stride = max(len(positions) // 4096, 1)
    candidate_indices = np.arange(0, len(positions), candidate_stride, dtype=np.int64)
    selected_local = _farthest_point_indices(positions[candidate_indices], args.source_anchor_count)
    anchor_indices = candidate_indices[selected_local]

    rng = np.random.default_rng(args.seed)
    pose_values: list[np.ndarray] = [positions[index].copy() for index in anchor_indices]
    pose_metadata: list[dict[str, object]] = [
        {
            "pose_id": f"source_anchor_{index + 1:03d}",
            "kind": "measured_source_anchor",
            "source_row": int(source_index),
            "source_time_s": float(times[source_index]),
        }
        for index, source_index in enumerate(anchor_indices)
    ]

    while len(pose_values) < args.bank_size:
        parent_count = int(rng.integers(2, 5))
        parent_indices = rng.choice(len(positions), size=parent_count, replace=False)
        weights = rng.dirichlet(np.full(parent_count, 1.5, dtype=np.float64))
        pose = np.sum(positions[parent_indices] * weights[:, None], axis=0)
        generated_index = len(pose_values) - args.source_anchor_count + 1
        pose_values.append(pose)
        pose_metadata.append(
            {
                "pose_id": f"random_convex_pose_{generated_index:04d}",
                "kind": "convex_synthesized",
                "parent_source_rows": [int(value) for value in parent_indices],
                "parent_source_times_s": [float(times[value]) for value in parent_indices],
                "parent_weights": [float(value) for value in weights],
            }
        )

    pose_bank = np.asarray(pose_values, dtype=np.float64)
    source_lower = np.min(positions, axis=0)
    source_upper = np.max(positions, axis=0)
    if np.any(pose_bank < source_lower[None, :] - 1.0e-10) or np.any(
        pose_bank > source_upper[None, :] + 1.0e-10
    ):
        raise RuntimeError("Generated pose left the measured joint-wise source range.")

    payload: dict[str, object] = {
        "schema_version": 1,
        "data_scope": "arm_only_14_dof",
        "units": {"joint_position": "rad", "joint_velocity": "rad/s", "time": "s"},
        "source": {
            "path": args.source.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(args.source),
            "rows_total": int(len(all_positions)),
            "duration_s": float(all_times[-1]),
            "training_reachable_rows": int(len(positions)),
            "training_reachable_time_range_s": [0.0, float(times[-1])],
            "end_margin_s": float(args.end_margin_s),
        },
        "generation": {
            "seed": int(args.seed),
            "bank_size": int(args.bank_size),
            "source_anchor_count": int(args.source_anchor_count),
            "random_convex_pose_count": int(args.bank_size - args.source_anchor_count),
            "method": (
                "measured farthest-point anchors plus 2-to-4-parent Dirichlet convex combinations "
                "of measured training-reachable arm poses"
            ),
            "independent_per_joint_uniform_sampling": False,
            "contains_future_policy_observation": False,
        },
        "joint_names": joint_names,
        "joint_statistics": statistics,
        "interpolation_velocity_limits_rad_s": velocity_limits,
        "poses": [
            {**metadata, "positions_rad": [float(value) for value in pose]}
            for metadata, pose in zip(pose_metadata, pose_bank, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE)
    parser.add_argument("--source-anchor-count", type=int, default=DEFAULT_SOURCE_ANCHOR_COUNT)
    parser.add_argument("--end-margin-s", type=float, default=DEFAULT_END_MARGIN_S)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    payload = build_pose_bank(args)
    print(f"Built ArmHack Stand random-pose bank: {args.output}")
    print(f"Source SHA-256: {payload['source']['sha256']}")
    print(
        f"Pose bank: {payload['generation']['bank_size']} poses = "
        f"{payload['generation']['source_anchor_count']} measured anchors + "
        f"{payload['generation']['random_convex_pose_count']} new convex poses"
    )


if __name__ == "__main__":
    main()
