#!/usr/bin/env python3
"""Generate one reproducible, arm-only ArmHack Stand MuJoCo trajectory.

The waypoints are sampled without replacement from the 512-pose randomized
training bank.  Adjacent waypoints are connected with quintic minimum-jerk
interpolation.  Every transition is lengthened when necessary so its peak
joint speed stays within the per-joint limits stored in the pose bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(
    pose_bank_path: Path,
    manifest_path: Path,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    pose_bank = json.loads(pose_bank_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if pose_bank.get("schema_version") != 1 or pose_bank.get("data_scope") != "arm_only_14_dof":
        raise ValueError(f"Unsupported ArmHack Stand pose-bank schema: {pose_bank_path}")
    if manifest.get("schema_version") != 5 or manifest.get("data_scope") != "arm_only_14_dof":
        raise ValueError(f"Unsupported ArmHack Stand test manifest: {manifest_path}")
    if manifest.get("contains_full_body_state") is not False:
        raise ValueError("ArmHack Stand test manifest must explicitly exclude full-body state.")

    joint_names = manifest.get("generation", {}).get("controlled_joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != 14 or len(set(joint_names)) != 14:
        raise ValueError("ArmHack Stand manifest must define exactly 14 unique controlled arm joints.")
    non_arm_joints = [
        name
        for name in joint_names
        if not isinstance(name, str)
        or not name.startswith(("left_", "right_"))
        or not any(part in name for part in ("shoulder", "elbow", "wrist"))
    ]
    if non_arm_joints:
        raise ValueError(f"ArmHack Stand generated trajectory contains non-arm joints: {non_arm_joints}")

    bank_joint_names = pose_bank.get("joint_names")
    if not isinstance(bank_joint_names, list) or set(bank_joint_names) != set(joint_names):
        raise ValueError("Pose-bank joints do not match the 14-DoF ArmHack Stand manifest.")
    entries = pose_bank.get("poses")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("ArmHack Stand random pose bank must contain at least two poses.")

    bank_poses = np.asarray([entry.get("positions_rad") for entry in entries], dtype=np.float64)
    bank_velocity_limits = np.asarray(
        pose_bank.get("interpolation_velocity_limits_rad_s"), dtype=np.float64
    )
    reorder = [bank_joint_names.index(name) for name in joint_names]
    poses = bank_poses[:, reorder]
    velocity_limits = bank_velocity_limits[reorder]
    if poses.shape != (len(entries), len(joint_names)):
        raise ValueError(f"Pose bank has invalid shape {poses.shape}; expected ({len(entries)}, 14).")
    if velocity_limits.shape != (len(joint_names),) or np.any(velocity_limits <= 0.0):
        raise ValueError("Pose-bank interpolation velocity limits must be 14 positive values.")
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(velocity_limits)):
        raise ValueError("Pose bank contains non-finite poses or velocity limits.")

    expected_sha = manifest.get("random_pose_bank", {}).get("sha256")
    actual_sha = _sha256(pose_bank_path)
    if expected_sha != actual_sha:
        raise ValueError(
            "Pose-bank SHA-256 does not match manifest: "
            f"manifest={expected_sha}, actual={actual_sha}"
        )
    return joint_names, poses, entries, velocity_limits, pose_bank


def _minimum_jerk(start: np.ndarray, end: np.ndarray, frame_count: int) -> np.ndarray:
    phase = np.arange(1, frame_count + 1, dtype=np.float64) / frame_count
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return start[None, :] + blend[:, None] * (end - start)[None, :]


def _write_csv(path: Path, joint_names: list[str], trajectory: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["time_s", *joint_names])
        for frame_index, pose in enumerate(trajectory):
            writer.writerow(
                [f"{frame_index / fps:.8f}", *(f"{value:.8f}" for value in pose)]
            )
    temporary_path.replace(path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 0 or args.seed > 2**63 - 1:
        raise ValueError("seed must be within [0, 2^63-1].")
    if args.waypoint_count < 2:
        raise ValueError("waypoint-count must be at least 2.")
    if args.fps <= 0.0 or not math.isfinite(args.fps):
        raise ValueError("fps must be finite and positive.")
    if args.hold_s < 0.0 or not math.isfinite(args.hold_s):
        raise ValueError("hold-s must be finite and non-negative.")
    if args.transition_s <= 0.0 or not math.isfinite(args.transition_s):
        raise ValueError("transition-s must be finite and positive.")

    pose_bank_path = args.pose_bank.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    for path, label in ((pose_bank_path, "pose bank"), (manifest_path, "manifest")):
        if not path.is_file():
            raise FileNotFoundError(f"ArmHack Stand {label} does not exist: {path}")

    joint_names, poses, entries, velocity_limits, pose_bank = _load_contract(
        pose_bank_path, manifest_path
    )
    if args.waypoint_count > len(poses):
        raise ValueError(
            f"waypoint-count={args.waypoint_count} exceeds pose-bank size {len(poses)}."
        )

    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(poses), size=args.waypoint_count, replace=False)
    selected_poses = poses[selected_indices]
    hold_frame_count = max(int(round(args.hold_s * args.fps)), 1)

    frames: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    transition_durations_s: list[float] = []
    cursor = 0
    for waypoint_index, (bank_index, pose) in enumerate(
        zip(selected_indices, selected_poses, strict=True)
    ):
        pose_id = str(entries[int(bank_index)].get("pose_id", f"pose_{int(bank_index):04d}"))
        if waypoint_index > 0:
            previous_pose = selected_poses[waypoint_index - 1]
            required_duration_s = float(
                np.max(1.875 * np.abs(pose - previous_pose) / velocity_limits)
            )
            requested_duration_s = max(args.transition_s, required_duration_s)
            transition_frame_count = max(int(math.ceil(requested_duration_s * args.fps)), 1)
            actual_duration_s = transition_frame_count / args.fps
            transition_start_s = cursor / args.fps
            frames.extend(_minimum_jerk(previous_pose, pose, transition_frame_count))
            cursor += transition_frame_count
            transition_durations_s.append(actual_duration_s)
            timeline.append(
                {
                    "kind": "velocity_safe_transition",
                    "label": f"waypoint_{waypoint_index:02d}_to_{waypoint_index + 1:02d}",
                    "start_s": transition_start_s,
                    "end_s": cursor / args.fps,
                    "duration_s": actual_duration_s,
                    "minimum_required_duration_s": required_duration_s,
                }
            )
        hold_start_s = cursor / args.fps
        frames.extend(np.repeat(pose[None, :], hold_frame_count, axis=0))
        cursor += hold_frame_count
        timeline.append(
            {
                "kind": "static_hold",
                "label": f"waypoint_{waypoint_index + 1:02d}:{pose_id}",
                "start_s": hold_start_s,
                "end_s": cursor / args.fps,
            }
        )

    trajectory = np.asarray(frames, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 14 or not np.all(np.isfinite(trajectory)):
        raise RuntimeError("Generated ArmHack Stand trajectory is not a finite (N, 14) array.")
    bank_min = np.min(poses, axis=0)
    bank_max = np.max(poses, axis=0)
    generated_min = np.min(trajectory, axis=0)
    generated_max = np.max(trajectory, axis=0)
    tolerance = 1.0e-10
    if np.any(generated_min < bank_min - tolerance) or np.any(generated_max > bank_max + tolerance):
        raise RuntimeError("Generated trajectory exceeded the randomized training pose-bank range.")

    measured_velocity = np.abs(np.diff(trajectory, axis=0)) * args.fps
    measured_max_velocity = (
        np.max(measured_velocity, axis=0) if len(measured_velocity) else np.zeros(14)
    )
    if np.any(measured_max_velocity > velocity_limits + 1.0e-8):
        raise RuntimeError("Generated trajectory exceeded a pose-bank interpolation velocity limit.")

    _write_csv(output_path, joint_names, trajectory, args.fps)
    metadata = {
        "schema_version": 1,
        "data_scope": "arm_only_14_dof",
        "contains_full_body_state": False,
        "generation": {
            "method": "seeded pose-bank sampling without replacement plus quintic minimum-jerk interpolation",
            "seed": args.seed,
            "waypoint_count": args.waypoint_count,
            "requested_hold_s": args.hold_s,
            "actual_hold_s": hold_frame_count / args.fps,
            "nominal_transition_s": args.transition_s,
            "fps": args.fps,
            "frame_count": len(trajectory),
            "duration_s": (len(trajectory) - 1) / args.fps,
            "transition_duration_min_s": min(transition_durations_s),
            "transition_duration_max_s": max(transition_durations_s),
        },
        "source": {
            "pose_bank_path": str(pose_bank_path),
            "pose_bank_sha256": _sha256(pose_bank_path),
            "pose_bank_schema_version": pose_bank.get("schema_version"),
            "pose_bank_size": len(poses),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        "selected_waypoints": [
            {
                "sequence_index": sequence_index,
                "bank_index": int(bank_index),
                "pose_id": str(entries[int(bank_index)].get("pose_id", "")),
            }
            for sequence_index, bank_index in enumerate(selected_indices, start=1)
        ],
        "joint_names": joint_names,
        "joint_range_and_speed": {
            name: {
                "pose_bank_min_rad": float(bank_min[index]),
                "pose_bank_max_rad": float(bank_max[index]),
                "generated_min_rad": float(generated_min[index]),
                "generated_max_rad": float(generated_max[index]),
                "velocity_limit_rad_s": float(velocity_limits[index]),
                "measured_max_abs_velocity_rad_s": float(measured_max_velocity[index]),
            }
            for index, name in enumerate(joint_names)
        },
        "timeline": timeline,
        "output": {
            "csv_path": str(output_path),
            "csv_sha256": _sha256(output_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    temporary_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata_path.replace(metadata_path)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--waypoint-count", type=int, default=8)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--transition-s", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    metadata = generate(_parse_args())
    generation = metadata["generation"]
    print(
        "[ArmHack Stand trajectory] "
        f"seed={generation['seed']}, waypoints={generation['waypoint_count']}, "
        f"frames={generation['frame_count']}, duration={generation['duration_s']:.3f}s"
    )
    print(f"[ArmHack Stand trajectory] CSV: {metadata['output']['csv_path']}")


if __name__ == "__main__":
    main()
