"""Repository-relative ArmHack reference-data paths and loaders."""

from __future__ import annotations

import json
import math
from pathlib import Path

from legged_lab import LEGGED_LAB_ROOT_DIR


REFERENCE_DATA_ROOT_RELATIVE = Path("Reference Data") / "ArmHack"
STAND_ARM_MOTION_RELATIVE_PATH = (
    REFERENCE_DATA_ROOT_RELATIVE / "StandPerturb" / "g1_arm_trajectory_named_50hz.csv"
)
WALK_ARM_POSE_SET_RELATIVE_PATH = (
    REFERENCE_DATA_ROOT_RELATIVE / "WalkPerturbFinetune" / "g1_arm_pose_set.json"
)

_LEGGED_LAB_PROJECT_DIR = Path(LEGGED_LAB_ROOT_DIR).resolve().parents[2]


def resolve_reference_data_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the ``legged_lab`` project directory."""

    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"ArmHack reference-data path must be repository-relative, got: {path}")
    resolved_path = (_LEGGED_LAB_PROJECT_DIR / path).resolve()
    if not resolved_path.is_relative_to(_LEGGED_LAB_PROJECT_DIR):
        raise ValueError(f"ArmHack reference-data path escapes the project directory: {path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"ArmHack reference-data file not found: {resolved_path}")
    return resolved_path


def load_walk_arm_pose_set(relative_path: str | Path = WALK_ARM_POSE_SET_RELATIVE_PATH) -> list[list[float]]:
    """Load and validate named left/right 7-DoF arm poses from JSON."""

    resolved_path = resolve_reference_data_path(relative_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected_joint_order = [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    if payload.get("units") != "rad":
        raise ValueError(f"Walk arm-pose units must be 'rad': {resolved_path}")
    if payload.get("joint_order_per_arm") != expected_joint_order:
        raise ValueError(f"Unexpected walk arm-pose joint order: {resolved_path}")

    poses = payload.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"Walk arm-pose JSON must contain a non-empty 'poses' list: {resolved_path}")

    paired_poses: list[list[float]] = []
    pose_names: set[str] = set()
    for pose in poses:
        if not isinstance(pose, dict):
            raise ValueError(f"Each walk arm pose must be an object: {resolved_path}")
        name = pose.get("name")
        if not isinstance(name, str) or not name or name in pose_names:
            raise ValueError(f"Walk arm-pose names must be non-empty and unique: {resolved_path}")
        pose_names.add(name)

        left = pose.get("left")
        right = pose.get("right")
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != 7 or len(right) != 7:
            raise ValueError(f"Pose {name!r} must contain exactly seven left and seven right values.")
        values = [float(value) for value in left + right]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Pose {name!r} contains a non-finite joint value.")

        paired_poses.append(
            [
                float(left[0]),
                float(right[0]),
                float(left[1]),
                float(right[1]),
                float(left[2]),
                float(right[2]),
                float(left[3]),
                float(right[3]),
                float(left[4]),
                float(right[4]),
                float(left[5]),
                float(right[5]),
                float(left[6]),
                float(right[6]),
            ]
        )
    return paired_poses
