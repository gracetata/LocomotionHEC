#!/usr/bin/env python3
"""Validate the two ArmHack reference-data packages without starting Isaac Sim."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = PROJECT_ROOT / "Reference Data" / "ArmHack"
STAND_RAW_CSV = REFERENCE_ROOT / "StandPerturb" / "raw" / "g1_full_body_motion_sdk_50hz.csv"
STAND_ARM_CSV = REFERENCE_ROOT / "StandPerturb" / "g1_arm_trajectory_named_50hz.csv"
WALK_POSE_JSON = REFERENCE_ROOT / "WalkPerturbFinetune" / "g1_arm_pose_set.json"
WALK_NAV2_CSV = REFERENCE_ROOT / "WalkPerturbFinetune" / "nav2_cmd_vel_raw_success.csv"
STAND_TEST_DATA_ROOT = REFERENCE_ROOT / "StandPerturb" / "TestData" / "ArmOnly"
STAND_TEST_DATA_MANIFEST = STAND_TEST_DATA_ROOT / "manifest.json"

EXPECTED_HASHES = {
    STAND_RAW_CSV: "b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50",
    STAND_ARM_CSV: "afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601",
    WALK_NAV2_CSV: "76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a",
}

ARM_JOINT_COLUMNS = [
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"ArmHack reference data is missing: {path}")


def check_hashes(stand_only: bool = False) -> None:
    for path, expected in EXPECTED_HASHES.items():
        if stand_only and path == WALK_NAV2_CSV:
            continue
        require_file(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
        print(f"[OK] SHA-256 {path.relative_to(PROJECT_ROOT)}: {actual}")


def check_stand_arm_csv() -> None:
    with STAND_ARM_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_header = ["time_s", *ARM_JOINT_COLUMNS]
        if header != expected_header:
            raise ValueError(f"Unexpected Stand arm CSV columns: {header}")
        rows = 0
        for row in reader:
            if len(row) != len(expected_header):
                raise ValueError(f"Stand arm CSV row {rows + 2} has {len(row)} columns")
            values = [float(value) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Stand arm CSV row {rows + 2} contains non-finite values")
            rows += 1
    if rows != 20122:
        raise ValueError(f"Stand arm CSV expected 20122 rows, got {rows}")
    print(f"[OK] Stand arm trajectory: {rows} rows, {len(expected_header)} columns")


def check_walk_pose_json() -> None:
    require_file(WALK_POSE_JSON)
    with WALK_POSE_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_order = [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    if payload.get("units") != "rad" or payload.get("joint_order_per_arm") != expected_order:
        raise ValueError("Walk pose JSON units or joint order is invalid")
    poses = payload.get("poses", [])
    if [pose.get("name") for pose in poses] != ["pos1_back", "pos2_down", "pos3_front"]:
        raise ValueError("Walk pose JSON must contain pos1_back, pos2_down and pos3_front in order")
    for pose in poses:
        values = [float(value) for side in ("left", "right") for value in pose.get(side, [])]
        if len(values) != 14 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Walk pose {pose.get('name')} must contain 14 finite values")
    print("[OK] Walk arm pose set: 3 poses, 14 joints per pose")


def check_walk_nav2_csv() -> None:
    require_file(WALK_NAV2_CSV)
    required = {
        "combo",
        "planner",
        "controller",
        "scenario",
        "scenario_family",
        "goal_id",
        "augmentation",
        "t",
        "vx",
        "vy",
        "wz",
    }
    velocity_ranges = {name: [math.inf, -math.inf] for name in ("vx", "vy", "wz")}
    augmentations: Counter[str] = Counter()
    groups: set[tuple[str, ...]] = set()
    rows = 0
    with WALK_NAV2_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Nav2 CSV is missing columns: {sorted(missing)}")
        for row in reader:
            values = {name: float(row[name]) for name in velocity_ranges}
            time_value = float(row["t"])
            if not math.isfinite(time_value) or not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Nav2 CSV row {rows + 2} contains non-finite command data")
            for name, value in values.items():
                velocity_ranges[name][0] = min(velocity_ranges[name][0], value)
                velocity_ranges[name][1] = max(velocity_ranges[name][1], value)
            augmentation = row.get("augmentation", "none") or "none"
            augmentations[augmentation] += 1
            groups.add(
                tuple(
                    row.get(name, "")
                    for name in (
                        "combo",
                        "planner",
                        "controller",
                        "scenario",
                        "goal_id",
                        "augmentation",
                        "scenario_family",
                    )
                )
            )
            rows += 1
    if rows != 331010 or len(groups) != 445 or augmentations != Counter({"none": 331010}):
        raise ValueError(
            f"Unexpected Nav2 dataset shape: rows={rows}, groups={len(groups)}, augmentations={dict(augmentations)}"
        )
    print(
        f"[OK] Walk Nav2 commands: {rows} rows, {len(groups)} groups, "
        f"ranges={velocity_ranges}, augmentations={dict(augmentations)}"
    )


def _stand_arm_global_bounds() -> tuple[list[float], list[float]]:
    lower = [math.inf] * len(ARM_JOINT_COLUMNS)
    upper = [-math.inf] * len(ARM_JOINT_COLUMNS)
    with STAND_ARM_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for index, joint_name in enumerate(ARM_JOINT_COLUMNS):
                value = float(row[joint_name])
                lower[index] = min(lower[index], value)
                upper[index] = max(upper[index], value)
    return lower, upper


def _check_visualization_csv(path: Path, source_lower: list[float], source_upper: list[float]) -> tuple[int, float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_header = ["time_s", *ARM_JOINT_COLUMNS]
        if header != expected_header:
            raise ValueError(f"Unexpected visualization CSV columns in {path.name}: {header}")

        rows = 0
        previous_time: float | None = None
        previous_pose: list[float] | None = None
        final_time = 0.0
        max_frame_delta = 0.0
        for row in reader:
            if len(row) != len(expected_header):
                raise ValueError(f"Visualization CSV {path.name} row {rows + 2} has {len(row)} columns")
            values = [float(value) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Visualization CSV {path.name} row {rows + 2} contains non-finite values")
            time_value, pose = values[0], values[1:]
            if rows == 0 and abs(time_value) > 1.0e-9:
                raise ValueError(f"Visualization CSV {path.name} must start at time_s=0")
            if previous_time is not None and not math.isclose(time_value - previous_time, 0.02, abs_tol=1.0e-8):
                raise ValueError(f"Visualization CSV {path.name} is not uniformly sampled at 50 Hz")
            if previous_pose is not None:
                max_frame_delta = max(
                    max_frame_delta,
                    max(abs(value - previous) for value, previous in zip(pose, previous_pose, strict=True)),
                )
            for index, value in enumerate(pose):
                if value < source_lower[index] - 1.0e-7 or value > source_upper[index] + 1.0e-7:
                    raise ValueError(
                        f"Visualization CSV {path.name} leaves source joint bounds at "
                        f"{ARM_JOINT_COLUMNS[index]}={value}"
                    )
            previous_time = time_value
            previous_pose = pose
            final_time = time_value
            rows += 1
    if rows == 0:
        raise ValueError(f"Visualization CSV {path.name} has no data rows")
    return rows, final_time, max_frame_delta


def check_stand_arm_only_test_data() -> None:
    require_file(STAND_TEST_DATA_MANIFEST)
    with STAND_TEST_DATA_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    source = manifest.get("source", {})
    generation = manifest.get("generation", {})
    if source.get("sha256") != EXPECTED_HASHES[STAND_ARM_CSV]:
        raise ValueError("Stand test data was not built from the canonical arm CSV")
    if manifest.get("data_scope") != "arm_only_14_dof" or manifest.get("contains_full_body_state") is not False:
        raise ValueError("Stand test data must explicitly contain arm-only 14-DoF targets")
    if generation.get("controlled_joint_names") != ARM_JOINT_COLUMNS:
        raise ValueError("Stand test data joint list must contain exactly the canonical 14 arm joints")
    if not math.isclose(float(generation.get("trajectory_speed_scale", -1.0)), 1.0, abs_tol=1.0e-12):
        raise ValueError("Stand trajectory tests must use the original 1.0x arm-trajectory speed")
    if generation.get("seed") != 20260714 or generation.get("runtime_random_sampling") is not False:
        raise ValueError("Stand visualization suite must use seed 20260714 and disable runtime sampling")
    if len(manifest.get("representative_poses", [])) != 6:
        raise ValueError("Stand visualization suite must contain 6 representative poses")
    if len(manifest.get("synthesized_poses", [])) != 3:
        raise ValueError("Stand visualization suite must contain 3 synthesized poses")
    if len(manifest.get("representative_trajectories", [])) != 4:
        raise ValueError("Stand visualization suite must contain 4 representative trajectories")
    if len(manifest.get("synthesized_trajectories", [])) != 3:
        raise ValueError("Stand visualization suite must contain 3 synthesized trajectories")
    for trajectory in [
        *manifest.get("representative_trajectories", []),
        *manifest.get("synthesized_trajectories", []),
    ]:
        if not math.isclose(float(trajectory.get("equivalent_source_speed", -1.0)), 1.0, abs_tol=1.0e-12):
            raise ValueError(f"Stand trajectory is not 1.0x: {trajectory.get('path', '<unknown>')}")

    generated_hashes = manifest.get("generated_file_sha256", {})
    actual_csv_names = {
        path.relative_to(STAND_TEST_DATA_ROOT).as_posix() for path in STAND_TEST_DATA_ROOT.rglob("*.csv")
    }
    if set(generated_hashes) != actual_csv_names:
        raise ValueError("Stand visualization manifest and generated CSV file set differ")

    source_lower, source_upper = _stand_arm_global_bounds()
    checked_trajectory_files = 0
    for file_name, expected_hash in sorted(generated_hashes.items()):
        path = STAND_TEST_DATA_ROOT / file_name
        require_file(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for visualization file {file_name}")
        if file_name == "poses/arm_pose_catalog.csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 9:
                raise ValueError(f"Visualization pose catalog expected 9 rows, got {len(rows)}")
            continue

        rows, final_time, max_frame_delta = _check_visualization_csv(path, source_lower, source_upper)
        checked_trajectory_files += 1
        if file_name.startswith("poses/representative/") or file_name.startswith("poses/synthesized/"):
            if rows != 1001 or not math.isclose(final_time, 20.0, abs_tol=1.0e-8):
                raise ValueError(f"Individual arm pose {file_name} must hold for 20 s at 50 Hz")
            if max_frame_delta > 1.0e-10:
                raise ValueError(f"Static arm pose {file_name} must not move")
        if rows > 1 and max_frame_delta > 0.04:
            raise ValueError(
                f"Visualization CSV {file_name} has a discontinuous frame jump: {max_frame_delta:.6f} rad"
            )
        if file_name.startswith("trajectories/representative/") and (
            rows != 251 or not math.isclose(final_time, 5.0, abs_tol=1.0e-8)
        ):
            raise ValueError(f"Representative trajectory {file_name} must contain 5 s at 50 Hz and 1.0x speed")
        if file_name.startswith("trajectories/synthesized/") and (
            rows != 251 or not math.isclose(final_time, 5.0, abs_tol=1.0e-8)
        ):
            raise ValueError(f"Synthesized trajectory {file_name} must contain 5 s at 50 Hz and 1.0x speed")
        if final_time < 0.0:
            raise ValueError(f"Visualization CSV {file_name} has a negative duration")

    all_file = manifest.get("files", {}).get("all", {})
    if float(all_file.get("duration_s", -1.0)) < 100.0:
        raise ValueError("Unexpectedly short deterministic all-sequence duration")
    print(
        "[OK] Stand arm-only test data: 6 representative poses, 4 representative trajectories, "
        f"3 synthesized poses, 3 synthesized trajectories, {checked_trajectory_files} playback CSVs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stand-only", action="store_true", help="Validate only ArmHack Stand data.")
    args = parser.parse_args()
    check_hashes(stand_only=args.stand_only)
    check_stand_arm_csv()
    check_stand_arm_only_test_data()
    if not args.stand_only:
        check_walk_pose_json()
        check_walk_nav2_csv()
    print("[PASS] ArmHack Stand data checks passed." if args.stand_only else "[PASS] All ArmHack data checks passed.")


if __name__ == "__main__":
    main()
