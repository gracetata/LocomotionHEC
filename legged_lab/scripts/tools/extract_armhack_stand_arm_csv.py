#!/usr/bin/env python3
"""Extract the 14 named G1 arm joints from the ArmHack full-body SDK CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
G1_ARM_JOINT_NAMES = G1_FULL_BODY_SDK_JOINT_NAMES[15:]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_input() -> Path:
    return (
        _project_root()
        / "Reference Data"
        / "ArmHack"
        / "StandPerturb"
        / "raw"
        / "g1_full_body_motion_sdk_50hz.csv"
    )


def _default_output() -> Path:
    return (
        _project_root()
        / "Reference Data"
        / "ArmHack"
        / "StandPerturb"
        / "g1_arm_trajectory_named_50hz.csv"
    )


def extract_arm_trajectory(input_path: Path, output_path: Path) -> int:
    """Write a strict CSV containing time and the 14 named arm joint positions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.reader(source_handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Source CSV has no header: {input_path}")
        if "time_s" not in header:
            raise ValueError(f"Source CSV is missing time_s: {input_path}")
        time_index = header.index("time_s")
        q_indices = [header.index(f"q{index}") for index in range(29)]

        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.writer(output_handle, lineterminator="\n")
            writer.writerow(["time_s", *G1_ARM_JOINT_NAMES])
            for row in reader:
                if len(row) == len(header):
                    full_body_values = [float(row[index]) for index in q_indices]
                else:
                    # The source has unquoted commas in some natural-language fields.
                    # Its final 29 values remain the SDK-order q0..q28 joint positions.
                    full_body_values = [float(value) for value in row[-29:]]
                writer.writerow([float(row[time_index]), *full_body_values[15:]])
                row_count += 1
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_default_input())
    parser.add_argument("--output", type=Path, default=_default_output())
    args = parser.parse_args()

    row_count = extract_arm_trajectory(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    print(f"Wrote {row_count} arm-trajectory rows to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
