#!/usr/bin/env python3
"""Generate deterministic 50 Hz hold CSVs for explicit ArmHack arm poses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


JOINT_NAMES = [
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

POSES = {
    "back": [0.91, 0.52, 0.11, 0.01, -0.12, -1.03, 0.01, 0.91, -0.52, -0.11, 0.01, 0.12, -1.03, -0.01],
    "down": [0.2504, 0.265, -0.0919, 0.8356, 0.0031, 0.0104, -0.0102, 0.2504, -0.265, 0.0919, 0.8356, -0.0031, 0.0104, 0.0102],
    "front": [0.27, 0.79, -0.22, -0.49, 0.85, 0.4, 0.05, 0.27, -0.79, 0.22, -0.49, -0.85, 0.4, -0.05],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.rate_hz <= 0.0:
        raise ValueError("duration and rate-hz must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = int(round(args.duration * args.rate_hz)) + 1
    for pose_name, positions in POSES.items():
        output_path = args.output_dir / f"armhack_explicit_{pose_name}_hold_{args.duration:g}s_{args.rate_hz:g}hz.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", *JOINT_NAMES])
            for sample_index in range(sample_count):
                writer.writerow([f"{sample_index / args.rate_hz:.8f}", *[f"{value:.8f}" for value in positions]])
        print(output_path)


if __name__ == "__main__":
    main()
