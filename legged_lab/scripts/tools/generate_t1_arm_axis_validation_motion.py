#!/usr/bin/env python3
"""Generate a T1 arm-axis validation motion for IsaacLab playback.

The output keeps the robot at the current Lab zero/natural arm pose and moves
only Shoulder Pitch plus Elbow Yaw. It is meant for visual validation of arm
joint semantics through ``visualize_t1_motion_as_policy_actions.py``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import joblib
import numpy as np


LAB_DOF_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

STANDING_LEG_POSE = {
    "Left_Hip_Pitch": -0.2,
    "Right_Hip_Pitch": -0.2,
    "Left_Knee_Pitch": 0.45,
    "Right_Knee_Pitch": 0.45,
    "Left_Ankle_Pitch": -0.25,
    "Right_Ankle_Pitch": -0.25,
}

DEFAULT_TEMPLATE = (
    "source/legged_lab/legged_lab/data/MotionData/"
    "t1_29dof_accad_g1used_50hz_amp_official/B10_-__Walk_turn_left_45_stageii.pkl"
)
DEFAULT_OUTPUT = "outputs/t1_arm_axis_validation_20260506/t1_lab_zero_shoulder_pitch_elbow_yaw_swing.pkl"


def install_numpy_core_pickle_shim() -> None:
    if hasattr(np, "_core"):
        return

    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule_name in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        sys.modules.setdefault(
            f"numpy._core.{submodule_name}",
            importlib.import_module(f"numpy.core.{submodule_name}"),
        )


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="29-DoF AMP pickle to copy metadata from.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output pickle path.")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--neutral-seconds", type=float, default=1.5)
    parser.add_argument("--shoulder-amp", type=float, default=0.25)
    parser.add_argument("--elbow-yaw-amp", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_numpy_core_pickle_shim()

    template_path = resolve_repo_path(args.template)
    output_path = resolve_repo_path(args.output)
    motion = joblib.load(template_path)

    frame_count = int(round(args.duration * args.fps))
    if frame_count < 2:
        raise ValueError("duration must produce at least two frames")

    dof_pos = np.zeros((frame_count, len(LAB_DOF_NAMES)), dtype=np.float32)
    for joint_name, joint_value in STANDING_LEG_POSE.items():
        dof_pos[:, LAB_DOF_NAMES.index(joint_name)] = joint_value

    neutral_frames = min(frame_count, int(round(args.neutral_seconds * args.fps)))
    active_frames = frame_count - neutral_frames
    if active_frames > 0:
        phase = np.linspace(0.0, 2.0 * np.pi, active_frames, endpoint=False, dtype=np.float32)
        swing = np.sin(phase)
        dof_pos[neutral_frames:, LAB_DOF_NAMES.index("Left_Shoulder_Pitch")] = args.shoulder_amp * swing
        dof_pos[neutral_frames:, LAB_DOF_NAMES.index("Right_Shoulder_Pitch")] = -args.shoulder_amp * swing
        dof_pos[neutral_frames:, LAB_DOF_NAMES.index("Left_Elbow_Yaw")] = args.elbow_yaw_amp * swing
        dof_pos[neutral_frames:, LAB_DOF_NAMES.index("Right_Elbow_Yaw")] = args.elbow_yaw_amp * swing

    generated = dict(motion)
    for key in ("root_pos", "root_rot", "key_body_pos"):
        value = motion[key]
        if value.shape[0] >= frame_count:
            generated[key] = value[:frame_count].astype(np.float32, copy=True)
        else:
            repeat_count = int(np.ceil(frame_count / value.shape[0]))
            generated[key] = np.concatenate([value] * repeat_count, axis=0)[:frame_count].astype(np.float32, copy=True)

    generated["fps"] = int(args.fps)
    generated["dof_pos"] = dof_pos
    generated["loop_mode"] = int(motion.get("loop_mode", 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(generated, output_path)
    print(output_path)
    print(f"frames={frame_count} fps={args.fps}")
    print(f"shoulder_amp={args.shoulder_amp} elbow_yaw_amp={args.elbow_yaw_amp}")
    print("moving_joints=Left_Shoulder_Pitch,Right_Shoulder_Pitch,Left_Elbow_Yaw,Right_Elbow_Yaw")


if __name__ == "__main__":
    main()