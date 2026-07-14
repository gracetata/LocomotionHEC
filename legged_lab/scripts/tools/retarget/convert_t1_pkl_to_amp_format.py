#!/usr/bin/env python3
"""
Convert T1 retarget-format pkl files to legged_lab AMP format.

Differences between formats:
    1. T1 has no `loop_mode` field -> default to 0 (CLAMP)
    2. T1 `root_rot` is xyzw -> must reorder to wxyz
    3. T1 `dof_pos` has 27 columns (no head joints) -> expand to 29 with zeros at head positions
    4. GMR and Lab joint angles share the same XML/URDF zero-pose convention
    5. T1 has no `key_body_pos` -> derive from `local_body_pos` for 6 key bodies

Usage:
  python scripts/tools/retarget/convert_t1_pkl_to_amp_format.py

Output is written to:
    source/legged_lab/legged_lab/data/MotionData/t1_29dof_accad_g1used_50hz_amp_official/
"""

import argparse
import sys
import os
import types
import glob
import importlib

import numpy as np

# ---------------------------------------------------------------------------
# Numpy shim: pkl files saved with numpy >= 2.0 reference 'numpy._core',
# which doesn't exist in numpy 1.26.  Patch sys.modules before importing joblib.
# ---------------------------------------------------------------------------
if not hasattr(np, "_core"):
    import numpy.core as _nc
    _s = types.ModuleType("numpy._core")
    _s.__dict__.update(_nc.__dict__)
    sys.modules.setdefault("numpy._core", _s)
    for _sub in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        _full = "numpy._core." + _sub
        if _full not in sys.modules:
            try:
                sys.modules[_full] = importlib.import_module("numpy.core." + _sub)
            except ImportError:
                pass

import joblib  # noqa: E402  (import after shim)

from t1_lab_joint_offsets import T1_LAB_BAKED_ZERO_OFFSETS, compensate_t1_lab_baked_zero_offsets


def quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by xyzw quaternions.

    Args:
        quat: shape (T, 4), xyzw.
        vec: shape (T, K, 3).
    Returns:
        Rotated vectors with shape (T, K, 3).
    """
    q_xyz = quat[:, :3]  # (T, 3)
    q_w = quat[:, 3:4]   # (T, 1)
    q_xyz_e = q_xyz[:, None, :]  # (T, 1, 3)
    q_w_e = q_w[:, None, :]      # (T, 1, 1)
    t = 2.0 * np.cross(q_xyz_e, vec, axis=-1)
    return vec + q_w_e * t + np.cross(q_xyz_e, t, axis=-1)

# ---------------------------------------------------------------------------
# Joint order definitions
# ---------------------------------------------------------------------------

# Dataset dof order (27 joints, no head joints):
DATASET_DOF_NAMES = [
    "Left_Shoulder_Pitch",   # 0
    "Left_Shoulder_Roll",    # 1
    "Left_Elbow_Pitch",      # 2
    "Left_Elbow_Yaw",        # 3
    "Left_Wrist_Pitch",      # 4
    "Left_Wrist_Yaw",        # 5
    "Left_Hand_Roll",        # 6
    "Right_Shoulder_Pitch",  # 7
    "Right_Shoulder_Roll",   # 8
    "Right_Elbow_Pitch",     # 9
    "Right_Elbow_Yaw",       # 10
    "Right_Wrist_Pitch",     # 11
    "Right_Wrist_Yaw",       # 12
    "Right_Hand_Roll",       # 13
    "Waist",                 # 14
    "Left_Hip_Pitch",        # 15
    "Left_Hip_Roll",         # 16
    "Left_Hip_Yaw",          # 17
    "Left_Knee_Pitch",       # 18
    "Left_Ankle_Pitch",      # 19
    "Left_Ankle_Roll",       # 20
    "Right_Hip_Pitch",       # 21
    "Right_Hip_Roll",        # 22
    "Right_Hip_Yaw",         # 23
    "Right_Knee_Pitch",      # 24
    "Right_Ankle_Pitch",     # 25
    "Right_Ankle_Roll",      # 26
]

# URDF / IsaacLab joint order (29 joints, with head joints at 0 and 1):
URDF_DOF_NAMES = [
    "AAHead_yaw",            # 0
    "Head_pitch",            # 1
    "Left_Shoulder_Pitch",   # 2
    "Left_Shoulder_Roll",    # 3
    "Left_Elbow_Pitch",      # 4
    "Left_Elbow_Yaw",        # 5
    "Left_Wrist_Pitch",      # 6
    "Left_Wrist_Yaw",        # 7
    "Left_Hand_Roll",        # 8
    "Right_Shoulder_Pitch",  # 9
    "Right_Shoulder_Roll",   # 10
    "Right_Elbow_Pitch",     # 11
    "Right_Elbow_Yaw",       # 12
    "Right_Wrist_Pitch",     # 13
    "Right_Wrist_Yaw",       # 14
    "Right_Hand_Roll",       # 15
    "Waist",                 # 16
    "Left_Hip_Pitch",        # 17
    "Left_Hip_Roll",         # 18
    "Left_Hip_Yaw",          # 19
    "Left_Knee_Pitch",       # 20
    "Left_Ankle_Pitch",      # 21
    "Left_Ankle_Roll",       # 22
    "Right_Hip_Pitch",       # 23
    "Right_Hip_Roll",        # 24
    "Right_Hip_Yaw",         # 25
    "Right_Knee_Pitch",      # 26
    "Right_Ankle_Pitch",     # 27
    "Right_Ankle_Roll",      # 28
]

# Precompute: for each URDF index, which dataset index to copy from (-1 = zero)
_dataset_name_to_idx = {name: i for i, name in enumerate(DATASET_DOF_NAMES)}
URDF_FROM_DATASET = [
    _dataset_name_to_idx.get(name, -1) for name in URDF_DOF_NAMES
]

# ---------------------------------------------------------------------------
# Key body configuration (must match KEY_BODY_NAMES in t1_amp_env_cfg.py)
# ---------------------------------------------------------------------------
KEY_BODY_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
    "AL3",   # body after left shoulder-roll (equiv. G1 left_shoulder_roll_link)
    "AR3",   # body after right shoulder-roll
]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_pkl(src_path: str, dst_path: str) -> None:
    data = joblib.load(src_path)

    T = data["root_pos"].shape[0]

    # 1. loop_mode (default CLAMP = 0)
    loop_mode = int(data.get("loop_mode", 0))

    # 2. root_rot: xyzw → wxyz
    rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)  # (T, 4)
    rot_wxyz = np.concatenate(
        [rot_xyzw[:, 3:4], rot_xyzw[:, :3]], axis=1
    ).astype(np.float32)

    # 3. dof_pos: 27 → 29  (head joints set to 0)
    dof_27 = np.asarray(data["dof_pos"], dtype=np.float32)  # (T, 27)
    dof_29 = np.zeros((T, 29), dtype=np.float32)
    for urdf_i, ds_i in enumerate(URDF_FROM_DATASET):
        if ds_i >= 0:
            dof_29[:, urdf_i] = dof_27[:, ds_i]
    dof_29 = compensate_t1_lab_baked_zero_offsets(dof_29, URDF_DOF_NAMES)

    # 4. key_body_pos in world frame from local_body_pos + root transform
    link_list = data.get("link_body_list", [])
    local_pos = np.asarray(data.get("local_body_pos", np.zeros((T, 0, 3))), dtype=np.float32)
    key_body_pos = np.zeros((T, len(KEY_BODY_NAMES), 3), dtype=np.float32)
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    for i, name in enumerate(KEY_BODY_NAMES):
        if name in link_list:
            idx = link_list.index(name)
            if idx < local_pos.shape[1]:
                local_body = local_pos[:, idx : idx + 1, :]  # (T, 1, 3)
                world_body = quat_rotate_xyzw(root_rot_xyzw, local_body)[:, 0, :] + root_pos
                key_body_pos[:, i, :] = world_body

    out = {
        "fps": data["fps"],
        "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
        "root_rot": rot_wxyz,
        "dof_pos": dof_29,
        "key_body_pos": key_body_pos,
        "loop_mode": loop_mode,
    }
    for optional_key in ("task", "cleaning"):
        if optional_key in data:
            out[optional_key] = data[optional_key]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    joblib.dump(out, dst_path, protocol=4)
    print(f"  {os.path.basename(src_path)}: {T} frames → saved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default=None, help="Input directory containing raw 27-DoF T1 GMR pkl files.")
    parser.add_argument("--dst-dir", default=None, help="Output directory for 29-DoF AMP pkl files.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../../.."))

    src_dir = args.src_dir or os.path.join(
        workspace_root,
        "source/legged_lab/legged_lab/data/MotionData/t1_29dof_accad_g1used_50hz",
    )
    dst_dir = args.dst_dir or os.path.join(
        workspace_root,
        "source/legged_lab/legged_lab/data/MotionData/t1_29dof_accad_g1used_50hz_amp_official",
    )

    if not os.path.isdir(src_dir):
        print(f"ERROR: source directory not found: {src_dir}")
        sys.exit(1)

    pkl_files = sorted(glob.glob(os.path.join(src_dir, "*.pkl")))
    if not pkl_files:
        print(f"ERROR: no .pkl files found in {src_dir}")
        sys.exit(1)

    print(f"Converting {len(pkl_files)} pkl files")
    print(f"  src: {src_dir}")
    print(f"  dst: {dst_dir}")
    print()

    for src_path in pkl_files:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        convert_pkl(src_path, dst_path)

    print(f"\nDone. {len(pkl_files)} files written to {dst_dir}")


if __name__ == "__main__":
    main()
