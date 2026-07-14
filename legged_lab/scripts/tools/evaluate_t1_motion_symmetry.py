#!/usr/bin/env python3
"""Evaluate a T1 motion with the current T1 mirror transform."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import torch


DEFAULT_MOTION = (
    "source/legged_lab/legged_lab/data/MotionData/"
    "t1_29dof_accad_g1used_50hz_amp_official/B10_-__Walk_turn_left_45_stageii.pkl"
)

NO_HEAD_JOINT_NAMES = [
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


def install_numpy_core_pickle_shim() -> None:
    if hasattr(np, "_core"):
        return

    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule_name in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        full_name = f"numpy._core.{submodule_name}"
        if full_name not in sys.modules:
            sys.modules[full_name] = importlib.import_module(f"numpy.core.{submodule_name}")


def load_t1_symmetry(repo_root: Path):
    symmetry_path = repo_root / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/symmetry/t1.py"
    module_spec = importlib.util.spec_from_file_location("t1_symmetry_eval", symmetry_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def evaluate_window(t1_symmetry, joint_pos_eval: torch.Tensor, fps: float, max_lag: int | None):
    if joint_pos_eval.shape[0] < 3:
        raise ValueError(f"Need at least 3 frames for lag search, got {joint_pos_eval.shape[0]}")

    mirrored = t1_symmetry._switch_t1_27dof_joints_left_right(joint_pos_eval)
    framewise_rmse = torch.sqrt(torch.mean((mirrored - joint_pos_eval) ** 2)).item()
    mean_joint_std = torch.mean(torch.std(joint_pos_eval, dim=0)).item()

    lag_limit = joint_pos_eval.shape[0] - 2
    if max_lag is not None:
        lag_limit = min(lag_limit, max_lag)
    if lag_limit < 1:
        raise ValueError(f"No valid lag for {joint_pos_eval.shape[0]} frames")

    best_lag = 1
    best_mse = float("inf")
    for lag in range(1, lag_limit + 1):
        mse = torch.mean((mirrored[:-lag] - joint_pos_eval[lag:]) ** 2).item()
        if mse < best_mse:
            best_lag = lag
            best_mse = mse

    best_rmse = best_mse**0.5
    normalized_best_rmse = best_rmse / (mean_joint_std + 1e-8)
    per_joint_rmse = torch.sqrt(torch.mean((mirrored[:-best_lag] - joint_pos_eval[best_lag:]) ** 2, dim=0))
    return {
        "frames": joint_pos_eval.shape[0],
        "framewise_rmse": framewise_rmse,
        "mean_joint_std": mean_joint_std,
        "best_lag": best_lag,
        "best_lag_seconds": best_lag / fps,
        "best_rmse": best_rmse,
        "normalized_best_rmse": normalized_best_rmse,
        "per_joint_rmse": per_joint_rmse,
    }


def print_result(prefix: str, result: dict, top_k: int) -> None:
    print(f"{prefix} frames={result['frames']}")
    print(f"{prefix} framewise_mirror_rmse={result['framewise_rmse']:.6f} rad")
    print(
        f"{prefix} best_temporal_lag="
        f"{result['best_lag']} frames ({result['best_lag_seconds']:.3f}s), "
        f"best_rmse={result['best_rmse']:.6f} rad, "
        f"best_rmse/mean_joint_std={result['normalized_best_rmse']:.6f}"
    )
    print(f"{prefix} largest per-joint RMSE at best lag:")
    per_joint_rmse = result["per_joint_rmse"]
    for joint_index in torch.argsort(per_joint_rmse, descending=True)[:top_k].tolist():
        print(f"  {joint_index:2d} {NO_HEAD_JOINT_NAMES[joint_index]:24s} {per_joint_rmse[joint_index].item():.6f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate T1 motion data symmetry using the current mirror transform.")
    parser.add_argument("--motion", type=str, default=DEFAULT_MOTION, help="Path to a converted 29-DoF T1 AMP pickle.")
    parser.add_argument("--start_frame", type=int, default=0, help="First frame included in the evaluation window.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames included in the evaluation window.")
    parser.add_argument("--trim_frac", type=float, default=0.1, help="Fraction trimmed from each end for lag search.")
    parser.add_argument("--max_lag", type=int, default=None, help="Maximum temporal lag to consider, in frames.")
    parser.add_argument("--scan_window_frames", type=int, default=None, help="Scan all local windows of this frame length.")
    parser.add_argument("--scan_stride_frames", type=int, default=None, help="Stride for local window scanning.")
    parser.add_argument("--scan_top_k", type=int, default=5, help="Number of lowest-error local windows to print.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of highest-error joints to print.")
    args = parser.parse_args()

    install_numpy_core_pickle_shim()
    repo_root = Path(__file__).resolve().parents[2]
    motion_path = Path(args.motion)
    if not motion_path.is_absolute():
        motion_path = repo_root / motion_path

    t1_symmetry = load_t1_symmetry(repo_root)
    motion = joblib.load(motion_path)
    fps = float(motion.get("fps", 50.0))
    joint_pos = torch.as_tensor(motion["dof_pos"], dtype=torch.float32)
    if joint_pos.shape[-1] == 29:
        joint_pos = joint_pos[:, 2:]
    elif joint_pos.shape[-1] != 27:
        raise ValueError(f"Expected 27-DoF or 29-DoF motion, got shape {tuple(joint_pos.shape)}")

    raw_start_frame = args.start_frame
    raw_end_frame = joint_pos.shape[0] if args.num_frames is None else args.start_frame + args.num_frames
    joint_pos_window = joint_pos[raw_start_frame:raw_end_frame]

    trim = int(joint_pos_window.shape[0] * args.trim_frac)
    if trim > 0:
        joint_pos_eval = joint_pos_window[trim:-trim]
        eval_start_frame = raw_start_frame + trim
        eval_end_frame = raw_end_frame - trim
    else:
        joint_pos_eval = joint_pos_window
        eval_start_frame = raw_start_frame
        eval_end_frame = raw_end_frame

    result = evaluate_window(t1_symmetry, joint_pos_eval, fps=fps, max_lag=args.max_lag)

    print("[T1 Motion Symmetry Eval] Motion:", motion_path)
    print(
        f"[T1 Motion Symmetry Eval] total_frames={joint_pos.shape[0]} fps={fps:g} "
        f"raw_window=[{raw_start_frame}, {raw_end_frame}) trim_frac={args.trim_frac:g} "
        f"eval_window=[{eval_start_frame}, {eval_end_frame})"
    )
    print_result("[T1 Motion Symmetry Eval]", result, args.top_k)

    if args.scan_window_frames is not None:
        stride = args.scan_stride_frames if args.scan_stride_frames is not None else max(1, args.scan_window_frames // 10)
        scan_results = []
        for start_frame in range(0, joint_pos.shape[0] - args.scan_window_frames + 1, stride):
            window = joint_pos[start_frame : start_frame + args.scan_window_frames]
            window_result = evaluate_window(t1_symmetry, window, fps=fps, max_lag=args.max_lag)
            scan_results.append((window_result["best_rmse"], start_frame, window_result))

        print(
            f"[T1 Motion Symmetry Eval] best {args.scan_top_k} local windows "
            f"for scan_window_frames={args.scan_window_frames}, stride={stride}:"
        )
        for _, start_frame, window_result in sorted(scan_results, key=lambda item: item[0])[: args.scan_top_k]:
            print(
                f"  start={start_frame:3d} end={start_frame + args.scan_window_frames:3d} "
                f"lag={window_result['best_lag']:2d} ({window_result['best_lag_seconds']:.3f}s) "
                f"best_rmse={window_result['best_rmse']:.6f} rad "
                f"norm={window_result['normalized_best_rmse']:.6f} "
                f"framewise={window_result['framewise_rmse']:.6f} rad"
            )


if __name__ == "__main__":
    main()