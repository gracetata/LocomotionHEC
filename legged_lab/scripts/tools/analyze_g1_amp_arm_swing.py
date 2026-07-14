#!/usr/bin/env python3
"""Compare G1 AMP policy arm swing against CMU washed reference motions.

The script runs a headless policy rollout, extracts arm joints and wrist paths,
selects reference clips with similar root velocity, synchronizes both sides by
left-foot touchdown gait cycles, and writes a JSON report plus phase plots.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "legged_lab"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))

import cli_args  # isort: skip


DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/g1_amp/"
    "2026-06-12_19-45-54_orig_g1_29dof_cmu_walk_washed_strict_upright_resume3999_2000/"
    "model_5998.pt"
)
DEFAULT_MOTION_DIR = (
    "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_washed_50hz"
)
DEFAULT_OUTPUT_DIR = (
    "logs/rsl_rl/g1_amp/"
    "2026-06-12_19-45-54_orig_g1_29dof_cmu_walk_washed_strict_upright_resume3999_2000/"
    "arm_swing_analysis"
)

ARM_JOINT_NAMES = [
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
CORE_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:8]
LEG_COUPLING_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
]
G1_LOCOMOTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
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
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
WRIST_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
SHOULDER_BODY_NAMES = ["left_shoulder_roll_link", "right_shoulder_roll_link"]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Strict-v0",
    help="Training task used for checkpoint loading and rollout.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--motion_dir", type=str, default=DEFAULT_MOTION_DIR, help="Reference motion pickle directory.")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/NPZ/PNG outputs.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of rollout environments.")
parser.add_argument("--max_steps", type=int, default=900, help="Number of policy steps to sample.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Initial rollout steps excluded from analysis.")
parser.add_argument("--phase_bins", type=int, default=128, help="Gait phase bins per left-foot cycle.")
parser.add_argument("--reference_top_k", type=int, default=8, help="Number of closest reference clips to aggregate.")
parser.add_argument("--command_lin_x", type=float, default=0.82, help="Fixed body-frame x velocity command.")
parser.add_argument("--command_lin_y", type=float, default=0.0, help="Fixed body-frame y velocity command.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw-rate command.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--robot_asset", type=str, default="g1_29dof", help="G1 robot asset preset.")
parser.add_argument("--disable_rsi", action="store_true", default=True, help="Disable reset-from-reference during rollout.")
parser.add_argument(
    "--keep_rsi",
    action="store_false",
    dest="disable_rsi",
    help="Keep configured reset-from-reference event.",
)
parser.add_argument(
    "--disable_randomization",
    action="store_true",
    default=True,
    help="Disable domain randomization events for cleaner arm comparison.",
)
parser.add_argument(
    "--keep_randomization",
    action="store_false",
    dest="disable_randomization",
    help="Keep configured randomization events.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(checkpoint=DEFAULT_CHECKPOINT, device="cuda:0")
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
os.environ.setdefault("LEGGED_LAB_G1_AMP_ROBOT_ASSET", args_cli.robot_asset)
os.environ.setdefault("OMNI_LOG_DEFAULT_LEVEL", "error")
os.environ.setdefault("OMNI_KIT_QUIET", "1")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import joblib  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import legged_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@dataclass
class PhaseData:
    curves: np.ndarray
    names: list[str]
    cycles: int


def install_numpy_pickle_shim() -> None:
    """Allow numpy>=2 pickles to load in numpy<2 runtimes."""

    if hasattr(np, "_core"):
        return
    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule_name in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        full_name = f"numpy._core.{submodule_name}"
        if full_name not in sys.modules:
            try:
                sys.modules[full_name] = importlib.import_module(f"numpy.core.{submodule_name}")
            except ImportError:
                pass


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def quat_apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros((*vector.shape[:-1], 1), dtype=vector.dtype)
    vector_quat = np.concatenate([zeros, vector], axis=-1)
    return quat_mul(quat_mul(quat, vector_quat), quat_conjugate(quat))[..., 1:]


def quat_apply_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros((*vector.shape[:-1], 1), dtype=vector.dtype)
    vector_quat = np.concatenate([zeros, vector], axis=-1)
    return quat_mul(quat_mul(quat_conjugate(quat), vector_quat), quat)[..., 1:]


def yaw_from_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return velocity
    velocity[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    velocity[0] = (values[1] - values[0]) * fps
    velocity[-1] = (values[-1] - values[-2]) * fps
    return velocity


def stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {key: float("nan") for key in ("mean", "std", "p05", "p50", "p95", "min", "max")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def centered_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    lhs = lhs[mask] - np.mean(lhs[mask])
    rhs = rhs[mask] - np.mean(rhs[mask])
    denom = np.std(lhs) * np.std(rhs)
    if denom < 1.0e-8:
        return 0.0
    return float(np.mean(lhs * rhs) / denom)


def cross_correlation_phase_shift(policy_curve: np.ndarray, ref_curve: np.ndarray) -> dict[str, float]:
    policy = np.asarray(policy_curve, dtype=np.float64)
    ref = np.asarray(ref_curve, dtype=np.float64)
    policy = policy - np.mean(policy)
    ref = ref - np.mean(ref)
    if np.std(policy) < 1.0e-8 or np.std(ref) < 1.0e-8:
        return {"shift_bins": 0.0, "shift_phase": 0.0, "corr": 0.0}
    correlations = []
    for shift in range(policy.shape[0]):
        correlations.append(centered_corr(np.roll(policy, shift), ref))
    best_shift = int(np.nanargmax(correlations))
    signed_shift = best_shift
    if signed_shift > policy.shape[0] // 2:
        signed_shift -= policy.shape[0]
    return {
        "shift_bins": float(signed_shift),
        "shift_phase": float(signed_shift / policy.shape[0]),
        "corr": float(correlations[best_shift]),
    }


def high_frequency_ratio(values: np.ndarray, dt: float, cutoff_hz: float = 3.0) -> float:
    series = np.asarray(values, dtype=np.float64)
    if series.shape[0] < 8:
        return float("nan")
    series = series - np.mean(series, axis=0, keepdims=True)
    spectrum = np.fft.rfft(series, axis=0)
    freqs = np.fft.rfftfreq(series.shape[0], dt)
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power[freqs > 0.1]))
    if total < 1.0e-12:
        return 0.0
    return float(np.sum(power[freqs >= cutoff_hz]) / total)


def touchdown_indices(contact: np.ndarray) -> np.ndarray:
    contact = np.asarray(contact, dtype=bool)
    return np.flatnonzero(contact[1:] & ~contact[:-1]) + 1


def cycles_from_touchdowns(left_contact: np.ndarray, min_len: int, max_len: int) -> list[tuple[int, int]]:
    touches = touchdown_indices(left_contact)
    cycles: list[tuple[int, int]] = []
    for start, end in zip(touches[:-1], touches[1:]):
        length = int(end - start)
        if min_len <= length <= max_len:
            cycles.append((int(start), int(end)))
    return cycles


def resample_cycle(values: np.ndarray, start: int, end: int, bins: int) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, end - start, endpoint=False)
    target_x = np.linspace(0.0, 1.0, bins, endpoint=False)
    segment = np.asarray(values[start:end], dtype=np.float64)
    if segment.ndim == 1:
        return np.interp(target_x, source_x, segment)
    output = np.empty((bins, segment.shape[1]), dtype=np.float64)
    for column in range(segment.shape[1]):
        output[:, column] = np.interp(target_x, source_x, segment[:, column])
    return output


def phase_fold(
    values: np.ndarray,
    left_contact: np.ndarray,
    dt: float,
    bins: int,
    min_cycle_s: float = 0.45,
    max_cycle_s: float = 1.60,
) -> tuple[np.ndarray, int]:
    min_len = max(3, int(round(min_cycle_s / dt)))
    max_len = max(min_len + 1, int(round(max_cycle_s / dt)))
    all_cycles = []
    values = np.asarray(values)
    left_contact = np.asarray(left_contact)
    if values.ndim == 2:
        values = values[:, None, :]
    if left_contact.ndim == 1:
        left_contact = left_contact[:, None]
    for env_id in range(values.shape[1]):
        cycles = cycles_from_touchdowns(left_contact[:, env_id], min_len=min_len, max_len=max_len)
        for start, end in cycles:
            all_cycles.append(resample_cycle(values[:, env_id, :], start, end, bins))
    if not all_cycles:
        return np.full((bins, values.shape[-1]), np.nan), 0
    return np.nanmean(np.stack(all_cycles, axis=0), axis=0), len(all_cycles)


def infer_reference_contact(foot_z: np.ndarray, fps: float) -> np.ndarray:
    """Infer contact from low foot height and low vertical velocity."""

    foot_z = np.asarray(foot_z, dtype=np.float64)
    foot_floor = np.percentile(foot_z, 8, axis=0, keepdims=True)
    rel_z = foot_z - foot_floor
    vel_z = finite_difference(foot_z.astype(np.float32), fps)
    low = rel_z < 0.035
    slow = np.abs(vel_z) < 0.45
    contact = low & slow
    # Fill tiny holes so touchdown cycles are not fragmented by retargeting noise.
    for foot_id in range(contact.shape[1]):
        series = contact[:, foot_id].copy()
        for index in range(1, series.shape[0] - 1):
            if not series[index] and series[index - 1] and series[index + 1]:
                series[index] = True
        contact[:, foot_id] = series
    return contact


def load_reference_clip(path: Path, target_joint_names: list[str]) -> dict:
    raw = joblib.load(path)
    fps = float(raw.get("fps", 50.0))
    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    root_quat = quat_xyzw_to_wxyz(np.asarray(raw["root_rot"], dtype=np.float32))
    source_joint_names = list(raw["dof_names"])
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    missing = [name for name in target_joint_names if name not in source_index]
    if missing:
        raise ValueError(f"{path} missing joints: {missing}")
    dof_pos_raw = np.asarray(raw["dof_pos"], dtype=np.float32)
    joint_pos = dof_pos_raw[:, [source_index[name] for name in target_joint_names]]

    local_body_pos = np.asarray(raw["local_body_pos"], dtype=np.float32)
    body_names = list(raw["link_body_list"])
    body_index = {name: index for index, name in enumerate(body_names)}
    required_bodies = FOOT_BODY_NAMES + WRIST_BODY_NAMES + SHOULDER_BODY_NAMES
    missing_bodies = [name for name in required_bodies if name not in body_index]
    if missing_bodies:
        raise ValueError(f"{path} missing bodies: {missing_bodies}")
    body_local = local_body_pos[:, [body_index[name] for name in required_bodies], :]
    body_world = root_pos[:, None, :] + quat_apply(root_quat[:, None, :], body_local)
    foot_z = body_world[:, 0:2, 2]
    contact = infer_reference_contact(foot_z, fps)
    wrist_rel = body_local[:, 2:4, :] - body_local[:, 4:6, :]

    root_vel_w = finite_difference(root_pos, fps)
    root_vel_b = quat_apply_inverse(root_quat, root_vel_w)
    yaw = np.unwrap(yaw_from_wxyz(root_quat))
    yaw_rate = finite_difference(yaw[:, None].astype(np.float32), fps)[:, 0]
    return {
        "name": path.stem,
        "fps": fps,
        "dt": 1.0 / fps,
        "joint_pos": joint_pos,
        "contact": contact,
        "wrist_rel": wrist_rel.reshape(wrist_rel.shape[0], -1),
        "root_vel_b_mean": np.mean(root_vel_b, axis=0),
        "yaw_rate_mean": float(np.mean(yaw_rate)),
    }


def reference_score(clip: dict, target_velocity: np.ndarray) -> float:
    ref_velocity = np.array(
        [clip["root_vel_b_mean"][0], clip["root_vel_b_mean"][1], clip["yaw_rate_mean"]],
        dtype=np.float64,
    )
    scale = np.array([0.25, 0.20, 0.35], dtype=np.float64)
    return float(np.linalg.norm((ref_velocity - target_velocity) / scale))


def load_reference_set(motion_dir: Path, target_velocity: np.ndarray, top_k: int, bins: int) -> tuple[dict, list[dict]]:
    install_numpy_pickle_shim()
    motion_paths = sorted(motion_dir.glob("*.pkl"))
    if not motion_paths:
        raise FileNotFoundError(f"No .pkl files found in {motion_dir}")
    target_joint_names = ARM_JOINT_NAMES + LEG_COUPLING_JOINT_NAMES
    scored = []
    for path in motion_paths:
        clip = load_reference_clip(path, target_joint_names)
        scored.append((reference_score(clip, target_velocity), clip))
    selected = [clip for _, clip in sorted(scored, key=lambda item: item[0])[:top_k]]

    joint_cycles = []
    wrist_cycles = []
    total_joint_cycles = 0
    total_wrist_cycles = 0
    for clip in selected:
        joint_curve, joint_count = phase_fold(
            clip["joint_pos"], clip["contact"][:, 0], clip["dt"], bins=bins
        )
        wrist_curve, wrist_count = phase_fold(
            clip["wrist_rel"], clip["contact"][:, 0], clip["dt"], bins=bins
        )
        if joint_count:
            joint_cycles.append(joint_curve)
            total_joint_cycles += joint_count
        if wrist_count:
            wrist_cycles.append(wrist_curve)
            total_wrist_cycles += wrist_count
    if not joint_cycles:
        raise RuntimeError("Could not infer any reference gait cycles from selected clips.")
    aggregate = {
        "joint_phase": np.nanmean(np.stack(joint_cycles, axis=0), axis=0),
        "wrist_phase": np.nanmean(np.stack(wrist_cycles, axis=0), axis=0),
        "joint_names": target_joint_names,
        "wrist_names": [
            "left_wrist_rel_x",
            "left_wrist_rel_y",
            "left_wrist_rel_z",
            "right_wrist_rel_x",
            "right_wrist_rel_y",
            "right_wrist_rel_z",
        ],
        "joint_cycles": total_joint_cycles,
        "wrist_cycles": total_wrist_cycles,
    }
    return aggregate, selected


def configure_clean_rollout(env_cfg) -> None:
    command = env_cfg.commands.base_velocity
    if hasattr(command, "heading_command"):
        command.heading_command = False
    if hasattr(command, "rel_heading_envs"):
        command.rel_heading_envs = 1.0
    if hasattr(command, "rel_standing_envs"):
        command.rel_standing_envs = 0.0
    if hasattr(command, "resampling_time_range"):
        command.resampling_time_range = (1000.0, 1000.0)
    if hasattr(command, "low_speed_threshold"):
        command.low_speed_threshold = 0.40
    if hasattr(command, "high_speed_ang_vel_z_mean"):
        command.high_speed_ang_vel_z_mean = args_cli.command_yaw
    if hasattr(command, "high_speed_ang_vel_z_std"):
        command.high_speed_ang_vel_z_std = 0.0
    command.ranges.lin_vel_x = (args_cli.command_lin_x, args_cli.command_lin_x)
    command.ranges.lin_vel_y = (args_cli.command_lin_y, args_cli.command_lin_y)
    command.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
    if hasattr(command.ranges, "low_speed_ang_vel_z"):
        command.ranges.low_speed_ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)

    if args_cli.disable_rsi and hasattr(env_cfg.events, "reset_from_ref"):
        env_cfg.events.reset_from_ref = None
    if args_cli.disable_randomization:
        for name in (
            "physics_material",
            "add_base_mass",
            "randomize_rigid_body_com",
            "scale_link_mass",
            "scale_actuator_gains",
            "scale_joint_parameters",
            "push_robot",
        ):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)


def load_runner(env, agent_cfg, checkpoint: str):
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint, map_location=agent_cfg.device)
    return runner


def policy_reset_fn(runner):
    try:
        return runner.alg.policy.reset
    except AttributeError:
        try:
            return runner.alg.actor_critic.reset
        except AttributeError:
            return lambda dones: None


def collect_policy_rollout(env, runner, bins: int) -> dict:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    action_term = base_env.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names)
    action_joint_ids = list(action_term._joint_ids)
    arm_action_ids = [action_joint_names.index(name) for name in ARM_JOINT_NAMES]
    arm_joint_ids = [action_joint_ids[index] for index in arm_action_ids]
    coupling_joint_ids, coupling_joint_names = robot.find_joints(LEG_COUPLING_JOINT_NAMES, preserve_order=True)
    foot_body_ids, _ = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    sensor_foot_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    wrist_body_ids, _ = robot.find_bodies(WRIST_BODY_NAMES, preserve_order=True)
    shoulder_body_ids, _ = robot.find_bodies(SHOULDER_BODY_NAMES, preserve_order=True)

    offset = action_term._offset
    scale = action_term._scale
    if not isinstance(offset, torch.Tensor):
        offset = torch.full((base_env.num_envs, len(action_joint_names)), float(offset), device=base_env.device)
    if not isinstance(scale, torch.Tensor):
        scale = torch.full((base_env.num_envs, len(action_joint_names)), float(scale), device=base_env.device)

    policy = runner.get_inference_policy(device=base_env.device)
    reset_policy = policy_reset_fn(runner)
    obs = env.get_observations()
    records = {
        "arm_joint_pos": [],
        "arm_target_pos": [],
        "arm_joint_vel": [],
        "coupling_joint_pos": [],
        "contact": [],
        "wrist_rel": [],
        "root_vel_b": [],
        "root_ang_vel_b": [],
        "command": [],
        "actions": [],
    }

    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            actions = policy(obs)
            target_joint_pos = offset + scale * actions
            obs, _, dones, _ = env.step(actions)
            reset_policy(dones)
            if step < args_cli.warmup_steps:
                continue

            contact = contact_sensor.data.current_contact_time[:, sensor_foot_ids] > 0.0
            wrist_rel = (
                robot.data.body_pos_w[:, wrist_body_ids, :] - robot.data.body_pos_w[:, shoulder_body_ids, :]
            )
            command = base_env.command_manager.get_command("base_velocity")
            records["arm_joint_pos"].append(robot.data.joint_pos[:, arm_joint_ids].detach().cpu().numpy())
            records["arm_target_pos"].append(target_joint_pos[:, arm_action_ids].detach().cpu().numpy())
            records["arm_joint_vel"].append(robot.data.joint_vel[:, arm_joint_ids].detach().cpu().numpy())
            records["coupling_joint_pos"].append(robot.data.joint_pos[:, coupling_joint_ids].detach().cpu().numpy())
            records["contact"].append(contact.detach().cpu().numpy())
            records["wrist_rel"].append(wrist_rel.reshape(base_env.num_envs, -1).detach().cpu().numpy())
            records["root_vel_b"].append(robot.data.root_lin_vel_b.detach().cpu().numpy())
            records["root_ang_vel_b"].append(robot.data.root_ang_vel_b.detach().cpu().numpy())
            records["command"].append(command.detach().cpu().numpy())
            records["actions"].append(actions[:, arm_action_ids].detach().cpu().numpy())

    arrays = {key: np.stack(value, axis=0) for key, value in records.items()}
    policy_joint = np.concatenate([arrays["arm_joint_pos"], arrays["coupling_joint_pos"]], axis=2)
    joint_phase, joint_cycles = phase_fold(
        policy_joint, arrays["contact"][:, :, 0], base_env.step_dt, bins=bins
    )
    target_phase, target_cycles = phase_fold(
        arrays["arm_target_pos"], arrays["contact"][:, :, 0], base_env.step_dt, bins=bins
    )
    wrist_phase, wrist_cycles = phase_fold(
        arrays["wrist_rel"], arrays["contact"][:, :, 0], base_env.step_dt, bins=bins
    )
    arrays["joint_phase"] = joint_phase
    arrays["target_phase"] = target_phase
    arrays["wrist_phase"] = wrist_phase
    arrays["joint_phase_cycles"] = np.array([joint_cycles], dtype=np.int32)
    arrays["target_phase_cycles"] = np.array([target_cycles], dtype=np.int32)
    arrays["wrist_phase_cycles"] = np.array([wrist_cycles], dtype=np.int32)
    arrays["joint_phase_names"] = np.array(ARM_JOINT_NAMES + coupling_joint_names, dtype=object)
    arrays["arm_joint_names"] = np.array(ARM_JOINT_NAMES, dtype=object)
    arrays["wrist_names"] = np.array(
        [
            "left_wrist_rel_x",
            "left_wrist_rel_y",
            "left_wrist_rel_z",
            "right_wrist_rel_x",
            "right_wrist_rel_y",
            "right_wrist_rel_z",
        ],
        dtype=object,
    )
    return arrays


def indexed(curve: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return curve[:, names.index(name)]


def build_report(policy: dict, reference: dict, selected_clips: list[dict], dt: float) -> dict:
    joint_names = list(policy["joint_phase_names"])
    ref_joint_names = reference["joint_names"]
    wrist_names = list(policy["wrist_names"])
    ref_wrist_names = reference["wrist_names"]
    report = {
        "checkpoint": str(resolve_path(args_cli.checkpoint)),
        "task": args_cli.task,
        "command": {
            "lin_x": args_cli.command_lin_x,
            "lin_y": args_cli.command_lin_y,
            "yaw": args_cli.command_yaw,
        },
        "rollout": {
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "warmup_steps": args_cli.warmup_steps,
            "dt": dt,
            "phase_bins": args_cli.phase_bins,
            "joint_cycles": int(policy["joint_phase_cycles"][0]),
            "target_cycles": int(policy["target_phase_cycles"][0]),
            "wrist_cycles": int(policy["wrist_phase_cycles"][0]),
            "root_vel_b": stats(policy["root_vel_b"]),
            "root_ang_vel_b": stats(policy["root_ang_vel_b"]),
        },
        "reference": {
            "motion_dir": str(resolve_path(args_cli.motion_dir)),
            "selected": [
                {
                    "name": clip["name"],
                    "mean_vx": float(clip["root_vel_b_mean"][0]),
                    "mean_vy": float(clip["root_vel_b_mean"][1]),
                    "mean_yaw_rate": float(clip["yaw_rate_mean"]),
                }
                for clip in selected_clips
            ],
            "joint_cycles": int(reference["joint_cycles"]),
            "wrist_cycles": int(reference["wrist_cycles"]),
        },
        "arm_joint_metrics": {},
        "coupling": {},
        "diagnosis": [],
    }

    for name in CORE_ARM_JOINT_NAMES:
        p_curve = indexed(policy["joint_phase"], joint_names, name)
        r_curve = indexed(reference["joint_phase"], ref_joint_names, name)
        p_amp = float(np.nanpercentile(p_curve, 95) - np.nanpercentile(p_curve, 5))
        r_amp = float(np.nanpercentile(r_curve, 95) - np.nanpercentile(r_curve, 5))
        shift = cross_correlation_phase_shift(p_curve, r_curve)
        rmse = float(np.sqrt(np.nanmean((p_curve - r_curve) ** 2)))
        report["arm_joint_metrics"][name] = {
            "policy_mean": float(np.nanmean(p_curve)),
            "reference_mean": float(np.nanmean(r_curve)),
            "mean_bias": float(np.nanmean(p_curve - r_curve)),
            "policy_amp_p95_p05": p_amp,
            "reference_amp_p95_p05": r_amp,
            "amp_ratio_policy_over_ref": float(p_amp / max(r_amp, 1.0e-6)),
            "phase_shift_fraction": shift["shift_phase"],
            "phase_corr_after_best_shift": shift["corr"],
            "phase_rmse_rad": rmse,
        }

    p_left = indexed(policy["joint_phase"], joint_names, "left_shoulder_pitch_joint")
    p_right = indexed(policy["joint_phase"], joint_names, "right_shoulder_pitch_joint")
    r_left = indexed(reference["joint_phase"], ref_joint_names, "left_shoulder_pitch_joint")
    r_right = indexed(reference["joint_phase"], ref_joint_names, "right_shoulder_pitch_joint")
    p_lhip = indexed(policy["joint_phase"], joint_names, "left_hip_pitch_joint")
    p_rhip = indexed(policy["joint_phase"], joint_names, "right_hip_pitch_joint")
    r_lhip = indexed(reference["joint_phase"], ref_joint_names, "left_hip_pitch_joint")
    r_rhip = indexed(reference["joint_phase"], ref_joint_names, "right_hip_pitch_joint")
    report["coupling"] = {
        "policy_left_right_shoulder_pitch_corr": centered_corr(p_left, p_right),
        "reference_left_right_shoulder_pitch_corr": centered_corr(r_left, r_right),
        "policy_left_shoulder_left_hip_corr": centered_corr(p_left, p_lhip),
        "reference_left_shoulder_left_hip_corr": centered_corr(r_left, r_lhip),
        "policy_left_shoulder_right_hip_corr": centered_corr(p_left, p_rhip),
        "reference_left_shoulder_right_hip_corr": centered_corr(r_left, r_rhip),
        "policy_right_shoulder_left_hip_corr": centered_corr(p_right, p_lhip),
        "reference_right_shoulder_left_hip_corr": centered_corr(r_right, r_lhip),
        "policy_right_shoulder_right_hip_corr": centered_corr(p_right, p_rhip),
        "reference_right_shoulder_right_hip_corr": centered_corr(r_right, r_rhip),
    }

    for name in (
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
    ):
        metric = report["arm_joint_metrics"][name]
        if metric["amp_ratio_policy_over_ref"] < 0.65:
            report["diagnosis"].append(f"{name}: policy amplitude is much smaller than reference.")
        elif metric["amp_ratio_policy_over_ref"] > 1.45:
            report["diagnosis"].append(f"{name}: policy amplitude is much larger than reference.")
        if abs(metric["phase_shift_fraction"]) > 0.14 and metric["phase_corr_after_best_shift"] > 0.35:
            report["diagnosis"].append(f"{name}: policy phase is shifted by {metric['phase_shift_fraction']:.2f} gait cycles.")
        if metric["phase_corr_after_best_shift"] < 0.25:
            report["diagnosis"].append(f"{name}: policy phase curve does not match the reference shape well.")

    p_hf = high_frequency_ratio(policy["arm_joint_pos"][:, :, : len(ARM_JOINT_NAMES)], dt)
    p_action_hf = high_frequency_ratio(policy["actions"], dt)
    report["rollout"]["arm_joint_high_frequency_ratio_gt_3hz"] = p_hf
    report["rollout"]["arm_action_high_frequency_ratio_gt_3hz"] = p_action_hf
    if p_action_hf > 0.30:
        report["diagnosis"].append("Arm action has large >3Hz content; visible arm jitter is likely action-level.")

    for axis_name in ("left_wrist_rel_x", "right_wrist_rel_x"):
        p_curve = indexed(policy["wrist_phase"], wrist_names, axis_name)
        r_curve = indexed(reference["wrist_phase"], ref_wrist_names, axis_name)
        p_amp = float(np.nanpercentile(p_curve, 95) - np.nanpercentile(p_curve, 5))
        r_amp = float(np.nanpercentile(r_curve, 95) - np.nanpercentile(r_curve, 5))
        report.setdefault("wrist_path_metrics", {})[axis_name] = {
            "policy_amp_m": p_amp,
            "reference_amp_m": r_amp,
            "amp_ratio_policy_over_ref": float(p_amp / max(r_amp, 1.0e-6)),
            "phase_corr": centered_corr(p_curve, r_curve),
        }
        if p_amp / max(r_amp, 1.0e-6) < 0.65:
            report["diagnosis"].append(f"{axis_name}: wrist forward/back path is too small.")

    if not report["diagnosis"]:
        report["diagnosis"].append("No large numeric mismatch found in the selected straight-walk arm metrics.")
    return report


def write_plots(output_dir: Path, policy: dict, reference: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = np.linspace(0.0, 1.0, args_cli.phase_bins, endpoint=False)
    joint_names = list(policy["joint_phase_names"])
    ref_joint_names = reference["joint_names"]
    wrist_names = list(policy["wrist_names"])
    ref_wrist_names = reference["wrist_names"]

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    joint_pairs = [
        ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
        ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
        ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint"),
        ("left_elbow_joint", "right_elbow_joint"),
    ]
    for row, pair in enumerate(joint_pairs):
        for col, name in enumerate(pair):
            ax = axes[row, col]
            ax.plot(phase, indexed(reference["joint_phase"], ref_joint_names, name), label="reference", linewidth=2.2)
            ax.plot(phase, indexed(policy["joint_phase"], joint_names, name), label="policy actual", linewidth=1.8)
            ax.plot(phase, indexed(policy["target_phase"], ARM_JOINT_NAMES, name), label="policy target", linewidth=1.1, alpha=0.65)
            ax.set_title(name.replace("_joint", ""))
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("gait phase, left touchdown = 0")
    axes[-1, 1].set_xlabel("gait phase, left touchdown = 0")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "arm_joint_phase_compare.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    wrist_pairs = [
        ("left_wrist_rel_x", "right_wrist_rel_x"),
        ("left_wrist_rel_y", "right_wrist_rel_y"),
        ("left_wrist_rel_z", "right_wrist_rel_z"),
    ]
    for row, pair in enumerate(wrist_pairs):
        for col, name in enumerate(pair):
            ax = axes[row, col]
            ax.plot(phase, indexed(reference["wrist_phase"], ref_wrist_names, name), label="reference", linewidth=2.2)
            ax.plot(phase, indexed(policy["wrist_phase"], wrist_names, name), label="policy", linewidth=1.8)
            ax.set_title(name)
            ax.set_ylabel("m")
            ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("gait phase, left touchdown = 0")
    axes[-1, 1].set_xlabel("gait phase, left touchdown = 0")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "wrist_path_phase_compare.png", dpi=160)
    plt.close(fig)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    checkpoint = retrieve_file_path(str(resolve_path(args_cli.checkpoint)))
    output_dir = resolve_path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.log_dir = os.path.dirname(checkpoint)
    configure_clean_rollout(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner(env, agent_cfg, checkpoint)
    base_env = env.unwrapped

    print(f"[G1 Arm Swing] checkpoint={checkpoint}")
    print(
        "[G1 Arm Swing] fixed command "
        f"vx={args_cli.command_lin_x:.3f} vy={args_cli.command_lin_y:.3f} yaw={args_cli.command_yaw:.3f}"
    )
    policy = collect_policy_rollout(env, runner, bins=args_cli.phase_bins)
    target_velocity = np.array([args_cli.command_lin_x, args_cli.command_lin_y, args_cli.command_yaw], dtype=np.float64)
    reference, selected_clips = load_reference_set(
        resolve_path(args_cli.motion_dir), target_velocity, top_k=args_cli.reference_top_k, bins=args_cli.phase_bins
    )
    report = build_report(policy, reference, selected_clips, dt=base_env.step_dt)

    np.savez_compressed(
        output_dir / "arm_swing_phase_data.npz",
        policy_joint_phase=policy["joint_phase"],
        policy_target_phase=policy["target_phase"],
        policy_wrist_phase=policy["wrist_phase"],
        reference_joint_phase=reference["joint_phase"],
        reference_wrist_phase=reference["wrist_phase"],
        joint_names=np.array(list(policy["joint_phase_names"]), dtype=object),
        arm_joint_names=np.array(ARM_JOINT_NAMES, dtype=object),
        wrist_names=np.array(list(policy["wrist_names"]), dtype=object),
    )
    with (output_dir / "arm_swing_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    write_plots(output_dir, policy, reference)

    print(f"[G1 Arm Swing] output_dir={output_dir}")
    print("[G1 Arm Swing] selected reference clips:")
    for clip in selected_clips:
        print(
            f"  {clip['name']:28s} vx={clip['root_vel_b_mean'][0]: .3f} "
            f"vy={clip['root_vel_b_mean'][1]: .3f} yaw={clip['yaw_rate_mean']: .3f}"
        )
    print(
        f"[G1 Arm Swing] cycles policy={report['rollout']['joint_cycles']} "
        f"reference={report['reference']['joint_cycles']}"
    )
    print("[G1 Arm Swing] diagnosis:")
    for item in report["diagnosis"]:
        print(f"  - {item}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
