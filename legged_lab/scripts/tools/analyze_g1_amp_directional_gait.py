#!/usr/bin/env python3
"""Diagnose directional gait quality for a G1 AMP policy.

The script runs one fixed velocity command in headless IsaacLab, records foot
contacts and root/foot motion, and compares the command with matching frames in
the reference motion dataset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "legged_lab"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))

import cli_args  # isort: skip


DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/g1_amp/"
    "2026-06-12_23-16-11_s3_g1_29dof_cmu_walk_washed_strict_armprior_scratch_5000/"
    "model_4999.pt"
)
DEFAULT_TASK = "LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Strict-ArmPrior-v0"
DEFAULT_MOTION_DIR = "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_washed_50hz"
DEFAULT_OUTPUT_DIR = (
    "logs/rsl_rl/g1_amp/"
    "2026-06-12_23-16-11_s3_g1_29dof_cmu_walk_washed_strict_armprior_scratch_5000/"
    "directional_gait_analysis"
)
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--motion_dir", type=str, default=DEFAULT_MOTION_DIR)
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--label", type=str, default="normal_walk")
parser.add_argument("--command_lin_x", type=float, default=0.82)
parser.add_argument("--command_lin_y", type=float, default=0.0)
parser.add_argument("--command_yaw", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=24)
parser.add_argument("--max_steps", type=int, default=700)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--robot_asset", type=str, default="s3_g1_29dof")
parser.add_argument("--disable_rsi", action="store_true", default=True)
parser.add_argument("--keep_rsi", action="store_false", dest="disable_rsi")
parser.add_argument("--disable_randomization", action="store_true", default=True)
parser.add_argument("--keep_randomization", action="store_false", dest="disable_randomization")
parser.add_argument("--match_vx_tol", type=float, default=0.15)
parser.add_argument("--match_vy_tol", type=float, default=0.12)
parser.add_argument("--match_yaw_tol", type=float, default=0.15)
parser.add_argument("--top_k_clips", type=int, default=8)
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


def install_numpy_pickle_shim() -> None:
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


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return velocity
    velocity[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    velocity[0] = (values[1] - values[0]) * fps
    velocity[-1] = (values[-1] - values[-2]) * fps
    return velocity


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
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def infer_reference_contact(foot_z: np.ndarray, fps: float) -> np.ndarray:
    foot_floor = np.percentile(foot_z, 8, axis=0, keepdims=True)
    rel_z = foot_z - foot_floor
    vel_z = finite_difference(foot_z.astype(np.float32), fps)
    contact = (rel_z < 0.035) & (np.abs(vel_z) < 0.45)
    for foot_id in range(contact.shape[1]):
        series = contact[:, foot_id].copy()
        for index in range(1, series.shape[0] - 1):
            if not series[index] and series[index - 1] and series[index + 1]:
                series[index] = True
        contact[:, foot_id] = series
    return contact


def segment_durations(mask: np.ndarray, dt: float) -> np.ndarray:
    series = np.asarray(mask, dtype=bool).reshape(-1)
    durations = []
    start = None
    for index, value in enumerate(series):
        if value and start is None:
            start = index
        elif not value and start is not None:
            durations.append((index - start) * dt)
            start = None
    if start is not None:
        durations.append((len(series) - start) * dt)
    return np.asarray(durations, dtype=np.float64)


def contact_summary(contact: np.ndarray, dt: float) -> dict:
    contact = np.asarray(contact, dtype=bool)
    if contact.ndim == 2:
        contact = contact[:, None, :]
    double_stance = contact[:, :, 0] & contact[:, :, 1]
    double_air = ~contact[:, :, 0] & ~contact[:, :, 1]
    single_stance = contact[:, :, 0] ^ contact[:, :, 1]
    left_air = ~contact[:, :, 0]
    right_air = ~contact[:, :, 1]
    double_air_durations = []
    left_air_durations = []
    right_air_durations = []
    for env_id in range(contact.shape[1]):
        double_air_durations.append(segment_durations(double_air[:, env_id], dt))
        left_air_durations.append(segment_durations(left_air[:, env_id], dt))
        right_air_durations.append(segment_durations(right_air[:, env_id], dt))
    double_air_durations = (
        np.concatenate(double_air_durations, axis=0) if double_air_durations else np.empty((0,), dtype=np.float64)
    )
    left_air_durations = (
        np.concatenate(left_air_durations, axis=0) if left_air_durations else np.empty((0,), dtype=np.float64)
    )
    right_air_durations = (
        np.concatenate(right_air_durations, axis=0) if right_air_durations else np.empty((0,), dtype=np.float64)
    )
    return {
        "left_contact_fraction": float(np.mean(contact[:, :, 0])),
        "right_contact_fraction": float(np.mean(contact[:, :, 1])),
        "contact_fraction_diff_abs": float(abs(np.mean(contact[:, :, 0]) - np.mean(contact[:, :, 1]))),
        "single_stance_fraction": float(np.mean(single_stance)),
        "double_stance_fraction": float(np.mean(double_stance)),
        "double_air_fraction": float(np.mean(double_air)),
        "double_air_duration_s": stats(double_air_durations),
        "left_air_duration_s": stats(left_air_durations),
        "right_air_duration_s": stats(right_air_durations),
        "left_right_air_duration_mean_diff_abs_s": float(
            abs(np.mean(left_air_durations) - np.mean(right_air_durations))
        )
        if left_air_durations.size and right_air_durations.size
        else float("nan"),
    }


def _fixed_command_sampling_config_path(output_dir: Path) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args_cli.label).strip("_") or "fixed"
    return output_dir / f"{safe_label}_fixed_command_sampling_config.json"


def _write_fixed_command_sampling_config(output_dir: Path) -> Path:
    config_path = _fixed_command_sampling_config_path(output_dir)
    mode_name = f"fixed_{args_cli.label}"
    config = {
        "command_frame": "robot_base_local",
        "fps": 50,
        "mode_weights": {mode_name: 1.0},
        "mode_name_by_id": {"0": mode_name},
        "modes": {
            mode_name: {
                "lin_vel_x": [args_cli.command_lin_x, args_cli.command_lin_x],
                "lin_vel_y": [args_cli.command_lin_y, args_cli.command_lin_y],
                "ang_vel_z": [args_cli.command_yaw, args_cli.command_yaw],
            }
        },
        "source_dataset": "fixed_command_eval",
    }
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    return config_path


def configure_clean_rollout(env_cfg, output_dir: Path) -> Path | None:
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
    fixed_sampling_config_path = None
    if hasattr(command, "sampling_config_path"):
        fixed_sampling_config_path = _write_fixed_command_sampling_config(output_dir)
        command.sampling_config_path = str(fixed_sampling_config_path)

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
    return fixed_sampling_config_path


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


def collect_policy_rollout(env, runner) -> tuple[dict, float]:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    foot_body_ids, _ = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    sensor_foot_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    policy = runner.get_inference_policy(device=base_env.device)
    reset_policy = policy_reset_fn(runner)
    obs = env.get_observations()
    records = {
        "contact": [],
        "root_vel_b": [],
        "root_ang_vel_b": [],
        "root_pos_w": [],
        "foot_pos_w": [],
        "foot_vel_w": [],
        "command": [],
    }
    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            reset_policy(dones)
            if step < args_cli.warmup_steps:
                continue
            contact = contact_sensor.data.current_contact_time[:, sensor_foot_ids] > 0.0
            records["contact"].append(contact.detach().cpu().numpy())
            records["root_vel_b"].append(robot.data.root_lin_vel_b.detach().cpu().numpy())
            records["root_ang_vel_b"].append(robot.data.root_ang_vel_b.detach().cpu().numpy())
            records["root_pos_w"].append(robot.data.root_pos_w.detach().cpu().numpy())
            records["foot_pos_w"].append(robot.data.body_pos_w[:, foot_body_ids, :].detach().cpu().numpy())
            records["foot_vel_w"].append(robot.data.body_lin_vel_w[:, foot_body_ids, :].detach().cpu().numpy())
            records["command"].append(base_env.command_manager.get_command("base_velocity").detach().cpu().numpy())
    return {key: np.stack(value, axis=0) for key, value in records.items()}, float(base_env.step_dt)


def summarize_policy(records: dict, dt: float) -> dict:
    contact = records["contact"]
    root_vel_b = records["root_vel_b"]
    root_ang_vel_b = records["root_ang_vel_b"]
    root_pos_w = records["root_pos_w"]
    foot_pos_w = records["foot_pos_w"]
    foot_vel_w = records["foot_vel_w"]
    command = records["command"]
    foot_clearance = foot_pos_w[:, :, :, 2] - np.min(foot_pos_w[:, :, :, 2], axis=2, keepdims=True)
    contact_expanded = contact[:, :, :, None]
    contact_foot_speed = np.linalg.norm(foot_vel_w[:, :, :, :2], axis=-1)[contact]
    swing_foot_clearance = foot_clearance[~contact]
    double_air = ~contact[:, :, 0] & ~contact[:, :, 1]
    return {
        "command_mean": stats(command),
        "root_vel_b": {
            "x": stats(root_vel_b[:, :, 0]),
            "y": stats(root_vel_b[:, :, 1]),
            "z": stats(root_vel_b[:, :, 2]),
        },
        "root_ang_vel_b_z": stats(root_ang_vel_b[:, :, 2]),
        "tracking_mae": {
            "lin_xy": float(np.mean(np.linalg.norm(root_vel_b[:, :, :2] - command[:, :, :2], axis=-1))),
            "yaw": float(np.mean(np.abs(root_ang_vel_b[:, :, 2] - command[:, :, 2]))),
        },
        "root_height": {
            "z": stats(root_pos_w[:, :, 2]),
            "vertical_speed_abs": stats(np.abs(root_vel_b[:, :, 2])),
            "double_air_vertical_speed_abs": stats(np.abs(root_vel_b[:, :, 2][double_air])),
        },
        "contact": contact_summary(contact, dt),
        "feet": {
            "contact_foot_speed_xy": stats(contact_foot_speed),
            "swing_clearance_relative_to_lower_foot_m": stats(swing_foot_clearance),
            "all_foot_speed_xy": stats(np.linalg.norm(foot_vel_w[:, :, :, :2], axis=-1)),
        },
    }


def load_reference_frames(motion_dir: Path, target: np.ndarray) -> dict:
    install_numpy_pickle_shim()
    motion_paths = sorted(motion_dir.glob("*.pkl"))
    if not motion_paths:
        raise FileNotFoundError(f"No .pkl files found in {motion_dir}")
    all_vel = []
    all_contact = []
    all_root_z = []
    all_foot_speed = []
    per_clip = []
    for path in motion_paths:
        raw = joblib.load(path)
        fps = float(raw.get("fps", 50.0))
        root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
        root_quat = quat_xyzw_to_wxyz(np.asarray(raw["root_rot"], dtype=np.float32))
        body_names = list(raw["link_body_list"])
        body_index = {name: index for index, name in enumerate(body_names)}
        missing = [name for name in FOOT_BODY_NAMES if name not in body_index]
        if missing:
            raise ValueError(f"{path} missing bodies: {missing}")
        local_body = np.asarray(raw["local_body_pos"], dtype=np.float32)[:, [body_index[name] for name in FOOT_BODY_NAMES], :]
        foot_world = root_pos[:, None, :] + quat_apply(root_quat[:, None, :], local_body)
        foot_vel = finite_difference(foot_world, fps)
        foot_z = foot_world[:, :, 2]
        contact = infer_reference_contact(foot_z, fps)
        root_vel_w = finite_difference(root_pos, fps)
        root_vel_b = quat_apply_inverse(root_quat, root_vel_w)
        yaw = np.unwrap(yaw_from_wxyz(root_quat))
        yaw_rate = finite_difference(yaw[:, None].astype(np.float32), fps)[:, 0]
        velocity = np.column_stack([root_vel_b[:, 0], root_vel_b[:, 1], yaw_rate])
        all_vel.append(velocity)
        all_contact.append(contact)
        all_root_z.append(root_pos[:, 2])
        all_foot_speed.append(np.linalg.norm(foot_vel[:, :, :2], axis=-1))
        scale = np.array([0.25, 0.20, 0.35], dtype=np.float64)
        score = float(np.mean(np.linalg.norm((velocity - target[None, :]) / scale[None, :], axis=1)))
        per_clip.append(
            {
                "name": path.name,
                "score": score,
                "mean_vx": float(np.mean(velocity[:, 0])),
                "mean_vy": float(np.mean(velocity[:, 1])),
                "mean_yaw": float(np.mean(velocity[:, 2])),
                "straight_ratio_abs_yaw_lt_0p15": float(np.mean(np.abs(velocity[:, 2]) < 0.15)),
                "frames": int(velocity.shape[0]),
            }
        )
    velocity = np.concatenate(all_vel, axis=0)
    contact = np.concatenate(all_contact, axis=0)
    root_z = np.concatenate(all_root_z, axis=0)
    foot_speed = np.concatenate(all_foot_speed, axis=0)
    match = (
        (np.abs(velocity[:, 0] - target[0]) <= args_cli.match_vx_tol)
        & (np.abs(velocity[:, 1] - target[1]) <= args_cli.match_vy_tol)
        & (np.abs(velocity[:, 2] - target[2]) <= args_cli.match_yaw_tol)
    )
    matched_contact = contact[match]
    matched_foot_speed = foot_speed[match]
    result = {
        "target": {"vx": float(target[0]), "vy": float(target[1]), "yaw": float(target[2])},
        "total_frames": int(velocity.shape[0]),
        "match_frames": int(np.count_nonzero(match)),
        "match_ratio": float(np.mean(match)),
        "velocity_all": {
            "vx": stats(velocity[:, 0]),
            "vy": stats(velocity[:, 1]),
            "yaw": stats(velocity[:, 2]),
        },
        "velocity_matched": {
            "vx": stats(velocity[match, 0]),
            "vy": stats(velocity[match, 1]),
            "yaw": stats(velocity[match, 2]),
        },
        "root_z_matched": stats(root_z[match]),
        "contact_matched": contact_summary(matched_contact, 1.0 / 50.0) if matched_contact.size else {},
        "matched_contact_foot_speed_xy": stats(matched_foot_speed[matched_contact]) if matched_contact.size else stats([]),
        "top_clips_by_mean_velocity_score": sorted(per_clip, key=lambda item: item["score"])[: args_cli.top_k_clips],
    }
    return result


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
    fixed_sampling_config_path = configure_clean_rollout(env_cfg, output_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner(env, agent_cfg, checkpoint)
    records, dt = collect_policy_rollout(env, runner)
    policy_summary = summarize_policy(records, dt)
    target = np.array([args_cli.command_lin_x, args_cli.command_lin_y, args_cli.command_yaw], dtype=np.float64)
    reference_summary = load_reference_frames(resolve_path(args_cli.motion_dir), target)
    report = {
        "label": args_cli.label,
        "checkpoint": checkpoint,
        "task": args_cli.task,
        "robot_asset": args_cli.robot_asset,
        "fixed_command_sampling_config": str(fixed_sampling_config_path) if fixed_sampling_config_path else None,
        "rollout": {
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "warmup_steps": args_cli.warmup_steps,
            "dt": dt,
        },
        "policy": policy_summary,
        "reference": reference_summary,
    }
    output_path = output_dir / f"{args_cli.label}_directional_gait_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)

    print(f"[Directional Gait] label={args_cli.label} output={output_path}")
    print(
        "[Directional Gait] policy "
        f"cmd=({args_cli.command_lin_x:+.2f},{args_cli.command_lin_y:+.2f},{args_cli.command_yaw:+.2f}) "
        f"vel=({policy_summary['root_vel_b']['x']['mean']:+.3f},{policy_summary['root_vel_b']['y']['mean']:+.3f},"
        f"{policy_summary['root_ang_vel_b_z']['mean']:+.3f}) "
        f"lin_mae={policy_summary['tracking_mae']['lin_xy']:.3f} yaw_mae={policy_summary['tracking_mae']['yaw']:.3f}"
    )
    contact = policy_summary["contact"]
    print(
        "[Directional Gait] policy contact "
        f"single={contact['single_stance_fraction']:.3f} double_stance={contact['double_stance_fraction']:.3f} "
        f"double_air={contact['double_air_fraction']:.3f} "
        f"double_air_p95_s={contact['double_air_duration_s']['p95']:.3f}"
    )
    print(
        "[Directional Gait] reference "
        f"match_frames={reference_summary['match_frames']}/{reference_summary['total_frames']} "
        f"ratio={reference_summary['match_ratio']:.4f}"
    )
    if reference_summary["contact_matched"]:
        ref_contact = reference_summary["contact_matched"]
        print(
            "[Directional Gait] reference contact "
            f"single={ref_contact['single_stance_fraction']:.3f} "
            f"double_stance={ref_contact['double_stance_fraction']:.3f} "
            f"double_air={ref_contact['double_air_fraction']:.3f}"
        )
    print("[Directional Gait] closest clips:")
    for clip in reference_summary["top_clips_by_mean_velocity_score"]:
        print(
            f"  {clip['name']:28s} score={clip['score']:.2f} "
            f"mean=({clip['mean_vx']:+.3f},{clip['mean_vy']:+.3f},{clip['mean_yaw']:+.3f}) "
            f"straight={clip['straight_ratio_abs_yaw_lt_0p15']:.3f}"
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
