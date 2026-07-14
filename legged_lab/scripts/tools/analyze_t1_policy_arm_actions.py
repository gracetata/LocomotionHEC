#!/usr/bin/env python3
"""Analyze T1 policy arm actions against the AMP motion data."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))

import cli_args  # isort: skip


DEFAULT_CHECKPOINT = (
    "/home/user/Code/T1-Locomotion/legged_lab/logs/rsl_rl/t1_amp/"
    "2026-05-04_20-48-11_fix_joint_order_gp_20_iter5000/model_1000.pt"
)
DEFAULT_MOTION_DIR = (
    "source/legged_lab/legged_lab/data/MotionData/"
    "t1_29dof_accad_g1used_50hz_amp_official"
)
SHOULDER_JOINTS = ["Left_Shoulder_Pitch", "Right_Shoulder_Pitch"]
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


parser = argparse.ArgumentParser(description="Analyze T1 policy shoulder-pitch actions.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-T1-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of envs for rollout sampling.")
parser.add_argument("--max_steps", type=int, default=400, help="Number of policy steps to sample.")
parser.add_argument("--motion_dir", type=str, default=DEFAULT_MOTION_DIR, help="AMP motion directory for comparison.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed override.")
cli_args.add_rsl_rl_args(parser)
parser.set_defaults(checkpoint=DEFAULT_CHECKPOINT)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import joblib  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import legged_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


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


def summarize(name: str, values: np.ndarray) -> None:
    flat = values.reshape(-1)
    print(
        f"{name:42s} mean={flat.mean(): .5f} std={flat.std(): .5f} "
        f"min={flat.min(): .5f} p05={np.percentile(flat, 5): .5f} "
        f"p50={np.percentile(flat, 50): .5f} p95={np.percentile(flat, 95): .5f} max={flat.max(): .5f}"
    )


def temporal_periodicity(values: np.ndarray, dt: float, min_lag_s: float = 0.2, max_lag_s: float = 1.2) -> dict[str, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    temporal_std = centered.std(axis=0)
    valid_envs = temporal_std > 1.0e-5
    if not np.any(valid_envs):
        return {"temporal_std": 0.0, "best_lag": 0.0, "best_corr": 0.0}

    centered = centered[:, valid_envs]
    temporal_std = temporal_std[valid_envs]
    min_lag = max(1, int(round(min_lag_s / dt)))
    max_lag = min(values.shape[0] - 2, int(round(max_lag_s / dt)))
    best_lag = 0
    best_corr = -1.0
    for lag in range(min_lag, max_lag + 1):
        lhs = centered[:-lag]
        rhs = centered[lag:]
        denom = lhs.std(axis=0) * rhs.std(axis=0) + 1.0e-8
        corr = np.mean(np.mean(lhs * rhs, axis=0) / denom)
        if corr > best_corr:
            best_corr = float(corr)
            best_lag = lag

    return {
        "temporal_std": float(np.mean(temporal_std)),
        "best_lag": float(best_lag),
        "best_lag_s": float(best_lag * dt),
        "best_corr": float(best_corr),
    }


def pair_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    denom = left_centered.std(axis=0) * right_centered.std(axis=0) + 1.0e-8
    same_phase = np.mean(np.mean(left_centered * right_centered, axis=0) / denom)
    anti_phase = np.mean(np.mean(left_centered * -right_centered, axis=0) / denom)
    return float(same_phase), float(anti_phase)


def load_motion_stats(repo_root: Path) -> dict[str, np.ndarray]:
    install_numpy_core_pickle_shim()
    motion_dir = Path(args_cli.motion_dir)
    if not motion_dir.is_absolute():
        motion_dir = repo_root / motion_dir

    motion_values: dict[str, list[np.ndarray]] = {joint_name: [] for joint_name in SHOULDER_JOINTS}
    name_to_index = {name: index for index, name in enumerate(LAB_DOF_NAMES)}
    for motion_path in sorted(motion_dir.glob("*.pkl")):
        motion = joblib.load(motion_path)
        joint_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
        for joint_name in SHOULDER_JOINTS:
            motion_values[joint_name].append(joint_pos[:, name_to_index[joint_name]])
    return {joint_name: np.concatenate(values) for joint_name, values in motion_values.items()}


def tensor_column(values: torch.Tensor, column_index: int) -> np.ndarray:
    return values[:, column_index].detach().cpu().numpy()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    repo_root = Path(__file__).resolve().parents[2]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names)
    action_joint_ids = list(action_term._joint_ids)
    shoulder_action_indices = [action_joint_names.index(joint_name) for joint_name in SHOULDER_JOINTS]
    shoulder_joint_ids = [action_joint_ids[action_index] for action_index in shoulder_action_indices]

    print("[T1 Arm Policy Analysis] checkpoint:", checkpoint_path)
    print("[T1 Arm Policy Analysis] action joints:")
    for joint_name, action_index, joint_id in zip(SHOULDER_JOINTS, shoulder_action_indices, shoulder_joint_ids):
        default = robot.data.default_joint_pos[0, joint_id].item()
        limits = robot.data.soft_joint_pos_limits[0, joint_id].detach().cpu().numpy()
        print(
            f"  {joint_name:22s} action_index={action_index:2d} physx_joint_id={joint_id:2d} "
            f"default={default:.5f} soft_limits=[{limits[0]:.5f}, {limits[1]:.5f}]"
        )

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=base_env.device)

    obs = env.get_observations()
    raw_records = {joint_name: [] for joint_name in SHOULDER_JOINTS}
    target_records = {joint_name: [] for joint_name in SHOULDER_JOINTS}
    joint_pos_records = {joint_name: [] for joint_name in SHOULDER_JOINTS}
    joint_vel_records = {joint_name: [] for joint_name in SHOULDER_JOINTS}

    offset = action_term._offset
    scale = action_term._scale
    if not isinstance(offset, torch.Tensor):
        offset = torch.full((base_env.num_envs, len(action_joint_names)), float(offset), device=base_env.device)
    if not isinstance(scale, torch.Tensor):
        scale = torch.full((base_env.num_envs, len(action_joint_names)), float(scale), device=base_env.device)

    with torch.inference_mode():
        for _ in range(args_cli.max_steps):
            actions = policy(obs)
            target_joint_pos = offset + scale * actions
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)

            for joint_name, action_index, joint_id in zip(SHOULDER_JOINTS, shoulder_action_indices, shoulder_joint_ids):
                raw_records[joint_name].append(tensor_column(actions, action_index))
                target_records[joint_name].append(tensor_column(target_joint_pos, action_index))
                joint_pos_records[joint_name].append(tensor_column(robot.data.joint_pos, joint_id))
                joint_vel_records[joint_name].append(tensor_column(robot.data.joint_vel, joint_id))

    motion_stats = load_motion_stats(repo_root)

    print("\n[T1 Arm Policy Analysis] Motion data shoulder pitch stats (rad):")
    for joint_name in SHOULDER_JOINTS:
        summarize(f"motion {joint_name}", motion_stats[joint_name])

    print("\n[T1 Arm Policy Analysis] Policy rollout stats:")
    joint_pos_time_series = {}
    target_time_series = {}
    for joint_name in SHOULDER_JOINTS:
        raw_ts = np.stack(raw_records[joint_name], axis=0)
        target_ts = np.stack(target_records[joint_name], axis=0)
        joint_pos_ts = np.stack(joint_pos_records[joint_name], axis=0)
        joint_vel_ts = np.stack(joint_vel_records[joint_name], axis=0)
        target_time_series[joint_name] = target_ts
        joint_pos_time_series[joint_name] = joint_pos_ts
        raw = raw_ts.reshape(-1)
        target = target_ts.reshape(-1)
        joint_pos = joint_pos_ts.reshape(-1)
        joint_vel = joint_vel_ts.reshape(-1)
        summarize(f"policy raw action {joint_name}", raw)
        summarize(f"policy PD target {joint_name}", target)
        summarize(f"actual joint pos {joint_name}", joint_pos)
        summarize(f"actual joint vel {joint_name}", joint_vel)

        target_period = temporal_periodicity(target_ts, dt=base_env.step_dt)
        joint_period = temporal_periodicity(joint_pos_ts, dt=base_env.step_dt)
        print(
            f"periodicity target {joint_name:22s} temporal_std={target_period['temporal_std']:.5f} "
            f"best_lag={target_period['best_lag']:.0f} ({target_period['best_lag_s']:.3f}s) "
            f"autocorr={target_period['best_corr']:.5f}"
        )
        print(
            f"periodicity actual {joint_name:22s} temporal_std={joint_period['temporal_std']:.5f} "
            f"best_lag={joint_period['best_lag']:.0f} ({joint_period['best_lag_s']:.3f}s) "
            f"autocorr={joint_period['best_corr']:.5f}"
        )

    target_same, target_anti = pair_correlation(
        target_time_series["Left_Shoulder_Pitch"], target_time_series["Right_Shoulder_Pitch"]
    )
    actual_same, actual_anti = pair_correlation(
        joint_pos_time_series["Left_Shoulder_Pitch"], joint_pos_time_series["Right_Shoulder_Pitch"]
    )
    print("\n[T1 Arm Policy Analysis] Left/right shoulder-pitch phase relation:")
    print(f"target same_phase_corr={target_same:.5f} anti_phase_corr={target_anti:.5f}")
    print(f"actual same_phase_corr={actual_same:.5f} anti_phase_corr={actual_anti:.5f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()