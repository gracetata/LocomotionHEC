#!/usr/bin/env python3
"""Analyze T1 gait timing, foot clearance, and root motion for trained policies."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))

import cli_args  # isort: skip


LEG_JOINTS = [
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


parser = argparse.ArgumentParser(description="Analyze T1 policy gait timing and clearance.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-T1-v0")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=360)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import legged_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def segment_durations(mask: np.ndarray, dt: float) -> list[float]:
    durations: list[float] = []
    flat = mask.astype(bool).reshape(mask.shape[0], -1)
    for column in range(flat.shape[1]):
        start = None
        for index, value in enumerate(flat[:, column]):
            if value and start is None:
                start = index
            if (not value or index == flat.shape[0] - 1) and start is not None:
                end = index if not value else index + 1
                if end - start >= 2:
                    durations.append((end - start) * dt)
                start = None
    return durations


def stats(values: np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def print_stats(label: str, values: np.ndarray | list[float]) -> None:
    result = stats(values)
    print(
        f"{label}: mean={result['mean']:.4f} p50={result['p50']:.4f} "
        f"p90={result['p90']:.4f} p95={result['p95']:.4f} max={result['max']:.4f}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    checkpoint = Path(args_cli.checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    foot_body_ids, foot_body_names = robot.find_bodies(["left_foot_link", "right_foot_link"], preserve_order=True)
    sensor_foot_ids, _ = contact_sensor.find_bodies(["left_foot_link", "right_foot_link"], preserve_order=True)
    leg_joint_ids, leg_joint_names = robot.find_joints(LEG_JOINTS, preserve_order=True)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(str(checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=base_env.device)
    obs, _ = env.reset()

    foot_z_records = []
    air_time_records = []
    contact_time_records = []
    contact_records = []
    single_stance_reward_records = {0.2: [], 0.3: [], 0.4: []}
    root_lin_records = []
    root_ang_records = []
    leg_joint_records = []

    with torch.no_grad():
        for _ in range(args_cli.max_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)

            foot_z = robot.data.body_pos_w[:, foot_body_ids, 2]
            air_time = contact_sensor.data.current_air_time[:, sensor_foot_ids]
            contact_time = contact_sensor.data.current_contact_time[:, sensor_foot_ids]
            in_contact = contact_time > 0.0
            in_mode_time = torch.where(in_contact, contact_time, air_time)
            single_stance = torch.sum(in_contact.int(), dim=1) == 1
            for threshold in single_stance_reward_records:
                reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
                single_stance_reward_records[threshold].append(torch.clamp(reward, max=threshold).detach().cpu().numpy())

            foot_z_records.append(foot_z.detach().cpu().numpy())
            air_time_records.append(air_time.detach().cpu().numpy())
            contact_time_records.append(contact_time.detach().cpu().numpy())
            contact_records.append(in_contact.detach().cpu().numpy())
            root_lin_records.append(robot.data.root_lin_vel_b.detach().cpu().numpy())
            root_ang_records.append(robot.data.root_ang_vel_b.detach().cpu().numpy())
            leg_joint_records.append(robot.data.joint_pos[:, leg_joint_ids].detach().cpu().numpy())

    foot_z = np.stack(foot_z_records, axis=0)
    air_time = np.stack(air_time_records, axis=0)
    contact_time = np.stack(contact_time_records, axis=0)
    contact = np.stack(contact_records, axis=0)
    root_lin = np.stack(root_lin_records, axis=0)
    root_ang = np.stack(root_ang_records, axis=0)
    leg_joint = np.stack(leg_joint_records, axis=0)
    foot_floor = np.percentile(foot_z, 5, axis=0, keepdims=True)
    clearance = np.maximum(foot_z - foot_floor, 0.0)
    air_segments = segment_durations(~contact, base_env.step_dt)
    single_segments = segment_durations(np.sum(contact.astype(np.int32), axis=2) == 1, base_env.step_dt)

    print(f"checkpoint={checkpoint}")
    print(f"iteration={checkpoint_iteration(checkpoint)} num_envs={args_cli.num_envs} steps={args_cli.max_steps} dt={base_env.step_dt}")
    print(f"feet={foot_body_names}")
    print_stats("foot_clearance_all_m", clearance)
    print_stats("foot_clearance_swing_m", clearance[~contact])
    print_stats("current_air_time_s", air_time)
    print_stats("current_contact_time_s", contact_time)
    print_stats("air_segment_duration_s", air_segments)
    print_stats("single_stance_duration_s", single_segments)
    for threshold, values in single_stance_reward_records.items():
        print_stats(f"single_stance_reward_raw_threshold_{threshold:.1f}", np.stack(values, axis=0))
    print(f"single_stance_fraction={np.mean(np.sum(contact.astype(np.int32), axis=2) == 1):.4f}")
    print(f"double_contact_fraction={np.mean(np.sum(contact.astype(np.int32), axis=2) == 2):.4f}")
    print(f"flight_fraction={np.mean(np.sum(contact.astype(np.int32), axis=2) == 0):.4f}")
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        print_stats(f"root_lin_vel_b_{axis}", root_lin[:, :, index])
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        print_stats(f"root_ang_vel_b_{axis}", root_ang[:, :, index])
    for index, joint_name in enumerate(leg_joint_names):
        values = leg_joint[:, :, index]
        print_stats(f"joint_{joint_name}_rad", values)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()