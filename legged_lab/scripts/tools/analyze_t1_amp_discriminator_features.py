#!/usr/bin/env python3
"""Analyze T1 AMP discriminator feature gaps and saliency for a checkpoint."""

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


parser = argparse.ArgumentParser(description="Analyze T1 AMP discriminator feature gaps and saliency.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-T1-v0")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=120)
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


FEATURE_RANGES = [
    ("root_rot", 0, 6),
    ("root_lin_vel_b", 6, 9),
    ("root_ang_vel_b", 9, 12),
    ("joint_pos", 12, 39),
    ("joint_vel", 39, 66),
]


def feature_label(index: int) -> str:
    if index < 6:
        return f"root_rot[{index}]"
    if index < 9:
        return f"root_lin_vel_b[{index - 6}]"
    if index < 12:
        return f"root_ang_vel_b[{index - 9}]"
    if index < 39:
        return f"joint_pos/{NO_HEAD_JOINT_NAMES[index - 12]}"
    return f"joint_vel/{NO_HEAD_JOINT_NAMES[index - 39]}"


def print_group_table(name: str, values: np.ndarray) -> None:
    print(name)
    for group_name, start, end in FEATURE_RANGES:
        block = values[start:end]
        print(f"  {group_name:16s} mean={block.mean():.5f} max={block.max():.5f}")


def print_top(name: str, values: np.ndarray, limit: int = 16) -> None:
    order = np.argsort(values)[::-1][:limit]
    print(name)
    for index in order:
        print(f"  {feature_label(int(index)):40s} value={values[index]:.5f}")


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
    policy = runner.get_inference_policy(device=env.device)
    discriminator = runner.alg.amp_discriminator
    obs, _ = env.reset()

    policy_disc_obs = []
    demo_disc_obs = []
    disc_scores = []
    style_rewards = []
    with torch.no_grad():
        for _ in range(args_cli.max_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)
            policy_obs = discriminator.get_disc_obs(obs, flatten_history_dim=False)
            demo_obs = discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
            rewards, scores = discriminator.predict_style_reward(policy_obs, dt=env.unwrapped.step_dt)
            policy_disc_obs.append(policy_obs.detach())
            demo_disc_obs.append(demo_obs.detach())
            disc_scores.append(scores.detach())
            style_rewards.append(rewards.detach())

    policy_disc = torch.cat(policy_disc_obs, dim=0)
    demo_disc = torch.cat(demo_disc_obs, dim=0)
    policy_norm = discriminator.normalize_disc_obs(policy_disc)
    demo_norm = discriminator.normalize_disc_obs(demo_disc)
    feature_gap = torch.mean(torch.abs(policy_norm - demo_norm), dim=(0, 1)).detach().cpu().numpy()
    policy_abs_z = torch.mean(torch.abs(policy_norm), dim=(0, 1)).detach().cpu().numpy()
    demo_abs_z = torch.mean(torch.abs(demo_norm), dim=(0, 1)).detach().cpu().numpy()

    flat_policy = policy_norm.reshape(policy_norm.shape[0], -1).detach().requires_grad_(True)
    score = discriminator(flat_policy).mean()
    (grad,) = torch.autograd.grad(score, flat_policy)
    saliency = grad.abs().reshape(policy_norm.shape[0], discriminator.disc_obs_steps, discriminator.disc_obs_dim)
    saliency_by_dim = saliency.mean(dim=(0, 1)).detach().cpu().numpy()

    policy_score = torch.cat(disc_scores).detach().cpu().numpy()
    policy_style = torch.cat(style_rewards).detach().cpu().numpy()
    print(f"checkpoint={checkpoint}")
    print(f"num_samples={policy_disc.shape[0]} history={policy_disc.shape[1]} dim={policy_disc.shape[2]}")
    print(f"policy_disc_score mean={policy_score.mean():.5f} p05={np.percentile(policy_score, 5):.5f} p95={np.percentile(policy_score, 95):.5f}")
    print(f"policy_style_reward mean={policy_style.mean():.7f} p95={np.percentile(policy_style, 95):.7f}")
    print_group_table("FEATURE_GAP_POLICY_DEMO_ABS_Z", feature_gap)
    print_group_table("POLICY_ABS_Z", policy_abs_z)
    print_group_table("DEMO_ABS_Z", demo_abs_z)
    print_group_table("POLICY_SCORE_GRAD_SALIENCY", saliency_by_dim)
    print_top("TOP_FEATURE_GAPS", feature_gap)
    print_top("TOP_POLICY_SCORE_GRAD_SALIENCY", saliency_by_dim)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()