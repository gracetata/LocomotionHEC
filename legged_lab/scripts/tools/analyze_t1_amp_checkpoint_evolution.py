#!/usr/bin/env python3
"""Analyze T1 AMP policy/style evolution across checkpoints.

This tool loads one IsaacLab T1 AMP environment, iterates over saved RSL-RL
checkpoints, and reports whether arm style drifts away from the demonstration
distribution as training progresses.

Core outputs:
- per-checkpoint discriminator score and AMP style reward statistics;
- policy-vs-demo z-score gaps for upper-body joint position/velocity features;
- shoulder-roll natural-arm error, using demo/natural roll signs as reference;
- rollout reset fractions split into timeout and non-timeout termination.

Usage:
    PYTHONPATH="$PWD/source/legged_lab${PYTHONPATH:+:$PYTHONPATH}" \
    /home/hecggdz/miniconda3/envs/env_leglab/bin/python \
      scripts/tools/analyze_t1_amp_checkpoint_evolution.py \
      --task LeggedLab-Isaac-AMP-T1-CmuWalkCore-DemoNorm-v0 \
      --run_dir logs/rsl_rl/t1_amp/2026-05-06_17-43-08_cmu_walk_core_natural_init_rsi05_disc1e5_demo_static_norm_iter5000 \
      --num_envs 64 \
      --max_steps 120
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

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
ARM_JOINT_NAMES = NO_HEAD_JOINT_NAMES[:14]
NATURAL_ROLL_REFERENCE = {
    "Left_Shoulder_Roll": -1.25,
    "Right_Shoulder_Roll": 1.25,
}


parser = argparse.ArgumentParser(description="Analyze T1 AMP style evolution across checkpoints.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-T1-CmuWalkCore-DemoNorm-v0")
parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing model_*.pt checkpoints.")
parser.add_argument("--checkpoint_every", type=int, default=200, help="Keep checkpoints divisible by this value.")
parser.add_argument("--include_final", action="store_true", default=True, help="Always include the largest checkpoint index.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=120)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
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
    """Return the integer suffix from a model_N.pt checkpoint path."""

    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Unsupported checkpoint name: {path.name}")
    return int(match.group(1))


def resolve_checkpoints(run_dir: Path, checkpoint_every: int, include_final: bool) -> list[Path]:
    """Resolve checkpoints to evaluate, sorted by training iteration."""

    checkpoints = sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")

    selected = [path for path in checkpoints if checkpoint_iteration(path) % checkpoint_every == 0]
    if include_final and checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    return sorted(selected, key=checkpoint_iteration)


def make_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg):
    """Construct the configured RSL-RL runner for checkpoint loading."""

    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        return AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def tensor_mean(values: list[torch.Tensor]) -> float:
    """Return the scalar mean for a list of tensors, or 0 for empty lists."""

    if not values:
        return 0.0
    return float(torch.cat([value.reshape(-1).detach().float().cpu() for value in values]).mean().item())


def percentile(values: torch.Tensor, q: float) -> float:
    """Return a percentile for a tensor as a Python float."""

    array = values.detach().reshape(-1).float().cpu().numpy()
    return float(np.percentile(array, q))


def summarize_checkpoint(
    checkpoint: Path,
    runner,
    env: RslRlVecEnvWrapper,
    max_steps: int,
) -> dict[str, Any]:
    """Run deterministic playback for one checkpoint and summarize style metrics."""

    runner.load(str(checkpoint), map_location=runner.device)
    policy = runner.get_inference_policy(device=env.device)
    discriminator = runner.alg.amp_discriminator
    obs, _ = env.reset()

    policy_disc_obs: list[torch.Tensor] = []
    demo_disc_obs: list[torch.Tensor] = []
    disc_scores: list[torch.Tensor] = []
    style_rewards: list[torch.Tensor] = []
    terminated_flags: list[torch.Tensor] = []
    timeout_flags: list[torch.Tensor] = []

    base_env = env.unwrapped
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)
            policy_obs = discriminator.get_disc_obs(obs, flatten_history_dim=False)
            demo_obs = discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
            rewards, scores = discriminator.predict_style_reward(policy_obs, dt=base_env.step_dt)
            policy_disc_obs.append(policy_obs.detach())
            demo_disc_obs.append(demo_obs.detach())
            disc_scores.append(scores.detach())
            style_rewards.append(rewards.detach())
            terminated_flags.append(base_env.reset_terminated.detach().float().cpu())
            timeout_flags.append(base_env.reset_time_outs.detach().float().cpu())

    policy_disc = torch.cat(policy_disc_obs, dim=0)
    demo_disc = torch.cat(demo_disc_obs, dim=0)
    policy_norm = discriminator.normalize_disc_obs(policy_disc)
    demo_norm = discriminator.normalize_disc_obs(demo_disc)
    feature_gap = torch.mean(torch.abs(policy_norm - demo_norm), dim=(0, 1)).detach().cpu().numpy()

    joint_start = 12
    joint_vel_start = 39
    arm_joint_offsets = [NO_HEAD_JOINT_NAMES.index(name) for name in ARM_JOINT_NAMES]
    arm_joint_indices = [joint_start + offset for offset in arm_joint_offsets]
    arm_vel_indices = [joint_vel_start + offset for offset in arm_joint_offsets]

    policy_joint_pos = policy_disc[..., joint_start:joint_vel_start]
    demo_joint_pos = demo_disc[..., joint_start:joint_vel_start]
    policy_joint_vel = policy_disc[..., joint_vel_start:]
    demo_joint_vel = demo_disc[..., joint_vel_start:]
    arm_offsets_tensor = torch.tensor(arm_joint_offsets, device=policy_joint_pos.device)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "iteration": checkpoint_iteration(checkpoint),
        "policy_disc_score_mean": tensor_mean(disc_scores),
        "policy_disc_score_p05": percentile(torch.cat(disc_scores), 5),
        "policy_disc_score_p95": percentile(torch.cat(disc_scores), 95),
        "policy_style_reward_mean": tensor_mean(style_rewards),
        "policy_style_reward_p95": percentile(torch.cat(style_rewards), 95),
        "termination_rate": tensor_mean(terminated_flags),
        "timeout_rate": tensor_mean(timeout_flags),
        "arm_joint_pos_gap_z": float(feature_gap[arm_joint_indices].mean()),
        "arm_joint_vel_gap_z": float(feature_gap[arm_vel_indices].mean()),
        "arm_joint_pos_raw_abs_gap": float(
            torch.mean(torch.abs(policy_joint_pos.index_select(-1, arm_offsets_tensor) - demo_joint_pos.index_select(-1, arm_offsets_tensor))).item()
        ),
        "arm_joint_vel_raw_abs_gap": float(
            torch.mean(torch.abs(policy_joint_vel.index_select(-1, arm_offsets_tensor) - demo_joint_vel.index_select(-1, arm_offsets_tensor))).item()
        ),
    }

    natural_roll_errors = []
    for joint_name in ARM_JOINT_NAMES:
        offset = NO_HEAD_JOINT_NAMES.index(joint_name)
        policy_values = policy_joint_pos[..., offset]
        demo_values = demo_joint_pos[..., offset]
        key = joint_name.lower()
        result[f"policy_{key}_mean"] = float(policy_values.mean().item())
        result[f"policy_{key}_std"] = float(policy_values.std().item())
        result[f"demo_{key}_mean"] = float(demo_values.mean().item())
        result[f"gap_{key}_abs"] = float(torch.mean(torch.abs(policy_values - demo_values)).item())
        if joint_name in NATURAL_ROLL_REFERENCE:
            natural_roll_errors.append(torch.mean(torch.abs(policy_values - NATURAL_ROLL_REFERENCE[joint_name])))
            result[f"natural_error_{key}"] = float(natural_roll_errors[-1].item())

    result["natural_shoulder_roll_error"] = float(torch.stack(natural_roll_errors).mean().item())
    return result


def write_outputs(rows: list[dict[str, Any]], output_csv: Path, output_json: Path) -> None:
    """Write checkpoint summaries as CSV and JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print the high-signal evolution metrics to stdout."""

    columns = [
        "iteration",
        "policy_style_reward_mean",
        "policy_disc_score_mean",
        "arm_joint_pos_gap_z",
        "arm_joint_vel_gap_z",
        "natural_shoulder_roll_error",
        "termination_rate",
        "timeout_rate",
        "policy_left_shoulder_roll_mean",
        "policy_right_shoulder_roll_mean",
        "policy_left_elbow_yaw_mean",
        "policy_right_elbow_yaw_mean",
    ]
    print("\t".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        print("\t".join(values))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Create the environment once, analyze every selected checkpoint, and save results."""

    run_dir = Path(args_cli.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    checkpoints = resolve_checkpoints(run_dir, args_cli.checkpoint_every, args_cli.include_final)
    print(f"[INFO] Selected {len(checkpoints)} checkpoints from {run_dir}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = str(run_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_runner(env, agent_cfg)

    rows = []
    for checkpoint in checkpoints:
        print(f"[INFO] Analyzing {checkpoint.name}...")
        rows.append(summarize_checkpoint(checkpoint, runner, env, args_cli.max_steps))

    output_csv = Path(args_cli.output_csv) if args_cli.output_csv else run_dir / "analysis" / "checkpoint_evolution.csv"
    output_json = Path(args_cli.output_json) if args_cli.output_json else run_dir / "analysis" / "checkpoint_evolution.json"
    if not output_csv.is_absolute():
        output_csv = REPO_ROOT / output_csv
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    write_outputs(rows, output_csv, output_json)
    print_table(rows)
    print(f"[INFO] Wrote CSV: {output_csv}")
    print(f"[INFO] Wrote JSON: {output_json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
