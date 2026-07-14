#!/usr/bin/env python3
"""Batch-analyze T1 arm DoFs for trained policies against AMP motion data."""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))

import cli_args  # isort: skip


DEFAULT_MOTION_DIR = (
    "source/legged_lab/legged_lab/data/MotionData/"
    "t1_29dof_accad_g1used_50hz_amp_official"
)

ARM_JOINTS = [
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
]

ARM_PAIRS = [
    ("Shoulder_Pitch", "Left_Shoulder_Pitch", "Right_Shoulder_Pitch"),
    ("Shoulder_Roll", "Left_Shoulder_Roll", "Right_Shoulder_Roll"),
    ("Elbow_Pitch", "Left_Elbow_Pitch", "Right_Elbow_Pitch"),
    ("Elbow_Yaw", "Left_Elbow_Yaw", "Right_Elbow_Yaw"),
    ("Wrist_Pitch", "Left_Wrist_Pitch", "Right_Wrist_Pitch"),
    ("Wrist_Yaw", "Left_Wrist_Yaw", "Right_Wrist_Yaw"),
    ("Hand_Roll", "Left_Hand_Roll", "Right_Hand_Roll"),
]

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


parser = argparse.ArgumentParser(description="Batch-analyze T1 policy arm DoFs.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-T1-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of envs for rollout sampling.")
parser.add_argument("--max_steps", type=int, default=360, help="Number of policy steps per checkpoint.")
parser.add_argument("--motion_dir", type=str, default=DEFAULT_MOTION_DIR, help="AMP motion directory for comparison.")
parser.add_argument("--run_dir", action="append", default=[], help="Training run directory to sample checkpoints from.")
parser.add_argument("--checkpoint_path", action="append", default=[], help="Explicit checkpoint path to analyze.")
parser.add_argument(
    "--iterations",
    type=str,
    default="200,1000,2000,4000,5999",
    help="Comma-separated model iterations to collect from each run_dir.",
)
parser.add_argument("--include_latest", action="store_true", help="Also include the latest model_*.pt per run_dir.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory for CSV and markdown outputs.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed override.")
cli_args.add_rsl_rl_args(parser)
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


def parse_iterations(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def run_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"gp_(\d+)(?:p(\d+))?_iter", path.name)
    if not match:
        return (10**9, path.name)
    integer = int(match.group(1))
    decimal = match.group(2) or "0"
    return (int(float(f"{integer}.{decimal}")), path.name)


def collect_checkpoints(repo_root: Path) -> list[Path]:
    checkpoints: list[Path] = []
    seen: set[Path] = set()
    iterations = parse_iterations(args_cli.iterations)

    for run_dir_text in args_cli.run_dir:
        run_dir = Path(run_dir_text)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
        for iteration in iterations:
            candidate = run_dir / f"model_{iteration}.pt"
            if candidate.exists() and candidate not in seen:
                checkpoints.append(candidate)
                seen.add(candidate)
        if args_cli.include_latest:
            available = sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)
            if available and available[-1] not in seen:
                checkpoints.append(available[-1])
                seen.add(available[-1])

    for checkpoint_text in args_cli.checkpoint_path:
        checkpoint = Path(checkpoint_text)
        if not checkpoint.is_absolute():
            checkpoint = repo_root / checkpoint
        if checkpoint.exists() and checkpoint not in seen:
            checkpoints.append(checkpoint)
            seen.add(checkpoint)

    checkpoints.sort(key=lambda path: (run_sort_key(path.parent), checkpoint_iteration(path)))
    return checkpoints


def side_and_dof(joint_name: str) -> tuple[str, str]:
    if joint_name.startswith("Left_"):
        return "Left", joint_name.removeprefix("Left_")
    if joint_name.startswith("Right_"):
        return "Right", joint_name.removeprefix("Right_")
    return "", joint_name


def base_stats(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1)
    p05 = float(np.percentile(flat, 5))
    p50 = float(np.percentile(flat, 50))
    p95 = float(np.percentile(flat, 95))
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "max": float(flat.max()),
        "range_p95_p05": p95 - p05,
    }


def temporal_periodicity(values: np.ndarray, dt: float, min_lag_s: float = 0.2, max_lag_s: float = 1.2) -> dict[str, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    temporal_std = centered.std(axis=0)
    temporal_range = np.percentile(values, 95, axis=0) - np.percentile(values, 5, axis=0)
    valid_envs = temporal_std > 1.0e-5
    if not np.any(valid_envs):
        return {
            "temporal_std": 0.0,
            "temporal_range_p95_p05": float(np.mean(temporal_range)),
            "best_lag": 0.0,
            "best_lag_s": 0.0,
            "best_corr": 0.0,
        }

    centered = centered[:, valid_envs]
    temporal_std = temporal_std[valid_envs]
    temporal_range = temporal_range[valid_envs]
    min_lag = max(1, int(round(min_lag_s / dt)))
    max_lag = min(values.shape[0] - 2, int(round(max_lag_s / dt)))
    best_lag = 0
    best_corr = -1.0
    for lag in range(min_lag, max_lag + 1):
        lhs = centered[:-lag]
        rhs = centered[lag:]
        denom = lhs.std(axis=0) * rhs.std(axis=0) + 1.0e-8
        corr = float(np.mean(np.mean(lhs * rhs, axis=0) / denom))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return {
        "temporal_std": float(np.mean(temporal_std)),
        "temporal_range_p95_p05": float(np.mean(temporal_range)),
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


def motion_time_metrics(sequences: list[np.ndarray], dt: float) -> dict[str, float]:
    metrics = [temporal_periodicity(sequence[:, None], dt=dt) for sequence in sequences if len(sequence) > 4]
    if not metrics:
        return temporal_periodicity(np.zeros((2, 1), dtype=np.float32), dt=dt)
    return {key: float(np.mean([metric[key] for metric in metrics])) for key in metrics[0]}


def load_motion_sequences(repo_root: Path) -> dict[str, list[np.ndarray]]:
    install_numpy_core_pickle_shim()
    motion_dir = Path(args_cli.motion_dir)
    if not motion_dir.is_absolute():
        motion_dir = repo_root / motion_dir

    name_to_index = {name: index for index, name in enumerate(LAB_DOF_NAMES)}
    motion_values: dict[str, list[np.ndarray]] = {joint_name: [] for joint_name in ARM_JOINTS}
    for motion_path in sorted(motion_dir.glob("*.pkl")):
        motion = joblib.load(motion_path)
        joint_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
        for joint_name in ARM_JOINTS:
            motion_values[joint_name].append(joint_pos[:, name_to_index[joint_name]])
    return motion_values


def make_metric_row(
    *,
    run_name: str,
    checkpoint: str,
    iteration: int,
    source: str,
    joint_name: str,
    values: np.ndarray,
    dt: float,
    motion_range: float | None = None,
    soft_lower: float | None = None,
    soft_upper: float | None = None,
) -> dict[str, object]:
    side, dof_name = side_and_dof(joint_name)
    row: dict[str, object] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "iteration": iteration,
        "source": source,
        "joint_name": joint_name,
        "side": side,
        "dof_name": dof_name,
    }
    row.update(base_stats(values))
    row.update(temporal_periodicity(values, dt=dt))
    row["motion_range_p95_p05"] = "" if motion_range is None else motion_range
    row["range_ratio_to_motion"] = "" if motion_range in (None, 0.0) else row["range_p95_p05"] / motion_range
    if soft_lower is not None and soft_upper is not None:
        row["frac_below_soft"] = float(np.mean(values < soft_lower))
        row["frac_above_soft"] = float(np.mean(values > soft_upper))
    else:
        row["frac_below_soft"] = ""
        row["frac_above_soft"] = ""
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, metric_rows: list[dict[str, object]], phase_rows: list[dict[str, object]]) -> None:
    final_actual = [
        row
        for row in metric_rows
        if row["source"] == "actual" and row["iteration"] in (4000, 5999) and "gp_" in str(row["run_name"])
    ]
    final_target = [
        row
        for row in metric_rows
        if row["source"] == "target" and row["iteration"] in (4000, 5999) and "gp_" in str(row["run_name"])
    ]
    shoulder_actual = [row for row in final_actual if row["dof_name"] in ("Shoulder_Pitch", "Shoulder_Roll")]
    wrist_actual = [row for row in final_actual if row["dof_name"] in ("Wrist_Pitch", "Wrist_Yaw", "Hand_Roll")]
    shoulder_actual.sort(key=lambda row: float(row["range_ratio_to_motion"] or 0.0))
    wrist_actual.sort(key=lambda row: float(row["range_ratio_to_motion"] or 0.0), reverse=True)
    final_target.sort(key=lambda row: float(row["range_ratio_to_motion"] or 0.0), reverse=True)

    lines = ["# T1 Arm Sweep Analysis", ""]
    lines.append("## Lowest Shoulder Actual Amplitude Ratios")
    lines.append("run | iter | joint | actual_range | motion_range | ratio | temporal_std")
    lines.append("--- | ---: | --- | ---: | ---: | ---: | ---:")
    for row in shoulder_actual[:20]:
        lines.append(
            f"{row['run_name']} | {row['iteration']} | {row['joint_name']} | "
            f"{float(row['range_p95_p05']):.4f} | {float(row['motion_range_p95_p05']):.4f} | "
            f"{float(row['range_ratio_to_motion']):.3f} | {float(row['temporal_std']):.4f}"
        )

    lines.extend(["", "## Highest Wrist/Hand Actual Amplitude Ratios"])
    lines.append("run | iter | joint | actual_range | motion_range | ratio | temporal_std")
    lines.append("--- | ---: | --- | ---: | ---: | ---: | ---:")
    for row in wrist_actual[:20]:
        lines.append(
            f"{row['run_name']} | {row['iteration']} | {row['joint_name']} | "
            f"{float(row['range_p95_p05']):.4f} | {float(row['motion_range_p95_p05']):.4f} | "
            f"{float(row['range_ratio_to_motion']):.3f} | {float(row['temporal_std']):.4f}"
        )

    lines.extend(["", "## Largest Target Amplitude Ratios"])
    lines.append("run | iter | joint | target_range | motion_range | ratio | target_soft_high_frac")
    lines.append("--- | ---: | --- | ---: | ---: | ---: | ---:")
    for row in final_target[:20]:
        lines.append(
            f"{row['run_name']} | {row['iteration']} | {row['joint_name']} | "
            f"{float(row['range_p95_p05']):.4f} | {float(row['motion_range_p95_p05']):.4f} | "
            f"{float(row['range_ratio_to_motion']):.3f} | {float(row['frac_above_soft']):.3f}"
        )

    lines.extend(["", "## Shoulder Pitch Left/Right Phase"])
    lines.append("run | iter | source | same_phase | anti_phase")
    lines.append("--- | ---: | --- | ---: | ---:")
    for row in phase_rows:
        if row["dof_name"] == "Shoulder_Pitch" and row["source"] in ("motion", "actual", "target"):
            lines.append(
                f"{row['run_name']} | {row['iteration']} | {row['source']} | "
                f"{float(row['same_phase_corr']):.4f} | {float(row['anti_phase_corr']):.4f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    repo_root = Path(__file__).resolve().parents[2]
    checkpoints = collect_checkpoints(repo_root)
    if not checkpoints:
        raise ValueError("No checkpoints found. Provide --run_dir and/or --checkpoint.")

    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else repo_root / "outputs" / "t1_arm_sweep_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = str(output_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names)
    action_joint_ids = list(action_term._joint_ids)
    action_indices = [action_joint_names.index(joint_name) for joint_name in ARM_JOINTS]
    joint_ids = [action_joint_ids[action_index] for action_index in action_indices]
    soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    offset = action_term._offset
    scale = action_term._scale
    if not isinstance(offset, torch.Tensor):
        offset = torch.full((base_env.num_envs, len(action_joint_names)), float(offset), device=base_env.device)
    if not isinstance(scale, torch.Tensor):
        scale = torch.full((base_env.num_envs, len(action_joint_names)), float(scale), device=base_env.device)

    print("[T1 Arm Sweep Analysis] Output:", output_dir)
    print("[T1 Arm Sweep Analysis] Checkpoints:")
    for checkpoint in checkpoints:
        print("  ", checkpoint)

    motion_sequences = load_motion_sequences(repo_root)
    motion_ranges = {}
    metric_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    motion_dt = 0.02
    for joint_name in ARM_JOINTS:
        values = np.concatenate(motion_sequences[joint_name])[:, None]
        motion_time = motion_time_metrics(motion_sequences[joint_name], dt=motion_dt)
        row = make_metric_row(
            run_name="motion_dataset",
            checkpoint="",
            iteration=-1,
            source="motion",
            joint_name=joint_name,
            values=values,
            dt=motion_dt,
        )
        row.update(motion_time)
        motion_ranges[joint_name] = float(row["range_p95_p05"])
        metric_rows.append(row)

    for dof_name, left_name, right_name in ARM_PAIRS:
        same_values = []
        anti_values = []
        for left, right in zip(motion_sequences[left_name], motion_sequences[right_name]):
            same, anti = pair_correlation(left[:, None], right[:, None])
            same_values.append(same)
            anti_values.append(anti)
        phase_rows.append(
            {
                "run_name": "motion_dataset",
                "checkpoint": "",
                "iteration": -1,
                "source": "motion",
                "dof_name": dof_name,
                "same_phase_corr": float(np.mean(same_values)),
                "anti_phase_corr": float(np.mean(anti_values)),
            }
        )

    for checkpoint in checkpoints:
        checkpoint_path = Path(retrieve_file_path(str(checkpoint)))
        run_name = checkpoint_path.parent.name
        iteration = checkpoint_iteration(checkpoint_path)
        print(f"[T1 Arm Sweep Analysis] Loading {run_name}/model_{iteration}.pt")
        runner.load(str(checkpoint_path), map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=base_env.device)
        obs, _ = env.reset()
        raw_records = []
        target_records = []
        actual_records = []
        velocity_records = []
        with torch.no_grad():
            for _ in range(args_cli.max_steps):
                actions = policy(obs)
                target_joint_pos = offset + scale * actions
                obs, _, dones, _ = env.step(actions)
                runner.alg.policy.reset(dones)
                raw_records.append(actions[:, action_indices].detach().cpu().numpy())
                target_records.append(target_joint_pos[:, action_indices].detach().cpu().numpy())
                actual_records.append(robot.data.joint_pos[:, joint_ids].detach().cpu().numpy())
                velocity_records.append(robot.data.joint_vel[:, joint_ids].detach().cpu().numpy())

        raw_ts = np.stack(raw_records, axis=0)
        target_ts = np.stack(target_records, axis=0)
        actual_ts = np.stack(actual_records, axis=0)
        velocity_ts = np.stack(velocity_records, axis=0)

        for joint_index, joint_name in enumerate(ARM_JOINTS):
            soft_lower = float(soft_limits[joint_index, 0])
            soft_upper = float(soft_limits[joint_index, 1])
            for source, values in (
                ("raw_action", raw_ts[:, :, joint_index]),
                ("target", target_ts[:, :, joint_index]),
                ("actual", actual_ts[:, :, joint_index]),
                ("actual_vel", velocity_ts[:, :, joint_index]),
            ):
                metric_rows.append(
                    make_metric_row(
                        run_name=run_name,
                        checkpoint=str(checkpoint_path),
                        iteration=iteration,
                        source=source,
                        joint_name=joint_name,
                        values=values,
                        dt=base_env.step_dt,
                        motion_range=motion_ranges[joint_name],
                        soft_lower=soft_lower if source in ("target", "actual") else None,
                        soft_upper=soft_upper if source in ("target", "actual") else None,
                    )
                )

        for dof_name, left_name, right_name in ARM_PAIRS:
            left_index = ARM_JOINTS.index(left_name)
            right_index = ARM_JOINTS.index(right_name)
            for source, values in (("target", target_ts), ("actual", actual_ts), ("raw_action", raw_ts)):
                same, anti = pair_correlation(values[:, :, left_index], values[:, :, right_index])
                phase_rows.append(
                    {
                        "run_name": run_name,
                        "checkpoint": str(checkpoint_path),
                        "iteration": iteration,
                        "source": source,
                        "dof_name": dof_name,
                        "same_phase_corr": same,
                        "anti_phase_corr": anti,
                    }
                )

    metrics_path = output_dir / "arm_metrics.csv"
    phase_path = output_dir / "arm_phase.csv"
    report_path = output_dir / "arm_report.md"
    write_csv(metrics_path, metric_rows)
    write_csv(phase_path, phase_rows)
    write_report(report_path, metric_rows, phase_rows)
    print("[T1 Arm Sweep Analysis] Metrics:", metrics_path)
    print("[T1 Arm Sweep Analysis] Phase:", phase_path)
    print("[T1 Arm Sweep Analysis] Report:", report_path)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()