#!/usr/bin/env python3
"""Run controlled T1 forward/back AMP experiments sequentially."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    label: str
    task: str
    normalizer_mode: str


def stream_process_output(proc: subprocess.Popen, log_file: Path) -> int:
    with log_file.open("w", encoding="utf-8") as file:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            file.write(line)
            file.flush()
    return proc.wait()


def run_command(cmd: list[str], repo_root: Path, log_file: Path, dry_run: bool) -> int | None:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    if dry_run:
        return None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return stream_process_output(proc, log_file)


def latest_run_dir(repo_root: Path, run_name: str) -> Path | None:
    matches = sorted((repo_root / "logs" / "rsl_rl" / "t1_amp").glob(f"*_{run_name}"))
    return matches[-1] if matches else None


def final_checkpoint(run_dir: Path, max_iterations: int) -> Path:
    checkpoint = run_dir / f"model_{max_iterations - 1}.pt"
    if checkpoint.exists():
        return checkpoint
    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def build_train_command(
    python_exec: str,
    experiment: Experiment,
    run_name: str,
    num_envs: int,
    max_iterations: int,
    grad_penalty: float,
    seed: int,
    extra_args: list[str],
) -> list[str]:
    return [
        python_exec,
        "scripts/rsl_rl/train.py",
        "--task",
        experiment.task,
        "--headless",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--seed",
        str(seed),
        "--run_name",
        run_name,
        f"agent.algorithm.amp_cfg.grad_penalty_scale={grad_penalty}",
        f"agent.algorithm.amp_cfg.normalizer_mode={experiment.normalizer_mode}",
        *extra_args,
    ]


def build_analysis_commands(
    python_exec: str,
    task: str,
    checkpoint: Path,
    output_dir: Path,
    analysis_num_envs: int,
    analysis_steps: int,
) -> list[tuple[str, list[str]]]:
    return [
        (
            "arm",
            [
                python_exec,
                "scripts/tools/analyze_t1_arm_sweep.py",
                "--task",
                task,
                "--checkpoint_path",
                str(checkpoint),
                "--num_envs",
                str(analysis_num_envs),
                "--max_steps",
                str(analysis_steps),
                "--motion_dir",
                "source/legged_lab/legged_lab/data/MotionData/t1_29dof_forward_back_filtered_50hz",
                "--output_dir",
                str(output_dir / "arm"),
            ],
        ),
        (
            "gait",
            [
                python_exec,
                "scripts/tools/analyze_t1_gait_style.py",
                "--task",
                task,
                "--checkpoint_path",
                str(checkpoint),
                "--num_envs",
                str(analysis_num_envs),
                "--max_steps",
                str(analysis_steps),
            ],
        ),
        (
            "disc_features",
            [
                python_exec,
                "scripts/tools/analyze_t1_amp_discriminator_features.py",
                "--task",
                task,
                "--checkpoint_path",
                str(checkpoint),
                "--num_envs",
                str(analysis_num_envs),
                "--max_steps",
                str(analysis_steps),
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--grad_penalty", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs/sweeps/t1_fwdback_amp")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--analysis_num_envs", type=int, default=64)
    parser.add_argument("--analysis_steps", type=int, default=360)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a quick 2-iteration smoke test with 128 envs.")
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to train.py, e.g. --extra_args --device cuda:1 agent.device=cuda:1",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    python_exec = sys.executable
    max_iterations = 2 if args.smoke else args.max_iterations
    num_envs = 128 if args.smoke else args.num_envs
    tag = "smoke" if args.smoke else f"iter{max_iterations}"
    gp_tag = f"gp{args.grad_penalty:g}".replace(".", "p")
    start = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    scheduler_dir = (repo_root / args.log_dir / start).resolve()
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    summary_path = scheduler_dir / "summary.json"

    experiments = [
        Experiment("baseline_policy_norm", "LeggedLab-Isaac-AMP-T1-FwdBack-v0", "policy"),
        Experiment("demo_static_norm", "LeggedLab-Isaac-AMP-T1-FwdBack-DemoNorm-v0", "demo_static"),
    ]
    summary = {
        "start_time": dt.datetime.now().isoformat(),
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "grad_penalty": args.grad_penalty,
        "seed": args.seed,
        "runs": [],
    }

    for index, experiment in enumerate(experiments, start=1):
        run_name = f"fwdback_{gp_tag}_{experiment.label}_{tag}"
        train_log = scheduler_dir / f"{index:02d}_{run_name}.train.log"
        run_meta = {
            "index": index,
            "label": experiment.label,
            "task": experiment.task,
            "normalizer_mode": experiment.normalizer_mode,
            "run_name": run_name,
            "train_log": str(train_log),
            "start_time": dt.datetime.now().isoformat(),
        }
        train_cmd = build_train_command(
            python_exec,
            experiment,
            run_name,
            num_envs,
            max_iterations,
            args.grad_penalty,
            args.seed,
            args.extra_args,
        )
        run_meta["train_command"] = train_cmd
        rc = run_command(train_cmd, repo_root, train_log, args.dry_run)
        run_meta["train_return_code"] = rc
        if rc not in (0, None):
            run_meta["status"] = "failed"
            summary["runs"].append(run_meta)
            summary["status"] = "failed"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return int(rc)

        if args.dry_run:
            run_meta["status"] = "dry_run"
            summary["runs"].append(run_meta)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            continue

        run_dir = latest_run_dir(repo_root, run_name)
        if run_dir is None:
            raise FileNotFoundError(f"Could not find log directory for run_name={run_name}")
        checkpoint = final_checkpoint(run_dir, max_iterations)
        run_meta["run_dir"] = str(run_dir)
        run_meta["checkpoint"] = str(checkpoint)

        if not args.skip_analysis:
            analysis_dir = scheduler_dir / f"{index:02d}_{run_name}.analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            run_meta["analysis_dir"] = str(analysis_dir)
            analysis_results = []
            for analysis_name, analysis_cmd in build_analysis_commands(
                python_exec,
                experiment.task,
                checkpoint,
                analysis_dir,
                args.analysis_num_envs,
                args.analysis_steps,
            ):
                analysis_log = analysis_dir / f"{analysis_name}.log"
                analysis_rc = run_command(analysis_cmd, repo_root, analysis_log, args.dry_run)
                analysis_results.append({"name": analysis_name, "return_code": analysis_rc, "log": str(analysis_log)})
                if analysis_rc not in (0, None):
                    run_meta["analysis"] = analysis_results
                    run_meta["status"] = "analysis_failed"
                    summary["runs"].append(run_meta)
                    summary["status"] = "failed"
                    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    return int(analysis_rc)
            run_meta["analysis"] = analysis_results

        run_meta["status"] = "completed"
        run_meta["end_time"] = dt.datetime.now().isoformat()
        summary["runs"].append(run_meta)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["status"] = "completed" if not args.dry_run else "dry_run"
    summary["end_time"] = dt.datetime.now().isoformat()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[T1 FwdBack AMP Experiments] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())