#!/usr/bin/env python3
"""Sequential sweep for AMP discriminator gradient penalty scale.

Behavior:
1) Wait for a running training PID to finish (optional).
2) Launch trainings one-by-one (never overlap).
3) Stop immediately if any run fails.
4) Save full per-run logs and a summary json.

Default sweep reruns the current T1 AMP baseline grad_penalty_scale=20.0
after reward changes, then covers higher values: [20.0, 50.0, 100.0, 200.0, 500.0].
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def pid_alive(pid: int) -> bool:
    """Return True if process exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, poll_sec: float = 10.0) -> None:
    """Block until PID exits."""
    print(f"[Scheduler] Waiting for PID {pid} to finish...")
    while pid_alive(pid):
        print(f"[Scheduler] PID {pid} is still running. Sleep {poll_sec:.0f}s.")
        time.sleep(poll_sec)
    print(f"[Scheduler] PID {pid} finished.")


def format_coef_for_name(coef: float) -> str:
    text = f"{coef:g}"
    return text.replace("-", "m").replace(".", "p")


def stream_process_output(proc: subprocess.Popen, log_file: Path) -> int:
    """Stream subprocess stdout to terminal and log file."""
    with log_file.open("w", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
    return proc.wait()


def parse_coefficients(values: str) -> list[float]:
    coeffs = []
    for item in values.split(","):
        item = item.strip()
        if not item:
            continue
        coeffs.append(float(item))
    if not coeffs:
        raise ValueError("No valid coefficients provided.")
    return coeffs


def build_command(
    python_exec: str,
    task: str,
    num_envs: int,
    max_iterations: int,
    run_name: str,
    coef: float,
    extra_args: Iterable[str],
) -> list[str]:
    cmd = [
        python_exec,
        "scripts/rsl_rl/train.py",
        "--task",
        task,
        "--headless",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--run_name",
        run_name,
        f"agent.algorithm.amp_cfg.grad_penalty_scale={coef}",
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential sweep for discriminator grad_penalty_scale.")
    parser.add_argument("--wait_pid", type=int, default=0, help="Wait for this PID before starting sweep.")
    parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-T1-v0", help="Task name.")
    parser.add_argument("--num_envs", type=int, default=8192, help="Number of envs.")
    parser.add_argument("--max_iterations", type=int, default=6000, help="Iterations per run.")
    parser.add_argument("--base_coef", type=float, default=20.0, help="Current grad_penalty_scale baseline.")
    parser.add_argument(
        "--coefficients",
        type=str,
        default="20.0,50.0,100.0,200.0,500.0",
        help="Comma-separated grad_penalty_scale values. Default includes a baseline rerun at base_coef.",
    )
    parser.add_argument(
        "--strict_upward",
        action="store_true",
        help="Require all coefficients to be > base_coef. Disabled by default so baseline reruns are allowed.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/sweeps/disc_grad_penalty",
        help="Directory for sweep logs and summary.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to train command. Example: --extra_args --device cuda:1 agent.device=cuda:1",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print planned commands.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    coeffs = parse_coefficients(args.coefficients)
    if args.strict_upward:
        non_upward = [c for c in coeffs if c <= args.base_coef]
        if non_upward:
            raise ValueError(
                f"Found non-upward coefficients {non_upward}. "
                f"All values must be > base_coef ({args.base_coef}) unless --strict_upward is omitted."
            )
    python_exec = sys.executable

    sweep_start = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (repo_root / args.log_dir / sweep_start).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"

    summary = {
        "start_time": dt.datetime.now().isoformat(),
        "task": args.task,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "coefficients": coeffs,
        "wait_pid": args.wait_pid,
        "runs": [],
    }

    print(f"[Scheduler] Current discriminator grad_penalty_scale (T1 config): {args.base_coef}")
    print(f"[Scheduler] Sweep coefficients: {coeffs}")
    print(f"[Scheduler] Logs will be saved to: {log_dir}")

    if args.wait_pid > 0 and pid_alive(args.wait_pid):
        if args.dry_run:
            print(f"[Scheduler][DryRun] Would wait for PID {args.wait_pid}.")
        else:
            wait_for_pid(args.wait_pid)
    else:
        print(f"[Scheduler] wait_pid={args.wait_pid} not running. Start immediately.")

    for idx, coef in enumerate(coeffs, start=1):
        coef_tag = format_coef_for_name(coef)
        run_name = f"gp_{coef_tag}_iter{args.max_iterations}"
        cmd = build_command(
            python_exec=python_exec,
            task=args.task,
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            run_name=run_name,
            coef=coef,
            extra_args=args.extra_args,
        )

        run_log = log_dir / f"{idx:02d}_{run_name}.log"
        run_meta = {
            "index": idx,
            "coef": coef,
            "run_name": run_name,
            "command": cmd,
            "log_file": str(run_log),
            "start_time": dt.datetime.now().isoformat(),
        }

        print("\n" + "=" * 100)
        print(f"[Scheduler] Run {idx}/{len(coeffs)}: grad_penalty_scale={coef}")
        print("[Scheduler] Command:")
        print(" ".join(cmd))
        print("=" * 100)

        if args.dry_run:
            run_meta["status"] = "dry_run"
            run_meta["end_time"] = dt.datetime.now().isoformat()
            run_meta["return_code"] = None
            summary["runs"].append(run_meta)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            continue

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # pragma: no cover
            run_meta["status"] = "spawn_failed"
            run_meta["error"] = str(exc)
            run_meta["end_time"] = dt.datetime.now().isoformat()
            run_meta["return_code"] = None
            summary["runs"].append(run_meta)
            summary["status"] = "failed"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[Scheduler][ERROR] Failed to start run: {exc}")
            return 1

        rc = stream_process_output(proc, run_log)
        run_meta["end_time"] = dt.datetime.now().isoformat()
        run_meta["return_code"] = rc

        if rc == 0:
            run_meta["status"] = "completed"
            print(f"[Scheduler] Run completed: grad_penalty_scale={coef}")
        else:
            run_meta["status"] = "failed"
            summary["runs"].append(run_meta)
            summary["status"] = "failed"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[Scheduler][ERROR] Run failed with return code {rc}. Stopping sweep.")
            return rc

        summary["runs"].append(run_meta)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["status"] = "completed"
    summary["end_time"] = dt.datetime.now().isoformat()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[Scheduler] Sweep completed successfully.")
    print(f"[Scheduler] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
