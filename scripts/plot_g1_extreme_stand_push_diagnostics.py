#!/usr/bin/env python3
"""Plot post-push actor, joint and actuator diagnostics from a MuJoCo run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _columns(rows: list[dict[str, str]], prefix: str) -> tuple[list[str], np.ndarray]:
    names = [name for name in rows[0] if name.startswith(prefix)]
    values = np.asarray(
        [[float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    return names, values


def _row_rms(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=1)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    result[valid] = np.sqrt(
        np.sum(np.where(finite, np.square(values), 0.0), axis=1)[valid]
        / counts[valid]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--motion-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    with args.motion_trace.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader if row]
    if len(rows) < 8:
        raise SystemExit("Motion trace requires at least 8 samples.")

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    _, positions = _columns(rows, "qpos_rad/")
    velocity_names, velocities = _columns(rows, "qvel_rad_s/")
    _, jerks = _columns(rows, "jerk_fd_rad_s3/")
    action_names, actions = _columns(rows, "actor_action/")
    _, torques = _columns(rows, "pd_torque_command_nm/")
    _, torque_limits = _columns(rows, "actuator_torque_limit_nm/")
    dt = float(np.median(np.diff(times)))
    action_rate = np.vstack(
        [np.full((1, actions.shape[1]), np.nan), np.diff(actions, axis=0) / dt]
    )
    velocity_rms = _row_rms(velocities)
    jerk_rms = _row_rms(jerks)
    action_rate_rms = _row_rms(action_rate)
    torque_rms = _row_rms(torques)
    torque_max = np.nanmax(np.abs(torques), axis=1)
    finite_limit = np.logical_and(np.isfinite(torque_limits), torque_limits > 1.0e-9)
    saturation = np.mean(
        np.logical_and(
            finite_limit,
            np.abs(torques) >= 0.98 * torque_limits,
        ),
        axis=1,
    )

    stand = report["extreme_stand_recovery"]
    diagnostics = stand.get("large_push", {}).get("post_push_diagnostics", {})
    events = diagnostics.get("events", [])
    analyzed_event = diagnostics.get("analyzed_event", {})
    post_start = float(analyzed_event.get("end_time_s", 0.0)) + float(
        diagnostics.get("post_push_settle_s", 0.0)
    )
    top_joint_names = [
        item["joint"]
        for item in diagnostics.get("post_push", {}).get(
            "highest_position_hf_joints", []
        )[:4]
    ]
    joint_name_to_index = {
        name.removeprefix("qvel_rad_s/"): index
        for index, name in enumerate(velocity_names)
    }

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for event in events:
        start = float(event["time_s"])
        end = float(event.get("end_time_s", start))
        for axis in axes:
            axis.axvspan(start, end, color="#d62728", alpha=0.20)
    for axis in axes:
        if post_start > 0.0:
            axis.axvline(post_start, color="#9467bd", linestyle="--", linewidth=1.2)
        axis.grid(True, alpha=0.25)

    axes[0].plot(times, velocity_rms, label="joint velocity RMS (rad/s)")
    axes[0].plot(times, jerk_rms / 1000.0, label="joint jerk RMS / 1000")
    axes[0].set_ylabel("state motion")
    axes[0].legend(loc="upper right")

    axes[1].plot(times, action_rate_rms, color="#ff7f0e")
    axes[1].set_ylabel("actor Δaction/s RMS")

    axes[2].plot(times, torque_rms, label="PD torque RMS")
    axes[2].plot(times, torque_max, label="PD torque max |τ|", alpha=0.8)
    axes[2].set_ylabel("torque (Nm)")
    axes[2].legend(loc="upper right")

    axes[3].plot(times, 100.0 * saturation, color="#8c564b")
    axes[3].set_ylabel("torque saturation (%)")

    if top_joint_names:
        for joint_name in top_joint_names:
            index = joint_name_to_index.get(joint_name)
            if index is None:
                continue
            centered = positions[:, index] - np.mean(positions[times >= post_start, index])
            axes[4].plot(times, centered, label=joint_name)
        axes[4].legend(loc="upper right", ncol=2, fontsize=8)
    else:
        axes[4].plot(times, np.sqrt(np.nanmean(np.square(positions), axis=1)))
    axes[4].set_ylabel("top joint q - post mean (rad)")
    axes[4].set_xlabel("MuJoCo time (s)")

    diagnosis = diagnostics.get("diagnosis", "not_available")
    force = stand.get("large_push", {}).get("force_n")
    duration = stand.get("large_push", {}).get("duration_s")
    fig.suptitle(
        f"Extreme Stand large torso push: {force} N × {duration} s\n"
        f"diagnosis={diagnosis}; red=push, purple=post-push analysis start"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
