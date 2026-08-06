#!/usr/bin/env python3
"""Analyze Extreme Stand interactive CSV trials and plot joint diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _matrix(frame: pd.DataFrame, prefix: str) -> tuple[list[str], np.ndarray]:
    columns = [name for name in frame.columns if name.startswith(f"{prefix}/")]
    return columns, frame[columns].to_numpy(dtype=np.float64)


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def _rolling_joint_rms(jerk: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(jerk), np.nan, dtype=np.float64)
    squared = np.mean(np.square(jerk), axis=1)
    if len(jerk) >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        result[window - 1 :] = np.sqrt(np.convolve(squared, kernel, mode="valid"))
    return result


def _longest_true_duration(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def _highpass_rms(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 5.0) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(centered, axis=0)
    spectrum[frequencies < cutoff_hz] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(centered), axis=0)
    return np.sqrt(np.mean(np.square(filtered), axis=0))


def _load_trial(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    time = frame["space_trial_time_s"].to_numpy(dtype=np.float64)
    dt = float(np.median(np.diff(time)))
    jerk_columns, jerk = _matrix(frame, "jerk_fd_rad_s3")
    _, torque = _matrix(frame, "actuator_force_nm")
    _, torque_command = _matrix(frame, "pd_torque_command_nm")
    _, torque_limit = _matrix(frame, "actuator_torque_limit_nm")
    _, position = _matrix(frame, "qpos_rad")
    _, target_position = _matrix(frame, "target_qpos_rad")
    _, action = _matrix(frame, "actor_action")
    return {
        "path": path,
        "frame": frame,
        "time": time,
        "dt": dt,
        "joint_names": [name.split("/", 1)[1] for name in jerk_columns],
        "jerk": jerk,
        "torque": torque,
        "torque_command": torque_command,
        "torque_limit": torque_limit,
        "position": position,
        "target_position": target_position,
        "action": action,
    }


def _summarize_trial(data: dict[str, object]) -> dict[str, object]:
    frame = data["frame"]
    time = data["time"]
    dt = data["dt"]
    jerk = data["jerk"]
    torque = data["torque"]
    torque_limit = data["torque_limit"]
    position = data["position"]
    target_position = data["target_position"]
    action = data["action"]
    after_half_second = time >= 0.5
    final_second = time >= max(0.0, float(time[-1]) - 1.0)
    after_three_seconds = time >= 3.0
    rolling_jerk = _rolling_joint_rms(jerk, max(2, round(0.5 / dt)))
    torque_rate = np.diff(torque, axis=0) / dt
    action_rate = np.diff(action, axis=0) / dt
    action_second_difference = np.diff(action, n=2, axis=0) / (dt * dt)
    foot_error = frame["feet/planar_distance_error_m"].to_numpy(dtype=np.float64)
    return {
        "trial": data["path"].stem.split("_")[1],
        "scenario": str(frame["space_trial_scenario"].iloc[0]),
        "samples": len(frame),
        "duration_s": float(time[-1]),
        "jerk_rms_after_0p5_rad_s3": _rms(jerk[after_half_second]),
        "jerk_p99_after_0p5_rad_s3": float(np.percentile(np.abs(jerk[after_half_second]), 99)),
        "jerk_max_after_0p5_rad_s3": float(np.max(np.abs(jerk[after_half_second]))),
        "jerk_rms_final_1s_rad_s3": _rms(jerk[final_second]),
        "rolling_jerk_rms_max_rad_s3": float(np.nanmax(rolling_jerk)),
        "rolling_jerk_over_100_duration_s": _longest_true_duration(
            np.nan_to_num(rolling_jerk) > 100.0, dt
        ),
        "torque_rms_after_0p5_nm": _rms(torque[after_half_second]),
        "torque_max_after_0p5_nm": float(np.max(np.abs(torque[after_half_second]))),
        "torque_rate_p99_nm_s": float(np.percentile(np.abs(torque_rate), 99)),
        "torque_rate_max_nm_s": float(np.max(np.abs(torque_rate))),
        "torque_saturation_fraction": float(
            np.mean(np.abs(torque) >= 0.95 * np.maximum(torque_limit, 1.0e-12))
        ),
        "position_tracking_error_rms_rad": _rms(position - target_position),
        "position_variation_mean_std_rad": float(
            np.mean(np.std(position[after_half_second], axis=0))
        ),
        "action_rate_rms_s_inv": _rms(action_rate),
        "action_second_diff_rms_s2_inv": _rms(action_second_difference),
        "foot_distance_error_rms_m": _rms(foot_error),
        "foot_distance_error_max_m": float(np.max(np.abs(foot_error))),
        "settled_jerk_rms_after_3s_rad_s3": (
            _rms(jerk[after_three_seconds]) if np.any(after_three_seconds) else math.nan
        ),
        "settled_torque_rms_after_3s_nm": (
            _rms(torque[after_three_seconds]) if np.any(after_three_seconds) else math.nan
        ),
    }


def _plot_trial_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    labels = [f"T{int(value):02d}" for value in summary["trial"]]
    x = np.arange(len(summary))
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axis = axes[0, 0]
    axis.bar(x - 0.18, summary["jerk_rms_after_0p5_rad_s3"], 0.36, label="RMS after 0.5 s")
    axis.bar(x + 0.18, summary["jerk_rms_final_1s_rad_s3"], 0.36, label="RMS final 1 s")
    axis.set_yscale("log")
    axis.set_ylabel("Joint jerk RMS [rad/s³]")
    axis.set_title("Jerk: transient vs final second")
    axis.set_xticks(x, labels)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    axis = axes[0, 1]
    axis.bar(x - 0.18, summary["torque_rms_after_0p5_nm"], 0.36, label="RMS")
    axis.bar(x + 0.18, summary["torque_max_after_0p5_nm"], 0.36, label="Max |torque|")
    axis.set_ylabel("Actuator torque [N·m]")
    axis.set_title("Torque magnitude")
    axis.set_xticks(x, labels)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    axis = axes[1, 0]
    axis.bar(x, summary["rolling_jerk_over_100_duration_s"])
    axis.set_ylabel("Longest duration [s]")
    axis.set_title("Continuous 0.5 s-window jerk RMS > 100 rad/s³")
    axis.set_xticks(x, labels)
    axis.grid(True, axis="y", alpha=0.25)
    axis = axes[1, 1]
    axis.bar(x - 0.18, 100.0 * summary["torque_saturation_fraction"], 0.36, label="Saturation")
    second_axis = axis.twinx()
    second_axis.bar(
        x + 0.18,
        100.0 * summary["foot_distance_error_max_m"],
        0.36,
        color="tab:orange",
        label="Max foot-distance error",
    )
    axis.set_ylabel("Torque saturation samples [%]")
    second_axis.set_ylabel("Foot-distance error [cm]")
    axis.set_title("Limits and feet geometry")
    axis.set_xticks(x, labels)
    handles, names = axis.get_legend_handles_labels()
    handles_2, names_2 = second_axis.get_legend_handles_labels()
    axis.legend(handles + handles_2, names + names_2, loc="upper left")
    figure.suptitle("Extreme Stand V4 interactive trial comparison", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _analyze_last_trial(data: dict[str, object], output_dir: Path) -> dict[str, object]:
    frame = data["frame"]
    time = data["time"]
    dt = data["dt"]
    sample_rate_hz = 1.0 / dt
    joints = data["joint_names"]
    jerk = data["jerk"]
    torque = data["torque"]
    torque_command = data["torque_command"]
    torque_limit = data["torque_limit"]
    position = data["position"]
    target_position = data["target_position"]
    force_columns = [name for name in frame if name.startswith("external/force_")]
    force = np.linalg.norm(frame[force_columns].to_numpy(dtype=np.float64), axis=1)
    rolling_jerk = _rolling_joint_rms(jerk, max(2, round(0.5 / dt)))
    torque_rms = np.sqrt(np.mean(np.square(torque), axis=1))
    torque_max = np.max(np.abs(torque), axis=1)
    tracking_error = np.max(np.abs(position - target_position), axis=1)
    foot_error = frame["feet/planar_distance_error_m"].to_numpy(dtype=np.float64)
    root_xy = np.hypot(
        frame["root/pos_x_m"] - frame["root/pos_x_m"].iloc[0],
        frame["root/pos_y_m"] - frame["root/pos_y_m"].iloc[0],
    )

    figure, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True, constrained_layout=True)
    axes[0].plot(time, rolling_jerk, color="tab:red", label="0.5 s rolling all-joint jerk RMS")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("Jerk [rad/s³]")
    axes[0].legend()
    force_axis = axes[0].twinx()
    force_axis.fill_between(time, 0.0, force, color="magenta", alpha=0.18)
    force_axis.set_ylabel("Force [N]")
    axes[1].plot(time, torque_rms, label="Joint torque RMS")
    axes[1].plot(time, torque_max, label="Max |joint torque|")
    axes[1].set_ylabel("Torque [N·m]")
    axes[1].legend()
    axes[2].plot(time, tracking_error, color="tab:orange", label="Max |q - q_target|")
    axes[2].set_ylabel("Tracking error [rad]")
    axes[2].legend()
    axes[3].plot(time, 100.0 * foot_error, color="tab:green", label="Foot-distance error")
    axes[3].plot(time, 100.0 * root_xy, color="tab:purple", label="Root XY displacement")
    axes[3].set_ylabel("Distance [cm]")
    axes[3].set_xlabel("Trial time [s]")
    axes[3].legend()
    for axis in axes:
        axis.axvspan(0.0, 0.2, color="magenta", alpha=0.08)
        axis.grid(True, alpha=0.25)
    figure.suptitle(f"{data['path'].stem}: push, recovery transient, and settling", fontsize=15)
    figure.savefig(output_dir / "trial_011_overview.png", dpi=180)
    plt.close(figure)

    recovery = np.logical_and(time >= 0.2, time < 3.0)
    settled = time >= 3.0
    ranks = np.argsort(np.sqrt(np.mean(np.square(jerk[recovery]), axis=0)))[::-1]
    top = ranks[:6]
    highpass_torque = _highpass_rms(torque[settled], sample_rate_hz)
    highpass_position = _highpass_rms(position[settled], sample_rate_hz)
    joint_rows = []
    for index, joint in enumerate(joints):
        joint_rows.append(
            {
                "joint": joint,
                "recovery_jerk_rms_rad_s3": _rms(jerk[recovery, index]),
                "recovery_jerk_max_rad_s3": float(np.max(np.abs(jerk[recovery, index]))),
                "recovery_torque_rms_nm": _rms(torque[recovery, index]),
                "recovery_torque_max_nm": float(np.max(np.abs(torque[recovery, index]))),
                "settled_after_3s_jerk_rms_rad_s3": _rms(jerk[settled, index]),
                "settled_after_3s_position_std_rad": float(np.std(position[settled, index])),
                "settled_after_3s_position_highpass_5hz_rms_rad": float(highpass_position[index]),
                "settled_after_3s_torque_highpass_5hz_rms_nm": float(highpass_torque[index]),
                "torque_limit_nm": float(np.max(torque_limit[:, index])),
                "max_torque_limit_ratio": float(
                    np.max(np.abs(torque[:, index]) / np.maximum(torque_limit[:, index], 1.0e-12))
                ),
            }
        )
    joint_summary = pd.DataFrame(joint_rows).sort_values(
        "recovery_jerk_rms_rad_s3", ascending=False
    )
    joint_summary.to_csv(output_dir / "trial_011_joint_ranking.csv", index=False)

    figure, axes = plt.subplots(6, 3, figsize=(18, 15), sharex=True, constrained_layout=True)
    for row, index in enumerate(top):
        label = joints[index].replace("_joint", "")
        axes[row, 0].plot(time, position[:, index], label="actual q")
        axes[row, 0].plot(time, target_position[:, index], label="target q", alpha=0.75)
        axes[row, 0].set_ylabel(f"{label}\nrad")
        axes[row, 1].plot(time, torque[:, index], label="actual torque")
        axes[row, 1].plot(time, torque_command[:, index], "--", label="PD command", alpha=0.6)
        axes[row, 1].axhline(np.max(torque_limit[:, index]), color="gray", linestyle=":")
        axes[row, 1].axhline(-np.max(torque_limit[:, index]), color="gray", linestyle=":")
        axes[row, 1].set_ylabel("N·m")
        axes[row, 2].plot(time, jerk[:, index], color="tab:red")
        axes[row, 2].set_yscale("symlog", linthresh=10.0)
        axes[row, 2].set_ylabel("rad/s³")
        for axis in axes[row]:
            axis.axvspan(0.0, 0.2, color="magenta", alpha=0.07)
            axis.grid(True, alpha=0.2)
        if row == 0:
            axes[row, 0].legend(fontsize=8)
            axes[row, 1].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Trial time [s]")
    axes[0, 0].set_title("Joint position: actual vs target")
    axes[0, 1].set_title("Actual vs commanded torque")
    axes[0, 2].set_title("Finite-difference jerk (symlog)")
    figure.suptitle("Trial 011: six joints with highest recovery-transient jerk", fontsize=15)
    figure.savefig(output_dir / "trial_011_top6_joint_curves.png", dpi=180)
    plt.close(figure)

    groups = {
        "Left leg": [i for i, name in enumerate(joints) if name.startswith("left_") and any(key in name for key in ("hip", "knee", "ankle"))],
        "Right leg": [i for i, name in enumerate(joints) if name.startswith("right_") and any(key in name for key in ("hip", "knee", "ankle"))],
        "Waist": [i for i, name in enumerate(joints) if name.startswith("waist_")],
        "Left arm": [i for i, name in enumerate(joints) if name.startswith("left_") and any(key in name for key in ("shoulder", "elbow", "wrist"))],
        "Right arm": [i for i, name in enumerate(joints) if name.startswith("right_") and any(key in name for key in ("shoulder", "elbow", "wrist"))],
    }
    figure, axes = plt.subplots(5, 2, figsize=(18, 17), sharex=True, constrained_layout=True)
    for row, (group, indices) in enumerate(groups.items()):
        for index in indices:
            label = joints[index].replace("_joint", "")
            axes[row, 0].plot(time, position[:, index], linewidth=0.9, label=label)
            axes[row, 1].plot(time, torque[:, index], linewidth=0.9, label=label)
        axes[row, 0].set_ylabel(f"{group}\nq [rad]")
        axes[row, 1].set_ylabel(f"{group}\nτ [N·m]")
        for axis in axes[row]:
            axis.axvspan(0.0, 0.2, color="magenta", alpha=0.07)
            axis.grid(True, alpha=0.2)
            axis.legend(ncol=2, fontsize=7)
    axes[0, 0].set_title("All joint positions by group")
    axes[0, 1].set_title("All actual actuator torques by group")
    axes[-1, 0].set_xlabel("Trial time [s]")
    axes[-1, 1].set_xlabel("Trial time [s]")
    figure.suptitle("Trial 011 complete joint position and torque curves", fontsize=15)
    figure.savefig(output_dir / "trial_011_all_joint_position_torque.png", dpi=180)
    plt.close(figure)

    peak = np.unravel_index(np.argmax(np.abs(jerk)), jerk.shape)
    final_window = time >= 15.0
    return {
        "source": str(data["path"].resolve()),
        "trials": len(list((data["path"].parent).glob("trial_*.csv"))),
        "last_trial": data["path"].stem,
        "last_trial_duration_s": float(time[-1]),
        "push_force_n": float(np.max(force)),
        "push_active_until_s": float(time[np.flatnonzero(force > 0)[-1]]),
        "peak_jerk": {
            "time_s": float(time[peak[0]]),
            "joint": joints[peak[1]],
            "value_rad_s3": float(jerk[peak]),
        },
        "recovery_0p2_to_3s": {
            "jerk_rms_rad_s3": _rms(jerk[recovery]),
            "torque_rms_nm": _rms(torque[recovery]),
            "max_abs_torque_nm": float(np.max(np.abs(torque[recovery]))),
        },
        "settled_after_3s": {
            "jerk_rms_rad_s3": _rms(jerk[settled]),
            "torque_rms_nm": _rms(torque[settled]),
            "max_abs_torque_nm": float(np.max(np.abs(torque[settled]))),
            "position_highpass_5hz_rms_max_rad": float(np.max(highpass_position)),
            "torque_highpass_5hz_rms_max_nm": float(np.max(highpass_torque)),
        },
        "final_after_15s": {
            "jerk_rms_rad_s3": _rms(jerk[final_window]),
            "torque_rms_nm": _rms(torque[final_window]),
            "max_abs_torque_nm": float(np.max(np.abs(torque[final_window]))),
            "position_mean_joint_std_rad": float(np.mean(np.std(position[final_window], axis=0))),
        },
        "max_torque_limit_ratio": float(
            np.max(np.abs(torque) / np.maximum(torque_limit, 1.0e-12))
        ),
        "torque_saturation_fraction": float(
            np.mean(np.abs(torque) >= 0.95 * np.maximum(torque_limit, 1.0e-12))
        ),
        "pd_vs_actual_torque_max_abs_diff_nm": float(
            np.max(np.abs(torque_command - torque))
        ),
        "top_recovery_jerk_joints": joint_summary.head(6).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    trial_paths = sorted((args.session_dir / "space_trials").glob("trial_*.csv"))
    if not trial_paths:
        raise FileNotFoundError(f"No trial CSV files below {args.session_dir}")
    trials = [_load_trial(path) for path in trial_paths]
    output_dir = args.session_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([_summarize_trial(trial) for trial in trials])
    summary.to_csv(output_dir / "all_trials_summary.csv", index=False)
    _plot_trial_comparison(summary, output_dir / "all_trials_comparison.png")
    result = _analyze_last_trial(trials[-1], output_dir)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Analysis outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
