#!/usr/bin/env python3
"""Aggregate the multi-profile Extreme Stand MuJoCo suite into JSON and Chinese Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROFILE_ORDER = (
    "nominal",
    "pose_recovery",
    "feet_distance_recovery",
    "recovery",
    "robust",
    "stress",
    "large_push",
)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.fmean(finite)) if finite else 0.0


def _maximum(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(max(finite)) if finite else 0.0


def _minimum(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(min(finite)) if finite else 0.0


def _nested(report: dict[str, Any], *keys: str, default: Any = math.nan) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _finite_float(value: Any, *, unavailable: float = 0.0) -> float:
    if value is None:
        return float(unavailable)
    result = float(value)
    # Reports are strict JSON (allow_nan=False).  Optional diagnostics such as
    # large-push fields are absent in nominal/recovery runs, so represent an
    # unavailable scalar as 0.0 and use the accompanying profile/type flags to
    # distinguish "not measured" from a real zero.  This also keeps historical
    # metrics files readable after new report fields are added.
    return result if math.isfinite(result) else 0.0


def _file_metadata(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"path": "", "exists": False, "size_bytes": 0, "sha256": ""}
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False, "size_bytes": 0, "sha256": ""}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_runs(results_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(results_root.glob("*/seed_*/metrics.json")):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        profile = metrics_path.parent.parent.name
        seed = int(metrics_path.parent.name.removeprefix("seed_"))
        stand = report.get("extreme_stand_recovery", {})
        health = report.get("health", {})
        important = report.get("important_metrics", {})
        tracking = report.get("task_tracking", {})
        score = report.get("score", {})
        pose_recovery = stand.get("default_pose_recovery", {})
        motion_quality = stand.get("motion_quality", {})
        jerk = motion_quality.get("joint_jerk_rad_s3", {})
        high_frequency = motion_quality.get(
            "joint_position_high_frequency_20_25hz_rms_rad", {}
        )
        actor_action = motion_quality.get("actor_action", {})
        target_position = motion_quality.get("target_joint_position_rad", {})
        joint_velocity = motion_quality.get("joint_velocity_rad_s", {})
        joint_acceleration = motion_quality.get("joint_acceleration_rad_s2", {})
        mechanical_power = motion_quality.get("mechanical_power_w", {})
        actuator_effort = motion_quality.get("actuator_effort_nm", {})
        control_chain_metrics_available = bool(
            "second_difference_per_step" in actor_action
            and "default_error" in target_position
            and "velocity_rad_s" in target_position
            and "acceleration_rad_s2" in target_position
            and "mechanical_power_w" in motion_quality
            and "pd_command_rate_nm_s" in actuator_effort
            and "relative_pd_command" in actuator_effort
            and "soft_peak_above_60pct" in actuator_effort
        )
        feet_distance = motion_quality.get("feet_planar_distance_m", {})
        feet_recovery = stand.get("foot_spacing_recovery", {})
        large_push = stand.get("large_push", {})
        push_diagnostics = large_push.get("post_push_diagnostics", {})
        push_pre = push_diagnostics.get("pre_push", {})
        push_post = push_diagnostics.get("post_push", {})
        push_late = push_diagnostics.get("late_post_push", {})
        push_ratios = push_diagnostics.get("post_over_pre", {})
        push_late_ratios = push_diagnostics.get("late_post_over_pre", {})
        push_flags = push_diagnostics.get("flags", {})
        push_settling = push_diagnostics.get("actor_action_rate_settling", {})
        sim_time_s = _finite_float(report.get("sim_time", 0.0))
        settling_time_value = push_settling.get("settling_time_after_push_end_s")
        settling_time_available = settling_time_value is not None
        command = [
            _finite_float(tracking.get("mean_command_lin_vel_x", math.nan)),
            _finite_float(tracking.get("mean_command_lin_vel_y", math.nan)),
            _finite_float(tracking.get("mean_command_yaw_rate", math.nan)),
        ]
        runs.append(
            {
                "profile": profile,
                "seed": seed,
                "metrics_path": str(metrics_path.resolve()),
                "healthy": bool(health.get("healthy", False)),
                "fallen": bool(health.get("fallen", True)),
                "fall_time_s": health.get("fall_time"),
                "sim_time_s": sim_time_s,
                "min_root_height_m": _finite_float(health.get("min_root_height", math.nan)),
                "max_abs_roll_rad": _finite_float(health.get("max_abs_roll", math.nan)),
                "max_abs_pitch_rad": _finite_float(health.get("max_abs_pitch", math.nan)),
                "lin_vel_xy_mae_m_s": _finite_float(tracking.get("lin_vel_xy_mae", math.nan)),
                "yaw_rate_mae_rad_s": _finite_float(tracking.get("yaw_rate_mae", math.nan)),
                "torso_roll_error_rad": _finite_float(important.get("torso_roll_error_rad", math.nan)),
                "torso_pitch_error_rad": _finite_float(important.get("torso_pitch_error_rad", math.nan)),
                "torso_height_error_m": _finite_float(important.get("torso_height_error_m", math.nan)),
                "torso_trace_path_m": _finite_float(_nested(report, "torso_trace", "path_length_m")),
                "total_score": _finite_float(score.get("total_score", math.nan)),
                "command_mean": command,
                "zero_command": all(math.isfinite(value) and abs(value) <= 1.0e-6 for value in command),
                "action_override": stand.get("action_override"),
                "wrench_event_count": int(_nested(stand, "wrench", "event_count", default=0)),
                "joint_limit_clip_count": int(
                    _nested(stand, "initial_noise", "joint_limit_clip_count", default=0)
                ),
                "pose_recovered": bool(pose_recovery.get("pose_recovered", False)),
                "initial_joint_mae_rad": _finite_float(
                    pose_recovery.get("initial_joint_mae_rad", math.nan)
                ),
                "final_joint_mae_rad": _finite_float(
                    pose_recovery.get("final_joint_mae_rad", math.nan)
                ),
                "final_joint_max_abs_error_rad": _finite_float(
                    pose_recovery.get("final_joint_max_abs_error_rad", math.nan)
                ),
                "joint_recovery_ratio": _finite_float(
                    pose_recovery.get("recovery_ratio", math.nan)
                ),
                "joint_recovery_time_s": pose_recovery.get("recovery_time_s"),
                "final_error_by_joint_rad": dict(
                    pose_recovery.get("final_mean_abs_error_by_joint_rad", {})
                ),
                "steady_start_s": _finite_float(
                    motion_quality.get("steady_start_s", math.nan)
                ),
                "steady_sample_count": int(
                    motion_quality.get("steady_sample_count", 0)
                ),
                "control_sample_rate_hz": _finite_float(
                    motion_quality.get("control_sample_rate_hz", math.nan)
                ),
                "joint_jerk_rms_rad_s3": _finite_float(jerk.get("rms", math.nan)),
                "joint_jerk_p95_abs_rad_s3": _finite_float(
                    jerk.get("p95_abs", math.nan)
                ),
                "joint_jerk_max_abs_rad_s3": _finite_float(
                    jerk.get("max_abs", math.nan)
                ),
                "joint_jerk_weighted_reward_equivalent": _finite_float(
                    jerk.get("training_weighted_mean_reward_equivalent", math.nan)
                ),
                "joint_jerk_per_joint_rms_rad_s3": dict(
                    jerk.get("per_joint_rms", {})
                ),
                "joint_position_hf_mean_rms_rad": _finite_float(
                    high_frequency.get("mean_across_joints", math.nan)
                ),
                "joint_position_hf_max_rms_rad": _finite_float(
                    high_frequency.get("max_across_joints", math.nan)
                ),
                "joint_position_hf_per_joint_rms_rad": dict(
                    high_frequency.get("per_joint", {})
                ),
                "control_chain_metrics_available": control_chain_metrics_available,
                "actor_action_rms": _finite_float(
                    _nested(actor_action, "value", "rms", default=0.0)
                ),
                "actor_action_rate_rms_per_s": _finite_float(
                    _nested(actor_action, "delta_rate_per_s", "rms", default=0.0)
                ),
                "actor_action_second_difference_rms": _finite_float(
                    _nested(
                        actor_action,
                        "second_difference_per_step",
                        "rms",
                        default=0.0,
                    )
                ),
                "actor_action_hf_mean_rms": _finite_float(
                    _nested(
                        actor_action,
                        "high_frequency_8_25hz_rms",
                        "mean_across_joints",
                        default=0.0,
                    )
                ),
                "target_default_error_rms_rad": _finite_float(
                    _nested(target_position, "default_error", "rms", default=0.0)
                ),
                "target_velocity_rms_rad_s": _finite_float(
                    _nested(target_position, "velocity_rad_s", "rms", default=0.0)
                ),
                "target_acceleration_rms_rad_s2": _finite_float(
                    _nested(
                        target_position,
                        "acceleration_rad_s2",
                        "rms",
                        default=0.0,
                    )
                ),
                "target_position_hf_mean_rms_rad": _finite_float(
                    _nested(
                        target_position,
                        "high_frequency_8_25hz_rms",
                        "mean_across_joints",
                        default=0.0,
                    )
                ),
                "joint_velocity_rms_rad_s": _finite_float(
                    joint_velocity.get("rms", 0.0)
                ),
                "joint_acceleration_rms_rad_s2": _finite_float(
                    joint_acceleration.get("rms", 0.0)
                ),
                "mechanical_power_rms_w": _finite_float(
                    mechanical_power.get("rms", 0.0)
                ),
                "mechanical_power_mean_square_w2": _finite_float(
                    mechanical_power.get("mean_square", 0.0)
                ),
                "pd_torque_rms_nm": _finite_float(
                    _nested(actuator_effort, "pd_command", "rms", default=0.0)
                ),
                "pd_torque_rate_rms_nm_s": _finite_float(
                    _nested(
                        actuator_effort,
                        "pd_command_rate_nm_s",
                        "rms",
                        default=0.0,
                    )
                ),
                "relative_pd_torque_rms": _finite_float(
                    _nested(
                        actuator_effort,
                        "relative_pd_command",
                        "rms",
                        default=0.0,
                    )
                ),
                "soft_peak_pd_torque_rms": _finite_float(
                    _nested(
                        actuator_effort,
                        "soft_peak_above_60pct",
                        "rms",
                        default=0.0,
                    )
                ),
                "pd_torque_hf_mean_rms_nm": _finite_float(
                    _nested(
                        actuator_effort,
                        "pd_command_high_frequency_8_25hz_rms",
                        "mean_across_joints",
                        default=0.0,
                    )
                ),
                "pd_torque_saturation_fraction": _finite_float(
                    actuator_effort.get("command_saturation_fraction", 0.0)
                ),
                "default_feet_distance_m": _finite_float(
                    feet_distance.get("default", math.nan)
                ),
                "mean_feet_distance_m": _finite_float(
                    feet_distance.get("mean", math.nan)
                ),
                "feet_distance_error_mean_abs_m": _finite_float(
                    feet_distance.get("error_mean_abs", math.nan)
                ),
                "feet_distance_error_rms_m": _finite_float(
                    feet_distance.get("error_rms", math.nan)
                ),
                "feet_distance_error_p95_abs_m": _finite_float(
                    feet_distance.get("error_p95_abs", math.nan)
                ),
                "feet_distance_error_max_abs_m": _finite_float(
                    feet_distance.get("error_max_abs", math.nan)
                ),
                "feet_distance_gaussian_mean": _finite_float(
                    feet_distance.get("gaussian_mean", math.nan)
                ),
                "feet_distance_within_1cm_fraction": _finite_float(
                    feet_distance.get("within_1cm_fraction", math.nan)
                ),
                "feet_distance_within_2cm_fraction": _finite_float(
                    feet_distance.get("within_2cm_fraction", math.nan)
                ),
                "feet_recovery_tested": bool(feet_recovery.get("tested", False)),
                "feet_perturbation_applied": bool(
                    feet_recovery.get("perturbation_applied", False)
                ),
                "feet_distance_recovered": bool(
                    feet_recovery.get("distance_recovered", False)
                ),
                "feet_initial_distance_m": _finite_float(
                    feet_recovery.get("actual_initial_distance_m", 0.0)
                ),
                "feet_initial_error_m": _finite_float(
                    feet_recovery.get("initial_error_m", 0.0)
                ),
                "feet_final_error_mean_abs_m": _finite_float(
                    feet_recovery.get("final_error_mean_abs_m", 0.0)
                ),
                "feet_final_error_max_abs_m": _finite_float(
                    feet_recovery.get("final_error_max_abs_m", 0.0)
                ),
                "feet_recovery_time_s": feet_recovery.get("recovery_time_s"),
                "motion_trace_csv_path": str(
                    motion_quality.get("trace_csv_path", "")
                ),
                "large_push_event_count": int(large_push.get("event_count", 0)),
                "large_push_force_n": _finite_float(
                    large_push.get("force_n", math.nan)
                ),
                "large_push_impulse_n_s": _finite_float(
                    large_push.get("impulse_n_s", math.nan)
                ),
                "large_push_diagnosis": str(
                    push_diagnostics.get("diagnosis", "")
                ),
                "large_push_persistent_joint_vibration": bool(
                    push_flags.get("persistent_joint_vibration", False)
                ),
                "large_push_transient_joint_vibration": bool(
                    push_flags.get("transient_joint_vibration", False)
                ),
                "large_push_policy_action_high_frequency": bool(
                    push_flags.get("policy_action_high_frequency", False)
                ),
                "large_push_transient_policy_action_high_frequency": bool(
                    push_flags.get(
                        "transient_policy_action_high_frequency",
                        False,
                    )
                ),
                "large_push_pd_torque_saturation": bool(
                    push_flags.get("pd_torque_saturation", False)
                ),
                "large_push_position_hf_ratio": _finite_float(
                    push_ratios.get(
                        "joint_position_hf_8_25hz_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_action_hf_ratio": _finite_float(
                    push_ratios.get(
                        "actor_action_hf_8_25hz_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_action_rate_ratio": _finite_float(
                    push_ratios.get(
                        "actor_action_delta_rate_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_torque_hf_ratio": _finite_float(
                    push_ratios.get(
                        "pd_torque_hf_8_25hz_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_pre_position_hf_rms_rad": _finite_float(
                    push_pre.get(
                        "joint_position_hf_8_25hz_rms_mean_rad",
                        math.nan,
                    )
                ),
                "large_push_post_position_hf_rms_rad": _finite_float(
                    push_post.get(
                        "joint_position_hf_8_25hz_rms_mean_rad",
                        math.nan,
                    )
                ),
                "large_push_pre_action_hf_rms": _finite_float(
                    push_pre.get("actor_action_hf_8_25hz_rms_mean", math.nan)
                ),
                "large_push_post_action_hf_rms": _finite_float(
                    push_post.get("actor_action_hf_8_25hz_rms_mean", math.nan)
                ),
                "large_push_post_torque_rms_nm": _finite_float(
                    push_post.get("pd_torque_rms_nm", math.nan)
                ),
                "large_push_post_torque_max_abs_nm": _finite_float(
                    push_post.get("pd_torque_max_abs_nm", math.nan)
                ),
                "large_push_post_torque_saturation_fraction": _finite_float(
                    push_post.get("pd_torque_saturation_fraction", math.nan)
                ),
                "large_push_late_position_hf_ratio": _finite_float(
                    push_late_ratios.get(
                        "joint_position_hf_8_25hz_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_late_action_hf_ratio": _finite_float(
                    push_late_ratios.get(
                        "actor_action_hf_8_25hz_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_late_action_rate_ratio": _finite_float(
                    push_late_ratios.get(
                        "actor_action_delta_rate_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_late_torque_hf_ratio": _finite_float(
                    push_late_ratios.get(
                        "pd_torque_hf_8_25hz_rms_ratio",
                        math.nan,
                    )
                ),
                "large_push_late_position_hf_rms_rad": _finite_float(
                    push_late.get(
                        "joint_position_hf_8_25hz_rms_mean_rad",
                        math.nan,
                    )
                ),
                "large_push_late_action_hf_rms": _finite_float(
                    push_late.get("actor_action_hf_8_25hz_rms_mean", math.nan)
                ),
                "large_push_action_rate_settled": settling_time_available,
                # A missing settling time means that the action rate never met
                # the threshold for the required hold interval.  Use the full
                # rollout duration as a finite failure sentinel; mapping it to
                # zero would incorrectly pass the <=6 s comparison gate.
                "large_push_action_rate_settling_time_s": _finite_float(
                    settling_time_value,
                    unavailable=sim_time_s,
                ),
            }
        )
    if not runs:
        raise SystemExit(f"No metrics.json found below {results_root}")
    return runs


def summarize_profile(profile: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [run for run in runs if run["profile"] == profile]
    healthy_count = sum(int(run["healthy"]) for run in selected)
    return {
        "profile": profile,
        "run_count": len(selected),
        "healthy_count": healthy_count,
        "healthy_rate": healthy_count / len(selected) if selected else 0.0,
        "fall_count": sum(int(run["fallen"]) for run in selected),
        "all_zero_command": all(run["zero_command"] for run in selected),
        "all_actor_outputs_unmodified": all(run["action_override"] is False for run in selected),
        "mean_lin_vel_xy_mae_m_s": _mean([run["lin_vel_xy_mae_m_s"] for run in selected]),
        "mean_yaw_rate_mae_rad_s": _mean([run["yaw_rate_mae_rad_s"] for run in selected]),
        "mean_torso_roll_error_rad": _mean([run["torso_roll_error_rad"] for run in selected]),
        "mean_torso_pitch_error_rad": _mean([run["torso_pitch_error_rad"] for run in selected]),
        "mean_torso_height_error_m": _mean([run["torso_height_error_m"] for run in selected]),
        "mean_torso_trace_path_m": _mean([run["torso_trace_path_m"] for run in selected]),
        "mean_total_score": _mean([run["total_score"] for run in selected]),
        "minimum_root_height_m": _minimum([run["min_root_height_m"] for run in selected]),
        "maximum_abs_roll_rad": _maximum([run["max_abs_roll_rad"] for run in selected]),
        "maximum_abs_pitch_rad": _maximum([run["max_abs_pitch_rad"] for run in selected]),
        "total_wrench_events": sum(run["wrench_event_count"] for run in selected),
        "total_large_push_events": sum(
            run["large_push_event_count"] for run in selected
        ),
        "total_initial_joint_limit_clips": sum(run["joint_limit_clip_count"] for run in selected),
        "pose_recovered_count": sum(int(run["pose_recovered"]) for run in selected),
        "pose_recovered_rate": (
            sum(int(run["pose_recovered"]) for run in selected) / len(selected)
            if selected
            else 0.0
        ),
        "mean_initial_joint_mae_rad": _mean([run["initial_joint_mae_rad"] for run in selected]),
        "mean_final_joint_mae_rad": _mean([run["final_joint_mae_rad"] for run in selected]),
        "mean_final_joint_max_abs_error_rad": _mean(
            [run["final_joint_max_abs_error_rad"] for run in selected]
        ),
        "mean_joint_recovery_ratio": _mean([run["joint_recovery_ratio"] for run in selected]),
        "mean_steady_sample_count": _mean(
            [float(run["steady_sample_count"]) for run in selected]
        ),
        "mean_joint_jerk_rms_rad_s3": _mean(
            [run["joint_jerk_rms_rad_s3"] for run in selected]
        ),
        "mean_joint_jerk_p95_abs_rad_s3": _mean(
            [run["joint_jerk_p95_abs_rad_s3"] for run in selected]
        ),
        "maximum_joint_jerk_abs_rad_s3": _maximum(
            [run["joint_jerk_max_abs_rad_s3"] for run in selected]
        ),
        "mean_joint_jerk_weighted_reward_equivalent": _mean(
            [run["joint_jerk_weighted_reward_equivalent"] for run in selected]
        ),
        "mean_joint_position_hf_rms_rad": _mean(
            [run["joint_position_hf_mean_rms_rad"] for run in selected]
        ),
        "maximum_joint_position_hf_rms_rad": _maximum(
            [run["joint_position_hf_max_rms_rad"] for run in selected]
        ),
        "control_chain_metrics_available_count": sum(
            int(run["control_chain_metrics_available"]) for run in selected
        ),
        "mean_actor_action_rms": _mean(
            [run["actor_action_rms"] for run in selected]
        ),
        "mean_actor_action_rate_rms_per_s": _mean(
            [run["actor_action_rate_rms_per_s"] for run in selected]
        ),
        "mean_actor_action_second_difference_rms": _mean(
            [run["actor_action_second_difference_rms"] for run in selected]
        ),
        "mean_actor_action_hf_rms": _mean(
            [run["actor_action_hf_mean_rms"] for run in selected]
        ),
        "mean_target_default_error_rms_rad": _mean(
            [run["target_default_error_rms_rad"] for run in selected]
        ),
        "mean_target_velocity_rms_rad_s": _mean(
            [run["target_velocity_rms_rad_s"] for run in selected]
        ),
        "mean_target_acceleration_rms_rad_s2": _mean(
            [run["target_acceleration_rms_rad_s2"] for run in selected]
        ),
        "mean_target_position_hf_rms_rad": _mean(
            [run["target_position_hf_mean_rms_rad"] for run in selected]
        ),
        "mean_joint_velocity_rms_rad_s": _mean(
            [run["joint_velocity_rms_rad_s"] for run in selected]
        ),
        "mean_joint_acceleration_rms_rad_s2": _mean(
            [run["joint_acceleration_rms_rad_s2"] for run in selected]
        ),
        "mean_mechanical_power_rms_w": _mean(
            [run["mechanical_power_rms_w"] for run in selected]
        ),
        "mean_mechanical_power_mean_square_w2": _mean(
            [run["mechanical_power_mean_square_w2"] for run in selected]
        ),
        "mean_pd_torque_rms_nm": _mean(
            [run["pd_torque_rms_nm"] for run in selected]
        ),
        "mean_pd_torque_rate_rms_nm_s": _mean(
            [run["pd_torque_rate_rms_nm_s"] for run in selected]
        ),
        "mean_relative_pd_torque_rms": _mean(
            [run["relative_pd_torque_rms"] for run in selected]
        ),
        "mean_soft_peak_pd_torque_rms": _mean(
            [run["soft_peak_pd_torque_rms"] for run in selected]
        ),
        "mean_pd_torque_hf_rms_nm": _mean(
            [run["pd_torque_hf_mean_rms_nm"] for run in selected]
        ),
        "mean_pd_torque_saturation_fraction": _mean(
            [run["pd_torque_saturation_fraction"] for run in selected]
        ),
        "mean_default_feet_distance_m": _mean(
            [run["default_feet_distance_m"] for run in selected]
        ),
        "mean_feet_distance_m": _mean(
            [run["mean_feet_distance_m"] for run in selected]
        ),
        "mean_feet_distance_error_mean_abs_m": _mean(
            [run["feet_distance_error_mean_abs_m"] for run in selected]
        ),
        "mean_feet_distance_error_rms_m": _mean(
            [run["feet_distance_error_rms_m"] for run in selected]
        ),
        "mean_feet_distance_error_p95_abs_m": _mean(
            [run["feet_distance_error_p95_abs_m"] for run in selected]
        ),
        "maximum_feet_distance_error_abs_m": _maximum(
            [run["feet_distance_error_max_abs_m"] for run in selected]
        ),
        "mean_feet_distance_gaussian": _mean(
            [run["feet_distance_gaussian_mean"] for run in selected]
        ),
        "mean_feet_distance_within_1cm_fraction": _mean(
            [run["feet_distance_within_1cm_fraction"] for run in selected]
        ),
        "mean_feet_distance_within_2cm_fraction": _mean(
            [run["feet_distance_within_2cm_fraction"] for run in selected]
        ),
        "feet_recovery_tested_count": sum(
            int(run["feet_recovery_tested"]) for run in selected
        ),
        "feet_perturbation_applied_count": sum(
            int(run["feet_perturbation_applied"]) for run in selected
        ),
        "feet_distance_recovered_count": sum(
            int(run["feet_distance_recovered"]) for run in selected
        ),
        "feet_distance_recovered_rate": (
            sum(int(run["feet_distance_recovered"]) for run in selected) / len(selected)
            if selected
            else 0.0
        ),
        "mean_feet_initial_distance_m": _mean(
            [run["feet_initial_distance_m"] for run in selected]
        ),
        "mean_feet_initial_error_abs_m": _mean(
            [abs(run["feet_initial_error_m"]) for run in selected]
        ),
        "mean_feet_final_error_abs_m": _mean(
            [run["feet_final_error_mean_abs_m"] for run in selected]
        ),
        "maximum_feet_final_error_abs_m": _maximum(
            [run["feet_final_error_max_abs_m"] for run in selected]
        ),
        "persistent_joint_vibration_count": sum(
            int(run["large_push_persistent_joint_vibration"]) for run in selected
        ),
        "transient_joint_vibration_count": sum(
            int(run["large_push_transient_joint_vibration"]) for run in selected
        ),
        "policy_action_high_frequency_count": sum(
            int(run["large_push_policy_action_high_frequency"]) for run in selected
        ),
        "transient_policy_action_high_frequency_count": sum(
            int(run["large_push_transient_policy_action_high_frequency"])
            for run in selected
        ),
        "pd_torque_saturation_count": sum(
            int(run["large_push_pd_torque_saturation"]) for run in selected
        ),
        "mean_large_push_position_hf_ratio": _mean(
            [run["large_push_position_hf_ratio"] for run in selected]
        ),
        "mean_large_push_action_hf_ratio": _mean(
            [run["large_push_action_hf_ratio"] for run in selected]
        ),
        "mean_large_push_action_rate_ratio": _mean(
            [run["large_push_action_rate_ratio"] for run in selected]
        ),
        "mean_large_push_torque_hf_ratio": _mean(
            [run["large_push_torque_hf_ratio"] for run in selected]
        ),
        "mean_large_push_post_torque_rms_nm": _mean(
            [run["large_push_post_torque_rms_nm"] for run in selected]
        ),
        "maximum_large_push_post_torque_abs_nm": _maximum(
            [run["large_push_post_torque_max_abs_nm"] for run in selected]
        ),
        "mean_large_push_post_torque_saturation_fraction": _mean(
            [run["large_push_post_torque_saturation_fraction"] for run in selected]
        ),
        "mean_large_push_late_position_hf_ratio": _mean(
            [run["large_push_late_position_hf_ratio"] for run in selected]
        ),
        "mean_large_push_late_action_hf_ratio": _mean(
            [run["large_push_late_action_hf_ratio"] for run in selected]
        ),
        "mean_large_push_late_action_rate_ratio": _mean(
            [run["large_push_late_action_rate_ratio"] for run in selected]
        ),
        "mean_large_push_late_torque_hf_ratio": _mean(
            [run["large_push_late_torque_hf_ratio"] for run in selected]
        ),
        "mean_large_push_action_rate_settling_time_s": _mean(
            [run["large_push_action_rate_settling_time_s"] for run in selected]
        ),
        "large_push_action_rate_settled_count": sum(
            int(run["large_push_action_rate_settled"]) for run in selected
        ),
    }


def build_summary(
    results_root: Path,
    *,
    model_label: str = "",
    checkpoint: str | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    runs = load_runs(results_root)
    profiles = [summarize_profile(profile, runs) for profile in PROFILE_ORDER if any(run["profile"] == profile for run in runs)]
    profile_map = {profile["profile"]: profile for profile in profiles}
    profile_names = set(profile_map)
    pose_only = profile_names == {"pose_recovery"}
    feet_only = profile_names == {"feet_distance_recovery"}
    push_only = profile_names == {"large_push"}
    mandatory_profiles = ("nominal", "recovery", "robust")
    required_present = (
        pose_only
        or feet_only
        or push_only
        or all(name in profile_map for name in mandatory_profiles)
    )
    if pose_only:
        acceptance_pass = bool(
            profile_map["pose_recovery"]["healthy_rate"] == 1.0
            and profile_map["pose_recovery"]["pose_recovered_rate"] >= 0.8
            and profile_map["pose_recovery"]["all_zero_command"]
            and profile_map["pose_recovery"]["all_actor_outputs_unmodified"]
        )
    elif feet_only:
        acceptance_pass = bool(
            profile_map["feet_distance_recovery"]["healthy_rate"] == 1.0
            and profile_map["feet_distance_recovery"][
                "feet_perturbation_applied_count"
            ]
            == profile_map["feet_distance_recovery"]["run_count"]
            and profile_map["feet_distance_recovery"][
                "feet_distance_recovered_rate"
            ]
            >= 0.8
            and profile_map["feet_distance_recovery"]["all_zero_command"]
            and profile_map["feet_distance_recovery"][
                "all_actor_outputs_unmodified"
            ]
        )
    elif push_only:
        acceptance_pass = bool(
            profile_map["large_push"]["healthy_rate"] == 1.0
            and profile_map["large_push"]["total_large_push_events"]
            == profile_map["large_push"]["run_count"]
            and profile_map["large_push"]["all_zero_command"]
            and profile_map["large_push"]["all_actor_outputs_unmodified"]
        )
    else:
        acceptance_pass = bool(
            required_present
            and profile_map["nominal"]["healthy_rate"] == 1.0
            and profile_map["recovery"]["healthy_rate"] == 1.0
            and profile_map["robust"]["healthy_rate"] >= 2.0 / 3.0
            and all(profile_map[name]["all_zero_command"] for name in mandatory_profiles)
            and all(profile_map[name]["all_actor_outputs_unmodified"] for name in mandatory_profiles)
            and (
                "pose_recovery" not in profile_map
                or (
                    profile_map["pose_recovery"]["healthy_rate"] == 1.0
                    and profile_map["pose_recovery"]["pose_recovered_rate"] >= 2.0 / 3.0
                )
            )
            and (
                "feet_distance_recovery" not in profile_map
                or (
                    profile_map["feet_distance_recovery"]["healthy_rate"] == 1.0
                    and profile_map["feet_distance_recovery"][
                        "feet_perturbation_applied_count"
                    ]
                    == profile_map["feet_distance_recovery"]["run_count"]
                    and profile_map["feet_distance_recovery"][
                        "feet_distance_recovered_rate"
                    ]
                    >= 2.0 / 3.0
                )
            )
        )
    return {
        "schema_version": 1,
        "results_root": str(results_root.resolve()),
        "model": {
            "label": model_label,
            "checkpoint": _file_metadata(checkpoint),
            "policy": _file_metadata(policy),
        },
        "run_count": len(runs),
        "profiles": profiles,
        "runs": runs,
        "acceptance": {
            "pass": acceptance_pass,
            "required_profiles_present": required_present,
            "criteria": {
                "nominal_healthy_rate": 1.0,
                "recovery_healthy_rate": 1.0,
                "robust_min_healthy_rate": 2.0 / 3.0,
                "zero_command_required": True,
                "action_override_must_be_false": True,
                "stress_profile_is_informational": True,
                "pose_recovery_min_rate": 0.8 if pose_only else 2.0 / 3.0,
                "feet_distance_recovery_min_rate": (
                    0.8 if feet_only else 2.0 / 3.0
                ),
            },
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    profile_names = {profile["profile"] for profile in summary["profiles"]}
    pose_only = profile_names == {"pose_recovery"}
    feet_only = profile_names == {"feet_distance_recovery"}
    push_only = profile_names == {"large_push"}
    model = summary.get("model", {})
    checkpoint = model.get("checkpoint", {})
    policy = model.get("policy", {})
    lines = [
        "# G1 Extreme Stand Recovery MuJoCo 全面测试报告",
        "",
        f"- 模型：`{model.get('label', '') or '未指定'}`",
        (
            f"- Checkpoint：`{checkpoint.get('path', '')}`"
            f"（SHA-256 `{checkpoint.get('sha256', '')}`）"
        ),
        (
            f"- 推理模型：`{policy.get('path', '')}`"
            f"（SHA-256 `{policy.get('sha256', '')}`）"
        ),
        f"- 测试目录：`{summary['results_root']}`",
        f"- 总运行数：{summary['run_count']}",
        f"- 基础验收：{'通过' if summary['acceptance']['pass'] else '未通过'}",
        (
            "- 本报告只验收随机关节初始姿态能否恢复到严格默认全身姿态。"
            if pose_only
            else (
                "- 本报告只验收随机双脚初始间距能否恢复到资产默认距离。"
                if feet_only
                else (
                    "- 本报告对固定躯干大推力后的长期抖动进行诊断；通过仅表示测试链路完整且未摔倒，不代表没有抖动。"
                    if push_only
                    else "- `stress` 和 `large_push` 是诊断场景，不计入基础验收。"
                )
            )
        ),
        "",
        "## 分场景汇总",
        "",
        "| 场景 | 健康运行 | 健康率 | 水平速度 MAE m/s | yaw-rate MAE rad/s | torso roll/pitch rad | 最低 root m | 最大 roll/pitch rad | torso 路径 m | 总分 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for profile in summary["profiles"]:
        lines.append(
            "| {profile} | {healthy_count}/{run_count} | {healthy_rate:.1%} | "
            "{mean_lin_vel_xy_mae_m_s:.4f} | {mean_yaw_rate_mae_rad_s:.4f} | "
            "{mean_torso_roll_error_rad:.4f}/{mean_torso_pitch_error_rad:.4f} | "
            "{minimum_root_height_m:.4f} | {maximum_abs_roll_rad:.4f}/{maximum_abs_pitch_rad:.4f} | "
            "{mean_torso_trace_path_m:.4f} | {mean_total_score:.2f} |".format(**profile)
        )
    pose_profiles = [
        profile for profile in summary["profiles"] if profile["profile"] == "pose_recovery"
    ]
    if pose_profiles:
        pose = pose_profiles[0]
        lines.extend(
            [
                "",
                "## 随机姿态恢复汇总",
                "",
                "| 恢复成功 | 初始关节 MAE rad | 最终关节 MAE rad | 最终最大关节误差 rad | 平均误差下降比例 |",
                "| ---: | ---: | ---: | ---: | ---: |",
                f"| {pose['pose_recovered_count']}/{pose['run_count']} | "
                f"{pose['mean_initial_joint_mae_rad']:.4f} | {pose['mean_final_joint_mae_rad']:.4f} | "
                f"{pose['mean_final_joint_max_abs_error_rad']:.4f} | {pose['mean_joint_recovery_ratio']:.1%} |",
            ]
        )
    feet_profiles = [
        profile
        for profile in summary["profiles"]
        if profile["profile"] == "feet_distance_recovery"
    ]
    if feet_profiles:
        feet_profile = feet_profiles[0]
        lines.extend(
            [
                "",
                "## 随机双脚间距恢复汇总",
                "",
                "| 有效间距扰动 | 恢复成功 | 默认/初始足距 m | 初始偏差 cm | 最终 MAE/最大误差 cm |",
                "| ---: | ---: | --- | ---: | --- |",
                f"| {feet_profile['feet_perturbation_applied_count']}/{feet_profile['run_count']} | "
                f"{feet_profile['feet_distance_recovered_count']}/{feet_profile['run_count']} | "
                f"{feet_profile['mean_default_feet_distance_m']:.4f}/"
                f"{feet_profile['mean_feet_initial_distance_m']:.4f} | "
                f"{100.0 * feet_profile['mean_feet_initial_error_abs_m']:.2f} | "
                f"{100.0 * feet_profile['mean_feet_final_error_abs_m']:.2f}/"
                f"{100.0 * feet_profile['maximum_feet_final_error_abs_m']:.2f} |",
            ]
        )
    push_profiles = [
        profile
        for profile in summary["profiles"]
        if profile["profile"] == "large_push"
    ]
    if push_profiles:
        push = push_profiles[0]
        lines.extend(
            [
                "",
                "## 躯干大推力后持续抖动诊断",
                "",
                "这里分别比较推力前 2 秒、推力结束并等待 `POST_PUSH_SETTLE_S` 后的恢复窗口，"
                "以及测试最终 5 秒的长期稳态窗口。位置、策略 action 和 PD 力矩均统计 8–25 Hz；"
                "力矩饱和按控制上限的 98% 判定。",
                "",
                "| 大推力事件 | 恢复段振动/Action 高频 | 最终稳态振动/Action 高频 | PD 力矩饱和 | 恢复段位置/Action/Action-rate/力矩比 | 最终稳态对应比 | Action-rate 稳定时间 s |",
                "| ---: | --- | --- | ---: | --- | --- | ---: |",
                f"| {push['total_large_push_events']} | "
                f"{push['transient_joint_vibration_count']}/{push['run_count']} / "
                f"{push['transient_policy_action_high_frequency_count']}/{push['run_count']} | "
                f"{push['persistent_joint_vibration_count']}/{push['run_count']} / "
                f"{push['policy_action_high_frequency_count']}/{push['run_count']} | "
                f"{push['pd_torque_saturation_count']}/{push['run_count']} | "
                f"{push['mean_large_push_position_hf_ratio']:.2f}/"
                f"{push['mean_large_push_action_hf_ratio']:.2f}/"
                f"{push['mean_large_push_action_rate_ratio']:.2f}/"
                f"{push['mean_large_push_torque_hf_ratio']:.2f} | "
                f"{push['mean_large_push_late_position_hf_ratio']:.2f}/"
                f"{push['mean_large_push_late_action_hf_ratio']:.2f}/"
                f"{push['mean_large_push_late_action_rate_ratio']:.2f}/"
                f"{push['mean_large_push_late_torque_hf_ratio']:.2f} | "
                f"{push['mean_large_push_action_rate_settling_time_s']:.2f} "
                f"({push['large_push_action_rate_settled_count']}/{push['run_count']}收敛) |",
                "",
                "### 单次诊断",
                "",
                "| seed | 诊断 | 恢复段振动/Action 高频 | 最终稳态振动/Action 高频 | 力矩饱和 | 位置 HF 前→恢复→最终 rad | Action HF 前→恢复→最终 | Action-rate 稳定 s |",
                "| ---: | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for run in summary["runs"]:
            if run["profile"] != "large_push":
                continue
            lines.append(
                f"| {run['seed']} | `{run['large_push_diagnosis']}` | "
                f"{'是' if run['large_push_transient_joint_vibration'] else '否'}/"
                f"{'是' if run['large_push_transient_policy_action_high_frequency'] else '否'} | "
                f"{'是' if run['large_push_persistent_joint_vibration'] else '否'}/"
                f"{'是' if run['large_push_policy_action_high_frequency'] else '否'} | "
                f"{'是' if run['large_push_pd_torque_saturation'] else '否'} | "
                f"{run['large_push_pre_position_hf_rms_rad']:.6f}→"
                f"{run['large_push_post_position_hf_rms_rad']:.6f}→"
                f"{run['large_push_late_position_hf_rms_rad']:.6f} | "
                f"{run['large_push_pre_action_hf_rms']:.6f}→"
                f"{run['large_push_post_action_hf_rms']:.6f}→"
                f"{run['large_push_late_action_hf_rms']:.6f} | "
                f"{run['large_push_action_rate_settling_time_s']:.2f}"
                f"{'（未收敛）' if not run['large_push_action_rate_settled'] else ''} |"
            )
    lines.extend(
        [
            "",
            "## 长期 jerk 与双脚距离",
            "",
            "以下指标按 50 Hz 控制步计算，并剔除 `STEADY_START_S` 指定的恢复段（长期测试默认前 10 秒）。"
            "`jerk RMS` 来自关节速度二阶差分；`20–25 Hz` 是关节位置高频频带 RMS，"
            "用于发现长期两帧振荡。双脚距离以资产默认姿态的左右足 body 平面距离为目标，"
            "不是越近或越远越好。",
            "",
            "| 场景 | 稳态样本/次 | jerk RMS rad/s³ | jerk P95 rad/s³ | jerk 加权等价奖励 | 20–25 Hz 位置 RMS 平均/最大 rad | 默认/实际足距 m | 足距 MAE/RMS/P95/最大 cm | ±1 cm | 足距高斯均值 |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for profile in summary["profiles"]:
        lines.append(
            "| {profile} | {mean_steady_sample_count:.0f} | "
            "{mean_joint_jerk_rms_rad_s3:.2f} | {mean_joint_jerk_p95_abs_rad_s3:.2f} | "
            "{mean_joint_jerk_weighted_reward_equivalent:.4f} | "
            "{mean_joint_position_hf_rms_rad:.6f}/{maximum_joint_position_hf_rms_rad:.6f} | "
            "{mean_default_feet_distance_m:.4f}/{mean_feet_distance_m:.4f} | "
            "{mae:.3f}/{rms:.3f}/{p95:.3f}/{maximum:.3f} | "
            "{within:.1%} | {mean_feet_distance_gaussian:.4f} |".format(
                **profile,
                mae=100.0 * profile["mean_feet_distance_error_mean_abs_m"],
                rms=100.0 * profile["mean_feet_distance_error_rms_m"],
                p95=100.0 * profile["mean_feet_distance_error_p95_abs_m"],
                maximum=100.0 * profile["maximum_feet_distance_error_abs_m"],
                within=profile["mean_feet_distance_within_1cm_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "### 控制链平滑指标",
            "",
            "下表使用同一个稳态窗口。`action Δ²` 是与训练一致的每控制步二阶差分；目标角速度/加速度由实际送入 PD 的目标有限差分得到；机械功率使用 MuJoCo 实际执行器力矩乘关节速度。",
            "",
            "| 场景 | action RMS/变化率/Δ² | action/目标角 8–25 Hz RMS | 目标默认误差/速度/加速度 | 关节速度/加速度 RMS | PD 力矩/变化率 RMS | 相对力矩/60%软峰值 | 力矩高频/饱和率 | 机械功率 RMS |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for profile in summary["profiles"]:
        lines.append(
            "| {profile} | {mean_actor_action_rms:.4f}/{mean_actor_action_rate_rms_per_s:.3f}/{mean_actor_action_second_difference_rms:.5f} | "
            "{mean_actor_action_hf_rms:.6f}/{mean_target_position_hf_rms_rad:.6f} | "
            "{mean_target_default_error_rms_rad:.4f}/{mean_target_velocity_rms_rad_s:.3f}/{mean_target_acceleration_rms_rad_s2:.2f} | "
            "{mean_joint_velocity_rms_rad_s:.3f}/{mean_joint_acceleration_rms_rad_s2:.2f} | "
            "{mean_pd_torque_rms_nm:.2f}/{mean_pd_torque_rate_rms_nm_s:.2f} | "
            "{mean_relative_pd_torque_rms:.4f}/{mean_soft_peak_pd_torque_rms:.4f} | "
            "{mean_pd_torque_hf_rms_nm:.5f}/{mean_pd_torque_saturation_fraction:.3%} | "
            "{mean_mechanical_power_rms_w:.3f} |".format(**profile)
        )
    lines.extend(
        [
            "",
            "### 单次长期指标",
            "",
            "| 场景 | seed | jerk RMS | jerk P95 | jerk 最大值 | 20–25 Hz 平均/最大 | 足距 RMS/P95/最大 cm | ±1 cm / ±2 cm |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for run in summary["runs"]:
        lines.append(
            f"| {run['profile']} | {run['seed']} | {run['joint_jerk_rms_rad_s3']:.2f} | "
            f"{run['joint_jerk_p95_abs_rad_s3']:.2f} | {run['joint_jerk_max_abs_rad_s3']:.2f} | "
            f"{run['joint_position_hf_mean_rms_rad']:.6f}/{run['joint_position_hf_max_rms_rad']:.6f} | "
            f"{100.0 * run['feet_distance_error_rms_m']:.3f}/"
            f"{100.0 * run['feet_distance_error_p95_abs_m']:.3f}/"
            f"{100.0 * run['feet_distance_error_max_abs_m']:.3f} | "
            f"{run['feet_distance_within_1cm_fraction']:.1%}/"
            f"{run['feet_distance_within_2cm_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 单次运行",
            "",
            "| 场景 | seed | 健康 | fall time s | 水平速度 MAE | yaw-rate MAE | torso roll/pitch | root 最低高度 | 外力次数 | 关节限位裁剪数 |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for run in summary["runs"]:
        fall_time = "-" if run["fall_time_s"] is None else f"{float(run['fall_time_s']):.3f}"
        lines.append(
            f"| {run['profile']} | {run['seed']} | {'是' if run['healthy'] else '否'} | {fall_time} | "
            f"{run['lin_vel_xy_mae_m_s']:.4f} | {run['yaw_rate_mae_rad_s']:.4f} | "
            f"{run['torso_roll_error_rad']:.4f}/{run['torso_pitch_error_rad']:.4f} | "
            f"{run['min_root_height_m']:.4f} | {run['wrench_event_count']} | {run['joint_limit_clip_count']} |"
        )
    pose_runs = [run for run in summary["runs"] if run["profile"] == "pose_recovery"]
    if pose_runs:
        lines.extend(
            [
                "",
                "## 随机姿态恢复单次结果",
                "",
                "| seed | 存活 | 恢复默认姿态 | 初始 MAE | 最终 MAE | 最终最大误差 | 误差下降比例 | 首次持续进入误差带 s |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in pose_runs:
            recovery_time = (
                "-"
                if run["joint_recovery_time_s"] is None
                else f"{float(run['joint_recovery_time_s']):.3f}"
            )
            lines.append(
                f"| {run['seed']} | {'是' if run['healthy'] else '否'} | "
                f"{'是' if run['pose_recovered'] else '否'} | "
                f"{run['initial_joint_mae_rad']:.4f} | {run['final_joint_mae_rad']:.4f} | "
                f"{run['final_joint_max_abs_error_rad']:.4f} | {run['joint_recovery_ratio']:.1%} | "
                f"{recovery_time} |"
            )
        joint_values: dict[str, list[float]] = {}
        for run in pose_runs:
            for joint_name, value in run["final_error_by_joint_rad"].items():
                joint_values.setdefault(joint_name, []).append(float(value))
        worst_joints = sorted(
            (
                (joint_name, statistics.fmean(values), max(values))
                for joint_name, values in joint_values.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        lines.extend(
            [
                "",
                "## 最终偏差最大的关节",
                "",
                f"| 关节 | {len(pose_runs)}次测试平均最终绝对误差 rad | 最差 seed 最终绝对误差 rad |",
                "| --- | ---: | ---: |",
            ]
        )
        for joint_name, mean_error, max_error in worst_joints:
            lines.append(f"| `{joint_name}` | {mean_error:.4f} | {max_error:.4f} |")
    feet_runs = [
        run
        for run in summary["runs"]
        if run["profile"] == "feet_distance_recovery"
    ]
    if feet_runs:
        lines.extend(
            [
                "",
                "## 随机双脚间距恢复单次结果",
                "",
                "| seed | 存活 | 有效扰动 | 恢复默认距离 | 默认/初始足距 m | 初始偏差 cm | 最终 MAE/最大误差 cm | 首次持续进入 ±2 cm s |",
                "| ---: | --- | --- | --- | --- | ---: | --- | ---: |",
            ]
        )
        for run in feet_runs:
            recovery_time = (
                "-"
                if run["feet_recovery_time_s"] is None
                else f"{float(run['feet_recovery_time_s']):.3f}"
            )
            lines.append(
                f"| {run['seed']} | {'是' if run['healthy'] else '否'} | "
                f"{'是' if run['feet_perturbation_applied'] else '否'} | "
                f"{'是' if run['feet_distance_recovered'] else '否'} | "
                f"{run['default_feet_distance_m']:.4f}/{run['feet_initial_distance_m']:.4f} | "
                f"{100.0 * abs(run['feet_initial_error_m']):.2f} | "
                f"{100.0 * run['feet_final_error_mean_abs_m']:.2f}/"
                f"{100.0 * run['feet_final_error_max_abs_m']:.2f} | "
                f"{recovery_time} |"
            )
    jerk_by_joint: dict[str, list[float]] = {}
    hf_by_joint: dict[str, list[float]] = {}
    for run in summary["runs"]:
        for joint_name, value in run["joint_jerk_per_joint_rms_rad_s3"].items():
            if math.isfinite(float(value)):
                jerk_by_joint.setdefault(joint_name, []).append(float(value))
        for joint_name, value in run["joint_position_hf_per_joint_rms_rad"].items():
            if math.isfinite(float(value)):
                hf_by_joint.setdefault(joint_name, []).append(float(value))
    highest_jerk_joints = sorted(
        (
            (
                joint_name,
                statistics.fmean(values),
                statistics.fmean(hf_by_joint.get(joint_name, [math.nan])),
            )
            for joint_name, values in jerk_by_joint.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    if highest_jerk_joints:
        lines.extend(
            [
                "",
                "## 平均 jerk 最大的关节",
                "",
                "| 关节 | jerk RMS rad/s³ | 20–25 Hz 位置 RMS rad |",
                "| --- | ---: | ---: |",
            ]
        )
        for joint_name, jerk_rms, hf_rms in highest_jerk_joints:
            lines.append(f"| `{joint_name}` | {jerk_rms:.2f} | {hf_rms:.6f} |")
    lines.extend(
        [
            "",
            "## 合同检查",
            "",
            "基础验收要求 nominal/recovery 全部健康、robust 至少 2/3 健康，所有基础场景速度指令必须恒为零，且 `action_override=false`。随机姿态恢复还要求最终全身关节 MAE 和任一关节最大误差同时进入配置阈值；随机双脚间距场景要求确实施加至少 5 cm 初始偏差，并在最终窗口持续恢复到默认距离 ±2 cm；压力测试结果只描述超训练分布余量，不代表真机允许施加同等扰动。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    summary = build_summary(
        args.results_root,
        model_label=args.model_label,
        checkpoint=args.checkpoint,
        policy=args.policy,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(f"MuJoCo suite summary: {args.output_markdown}")
    print(f"Acceptance: {'PASS' if summary['acceptance']['pass'] else 'FAIL'}")
    if args.require_pass and not summary["acceptance"]["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
