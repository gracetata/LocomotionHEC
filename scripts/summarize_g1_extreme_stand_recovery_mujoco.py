#!/usr/bin/env python3
"""Aggregate the multi-profile Extreme Stand MuJoCo suite into JSON and Chinese Markdown."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROFILE_ORDER = ("nominal", "pose_recovery", "recovery", "robust", "stress")


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


def _finite_float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else math.nan


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
                "sim_time_s": _finite_float(report.get("sim_time", 0.0)),
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
    }


def build_summary(results_root: Path) -> dict[str, Any]:
    runs = load_runs(results_root)
    profiles = [summarize_profile(profile, runs) for profile in PROFILE_ORDER if any(run["profile"] == profile for run in runs)]
    profile_map = {profile["profile"]: profile for profile in profiles}
    profile_names = set(profile_map)
    pose_only = profile_names == {"pose_recovery"}
    mandatory_profiles = ("nominal", "recovery", "robust")
    required_present = pose_only or all(name in profile_map for name in mandatory_profiles)
    if pose_only:
        acceptance_pass = bool(
            profile_map["pose_recovery"]["healthy_rate"] == 1.0
            and profile_map["pose_recovery"]["pose_recovered_rate"] >= 0.8
            and profile_map["pose_recovery"]["all_zero_command"]
            and profile_map["pose_recovery"]["all_actor_outputs_unmodified"]
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
        )
    return {
        "schema_version": 1,
        "results_root": str(results_root.resolve()),
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
            },
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    pose_only = {profile["profile"] for profile in summary["profiles"]} == {
        "pose_recovery"
    }
    lines = [
        "# G1 Extreme Stand Recovery MuJoCo 全面测试报告",
        "",
        f"- 测试目录：`{summary['results_root']}`",
        f"- 总运行数：{summary['run_count']}",
        f"- 基础验收：{'通过' if summary['acceptance']['pass'] else '未通过'}",
        (
            "- 本报告只验收随机关节初始姿态能否恢复到严格默认全身姿态。"
            if pose_only
            else "- `stress` 是超训练分布压力测试，不计入基础验收。"
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
    lines.extend(
        [
            "",
            "## 合同检查",
            "",
            "基础验收要求 nominal/recovery 全部健康、robust 至少 2/3 健康，所有基础场景速度指令必须恒为零，且 `action_override=false`。随机姿态恢复还要求最终全身关节 MAE 和任一关节最大误差同时进入配置阈值；压力测试结果只描述超训练分布余量，不代表真机允许施加同等扰动。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    summary = build_summary(args.results_root)
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
