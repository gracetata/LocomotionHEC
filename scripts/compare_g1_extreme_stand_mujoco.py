#!/usr/bin/env python3
"""Compare fixed-baseline Extreme Stand MuJoCo reports.

The comparison is deliberately stricter than the per-model smoke acceptance:
the candidate must remain healthy, preserve pose/feet recovery, reduce steady
jerk, and avoid persistent post-push vibration.  Stress and very large pushes
remain diagnostic, but they are still checked for regressions against the
baseline policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CORE_PROFILES = (
    "nominal",
    "pose_recovery",
    "feet_distance_recovery",
    "recovery",
    "robust",
    "stress",
)
JERK_PROFILES = (
    "nominal",
    "pose_recovery",
    "feet_distance_recovery",
    "recovery",
    "robust",
)
PUSH_LEVELS_N = (120, 180, 240, 360)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profiles(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["profile"]: item for item in summary["profiles"]}


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def _ratio(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return math.nan
    if abs(baseline) <= 1.0e-12:
        return 1.0 if abs(candidate) <= 1.0e-12 else math.inf
    return candidate / baseline


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def compare(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    baseline_core = _load(baseline_root / "core" / "summary.json")
    candidate_core = _load(candidate_root / "core" / "summary.json")
    baseline_profiles = _profiles(baseline_core)
    candidate_profiles = _profiles(candidate_core)
    checks: list[dict[str, Any]] = []

    missing = [name for name in CORE_PROFILES if name not in candidate_profiles]
    _check(checks, "候选核心场景完整", not missing, f"missing={missing}")
    _check(
        checks,
        "候选基础验收通过",
        bool(candidate_core.get("acceptance", {}).get("pass", False)),
        f"acceptance={candidate_core.get('acceptance', {}).get('pass')}",
    )
    baseline_control_metrics_complete = all(
        baseline_profiles[name].get("control_chain_metrics_available_count", 0)
        == baseline_profiles[name]["run_count"]
        for name in CORE_PROFILES
        if name in baseline_profiles
    )
    candidate_control_metrics_complete = all(
        candidate_profiles[name].get("control_chain_metrics_available_count", 0)
        == candidate_profiles[name]["run_count"]
        for name in CORE_PROFILES
        if name in candidate_profiles
    )
    _check(
        checks,
        "V4/V5 控制链平滑指标完整",
        baseline_control_metrics_complete and candidate_control_metrics_complete,
        (
            f"baseline={baseline_control_metrics_complete}, "
            f"candidate={candidate_control_metrics_complete}"
        ),
    )

    profile_rows: list[dict[str, Any]] = []
    for name in CORE_PROFILES:
        if name not in baseline_profiles or name not in candidate_profiles:
            continue
        baseline = baseline_profiles[name]
        candidate = candidate_profiles[name]
        health_ok = candidate["healthy_rate"] + 1.0e-12 >= baseline["healthy_rate"]
        falls_ok = candidate["fall_count"] <= baseline["fall_count"]
        _check(
            checks,
            f"{name} 生存率不退化",
            health_ok and falls_ok,
            (
                f"healthy {baseline['healthy_rate']:.3f}→{candidate['healthy_rate']:.3f}; "
                f"falls {baseline['fall_count']}→{candidate['fall_count']}"
            ),
        )
        profile_rows.append(
            {
                "profile": name,
                "baseline_healthy_rate": baseline["healthy_rate"],
                "candidate_healthy_rate": candidate["healthy_rate"],
                "baseline_jerk_rms": baseline["mean_joint_jerk_rms_rad_s3"],
                "candidate_jerk_rms": candidate["mean_joint_jerk_rms_rad_s3"],
                "jerk_ratio": _ratio(
                    candidate["mean_joint_jerk_rms_rad_s3"],
                    baseline["mean_joint_jerk_rms_rad_s3"],
                ),
                "baseline_position_hf_rms": baseline["mean_joint_position_hf_rms_rad"],
                "candidate_position_hf_rms": candidate["mean_joint_position_hf_rms_rad"],
                "position_hf_ratio": _ratio(
                    candidate["mean_joint_position_hf_rms_rad"],
                    baseline["mean_joint_position_hf_rms_rad"],
                ),
                "action_rate_ratio": _ratio(
                    candidate["mean_actor_action_rate_rms_per_s"],
                    baseline["mean_actor_action_rate_rms_per_s"],
                ),
                "action_second_difference_ratio": _ratio(
                    candidate["mean_actor_action_second_difference_rms"],
                    baseline["mean_actor_action_second_difference_rms"],
                ),
                "target_acceleration_ratio": _ratio(
                    candidate["mean_target_acceleration_rms_rad_s2"],
                    baseline["mean_target_acceleration_rms_rad_s2"],
                ),
                "torque_rate_ratio": _ratio(
                    candidate["mean_pd_torque_rate_rms_nm_s"],
                    baseline["mean_pd_torque_rate_rms_nm_s"],
                ),
                "mechanical_power_ratio": _ratio(
                    candidate["mean_mechanical_power_rms_w"],
                    baseline["mean_mechanical_power_rms_w"],
                ),
            }
        )

    baseline_jerk = _mean(
        [baseline_profiles[name]["mean_joint_jerk_rms_rad_s3"] for name in JERK_PROFILES]
    )
    candidate_jerk = _mean(
        [candidate_profiles[name]["mean_joint_jerk_rms_rad_s3"] for name in JERK_PROFILES]
    )
    jerk_ratio = _ratio(candidate_jerk, baseline_jerk)
    _check(
        checks,
        "核心场景稳态 jerk 至少降低 15%",
        jerk_ratio <= 0.85,
        f"mean jerk {_fmt(baseline_jerk, 2)}→{_fmt(candidate_jerk, 2)} rad/s³; ratio={_fmt(jerk_ratio, 3)}",
    )

    nominal_jerk_ratio = next(
        (row["jerk_ratio"] for row in profile_rows if row["profile"] == "nominal"),
        math.nan,
    )
    _check(
        checks,
        "无扰动长期 jerk 不退化",
        nominal_jerk_ratio <= 1.0,
        f"nominal jerk ratio={_fmt(nominal_jerk_ratio, 3)}",
    )

    # A lower joint jerk is not sufficient if the policy merely moves the
    # oscillation upstream into action/target commands or downstream into PD
    # torque.  Compare all five recovery-relevant profiles on the same window.
    smooth_metrics = (
        ("mean_actor_action_rate_rms_per_s", "action 变化率", 0.95),
        ("mean_actor_action_second_difference_rms", "action 二阶差分", 0.95),
        ("mean_target_acceleration_rms_rad_s2", "目标角加速度", 0.95),
        ("mean_pd_torque_rate_rms_nm_s", "PD 力矩变化率", 0.95),
        ("mean_mechanical_power_rms_w", "机械功率", 1.05),
    )
    for field, label, maximum_ratio in smooth_metrics:
        baseline_value = _mean([baseline_profiles[name][field] for name in JERK_PROFILES])
        candidate_value = _mean([candidate_profiles[name][field] for name in JERK_PROFILES])
        value_ratio = _ratio(candidate_value, baseline_value)
        _check(
            checks,
            f"核心场景{label}满足门限",
            value_ratio <= maximum_ratio,
            (
                f"{_fmt(baseline_value, 4)}→{_fmt(candidate_value, 4)}; "
                f"ratio={_fmt(value_ratio, 3)}, required<={maximum_ratio:.2f}"
            ),
        )

    baseline_target_error = _mean(
        [baseline_profiles[name]["mean_target_default_error_rms_rad"] for name in JERK_PROFILES]
    )
    candidate_target_error = _mean(
        [candidate_profiles[name]["mean_target_default_error_rms_rad"] for name in JERK_PROFILES]
    )
    target_error_limit = max(baseline_target_error * 1.10, baseline_target_error + 0.01)
    _check(
        checks,
        "目标角默认姿态误差不退化",
        candidate_target_error <= target_error_limit,
        (
            f"{_fmt(baseline_target_error, 4)}→{_fmt(candidate_target_error, 4)} rad; "
            f"limit={_fmt(target_error_limit, 4)}"
        ),
    )

    baseline_saturation = _mean(
        [baseline_profiles[name]["mean_pd_torque_saturation_fraction"] for name in JERK_PROFILES]
    )
    candidate_saturation = _mean(
        [candidate_profiles[name]["mean_pd_torque_saturation_fraction"] for name in JERK_PROFILES]
    )
    _check(
        checks,
        "PD 力矩饱和率不退化",
        candidate_saturation <= baseline_saturation + 1.0e-4,
        (
            f"{baseline_saturation:.6%}→{candidate_saturation:.6%}; "
            f"limit={baseline_saturation + 1.0e-4:.6%}"
        ),
    )

    for profile_name, field, label, minimum in (
        ("pose_recovery", "pose_recovered_rate", "随机姿态恢复率", 2.0 / 3.0),
        ("feet_distance_recovery", "feet_distance_recovered_rate", "脚距恢复率", 2.0 / 3.0),
    ):
        baseline = baseline_profiles[profile_name][field]
        candidate = candidate_profiles[profile_name][field]
        _check(
            checks,
            f"{label}不退化",
            candidate + 1.0e-12 >= max(minimum, baseline),
            f"{baseline:.3f}→{candidate:.3f}; required>={max(minimum, baseline):.3f}",
        )

    # Do not trade jerk for a visibly looser stance or larger torso drift.
    for profile_name in ("nominal", "recovery", "robust"):
        baseline = baseline_profiles[profile_name]
        candidate = candidate_profiles[profile_name]
        for field, label, abs_margin in (
            ("mean_torso_roll_error_rad", "躯干 roll", 0.005),
            ("mean_torso_pitch_error_rad", "躯干 pitch", 0.005),
            ("mean_feet_distance_error_rms_m", "脚距 RMS", 0.005),
        ):
            limit = max(float(baseline[field]) * 1.10, float(baseline[field]) + abs_margin)
            _check(
                checks,
                f"{profile_name} {label}不明显退化",
                float(candidate[field]) <= limit,
                f"{baseline[field]:.5f}→{candidate[field]:.5f}; limit={limit:.5f}",
            )

    push_rows: list[dict[str, Any]] = []
    for force_n in PUSH_LEVELS_N:
        baseline_path = baseline_root / f"push_{force_n}n" / "summary.json"
        candidate_path = candidate_root / f"push_{force_n}n" / "summary.json"
        if not baseline_path.is_file() or not candidate_path.is_file():
            _check(
                checks,
                f"{force_n} N 四方向推力报告完整",
                False,
                f"baseline={baseline_path.is_file()}, candidate={candidate_path.is_file()}",
            )
            continue
        baseline = _profiles(_load(baseline_path))["large_push"]
        candidate = _profiles(_load(candidate_path))["large_push"]
        persistent = int(candidate["persistent_joint_vibration_count"])
        policy_hf = int(candidate["policy_action_high_frequency_count"])
        health_ok = candidate["healthy_rate"] + 1.0e-12 >= baseline["healthy_rate"]
        if force_n <= 180:
            health_ok = health_ok and candidate["healthy_rate"] == 1.0
        late_position_ratio = float(candidate["mean_large_push_late_position_hf_ratio"])
        late_action_ratio = float(candidate["mean_large_push_late_action_hf_ratio"])
        late_torque_ratio = float(candidate["mean_large_push_late_torque_hf_ratio"])
        settle_time = float(candidate["mean_large_push_action_rate_settling_time_s"])
        no_long_term_jitter = (
            persistent == 0
            and policy_hf == 0
            and late_position_ratio <= 1.5
            and late_action_ratio <= 1.5
            and late_torque_ratio <= 1.5
            and settle_time <= 6.0
        )
        _check(
            checks,
            f"{force_n} N 推力恢复能力不退化",
            health_ok,
            f"healthy {baseline['healthy_rate']:.3f}→{candidate['healthy_rate']:.3f}",
        )
        _check(
            checks,
            f"{force_n} N 推力后无持续 jerk",
            no_long_term_jitter,
            (
                f"persistent={persistent}, policy_hf={policy_hf}, "
                f"late ratios pos/action/torque={late_position_ratio:.2f}/"
                f"{late_action_ratio:.2f}/{late_torque_ratio:.2f}, settle={settle_time:.2f}s"
            ),
        )
        _check(
            checks,
            f"{force_n} N 力矩饱和不增加",
            candidate["pd_torque_saturation_count"] <= baseline["pd_torque_saturation_count"],
            (
                f"saturation runs {baseline['pd_torque_saturation_count']}→"
                f"{candidate['pd_torque_saturation_count']}"
            ),
        )
        push_rows.append(
            {
                "force_n": force_n,
                "baseline_healthy_rate": baseline["healthy_rate"],
                "candidate_healthy_rate": candidate["healthy_rate"],
                "candidate_persistent_vibration_count": persistent,
                "candidate_policy_action_hf_count": policy_hf,
                "baseline_settling_time_s": baseline["mean_large_push_action_rate_settling_time_s"],
                "candidate_settling_time_s": settle_time,
                "baseline_post_torque_rms_nm": baseline["mean_large_push_post_torque_rms_nm"],
                "candidate_post_torque_rms_nm": candidate["mean_large_push_post_torque_rms_nm"],
                "candidate_late_position_hf_ratio": late_position_ratio,
                "candidate_late_action_hf_ratio": late_action_ratio,
                "candidate_late_torque_hf_ratio": late_torque_ratio,
            }
        )

    return {
        "schema_version": 1,
        "baseline_root": str(baseline_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "pass": all(item["pass"] for item in checks),
        "core_jerk": {
            "baseline_mean_rms_rad_s3": baseline_jerk,
            "candidate_mean_rms_rad_s3": candidate_jerk,
            "candidate_over_baseline": jerk_ratio,
            "required_max_ratio": 0.85,
        },
        "profile_rows": profile_rows,
        "push_rows": push_rows,
        "checks": checks,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# G1 Extreme Stand V4/V5 固定基线 MuJoCo 对比",
        "",
        f"- 总结：**{'通过' if report['pass'] else '未通过'}**",
        f"- V4 结果：`{report['baseline_root']}`",
        f"- V5 结果：`{report['candidate_root']}`",
        (
            "- 核心场景 jerk RMS："
            f"{_fmt(report['core_jerk']['baseline_mean_rms_rad_s3'], 2)} → "
            f"{_fmt(report['core_jerk']['candidate_mean_rms_rad_s3'], 2)} rad/s³，"
            f"比例 `{_fmt(report['core_jerk']['candidate_over_baseline'], 3)}`"
        ),
        "",
        "## 验收项",
        "",
        "| 结果 | 验收项 | 证据 |",
        "|---|---|---|",
    ]
    for item in report["checks"]:
        lines.append(f"| {'PASS' if item['pass'] else 'FAIL'} | {item['name']} | {item['detail']} |")
    lines.extend(
        [
            "",
            "## 核心场景长期 jerk",
            "",
            "| 场景 | V4/V5 生存率 | jerk RMS V4→V5 | jerk 比例 | 位置高频比例 | action 变化/Δ² 比 | 目标加速度比 | 力矩变化率比 | 功率比 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["profile_rows"]:
        lines.append(
            f"| {row['profile']} | {row['baseline_healthy_rate']:.3f}/{row['candidate_healthy_rate']:.3f} | "
            f"{row['baseline_jerk_rms']:.2f}→{row['candidate_jerk_rms']:.2f} | "
            f"{_fmt(row['jerk_ratio'], 3)} | {_fmt(row['position_hf_ratio'], 3)} | "
            f"{_fmt(row['action_rate_ratio'], 3)}/{_fmt(row['action_second_difference_ratio'], 3)} | "
            f"{_fmt(row['target_acceleration_ratio'], 3)} | {_fmt(row['torque_rate_ratio'], 3)} | "
            f"{_fmt(row['mechanical_power_ratio'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 四方向大推力恢复",
            "",
            "| 推力 | V4/V5 生存率 | V5 持续关节/策略高频次数 | V4→V5 收敛时间 s | V4→V5 推力后力矩 RMS Nm | V5 后期位置/action/力矩高频比 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["push_rows"]:
        lines.append(
            f"| {row['force_n']} N | {row['baseline_healthy_rate']:.3f}/{row['candidate_healthy_rate']:.3f} | "
            f"{row['candidate_persistent_vibration_count']}/{row['candidate_policy_action_hf_count']} | "
            f"{row['baseline_settling_time_s']:.2f}→{row['candidate_settling_time_s']:.2f} | "
            f"{row['baseline_post_torque_rms_nm']:.2f}→{row['candidate_post_torque_rms_nm']:.2f} | "
            f"{row['candidate_late_position_hf_ratio']:.2f}/{row['candidate_late_action_hf_ratio']:.2f}/"
            f"{row['candidate_late_torque_hf_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 判定原则",
            "",
            "核心长期场景要求平均 jerk RMS 至少降低 15%，nominal jerk 不得增加；随机姿态、脚距和基础生存率不得退化。120/180 N 四方向必须全部存活；全部推力档均不得出现持续关节或策略高频，推力结束后期位置、action、力矩高频比均需不大于 1.5，action-rate 最迟 6 秒内重新稳定。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = compare(args.baseline_root, args.candidate_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Comparison report: {args.output_markdown}")
    print(f"Closed-loop acceptance: {'PASS' if report['pass'] else 'FAIL'}")
    if args.require_pass and not report["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
