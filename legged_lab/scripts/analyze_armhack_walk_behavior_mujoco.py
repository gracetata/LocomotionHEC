#!/usr/bin/env python3
"""Evaluate one scheduled ArmHack Walk MuJoCo report against targeted behavior criteria."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _projection(segment: dict) -> float:
    command = segment["command"]
    actual = [segment["steady_mean_lin_vel_x"], segment["steady_mean_lin_vel_y"]]
    magnitude = math.hypot(command[0], command[1])
    if magnitude <= 1.0e-8:
        return 0.0
    return (actual[0] * command[0] + actual[1] * command[1]) / magnitude


def _check(condition: bool, label: str, value: float, limit: str) -> dict:
    return {"name": label, "passed": bool(condition), "value": float(value), "limit": limit}


def evaluate(report: dict, scenario_name: str) -> dict:
    checks: list[dict] = []
    health = report["health"]
    checks.append(_check(bool(health["healthy"]), "未摔倒", 1.0 if health["healthy"] else 0.0, "==1"))
    segments = report.get("command_segments", [])
    if not segments:
        raise ValueError("MuJoCo report has no command_segments")

    for segment in segments:
        name = str(segment.get("name", ""))
        command = segment["command"]
        command_norm = math.sqrt(sum(float(value) ** 2 for value in command))
        if command_norm <= 1.0e-6:
            planar_speed = math.hypot(
                segment["steady_mean_lin_vel_x"], segment["steady_mean_lin_vel_y"]
            )
            checks.extend(
                [
                    _check(planar_speed <= 0.035, f"{name}:零速平移", planar_speed, "<=0.035 m/s"),
                    _check(
                        abs(segment["steady_mean_yaw_rate"]) <= 0.06,
                        f"{name}:零速转动",
                        abs(segment["steady_mean_yaw_rate"]),
                        "<=0.06 rad/s",
                    ),
                    _check(
                        segment["steady_step_frequency_hz"] <= 0.5,
                        f"{name}:零速落脚频率",
                        segment["steady_step_frequency_hz"],
                        "<=0.5 Hz",
                    ),
                    _check(
                        segment["steady_planar_displacement_m"] <= 0.12,
                        f"{name}:零速漂移",
                        segment["steady_planar_displacement_m"],
                        "<=0.12 m",
                    ),
                ]
            )
            continue

        # All nonzero commands must produce real stepping and avoid sole overlap.
        checks.append(
            _check(
                segment["steady_touchdown_count"] >= 1,
                f"{name}:存在迈步",
                segment["steady_touchdown_count"],
                ">=1 touchdown",
            )
        )
        checks.append(
            _check(
                segment["steady_min_signed_sole_clearance_m"] >= -0.005,
                f"{name}:足底不相交",
                segment["steady_min_signed_sole_clearance_m"],
                ">=-0.005 m",
            )
        )

        translation = math.hypot(command[0], command[1])
        if translation > 0.0:
            projected_speed = _projection(segment)
            minimum = 0.015 if translation <= 0.10 else 0.08
            checks.append(
                _check(
                    projected_speed >= minimum,
                    f"{name}:指令方向速度",
                    projected_speed,
                    f">={minimum} m/s",
                )
            )
        if abs(command[2]) > 0.0:
            signed_yaw = math.copysign(1.0, command[2]) * segment["steady_mean_yaw_rate"]
            checks.extend(
                [
                    _check(
                        signed_yaw >= 0.15,
                        f"{name}:原地转向响应",
                        signed_yaw,
                        ">=0.15 rad/s",
                    ),
                    _check(
                        segment["steady_max_abs_roll_rad"] <= 0.25,
                        f"{name}:原地转向躯干侧倾",
                        segment["steady_max_abs_roll_rad"],
                        "<=0.25 rad",
                    ),
                ]
            )

    if scenario_name in {"lateral_left", "lateral_right", "diagonal_front_left", "diagonal_front_right"}:
        segment = segments[-1]
        checks.append(
            _check(
                segment["steady_sole_clearance_violation_fraction"] <= 0.20,
                "侧移/斜移足底安全间距占比",
                segment["steady_sole_clearance_violation_fraction"],
                "<=0.20",
            )
        )
    if scenario_name == "forward_cadence":
        frequency = segments[-1]["steady_step_frequency_hz"]
        checks.extend(
            [
                _check(frequency >= 0.5, "前向步频下限", frequency, ">=0.5 Hz"),
                _check(frequency <= 3.0, "前向步频上限", frequency, "<=3.0 Hz"),
            ]
        )

    return {
        "scenario": scenario_name,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = evaluate(report, args.scenario)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [check for check in result["checks"] if not check["passed"]]
    print(
        f"[BEHAVIOR {'PASS' if result['passed'] else 'FAIL'}] {args.scenario}: "
        f"{len(result['checks']) - len(failed)}/{len(result['checks'])} checks passed"
    )
    for check in failed:
        print(f"  - {check['name']}: value={check['value']:.6g}, required {check['limit']}")
    if args.enforce and failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
