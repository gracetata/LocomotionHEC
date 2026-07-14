#!/usr/bin/env python3
"""Summarize directional gait diagnostic JSON reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = (
    "logs/rsl_rl/g1_amp/"
    "2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500/"
    "directional_gait_analysis"
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)
parser.add_argument("--output_json", type=str, default="")
parser.add_argument("--output_md", type=str, default="")
args = parser.parse_args()


def resolve(path: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def get(data: dict[str, Any], path: str, default: float = float("nan")) -> Any:
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def fnum(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def fmt(value: Any, digits: int = 3) -> str:
    number = fnum(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def gait_flags(row: dict[str, Any]) -> list[str]:
    label = str(row["label"])
    flags: list[str] = []
    lin_mae = fnum(row["lin_mae"])
    yaw_mae = fnum(row["yaw_mae"])
    single = fnum(row["single"])
    double_stance = fnum(row["double_stance"])
    double_air = fnum(row["double_air"])
    contact_speed_p95 = fnum(row["contact_speed_p95"])
    clearance_p05 = fnum(row["clearance_p05"])
    clearance_p95 = fnum(row["clearance_p95"])

    if label.startswith("turn"):
        if lin_mae > 0.25:
            flags.append("lin_tracking")
        if yaw_mae > 0.35:
            flags.append("yaw_tracking")
    elif label in {"backward", "lateral_left", "lateral_right"}:
        if lin_mae > 0.18:
            flags.append("lin_tracking")
        if yaw_mae > 0.30:
            flags.append("yaw_tracking")
    else:
        if lin_mae > 0.12:
            flags.append("lin_tracking")
        if yaw_mae > 0.25:
            flags.append("yaw_tracking")

    if label != "stand":
        if single < 0.12 and double_stance > 0.75:
            flags.append("stuck_or_hop")
        if double_air > 0.08:
            flags.append("double_air")
        if contact_speed_p95 > 0.45:
            flags.append("foot_slide")
        if math.isfinite(clearance_p05) and clearance_p05 < 0.015:
            flags.append("low_clearance")
        if math.isfinite(clearance_p95) and clearance_p95 > 0.16:
            flags.append("high_clearance")
    return flags


def row_from_report(report: dict[str, Any]) -> dict[str, Any]:
    policy = report["policy"]
    reference = report.get("reference", {})
    row = {
        "label": report.get("label", ""),
        "cmd_vx": get(reference, "target.vx"),
        "cmd_vy": get(reference, "target.vy"),
        "cmd_wz": get(reference, "target.yaw"),
        "vx": get(policy, "root_vel_b.x.mean"),
        "vy": get(policy, "root_vel_b.y.mean"),
        "wz": get(policy, "root_ang_vel_b_z.mean"),
        "lin_mae": get(policy, "tracking_mae.lin_xy"),
        "yaw_mae": get(policy, "tracking_mae.yaw"),
        "single": get(policy, "contact.single_stance_fraction"),
        "double_stance": get(policy, "contact.double_stance_fraction"),
        "double_air": get(policy, "contact.double_air_fraction"),
        "double_air_p95_s": get(policy, "contact.double_air_duration_s.p95"),
        "contact_diff": get(policy, "contact.contact_fraction_diff_abs"),
        "air_diff_s": get(policy, "contact.left_right_air_duration_mean_diff_abs_s"),
        "contact_speed_p95": get(policy, "feet.contact_foot_speed_xy.p95"),
        "clearance_p05": get(policy, "feet.swing_clearance_relative_to_lower_foot_m.p05"),
        "clearance_p50": get(policy, "feet.swing_clearance_relative_to_lower_foot_m.p50"),
        "clearance_p95": get(policy, "feet.swing_clearance_relative_to_lower_foot_m.p95"),
        "height_mean": get(policy, "root_height.z.mean"),
        "height_p05": get(policy, "root_height.z.p05"),
        "ref_match_frames": int(fnum(get(reference, "match_frames", 0.0))),
        "ref_match_ratio": get(reference, "match_ratio"),
        "ref_single": get(reference, "contact_matched.single_stance_fraction"),
        "ref_double_stance": get(reference, "contact_matched.double_stance_fraction"),
        "ref_double_air": get(reference, "contact_matched.double_air_fraction"),
    }
    row["flags"] = gait_flags(row)
    return row


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "mode",
        "cmd",
        "vel",
        "lin_mae",
        "yaw_mae",
        "single",
        "double_st",
        "double_air",
        "slide_p95",
        "clr_p05/p95",
        "ref_frames",
        "flags",
    ]
    lines = [
        "# Directional Gait Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        command = f"({fmt(row['cmd_vx'], 2)},{fmt(row['cmd_vy'], 2)},{fmt(row['cmd_wz'], 2)})"
        velocity = f"({fmt(row['vx'], 2)},{fmt(row['vy'], 2)},{fmt(row['wz'], 2)})"
        clearance = f"{fmt(row['clearance_p05'], 3)}/{fmt(row['clearance_p95'], 3)}"
        flags = ", ".join(row["flags"]) if row["flags"] else "ok"
        values = [
            str(row["label"]),
            command,
            velocity,
            fmt(row["lin_mae"]),
            fmt(row["yaw_mae"]),
            fmt(row["single"]),
            fmt(row["double_stance"]),
            fmt(row["double_air"]),
            fmt(row["contact_speed_p95"]),
            clearance,
            str(row["ref_match_frames"]),
            flags,
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Flags are heuristic triage signals, not final acceptance criteria.")
    return "\n".join(lines) + "\n"


def main() -> None:
    report_dir = resolve(args.report_dir)
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Report directory does not exist: {report_dir}")
    report_paths = sorted(report_dir.glob("*_directional_gait_report.json"))
    if not report_paths:
        raise FileNotFoundError(f"No directional gait reports found in {report_dir}")

    rows = []
    for path in report_paths:
        with path.open(encoding="utf-8") as file:
            report = json.load(file)
        rows.append(row_from_report(report))
    rows.sort(key=lambda row: str(row["label"]))

    output_json = resolve(args.output_json) if args.output_json else report_dir / "directional_gait_summary.json"
    output_md = resolve(args.output_md) if args.output_md else report_dir / "directional_gait_summary.md"
    with output_json.open("w", encoding="utf-8") as file:
        json.dump({"report_dir": str(report_dir), "rows": rows}, file, indent=2, sort_keys=True)
    output_md.write_text(markdown_table(rows), encoding="utf-8")

    print(markdown_table(rows))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
