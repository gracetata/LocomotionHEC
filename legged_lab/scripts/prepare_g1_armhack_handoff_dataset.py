#!/usr/bin/env python3
"""Merge and filter real producer states for a safe ArmHack handoff stage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-role", required=True)
    parser.add_argument("--to-role", required=True)
    parser.add_argument("--max-command-norm", type=float, default=0.02)
    parser.add_argument("--max-linear-speed", type=float, default=0.10)
    parser.add_argument("--max-yaw-rate", type=float, default=0.15)
    parser.add_argument("--max-joint-speed", type=float, default=4.0)
    args = parser.parse_args()

    accepted, rejected = [], {"role": 0, "command": 0, "linear": 0, "yaw": 0, "joint": 0}
    for input_path in args.input:
        payload = json.loads(input_path.expanduser().resolve().read_text(encoding="utf-8"))
        for state in payload["policy_switch_states"]:
            if state.get("from") != args.from_role or state.get("to") != args.to_role:
                rejected["role"] += 1
                continue
            if math.sqrt(sum(float(v) ** 2 for v in state["command"])) > args.max_command_norm:
                rejected["command"] += 1
                continue
            velocity = state["root_velocity_world"]
            if math.hypot(float(velocity[0]), float(velocity[1])) > args.max_linear_speed:
                rejected["linear"] += 1
                continue
            if abs(float(velocity[5])) > args.max_yaw_rate:
                rejected["yaw"] += 1
                continue
            if max(abs(float(v)) for v in state["joint_velocities"].values()) > args.max_joint_speed:
                rejected["joint"] += 1
                continue
            accepted.append(state)
    if len(accepted) < 32:
        raise RuntimeError(f"only {len(accepted)} states passed the handoff gates")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "contract": {
            "from": args.from_role,
            "to": args.to_role,
            "max_command_norm": args.max_command_norm,
            "max_linear_speed": args.max_linear_speed,
            "max_yaw_rate": args.max_yaw_rate,
            "max_joint_speed": args.max_joint_speed,
        },
        "summary": {"accepted": len(accepted), "rejected": rejected},
        "policy_switch_states": accepted,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
