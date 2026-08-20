#!/usr/bin/env python3
"""Add a symmetric hip-roll output bias before 30-cm ankle-spacing PPO."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


LEFT_HIP_ROLL_ACTION_INDEX = 3
RIGHT_HIP_ROLL_ACTION_INDEX = 4
ACTION_SCALE_RAD = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--action-bias", type=float, required=True)
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    actual_sha = sha256(source)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"source SHA-256 mismatch: {actual_sha}")
    if not 0.0 < args.action_bias <= 0.60:
        raise ValueError("action-bias must be in (0, 0.60]")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    bias = state["actor.6.bias"].clone()
    if tuple(bias.shape) != (29,):
        raise RuntimeError(f"unexpected output bias shape: {tuple(bias.shape)}")
    bias[LEFT_HIP_ROLL_ACTION_INDEX] += args.action_bias
    bias[RIGHT_HIP_ROLL_ACTION_INDEX] -= args.action_bias
    state = dict(state)
    state["actor.6.bias"] = bias
    checkpoint["model_state_dict"] = state
    checkpoint["optimizer_state_dict"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        **dict(checkpoint.get("infos") or {}),
        "ankle_spacing_bootstrap_source_sha256": actual_sha,
        "ankle_spacing_hip_roll_action_bias": float(args.action_bias),
        "ankle_spacing_hip_roll_joint_offset_rad": float(
            args.action_bias * ACTION_SCALE_RAD
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"output={output}")
    print(f"sha256={sha256(output)}")
    print(
        "hip_roll_joint_offsets_rad="
        f"(+{args.action_bias * ACTION_SCALE_RAD:.6f},"
        f"-{args.action_bias * ACTION_SCALE_RAD:.6f})"
    )


if __name__ == "__main__":
    main()
