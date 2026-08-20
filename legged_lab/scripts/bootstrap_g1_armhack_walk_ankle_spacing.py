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
    state = dict(state)
    actor_bias_keys = [
        key
        for key in (
            "actor.6.bias",
            "lateral_expert_actor.6.bias",
            "pure_yaw_expert_actor.6.bias",
        )
        if key in state
    ]
    if not actor_bias_keys:
        raise RuntimeError("checkpoint has no supported actor output bias")
    for key in actor_bias_keys:
        bias = state[key].clone()
        if tuple(bias.shape) != (29,):
            raise RuntimeError(f"unexpected {key} shape: {tuple(bias.shape)}")
        bias[LEFT_HIP_ROLL_ACTION_INDEX] += args.action_bias
        bias[RIGHT_HIP_ROLL_ACTION_INDEX] -= args.action_bias
        state[key] = bias
    checkpoint["model_state_dict"] = state
    checkpoint["optimizer_state_dict"] = {}
    checkpoint["iter"] = 0
    infos = dict(checkpoint.get("infos") or {})
    cumulative_bias = float(infos.get("ankle_spacing_hip_roll_action_bias", 0.0))
    checkpoint["infos"] = {
        **infos,
        "ankle_spacing_bootstrap_source_sha256": actual_sha,
        "ankle_spacing_hip_roll_action_bias_delta": float(args.action_bias),
        "ankle_spacing_hip_roll_action_bias": cumulative_bias + float(args.action_bias),
        "ankle_spacing_hip_roll_joint_offset_rad": float(
            (cumulative_bias + args.action_bias) * ACTION_SCALE_RAD
        ),
        "ankle_spacing_adjusted_actor_bias_keys": actor_bias_keys,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"output={output}")
    print(f"sha256={sha256(output)}")
    print(f"adjusted_actors={','.join(actor_bias_keys)}")
    print(
        "cumulative_hip_roll_joint_offsets_rad="
        f"(+{(cumulative_bias + args.action_bias) * ACTION_SCALE_RAD:.6f},"
        f"-{(cumulative_bias + args.action_bias) * ACTION_SCALE_RAD:.6f})"
    )


if __name__ == "__main__":
    main()
