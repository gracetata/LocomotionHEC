#!/usr/bin/env python3
"""Apply a symmetric hip-pitch output calibration to one pure-yaw actor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


LEFT_HIP_PITCH_ACTION_INDEX = 0
RIGHT_HIP_PITCH_ACTION_INDEX = 1


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
    parser.add_argument("--hip-pitch-action-bias", type=float, required=True)
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    actual_sha = sha256(source)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"source SHA-256 mismatch: {actual_sha}")
    if abs(args.hip_pitch_action_bias) > 0.20:
        raise ValueError("hip-pitch-action-bias magnitude must be <= 0.20")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = dict(checkpoint["model_state_dict"])
    bias = state["actor.6.bias"].clone()
    if tuple(bias.shape) != (29,):
        raise RuntimeError(f"unexpected output bias shape: {tuple(bias.shape)}")
    bias[LEFT_HIP_PITCH_ACTION_INDEX] += args.hip_pitch_action_bias
    bias[RIGHT_HIP_PITCH_ACTION_INDEX] += args.hip_pitch_action_bias
    state["actor.6.bias"] = bias
    checkpoint["model_state_dict"] = state
    checkpoint["optimizer_state_dict"] = {}
    checkpoint["infos"] = {
        **dict(checkpoint.get("infos") or {}),
        "yaw_drift_calibration_source_sha256": actual_sha,
        "yaw_drift_hip_pitch_action_bias": float(args.hip_pitch_action_bias),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"output={output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
