#!/usr/bin/env python3
"""Scale one deployed model_9996 command-residual output without touching its base actor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


MODEL9996_SIZE = 16_202_421
MODEL9996_SHA256 = "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"
BRANCH_PREFIX = {
    "lateral": "lateral_command_residual",
    "pure_yaw": "pure_yaw_command_residual",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a strict lateral or pure-yaw residual by scaling its final linear layer."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=tuple(BRANCH_PREFIX), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--model9996", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    model9996 = args.model9996.resolve()
    if not source.is_file() or not model9996.is_file():
        raise FileNotFoundError("Source and protected model_9996 checkpoints must exist.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite residual calibration output: {output}")
    if not 0.0 < args.scale <= 1.0:
        raise ValueError("Residual scale must be in (0, 1].")
    if model9996.stat().st_size != MODEL9996_SIZE or sha256(model9996) != MODEL9996_SHA256:
        raise RuntimeError("Protected model_9996 failed its size/SHA-256 contract.")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    baseline = torch.load(model9996, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    baseline_state = baseline["model_state_dict"]
    if float(state["fixed_command_bridge_fraction"]) != 0.0:
        raise RuntimeError("Calibration requires a deployed bridge fraction of exactly zero.")
    for key, value in baseline_state.items():
        if key.startswith("actor.") and not torch.equal(state[key], value):
            raise RuntimeError(f"Candidate does not retain the exact model_9996 actor: {key}")

    prefix = BRANCH_PREFIX[args.branch]
    scaled_keys = (f"{prefix}.2.weight", f"{prefix}.2.bias")
    for key in scaled_keys:
        if key not in state:
            raise KeyError(f"Missing residual output tensor: {key}")
        state[key] = state[key] * float(args.scale)
    checkpoint["residual_calibration"] = {
        "source": str(source),
        "source_sha256": sha256(source),
        "model9996_sha256": MODEL9996_SHA256,
        "branch": args.branch,
        "final_linear_scale": float(args.scale),
        "scaled_keys": scaled_keys,
    }
    if not finite_tree(checkpoint):
        raise RuntimeError("Residual calibration produced a non-finite checkpoint.")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    if model9996.stat().st_size != MODEL9996_SIZE or sha256(model9996) != MODEL9996_SHA256:
        raise RuntimeError("Protected model_9996 changed during residual calibration.")
    print(f"Calibrated {args.branch} residual by {args.scale:.6f}: {output}")
    print(f"SHA-256: {sha256(output)}")


if __name__ == "__main__":
    main()
