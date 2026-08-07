#!/usr/bin/env python3
"""Compose independently trained strict-command residuals on model_9996.

The lateral and pure-yaw adapters are activated by disjoint command masks.
Composing their state dictionaries is therefore exact: lateral inference is
bit-for-bit identical to the lateral source, pure-yaw inference is identical
to the yaw source, and every other command uses the protected base actor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


MODEL9996_SIZE = 16_202_421
MODEL9996_SHA256 = "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"missing model_state_dict: {path}")
    for name, tensor in checkpoint["model_state_dict"].items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite model tensor {name}: {path}")
    return checkpoint


def require_equal_state(prefix: str, left: dict, right: dict) -> None:
    left_keys = {key for key in left if key.startswith(prefix)}
    right_keys = {key for key in right if key.startswith(prefix)}
    if left_keys != right_keys:
        raise RuntimeError(f"{prefix} key sets differ")
    for key in sorted(left_keys):
        if not torch.equal(left[key], right[key]):
            raise RuntimeError(f"shared frozen state differs at {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral-checkpoint", type=Path, required=True)
    parser.add_argument("--yaw-checkpoint", type=Path, required=True)
    parser.add_argument("--model9996", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lateral_path = args.lateral_checkpoint.resolve(strict=True)
    yaw_path = args.yaw_checkpoint.resolve(strict=True)
    model9996_path = args.model9996.resolve(strict=True)
    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output_path}")
    if model9996_path.stat().st_size != MODEL9996_SIZE or sha256(model9996_path) != MODEL9996_SHA256:
        raise RuntimeError("protected model_9996 size or SHA-256 mismatch")

    lateral = load(lateral_path)
    yaw = load(yaw_path)
    model9996 = load(model9996_path)
    lateral_state = lateral["model_state_dict"]
    yaw_state = yaw["model_state_dict"]
    base_state = model9996["model_state_dict"]

    require_equal_state("actor.", lateral_state, yaw_state)
    require_equal_state("actor.", lateral_state, base_state)
    for state, label in ((lateral_state, "lateral"), (yaw_state, "yaw")):
        bridge = state.get("fixed_command_bridge_fraction")
        if bridge is None or float(bridge.item()) != 0.0:
            raise RuntimeError(f"{label} source has a nonzero deployed carrier bridge")

    lateral_keys = sorted(key for key in lateral_state if key.startswith("lateral_command_residual."))
    yaw_keys = sorted(key for key in yaw_state if key.startswith("pure_yaw_command_residual."))
    if not lateral_keys or not yaw_keys:
        raise RuntimeError("both residual modules must exist")
    if not any(torch.count_nonzero(lateral_state[key]).item() for key in lateral_keys):
        raise RuntimeError("lateral residual source is identically zero")
    if not any(torch.count_nonzero(yaw_state[key]).item() for key in yaw_keys):
        raise RuntimeError("pure-yaw residual source is identically zero")

    merged = copy.deepcopy(lateral)
    merged_state = merged["model_state_dict"]
    for key in yaw_keys:
        merged_state[key] = yaw_state[key].clone()
    merged["iter"] = max(int(lateral.get("iter", 0)), int(yaw.get("iter", 0)))
    merged["two_goal_model9996_provenance"] = {
        "protected_model9996": str(model9996_path),
        "protected_model9996_sha256": MODEL9996_SHA256,
        "lateral_checkpoint": str(lateral_path),
        "lateral_sha256": sha256(lateral_path),
        "yaw_checkpoint": str(yaw_path),
        "yaw_sha256": sha256(yaw_path),
        "composition": "disjoint command-gated residual adapters",
    }

    # Verify exact branch preservation before writing.
    for key in lateral_keys:
        if not torch.equal(merged_state[key], lateral_state[key]):
            raise RuntimeError(f"lateral branch changed during composition: {key}")
    for key in yaw_keys:
        if not torch.equal(merged_state[key], yaw_state[key]):
            raise RuntimeError(f"yaw branch changed during composition: {key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f"wrote {output_path}")
    print(f"size={output_path.stat().st_size} sha256={sha256(output_path)}")


if __name__ == "__main__":
    main()
