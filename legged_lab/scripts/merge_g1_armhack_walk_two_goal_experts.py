#!/usr/bin/env python3
"""Merge the protected ArmHack actor with strict lateral and pure-yaw experts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


BASE_SIZE = 14_826_139
BASE_SHA256 = "1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"


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


def actor_state(checkpoint: dict, label: str) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", {})
    required = {
        "actor.0.weight": (512, 96),
        "actor.6.weight": (29, 128),
    }
    for key, shape in required.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise RuntimeError(f"{label} {key} must have shape {shape}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lateral-expert", type=Path, required=True)
    parser.add_argument("--yaw-expert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-forward", type=float, default=0.0)
    parser.add_argument("--lateral-yaw-abs", type=float, default=0.0)
    parser.add_argument("--yaw-forward", type=float, default=0.0)
    parser.add_argument("--yaw-lateral", type=float, default=0.0)
    parser.add_argument("--yaw-scale", type=float, default=1.0)
    args = parser.parse_args()

    base_path = args.base.expanduser().resolve()
    lateral_path = args.lateral_expert.expanduser().resolve()
    yaw_path = args.yaw_expert.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if base_path.stat().st_size != BASE_SIZE or sha256(base_path) != BASE_SHA256:
        raise RuntimeError("ArmHack model_10990 failed its immutable size/SHA-256 contract")

    base = torch.load(base_path, map_location="cpu", weights_only=False)
    lateral = torch.load(lateral_path, map_location="cpu", weights_only=False)
    yaw = torch.load(yaw_path, map_location="cpu", weights_only=False)
    if not all(finite_tree(item) for item in (base, lateral, yaw)):
        raise RuntimeError("an input checkpoint contains a non-finite tensor")
    base_state = actor_state(base, "base")
    lateral_state = actor_state(lateral, "lateral")
    yaw_state = actor_state(yaw, "yaw")

    merged = base
    state = merged["model_state_dict"]
    for prefix, source in (
        ("lateral_expert_actor.", lateral_state),
        ("pure_yaw_expert_actor.", yaw_state),
    ):
        for key, value in source.items():
            if key.startswith("actor."):
                state[prefix + key[len("actor.") :]] = value.clone()

    state["lateral_expert_forward_command"] = torch.tensor(float(args.lateral_forward))
    state["lateral_expert_same_yaw_abs"] = torch.tensor(float(args.lateral_yaw_abs))
    state["pure_yaw_expert_forward_command"] = torch.tensor(float(args.yaw_forward))
    state["pure_yaw_expert_lateral_command"] = torch.tensor(float(args.yaw_lateral))
    state["pure_yaw_expert_yaw_scale"] = torch.tensor(float(args.yaw_scale))
    merged["infos"] = dict(merged.get("infos") or {})
    merged["infos"].update(
        {
            "armhack_two_goal_base_sha256": BASE_SHA256,
            "armhack_two_goal_lateral_sha256": sha256(lateral_path),
            "armhack_two_goal_yaw_sha256": sha256(yaw_path),
            "lateral_expert_forward_command": float(args.lateral_forward),
            "lateral_expert_same_yaw_abs": float(args.lateral_yaw_abs),
            "pure_yaw_expert_forward_command": float(args.yaw_forward),
            "pure_yaw_expert_lateral_command": float(args.yaw_lateral),
            "pure_yaw_expert_yaw_scale": float(args.yaw_scale),
        }
    )
    for key, value in base_state.items():
        if key.startswith("actor.") and not torch.equal(state[key], value):
            raise RuntimeError(f"protected base actor changed during merge: {key}")
    if not finite_tree(merged):
        raise RuntimeError("merged checkpoint contains a non-finite tensor")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"Merged ArmHack gated experts: {output}")
    print(f"SHA-256: {sha256(output)}")


if __name__ == "__main__":
    main()
