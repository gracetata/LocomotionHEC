#!/usr/bin/env python3
"""Merge model_9996, a lateral expert, and an accepted pure-yaw residual."""

from __future__ import annotations

import argparse
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


def finite_tree(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model9996", type=Path, required=True)
    parser.add_argument("--lateral-expert", type=Path, required=True)
    parser.add_argument("--yaw-residual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-forward", type=float, default=-0.16)
    parser.add_argument("--lateral-yaw-abs", type=float, default=0.15)
    args = parser.parse_args()

    paths = tuple(path.expanduser().resolve() for path in (
        args.model9996, args.lateral_expert, args.yaw_residual
    ))
    model9996_path, lateral_path, yaw_path = paths
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if model9996_path.stat().st_size != MODEL9996_SIZE or sha256(model9996_path) != MODEL9996_SHA256:
        raise RuntimeError("model_9996 failed its immutable size/SHA-256 contract")

    base = torch.load(model9996_path, map_location="cpu", weights_only=False)
    lateral = torch.load(lateral_path, map_location="cpu", weights_only=False)
    yaw = torch.load(yaw_path, map_location="cpu", weights_only=False)
    if not all(finite_tree(item) for item in (base, lateral, yaw)):
        raise RuntimeError("an input checkpoint contains a non-finite tensor")

    base_state = base["model_state_dict"]
    lateral_state = lateral["model_state_dict"]
    yaw_state = yaw["model_state_dict"]
    for key, value in base_state.items():
        if key.startswith("actor.") and not torch.equal(yaw_state[key], value):
            raise RuntimeError(f"yaw residual did not preserve model_9996 base actor: {key}")
    for key in ("actor.0.weight", "actor.6.weight", "critic.0.weight"):
        if key not in lateral_state:
            raise RuntimeError(f"lateral expert lacks required tensor: {key}")
    if tuple(lateral_state["actor.0.weight"].shape) != (512, 96):
        raise RuntimeError("lateral expert input is not 96")
    if tuple(lateral_state["actor.6.weight"].shape) != (29, 128):
        raise RuntimeError("lateral expert output is not 29")

    merged = base
    state = merged["model_state_dict"]
    for key, value in lateral_state.items():
        if key.startswith("actor."):
            state["lateral_expert_actor." + key[len("actor.") :]] = value.clone()
    for prefix in ("lateral_command_residual.", "pure_yaw_command_residual."):
        for key, value in yaw_state.items():
            if key.startswith(prefix):
                state[key] = value.clone()
    for key in list(state):
        if key.startswith("lateral_command_residual.") and torch.is_tensor(state[key]):
            state[key].zero_()
    state["fixed_command_bridge_fraction"] = torch.tensor(0.0)
    state["lateral_expert_forward_command"] = torch.tensor(float(args.lateral_forward))
    state["lateral_expert_same_yaw_abs"] = torch.tensor(float(args.lateral_yaw_abs))
    merged["infos"] = dict(merged.get("infos") or {})
    merged["infos"].update(
        {
            "gated_expert_model9996_sha256": MODEL9996_SHA256,
            "gated_expert_lateral_sha256": sha256(lateral_path),
            "gated_expert_yaw_sha256": sha256(yaw_path),
            "lateral_expert_forward_command": float(args.lateral_forward),
            "lateral_expert_same_yaw_abs": float(args.lateral_yaw_abs),
        }
    )
    if not finite_tree(merged):
        raise RuntimeError("merged checkpoint contains a non-finite tensor")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"Merged gated experts: {output}")
    print(f"SHA-256: {sha256(output)}")


if __name__ == "__main__":
    main()
