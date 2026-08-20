#!/usr/bin/env python3
"""Split a deployed ArmHack gated checkpoint into trainable actor checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_keys(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    extracted = {
        "actor." + key[len(prefix) :]: value.clone()
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if set(key.split(".", 1)[1] for key in extracted) != {
        key[len("actor.") :] for key in state if key.startswith("actor.")
    }:
        raise RuntimeError(f"{prefix} does not match the base actor architecture")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    actual_sha = sha256(source)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"source SHA-256 mismatch: {actual_sha}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    branches = {
        "base": {key: value.clone() for key, value in state.items() if key.startswith("actor.")},
        "lateral": actor_keys(state, "lateral_expert_actor."),
        "yaw": actor_keys(state, "pure_yaw_expert_actor."),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for branch, actor in branches.items():
        output = output_dir / f"source_{branch}.pt"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
        branch_checkpoint = dict(checkpoint)
        branch_state = {
            key: value
            for key, value in state.items()
            if not key.startswith("actor.")
            and not key.startswith("lateral_expert_actor.")
            and not key.startswith("pure_yaw_expert_actor.")
            and key not in {
                "lateral_expert_forward_command",
                "lateral_expert_same_yaw_abs",
                "pure_yaw_expert_forward_command",
                "pure_yaw_expert_lateral_command",
                "pure_yaw_expert_yaw_scale",
            }
        }
        branch_state.update(actor)
        branch_checkpoint["model_state_dict"] = branch_state
        branch_checkpoint["optimizer_state_dict"] = {}
        branch_checkpoint["iter"] = 0
        branch_checkpoint["infos"] = {
            **dict(checkpoint.get("infos") or {}),
            "ankle_spacing_split_source_sha256": actual_sha,
            "ankle_spacing_branch": branch,
        }
        torch.save(branch_checkpoint, output)
        print(f"{branch}: {output} sha256={sha256(output)}")


if __name__ == "__main__":
    main()
