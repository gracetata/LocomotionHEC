#!/usr/bin/env python3
"""Merge three independently fine-tuned 30-cm-spacing actors for deployment."""

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


def load_actor(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", {})
    actor = {key: value for key, value in state.items() if key.startswith("actor.")}
    if tuple(actor["actor.0.weight"].shape) != (512, 96):
        raise RuntimeError(f"invalid actor input shape in {path}")
    if tuple(actor["actor.6.weight"].shape) != (29, 128):
        raise RuntimeError(f"invalid actor output shape in {path}")
    if not all(bool(torch.isfinite(value).all()) for value in actor.values()):
        raise RuntimeError(f"non-finite actor tensor in {path}")
    return checkpoint, actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lateral", type=Path, required=True)
    parser.add_argument("--yaw", type=Path, required=True)
    parser.add_argument("--gate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {name: value.expanduser().resolve() for name, value in vars(args).items() if name != "output"}
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    base_checkpoint, base_actor = load_actor(paths["base"])
    _, lateral_actor = load_actor(paths["lateral"])
    yaw_checkpoint, yaw_actor = load_actor(paths["yaw"])
    gate_checkpoint = torch.load(paths["gate_source"], map_location="cpu", weights_only=False)
    gate_state = gate_checkpoint["model_state_dict"]

    merged = dict(base_checkpoint)
    state = dict(base_checkpoint["model_state_dict"])
    for prefix, actor in (
        ("lateral_expert_actor.", lateral_actor),
        ("pure_yaw_expert_actor.", yaw_actor),
    ):
        for key, value in actor.items():
            state[prefix + key[len("actor.") :]] = value.clone()
    for key in (
        "lateral_expert_forward_command",
        "lateral_expert_same_yaw_abs",
        "pure_yaw_expert_forward_command",
        "pure_yaw_expert_lateral_command",
        "pure_yaw_expert_yaw_scale",
    ):
        state[key] = gate_state[key].clone()
    for key, value in base_actor.items():
        state[key] = value.clone()
    merged["model_state_dict"] = state
    merged["optimizer_state_dict"] = {}
    yaw_infos = dict(yaw_checkpoint.get("infos") or {})
    merged["infos"] = {
        **dict(base_checkpoint.get("infos") or {}),
        "ankle_distance_target_m": 0.30,
        "ankle_distance_kernel_std_m": 0.06,
        "ankle_distance_kernel_weight": 500.0,
        "ankle_spacing_hip_roll_action_bias": 0.30,
        "ankle_spacing_base_sha256": sha256(paths["base"]),
        "ankle_spacing_lateral_sha256": sha256(paths["lateral"]),
        "ankle_spacing_yaw_sha256": sha256(paths["yaw"]),
        "ankle_spacing_gate_source_sha256": sha256(paths["gate_source"]),
    }
    if "yaw_drift_hip_pitch_action_bias" in yaw_infos:
        merged["infos"]["yaw_drift_hip_pitch_action_bias"] = float(
            yaw_infos["yaw_drift_hip_pitch_action_bias"]
        )
        merged["infos"]["yaw_drift_calibration_source_sha256"] = yaw_infos[
            "yaw_drift_calibration_source_sha256"
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"merged: {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
