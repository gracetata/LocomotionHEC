#!/usr/bin/env python3
"""Distill a MuJoCo-stable filtered Stand hold controller into the hold actor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_actor(state: dict[str, torch.Tensor]) -> nn.Sequential:
    actor = nn.Sequential(
        nn.Linear(state["actor.0.weight"].shape[1], state["actor.0.weight"].shape[0]),
        nn.ELU(),
        nn.Linear(state["actor.2.weight"].shape[1], state["actor.2.weight"].shape[0]),
        nn.ELU(),
        nn.Linear(state["actor.4.weight"].shape[1], state["actor.4.weight"].shape[0]),
        nn.ELU(),
        nn.Linear(state["actor.6.weight"].shape[1], state["actor.6.weight"].shape[0]),
    )
    actor.load_state_dict(
        {key[len("actor.") :]: value for key, value in state.items() if key.startswith("actor.")}
    )
    return actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--retention-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    actual_sha = sha256(source)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"source SHA-256 mismatch: {actual_sha}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch size, and learning rate must be positive")

    obs_parts, action_parts = [], []
    lower_indices: np.ndarray | None = None
    for trace_path in args.trace:
        with np.load(trace_path.expanduser().resolve()) as trace:
            obs_parts.append(np.asarray(trace["observations"], dtype=np.float32))
            action_parts.append(np.asarray(trace["actions"], dtype=np.float32))
            trace_indices = np.asarray(trace["lower_action_indices"], dtype=np.int64)
            if lower_indices is None:
                lower_indices = trace_indices
            elif not np.array_equal(lower_indices, trace_indices):
                raise ValueError("trace lower-action indices do not match")
    observations = np.concatenate(obs_parts, axis=0)
    target_actions = np.concatenate(action_parts, axis=0)
    if observations.shape[0] < 100 or observations.shape[1] != 96 or target_actions.shape[1] != 29:
        raise ValueError(f"invalid distillation dataset shapes: {observations.shape}, {target_actions.shape}")
    assert lower_indices is not None

    torch.manual_seed(args.seed)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor = build_actor(state)
    teacher = build_actor(state).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    obs_tensor = torch.from_numpy(observations)
    target_tensor = torch.from_numpy(target_actions)
    lower_tensor = torch.from_numpy(lower_indices)
    permutation = torch.randperm(obs_tensor.shape[0])
    split = max(int(0.90 * obs_tensor.shape[0]), 1)
    train_ids, validation_ids = permutation[:split], permutation[split:]
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    def lower_mse(indices: torch.Tensor) -> float:
        with torch.inference_mode():
            prediction = actor(obs_tensor[indices])[:, lower_tensor]
            target = target_tensor[indices][:, lower_tensor]
            return float(torch.mean(torch.square(prediction - target)).item())

    initial_validation_mse = lower_mse(validation_ids)
    actor.train()
    for _ in range(args.epochs):
        shuffled = train_ids[torch.randperm(train_ids.numel())]
        for start in range(0, shuffled.numel(), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            obs = obs_tensor[batch]
            prediction = actor(obs)
            target = target_tensor[batch]
            with torch.no_grad():
                source_action = teacher(obs)
            imitation = torch.mean(
                torch.square(prediction[:, lower_tensor] - target[:, lower_tensor])
            )
            retention = torch.mean(torch.square(prediction - source_action))
            loss = imitation + float(args.retention_weight) * retention
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
    actor.eval()
    final_validation_mse = lower_mse(validation_ids)

    actor_state = actor.state_dict()
    for key, value in actor_state.items():
        state[f"actor.{key}"] = value.detach().clone()
    checkpoint["model_state_dict"] = state
    checkpoint["optimizer_state_dict"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        **dict(checkpoint.get("infos") or {}),
        "armhack_hold_distillation_source_sha256": actual_sha,
        "armhack_hold_distillation_samples": int(observations.shape[0]),
        "armhack_hold_distillation_initial_validation_mse": initial_validation_mse,
        "armhack_hold_distillation_final_validation_mse": final_validation_mse,
        "armhack_hold_distillation_filter_teacher": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"samples={observations.shape[0]}")
    print(f"validation_mse={initial_validation_mse:.8f}->{final_validation_mse:.8f}")
    print(f"output={output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
