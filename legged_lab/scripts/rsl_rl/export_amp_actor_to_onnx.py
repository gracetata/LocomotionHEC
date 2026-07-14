#!/usr/bin/env python3
"""
Export an RSL-RL AMP checkpoint actor without starting Isaac Sim.

This script loads only ``model_state_dict`` from an AMP checkpoint, rebuilds the
feed-forward actor trunk, and writes an ONNX graph with one input ``obs`` and one
output ``actions``. Discriminator weights, discriminator optimizer state, critic
weights, and PPO optimizer state are intentionally ignored.

Inputs:
    --checkpoint: RSL-RL ``model_*.pt`` checkpoint containing ``model_state_dict``.
    --output: Destination ONNX path. Defaults to ``<checkpoint_dir>/exported/policy.onnx``.
    --jit-output: Optional TorchScript path, e.g. ``<checkpoint_dir>/exported/policy.pt``.
    --metadata: Optional JSON path for deployment metadata. Defaults beside the ONNX file.
    --robot: Deployment metadata profile, ``t1`` by default for backward compatibility.

Outputs:
    ONNX actor policy with shape ``[1, obs_dim] -> [1, action_dim]``.
    Optional TorchScript actor policy consumable by unitree_sim2sim2real deployment.
    JSON metadata describing the selected robot observation/action layout and PD deployment constants.

Usage:
    python scripts/rsl_rl/export_amp_actor_to_onnx.py \
        --checkpoint logs/rsl_rl/t1_amp/run/model_400.pt \
        --output ../HEC_S3_Simulation/models/t1_amp_cmu_walk_core_model_400.onnx

    python scripts/rsl_rl/export_amp_actor_to_onnx.py \
        --robot g1 \
        --checkpoint logs/rsl_rl/g1_amp/run/model_1400.pt \
        --output logs/rsl_rl/g1_amp/run/exported/policy.onnx \
        --jit-output logs/rsl_rl/g1_amp/run/exported/policy.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


FULL_JOINT_NAMES: List[str] = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

ACTION_JOINT_NAMES: List[str] = FULL_JOINT_NAMES[2:]

DEFAULT_JOINT_POS: Dict[str, float] = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.25,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "Left_Wrist_Pitch": 0.0,
    "Left_Wrist_Yaw": 0.0,
    "Left_Hand_Roll": -0.26,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.25,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.09,
    "Right_Wrist_Pitch": 0.0,
    "Right_Wrist_Yaw": 0.0,
    "Right_Hand_Roll": 0.26,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.45,
    "Left_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.45,
    "Right_Ankle_Pitch": -0.25,
    "Right_Ankle_Roll": 0.0,
}

PD_GAINS: Dict[str, Dict[str, float]] = {
    "AAHead_yaw": {"kp": 40.0, "kd": 1.0},
    "Head_pitch": {"kp": 40.0, "kd": 1.0},
    "Left_Shoulder_Pitch": {"kp": 40.0, "kd": 1.0},
    "Left_Shoulder_Roll": {"kp": 40.0, "kd": 1.0},
    "Left_Elbow_Pitch": {"kp": 40.0, "kd": 1.0},
    "Left_Elbow_Yaw": {"kp": 40.0, "kd": 1.0},
    "Left_Wrist_Pitch": {"kp": 40.0, "kd": 1.0},
    "Left_Wrist_Yaw": {"kp": 40.0, "kd": 1.0},
    "Left_Hand_Roll": {"kp": 40.0, "kd": 1.0},
    "Right_Shoulder_Pitch": {"kp": 40.0, "kd": 1.0},
    "Right_Shoulder_Roll": {"kp": 40.0, "kd": 1.0},
    "Right_Elbow_Pitch": {"kp": 40.0, "kd": 1.0},
    "Right_Elbow_Yaw": {"kp": 40.0, "kd": 1.0},
    "Right_Wrist_Pitch": {"kp": 40.0, "kd": 1.0},
    "Right_Wrist_Yaw": {"kp": 40.0, "kd": 1.0},
    "Right_Hand_Roll": {"kp": 40.0, "kd": 1.0},
    "Waist": {"kp": 200.0, "kd": 5.0},
    "Left_Hip_Pitch": {"kp": 100.0, "kd": 2.0},
    "Left_Hip_Roll": {"kp": 100.0, "kd": 2.0},
    "Left_Hip_Yaw": {"kp": 100.0, "kd": 2.0},
    "Left_Knee_Pitch": {"kp": 150.0, "kd": 4.0},
    "Left_Ankle_Pitch": {"kp": 40.0, "kd": 2.0},
    "Left_Ankle_Roll": {"kp": 40.0, "kd": 2.0},
    "Right_Hip_Pitch": {"kp": 100.0, "kd": 2.0},
    "Right_Hip_Roll": {"kp": 100.0, "kd": 2.0},
    "Right_Hip_Yaw": {"kp": 100.0, "kd": 2.0},
    "Right_Knee_Pitch": {"kp": 150.0, "kd": 4.0},
    "Right_Ankle_Pitch": {"kp": 40.0, "kd": 2.0},
    "Right_Ankle_Roll": {"kp": 40.0, "kd": 2.0},
}

G1_JOINT_NAMES: List[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_DEFAULT_JOINT_POS: Dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "waist_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

G1_PD_GAINS: Dict[str, Dict[str, float]] = {
    "left_hip_pitch_joint": {"kp": 100.0, "kd": 2.0},
    "right_hip_pitch_joint": {"kp": 100.0, "kd": 2.0},
    "waist_yaw_joint": {"kp": 200.0, "kd": 5.0},
    "left_hip_roll_joint": {"kp": 100.0, "kd": 2.0},
    "right_hip_roll_joint": {"kp": 100.0, "kd": 2.0},
    "waist_roll_joint": {"kp": 40.0, "kd": 5.0},
    "left_hip_yaw_joint": {"kp": 100.0, "kd": 2.0},
    "right_hip_yaw_joint": {"kp": 100.0, "kd": 2.0},
    "waist_pitch_joint": {"kp": 40.0, "kd": 5.0},
    "left_knee_joint": {"kp": 150.0, "kd": 4.0},
    "right_knee_joint": {"kp": 150.0, "kd": 4.0},
    "left_shoulder_pitch_joint": {"kp": 40.0, "kd": 1.0},
    "right_shoulder_pitch_joint": {"kp": 40.0, "kd": 1.0},
    "left_ankle_pitch_joint": {"kp": 40.0, "kd": 2.0},
    "right_ankle_pitch_joint": {"kp": 40.0, "kd": 2.0},
    "left_shoulder_roll_joint": {"kp": 40.0, "kd": 1.0},
    "right_shoulder_roll_joint": {"kp": 40.0, "kd": 1.0},
    "left_ankle_roll_joint": {"kp": 40.0, "kd": 2.0},
    "right_ankle_roll_joint": {"kp": 40.0, "kd": 2.0},
    "left_shoulder_yaw_joint": {"kp": 40.0, "kd": 1.0},
    "right_shoulder_yaw_joint": {"kp": 40.0, "kd": 1.0},
    "left_elbow_joint": {"kp": 40.0, "kd": 1.0},
    "right_elbow_joint": {"kp": 40.0, "kd": 1.0},
    "left_wrist_roll_joint": {"kp": 40.0, "kd": 1.0},
    "right_wrist_roll_joint": {"kp": 40.0, "kd": 1.0},
    "left_wrist_pitch_joint": {"kp": 40.0, "kd": 1.0},
    "right_wrist_pitch_joint": {"kp": 40.0, "kd": 1.0},
    "left_wrist_yaw_joint": {"kp": 40.0, "kd": 1.0},
    "right_wrist_yaw_joint": {"kp": 40.0, "kd": 1.0},
}


class AmpActorExporter(torch.nn.Module):
    """Feed-forward AMP actor wrapper with optional actor-observation normalization."""

    def __init__(self, model_state: Dict[str, torch.Tensor], activation_name: str) -> None:
        super().__init__()
        self.obs_dim, self.action_dim = actor_dimensions(model_state)
        self.actor = build_actor(model_state, activation_name)
        normalizer = build_actor_normalizer(model_state)
        self.normalizer = normalizer if normalizer is not None else torch.nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


class FixedNormalizer(torch.nn.Module):
    """Static ``(x - mean) / std`` normalizer for checkpoints that contain actor stats."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / torch.clamp(self.std, min=1.0e-6)


def build_actor(model_state: Dict[str, torch.Tensor], activation_name: str) -> torch.nn.Sequential:
    linear_indices = sorted(
        int(key.split(".")[1])
        for key in model_state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    if not linear_indices:
        raise ValueError("No actor.*.weight tensors found in checkpoint model_state_dict.")

    activation_cls = activation_from_name(activation_name)
    layers: List[torch.nn.Module] = []
    for position, layer_index in enumerate(linear_indices):
        weight = model_state[f"actor.{layer_index}.weight"].float()
        bias = model_state[f"actor.{layer_index}.bias"].float()
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        layers.append(layer)
        if position != len(linear_indices) - 1:
            layers.append(activation_cls())
    return torch.nn.Sequential(*layers)


def build_actor_normalizer(model_state: Dict[str, torch.Tensor]) -> FixedNormalizer | None:
    for prefix in ("actor_obs_normalizer", "normalizer"):
        mean_key = f"{prefix}._mean"
        std_key = f"{prefix}._std"
        if mean_key in model_state and std_key in model_state:
            return FixedNormalizer(model_state[mean_key], model_state[std_key])
    return None


def activation_from_name(name: str) -> type[torch.nn.Module]:
    normalized = name.lower()
    if normalized == "elu":
        return torch.nn.ELU
    if normalized == "relu":
        return torch.nn.ReLU
    if normalized == "tanh":
        return torch.nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def actor_dimensions(model_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    first_weight_key = min(
        (key for key in model_state if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    last_weight_key = max(
        (key for key in model_state if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    obs_dim = int(model_state[first_weight_key].shape[1])
    action_dim = int(model_state[last_weight_key].shape[0])
    return obs_dim, action_dim


def default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "exported" / "policy.onnx"


def default_jit_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "exported" / "policy.pt"


def policy_obs_layout(robot: str) -> list[dict[str, int | str]]:
    if robot == "g1":
        return [
            {"name": "base_ang_vel", "start": 0, "end": 3},
            {"name": "projected_gravity", "start": 3, "end": 6},
            {"name": "velocity_commands", "start": 6, "end": 9},
            {"name": "joint_pos_rel_g1_29dof_lab_order", "start": 9, "end": 38},
            {"name": "joint_vel_g1_29dof_lab_order", "start": 38, "end": 67},
            {"name": "last_action_g1_29dof_lab_order", "start": 67, "end": 96},
        ]
    return [
        {"name": "base_ang_vel", "start": 0, "end": 3},
        {"name": "projected_gravity", "start": 3, "end": 6},
        {"name": "velocity_commands", "start": 6, "end": 9},
        {"name": "joint_pos_rel_no_head", "start": 9, "end": 36},
        {"name": "joint_vel_no_head", "start": 36, "end": 63},
        {"name": "last_action", "start": 63, "end": 90},
    ]


def robot_profile(robot: str) -> dict:
    if robot == "g1":
        return {
            "full_joint_names": G1_JOINT_NAMES,
            "action_joint_names": G1_JOINT_NAMES,
            "default_joint_pos": G1_DEFAULT_JOINT_POS,
            "pd_gains": G1_PD_GAINS,
            "default_command": {"lin_vel_x": 0.5, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
            "sim_dt": 0.002,
            "decimation": 10,
            "notes": [
                "AMP discriminator tensors are deliberately excluded from this export.",
                "The exported actor uses the 29-DoF IsaacLab AMP policy joint order from G1AmpEnvCfg.",
                "MuJoCo XML actuator order differs from policy order; use unitree_sim2sim2real/deploy/deploy_mujoco/configs/g1_amp.yaml for mapping.",
                "The export is offline and does not start Isaac Sim or GMR environments.",
            ],
        }
    return {
        "full_joint_names": FULL_JOINT_NAMES,
        "action_joint_names": ACTION_JOINT_NAMES,
        "default_joint_pos": DEFAULT_JOINT_POS,
        "pd_gains": PD_GAINS,
        "default_command": {"lin_vel_x": 0.6, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "sim_dt": 0.001,
        "decimation": 20,
        "hec_mujoco_default_root_height": 0.74,
        "notes": [
            "AMP discriminator tensors are deliberately excluded from this export.",
            "The exported actor uses the no-head 27-DoF action order from T1AmpEnvCfg.NO_HEAD_JOINT_NAMES.",
            "HEC MuJoCo T1.xml joint and actuator order matches the IsaacLab URDF full 29-DoF order; head joints are held at default targets during deployment.",
            "HEC headless MuJoCo validation uses ROOT_HEIGHT=0.74; ROOT_HEIGHT=0.82 was observed to fall in long rollout.",
        ],
    }


def write_metadata(
    path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    jit_path: Path | None,
    obs_dim: int,
    action_dim: int,
    robot: str,
) -> None:
    profile = robot_profile(robot)
    metadata = {
        "robot": robot,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "torchscript": str(jit_path) if jit_path is not None else None,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "control_frequency_hz": 50,
        "sim_dt": profile["sim_dt"],
        "decimation": profile["decimation"],
        "action_scale": 0.25,
        "action_formula": "q_target[action_joint] = default_joint_pos[action_joint] + 0.25 * action",
        "policy_obs_layout": policy_obs_layout(robot),
        "full_joint_names": profile["full_joint_names"],
        "action_joint_names": profile["action_joint_names"],
        "default_joint_pos": profile["default_joint_pos"],
        "pd_gains": profile["pd_gains"],
        "default_command": profile["default_command"],
        "notes": profile["notes"],
    }
    if "hec_mujoco_default_root_height" in profile:
        metadata["hec_mujoco_default_root_height"] = profile["hec_mujoco_default_root_height"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to RSL-RL model_*.pt checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Destination ONNX path.")
    parser.add_argument("--jit-output", type=Path, default=None, help="Optional TorchScript policy path.")
    parser.add_argument("--metadata", type=Path, default=None, help="Destination deployment metadata JSON path.")
    parser.add_argument("--robot", choices=["t1", "g1"], default="t1", help="Deployment metadata profile.")
    parser.add_argument("--activation", default="elu", choices=["elu", "relu", "tanh"], help="Actor activation.")
    parser.add_argument("--opset", default=11, type=int, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    onnx_path = (args.output or default_output_path(checkpoint_path)).expanduser().resolve()
    jit_path = args.jit_output.expanduser().resolve() if args.jit_output is not None else None
    metadata_path = (args.metadata or onnx_path.with_suffix(".deploy.json")).expanduser().resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")
    model_state = checkpoint["model_state_dict"]
    obs_dim, action_dim = actor_dimensions(model_state)

    actor = AmpActorExporter(model_state, activation_name=args.activation).eval().cpu()
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    if jit_path is not None:
        jit_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(actor).save(str(jit_path))

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_obs,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )

    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
    except Exception as exc:
        raise RuntimeError(f"ONNX checker failed for {onnx_path}: {exc}") from exc

    write_metadata(metadata_path, checkpoint_path, onnx_path, jit_path, obs_dim, action_dim, args.robot)

    with torch.no_grad():
        sample_output = actor(dummy_obs)
    print(f"[export_amp_actor_to_onnx] checkpoint: {checkpoint_path}")
    print(f"[export_amp_actor_to_onnx] onnx:       {onnx_path}")
    if jit_path is not None:
        print(f"[export_amp_actor_to_onnx] jit:        {jit_path}")
    print(f"[export_amp_actor_to_onnx] metadata:   {metadata_path}")
    print(f"[export_amp_actor_to_onnx] obs_dim={obs_dim} action_dim={action_dim}")
    print(f"[export_amp_actor_to_onnx] zero_obs_action_mean={sample_output.mean().item():.6f}")


if __name__ == "__main__":
    main()