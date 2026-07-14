#!/usr/bin/env python3
"""Train a lightweight T1 arm-style prior from demonstration motion data.

This script trains a small MLP that predicts the 14 upper-body arm joint
positions from root velocity plus non-arm joint positions/velocities. The
exported checkpoint stores model weights, normalization statistics, joint names,
and validation metrics so it can be used by the AMP MDP reward
``arm_style_prior_exp``.

Inputs:
- ``--motion_dir``: directory of Lab-format T1 motion pickles containing
  ``root_pos``, ``root_rot`` in wxyz order, and 29-DoF ``dof_pos``.

Outputs:
- ``--output``: ``torch.save`` checkpoint with ``model_state_dict``, normalizers,
  ``input_joint_names``, ``target_joint_names``, and training metrics;
- adjacent ``.json`` metadata summary for quick inspection.

Usage:
    /home/hecggdz/miniconda3/envs/env_leglab/bin/python \
      scripts/tools/train_t1_arm_style_prior.py \
      --motion_dir source/legged_lab/legged_lab/data/MotionData/T1_Lab/cmu_walk_50hz_core \
      --output logs/arm_style_prior/t1_cmu_walk_core_arm_prior.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LAB_DOF_NAMES = [
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
ARM_JOINT_NAMES = LAB_DOF_NAMES[2:16]
INPUT_JOINT_NAMES = LAB_DOF_NAMES[16:]


class ArmStylePrior(nn.Module):
    """Small MLP used to predict arm joint positions from gait state."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ELU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return conjugates for wxyz quaternions."""

    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multiply wxyz quaternions."""

    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def quat_apply_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors from world frame into the local frame of wxyz quaternions."""

    zeros = np.zeros((*vector.shape[:-1], 1), dtype=vector.dtype)
    vector_quat = np.concatenate([zeros, vector], axis=-1)
    rotated = quat_mul(quat_mul(quat_conjugate(quat), vector_quat), quat)
    return rotated[..., 1:]


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """Compute per-frame finite-difference velocity with endpoint padding."""

    velocity = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return velocity
    velocity[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    velocity[0] = (values[1] - values[0]) * fps
    velocity[-1] = (values[-1] - values[-2]) * fps
    return velocity


def local_angular_velocity(quat: np.ndarray, fps: float) -> np.ndarray:
    """Approximate local angular velocity from consecutive wxyz root quaternions."""

    angular_velocity = np.zeros((quat.shape[0], 3), dtype=np.float32)
    if quat.shape[0] < 2:
        return angular_velocity
    delta = quat_mul(quat_conjugate(quat[:-1]), quat[1:])
    sign = np.where(delta[:, :1] < 0.0, -1.0, 1.0)
    delta *= sign
    vector_norm = np.linalg.norm(delta[:, 1:], axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[:, :1], 1.0e-8, None))
    axis = delta[:, 1:] / np.clip(vector_norm, 1.0e-8, None)
    angular_velocity[:-1] = axis * angle * fps
    angular_velocity[-1] = angular_velocity[-2]
    return angular_velocity


def load_dataset(motion_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all motion files and build normalized-regression samples."""

    motion_paths = sorted(motion_dir.glob("*.pkl"))
    if not motion_paths:
        raise FileNotFoundError(f"No .pkl files found in {motion_dir}")

    name_to_index = {name: index for index, name in enumerate(LAB_DOF_NAMES)}
    input_joint_ids = [name_to_index[name] for name in INPUT_JOINT_NAMES]
    target_joint_ids = [name_to_index[name] for name in ARM_JOINT_NAMES]
    feature_blocks = []
    target_blocks = []
    for motion_path in motion_paths:
        motion = joblib.load(motion_path)
        fps = float(motion.get("fps", 50))
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_quat = np.asarray(motion["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
        dof_vel = finite_difference(dof_pos, fps)
        root_lin_vel_w = finite_difference(root_pos, fps)
        root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w).astype(np.float32)
        root_ang_vel_b = local_angular_velocity(root_quat, fps)
        features = np.concatenate(
            [
                root_lin_vel_b,
                root_ang_vel_b,
                dof_pos[:, input_joint_ids],
                dof_vel[:, input_joint_ids],
            ],
            axis=-1,
        )
        targets = dof_pos[:, target_joint_ids]
        finite_mask = np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1)
        feature_blocks.append(features[finite_mask])
        target_blocks.append(targets[finite_mask])

    return np.concatenate(feature_blocks, axis=0), np.concatenate(target_blocks, axis=0)


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized values plus mean/std arrays."""

    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.clip(std, 1.0e-5, None)
    return (values - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_model(args: argparse.Namespace) -> dict:
    """Train the MLP and return a serializable checkpoint payload."""

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, targets = load_dataset(Path(args.motion_dir))
    feature_norm, feature_mean, feature_std = standardize(features)
    target_norm, target_mean, target_std = standardize(targets)

    indices = np.arange(feature_norm.shape[0])
    np.random.shuffle(indices)
    split = int(round(indices.shape[0] * (1.0 - args.val_fraction)))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = TensorDataset(
        torch.as_tensor(feature_norm[train_indices], dtype=torch.float32),
        torch.as_tensor(target_norm[train_indices], dtype=torch.float32),
    )
    val_features = torch.as_tensor(feature_norm[val_indices], dtype=torch.float32, device=args.device)
    val_targets = torch.as_tensor(target_norm[val_indices], dtype=torch.float32, device=args.device)

    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value.strip()]
    model = ArmStylePrior(feature_norm.shape[1], target_norm.shape[1], hidden_dims).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_val = math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(args.device)
            batch_targets = batch_targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.square(model(batch_features) - batch_targets))
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * batch_features.shape[0]
            train_count += batch_features.shape[0]
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean(torch.square(model(val_features) - val_targets)).item()
        train_loss = train_loss_sum / max(train_count, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} best_val={best_val:.6f}")

    assert best_state is not None
    return {
        "model_state_dict": best_state,
        "input_mean": torch.as_tensor(feature_mean, dtype=torch.float32),
        "input_std": torch.as_tensor(feature_std, dtype=torch.float32),
        "target_mean": torch.as_tensor(target_mean, dtype=torch.float32),
        "target_std": torch.as_tensor(target_std, dtype=torch.float32),
        "input_joint_names": INPUT_JOINT_NAMES,
        "target_joint_names": ARM_JOINT_NAMES,
        "root_feature_names": ["root_lin_vel_b_x", "root_lin_vel_b_y", "root_lin_vel_b_z", "root_ang_vel_b_x", "root_ang_vel_b_y", "root_ang_vel_b_z"],
        "hidden_dims": hidden_dims,
        "activation": "elu",
        "num_samples": int(features.shape[0]),
        "num_train_samples": int(train_indices.shape[0]),
        "num_val_samples": int(val_indices.shape[0]),
        "best_epoch": int(best_epoch),
        "best_val_mse_normalized": float(best_val),
        "motion_dir": str(Path(args.motion_dir).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a T1 arm-style prior from Lab-format motion data.")
    parser.add_argument("--motion_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="logs/arm_style_prior/t1_cmu_walk_core_arm_prior.pt")
    parser.add_argument("--hidden_dims", type=str, default="64,64")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    payload = train_model(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    metadata = {
        key: value
        for key, value in payload.items()
        if key != "model_state_dict" and not isinstance(value, torch.Tensor)
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Saved arm style prior: {output_path}")
    print(f"[INFO] Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
