"""Termination helpers for AMP locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR, target_pose_success_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_pose_success_hold(
    env: ManagerBasedRLEnv,
    command_name: str,
    hold_steps: int = 8,
    position_threshold: float = 0.08,
    heading_threshold: float = 0.18,
    lin_vel_threshold: float = 0.08,
    yaw_vel_threshold: float = 0.12,
    mean_joint_threshold: float = 0.16,
    key_body_mean_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    key_body_asset_cfg: SceneEntityCfg | None = None,
    key_body_reference_attr: str = DEFAULT_TARGET_KEY_BODY_REFERENCE_ATTR,
) -> torch.Tensor:
    """Terminate after the target success pose is held for consecutive policy steps."""

    success = target_pose_success_mask(
        env,
        command_name=command_name,
        position_threshold=position_threshold,
        heading_threshold=heading_threshold,
        lin_vel_threshold=lin_vel_threshold,
        yaw_vel_threshold=yaw_vel_threshold,
        mean_joint_threshold=mean_joint_threshold,
        key_body_mean_threshold=key_body_mean_threshold,
        asset_cfg=asset_cfg,
        key_body_asset_cfg=key_body_asset_cfg,
        key_body_reference_attr=key_body_reference_attr,
    )

    hold_counts_by_command = getattr(env, "_to_target_success_hold_counts", None)
    if hold_counts_by_command is None:
        hold_counts_by_command = {}
        setattr(env, "_to_target_success_hold_counts", hold_counts_by_command)

    hold_count = hold_counts_by_command.get(command_name)
    if hold_count is None or hold_count.shape != success.shape or hold_count.device != success.device:
        hold_count = torch.zeros_like(success, dtype=torch.long)

    if hasattr(env, "episode_length_buf"):
        hold_count = torch.where(env.episode_length_buf <= 1, torch.zeros_like(hold_count), hold_count)
    hold_count = torch.where(success, hold_count + 1, torch.zeros_like(hold_count))
    hold_counts_by_command[command_name] = hold_count

    done = hold_count >= max(int(hold_steps), 1)
    setattr(env, "_to_target_success_done", done.detach().clone())
    setattr(env, "_to_target_success_hold_count", hold_count.detach().clone())
    return done
