
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal
import random

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAmpEnv


def cache_default_key_body_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_attr: str = "_to_target_default_key_body_offsets_b",
):
    """Cache root-yaw-frame key-body offsets for the robot's startup/default pose."""
    del env_ids  # Startup events receive this argument by convention; the cache covers every environment.
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    body_ids = list(asset_cfg.body_ids)
    if not body_ids:
        raise ValueError("cache_default_key_body_offsets requires at least one body id.")

    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    root_yaw_quat_w = math_utils.yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(body_ids), -1)
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    body_offsets_b = math_utils.quat_apply_inverse(root_yaw_quat_w, body_pos_w - root_pos_w)
    setattr(env, reference_attr, body_offsets_b.detach().clone())


def ref_state_init_root(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_rsi: bool = True,
    motion_dataset: str | None = None,
    height_offset: float = 0.05,
):
    """Reference State Initialization (RSI) for the root of the robot.
    Sample from the motion loader and set the root position and orientation.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_state_dict = _sample_motion_state(env, env_ids.shape[0], motion_dataset)
    motion_state_dict["root_pos_w"][:, 2] += height_offset
    
    if not pos_rsi:
        motion_state_dict["root_pos_w"][:, :2] = 0.0    # no offset in x and y
    ref_root_pos_w = motion_state_dict["root_pos_w"] + env.scene.env_origins[env_ids]
    ref_root_quat = motion_state_dict["root_quat"]
    ref_root_vel_w = motion_state_dict["root_vel_w"]
    ref_root_ang_vel_w = motion_state_dict["root_ang_vel_w"]
    
    asset.write_root_pose_to_sim(
        torch.cat([ref_root_pos_w, ref_root_quat], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim(
        torch.cat([ref_root_vel_w, ref_root_ang_vel_w], dim=-1),
        env_ids=env_ids,
    )
    

def ref_state_init_dof(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_dataset: str | None = None,
):
    """Reference State Initialization (RSI) for the joints (DoF) of the robot.
    Sample from the motion loader and set the joint positions and velocities.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_state_dict = _sample_motion_state(env, env_ids.shape[0], motion_dataset)

    joint_pos, joint_vel, joint_ids = _select_motion_joint_state(asset, asset_cfg, env_ids, motion_state_dict)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def ref_state_init_subset(
    env: ManagerBasedAmpEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rsi_ratio: float = 1.0,
    pos_rsi: bool = True,
    motion_dataset: str | None = None,
    height_offset: float = 0.05,
):
    """Apply reference-state initialization to a configurable subset of reset envs."""

    if env_ids.numel() == 0 or rsi_ratio <= 0.0:
        return

    rsi_ratio = min(float(rsi_ratio), 1.0)
    num_envs = env_ids.shape[0]
    num_rsi_envs = max(1, int(round(num_envs * rsi_ratio)))
    random_order = torch.randperm(num_envs, device=env_ids.device)
    rsi_env_ids = env_ids[random_order[:num_rsi_envs]]

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_state_dict = _sample_motion_state(env, rsi_env_ids.shape[0], motion_dataset)
    motion_state_dict["root_pos_w"][:, 2] += height_offset

    if not pos_rsi:
        motion_state_dict["root_pos_w"][:, :2] = 0.0
    ref_root_pos_w = motion_state_dict["root_pos_w"] + env.scene.env_origins[rsi_env_ids]
    asset.write_root_pose_to_sim(
        torch.cat([ref_root_pos_w, motion_state_dict["root_quat"]], dim=-1),
        env_ids=rsi_env_ids,
    )
    asset.write_root_velocity_to_sim(
        torch.cat([motion_state_dict["root_vel_w"], motion_state_dict["root_ang_vel_w"]], dim=-1),
        env_ids=rsi_env_ids,
    )

    joint_pos, joint_vel, joint_ids = _select_motion_joint_state(asset, asset_cfg, rsi_env_ids, motion_state_dict)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=rsi_env_ids)


def _select_motion_joint_state(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
    motion_state_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[int] | torch.Tensor | None]:
    """Return motion joint state aligned to the configured asset joints."""

    joint_pos = motion_state_dict["dof_pos"]
    joint_vel = motion_state_dict["dof_vel"]
    joint_ids = asset_cfg.joint_ids

    if isinstance(joint_ids, slice):
        joint_ids = None

    if joint_ids is not None:
        num_motion_joints = joint_pos.shape[-1]
        num_selected_joints = len(joint_ids)
        num_asset_joints = asset.data.joint_pos.shape[-1]
        if num_motion_joints == num_selected_joints:
            pass
        elif num_motion_joints == num_asset_joints:
            joint_pos = joint_pos[:, joint_ids]
            joint_vel = joint_vel[:, joint_ids]
        else:
            raise ValueError(
                "Reference motion DoF count does not match the configured RSI joint subset: "
                f"motion={num_motion_joints}, selected={num_selected_joints}, asset={num_asset_joints}."
            )

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    if joint_ids is not None:
        joint_pos_limits = joint_pos_limits[:, joint_ids]
        joint_vel_limits = joint_vel_limits[:, joint_ids]

    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    return joint_pos, joint_vel, joint_ids


def _sample_motion_state(env: ManagerBasedAmpEnv, num_envs: int, motion_dataset: str | None = None) -> dict[str, torch.Tensor]:
    dt = env.cfg.sim.dt * env.cfg.decimation
    if motion_dataset is None:
        term_weights = env.motion_data_manager.get_term_weights()
        motion_dataset = random.choices(list(term_weights.keys()), weights=list(term_weights.values()))[0]
    else:
        active_terms = env.motion_data_manager.active_terms
        if callable(active_terms):
            active_terms = active_terms()
        if motion_dataset not in active_terms:
            raise ValueError(f"Motion dataset '{motion_dataset}' not found in the active terms.")

    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time_end=dt)
    return motion_loader.get_motion_state(motion_ids, motion_times)
