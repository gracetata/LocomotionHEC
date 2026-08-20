
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


def sample_sequential_step_training_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    phase_one_probability: float = 0.0,
):
    """Select phase-one reset environments for right-step skill oversampling."""
    if not 0.0 <= float(phase_one_probability) <= 1.0:
        raise ValueError("phase_one_probability must be in [0, 1].")
    reset_phase_one = getattr(env, "_armhack_sequential_phase_one_reset", None)
    if reset_phase_one is None or reset_phase_one.shape[0] != env.num_envs:
        reset_phase_one = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, "_armhack_sequential_phase_one_reset", reset_phase_one)
    reset_phase_one[env_ids] = (
        torch.rand(env_ids.numel(), device=env.device) < float(phase_one_probability)
    )


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


def reset_joints_with_random_stance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float] = (0.08, 0.48),
    close_distance_range: tuple[float, float] = (0.08, 0.14),
    close_stance_probability: float = 0.40,
    nominal_distance_range: tuple[float, float] = (0.28, 0.32),
    nominal_stance_probability: float = 0.10,
    asymmetric_support_probability: float = 0.0,
    phase_one_probability: float = 0.0,
    phase_two_probability: float = 0.0,
    support_distance_range: tuple[float, float] = (0.20, 0.32),
    kinematic_nominal_distance: float = 0.237,
    kinematic_distance_per_rad: float = 1.22,
    position_scale_range: tuple[float, float] = (0.95, 1.05),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset G1 joints with symmetric or support-foot-wide ankle spacing.

    The sampled spacing is converted to a symmetric hip-roll angle using a
    locally calibrated G1 kinematic slope. Opposite ankle-roll angles keep
    both feet approximately parallel to the floor. Close stances are
    deliberately oversampled to cover the difficult feet-together case.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids.numel() == 0:
        return

    distance_min, distance_max = (float(value) for value in distance_range)
    close_min, close_max = (float(value) for value in close_distance_range)
    nominal_min, nominal_max = (float(value) for value in nominal_distance_range)
    close_probability = float(close_stance_probability)
    nominal_probability = float(nominal_stance_probability)
    asymmetric_probability = float(asymmetric_support_probability)
    phase_one_probability = float(phase_one_probability)
    phase_two_probability = float(phase_two_probability)
    support_distance_min, support_distance_max = (float(value) for value in support_distance_range)
    if not 0.0 < distance_min <= distance_max:
        raise ValueError("distance_range must satisfy 0 < min <= max.")
    if not distance_min <= close_min <= close_max <= distance_max:
        raise ValueError("close_distance_range must be contained in distance_range.")
    if not distance_min <= nominal_min <= nominal_max <= distance_max:
        raise ValueError("nominal_distance_range must be contained in distance_range.")
    if close_probability < 0.0 or nominal_probability < 0.0 or close_probability + nominal_probability > 1.0:
        raise ValueError("Stance mixture probabilities must be non-negative and sum to <= 1.")
    if not 0.0 <= asymmetric_probability <= 1.0:
        raise ValueError("asymmetric_support_probability must be in [0, 1].")
    if not 0.0 <= phase_one_probability <= 1.0:
        raise ValueError("phase_one_probability must be in [0, 1].")
    if not 0.0 <= phase_two_probability <= 1.0 or phase_one_probability + phase_two_probability > 1.0:
        raise ValueError("phase reset probabilities must be non-negative and sum to <= 1.")
    if not 0.0 < support_distance_min <= support_distance_max:
        raise ValueError("support_distance_range must satisfy 0 < min <= max.")
    if float(kinematic_distance_per_rad) <= 0.0:
        raise ValueError("kinematic_distance_per_rad must be positive.")

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos *= math_utils.sample_uniform(*position_scale_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    sample_count = env_ids.numel()
    mixture = torch.rand(sample_count, device=asset.device)
    sampled_distance = math_utils.sample_uniform(distance_min, distance_max, (sample_count,), asset.device)
    close_distance = math_utils.sample_uniform(close_min, close_max, (sample_count,), asset.device)
    nominal_distance = math_utils.sample_uniform(nominal_min, nominal_max, (sample_count,), asset.device)
    sampled_distance = torch.where(mixture < close_probability, close_distance, sampled_distance)
    sampled_distance = torch.where(
        (mixture >= close_probability) & (mixture < close_probability + nominal_probability),
        nominal_distance,
        sampled_distance,
    )

    roll_angle = (sampled_distance - float(kinematic_nominal_distance)) / float(kinematic_distance_per_rad)
    phase_draw = torch.rand(sample_count, device=asset.device)
    phase_two_mask = phase_draw < phase_two_probability
    phase_one_mask = (phase_draw >= phase_two_probability) & (
        phase_draw < phase_two_probability + phase_one_probability
    )
    reset_phase = getattr(env, "_armhack_sequential_reset_phase", None)
    if reset_phase is None or reset_phase.shape[0] != env.num_envs:
        reset_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, "_armhack_sequential_reset_phase", reset_phase)
    reset_phase[env_ids] = phase_one_mask.long() + 2 * phase_two_mask.long()
    reset_phase_one = getattr(env, "_armhack_sequential_phase_one_reset", None)
    if reset_phase_one is None or reset_phase_one.shape[0] != env.num_envs:
        reset_phase_one = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, "_armhack_sequential_phase_one_reset", reset_phase_one)
    reset_phase_one[env_ids] = phase_one_mask

    asymmetric_mask = (
        (torch.rand(sample_count, device=asset.device) < asymmetric_probability) | phase_one_mask
    ) & (~phase_two_mask)
    # In phase zero the active left foot is close and the right support foot
    # is at -15 cm.  A phase-one reset must be the physical continuation of a
    # completed left step: left support at +15 cm and active right foot close.
    active_roll_angle = (close_distance - float(kinematic_nominal_distance)) / float(
        kinematic_distance_per_rad
    )
    support_distance = math_utils.sample_uniform(
        support_distance_min, support_distance_max, (sample_count,), asset.device
    )
    support_roll_angles = (support_distance - float(kinematic_nominal_distance)) / float(
        kinematic_distance_per_rad
    )
    asymmetric_left_roll = torch.where(phase_one_mask, support_roll_angles, active_roll_angle)
    asymmetric_right_roll = torch.where(phase_one_mask, active_roll_angle, support_roll_angles)
    left_roll_angle = torch.where(asymmetric_mask, asymmetric_left_roll, roll_angle)
    right_roll_angle = torch.where(
        asymmetric_mask, asymmetric_right_roll, roll_angle
    )
    final_roll_angles = torch.full_like(
        roll_angle,
        (0.30 - float(kinematic_nominal_distance)) / float(kinematic_distance_per_rad),
    )
    left_roll_angle = torch.where(phase_two_mask, final_roll_angles, left_roll_angle)
    right_roll_angle = torch.where(phase_two_mask, final_roll_angles, right_roll_angle)
    joint_index = {name: index for index, name in enumerate(asset.joint_names)}
    stance_joint_names = (
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    )
    missing = [name for name in stance_joint_names if name not in joint_index]
    if missing:
        raise ValueError(f"Random stance reset is missing G1 joints: {missing}")

    signed_targets = {
        "left_hip_roll_joint": left_roll_angle,
        "right_hip_roll_joint": -right_roll_angle,
        "left_ankle_roll_joint": -left_roll_angle,
        "right_ankle_roll_joint": right_roll_angle,
    }
    for joint_name, values in signed_targets.items():
        joint_pos[:, joint_index[joint_name]] = values

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)

    state = getattr(env, "_armhack_initial_stance_distance_m", None)
    if state is None or state.shape != (env.num_envs,) or state.device != asset.device:
        state = torch.zeros(env.num_envs, dtype=torch.float32, device=asset.device)
        setattr(env, "_armhack_initial_stance_distance_m", state)
    asymmetric_distance = 0.5 * (close_distance + support_distance)
    reset_distance = torch.where(asymmetric_mask, asymmetric_distance, sampled_distance)
    state[env_ids] = torch.where(phase_two_mask, torch.full_like(reset_distance, 0.30), reset_distance)


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
