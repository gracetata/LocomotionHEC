from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm
    


def root_local_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    
    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    
    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)
    
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local) # shape: (num_envs, num_steps, 3, 3)
    
    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)
    
    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs

def ref_root_projected_gravity(
    env: ManagerBasedAnimationEnv, 
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=ref_root_quat.device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3)
    projected_gravity = math_utils.quat_apply_inverse(
        ref_root_quat, gravity_vec.expand(num_envs, -1, -1)
    )  # shape: (num_envs, num_steps, 3)
    
    if flatten_steps_dim:
        return projected_gravity.reshape(num_envs, -1)
    else:
        return projected_gravity
    
def ray_caster(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取激光雷达（RayCaster）传感器的距离数据"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    origin = sensor.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    hits = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    distances = torch.norm(hits - origin, dim=-1).clamp(min=0.2, max=5)  # [num_envs, num_rays]
    return distances


def sequential_phase_augmented_last_action(
    env: ManagerBasedEnv,
    phase_action_index: int = 27,
    lifted_action_index: int = 28,
    se2_action_indices: tuple[int, int, int] = (24, 25, 26),
    se2_xy_scale_m: float = 0.30,
    se2_yaw_scale_rad: float = 0.50,
    se2_injection_gain: float = 1.0,
    torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Encode step phase and reset-relative torso SE(2) in scripted-arm slots.

    ArmHack replaces the commanded arm targets downstream, so these two actor
    action-history entries do not control the robot.  Reusing them keeps the
    policy input at 96 dimensions and therefore permits policy-only loading of
    all existing checkpoints.
    """
    actions = env.action_manager.action.clone()
    if phase_action_index < 0 or lifted_action_index < 0:
        raise ValueError("Sequential phase observation indices must be non-negative.")
    if phase_action_index >= actions.shape[1] or lifted_action_index >= actions.shape[1]:
        raise ValueError(
            f"Sequential phase observation indices {(phase_action_index, lifted_action_index)} "
            f"exceed action dimension {actions.shape[1]}."
        )

    state = getattr(env, "_armhack_sequential_foot_step_state", None)
    if state is None:
        phase_signal = torch.zeros(env.num_envs, device=env.device)
        lifted_signal = torch.zeros(env.num_envs, device=env.device)
    else:
        phase = state["phase"]
        active_index = state["active_index"]
        # -1 means physical left is active, +1 means physical right is active;
        # completed double support is 0.  This lets one policy execute either
        # contact-force-selected order without adding observations.
        phase_signal = torch.where(
            phase >= 2,
            torch.zeros(env.num_envs, device=env.device),
            2.0 * active_index.float() - 1.0,
        )
        lifted = torch.stack((state["left_lifted"], state["right_lifted"]), dim=1)
        lifted_signal = lifted.gather(1, active_index.unsqueeze(1)).squeeze(1).float()
    actions[:, phase_action_index] = phase_signal
    actions[:, lifted_action_index] = lifted_signal
    if len(se2_action_indices) != 3 or min(se2_action_indices) < 0 or max(se2_action_indices) >= actions.shape[1]:
        raise ValueError("SE(2) action-history indices must contain three valid action indices.")
    if se2_xy_scale_m <= 0.0 or se2_yaw_scale_rad <= 0.0:
        raise ValueError("SE(2) observation scales must be positive.")
    asset: RigidObject = env.scene[torso_cfg.name]
    if len(torso_cfg.body_ids) != 1:
        raise ValueError("SE(2) observation requires exactly one torso body.")
    body_id = torso_cfg.body_ids[0]
    position = asset.data.body_pos_w[:, body_id, :]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.body_quat_w[:, body_id, :])
    initial_position = getattr(env, "_important_metric_initial_torso_pos_w", None)
    initial_yaw = getattr(env, "_important_metric_initial_torso_yaw_w", None)
    if initial_position is None or initial_yaw is None:
        initial_position = position.detach()
        initial_yaw = yaw.detach()
    delta_xy_w = position[:, :2] - initial_position[:, :2]
    cos_yaw = torch.cos(initial_yaw)
    sin_yaw = torch.sin(initial_yaw)
    delta_x = cos_yaw * delta_xy_w[:, 0] + sin_yaw * delta_xy_w[:, 1]
    delta_y = -sin_yaw * delta_xy_w[:, 0] + cos_yaw * delta_xy_w[:, 1]
    se2_signal = torch.stack(
        (
            delta_x / float(se2_xy_scale_m),
            delta_y / float(se2_xy_scale_m),
            math_utils.wrap_to_pi(yaw - initial_yaw) / float(se2_yaw_scale_rad),
        ),
        dim=1,
    ).clamp_(-1.0, 1.0)
    actions[:, list(se2_action_indices)] += float(se2_injection_gain) * se2_signal
    return actions
