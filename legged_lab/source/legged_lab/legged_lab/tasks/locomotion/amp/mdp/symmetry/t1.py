"""Functions to specify symmetry transforms for T1 AMP policy/action spaces.

T1 joint order (URDF declaration order, as used by IsaacLab):
  0 AAHead_yaw          1 Head_pitch
  2 Left_Shoulder_Pitch  3 Left_Shoulder_Roll  4 Left_Elbow_Pitch
  5 Left_Elbow_Yaw       6 Left_Wrist_Pitch    7 Left_Wrist_Yaw
  8 Left_Hand_Roll
  9 Right_Shoulder_Pitch 10 Right_Shoulder_Roll 11 Right_Elbow_Pitch
 12 Right_Elbow_Yaw      13 Right_Wrist_Pitch   14 Right_Wrist_Yaw
 15 Right_Hand_Roll
 16 Waist
 17 Left_Hip_Pitch  18 Left_Hip_Roll  19 Left_Hip_Yaw
 20 Left_Knee_Pitch 21 Left_Ankle_Pitch 22 Left_Ankle_Roll
 23 Right_Hip_Pitch 24 Right_Hip_Roll 25 Right_Hip_Yaw
 26 Right_Knee_Pitch 27 Right_Ankle_Pitch 28 Right_Ankle_Roll

Supported policy layouts:
    - 29-DoF (legacy): [base(9), joint_pos(29), joint_vel(29), actions(29)] → 96
    - 27-DoF (no-head): [base(9), joint_pos(27), joint_vel(27), actions(27)] → 90
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions with a left-right symmetry transformation."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the policy observation tensor.

        Policy obs layout:
            - 96 dims: [base(9), joint_pos(29), joint_vel(29), actions(29)]
            - 90 dims: [base(9), joint_pos(27), joint_vel(27), actions(27)]
    """
    obs = obs.clone()
    device = obs.device

    if obs.shape[-1] == 96:
        joint_num = 29
        # base_ang_vel:  negate roll(x) and yaw(z) components, keep pitch(y)
        obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
        # projected_gravity: negate y component
        obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
        # velocity_commands: negate vy and yaw
        obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
        # joint_pos, joint_vel, actions
        obs[:, 9 : 9 + joint_num] = _switch_t1_29dof_joints_left_right(obs[:, 9 : 9 + joint_num])
        obs[:, 38 : 38 + joint_num] = _switch_t1_29dof_joints_left_right(obs[:, 38 : 38 + joint_num])
        obs[:, 67 : 67 + joint_num] = _switch_t1_29dof_joints_left_right(obs[:, 67 : 67 + joint_num])
        return obs

    if obs.shape[-1] == 90:
        joint_num = 27
        # base_ang_vel:  negate roll(x) and yaw(z) components, keep pitch(y)
        obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
        # projected_gravity: negate y component
        obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
        # velocity_commands: negate vy and yaw
        obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
        # joint_pos, joint_vel, actions
        obs[:, 9 : 9 + joint_num] = _switch_t1_27dof_joints_left_right(obs[:, 9 : 9 + joint_num])
        obs[:, 36 : 36 + joint_num] = _switch_t1_27dof_joints_left_right(obs[:, 36 : 36 + joint_num])
        obs[:, 63 : 63 + joint_num] = _switch_t1_27dof_joints_left_right(obs[:, 63 : 63 + joint_num])
        return obs

    raise ValueError(f"Unexpected policy obs dimension: {obs.shape[-1]}. Expected 96 or 90.")


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the actions tensor."""
    actions = actions.clone()
    if actions.shape[-1] == 29:
        actions[:] = _switch_t1_29dof_joints_left_right(actions[:])
    elif actions.shape[-1] == 27:
        actions[:] = _switch_t1_27dof_joints_left_right(actions[:])
    else:
        raise ValueError(f"Unexpected action dimension: {actions.shape[-1]}. Expected 29 or 27.")
    return actions


def _switch_t1_27dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right for no-head 27-DoF order used by AMP control.

    Order:
      0..6   left arm
      7..13  right arm
      14     waist
      15..20 left leg
      21..26 right leg
    """
    joint_data_switched = torch.zeros_like(joint_data)

    # Waist (14): yaw about z, so it flips under a left-right mirror.
    joint_data_switched[..., 14] = -joint_data[..., 14]

    # Arms: left=[0..6], right=[7..13]
    left_arm = [0, 1, 2, 3, 4, 5, 6]
    right_arm = [7, 8, 9, 10, 11, 12, 13]
    # Legs: left=[15..20], right=[21..26]
    left_leg = [15, 16, 17, 18, 19, 20]
    right_leg = [21, 22, 23, 24, 25, 26]

    joint_data_switched[..., left_arm] = joint_data[..., right_arm]
    joint_data_switched[..., right_arm] = joint_data[..., left_arm]
    joint_data_switched[..., left_leg] = joint_data[..., right_leg]
    joint_data_switched[..., right_leg] = joint_data[..., left_leg]

    # Sign flips after swap
    # Roll axes: Shoulder Roll (1,8), Hand Roll (6,13), Hip Roll (16,22), Ankle Roll (20,26)
    roll_indices = [1, 8, 6, 13, 16, 22, 20, 26]
    # Yaw axes:  Elbow Yaw (3,10), Wrist Yaw (5,12), Hip Yaw (17,23)
    yaw_indices = [3, 10, 5, 12, 17, 23]

    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    return joint_data_switched


def _switch_t1_29dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints and apply axis-sign corrections for L-R mirror symmetry.

    Under a left-right mirror (negate Y axis):
    - Pitch joints: no sign change when swapped
    - Roll joints: sign flip when swapped   (rotation axis stays same, direction inverts)
    - Yaw  joints: sign flip when swapped
    - Head yaw (unpaired): sign flip, no swap
    - Head pitch (unpaired): no flip, no swap
    - Waist (unpaired, pitch-like): no flip, no swap
    """
    joint_data_switched = torch.zeros_like(joint_data)

    # --- Unpaired joints ---
    # Head yaw (0): flip sign (turning left <-> right)
    joint_data_switched[..., 0] = -joint_data[..., 0]
    # Head pitch (1): unchanged
    joint_data_switched[..., 1] = joint_data[..., 1]
    # Waist (16): yaw about z, so it flips under a left-right mirror.
    joint_data_switched[..., 16] = -joint_data[..., 16]

    # --- Paired joints: swap left <-> right ---
    # Arms: left=[2..8], right=[9..15]  (same relative order)
    left_arm = [2, 3, 4, 5, 6, 7, 8]
    right_arm = [9, 10, 11, 12, 13, 14, 15]
    # Legs: left=[17..22], right=[23..28]
    left_leg = [17, 18, 19, 20, 21, 22]
    right_leg = [23, 24, 25, 26, 27, 28]

    joint_data_switched[..., left_arm] = joint_data[..., right_arm]
    joint_data_switched[..., right_arm] = joint_data[..., left_arm]
    joint_data_switched[..., left_leg] = joint_data[..., right_leg]
    joint_data_switched[..., right_leg] = joint_data[..., left_leg]

    # --- Sign flips AFTER swap (indices in the *output* tensor) ---
    # Roll axes: Shoulder Roll (3,10), Hand Roll (8,15), Hip Roll (18,24), Ankle Roll (22,28)
    roll_indices = [3, 10, 8, 15, 18, 24, 22, 28]
    # Yaw axes:  Elbow Yaw (5,12), Wrist Yaw (7,14), Hip Yaw (19,25)
    yaw_indices = [5, 12, 7, 14, 19, 25]

    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    return joint_data_switched
