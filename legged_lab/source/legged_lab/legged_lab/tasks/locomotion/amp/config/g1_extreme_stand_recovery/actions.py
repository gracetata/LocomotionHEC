"""Action terms dedicated to low-jerk Extreme Stand recovery."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class StableDampedJointPositionAction(JointPositionAction):
    r"""Preserve V4 targets and add damping only after recovery has settled.

    For an implicit PD drive, adding ``-g Kd qdot`` to torque is exactly
    equivalent to sending ``q_target - g (Kd/Kp) qdot`` as its position target.
    The correction therefore suppresses physical high-frequency motion without
    phase-lagging or integrating the actor's recovery command.  It is latched
    only after a short stable window and fades out quickly on a disturbance.
    """

    cfg: "StableDampedJointPositionActionCfg"

    def __init__(self, cfg: "StableDampedJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._control_dt = float(env.step_dt)
        if self._control_dt <= 0.0:
            raise ValueError("Extreme Stand stable damping requires a positive control dt.")
        if cfg.stable_steps_before_damping <= 0:
            raise ValueError("Stable steps before damping must be positive.")
        if min(
            cfg.damping_gain,
            cfg.initial_damping_gain,
            cfg.blend_in_time_s,
            cfg.blend_out_time_s,
            cfg.initial_gain_duration_s,
        ) < 0.0:
            raise ValueError("Stable damping gains and times must be non-negative.")
        if not 0.0 <= cfg.recovery_damping_blend <= 1.0:
            raise ValueError("Recovery damping blend must be in [0, 1].")
        if not 0.0 < cfg.severe_disturbance_gravity_xy <= 1.0:
            raise ValueError("Severe-disturbance gravity threshold must be in (0, 1].")

        self._stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._damping_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._damping_blend = torch.zeros(self.num_envs, device=self.device)
        self._stable_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._severe_disturbance_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        desired_position = self._processed_actions.clone()

        gravity_xy = torch.linalg.vector_norm(
            self._asset.data.projected_gravity_b[:, :2], dim=1
        )
        linear_speed = torch.linalg.vector_norm(self._asset.data.root_lin_vel_b, dim=1)
        angular_speed = torch.linalg.vector_norm(self._asset.data.root_ang_vel_b, dim=1)
        stable_now = (
            (gravity_xy <= self.cfg.upright_gravity_xy_max)
            & (linear_speed <= self.cfg.root_linear_speed_max)
            & (angular_speed <= self.cfg.root_angular_speed_max)
        )
        self._stable_steps = torch.where(
            stable_now, self._stable_steps + 1, torch.zeros_like(self._stable_steps)
        )
        self._damping_latched |= self._stable_steps >= self.cfg.stable_steps_before_damping
        self._severe_disturbance_latched |= (
            self._damping_latched & (gravity_xy >= self.cfg.severe_disturbance_gravity_xy)
        )

        damping_now = self._damping_latched & stable_now
        self._stable_elapsed = torch.where(
            damping_now,
            self._stable_elapsed + self._control_dt,
            torch.zeros_like(self._stable_elapsed),
        )
        blend_target = torch.where(
            damping_now,
            torch.ones_like(self._damping_blend),
            torch.where(
                self._damping_latched & self._severe_disturbance_latched,
                torch.full_like(self._damping_blend, self.cfg.recovery_damping_blend),
                torch.zeros_like(self._damping_blend),
            ),
        )
        blend_time = torch.where(
            blend_target > self._damping_blend,
            torch.full_like(self._damping_blend, self.cfg.blend_in_time_s),
            torch.full_like(self._damping_blend, self.cfg.blend_out_time_s),
        )
        max_blend_step = self._control_dt / torch.clamp(blend_time, min=self._control_dt)
        self._damping_blend += torch.clamp(
            blend_target - self._damping_blend,
            min=-max_blend_step,
            max=max_blend_step,
        )

        damping_gain = torch.where(
            self._stable_elapsed < self.cfg.initial_gain_duration_s,
            torch.full_like(self._stable_elapsed, self.cfg.initial_damping_gain),
            torch.full_like(self._stable_elapsed, self.cfg.damping_gain),
        )
        stiffness = self._asset.data.joint_stiffness[:, self._joint_ids]
        damping = self._asset.data.joint_damping[:, self._joint_ids]
        joint_velocity = self._asset.data.joint_vel[:, self._joint_ids]
        target_correction = (
            damping_gain.unsqueeze(1)
            * self._damping_blend.unsqueeze(1)
            * damping
            / torch.clamp(stiffness, min=1.0e-6)
            * joint_velocity
        )
        self._processed_actions = desired_position - target_correction

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._stable_steps[env_ids] = 0
        self._damping_latched[env_ids] = False
        self._damping_blend[env_ids] = 0.0
        self._stable_elapsed[env_ids] = 0.0
        self._severe_disturbance_latched[env_ids] = False


@configclass
class StableDampedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`StableDampedJointPositionAction`."""

    class_type: type[ActionTerm] = StableDampedJointPositionAction

    upright_gravity_xy_max: float = 0.20
    root_linear_speed_max: float = 0.60
    root_angular_speed_max: float = 1.20
    stable_steps_before_damping: int = 10
    initial_damping_gain: float = 1.1
    initial_gain_duration_s: float = 3.0
    damping_gain: float = 1.05
    blend_in_time_s: float = 0.50
    blend_out_time_s: float = 0.05
    recovery_damping_blend: float = 0.25
    severe_disturbance_gravity_xy: float = 0.20
