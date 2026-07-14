"""AMP manager-based environment with TensorBoard torso/IMU gait metrics.

Core class:
    ManagerBasedAmpEnv wraps IsaacLab's ManagerBasedRLEnv for AMP training and
    preserves discriminator observations around reset boundaries.

Important metric output:
    step(action) returns the normal RSL-RL tuple and also writes scalar tensors
    under extras["log"]["Important Metrics/..."] for TensorBoard. Each field is
    an error from the task-perfect value and carries units in the metric name:
    torso roll/pitch, command tracking, vertical drift, angular drift, IMU
    specific-force bias, and task-aligned lateral path ratio. They are
    diagnostics only and do not change the reward.

Usage:
    gym.make("LeggedLab-Isaac-AMP-G1-v0", cfg=env_cfg) creates this class via
    the task registry. RSL-RL logs the extras dictionary during train/play.
"""

from __future__ import annotations

import torch
from typing import Any
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
from isaaclab.managers import ActionManager, ObservationManager, RecorderManager, CommandManager, CurriculumManager, RewardManager, TerminationManager
import isaaclab.utils.math as math_utils

from legged_lab.managers import MotionDataManager, AnimationManager
from .manager_based_animation_env import ManagerBasedAnimationEnv
from .manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg

GRAVITY_ACCELERATION_M_PER_S2 = 9.80665

class ManagerBasedAmpEnv(ManagerBasedAnimationEnv):
    
    """AMP Environment for locomotion tasks.

    This class inherits from the `ManagerBasedRLEnv` class and is used to create an environment for
    training and testing reinforcement learning agents on locomotion tasks using the AMP.
    
    In the original ManagerBasedRLEnv's `step` method, observations are lost if the environments are 
    reset. But in AMP we should record the observations before resetting the environments.
    This class overrides the `step` method to ensure that observations are retained even when
    environments are reset.
    """

    cfg: ManagerBasedAmpEnvCfg
    
    def __init__(self, cfg: ManagerBasedAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._important_metric_torso_body_id: int | None = None
        self._important_metric_prev_torso_pos_w: torch.Tensor | None = None
        self._important_metric_torso_height_target_m: torch.Tensor | None = None
        self._important_metric_torso_lateral_path_m: torch.Tensor | None = None
        self._important_metric_torso_forward_path_m: torch.Tensor | None = None
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.
        
        This function is almost identical to the parent class, except that:
            In the parent class's method, the observations are computed after the reset, which leads to
            the loss of observations for the reset environments. In this class, we compute the AMP observations
            before the reset and update the observations for the reset environments. 

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
            The AMP observations are included in the observations dictionary under the key "amp".
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update animation manager
        self.animation_manager.update(dt=self.step_dt)
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._compute_and_log_important_metrics()
        # # -- update AMP observations
        # amp_obs = self._get_amp_observations()

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            self._reset_important_metric_buffers(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        # if len(reset_env_ids) > 0:
        #     self.obs_buf["amp"][reset_env_ids] = amp_obs[reset_env_ids]
        
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _compute_and_log_important_metrics(self) -> None:
        """Compute torso-mounted IMU diagnostic errors and attach them to RSL-RL extras."""

        try:
            robot = self.scene["robot"]
        except KeyError:
            return

        torso_body_id = self._get_torso_body_id(robot)
        torso_pos_w = robot.data.body_pos_w[:, torso_body_id, :]
        torso_quat_w = robot.data.body_quat_w[:, torso_body_id, :]
        torso_lin_vel_w = robot.data.body_lin_vel_w[:, torso_body_id, :]
        torso_ang_vel_w = robot.data.body_ang_vel_w[:, torso_body_id, :]
        torso_lin_acc_w = robot.data.body_lin_acc_w[:, torso_body_id, :]
        torso_ang_acc_w = robot.data.body_ang_acc_w[:, torso_body_id, :]
        command = self._get_base_velocity_command()

        if self._important_metric_prev_torso_pos_w is None:
            self._initialize_important_metric_buffers(robot)

        assert self._important_metric_prev_torso_pos_w is not None
        assert self._important_metric_torso_height_target_m is not None
        assert self._important_metric_torso_lateral_path_m is not None
        assert self._important_metric_torso_forward_path_m is not None

        roll, pitch, _ = math_utils.euler_xyz_from_quat(torso_quat_w)
        yaw_quat_w = math_utils.yaw_quat(torso_quat_w)
        torso_lin_vel_yaw_b = math_utils.quat_apply_inverse(yaw_quat_w, torso_lin_vel_w)
        torso_ang_vel_b = math_utils.quat_apply_inverse(torso_quat_w, torso_ang_vel_w)
        torso_ang_acc_b = math_utils.quat_apply_inverse(torso_quat_w, torso_ang_acc_w)

        gravity_acc_w = torch.tensor(
            [0.0, 0.0, -GRAVITY_ACCELERATION_M_PER_S2], dtype=torso_lin_acc_w.dtype, device=self.device
        )
        specific_force_b = math_utils.quat_apply_inverse(torso_quat_w, torso_lin_acc_w - gravity_acc_w)

        task_forward_xy = self._get_task_forward_xy(torso_quat_w, command[:, :2])
        task_lateral_xy = torch.stack((-task_forward_xy[:, 1], task_forward_xy[:, 0]), dim=1)
        torso_delta_xy = torso_pos_w[:, :2] - self._important_metric_prev_torso_pos_w[:, :2]
        self._important_metric_torso_forward_path_m += torch.abs(torch.sum(torso_delta_xy * task_forward_xy, dim=1))
        self._important_metric_torso_lateral_path_m += torch.abs(torch.sum(torso_delta_xy * task_lateral_xy, dim=1))

        log_extras = self.extras.setdefault("log", {})
        log_extras["Important Metrics/torso_roll_error_rad"] = torch.abs(math_utils.wrap_to_pi(roll)).mean()
        log_extras["Important Metrics/torso_pitch_error_rad"] = torch.abs(math_utils.wrap_to_pi(pitch)).mean()
        log_extras["Important Metrics/torso_lin_vel_xy_cmd_error_m_per_s"] = torch.linalg.norm(
            torso_lin_vel_yaw_b[:, :2] - command[:, :2], dim=1
        ).mean()
        log_extras["Important Metrics/torso_lateral_vel_cmd_error_m_per_s"] = torch.abs(
            torso_lin_vel_yaw_b[:, 1] - command[:, 1]
        ).mean()
        log_extras["Important Metrics/torso_yaw_rate_cmd_error_rad_per_s"] = torch.abs(
            torso_ang_vel_b[:, 2] - command[:, 2]
        ).mean()
        log_extras["Important Metrics/torso_vertical_vel_error_m_per_s"] = torch.abs(torso_lin_vel_w[:, 2]).mean()
        log_extras["Important Metrics/torso_height_error_m"] = torch.abs(
            torso_pos_w[:, 2] - self._important_metric_torso_height_target_m
        ).mean()
        log_extras["Important Metrics/torso_ang_vel_xy_error_rad_per_s"] = torch.linalg.norm(
            torso_ang_vel_b[:, :2], dim=1
        ).mean()
        log_extras["Important Metrics/torso_ang_acc_xy_error_rad_per_s2"] = torch.linalg.norm(
            torso_ang_acc_b[:, :2], dim=1
        ).mean()
        log_extras["Important Metrics/torso_specific_force_xy_error_m_per_s2"] = torch.linalg.norm(
            specific_force_b[:, :2], dim=1
        ).mean()
        log_extras["Important Metrics/torso_specific_force_z_error_m_per_s2"] = torch.abs(
            specific_force_b[:, 2] - GRAVITY_ACCELERATION_M_PER_S2
        ).mean()
        log_extras["Important Metrics/torso_lateral_path_ratio"] = torch.mean(
            self._important_metric_torso_lateral_path_m
            / torch.clamp(self._important_metric_torso_forward_path_m, min=1.0e-4)
        )

        self._important_metric_prev_torso_pos_w = torso_pos_w.detach().clone()

    def _initialize_important_metric_buffers(self, robot) -> None:
        torso_pos_w = robot.data.body_pos_w[:, self._get_torso_body_id(robot), :]
        self._important_metric_prev_torso_pos_w = torso_pos_w.detach().clone()
        self._important_metric_torso_height_target_m = torso_pos_w[:, 2].detach().clone()
        self._important_metric_torso_lateral_path_m = torch.zeros(self.num_envs, device=self.device)
        self._important_metric_torso_forward_path_m = torch.zeros(self.num_envs, device=self.device)

    def _reset_important_metric_buffers(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if self._important_metric_prev_torso_pos_w is None:
            return
        try:
            robot = self.scene["robot"]
        except KeyError:
            return

        assert self._important_metric_torso_height_target_m is not None
        assert self._important_metric_torso_lateral_path_m is not None
        assert self._important_metric_torso_forward_path_m is not None

        torso_pos_w = robot.data.body_pos_w[:, self._get_torso_body_id(robot), :]
        self._important_metric_prev_torso_pos_w[env_ids] = torso_pos_w[env_ids]
        self._important_metric_torso_height_target_m[env_ids] = torso_pos_w[env_ids, 2]
        self._important_metric_torso_lateral_path_m[env_ids] = 0.0
        self._important_metric_torso_forward_path_m[env_ids] = 0.0

    def _get_base_velocity_command(self) -> torch.Tensor:
        try:
            return self.command_manager.get_command("base_velocity")
        except Exception:
            return torch.zeros(self.num_envs, 3, device=self.device)

    def _get_task_forward_xy(self, torso_quat_w: torch.Tensor, command_xy: torch.Tensor) -> torch.Tensor:
        command_norm = torch.linalg.norm(command_xy, dim=1, keepdim=True)
        base_forward = torch.tensor([1.0, 0.0, 0.0], dtype=torso_quat_w.dtype, device=self.device).expand(self.num_envs, -1)
        base_forward_xy = math_utils.quat_apply(torso_quat_w, base_forward)[:, :2]
        base_forward_xy = base_forward_xy / torch.clamp(
            torch.linalg.norm(base_forward_xy, dim=1, keepdim=True), min=1.0e-6
        )
        command_forward_xy = command_xy / torch.clamp(command_norm, min=1.0e-6)
        return torch.where(command_norm > 0.05, command_forward_xy, base_forward_xy)

    def _get_torso_body_id(self, robot) -> int:
        if self._important_metric_torso_body_id is None:
            body_names = list(getattr(robot, "body_names", []))
            try:
                self._important_metric_torso_body_id = body_names.index("torso_link")
            except ValueError:
                self._important_metric_torso_body_id = 0
        return self._important_metric_torso_body_id

