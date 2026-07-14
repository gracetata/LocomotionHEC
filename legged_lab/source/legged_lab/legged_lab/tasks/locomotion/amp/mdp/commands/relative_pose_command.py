"""Relative SE(2) target commands for short-range G1 precise positioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelativePose2dCommand(CommandTerm):
    """Sample a fixed target relative to the reset SE(2) pose.

    The policy-facing command remains 3-D for checkpoint compatibility:
    ``[target_dx_b, target_dy_b, target_dyaw]``. The XY terms are recomputed in
    the current yaw frame each step while the world-frame target stays fixed.
    """

    cfg: RelativePose2dCommandCfg

    def __init__(self, cfg: RelativePose2dCommandCfg, env: ManagerBasedEnv):
        self.robot: Articulation = env.scene[cfg.asset_name]
        super().__init__(cfg, env)

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.target_delta_initial = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.stop_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_lin_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_radius"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_latched"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["left_stop"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "RelativePose2dCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRadius range: {self.cfg.ranges.radius}\n"
        msg += f"\tHeading range: {self.cfg.ranges.heading}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Target pose error in the current base-yaw frame. Shape is ``(num_envs, 3)``."""
        return self.pose_command_b

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        resolved_env_ids = self._resolve_env_ids(env_ids)
        self.stop_latched[resolved_env_ids] = False
        return super().reset(resolved_env_ids)

    def _update_metrics(self):
        distance = torch.norm(self.target_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        heading_error = torch.abs(math_utils.wrap_to_pi(self.target_heading_w - self.robot.data.heading_w))
        lin_speed = torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_rate = torch.abs(self.robot.data.root_ang_vel_b[:, 2])
        left_stop = self.stop_latched & (
            (distance > float(self.cfg.stop_exit_position_threshold))
            | (heading_error > float(self.cfg.stop_exit_heading_threshold))
        )
        success = (
            (distance < float(self.cfg.success_position_threshold))
            & (heading_error < float(self.cfg.success_heading_threshold))
            & (lin_speed < float(self.cfg.success_lin_vel_threshold))
            & (yaw_rate < float(self.cfg.success_yaw_vel_threshold))
        )

        self.metrics["error_pos_2d"][:] = distance
        self.metrics["error_heading"][:] = heading_error
        self.metrics["error_lin_vel_xy"][:] = lin_speed
        self.metrics["error_yaw_rate"][:] = yaw_rate
        self.metrics["target_radius"][:] = torch.norm(self.target_delta_initial[:, :2], dim=1)
        self.metrics["stop_latched"][:] = self.stop_latched.float()
        self.metrics["left_stop"][:] = left_stop.float()
        self.metrics["success"][:] = success.float()

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._resolve_env_ids(env_ids)
        count = int(env_ids_tensor.numel())
        if count == 0:
            return

        radius_min, radius_max = self.cfg.ranges.radius
        radius_min = max(float(radius_min), 0.0)
        radius_max = max(float(radius_max), radius_min)
        radius = torch.sqrt(
            torch.empty(count, device=self.device).uniform_(radius_min * radius_min, radius_max * radius_max)
        )
        angle = torch.empty(count, device=self.device).uniform_(-torch.pi, torch.pi)
        delta_x = radius * torch.cos(angle)
        delta_y = radius * torch.sin(angle)
        heading_delta = torch.empty(count, device=self.device).uniform_(*self.cfg.ranges.heading)

        current_pos_w = self.robot.data.root_pos_w[env_ids_tensor]
        current_heading = self.robot.data.heading_w[env_ids_tensor]
        cos_yaw = torch.cos(current_heading)
        sin_yaw = torch.sin(current_heading)
        delta_w_x = cos_yaw * delta_x - sin_yaw * delta_y
        delta_w_y = sin_yaw * delta_x + cos_yaw * delta_y

        self.target_pos_w[env_ids_tensor, :2] = current_pos_w[:, :2] + torch.stack([delta_w_x, delta_w_y], dim=1)
        self.target_pos_w[env_ids_tensor, 2] = current_pos_w[:, 2]
        self.target_heading_w[env_ids_tensor] = math_utils.wrap_to_pi(current_heading + heading_delta)
        self.target_delta_initial[env_ids_tensor, :] = torch.stack([delta_x, delta_y, heading_delta], dim=1)
        self._update_command()

    def _update_command(self):
        target_vec_w = self.target_pos_w - self.robot.data.root_pos_w[:, :3]
        target_vec_w[:, 2] = 0.0
        target_vec_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w), target_vec_w)
        self.pose_command_b[:, :2] = target_vec_b[:, :2]
        self.pose_command_b[:, 2] = math_utils.wrap_to_pi(self.target_heading_w - self.robot.data.heading_w)
        self._update_stop_latch()

    def _update_stop_latch(self):
        position_error = torch.norm(self.pose_command_b[:, :2], dim=1)
        heading_error = torch.abs(math_utils.wrap_to_pi(self.pose_command_b[:, 2]))
        entered_stop = (
            (position_error < float(self.cfg.stop_latch_position_threshold))
            & (heading_error < float(self.cfg.stop_latch_heading_threshold))
        )
        self.stop_latched[:] = self.stop_latched | entered_stop

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_pose_visualizer"):
                self.target_pose_visualizer = VisualizationMarkers(self.cfg.target_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.target_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_pose_visualizer"):
                self.target_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        zeros = torch.zeros_like(self.target_heading_w)
        marker_height = float(self.cfg.target_root_height_offset)
        target_pos_w = self.target_pos_w.clone()
        target_pos_w[:, 2] += marker_height
        current_pos_w = self.robot.data.root_pos_w.clone()
        current_pos_w[:, 2] += marker_height

        self.target_pose_visualizer.visualize(
            translations=target_pos_w,
            orientations=math_utils.quat_from_euler_xyz(zeros, zeros, self.target_heading_w),
        )
        self.current_pose_visualizer.visualize(
            translations=current_pos_w,
            orientations=math_utils.quat_from_euler_xyz(zeros, zeros, self.robot.data.heading_w),
        )


@configclass
class RelativePose2dCommandCfg(CommandTermCfg):
    """Configuration for reset-relative 2-D pose target commands."""

    class_type: type = RelativePose2dCommand

    asset_name: str = MISSING
    """Name of the robot articulation."""

    target_asset_name: str | None = None
    """Deprecated compatibility field. Target debug visualization now uses markers, not an articulation."""

    target_root_height_offset: float = 0.0
    """Additional visual-only height offset applied to target/current pose arrows."""

    target_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/to_target_goal_pose"
    )
    """Marker used to visualize the sampled target pose."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/to_target_current_pose"
    )
    """Marker used to visualize the current robot base pose."""

    target_pose_visualizer_cfg.markers["arrow"].scale = (0.45, 0.45, 0.45)
    current_pose_visualizer_cfg.markers["arrow"].scale = (0.35, 0.35, 0.35)

    @configclass
    class Ranges:
        """Uniform target sampling ranges."""

        radius: tuple[float, float] = (0.0, 1.0)
        """Target distance from the initial root position in meters."""

        heading: tuple[float, float] = (-torch.pi, torch.pi)
        """Target yaw offset from the initial root heading in radians."""

    ranges: Ranges = MISSING

    success_position_threshold: float = 0.06
    """Position tolerance used by command metrics."""

    success_heading_threshold: float = 0.12
    """Heading tolerance used by command metrics."""

    success_lin_vel_threshold: float = 0.06
    """Linear speed tolerance used by command metrics."""

    success_yaw_vel_threshold: float = 0.10
    """Yaw-rate tolerance used by command metrics."""

    stop_latch_position_threshold: float = 0.08
    """Position tolerance for entering sticky stop mode."""

    stop_latch_heading_threshold: float = 0.18
    """Heading tolerance for entering sticky stop mode."""

    stop_exit_position_threshold: float = 0.12
    """Position tolerance for logging stop-mode escape."""

    stop_exit_heading_threshold: float = 0.28
    """Heading tolerance for logging stop-mode escape."""
