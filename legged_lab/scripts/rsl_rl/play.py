# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a Unitree G1 AMP checkpoint in IsaacLab.

Core behavior:
        Loads an RSL-RL/AMP checkpoint, runs policy inference, and prints task
        tracking plus Important Metrics emitted by ManagerBasedAmpEnv. Policy export
        is intentionally handled by scripts/export_g1_amp_policy.sh.

```bash
python scripts/rsl_rl/play.py \
        --task LeggedLab-Isaac-AMP-G1-Play-v0 \
    --num_envs 4 \
        --checkpoint logs/rsl_rl/g1_amp/<run>/model_2999.pt \
        --max_steps 200
```
Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of simulation steps to run before exiting.")
parser.add_argument("--follow_camera", action="store_true", default=False, help="Keep the play/recording camera tracking the robot.")
parser.add_argument(
    "--camera_view",
    type=str,
    default="front",
    choices=("front", "chase", "side"),
    help="Robot-relative camera view. 'front' looks at the robot from its front, 'chase' follows from behind.",
)
parser.add_argument("--camera_distance", type=float, default=3.0, help="Horizontal distance from the robot for follow camera.")
parser.add_argument("--camera_height", type=float, default=1.25, help="Follow camera eye height above the robot root.")
parser.add_argument("--camera_target_height", type=float, default=0.85, help="Height above robot root for the camera look-at target.")
parser.add_argument("--camera_lateral", type=float, default=0.0, help="Lateral offset for the follow camera, in robot-right meters.")
parser.add_argument(
    "--camera_smoothing",
    type=float,
    default=0.25,
    help="Low-pass smoothing for follow camera pose in [0, 0.95]. Higher values move the camera more slowly.",
)
parser.add_argument("--camera_env_index", type=int, default=0, help="Environment index whose robot is tracked by follow camera.")
parser.add_argument("--skip_export", action="store_true", default=False, help="Deprecated no-op; use export_g1_amp_policy.sh.")
parser.add_argument(
    "--strict_export",
    action="store_true",
    default=False,
    help="Deprecated no-op kept for script compatibility.",
)
parser.add_argument(
    "--armhack_stand_report_path",
    type=str,
    default=None,
    help="Write an ArmHack Stand per-joint fluctuation Markdown report after playback.",
)
parser.add_argument(
    "--armhack_stand_test_id",
    type=str,
    default="unspecified",
    help="Deterministic ArmHack Stand test item recorded in the report.",
)
parser.add_argument(
    "--armhack_stand_test_data",
    type=str,
    default=None,
    help="Arm-only test CSV recorded and verified in the report.",
)
parser.add_argument(
    "--armhack_stand_manifest",
    type=str,
    default=None,
    help="Deterministic ArmHack Stand manifest used for plot stage annotations.",
)
parser.add_argument(
    "--armhack_stand_payload_kg",
    type=float,
    default=0.0,
    help="Fixed added mass per wrist-yaw link recorded in the ArmHack Stand report.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import csv
import hashlib
import json
import math
import os
import time
import torch
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
import isaaclab.utils.math as math_utils
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)

TO_TARGET_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
TO_TARGET_KEY_BODY_REFERENCE_ATTR = "_to_target_default_key_body_offsets_b"



class _FollowRobotCameraWrapper(gym.Wrapper):
    """Update the viewport camera before RecordVideo captures each frame."""

    def __init__(
        self,
        env,
        view: str,
        distance: float,
        height: float,
        target_height: float,
        lateral: float,
        smoothing: float,
        env_index: int,
    ):
        super().__init__(env)
        self._view = view
        self._distance = max(float(distance), 0.01)
        self._height = float(height)
        self._target_height = float(target_height)
        self._lateral = float(lateral)
        self._smoothing = min(max(float(smoothing), 0.0), 0.95)
        self._env_index = max(int(env_index), 0)
        self._last_eye: torch.Tensor | None = None
        self._last_target: torch.Tensor | None = None
        self._warned = False

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._update_camera()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._update_camera()
        return result

    def _update_camera(self) -> None:
        try:
            base_env = self.unwrapped
            robot = base_env.scene["robot"]
            num_envs = int(robot.data.root_pos_w.shape[0])
            env_index = min(self._env_index, max(num_envs - 1, 0))

            root_pos = robot.data.root_pos_w[env_index].detach()
            root_quat = robot.data.root_quat_w[env_index].detach()
            device = root_pos.device
            dtype = root_pos.dtype

            yaw_quat = math_utils.yaw_quat(root_quat.unsqueeze(0))
            forward = math_utils.quat_apply(
                yaw_quat,
                torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype),
            )[0]
            forward = torch.stack((forward[0], forward[1], torch.zeros((), device=device, dtype=dtype)))
            forward = forward / torch.clamp(torch.linalg.norm(forward), min=1.0e-6)
            right = torch.stack((-forward[1], forward[0], torch.zeros((), device=device, dtype=dtype)))

            if self._view == "chase":
                camera_axis = -forward
            elif self._view == "side":
                camera_axis = right
            else:
                camera_axis = forward

            eye = root_pos + camera_axis * self._distance + right * self._lateral
            target = root_pos.clone()
            eye[2] = root_pos[2] + self._height
            target[2] = root_pos[2] + self._target_height

            if self._last_eye is not None and self._smoothing > 0.0:
                eye = self._smoothing * self._last_eye + (1.0 - self._smoothing) * eye
                target = self._smoothing * self._last_target + (1.0 - self._smoothing) * target

            self._last_eye = eye
            self._last_target = target
            base_env.sim.set_camera_view(
                eye=tuple(float(v) for v in eye.detach().cpu()),
                target=tuple(float(v) for v in target.detach().cpu()),
            )
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Failed to update follow camera: {exc}")
                self._warned = True

def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sum(values: list[float]) -> float:
    return float(sum(values)) if values else 0.0


def _ratio(numerators: list[float], denominators: list[float]) -> float:
    denominator = _sum(denominators)
    return _sum(numerators) / denominator if denominator > 0.0 else 0.0


def _exp_score(error: float, scale: float) -> float:
    return 100.0 * math.exp(-max(float(error), 0.0) / max(float(scale), 1.0e-6))


def _action_joint_ids(base_env) -> list[int] | slice:
    try:
        action_term = base_env.action_manager.get_term("joint_pos")
        return list(action_term._joint_ids)
    except Exception:
        return slice(None)


def _target_key_body_mean_error(base_env, robot) -> torch.Tensor | None:
    try:
        body_ids, _ = robot.find_bodies(TO_TARGET_KEY_BODY_NAMES, preserve_order=True)
    except Exception:
        return None
    if not body_ids:
        return None

    root_pos_w = robot.data.root_pos_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    root_yaw_quat_w = math_utils.yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, len(body_ids), -1)
    body_pos_w = robot.data.body_pos_w[:, body_ids, :]
    offsets = math_utils.quat_apply_inverse(root_yaw_quat_w, body_pos_w - root_pos_w)
    reference = getattr(base_env, TO_TARGET_KEY_BODY_REFERENCE_ATTR, None)
    if (
        reference is None
        or reference.shape != offsets.shape
        or reference.device != offsets.device
        or reference.dtype != offsets.dtype
    ):
        return None
    return torch.mean(torch.linalg.norm(offsets - reference, dim=-1), dim=1)


def _update_play_metrics(metrics: dict, env, rewards, dones, extras) -> None:
    metrics["reward"].append(_to_float(rewards))
    base_env = env.unwrapped
    done_tensor = dones.detach().bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=base_env.device).bool()
    metrics["done_rate"].append(_to_float(done_tensor.float()))
    metrics["done_events"].append(float(done_tensor.float().sum().detach().cpu().item()))
    try:
        commands = base_env.command_manager.get_command("base_velocity")
        robot = base_env.scene["robot"]
        lin_vel_b = robot.data.root_lin_vel_b[:, :2]
        yaw_rate = robot.data.root_ang_vel_b[:, 2]
        lin_error = torch.linalg.norm(lin_vel_b - commands[:, :2], dim=1).mean()
        yaw_error = torch.abs(yaw_rate - commands[:, 2]).mean()
        metrics["lin_vel_xy_errors"].append(_to_float(lin_error))
        metrics["yaw_rate_errors"].append(_to_float(yaw_error))
        metrics["cmd_lin_x"].append(_to_float(commands[:, 0]))
        metrics["cmd_lin_y"].append(_to_float(commands[:, 1]))
        metrics["cmd_yaw"].append(_to_float(commands[:, 2]))
        metrics["mean_lin_x"].append(_to_float(lin_vel_b[:, 0]))
        metrics["mean_lin_y"].append(_to_float(lin_vel_b[:, 1]))
        metrics["mean_yaw"].append(_to_float(yaw_rate))
        command_term = base_env.command_manager.get_term("base_velocity")
        for key, value in getattr(command_term, "metrics", {}).items():
            metrics["command"].setdefault(str(key), []).append(_to_float(value))
        joint_ids = _action_joint_ids(base_env)
        joint_error = torch.abs(robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids])
        metrics["target_joint_error"].append(_to_float(torch.mean(joint_error, dim=1)))
        key_body_error = _target_key_body_mean_error(base_env, robot)
        if key_body_error is not None:
            metrics["target_key_body_error"].append(_to_float(key_body_error))
        success_done = getattr(base_env, "_to_target_success_done", None)
        if success_done is not None:
            success_done_events = success_done.bool() & done_tensor
            metrics["target_success_done_events"].append(
                float(success_done_events.float().sum().detach().cpu().item())
            )
    except Exception:
        pass
    if isinstance(extras, dict):
        for key, value in extras.get("log", {}).items():
            if str(key).startswith("Important Metrics/"):
                metrics["important"].setdefault(str(key), []).append(_to_float(value))


def _print_play_report(metrics: dict, timestep: int) -> None:
    if not metrics["lin_vel_xy_errors"]:
        print("[METRIC] IsaacSim play metrics unavailable for this environment wrapper.")
        return
    lin_mae = _mean(metrics["lin_vel_xy_errors"])
    yaw_mae = _mean(metrics["yaw_rate_errors"])
    lin_score = _exp_score(lin_mae, 0.35)
    yaw_score = _exp_score(yaw_mae, 0.50)
    tracking_score = 0.7 * lin_score + 0.3 * yaw_score
    reward_score = max(0.0, min((_mean(metrics["reward"]) + 5.0) * 10.0, 100.0))
    total_score = 0.7 * tracking_score + 0.3 * reward_score
    print("[METRIC] IsaacSim play task tracking:")
    print(
        "  steps={steps} cmd_mean=({cmd_x:.3f}, {cmd_y:.3f}, {cmd_yaw:.3f}) "
        "vel_mean=({vel_x:.3f}, {vel_y:.3f}, {vel_yaw:.3f}) lin_vel_xy_mae={lin_mae:.3f} yaw_rate_mae={yaw_mae:.3f}".format(
            steps=timestep,
            cmd_x=_mean(metrics["cmd_lin_x"]),
            cmd_y=_mean(metrics["cmd_lin_y"]),
            cmd_yaw=_mean(metrics["cmd_yaw"]),
            vel_x=_mean(metrics["mean_lin_x"]),
            vel_y=_mean(metrics["mean_lin_y"]),
            vel_yaw=_mean(metrics["mean_yaw"]),
            lin_mae=lin_mae,
            yaw_mae=yaw_mae,
        )
    )
    print("[METRIC] IsaacSim play score:")
    print(f"  total={total_score:.1f} tracking={tracking_score:.1f} reward={reward_score:.1f} lin={lin_score:.1f} yaw={yaw_score:.1f}")
    if metrics["important"]:
        print("[METRIC] IsaacSim play Important Metrics:")
        for key in sorted(metrics["important"]):
            print(f"  {key}={_mean(metrics['important'][key]):.6f}")
    command_metrics = metrics.get("command", {})
    if "error_pos_2d" in command_metrics:
        print("[METRIC] IsaacSim ToTarget command metrics:")
        print(
            "  target_radius={radius:.3f} pos_error={pos:.3f} heading_error={heading:.3f} "
            "lin_speed={lin:.3f} yaw_rate={yaw:.3f} success={success:.3f} stop_latched={stop:.3f} "
            "left_stop={left_stop:.3f} done_rate={done:.3f} target_success_done={success_done:.3f} "
            "joint_error={joint:.3f} key_body_error={key_body:.3f}".format(
                radius=_mean(command_metrics.get("target_radius", [])),
                pos=_mean(command_metrics.get("error_pos_2d", [])),
                heading=_mean(command_metrics.get("error_heading", [])),
                lin=_mean(command_metrics.get("error_lin_vel_xy", [])),
                yaw=_mean(command_metrics.get("error_yaw_rate", [])),
                success=_mean(command_metrics.get("success", [])),
                stop=_mean(command_metrics.get("stop_latched", [])),
                left_stop=_mean(command_metrics.get("left_stop", [])),
                done=_mean(metrics.get("done_rate", [])),
                success_done=_ratio(
                    metrics.get("target_success_done_events", []),
                    metrics.get("done_events", []),
                ),
                joint=_mean(metrics.get("target_joint_error", [])),
                key_body=_mean(metrics.get("target_key_body_error", [])),
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArmHackStandJointFluctuationReport:
    """Accumulate joint and torso world-frame 6D displacement statistics."""

    _POSE_COMPONENTS = (
        ("delta_x_w", "m"),
        ("delta_y_w", "m"),
        ("delta_z_w", "m"),
        ("delta_roll_w", "rad"),
        ("delta_pitch_w", "rad"),
        ("delta_yaw_w", "rad"),
    )
    _POSE_NORM_COMPONENTS = (
        ("horizontal_translation_norm", "m"),
        ("translation_3d_norm", "m"),
        ("rpy_displacement_norm", "rad"),
    )

    def __init__(
        self,
        env,
        report_path: str,
        checkpoint_path: str,
        test_id: str,
        test_data_path: str | None,
        test_manifest_path: str | None,
        payload_kg: float,
    ):
        self._base_env = env.unwrapped
        self._robot = self._base_env.scene["robot"]
        self._report_path = Path(report_path).expanduser().resolve()
        self._plot_path = self._report_path.with_name(f"{self._report_path.stem}__torso_world_6d.png")
        self._checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._test_id = str(test_id)
        self._test_data_path = Path(test_data_path).expanduser().resolve() if test_data_path else None
        self._test_manifest_path = (
            Path(test_manifest_path).expanduser().resolve() if test_manifest_path else None
        )
        self._payload_kg = float(payload_kg)
        self._joint_names = list(self._robot.joint_names)
        self._arm_joint_names = set(getattr(self._base_env.cfg.upper_body_perturbation, "joint_names", []))
        num_joints = len(self._joint_names)
        device = self._robot.data.joint_pos.device
        self._sum = torch.zeros(num_joints, dtype=torch.float64, device=device)
        self._sum_sq = torch.zeros(num_joints, dtype=torch.float64, device=device)
        self._minimum = torch.full((num_joints,), float("inf"), dtype=torch.float64, device=device)
        self._maximum = torch.full((num_joints,), float("-inf"), dtype=torch.float64, device=device)
        self._abs_step_delta_sum = torch.zeros(num_joints, dtype=torch.float64, device=device)
        self._sample_count = 0
        self._delta_sample_count = 0
        self._termination_events = 0
        self._previous_joint_pos: torch.Tensor | None = None

        torso_body_ids, _ = self._robot.find_bodies("torso_link", preserve_order=True)
        if len(torso_body_ids) != 1:
            raise ValueError(f"ArmHack Stand report expected one torso_link, got {torso_body_ids}")
        self._torso_body_id = int(torso_body_ids[0])
        self._torso_pose_reference_w = self._current_torso_pose_w()
        self._initial_torso_pose_reference_mean_w = torch.mean(self._torso_pose_reference_w, dim=0).cpu()
        self._torso_pose_sum = torch.zeros(6, dtype=torch.float64, device=device)
        self._torso_pose_abs_sum = torch.zeros(6, dtype=torch.float64, device=device)
        self._torso_pose_sum_sq = torch.zeros(6, dtype=torch.float64, device=device)
        self._torso_pose_minimum = torch.full((6,), float("inf"), dtype=torch.float64, device=device)
        self._torso_pose_maximum = torch.full((6,), float("-inf"), dtype=torch.float64, device=device)
        self._torso_pose_sample_count = 0
        self._torso_norm_sum = torch.zeros(3, dtype=torch.float64, device=device)
        self._torso_norm_sum_sq = torch.zeros(3, dtype=torch.float64, device=device)
        self._torso_norm_maximum = torch.zeros(3, dtype=torch.float64, device=device)
        self._torso_pose_delta_trace: list[torch.Tensor] = []

    def _current_torso_pose_w(self) -> torch.Tensor:
        torso_pos_w = self._robot.data.body_pos_w[:, self._torso_body_id, :].detach().to(dtype=torch.float64)
        torso_quat_w = self._robot.data.body_quat_w[:, self._torso_body_id, :].detach()
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(torso_quat_w)
        torso_rpy_w = torch.stack((roll, pitch, yaw), dim=1).to(dtype=torch.float64)
        return torch.cat((torso_pos_w, torso_rpy_w), dim=1)

    def update(self, dones: torch.Tensor) -> None:
        done_mask = dones.detach().bool().reshape(-1)
        self._termination_events += int(torch.sum(done_mask).item())

        joint_pos = self._robot.data.joint_pos.detach().to(dtype=torch.float64)
        self._sum += torch.sum(joint_pos, dim=0)
        self._sum_sq += torch.sum(joint_pos * joint_pos, dim=0)
        self._minimum = torch.minimum(self._minimum, torch.amin(joint_pos, dim=0))
        self._maximum = torch.maximum(self._maximum, torch.amax(joint_pos, dim=0))
        self._sample_count += int(joint_pos.shape[0])
        if self._previous_joint_pos is not None:
            valid_mask = ~done_mask
            if torch.any(valid_mask):
                self._abs_step_delta_sum += torch.sum(
                    torch.abs(joint_pos[valid_mask] - self._previous_joint_pos[valid_mask]), dim=0
                )
                self._delta_sample_count += int(torch.sum(valid_mask).item())
        self._previous_joint_pos = joint_pos.clone()

        torso_pose_w = self._current_torso_pose_w()
        if torch.any(done_mask):
            self._torso_pose_reference_w[done_mask] = torso_pose_w[done_mask]
        torso_pose_delta_w = torso_pose_w - self._torso_pose_reference_w
        torso_pose_delta_w[:, 3:] = math_utils.wrap_to_pi(torso_pose_delta_w[:, 3:])
        self._torso_pose_sum += torch.sum(torso_pose_delta_w, dim=0)
        self._torso_pose_abs_sum += torch.sum(torch.abs(torso_pose_delta_w), dim=0)
        self._torso_pose_sum_sq += torch.sum(torso_pose_delta_w * torso_pose_delta_w, dim=0)
        self._torso_pose_minimum = torch.minimum(
            self._torso_pose_minimum, torch.amin(torso_pose_delta_w, dim=0)
        )
        self._torso_pose_maximum = torch.maximum(
            self._torso_pose_maximum, torch.amax(torso_pose_delta_w, dim=0)
        )
        self._torso_pose_sample_count += int(torso_pose_delta_w.shape[0])

        torso_pose_norms = torch.stack(
            (
                torch.linalg.norm(torso_pose_delta_w[:, :2], dim=1),
                torch.linalg.norm(torso_pose_delta_w[:, :3], dim=1),
                torch.linalg.norm(torso_pose_delta_w[:, 3:], dim=1),
            ),
            dim=1,
        )
        self._torso_norm_sum += torch.sum(torso_pose_norms, dim=0)
        self._torso_norm_sum_sq += torch.sum(torso_pose_norms * torso_pose_norms, dim=0)
        self._torso_norm_maximum = torch.maximum(
            self._torso_norm_maximum, torch.amax(torso_pose_norms, dim=0)
        )
        self._torso_pose_delta_trace.append(torch.mean(torso_pose_delta_w, dim=0).cpu())

    def _load_stage_timeline(self, total_duration_s: float) -> list[dict[str, float | str]]:
        timeline: list[dict[str, float | str]] = []
        if self._test_manifest_path is not None:
            if not self._test_manifest_path.is_file():
                raise FileNotFoundError(f"ArmHack Stand manifest does not exist: {self._test_manifest_path}")
            manifest = json.loads(self._test_manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files", {})
            if self._test_id in {
                "all",
                "representative_poses",
                "synthesized_poses",
                "randomized_poses",
                "representative_trajectories",
                "synthesized_trajectories",
                "randomized_trajectories",
                "down_to_horizontal",
                "default_forward_return_down",
            }:
                file_metadata = files.get(self._test_id, {})
                timeline = list(file_metadata.get("detailed_timeline") or file_metadata.get("timeline") or [])
            elif "_item" in self._test_id:
                mode, item_text = self._test_id.rsplit("_item", 1)
                item_index = int(item_text) - 1
                collection_key_by_mode = {
                    "representative_pose": ("representative_poses", "pose_id"),
                    "synthesized_pose": ("synthesized_poses", "pose_id"),
                    "randomized_pose": ("randomized_poses", "pose_id"),
                    "representative_trajectory": ("representative_trajectories", "trajectory_id"),
                    "synthesized_trajectory": ("synthesized_trajectories", "trajectory_id"),
                    "randomized_trajectory": ("randomized_trajectories", "trajectory_id"),
                }
                collection_key, label_key = collection_key_by_mode[mode]
                item = manifest[collection_key][item_index]
                duration_s = float(item.get("duration_s", item.get("playback_duration_s", total_duration_s)))
                timeline = [
                    {
                        "kind": "static_hold" if "pose" in mode else "trajectory",
                        "label": str(item[label_key]),
                        "start_s": 0.0,
                        "end_s": min(duration_s, total_duration_s),
                    }
                ]

        cleaned: list[dict[str, float | str]] = []
        for stage in timeline:
            start_s = max(float(stage.get("start_s", 0.0)), 0.0)
            end_s = min(float(stage.get("end_s", total_duration_s)), total_duration_s)
            if end_s <= start_s:
                continue
            cleaned.append(
                {
                    "kind": str(stage.get("kind", "stage")),
                    "label": str(stage.get("label", "unspecified")),
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
        cleaned.sort(key=lambda stage: float(stage["start_s"]))
        if not cleaned:
            cleaned.append(
                {"kind": "stage", "label": self._test_id, "start_s": 0.0, "end_s": total_duration_s}
            )
        last_end_s = max(float(stage["end_s"]) for stage in cleaned)
        if last_end_s < total_duration_s - 1.0e-9:
            cleaned.append(
                {"kind": "final_hold", "label": "final_hold", "start_s": last_end_s, "end_s": total_duration_s}
            )
        return cleaned

    @staticmethod
    def _stage_color(stage: dict[str, float | str]) -> str:
        kind = str(stage["kind"])
        label = str(stage["label"])
        if "transition" in kind or "bridge" in kind:
            return "#B0BEC5"
        if label == "arms_down_hold":
            return "#2F4B7C"
        if label in {"arms_default_initial_hold", "arms_default_returned_hold"}:
            return "#4E79A7"
        if label == "arms_natural_down_hold":
            return "#2F4B7C"
        if label == "arms_forward_horizontal_hold":
            return "#00A087"
        if label.startswith("representative_pose"):
            return "#4E79A7"
        if label.startswith("synth_pose"):
            return "#B07AA1"
        if label.startswith("randomized_pose"):
            return "#76B7B2"
        if label.startswith("representative_trajectory"):
            return "#59A14F"
        if label.startswith("synth_trajectory"):
            return "#F28E2B"
        if label.startswith("randomized_trajectory"):
            return "#EDC948"
        return "#BAB0AC"

    @staticmethod
    def _short_stage_label(stage: dict[str, float | str]) -> str:
        kind = str(stage["kind"])
        label = str(stage["label"])
        if "transition" in kind:
            transition_labels = {
                "arms_down_to_forward_horizontal": "D→H",
                "arms_default_to_forward_horizontal": "P0→F",
                "arms_forward_horizontal_to_default": "F→P0",
                "arms_default_to_natural_down": "P0→AD",
            }
            return transition_labels.get(label, "T")
        if "bridge" in kind:
            return "B"
        if label == "final_hold":
            return "H"
        if label == "arms_down_hold":
            return "AD"
        if label == "arms_default_initial_hold":
            return "P0"
        if label == "arms_default_returned_hold":
            return "P0R"
        if label == "arms_natural_down_hold":
            return "AD"
        if label == "arms_forward_horizontal_hold":
            return "AH"
        replacements = (
            ("representative_pose_", "RP"),
            ("synth_pose_", "SP"),
            ("randomized_pose_", "GP"),
            ("representative_trajectory_", "RT"),
            ("synth_trajectory_", "ST"),
            ("randomized_trajectory_", "GT"),
        )
        for prefix, short_prefix in replacements:
            if label.startswith(prefix):
                return short_prefix + label.removeprefix(prefix)
        return label[:8]

    def _write_torso_pose_plot(
        self,
        control_dt: float,
        total_duration_s: float,
        stage_timeline: list[dict[str, float | str]],
    ) -> None:
        if not self._torso_pose_delta_trace:
            raise RuntimeError("Cannot plot ArmHack torso pose without pose samples.")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch

        pose_delta = torch.stack(self._torso_pose_delta_trace).numpy()
        times_s = (np.arange(len(pose_delta), dtype=np.float64) + 1.0) * float(control_dt)
        self._plot_path.parent.mkdir(parents=True, exist_ok=True)

        figure = plt.figure(figsize=(18.0, 10.0), layout="constrained")
        grid = figure.add_gridspec(3, 1, height_ratios=(3.0, 3.0, 1.15), hspace=0.16)
        position_axis = figure.add_subplot(grid[0, 0])
        rotation_axis = figure.add_subplot(grid[1, 0], sharex=position_axis)
        stage_axis = figure.add_subplot(grid[2, 0], sharex=position_axis)

        position_colors = ("#1F77B4", "#D62728", "#2CA02C")
        rotation_colors = ("#9467BD", "#FF7F0E", "#17BECF")
        for index, (label, color) in enumerate(zip(("dx_w", "dy_w", "dz_w"), position_colors, strict=True)):
            position_axis.plot(times_s, pose_delta[:, index], label=label, color=color, linewidth=1.15)
        for index, (label, color) in enumerate(
            zip(("droll_w", "dpitch_w", "dyaw_w"), rotation_colors, strict=True), start=3
        ):
            rotation_axis.plot(times_s, pose_delta[:, index], label=label, color=color, linewidth=1.15)

        for stage in stage_timeline:
            start_s = float(stage["start_s"])
            end_s = float(stage["end_s"])
            color = self._stage_color(stage)
            position_axis.axvspan(start_s, end_s, color=color, alpha=0.055, linewidth=0.0)
            rotation_axis.axvspan(start_s, end_s, color=color, alpha=0.055, linewidth=0.0)
            stage_axis.axvspan(
                start_s,
                end_s,
                facecolor=color,
                alpha=0.88,
                linewidth=0.4,
                edgecolor="white",
            )
            duration_s = end_s - start_s
            stage_axis.text(
                0.5 * (start_s + end_s),
                0.5,
                self._short_stage_label(stage),
                ha="center",
                va="center",
                rotation=90 if duration_s < 2.5 else 0,
                fontsize=6.5,
                color="#111111",
                clip_on=True,
            )

        for axis in (position_axis, rotation_axis):
            axis.axhline(0.0, color="#333333", linewidth=0.7, alpha=0.65)
            axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.7)
            axis.legend(loc="upper right", ncol=3, frameon=True, fontsize=8)
        position_axis.set_ylabel("World translation displacement (m)")
        rotation_axis.set_ylabel("World RPY displacement (rad)")
        rotation_axis.set_xlabel("Test time (s)")
        position_axis.tick_params(labelbottom=False)

        stage_axis.set_ylim(0.0, 1.0)
        stage_axis.set_yticks([])
        stage_axis.set_xlabel("Test stage timeline (s)")
        stage_axis.set_xlim(0.0, max(total_duration_s, control_dt))
        stage_axis.set_title(
            "RP=representative pose, SP=measured-blend pose, GP=randomized-bank pose, "
            "RT=representative trajectory, ST=measured-blend trajectory, "
            "GT=randomized pose-interpolation trajectory, AD=arms down, "
            "D→H=down-to-horizontal transition, AH=arms horizontal, B=bridge",
            fontsize=8,
        )
        stage_axis.legend(
            handles=[
                Patch(color="#4E79A7", label="RP"),
                Patch(color="#B07AA1", label="SP"),
                Patch(color="#76B7B2", label="GP"),
                Patch(color="#59A14F", label="RT"),
                Patch(color="#F28E2B", label="ST"),
                Patch(color="#EDC948", label="GT"),
                Patch(color="#B0BEC5", label="transition / bridge"),
                Patch(color="#2F4B7C", label="AD"),
                Patch(color="#00A087", label="AH"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.48),
            ncol=9,
            fontsize=8,
            frameon=False,
        )
        figure.suptitle(
            f"ArmHack Stand torso world-frame 6D displacement — {self._test_id}",
            fontsize=14,
            fontweight="bold",
        )
        figure.savefig(self._plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

    def write(self, play_metrics: dict, steps: int, control_dt: float) -> None:
        if self._sample_count <= 0:
            raise RuntimeError("Cannot write ArmHack Stand report without joint samples.")
        if self._torso_pose_sample_count <= 0:
            raise RuntimeError("Cannot write ArmHack Stand report without torso pose samples.")

        mean = self._sum / self._sample_count
        variance = torch.clamp(self._sum_sq / self._sample_count - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        position_range = self._maximum - self._minimum
        if self._delta_sample_count > 0:
            mean_abs_step_delta = self._abs_step_delta_sum / self._delta_sample_count
        else:
            mean_abs_step_delta = torch.zeros_like(self._abs_step_delta_sum)

        mean = mean.cpu().tolist()
        std = std.cpu().tolist()
        position_range = position_range.cpu().tolist()
        mean_abs_step_delta = mean_abs_step_delta.cpu().tolist()

        torso_mean = self._torso_pose_sum / self._torso_pose_sample_count
        torso_mean_abs = self._torso_pose_abs_sum / self._torso_pose_sample_count
        torso_variance = torch.clamp(
            self._torso_pose_sum_sq / self._torso_pose_sample_count - torso_mean * torso_mean,
            min=0.0,
        )
        torso_std = torch.sqrt(torso_variance)
        torso_rms = torch.sqrt(self._torso_pose_sum_sq / self._torso_pose_sample_count)
        torso_max_abs = torch.maximum(torch.abs(self._torso_pose_minimum), torch.abs(self._torso_pose_maximum))
        torso_range = self._torso_pose_maximum - self._torso_pose_minimum
        torso_norm_mean = self._torso_norm_sum / self._torso_pose_sample_count
        torso_norm_variance = torch.clamp(
            self._torso_norm_sum_sq / self._torso_pose_sample_count - torso_norm_mean * torso_norm_mean,
            min=0.0,
        )
        torso_norm_std = torch.sqrt(torso_norm_variance)
        torso_norm_rms = torch.sqrt(self._torso_norm_sum_sq / self._torso_pose_sample_count)
        torso_values = [
            tensor.cpu().tolist()
            for tensor in (
                torso_mean,
                torso_mean_abs,
                torso_std,
                torso_rms,
                torso_max_abs,
                self._torso_pose_minimum,
                self._torso_pose_maximum,
                torso_range,
            )
        ]
        torso_norm_values = [
            tensor.cpu().tolist()
            for tensor in (torso_norm_mean, torso_norm_std, torso_norm_rms, self._torso_norm_maximum)
        ]

        test_data_columns: list[str] = []
        if self._test_data_path is not None:
            if not self._test_data_path.is_file():
                raise FileNotFoundError(f"ArmHack Stand test data does not exist: {self._test_data_path}")
            with self._test_data_path.open("r", encoding="utf-8", newline="") as handle:
                test_data_columns = next(csv.reader(handle), [])
            expected_columns = ["time_s", *list(self._base_env.cfg.upper_body_perturbation.joint_names)]
            if set(test_data_columns) != set(expected_columns) or len(test_data_columns) != 15:
                raise ValueError(
                    "ArmHack Stand report requires test CSV with time_s plus exactly 14 configured arm joints; "
                    f"got {test_data_columns}"
                )

        total_duration_s = steps * control_dt
        stage_timeline = self._load_stage_timeline(total_duration_s)
        self._write_torso_pose_plot(control_dt, total_duration_s, stage_timeline)
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        lines = [
            "# ArmHack Stand 关节与躯干 6D 位移测试报告",
            "",
            "## 测试身份",
            "",
            f"- 生成时间：`{generated_at}`",
            f"- checkpoint：`{self._checkpoint_path}`",
            f"- checkpoint SHA-256：`{_sha256(self._checkpoint_path)}`",
            f"- 测试项：`{self._test_id}`",
            f"- 测试数据：`{self._test_data_path}`" if self._test_data_path else "- 测试数据：未提供",
            (
                f"- 测试数据 SHA-256：`{_sha256(self._test_data_path)}`"
                if self._test_data_path is not None
                else "- 测试数据 SHA-256：未提供"
            ),
            f"- 测试清单：`{self._test_manifest_path}`" if self._test_manifest_path else "- 测试清单：未提供",
            (
                f"- 测试清单 SHA-256：`{_sha256(self._test_manifest_path)}`"
                if self._test_manifest_path is not None
                else "- 测试清单 SHA-256：未提供"
            ),
            f"- 6D 曲线图：`{self._plot_path}`",
            f"- 控制步数：`{steps}`",
            f"- 控制周期：`{control_dt:.6f} s`",
            f"- 仿真测试时长：`{total_duration_s:.3f} s`",
            f"- 环境采样数：`{self._sample_count}`",
            f"- termination/reset 事件数：`{self._termination_events}`",
            f"- 左/右腕末端附加质量：各 `{self._payload_kg:.6f} kg`（固定测试值）",
            "- 测试输入范围：仅双臂 14 关节；腰、腿和根节点不在测试 CSV 中。",
            "",
            "## 统计口径",
            "",
            "`平均逐步波动` 是同一 episode 内相邻 50 Hz 控制帧实际关节角之差绝对值的均值："
            " `mean(|q[t]-q[t-1]|)`。reset 前后的跳变不计入。",
            "",
            "躯干 6D 位移使用 `torso_link` 的世界坐标位姿，并相对每个 episode 的起始位姿计算："
            "位置为 `p_w(t)-p_w(0)`；姿态为世界系 XYZ Euler 角之差并 wrap 到 `[-pi, pi]`。"
            "发生 reset 时立即用 reset 后位姿重建参考，因此 reset 跳变不进入位移统计。",
            "",
            "## 每关节实际波动",
            "",
            "| 关节 | 分组 | 平均逐步波动 rad/step | 实际角均值 rad | 标准差 rad | 极差 rad |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for index, joint_name in enumerate(self._joint_names):
            group = "双臂输入关节" if joint_name in self._arm_joint_names else "平衡策略关节"
            lines.append(
                f"| `{joint_name}` | {group} | {mean_abs_step_delta[index]:.8f} | "
                f"{mean[index]:.8f} | {std[index]:.8f} | {position_range[index]:.8f} |"
            )

        lines.extend(
            [
                "",
                "## 躯干世界坐标系 6D 位移",
                "",
                "初始参考位姿（所有环境均值）为："
                f" `x={self._initial_torso_pose_reference_mean_w[0]:.6f} m, "
                f"y={self._initial_torso_pose_reference_mean_w[1]:.6f} m, "
                f"z={self._initial_torso_pose_reference_mean_w[2]:.6f} m, "
                f"roll={self._initial_torso_pose_reference_mean_w[3]:.6f} rad, "
                f"pitch={self._initial_torso_pose_reference_mean_w[4]:.6f} rad, "
                f"yaw={self._initial_torso_pose_reference_mean_w[5]:.6f} rad`。",
                "",
                "| 分量 | 单位 | 有符号均值 | 绝对值均值 | 标准差 | RMS | 最大绝对值 | 最小值 | 最大值 | 极差 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for index, (component, unit) in enumerate(self._POSE_COMPONENTS):
            lines.append(
                f"| `{component}` | {unit} | {torso_values[0][index]:.8f} | "
                f"{torso_values[1][index]:.8f} | {torso_values[2][index]:.8f} | "
                f"{torso_values[3][index]:.8f} | {torso_values[4][index]:.8f} | "
                f"{torso_values[5][index]:.8f} | {torso_values[6][index]:.8f} | "
                f"{torso_values[7][index]:.8f} |"
            )
        lines.extend(
            [
                "",
                "| 综合位移 | 单位 | 均值 | 标准差 | RMS | 最大值 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for index, (component, unit) in enumerate(self._POSE_NORM_COMPONENTS):
            lines.append(
                f"| `{component}` | {unit} | {torso_norm_values[0][index]:.8f} | "
                f"{torso_norm_values[1][index]:.8f} | {torso_norm_values[2][index]:.8f} | "
                f"{torso_norm_values[3][index]:.8f} |"
            )
        lines.extend(
            [
                "",
                "### 6D 位移曲线与测试阶段",
                "",
                f"![Torso world-frame 6D displacement]({self._plot_path.name})",
                "",
                "曲线是每个控制帧上所有测试环境 6D 位移的均值；确定性可视化默认只有 1 个环境。"
                "背景色和底部色带由 manifest 的详细时间线生成。",
                "",
                "| 开始 s | 结束 s | 时长 s | 类型 | 姿态或轨迹阶段 |",
                "|---:|---:|---:|---|---|",
            ]
        )
        for stage in stage_timeline:
            start_s = float(stage["start_s"])
            end_s = float(stage["end_s"])
            lines.append(
                f"| {start_s:.3f} | {end_s:.3f} | {end_s - start_s:.3f} | "
                f"`{stage['kind']}` | `{stage['label']}` |"
            )

        lines.extend(["", "## 躯干稳定指标", ""])
        important_metrics = play_metrics.get("important", {})
        if important_metrics:
            lines.extend(["| 指标 | 均值 |", "|---|---:|"])
            for key in sorted(important_metrics):
                lines.append(f"| `{key}` | {_mean(important_metrics[key]):.8f} |")
        else:
            lines.append("本次回放未提供 Important Metrics。")

        lines.extend(
            [
                "",
                "## 结论边界",
                "",
                "该报告记录仿真中的实际关节波动、torso 世界坐标系 6D 位移和 termination 次数。"
                "双臂是外部测试输入；腰腿的波动是策略为保持平衡产生的响应。"
                "若 termination/reset 事件数大于 0，不能把该测试项判定为完整稳定通过。",
                "",
            ]
        )
        self._report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[REPORT] ArmHack Stand report: {self._report_path}")
        print(f"[REPORT] ArmHack Stand torso world-frame 6D plot: {self._plot_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise ModuleNotFoundError(
                "`isaaclab.utils.pretrained_checkpoint` is not available in this IsaacLab version. "
                "Please provide `--checkpoint` or upgrade IsaacLab."
            )
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # update the viewport camera before video frames are captured
    if args_cli.follow_camera:
        camera_kwargs = {
            "view": args_cli.camera_view,
            "distance": args_cli.camera_distance,
            "height": args_cli.camera_height,
            "target_height": args_cli.camera_target_height,
            "lateral": args_cli.camera_lateral,
            "smoothing": args_cli.camera_smoothing,
            "env_index": args_cli.camera_env_index,
        }
        print("[INFO] Using robot-follow camera.")
        print_dict(camera_kwargs, nesting=4)
        env = _FollowRobotCameraWrapper(env, **camera_kwargs)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner
        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    if args_cli.skip_export or args_cli.strict_export:
        print("[INFO] Export flags are accepted for compatibility; use scripts/export_g1_amp_policy.sh for export.")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    play_metrics = {
        "reward": [],
        "lin_vel_xy_errors": [],
        "yaw_rate_errors": [],
        "cmd_lin_x": [],
        "cmd_lin_y": [],
        "cmd_yaw": [],
        "mean_lin_x": [],
        "mean_lin_y": [],
        "mean_yaw": [],
        "done_rate": [],
        "done_events": [],
        "target_success_done_events": [],
        "target_joint_error": [],
        "target_key_body_error": [],
        "command": {},
        "important": {},
    }
    armhack_stand_report = None
    if args_cli.armhack_stand_report_path:
        if not any(token in task_name for token in ("StandPerturb", "StandRandomizedPayload")):
            raise ValueError("--armhack_stand_report_path is only valid for an ArmHack Stand task.")
        armhack_stand_report = _ArmHackStandJointFluctuationReport(
            env=env,
            report_path=args_cli.armhack_stand_report_path,
            checkpoint_path=resume_path,
            test_id=args_cli.armhack_stand_test_id,
            test_data_path=args_cli.armhack_stand_test_data,
            test_manifest_path=args_cli.armhack_stand_manifest,
            payload_kg=args_cli.armhack_stand_payload_kg,
        )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            _update_play_metrics(play_metrics, env, rewards, dones, extras)
            if armhack_stand_report is not None:
                armhack_stand_report.update(dones)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            print(f"[INFO] Reached max_steps={args_cli.max_steps}. Exiting play loop.")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    _print_play_report(play_metrics, timestep)
    if armhack_stand_report is not None:
        armhack_stand_report.write(play_metrics=play_metrics, steps=timestep, control_dt=dt)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
