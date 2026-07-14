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
    """Accumulate actual 29-DoF joint statistics for a Stand playback run."""

    def __init__(self, env, report_path: str, checkpoint_path: str, test_id: str, test_data_path: str | None):
        self._base_env = env.unwrapped
        self._robot = self._base_env.scene["robot"]
        self._report_path = Path(report_path).expanduser().resolve()
        self._checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._test_id = str(test_id)
        self._test_data_path = Path(test_data_path).expanduser().resolve() if test_data_path else None
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

    def update(self, dones: torch.Tensor) -> None:
        joint_pos = self._robot.data.joint_pos.detach().to(dtype=torch.float64)
        self._sum += torch.sum(joint_pos, dim=0)
        self._sum_sq += torch.sum(joint_pos * joint_pos, dim=0)
        self._minimum = torch.minimum(self._minimum, torch.amin(joint_pos, dim=0))
        self._maximum = torch.maximum(self._maximum, torch.amax(joint_pos, dim=0))
        self._sample_count += int(joint_pos.shape[0])

        done_mask = dones.detach().bool().reshape(-1)
        self._termination_events += int(torch.sum(done_mask).item())
        if self._previous_joint_pos is not None:
            valid_mask = ~done_mask
            if torch.any(valid_mask):
                self._abs_step_delta_sum += torch.sum(
                    torch.abs(joint_pos[valid_mask] - self._previous_joint_pos[valid_mask]), dim=0
                )
                self._delta_sample_count += int(torch.sum(valid_mask).item())
        self._previous_joint_pos = joint_pos.clone()

    def write(self, play_metrics: dict, steps: int, control_dt: float) -> None:
        if self._sample_count <= 0:
            raise RuntimeError("Cannot write ArmHack Stand joint report without joint samples.")

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

        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        lines = [
            "# ArmHack Stand 关节波动测试报告",
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
            f"- 控制步数：`{steps}`",
            f"- 控制周期：`{control_dt:.6f} s`",
            f"- 仿真测试时长：`{steps * control_dt:.3f} s`",
            f"- 环境采样数：`{self._sample_count}`",
            f"- termination/reset 事件数：`{self._termination_events}`",
            "- 测试输入范围：仅双臂 14 关节；腰、腿和根节点不在测试 CSV 中。",
            "",
            "## 统计口径",
            "",
            "`平均逐步波动` 是同一 episode 内相邻 50 Hz 控制帧实际关节角之差绝对值的均值："
            " `mean(|q[t]-q[t-1]|)`。reset 前后的跳变不计入。另列出实际关节角均值、标准差和极差。",
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
                "该报告记录仿真中的实际关节波动和 termination 次数。双臂是外部测试输入；腰腿的波动是策略为保持平衡产生的响应。"
                "若 termination/reset 事件数大于 0，不能把该测试项判定为完整稳定通过。",
                "",
            ]
        )
        self._report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[REPORT] ArmHack Stand joint fluctuation report: {self._report_path}")


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
        if "StandPerturb" not in task_name:
            raise ValueError("--armhack_stand_report_path is only valid for an ArmHack StandPerturb task.")
        armhack_stand_report = _ArmHackStandJointFluctuationReport(
            env=env,
            report_path=args_cli.armhack_stand_report_path,
            checkpoint_path=resume_path,
            test_id=args_cli.armhack_stand_test_id,
            test_data_path=args_cli.armhack_stand_test_data,
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
