#!/usr/bin/env python3
"""ArmHack Stand real-robot runner with operator-switched arm-only trajectories.

The actor still controls all 29 policy outputs.  At every 50 Hz control frame,
the 14 arm entries are replaced by the current minimum-jerk arm position target;
the remaining waist/leg entries stay equal to the actor output.  The composed
action is written back to the next observation's action-history block.

Real mode intentionally requires an interactive terminal.  It keeps the current
pose while releasing Unitree's high-level motion service, moves to a validated
Stand startup pose, then waits for a second operator gate before policy rollout.

Offline validation (does not import Unitree SDK or initialize DDS):
    python deploy_real_g1_armhack_stand.py --self-test \
        --policy /path/to/stand.onnx --presets /path/to/stand_arm_presets.json \
        --config configs/g1_amp.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# S3 G1 29DoF limits from resources/robots/g1_description/g1_29dof.xml.
JOINT_LIMITS_RAD = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _minimum_jerk(alpha: float) -> float:
    """Quintic 10a^3 - 15a^4 + 6a^5 with zero endpoint velocity/acceleration."""
    value = float(np.clip(alpha, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


@dataclass(frozen=True)
class ArmPose:
    pose_id: str
    label_zh: str
    positions: np.ndarray
    source: str


def load_arm_presets(path: Path, limit_margin_rad: float = 0.0, verify_sources: bool = True) -> tuple[list[ArmPose], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported preset schema: {payload.get('schema_version')}")
    if payload.get("data_scope") != "arm_only_14_dof":
        raise ValueError("Preset data_scope must be arm_only_14_dof.")
    if payload.get("arm_joint_names") != ARM_JOINT_NAMES:
        raise ValueError("Preset arm_joint_names/order does not match the Stand arm contract.")
    transition_s = float(payload.get("default_transition_s", 4.0))
    if transition_s <= 0.0:
        raise ValueError("default_transition_s must be positive.")

    poses: list[ArmPose] = []
    for entry in payload.get("poses", []):
        values = np.asarray(entry.get("positions_rad"), dtype=np.float32)
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Pose {entry.get('id')} must contain 14 finite radians.")
        for name, value in zip(ARM_JOINT_NAMES, values):
            lower, upper = JOINT_LIMITS_RAD[name]
            if not lower + limit_margin_rad <= float(value) <= upper - limit_margin_rad:
                raise ValueError(
                    f"Pose {entry.get('id')} joint {name}={value:.6f} is outside "
                    f"safe range [{lower + limit_margin_rad:.6f}, {upper - limit_margin_rad:.6f}]."
                )
        source = str(entry.get("source", ""))
        if verify_sources and source:
            _verify_pose_source(path.parent, source, values)
        poses.append(
            ArmPose(
                pose_id=str(entry["id"]),
                label_zh=str(entry.get("label_zh", entry["id"])),
                positions=values,
                source=source,
            )
        )
    if len(poses) < 2:
        raise ValueError("At least two arm poses are required for SPACE switching.")
    return poses, transition_s


def _verify_pose_source(preset_dir: Path, source: str, expected: np.ndarray) -> None:
    source_path_text, _, selector = source.partition(":")
    if selector not in {"", "first_row"}:
        raise ValueError(f"Unsupported preset source selector: {selector}")
    source_path = (preset_dir / source_path_text).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Preset source CSV does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    actual = np.asarray([float(row[name]) for name in ARM_JOINT_NAMES], dtype=np.float32)
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"Preset values drifted from source CSV: {source_path}")


class ArmPresetSequencer:
    """Cycle arm postures and generate interruptible minimum-jerk transitions."""

    def __init__(self, poses: list[ArmPose], transition_s: float, start_time: float) -> None:
        self.poses = poses
        self.transition_s = float(transition_s)
        self.active_index = 0
        self.start_time = float(start_time)
        self.start_positions = poses[0].positions.copy()
        self.target_positions = poses[0].positions.copy()

    def sample(self, now: float) -> np.ndarray:
        alpha = (float(now) - self.start_time) / self.transition_s
        blend = _minimum_jerk(alpha)
        return self.start_positions + (self.target_positions - self.start_positions) * blend

    def switch_next(self, now: float) -> ArmPose:
        current = self.sample(now).copy()
        self.active_index = (self.active_index + 1) % len(self.poses)
        self.start_positions = current
        self.target_positions = self.poses[self.active_index].positions.copy()
        self.start_time = float(now)
        return self.poses[self.active_index]

    @property
    def active_pose(self) -> ArmPose:
        return self.poses[self.active_index]


def _load_config_contract(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(config["policy_joint_names"])
    if len(names) != 29 or set(names) != set(JOINT_LIMITS_RAD):
        raise ValueError("Real config must contain the exact S3 G1 29DoF joint set.")
    motor_names = list(config["motor_joint_names"])
    if len(motor_names) != 29 or set(motor_names) != set(names):
        raise ValueError("motor_joint_names must contain the same exact 29DoF joint set.")
    for key in ("motor_indices", "default_angles", "kps", "kds"):
        if len(config[key]) != 29:
            raise ValueError(f"Real config {key} must contain 29 entries.")
    if float(config["control_dt"]) != 0.02:
        raise ValueError("ArmHack Stand deployment requires control_dt=0.02 (50 Hz).")
    if float(config["action_scale"]) != 0.25:
        raise ValueError("ArmHack Stand deployment requires action_scale=0.25.")
    return config


def run_self_test(policy_path: Path, preset_path: Path, config_path: Path) -> None:
    config = _load_config_contract(config_path)
    poses, default_transition_s = load_arm_presets(preset_path, limit_margin_rad=0.05, verify_sources=True)
    transition_s = _env_float("G1_ARMHACK_STAND_TRANSITION_S", default_transition_s)
    if transition_s < 2.0:
        raise ValueError("G1_ARMHACK_STAND_TRANSITION_S must be >= 2.0s (the training range lower bound).")

    sequencer = ArmPresetSequencer(poses, transition_s, start_time=0.0)
    before = sequencer.sample(0.0)
    sequencer.switch_next(0.0)
    start = sequencer.sample(0.0)
    middle = sequencer.sample(transition_s * 0.5)
    end = sequencer.sample(transition_s)
    if not np.array_equal(before, start) or not np.allclose(end, poses[1].positions, atol=1.0e-7):
        raise AssertionError("Minimum-jerk switch endpoint/continuity test failed.")
    if not np.all(np.isfinite(middle)):
        raise AssertionError("Minimum-jerk transition produced non-finite values.")
    interrupted_before = sequencer.sample(transition_s * 0.5)
    sequencer.switch_next(transition_s * 0.5)
    interrupted_after = sequencer.sample(transition_s * 0.5)
    if not np.array_equal(interrupted_before, interrupted_after):
        raise AssertionError("Interrupted SPACE transition introduced a position discontinuity.")
    max_delta = max(float(np.max(np.abs(poses[(i + 1) % len(poses)].positions - poses[i].positions))) for i in range(len(poses)))
    max_velocity = 1.875 * max_delta / transition_s

    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(policy_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "obs" or list(inputs[0].shape) != [1, 96]:
        raise ValueError(f"Unexpected Stand actor input: {[(item.name, item.shape) for item in inputs]}")
    if len(outputs) != 1 or outputs[0].name != "actions" or list(outputs[0].shape) != [1, 29]:
        raise ValueError(f"Unexpected Stand actor output: {[(item.name, item.shape) for item in outputs]}")
    action = session.run(["actions"], {"obs": np.zeros((1, 96), dtype=np.float32)})[0]
    if action.shape != (1, 29) or not np.all(np.isfinite(action)):
        raise ValueError("Stand actor zero-observation inference failed finite [1,29] check.")
    rng = np.random.default_rng(20260718)
    for _ in range(16):
        obs = rng.normal(0.0, 0.5, size=(1, 96)).astype(np.float32)
        random_action = session.run(["actions"], {"obs": obs})[0]
        if random_action.shape != (1, 29) or not np.all(np.isfinite(random_action)):
            raise ValueError("Stand actor random-observation inference failed finite [1,29] check.")

    arm_policy_indices = [list(config["policy_joint_names"]).index(name) for name in ARM_JOINT_NAMES]
    print("[SELF-TEST PASS] 未初始化 Unitree DDS，未发送机器人命令。")
    print(f"  policy: {policy_path}")
    print("  actor : obs[1,96] -> actions[1,29], zero + 16 deterministic random inputs finite")
    print(f"  poses : {len(poses)} arm-only presets; ids={[pose.pose_id for pose in poses]}")
    print(f"  arms  : policy indices={arm_policy_indices}")
    print(f"  path  : minimum-jerk {transition_s:.3f}s; worst adjacent peak speed={max_velocity:.3f} rad/s")


def _quat_wxyz_to_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    norm = float(np.linalg.norm(quat))
    if not 0.5 <= norm <= 1.5:
        raise RuntimeError(f"Invalid IMU quaternion norm: {norm:.6f}")
    w, x, y, z = (quat / norm).tolist()
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def run_real(net: str, config_name: str) -> None:
    if os.environ.get("G1_ARMHACK_STAND_CONFIRM") != "I_UNDERSTAND":
        raise RuntimeError("Real mode requires G1_ARMHACK_STAND_CONFIRM=I_UNDERSTAND via the launcher.")
    # This import is intentionally delayed so --self-test and confirmation rejection work without Unitree SDK.
    import deploy_real_g1_amp as amp

    config_path = Path(__file__).resolve().parent / "configs" / config_name
    config = amp.Config(str(config_path))
    if config.msg_type != "hg" or len(config.motor_indices) != 29:
        raise RuntimeError("ArmHack Stand real deployment only supports the G1 29DoF HG LowCmd contract.")
    if config.command_mode != "fixed" or not np.allclose(config.cmd_init, 0.0):
        raise RuntimeError("Stand deployment refuses non-zero/remote velocity commands; require fixed [0,0,0].")
    if not config.release_motion_mode:
        raise RuntimeError("Stand deployment requires G1_AMP_RELEASE_MOTION_MODE=True.")

    preset_path = Path(os.environ["G1_ARMHACK_STAND_PRESET_PATH"]).expanduser().resolve()
    margin = _env_float("G1_ARMHACK_STAND_JOINT_LIMIT_MARGIN_RAD", 0.05)
    if margin < 0.0:
        raise ValueError(f"joint_limit_margin_rad must be non-negative, got {margin}")
    poses, default_transition_s = load_arm_presets(preset_path, limit_margin_rad=margin, verify_sources=True)
    transition_s = _env_float("G1_ARMHACK_STAND_TRANSITION_S", default_transition_s)
    startup_move_s = _env_float("G1_ARMHACK_STAND_STARTUP_MOVE_S", 5.0)
    lowstate_timeout_s = _env_float("G1_ARMHACK_STAND_LOWSTATE_TIMEOUT_S", 0.20)
    max_tilt_rad = _env_float("G1_ARMHACK_STAND_MAX_TILT_RAD", 0.60)
    max_target_speed = _env_float("G1_ARMHACK_STAND_MAX_TARGET_SPEED_RAD_S", 4.0)
    damping_exit_s = _env_float("G1_ARMHACK_STAND_DAMPING_EXIT_S", 1.0)
    joint_print_hz = _env_float("G1_ARMHACK_STAND_JOINT_PRINT_HZ", 1.0)
    for name, value in {
        "transition_s": transition_s,
        "startup_move_s": startup_move_s,
        "lowstate_timeout_s": lowstate_timeout_s,
        "max_tilt_rad": max_tilt_rad,
        "max_target_speed": max_target_speed,
        "damping_exit_s": damping_exit_s,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    if transition_s < 2.0:
        raise ValueError("transition_s must be >= 2.0s (the training range lower bound).")
    if startup_move_s < 3.0:
        raise ValueError("startup_move_s must be >= 3.0s for the real-robot handoff.")

    class StandController(amp.Controller):
        def __init__(self) -> None:
            self.last_lowstate_time = time.monotonic()
            super().__init__(config)
            self.arm_policy_indices = np.asarray(
                [config.policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64
            )
            self.lower_policy = np.asarray(
                [JOINT_LIMITS_RAD[name][0] + margin for name in config.policy_joint_names], dtype=np.float32
            )
            self.upper_policy = np.asarray(
                [JOINT_LIMITS_RAD[name][1] - margin for name in config.policy_joint_names], dtype=np.float32
            )
            self.hard_lower_policy = np.asarray(
                [JOINT_LIMITS_RAD[name][0] for name in config.policy_joint_names], dtype=np.float32
            )
            self.hard_upper_policy = np.asarray(
                [JOINT_LIMITS_RAD[name][1] for name in config.policy_joint_names], dtype=np.float32
            )
            if np.any(self.lower_policy >= self.upper_policy):
                raise ValueError("Joint limit margin leaves an empty safe interval.")
            self.sequencer = ArmPresetSequencer(poses, transition_s, time.monotonic())
            self.previous_target_policy: Optional[np.ndarray] = None
            self.next_joint_print_time = 0.0

        def low_state_hg_handler(self, msg: Any) -> None:
            super().low_state_hg_handler(msg)
            self.last_lowstate_time = time.monotonic()

        def low_state_go_handler(self, msg: Any) -> None:
            super().low_state_go_handler(msg)
            self.last_lowstate_time = time.monotonic()

        def _check_state(self, qj_policy: np.ndarray, quat: np.ndarray) -> None:
            age = time.monotonic() - self.last_lowstate_time
            if age > lowstate_timeout_s:
                raise RuntimeError(f"LowState stale for {age:.3f}s (limit {lowstate_timeout_s:.3f}s).")
            if not np.all(np.isfinite(qj_policy)) or not np.all(np.isfinite(quat)):
                raise RuntimeError("LowState contains NaN/Inf.")
            tolerance = 0.02
            if np.any(qj_policy < self.hard_lower_policy - tolerance) or np.any(
                qj_policy > self.hard_upper_policy + tolerance
            ):
                bad = np.flatnonzero(
                    (qj_policy < self.hard_lower_policy - tolerance)
                    | (qj_policy > self.hard_upper_policy + tolerance)
                )
                details = ", ".join(f"{config.policy_joint_names[i]}={qj_policy[i]:.3f}" for i in bad)
                raise RuntimeError(f"Measured joint position outside hardware safety range: {details}")
            roll, pitch = _quat_wxyz_to_roll_pitch(quat)
            if abs(roll) > max_tilt_rad or abs(pitch) > max_tilt_rad:
                raise RuntimeError(
                    f"Torso tilt exceeded: roll={roll:.3f}, pitch={pitch:.3f}, limit={max_tilt_rad:.3f} rad."
                )

        def _safe_target(self, target_policy: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(target_policy)):
                raise RuntimeError("Policy/arm target contains NaN/Inf.")
            clipped = np.clip(target_policy, self.lower_policy, self.upper_policy)
            if self.previous_target_policy is not None:
                max_delta = max_target_speed * config.control_dt
                clipped = self.previous_target_policy + np.clip(
                    clipped - self.previous_target_policy, -max_delta, max_delta
                )
            self.previous_target_policy = clipped.copy()
            return clipped

        def startup_target_policy(self) -> np.ndarray:
            target = config.default_angles.copy()
            target[self.arm_policy_indices] = poses[0].positions
            return np.clip(target, self.lower_policy, self.upper_policy)

        def hold_motor_target_until_enter(self, target_motor: np.ndarray, prompt: str) -> None:
            kp_motor = config.kps[config.policy_to_motor_order]
            kd_motor = config.kds[config.policy_to_motor_order]
            print(prompt)
            print("按 ENTER 继续；按 q 或 Ctrl-C 立即退出并进入阻尼。")
            with amp.TerminalKeyReader() as keys:
                while True:
                    self._write_motor_targets(target_motor, kp_motor, kd_motor)
                    self.send_cmd(self.low_cmd)
                    key = keys.read_key(config.control_dt)
                    if key in {"\r", "\n"}:
                        return
                    if key.lower() == "q" or self.remote_controller.button[amp.KeyMap.select] == 1:
                        raise KeyboardInterrupt

        def move_to_startup(self, target_policy: np.ndarray) -> None:
            start_motor = self._current_motor_positions()
            target_motor = target_policy[config.policy_to_motor_order]
            kp_motor = config.kps[config.policy_to_motor_order]
            kd_motor = config.kds[config.policy_to_motor_order]
            steps = max(1, int(round(startup_move_s / config.control_dt)))
            print(f"用 minimum-jerk 在 {startup_move_s:.2f}s 内移动到 Stand 启动姿态 {poses[0].pose_id}。")
            with amp.TerminalKeyReader() as keys:
                for step in range(steps):
                    blend = _minimum_jerk((step + 1) / steps)
                    target_motor_step = start_motor + (target_motor - start_motor) * blend
                    self._write_motor_targets(target_motor_step, kp_motor, kd_motor)
                    self.send_cmd(self.low_cmd)
                    key = keys.read_key(config.control_dt)
                    if key.lower() == "q" or self.remote_controller.button[amp.KeyMap.select] == 1:
                        raise KeyboardInterrupt
            self.previous_target_policy = target_policy.copy()

        def run_frame(self, key: str) -> bool:
            now = time.monotonic()
            if key.lower() == "q":
                return False
            if key == " ":
                selected = self.sequencer.switch_next(now)
                print(
                    f"[ARM SWITCH] -> {selected.pose_id} ({selected.label_zh}), "
                    f"minimum-jerk {transition_s:.2f}s",
                    flush=True,
                )

            qj_policy, dqj_policy, quat, ang_vel = self._update_state_arrays()
            self._check_state(qj_policy, quat)
            self.obs[0:3] = ang_vel * config.ang_vel_scale
            self.obs[3:6] = amp.get_gravity_orientation(quat)
            self.obs[6:9] = 0.0
            self.obs[9:38] = (qj_policy - config.default_angles) * config.dof_pos_scale
            self.obs[38:67] = dqj_policy * config.dof_vel_scale
            self.obs[67:96] = self.action_policy

            network_action = self.policy.infer(self.obs)
            if not np.all(np.isfinite(network_action)):
                raise RuntimeError("Stand policy output contains NaN/Inf.")
            executed_action = network_action.copy()
            arm_target = self.sequencer.sample(now)
            executed_action[self.arm_policy_indices] = (
                arm_target - config.default_angles[self.arm_policy_indices]
            ) / config.action_scale
            target_policy = config.default_angles + executed_action * config.action_scale
            target_policy = self._safe_target(target_policy)
            # action history must describe the target that actually survived safety filtering.
            self.action_policy = (target_policy - config.default_angles) / config.action_scale

            target_motor = target_policy[config.policy_to_motor_order]
            self._write_motor_targets(
                target_motor,
                config.kps[config.policy_to_motor_order],
                config.kds[config.policy_to_motor_order],
            )
            self.send_cmd(self.low_cmd)

            if joint_print_hz > 0.0 and now >= self.next_joint_print_time:
                self.next_joint_print_time = now + 1.0 / joint_print_hz
                print(
                    "[29DoF target rad] "
                    + " ".join(f"{name}={target_policy[i]:+.3f}" for i, name in enumerate(config.policy_joint_names)),
                    flush=True,
                )
            time.sleep(config.control_dt)
            return True

        def send_damping_for(self, duration_s: float) -> None:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                amp.create_damping_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
                time.sleep(config.control_dt)

    amp.ChannelFactoryInitialize(0, net)
    print("============================================================")
    print("  ArmHack Stand Real Controller")
    print("============================================================")
    print(f"policy={config.policy_path}")
    print(f"arm presets={[pose.pose_id for pose in poses]}")
    print(f"transition={transition_s:.2f}s, startup={startup_move_s:.2f}s, command=[0,0,0]")
    print("SPACE=下一双臂姿态/轨迹；q/Ctrl-C/遥控器 Select=退出并进入阻尼")
    print("============================================================")

    controller = StandController()
    stop_release_hold = None
    try:
        stop_release_hold = controller.start_current_pos_hold_thread()
        time.sleep(max(config.release_handoff_warmup_s, 0.3))
        amp.release_motion_mode()
        time.sleep(max(config.release_handoff_after_s, 0.2))
        stop_release_hold()
        stop_release_hold = None
        print("[DEBUG MODE] 高层 motion mode 已释放；当前由 rt/lowcmd 低层调试控制保持姿态。")

        current_motor = controller._current_motor_positions()
        controller.hold_motor_target_until_enter(
            current_motor,
            "调试控制已建立。确认机器人仍在吊架保护、关节方向正常后，准备移动到 Stand 启动姿态。",
        )
        startup_policy = controller.startup_target_policy()
        controller.move_to_startup(startup_policy)
        controller.hold_motor_target_until_enter(
            startup_policy[config.policy_to_motor_order],
            "Stand 启动姿态已到达。确认双脚/吊架、急停和周围环境后，准备启动 29DoF policy。",
        )

        controller.sequencer = ArmPresetSequencer(poses, transition_s, time.monotonic())
        controller.action_policy.fill(0.0)
        controller.previous_target_policy = startup_policy.copy()
        start_time = time.monotonic()
        print("[POLICY ON] 29DoF Stand 推理已启动；当前为 P0。按 SPACE 循环切换双臂轨迹。")
        with amp.TerminalKeyReader() as keys:
            while True:
                if not controller.run_frame(keys.read_key(0.0)):
                    break
                if controller.remote_controller.button[amp.KeyMap.select] == 1:
                    print("Remote Select received.")
                    break
                if config.run_duration > 0.0 and time.monotonic() - start_time >= config.run_duration:
                    print(f"Run duration reached: {config.run_duration:.2f}s")
                    break
    except KeyboardInterrupt:
        print("Operator stop received.")
    finally:
        if stop_release_hold is not None:
            stop_release_hold()
        print(f"进入低层阻尼并保持 {damping_exit_s:.2f}s。")
        try:
            controller.send_damping_for(damping_exit_s)
        finally:
            controller.close()
        if config.recover_native_on_exit:
            amp.recover_native_on_exit(config)
    print("Exit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy ArmHack Stand policy with SPACE-switched arm trajectories.")
    parser.add_argument("net", nargs="?", help="Unitree DDS network interface, e.g. enp11s0.")
    parser.add_argument("config_name", nargs="?", default="g1_amp.yaml")
    parser.add_argument("--self-test", action="store_true", help="Validate data/model only; never initialize DDS.")
    parser.add_argument("--policy", type=Path, help="ONNX actor used by --self-test.")
    parser.add_argument("--presets", type=Path, help="Arm preset JSON used by --self-test.")
    parser.add_argument("--config", type=Path, help="Real deployment YAML used by --self-test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        if args.policy is None or args.presets is None or args.config is None:
            raise SystemExit("--self-test requires --policy, --presets and --config.")
        run_self_test(args.policy.resolve(), args.presets.resolve(), args.config.resolve())
        return
    if not args.net:
        raise SystemExit("Real mode requires the DDS network interface positional argument.")
    if not sys.stdin.isatty():
        raise SystemExit("Real mode requires an interactive TTY for ENTER/SPACE/q safety controls.")
    run_real(args.net, args.config_name)


if __name__ == "__main__":
    main()
