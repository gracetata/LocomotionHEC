#!/usr/bin/env python3
"""Deploy ArmHack Walk with fixed/Joystick velocity and switchable arm poses.

The actor stays 96 -> 29.  Fourteen arm entries are replaced by one selected
Walk pose after inference and the composed action is stored in the next
observation.  SPACE cycles the three arm poses with minimum-jerk interpolation;
fixed mode uses V to switch between zero and its configured velocity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

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

JOINT_LIMITS_RAD = {
    "left_hip_pitch_joint": (-2.5307, 2.8798), "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576), "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236), "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798), "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576), "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236), "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618), "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704), "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618), "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222), "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704), "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618), "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222), "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _minimum_jerk(alpha: float) -> float:
    value = float(np.clip(alpha, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def load_walk_pose_set(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1 or payload.get("units") != "rad":
        raise ValueError("Walk pose JSON must use schema_version=1 and radians.")
    expected_order = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    if payload.get("joint_order_per_arm") != expected_order:
        raise ValueError("Walk pose JSON has an incompatible per-arm joint order.")
    entries = payload.get("poses", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Walk pose JSON must contain at least one pose.")
    pose_names: list[str] = []
    pose_values: list[np.ndarray] = []
    for entry in entries:
        pose_name = str(entry.get("name", "")).strip()
        if not pose_name or pose_name in pose_names:
            raise ValueError(f"Walk pose names must be non-empty and unique; got {pose_name!r}.")
        values = np.asarray(entry.get("left", []) + entry.get("right", []), dtype=np.float32)
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Walk pose {pose_name} must contain 14 finite radians.")
        for joint_name, value in zip(ARM_JOINT_NAMES, values):
            lower, upper = JOINT_LIMITS_RAD[joint_name]
            if not lower <= float(value) <= upper:
                raise ValueError(
                    f"Walk pose {pose_name}: {joint_name}={value:.6f} outside hardware range [{lower}, {upper}]."
                )
        pose_names.append(pose_name)
        pose_values.append(values)
    return tuple(pose_names), np.stack(pose_values)


def load_walk_pose(path: Path, pose_name: str) -> np.ndarray:
    pose_names, pose_values = load_walk_pose_set(path)
    if pose_names.count(pose_name) != 1:
        raise ValueError(f"POSE_NAME must select exactly one Walk pose; got {pose_name}")
    return pose_values[pose_names.index(pose_name)].copy()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Walk deployment contract must use schema_version=1.")
    if payload.get("robot_asset") != "s3_g1_29dof" or int(payload.get("control_frequency_hz", -1)) != 50:
        raise ValueError("Walk deployment contract must describe 50 Hz S3 G1.")
    return payload


def validate_command(command: np.ndarray, contract: dict[str, Any]) -> None:
    command = np.asarray(command, dtype=np.float32)
    bounds = contract["raw_command_component_bounds"]
    lower = np.asarray(bounds["min"], dtype=np.float32)
    upper = np.asarray(bounds["max"], dtype=np.float32)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("FIXED_COMMAND must contain three finite values.")
    if np.any(command < lower) or np.any(command > upper):
        raise ValueError(f"FIXED_COMMAND {command.tolist()} outside raw Nav2 range [{lower.tolist()}, {upper.tolist()}].")


def validate_joystick_ranges(ranges: np.ndarray, contract: dict[str, Any]) -> None:
    values = np.asarray(ranges, dtype=np.float32)
    bounds = contract["raw_command_component_bounds"]
    contract_lower = np.asarray(bounds["min"], dtype=np.float32)
    contract_upper = np.asarray(bounds["max"], dtype=np.float32)
    if values.shape != (3, 2) or not np.all(np.isfinite(values)):
        raise ValueError("Joystick ranges must be three finite [min,max] pairs.")
    if np.any(values[:, 0] > values[:, 1]):
        raise ValueError("Each Joystick range must satisfy min <= max.")
    if np.any(values[:, 0] < contract_lower) or np.any(values[:, 1] > contract_upper):
        raise ValueError(
            f"Joystick ranges {values.tolist()} exceed raw Nav2 bounds "
            f"[{contract_lower.tolist()}, {contract_upper.tolist()}]."
        )


def _load_config_contract(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy_names = list(config["policy_joint_names"])
    motor_names = list(config["motor_joint_names"])
    if len(policy_names) != 29 or set(policy_names) != set(JOINT_LIMITS_RAD):
        raise ValueError("Real config must contain the exact S3 G1 29DoF policy joint set.")
    if len(motor_names) != 29 or set(motor_names) != set(policy_names):
        raise ValueError("Real config motor_joint_names must contain the same 29 joints.")
    for key in ("motor_indices", "default_angles", "kps", "kds"):
        if len(config[key]) != 29:
            raise ValueError(f"Real config {key} must contain 29 entries.")
    if float(config["control_dt"]) != 0.02 or float(config["action_scale"]) != 0.25:
        raise ValueError("ArmHack Walk requires control_dt=0.02 and action_scale=0.25.")
    return config


def run_self_test(
    policy_path: Path,
    pose_path: Path,
    contract_path: Path,
    config_path: Path,
    pose_name: str,
    command_mode: str,
    command: np.ndarray,
    joystick_ranges: np.ndarray,
) -> None:
    config = _load_config_contract(config_path)
    contract = load_contract(contract_path)
    validate_command(command, contract)
    validate_joystick_ranges(joystick_ranges, contract)
    pose_names, pose_values = load_walk_pose_set(pose_path)
    if pose_name not in pose_names:
        raise ValueError(f"POSE_NAME {pose_name!r} not found; available={list(pose_names)}")
    pose = pose_values[pose_names.index(pose_name)]
    if command_mode not in {"fixed", "joystick"}:
        raise ValueError("COMMAND_MODE must be fixed or joystick.")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy_path), sess_options=options, providers=["CPUExecutionProvider"])
    if [(item.name, list(item.shape)) for item in session.get_inputs()] != [("obs", [1, 96])]:
        raise ValueError("Walk actor input must be obs[1,96].")
    if [(item.name, list(item.shape)) for item in session.get_outputs()] != [("actions", [1, 29])]:
        raise ValueError("Walk actor output must be actions[1,29].")
    rng = np.random.default_rng(20260718)
    for obs in [np.zeros((1, 96), dtype=np.float32)] + [rng.normal(0, 0.5, (1, 96)).astype(np.float32) for _ in range(16)]:
        action = session.run(["actions"], {"obs": obs})[0]
        if action.shape != (1, 29) or not np.all(np.isfinite(action)):
            raise ValueError("Walk actor failed finite [1,29] inference check.")
    names = list(config["policy_joint_names"])
    arm_indices = np.asarray([names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64)
    default = np.asarray(config["default_angles"], dtype=np.float32)
    composed = np.zeros(29, dtype=np.float32)
    composed[arm_indices] = (pose - default[arm_indices]) / 0.25
    if not np.allclose(default[arm_indices] + 0.25 * composed[arm_indices], pose, atol=1e-7):
        raise AssertionError("Walk arm action composition failed.")
    print("[SELF-TEST PASS] 未初始化 Unitree DDS，未发送机器人命令。")
    print(f"  actor   : obs[1,96] -> actions[1,29], zero + 16 random inputs finite")
    print(f"  poses   : {' -> '.join(pose_names)}; start={pose_name}; arm policy indices={arm_indices.tolist()}")
    if command_mode == "fixed":
        print(f"  command : V switches zero <-> {command.tolist()}, inside raw Nav2 CSV bounds")
    else:
        print(f"  command : Joystick ranges={joystick_ranges.tolist()}, inside raw Nav2 CSV bounds")
    print("  history : composed 29-D action reproduces the selected 14-D arm target")


def _quat_wxyz_to_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    norm = float(np.linalg.norm(quat))
    if not 0.5 <= norm <= 1.5:
        raise RuntimeError(f"Invalid IMU quaternion norm: {norm:.6f}")
    w, x, y, z = (quat / norm).tolist()
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def run_real(net: str, config_name: str) -> None:
    if os.environ.get("G1_ARMHACK_WALK_CONFIRM") != "I_UNDERSTAND":
        raise RuntimeError("Real mode requires G1_ARMHACK_WALK_CONFIRM=I_UNDERSTAND.")
    import deploy_real_g1_amp as amp

    config_path = Path(__file__).resolve().parent / "configs" / config_name
    config = amp.Config(str(config_path))
    if config.msg_type != "hg" or len(config.motor_indices) != 29:
        raise RuntimeError("ArmHack Walk real deployment only supports G1 29DoF HG LowCmd.")
    if config.command_mode not in {"fixed", "joystick"} or not config.release_motion_mode:
        raise RuntimeError("Walk deployment requires fixed/joystick command mode and ReleaseMode handoff.")

    pose_path = Path(os.environ["G1_ARMHACK_WALK_POSE_PATH"]).expanduser().resolve()
    contract_path = Path(os.environ["G1_ARMHACK_WALK_CONTRACT_PATH"]).expanduser().resolve()
    pose_name = os.environ.get("G1_ARMHACK_WALK_POSE_NAME", "pos2_down")
    pose_names, pose_values = load_walk_pose_set(pose_path)
    if pose_name not in pose_names:
        raise ValueError(f"POSE_NAME {pose_name!r} not found; available={list(pose_names)}")
    initial_pose_index = pose_names.index(pose_name)
    pose = pose_values[initial_pose_index]
    contract = load_contract(contract_path)
    fixed_command = np.asarray(config.cmd_init, dtype=np.float32)
    validate_command(fixed_command, contract)
    joystick_ranges = np.asarray(
        [
            config.joystick_ranges["lin_vel_x"],
            config.joystick_ranges["lin_vel_y"],
            config.joystick_ranges["yaw_rate"],
        ],
        dtype=np.float32,
    )
    validate_joystick_ranges(joystick_ranges, contract)
    startup_move_s = _env_float("G1_ARMHACK_WALK_STARTUP_MOVE_S", 5.0)
    arm_pose_switch_s = _env_float("G1_ARMHACK_WALK_ARM_POSE_SWITCH_S", 2.0)
    lowstate_timeout_s = _env_float("G1_ARMHACK_WALK_LOWSTATE_TIMEOUT_S", 0.20)
    max_tilt_rad = _env_float("G1_ARMHACK_WALK_MAX_TILT_RAD", 0.60)
    damping_exit_s = _env_float("G1_ARMHACK_WALK_DAMPING_EXIT_S", 1.0)
    joint_print_hz = _env_float("G1_ARMHACK_WALK_JOINT_PRINT_HZ", 1.0)
    if startup_move_s < 3.0 or arm_pose_switch_s <= 0.0 or lowstate_timeout_s <= 0.0 or max_tilt_rad <= 0.0 or damping_exit_s <= 0.0:
        raise ValueError("Invalid positive Walk safety/startup parameter.")

    class WalkController(amp.Controller):
        def __init__(self) -> None:
            self.last_lowstate_time = time.monotonic()
            super().__init__(config)
            self.arm_indices = np.asarray([config.policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64)
            self.lower = np.asarray([JOINT_LIMITS_RAD[name][0] for name in config.policy_joint_names], dtype=np.float32)
            self.upper = np.asarray([JOINT_LIMITS_RAD[name][1] for name in config.policy_joint_names], dtype=np.float32)
            self.command_active = False
            self.arm_pose_index = initial_pose_index
            self.arm_pose_start = pose.copy()
            self.arm_pose_target = pose.copy()
            self.arm_transition_started = time.monotonic() - arm_pose_switch_s
            self.next_joint_print_time = 0.0

        def low_state_hg_handler(self, msg: Any) -> None:
            super().low_state_hg_handler(msg)
            self.last_lowstate_time = time.monotonic()

        def low_state_go_handler(self, msg: Any) -> None:
            super().low_state_go_handler(msg)
            self.last_lowstate_time = time.monotonic()

        def _check_state(self, qj: np.ndarray, quat: np.ndarray) -> None:
            age = time.monotonic() - self.last_lowstate_time
            if age > lowstate_timeout_s:
                raise RuntimeError(f"LowState stale for {age:.3f}s (limit {lowstate_timeout_s:.3f}s).")
            if not np.all(np.isfinite(qj)) or not np.all(np.isfinite(quat)):
                raise RuntimeError("LowState contains NaN/Inf.")
            if np.any(qj < self.lower - 0.02) or np.any(qj > self.upper + 0.02):
                bad = np.flatnonzero((qj < self.lower - 0.02) | (qj > self.upper + 0.02))
                raise RuntimeError("Measured joint outside hardware range: " + ", ".join(config.policy_joint_names[i] for i in bad))
            roll, pitch = _quat_wxyz_to_roll_pitch(quat)
            if abs(roll) > max_tilt_rad or abs(pitch) > max_tilt_rad:
                raise RuntimeError(f"Torso tilt exceeded: roll={roll:.3f}, pitch={pitch:.3f}, limit={max_tilt_rad:.3f}.")

        def _bounded_target(self, target: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(target)):
                raise RuntimeError("Policy/arm target contains NaN/Inf.")
            return np.clip(target, self.lower, self.upper)

        def _arm_target_at(self, now: float) -> np.ndarray:
            alpha = (now - self.arm_transition_started) / arm_pose_switch_s
            blend = _minimum_jerk(alpha)
            return self.arm_pose_start + (self.arm_pose_target - self.arm_pose_start) * blend

        def _cycle_arm_pose(self, now: float) -> None:
            current = self._arm_target_at(now)
            self.arm_pose_index = (self.arm_pose_index + 1) % len(pose_names)
            self.arm_pose_start = current.copy()
            self.arm_pose_target = pose_values[self.arm_pose_index].copy()
            self.arm_transition_started = now
            print(
                f"[ARM POSE] -> {pose_names[self.arm_pose_index]} "
                f"(minimum-jerk {arm_pose_switch_s:.2f}s)",
                flush=True,
            )

        def hold_until_enter(self, target_policy: np.ndarray, prompt: str) -> None:
            target_motor = target_policy[config.policy_to_motor_order]
            kp = config.kps[config.policy_to_motor_order]
            kd = config.kds[config.policy_to_motor_order]
            print(prompt)
            print("按 ENTER 继续；q/Ctrl-C/遥控器 Select 立即退出并进入阻尼。")
            with amp.TerminalKeyReader() as keys:
                while True:
                    self._write_motor_targets(target_motor, kp, kd)
                    self.send_cmd(self.low_cmd)
                    key = keys.read_key(config.control_dt)
                    if key in {"\r", "\n"}:
                        return
                    if key.lower() == "q" or self.remote_controller.button[amp.KeyMap.select] == 1:
                        raise KeyboardInterrupt

        def move_arms_to_startup(self, start_policy: np.ndarray, target_policy: np.ndarray) -> None:
            kp = config.kps[config.policy_to_motor_order]
            kd = config.kds[config.policy_to_motor_order]
            steps = max(1, int(round(startup_move_s / config.control_dt)))
            print(f"用 minimum-jerk 在 {startup_move_s:.2f}s 内移动双臂到 {pose_name}；腰腿保持当前目标。")
            with amp.TerminalKeyReader() as keys:
                for step in range(steps):
                    blend = _minimum_jerk((step + 1) / steps)
                    target = start_policy + (target_policy - start_policy) * blend
                    self._write_motor_targets(target[config.policy_to_motor_order], kp, kd)
                    self.send_cmd(self.low_cmd)
                    key = keys.read_key(config.control_dt)
                    if key.lower() == "q" or self.remote_controller.button[amp.KeyMap.select] == 1:
                        raise KeyboardInterrupt

        def run_frame(self, key: str) -> bool:
            if key.lower() == "q":
                return False
            now = time.monotonic()
            if key == " ":
                self._cycle_arm_pose(now)
            if key.lower() == "v" and config.command_mode == "fixed":
                self.command_active = not self.command_active
                state = "FIXED" if self.command_active else "ZERO"
                command = fixed_command if self.command_active else np.zeros(3, dtype=np.float32)
                print(f"[COMMAND SWITCH] {state}: {command.tolist()}", flush=True)
            qj, dqj, quat, ang_vel = self._update_state_arrays()
            self._check_state(qj, quat)
            if config.command_mode == "fixed":
                target_command = fixed_command if self.command_active else np.zeros(3, dtype=np.float32)
            else:
                target_command = self._target_command_physical()
            self.command_physical = self._apply_command_ramp(target_command)
            self.obs[0:3] = ang_vel * config.ang_vel_scale
            self.obs[3:6] = amp.get_gravity_orientation(quat)
            self.obs[6:9] = self.command_physical
            self.obs[9:38] = (qj - config.default_angles) * config.dof_pos_scale
            self.obs[38:67] = dqj * config.dof_vel_scale
            self.obs[67:96] = self.action_policy
            network_action = self.policy.infer(self.obs)
            if network_action.shape != (29,) or not np.all(np.isfinite(network_action)):
                raise RuntimeError("Walk actor output is not finite shape (29,).")
            executed = network_action.copy()
            arm_target = self._arm_target_at(now)
            executed[self.arm_indices] = (arm_target - config.default_angles[self.arm_indices]) / config.action_scale
            target_policy = self._bounded_target(config.default_angles + executed * config.action_scale)
            self.action_policy = (target_policy - config.default_angles) / config.action_scale
            self._write_motor_targets(
                target_policy[config.policy_to_motor_order],
                config.kps[config.policy_to_motor_order],
                config.kds[config.policy_to_motor_order],
            )
            self.send_cmd(self.low_cmd)
            if joint_print_hz > 0.0 and now >= self.next_joint_print_time:
                self.next_joint_print_time = now + 1.0 / joint_print_hz
                print(
                    f"[Walk cmd] vx={self.command_physical[0]:+.3f} vy={self.command_physical[1]:+.3f} wz={self.command_physical[2]:+.3f}; "
                    f"pose={pose_names[self.arm_pose_index]}", flush=True
                )
            time.sleep(config.control_dt)
            return True

        def send_damping_for(self, duration: float) -> None:
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                amp.create_damping_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
                time.sleep(config.control_dt)

    amp.ChannelFactoryInitialize(0, net)
    print("============================================================")
    print(" ArmHack Walk Real Controller")
    print("============================================================")
    print(f"policy={config.policy_path}")
    print(f"pose={pose_name}; SPACE cycles {' -> '.join(pose_names)}")
    if config.command_mode == "fixed":
        print(f"command=fixed; starts ZERO; V toggles fixed={fixed_command.tolist()}")
    else:
        print(
            f"command=joystick; device={config.joystick_device}; "
            f"ranges={joystick_ranges.tolist()}"
        )
    print("q/Ctrl-C/remote Select -> low-level damping")
    print("============================================================")
    controller = WalkController()
    stop_hold = None
    try:
        stop_hold = controller.start_current_pos_hold_thread()
        time.sleep(max(config.release_handoff_warmup_s, 0.3))
        amp.release_motion_mode()
        time.sleep(max(config.release_handoff_after_s, 0.2))
        stop_hold()
        stop_hold = None
        print("[DEBUG MODE] 高层 motion mode 已释放，当前 LowCmd 保持实际姿态。")
        current_policy, _, _, _ = controller._update_state_arrays()
        controller.hold_until_enter(current_policy, "确认吊架、关节方向、急停后，准备只移动双臂到固定姿态。")
        startup_policy = current_policy.copy()
        startup_policy[controller.arm_indices] = pose
        startup_policy = np.clip(startup_policy, controller.lower, controller.upper)
        controller.move_arms_to_startup(current_policy, startup_policy)
        controller.hold_until_enter(startup_policy, "固定双臂姿态已到达；确认双脚着地后准备以零速度启动 Walk policy。")
        controller.action_policy = (startup_policy - config.default_angles) / config.action_scale
        controller.arm_pose_start = pose.copy()
        controller.arm_pose_target = pose.copy()
        controller.arm_transition_started = time.monotonic() - arm_pose_switch_s
        controller.command_physical.fill(0.0)
        start_time = time.monotonic()
        if config.command_mode == "fixed":
            print("[POLICY ON] 初始速度 [0,0,0]；V 切换固定速度/零速度；SPACE 切换双臂姿态。")
        else:
            print("[POLICY ON] Joystick 连续控制速度；SPACE 切换双臂姿态。")
        with amp.TerminalKeyReader() as keys:
            while True:
                if not controller.run_frame(keys.read_key(0.0)):
                    break
                if controller.remote_controller.button[amp.KeyMap.select] == 1:
                    break
                if config.run_duration > 0.0 and time.monotonic() - start_time >= config.run_duration:
                    print(f"Run duration reached: {config.run_duration:.2f}s")
                    break
    except KeyboardInterrupt:
        print("Operator stop received.")
    finally:
        if stop_hold is not None:
            stop_hold()
        print(f"进入低层阻尼并保持 {damping_exit_s:.2f}s。")
        try:
            controller.send_damping_for(damping_exit_s)
        finally:
            controller.close()
    print("Exit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("net", nargs="?")
    parser.add_argument("config_name", nargs="?", default="g1_amp.yaml")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pose-name", default="pos2_down")
    parser.add_argument("--command-mode", choices=("fixed", "joystick"), default="fixed")
    parser.add_argument("--fixed-command", nargs=3, type=float, default=[0.35, 0.0, 0.0])
    parser.add_argument("--joystick-lin-x-range", nargs=2, type=float, default=[-0.2, 0.6])
    parser.add_argument("--joystick-lin-y-range", nargs=2, type=float, default=[-0.3, 0.3])
    parser.add_argument("--joystick-yaw-range", nargs=2, type=float, default=[-0.5187280216217041, 0.6])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        if None in (args.policy, args.poses, args.contract, args.config):
            raise SystemExit("--self-test requires --policy, --poses, --contract and --config.")
        run_self_test(
            args.policy.resolve(), args.poses.resolve(), args.contract.resolve(), args.config.resolve(),
            args.pose_name, args.command_mode, np.asarray(args.fixed_command, dtype=np.float32),
            np.asarray(
                [args.joystick_lin_x_range, args.joystick_lin_y_range, args.joystick_yaw_range],
                dtype=np.float32,
            ),
        )
        return
    if not args.net:
        raise SystemExit("Real mode requires the DDS network interface positional argument.")
    if not sys.stdin.isatty():
        raise SystemExit("Real mode requires an interactive TTY for ENTER/SPACE/V/q controls.")
    run_real(args.net, args.config_name)


if __name__ == "__main__":
    main()
