"""Real-robot controller for IsaacLab-exported Unitree G1 29DoF AMP policies.

Core classes:
    Config reads deployment YAML and Controller builds the same 96-D observation
    used by the G1 AMP policy before sending PD position targets to Unitree DDS.

Inputs/outputs:
    Inputs are a TorchScript/ONNX policy, Unitree low-state DDS stream, and
    wireless remote, fixed, joystick, or navigation velocity commands. Outputs are low-level motor
    commands for the 29 configured G1 joints. The script requires an external
    shell confirmation in scripts/deploy_real_g1_amp.sh before launch.

Usage:
    python deploy_real_g1_amp.py eth0 g1_amp.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import yaml
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import MotorMode, create_damping_cmd, create_zero_cmd, init_cmd_go, init_cmd_hg
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data


UNITREE_ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT_DIR = UNITREE_ROOT_DIR.parent
LEGGED_LAB_ROOT_DIR = PROJECT_ROOT_DIR / "legged_lab"


def _resolve_path(value: str) -> str:
    return (
        str(value)
        .replace("{PROJECT_ROOT_DIR}", str(PROJECT_ROOT_DIR))
        .replace("{UNITREE_ROOT_DIR}", str(UNITREE_ROOT_DIR))
        .replace("{LEGGED_GYM_ROOT_DIR}", str(UNITREE_ROOT_DIR))
        .replace("{LEGGED_LAB_ROOT_DIR}", str(LEGGED_LAB_ROOT_DIR))
    )


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value}")


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_yaml_list(name: str, default: list) -> list:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = yaml.safe_load(raw_value)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a YAML list, got: {raw_value}")
    return value


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return list(default)
    value = yaml.safe_load(raw_value)
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [int(item) for item in value]
    raise ValueError(f"{name} must be an int list or comma-separated ints, got: {raw_value}")


def _env_yaml_vector(name: str, default: list[float], size: int) -> np.ndarray:
    value = _env_yaml_list(name, default)
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got: {value}")
    return array


class PolicyRunner:
    def __init__(self, policy_path: str, runtime: str) -> None:
        self.policy_path = policy_path
        self.runtime = runtime.lower()
        if self.runtime == "auto":
            self.runtime = "onnx" if policy_path.endswith(".onnx") else "torchscript"
        if self.runtime in {"torch", "jit"}:
            self.runtime = "torchscript"
        if self.runtime == "torchscript":
            self.policy = torch.jit.load(policy_path, map_location="cpu")
            self.policy.eval()
            self.session = None
            self.input_name = ""
            self.output_name = ""
        elif self.runtime == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required for ONNX deployment. "
                    "Install it in the UNITREE_PYTHON environment."
                ) from exc
            self.policy = None
            self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            raise ValueError(f"Invalid policy_runtime: {runtime}")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.runtime == "torchscript":
            with torch.inference_mode():
                action = self.policy(torch.from_numpy(obs_batch)).detach().cpu().numpy()
        else:
            action = self.session.run([self.output_name], {self.input_name: obs_batch})[0]
        action = np.asarray(action, dtype=np.float32).squeeze()
        if action.shape != (29,):
            raise RuntimeError(f"Policy output shape must be (29,), got {action.shape}")
        return action


class LinuxJoystickCommandReader:
    """Non-blocking Linux joystick reader matching the MuJoCo sim2sim mapping."""

    JS_EVENT_AXIS = 0x02

    def __init__(self, config: "Config") -> None:
        self.device_path = config.joystick_device
        self.axis_max = max(float(config.joystick_axis_max), 1.0)
        self.deadzone = max(float(config.joystick_deadzone), 0.0)
        self.axis_values: dict[int, int] = {}
        self.fd = os.open(self.device_path, os.O_RDONLY | os.O_NONBLOCK)
        self.axis_lin_x = config.joystick_axis_lin_x
        self.axis_lin_y = config.joystick_axis_lin_y
        self.axis_yaw = config.joystick_axis_yaw
        self.sign_lin_x = config.joystick_sign_lin_x
        self.sign_lin_y = config.joystick_sign_lin_y
        self.sign_yaw = config.joystick_sign_yaw
        self.ranges = config.joystick_ranges
        print(
            "[INFO] Joystick command mode opened: "
            f"device={self.device_path} axes=(x:{self.axis_lin_x}, y:{self.axis_lin_y}, yaw:{self.axis_yaw}) "
            f"signs=({self.sign_lin_x}, {self.sign_lin_y}, {self.sign_yaw}) deadzone={self.deadzone}"
        )

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def poll(self) -> None:
        while True:
            try:
                event = os.read(self.fd, 8)
            except BlockingIOError:
                return
            if len(event) != 8:
                return
            _, value, event_type, number = struct.unpack("IhBB", event)
            if event_type & self.JS_EVENT_AXIS:
                self.axis_values[int(number)] = int(value)

    def _axis_unit(self, axis_id: int, sign: float) -> float:
        raw_value = float(self.axis_values.get(axis_id, 0)) / self.axis_max
        value = float(np.clip(raw_value * sign, -1.0, 1.0))
        if abs(value) < self.deadzone:
            return 0.0
        return value

    @staticmethod
    def _map_signed_range(unit_value: float, value_range: list[float]) -> float:
        negative_limit = min(float(value_range[0]), float(value_range[1]), 0.0)
        positive_limit = max(float(value_range[0]), float(value_range[1]), 0.0)
        if unit_value >= 0.0:
            return float(unit_value * positive_limit)
        return float(-abs(unit_value) * abs(negative_limit))

    def read_command(self) -> np.ndarray:
        self.poll()
        lin_x_unit = self._axis_unit(self.axis_lin_x, self.sign_lin_x)
        lin_y_unit = self._axis_unit(self.axis_lin_y, self.sign_lin_y)
        yaw_unit = self._axis_unit(self.axis_yaw, self.sign_yaw)
        return np.asarray(
            [
                self._map_signed_range(lin_x_unit, self.ranges["lin_vel_x"]),
                self._map_signed_range(lin_y_unit, self.ranges["lin_vel_y"]),
                self._map_signed_range(yaw_unit, self.ranges["yaw_rate"]),
            ],
            dtype=np.float32,
        )


class NavUdpCommandReceiver:
    """Receive latest base-frame velocity command from a small UDP interface."""

    def __init__(self, config: "Config") -> None:
        self.bind_host = config.nav_udp_bind_host
        self.port = config.nav_udp_port
        self.timeout_s = config.nav_command_timeout_s
        self.stale_behavior = config.nav_stale_behavior
        self.clip_min = config.nav_command_clip_min
        self.clip_max = config.nav_command_clip_max
        self.latest_command = np.zeros(3, dtype=np.float32)
        self.last_rx_time: Optional[float] = None
        self._next_error_print_time = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_host, self.port))
        self.sock.setblocking(False)
        print(
            "[INFO] Navigation UDP command receiver opened: "
            f"bind={self.bind_host}:{self.port} timeout={self.timeout_s:.3f}s stale={self.stale_behavior}"
        )

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _parse_payload(self, payload: bytes) -> np.ndarray:
        text = payload.decode("utf-8").strip()
        if not text:
            raise ValueError("empty UDP payload")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            parts = [item for item in text.replace(",", " ").split() if item]
            if len(parts) < 3:
                raise ValueError(f"expected 'vx vy yaw', got: {text!r}")
            command = [float(parts[0]), float(parts[1]), float(parts[2])]
        else:
            if isinstance(value, dict):
                vx = value.get("vx", value.get("linear_x", value.get("lin_x")))
                vy = value.get("vy", value.get("linear_y", value.get("lin_y")))
                yaw = value.get("yaw", value.get("yaw_rate", value.get("wz", value.get("omega_z"))))
                if vx is None or vy is None or yaw is None:
                    raise ValueError(f"JSON command must contain vx, vy, and yaw/yaw_rate/wz: {text!r}")
                command = [float(vx), float(vy), float(yaw)]
            elif isinstance(value, list) and len(value) >= 3:
                command = [float(value[0]), float(value[1]), float(value[2])]
            else:
                raise ValueError(f"unsupported JSON command payload: {text!r}")
        return np.clip(np.asarray(command, dtype=np.float32), self.clip_min, self.clip_max)

    def poll(self) -> None:
        while True:
            try:
                payload, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                self.latest_command = self._parse_payload(payload)
                self.last_rx_time = time.monotonic()
            except Exception as exc:
                now = time.monotonic()
                if now >= self._next_error_print_time:
                    print(f"[WARN] Ignoring invalid navigation command: {exc}")
                    self._next_error_print_time = now + 1.0

    def read_command(self) -> np.ndarray:
        self.poll()
        if self.last_rx_time is None:
            return np.zeros(3, dtype=np.float32)
        if self.timeout_s > 0.0 and time.monotonic() - self.last_rx_time > self.timeout_s:
            if self.stale_behavior == "hold":
                return self.latest_command.copy()
            return np.zeros(3, dtype=np.float32)
        return self.latest_command.copy()


class MockNavPublisher:
    """Publish a constant command at 50 Hz through the same UDP path as navigation."""

    def __init__(self, config: "Config") -> None:
        self.target_host = config.nav_mock_target_host
        self.port = config.nav_udp_port
        self.command = config.nav_mock_cmd
        self.rate_hz = max(config.nav_mock_rate_hz, 1.0)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="g1_amp_mock_nav_50hz", daemon=True)

    def start(self) -> None:
        print(
            "[INFO] Mock navigation publisher starting: "
            f"{self.rate_hz:.1f}Hz -> {self.target_host}:{self.port} command={self.command.tolist()}"
        )
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        payload = json.dumps(
            {"vx": float(self.command[0]), "vy": float(self.command[1]), "yaw": float(self.command[2])}
        ).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not self.stop_event.is_set():
                sock.sendto(payload, (self.target_host, self.port))
                time.sleep(period)
        finally:
            sock.close()


class TerminalKeyReader:
    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            raise RuntimeError("Terminal space handoff requires an interactive TTY.")
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout_s: float) -> str:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return ""
        char = sys.stdin.read(1)
        if char == "\x03":
            raise KeyboardInterrupt
        return char


class Config:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.control_dt = float(config["control_dt"])
        self.msg_type = config["msg_type"]
        self.imu_type = config["imu_type"]
        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]
        policy_path = os.environ.get("G1_AMP_POLICY_PATH", os.environ.get("POLICY_PATH", config["policy_path"]))
        self.policy_path = _resolve_path(policy_path)
        self.policy_runtime = os.environ.get(
            "G1_AMP_POLICY_RUNTIME", str(config.get("policy_runtime", "auto"))
        ).strip().lower()
        self.policy_joint_names = list(config["policy_joint_names"])
        self.motor_joint_names = list(config["motor_joint_names"])
        self.motor_indices = list(config["motor_indices"])
        self.default_angles = np.asarray(config["default_angles"], dtype=np.float32)
        self.kps = np.asarray(config["kps"], dtype=np.float32)
        self.kds = np.asarray(config["kds"], dtype=np.float32)
        self.ang_vel_scale = float(config["ang_vel_scale"])
        self.dof_pos_scale = float(config["dof_pos_scale"])
        self.dof_vel_scale = float(config["dof_vel_scale"])
        self.action_scale = float(config["action_scale"])
        self.cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
        self.max_cmd = np.asarray(config["max_cmd"], dtype=np.float32)
        self.command_mode = os.environ.get(
            "G1_AMP_COMMAND_MODE", str(config.get("command_mode", "remote"))
        ).strip().lower()
        self.command_mode = {
            "nav": "nav_udp",
            "navigation": "nav_udp",
            "udp": "nav_udp",
            "mock_nav": "nav_mock",
            "nav_loopback": "nav_mock",
        }.get(self.command_mode, self.command_mode)
        self.cmd_init = np.asarray(_env_yaml_list("G1_AMP_CMD_INIT", config.get("cmd_init", [0.0, 0.0, 0.0])), dtype=np.float32)
        self.run_duration = _env_float("G1_AMP_RUN_DURATION", float(config.get("run_duration_s", 0.0)))
        self.release_motion_mode = _env_bool(
            "G1_AMP_RELEASE_MOTION_MODE", bool(config.get("release_motion_mode", False))
        )
        self.handoff_mode = os.environ.get(
            "G1_AMP_HANDOFF_MODE", str(config.get("handoff_mode", "zero"))
        ).strip().lower()
        command_ramp_default = bool(config.get("command_ramp", self.command_mode == "fixed"))
        self.command_ramp = _env_bool("G1_AMP_COMMAND_RAMP", command_ramp_default)
        self.command_max_linear_accel = _env_float(
            "G1_AMP_COMMAND_MAX_LINEAR_ACCEL", float(config.get("command_max_linear_accel", 0.5))
        )
        self.command_max_yaw_accel = _env_float(
            "G1_AMP_COMMAND_MAX_YAW_ACCEL", float(config.get("command_max_yaw_accel", 0.8))
        )
        self.wait_for_button_a = _env_bool(
            "G1_AMP_WAIT_FOR_BUTTON_A", bool(config.get("wait_for_button_a", False))
        )
        self.default_move_s = _env_float(
            "G1_AMP_DEFAULT_MOVE_S", float(config.get("default_move_s", 2.0))
        )
        self.default_hold_s = _env_float(
            "G1_AMP_DEFAULT_HOLD_S", float(config.get("default_hold_s", 0.0))
        )
        self.release_handoff_warmup_s = _env_float(
            "G1_AMP_RELEASE_HANDOFF_WARMUP_S", float(config.get("release_handoff_warmup_s", 0.3))
        )
        self.release_handoff_after_s = _env_float(
            "G1_AMP_RELEASE_HANDOFF_AFTER_S", float(config.get("release_handoff_after_s", 0.2))
        )
        self.recover_native_on_exit = _env_bool(
            "G1_AMP_RECOVER_NATIVE_ON_EXIT", bool(config.get("recover_native_on_exit", False))
        )
        self.native_recover_mode = os.environ.get(
            "G1_AMP_NATIVE_RECOVER_MODE", str(config.get("native_recover_mode", "damp"))
        ).strip().lower()
        self.native_recover_strict = _env_bool(
            "G1_AMP_NATIVE_RECOVER_STRICT", bool(config.get("native_recover_strict", False))
        )
        self.native_target_mode = os.environ.get(
            "G1_AMP_NATIVE_TARGET_MODE", str(config.get("native_target_mode", "ai"))
        ).strip()
        self.native_start_service = os.environ.get(
            "G1_AMP_NATIVE_START_SERVICE", str(config.get("native_start_service", "ai_sport"))
        ).strip()
        self.native_loco_service = os.environ.get(
            "G1_AMP_NATIVE_LOCO_SERVICE", str(config.get("native_loco_service", "sport"))
        ).strip()
        self.native_recover_timeout_s = _env_float(
            "G1_AMP_NATIVE_RECOVER_TIMEOUT_S", float(config.get("native_recover_timeout_s", 20.0))
        )
        self.native_standup_wait_s = _env_float(
            "G1_AMP_NATIVE_STANDUP_WAIT_S", float(config.get("native_standup_wait_s", 8.0))
        )
        self.native_min_standup_s = _env_float(
            "G1_AMP_NATIVE_MIN_STANDUP_S", float(config.get("native_min_standup_s", 6.0))
        )
        self.native_start_fsm_ids = _env_int_list(
            "G1_AMP_NATIVE_START_FSM_IDS", list(config.get("native_start_fsm_ids", [801, 802, 500]))
        )
        self.native_locomotion_fsm_ids = _env_int_list(
            "G1_AMP_NATIVE_LOCOMOTION_FSM_IDS", list(config.get("native_locomotion_fsm_ids", [802, 801, 500]))
        )
        self.terminal_space_handoff = _env_bool(
            "G1_AMP_TERMINAL_SPACE_HANDOFF", bool(config.get("terminal_space_handoff", False))
        )
        self.command_print_period_s = _env_float(
            "G1_AMP_COMMAND_PRINT_PERIOD_S", float(config.get("command_print_period_s", 0.5))
        )
        self.command_denominator = self.cmd_scale * self.max_cmd

        joystick_ranges = dict(config.get("joystick_ranges", {}))
        self.joystick_ranges = {
            "lin_vel_x": _env_yaml_list(
                "G1_AMP_JOYSTICK_LIN_X_RANGE", list(joystick_ranges.get("lin_vel_x", [-0.2, 1.5]))
            ),
            "lin_vel_y": _env_yaml_list(
                "G1_AMP_JOYSTICK_LIN_Y_RANGE", list(joystick_ranges.get("lin_vel_y", [-0.25, 0.25]))
            ),
            "yaw_rate": _env_yaml_list(
                "G1_AMP_JOYSTICK_YAW_RANGE", list(joystick_ranges.get("yaw_rate", [-0.6, 0.6]))
            ),
        }
        self.joystick_device = os.environ.get(
            "G1_AMP_JOYSTICK_DEVICE", str(config.get("joystick_device", "/dev/input/js0"))
        )
        self.joystick_axis_lin_x = _env_int(
            "G1_AMP_JOYSTICK_AXIS_LIN_X", int(config.get("joystick_axis_lin_x", 1))
        )
        self.joystick_axis_lin_y = _env_int(
            "G1_AMP_JOYSTICK_AXIS_LIN_Y", int(config.get("joystick_axis_lin_y", 0))
        )
        self.joystick_axis_yaw = _env_int(
            "G1_AMP_JOYSTICK_AXIS_YAW", int(config.get("joystick_axis_yaw", 3))
        )
        self.joystick_sign_lin_x = _env_float(
            "G1_AMP_JOYSTICK_SIGN_LIN_X", float(config.get("joystick_sign_lin_x", -1.0))
        )
        self.joystick_sign_lin_y = _env_float(
            "G1_AMP_JOYSTICK_SIGN_LIN_Y", float(config.get("joystick_sign_lin_y", -1.0))
        )
        self.joystick_sign_yaw = _env_float(
            "G1_AMP_JOYSTICK_SIGN_YAW", float(config.get("joystick_sign_yaw", -1.0))
        )
        self.joystick_axis_max = _env_float(
            "G1_AMP_JOYSTICK_AXIS_MAX", float(config.get("joystick_axis_max", 32768.0))
        )
        self.joystick_deadzone = _env_float(
            "G1_AMP_JOYSTICK_DEADZONE", float(config.get("joystick_deadzone", 0.05))
        )

        self.nav_udp_bind_host = os.environ.get(
            "G1_AMP_NAV_UDP_BIND_HOST", str(config.get("nav_udp_bind_host", "0.0.0.0"))
        )
        self.nav_udp_port = _env_int("G1_AMP_NAV_UDP_PORT", int(config.get("nav_udp_port", 15050)))
        self.nav_command_timeout_s = _env_float(
            "G1_AMP_NAV_COMMAND_TIMEOUT_S", float(config.get("nav_command_timeout_s", 0.25))
        )
        self.nav_stale_behavior = os.environ.get(
            "G1_AMP_NAV_STALE_BEHAVIOR", str(config.get("nav_stale_behavior", "zero"))
        ).strip().lower()
        self.nav_command_clip_min = _env_yaml_vector(
            "G1_AMP_NAV_COMMAND_CLIP_MIN", list(config.get("nav_command_clip_min", (-self.max_cmd).tolist())), 3
        )
        self.nav_command_clip_max = _env_yaml_vector(
            "G1_AMP_NAV_COMMAND_CLIP_MAX", list(config.get("nav_command_clip_max", self.max_cmd.tolist())), 3
        )
        self.nav_mock_cmd = _env_yaml_vector(
            "G1_AMP_NAV_MOCK_CMD", list(config.get("nav_mock_cmd", [0.7, 0.0, 0.0])), 3
        )
        self.nav_mock_rate_hz = _env_float(
            "G1_AMP_NAV_MOCK_RATE_HZ", float(config.get("nav_mock_rate_hz", 50.0))
        )
        self.nav_mock_target_host = os.environ.get(
            "G1_AMP_NAV_MOCK_TARGET_HOST", str(config.get("nav_mock_target_host", "127.0.0.1"))
        )

        if set(self.policy_joint_names) != set(self.motor_joint_names):
            raise ValueError("policy_joint_names and motor_joint_names must contain the same joint names.")
        if len(self.policy_joint_names) != 29:
            raise ValueError("G1 AMP deployment expects 29 policy joints.")
        if self.policy_runtime not in {"auto", "onnx", "torchscript", "torch", "jit"}:
            raise ValueError(f"Invalid policy_runtime: {self.policy_runtime}")
        if self.command_mode not in {"remote", "fixed", "joystick", "nav_udp", "nav_mock"}:
            raise ValueError(f"Invalid command_mode: {self.command_mode}")
        if self.handoff_mode not in {"zero", "stand"}:
            raise ValueError(f"Invalid handoff_mode: {self.handoff_mode}")
        if self.native_recover_mode not in {"service", "damp", "stand", "locomotion"}:
            raise ValueError(f"Invalid native_recover_mode: {self.native_recover_mode}")
        if self.nav_stale_behavior not in {"zero", "hold"}:
            raise ValueError(f"Invalid nav_stale_behavior: {self.nav_stale_behavior}")
        if self.cmd_init.shape != (3,):
            raise ValueError("cmd_init must contain three values: [lin_x, lin_y, yaw_rate].")
        if self.default_move_s < 0.0:
            raise ValueError("default_move_s must be non-negative.")
        if self.default_hold_s < 0.0:
            raise ValueError("default_hold_s must be non-negative.")
        for key, value in self.joystick_ranges.items():
            if len(value) != 2:
                raise ValueError(f"joystick range {key} must contain [negative, positive], got: {value}")
        if np.any(self.nav_command_clip_min > self.nav_command_clip_max):
            raise ValueError("nav_command_clip_min must be <= nav_command_clip_max.")
        if np.any(np.abs(self.command_denominator) < 1.0e-6):
            raise ValueError("cmd_scale * max_cmd must be non-zero for all command axes.")
        self.policy_to_motor_order = [self.policy_joint_names.index(name) for name in self.motor_joint_names]
        self.motor_to_policy_order = [self.motor_joint_names.index(name) for name in self.policy_joint_names]


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.policy = PolicyRunner(config.policy_path, config.policy_runtime)

        self.qj_motor = np.zeros(29, dtype=np.float32)
        self.dqj_motor = np.zeros(29, dtype=np.float32)
        self.action_policy = np.zeros(29, dtype=np.float32)
        self.obs = np.zeros(96, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.command_physical = np.zeros(3, dtype=np.float32)
        self.counter = 0
        self._cmd_lock = threading.Lock()
        self._next_command_print_time = 0.0
        self.joystick_reader: Optional[LinuxJoystickCommandReader] = None
        self.nav_receiver: Optional[NavUdpCommandReceiver] = None
        self.nav_mock_publisher: Optional[MockNavPublisher] = None

        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr = MotorMode.PR
            self.mode_machine = 0
            self.lowcmd_publisher = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.low_state_hg_handler, 10)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()
            self.lowcmd_publisher = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.low_state_go_handler, 10)
        else:
            raise ValueError(f"Invalid msg_type: {config.msg_type}")

        self.wait_for_low_state()
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine, self.mode_pr)
        else:
            init_cmd_go(self.low_cmd, weak_motor=[])
        self._init_command_sources()

    def _init_command_sources(self) -> None:
        if self.config.command_mode == "joystick":
            self.joystick_reader = LinuxJoystickCommandReader(self.config)
        if self.config.command_mode in {"nav_udp", "nav_mock"}:
            self.nav_receiver = NavUdpCommandReceiver(self.config)
        if self.config.command_mode == "nav_mock":
            self.nav_mock_publisher = MockNavPublisher(self.config)
            self.nav_mock_publisher.start()

    def close(self) -> None:
        if self.nav_mock_publisher is not None:
            self.nav_mock_publisher.close()
            self.nav_mock_publisher = None
        if self.nav_receiver is not None:
            self.nav_receiver.close()
            self.nav_receiver = None
        if self.joystick_reader is not None:
            self.joystick_reader.close()
            self.joystick_reader = None

    def low_state_hg_handler(self, msg: LowStateHG) -> None:
        self.low_state = msg
        self.mode_machine = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def low_state_go_handler(self, msg: LowStateGo) -> None:
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]) -> None:
        with self._cmd_lock:
            cmd.crc = CRC().Crc(cmd)
            self.lowcmd_publisher.Write(cmd)

    def wait_for_low_state(self) -> None:
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self) -> None:
        print("Enter zero torque state. Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def zero_torque_until_space(self) -> None:
        print("Enter zero torque state. Press SPACE to switch to damping mode.")
        with TerminalKeyReader() as keys:
            while True:
                create_zero_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
                if keys.read_key(self.config.control_dt) == " ":
                    return

    def damping_until_space(self) -> None:
        print("Enter damping mode.")
        print("WARNING: Before pressing SPACE again, make sure the robot is in a good initial posture and both feet are touching the ground.")
        print("Press SPACE to start the ONNX policy. Press CTRL-C to abort and keep damping.")
        with TerminalKeyReader() as keys:
            while True:
                create_damping_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
                if keys.read_key(self.config.control_dt) == " ":
                    return

    def move_to_default_pos(self) -> None:
        print(f"Moving to default pos over {self.config.default_move_s:.3f}s.")
        total_time = self.config.default_move_s
        num_steps = max(1, int(total_time / self.config.control_dt))
        default_motor = self.config.default_angles[self.config.policy_to_motor_order]
        kp_motor = self.config.kps[self.config.policy_to_motor_order]
        kd_motor = self.config.kds[self.config.policy_to_motor_order]
        init_motor = np.zeros(29, dtype=np.float32)
        for index, motor_idx in enumerate(self.config.motor_indices):
            init_motor[index] = self.low_state.motor_state[motor_idx].q
        for step in range(num_steps):
            alpha = 1.0 if total_time <= 0.0 else min(1.0, (step + 1) / num_steps)
            target = init_motor * (1.0 - alpha) + default_motor * alpha
            self._write_motor_targets(target, kp_motor, kd_motor)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self) -> None:
        print(f"Default pose reached. Auto-start policy after {self.config.default_hold_s:.3f}s hold.")
        default_motor = self.config.default_angles[self.config.policy_to_motor_order]
        kp_motor = self.config.kps[self.config.policy_to_motor_order]
        kd_motor = self.config.kds[self.config.policy_to_motor_order]
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.config.default_hold_s:
            self._write_motor_targets(default_motor, kp_motor, kd_motor)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _write_motor_targets(self, target_motor: np.ndarray, kp_motor: np.ndarray, kd_motor: np.ndarray) -> None:
        for index, motor_idx in enumerate(self.config.motor_indices):
            self.low_cmd.motor_cmd[motor_idx].q = float(target_motor[index])
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(kp_motor[index])
            self.low_cmd.motor_cmd[motor_idx].kd = float(kd_motor[index])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

    def _current_motor_positions(self) -> np.ndarray:
        current_motor = np.zeros(29, dtype=np.float32)
        for index, motor_idx in enumerate(self.config.motor_indices):
            current_motor[index] = self.low_state.motor_state[motor_idx].q
        return current_motor

    def start_current_pos_hold_thread(self) -> Callable[[], None]:
        target_motor = self._current_motor_positions()
        kp_motor = self.config.kps[self.config.policy_to_motor_order]
        kd_motor = self.config.kds[self.config.policy_to_motor_order]
        stop_event = threading.Event()
        started_event = threading.Event()

        def hold_loop() -> None:
            started_event.set()
            while not stop_event.is_set():
                self._write_motor_targets(target_motor, kp_motor, kd_motor)
                self.send_cmd(self.low_cmd)
                time.sleep(self.config.control_dt)

        thread = threading.Thread(target=hold_loop, name="g1_amp_release_handoff_hold", daemon=False)
        thread.start()
        if not started_event.wait(timeout=1.0):
            raise RuntimeError("Release handoff lowcmd hold thread did not start.")

        def stop_hold() -> None:
            stop_event.set()
            thread.join(timeout=1.0)

        return stop_hold

    def _update_state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        for index, motor_idx in enumerate(self.config.motor_indices):
            self.qj_motor[index] = self.low_state.motor_state[motor_idx].q
            self.dqj_motor[index] = self.low_state.motor_state[motor_idx].dq
        qj_policy = self.qj_motor[self.config.motor_to_policy_order]
        dqj_policy = self.dqj_motor[self.config.motor_to_policy_order]
        quat = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if self.config.imu_type == "torso":
            waist_index_in_motor = self.config.motor_joint_names.index("waist_yaw_joint")
            waist_yaw = self.qj_motor[waist_index_in_motor]
            waist_yaw_omega = self.dqj_motor[waist_index_in_motor]
            quat, ang_vel = transform_imu_data(waist_yaw, waist_yaw_omega, quat, np.asarray([ang_vel], dtype=np.float32))
        return qj_policy, dqj_policy, quat, ang_vel

    def _remote_command_physical(self) -> np.ndarray:
        remote_cmd = np.asarray(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx], dtype=np.float32
        )
        remote_cmd = np.clip(remote_cmd, -1.0, 1.0)
        return remote_cmd * self.config.command_denominator

    def _target_command_physical(self) -> np.ndarray:
        if self.config.command_mode == "fixed":
            return self.config.cmd_init.copy()
        if self.config.command_mode == "remote":
            return self._remote_command_physical()
        if self.config.command_mode == "joystick":
            if self.joystick_reader is None:
                raise RuntimeError("Joystick command reader is not initialized.")
            return self.joystick_reader.read_command()
        if self.config.command_mode in {"nav_udp", "nav_mock"}:
            if self.nav_receiver is None:
                raise RuntimeError("Navigation command receiver is not initialized.")
            return self.nav_receiver.read_command()
        raise RuntimeError(f"Unsupported command mode: {self.config.command_mode}")

    def _apply_command_ramp(self, target_command: np.ndarray) -> np.ndarray:
        if not self.config.command_ramp:
            return target_command
        max_step = np.asarray(
            [
                self.config.command_max_linear_accel * self.config.control_dt,
                self.config.command_max_linear_accel * self.config.control_dt,
                self.config.command_max_yaw_accel * self.config.control_dt,
            ],
            dtype=np.float32,
        )
        delta = np.clip(target_command - self.command_physical, -max_step, max_step)
        return self.command_physical + delta

    def _normalized_command(self, command_physical: np.ndarray) -> np.ndarray:
        return np.clip(command_physical / self.config.command_denominator, -1.0, 1.0)

    def _maybe_print_command(self) -> None:
        if self.config.command_print_period_s <= 0.0:
            return
        now = time.monotonic()
        if now < self._next_command_print_time:
            return
        self._next_command_print_time = now + self.config.command_print_period_s
        print(
            f"[COMMAND {self.config.command_mode}] "
            f"vx={self.command_physical[0]: .3f} m/s, "
            f"vy={self.command_physical[1]: .3f} m/s, "
            f"yaw={self.command_physical[2]: .3f} rad/s",
            flush=True,
        )

    def run(self) -> None:
        self.counter += 1
        qj_policy, dqj_policy, quat, ang_vel = self._update_state_arrays()
        target_command = self._target_command_physical()
        self.command_physical = self._apply_command_ramp(target_command)
        self.cmd = self._normalized_command(self.command_physical)
        self._maybe_print_command()

        self.obs[0:3] = ang_vel * self.config.ang_vel_scale
        self.obs[3:6] = get_gravity_orientation(quat)
        self.obs[6:9] = self.command_physical
        self.obs[9:38] = (qj_policy - self.config.default_angles) * self.config.dof_pos_scale
        self.obs[38:67] = dqj_policy * self.config.dof_vel_scale
        self.obs[67:96] = self.action_policy

        self.action_policy = self.policy.infer(self.obs)
        target_policy = self.config.default_angles + self.action_policy * self.config.action_scale
        target_motor = target_policy[self.config.policy_to_motor_order]
        kp_motor = self.config.kps[self.config.policy_to_motor_order]
        kd_motor = self.config.kds[self.config.policy_to_motor_order]
        self._write_motor_targets(target_motor, kp_motor, kd_motor)
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


def release_motion_mode() -> None:
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()
    for _ in range(5):
        status, result = motion_switcher.CheckMode()
        if status != 0:
            raise RuntimeError(f"MotionSwitcher CheckMode failed with code {status}.")
        mode_name = result.get("name", "") if result else ""
        print(f"MotionSwitcher mode: {result}")
        if not mode_name:
            print("No high-level motion mode is active.")
            return
        release_status, _ = motion_switcher.ReleaseMode()
        if release_status != 0:
            raise RuntimeError(f"MotionSwitcher ReleaseMode failed with code {release_status}.")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status, result = motion_switcher.CheckMode()
            if status == 0 and not (result.get("name", "") if result else ""):
                print(f"MotionSwitcher mode: {result}")
                print("No high-level motion mode is active.")
                return
            time.sleep(0.02)
    raise RuntimeError("High-level motion mode is still active after ReleaseMode retries.")


def select_native_motion_mode(config: Config) -> None:
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()
    deadline = time.monotonic() + config.native_recover_timeout_s
    select_code = None
    while time.monotonic() < deadline:
        select_code, _ = motion_switcher.SelectMode(config.native_target_mode)
        print(f"MotionSwitcher.SelectMode({config.native_target_mode!r}) recovery: code={select_code}")
        if select_code != 0:
            time.sleep(1.0)
            continue
        code, result = motion_switcher.CheckMode()
        print(f"MotionSwitcher.CheckMode recovery poll: code={code} data={result}")
        if code == 0 and result and result.get("name", "") == config.native_target_mode:
            return
        time.sleep(1.0)
    raise RuntimeError(
        f"MotionSwitcher did not report native target mode {config.native_target_mode!r}; "
        f"latest SelectMode code={select_code}."
    )


def start_native_service(config: Config) -> None:
    if not config.native_start_service:
        print("Native service start skipped because G1_AMP_NATIVE_START_SERVICE is empty.")
        return
    robot_state = RobotStateClient()
    robot_state.SetTimeout(5.0)
    robot_state.Init()
    code = robot_state.ServiceSwitch(config.native_start_service, True)
    print(f"RobotState.ServiceSwitch({config.native_start_service!r}, True) recovery: code={code}")
    if code != 0:
        raise RuntimeError(f"ServiceSwitch({config.native_start_service}, True) failed with code {code}.")
    time.sleep(1.0)


def wait_loco_available(client: LocoClient, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code, version = client.GetServerApiVersion()
        print(f"Loco GetServerApiVersion recovery poll: code={code} value={version}")
        if code == 0:
            return
        time.sleep(0.5)
    raise RuntimeError("Loco RPC did not become available during native recovery.")


def poll_loco_fsm(client: LocoClient, target_fsm_ids: list[int], timeout_s: float, label: str) -> bool:
    target_values = {int(value) for value in target_fsm_ids}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        fsm_code, fsm_id = client.GetFsmId()
        mode_code, fsm_mode = client.GetFsmMode()
        print(
            f"Loco FSM recovery poll ({label}): "
            f"GetFsmId code={fsm_code} value={fsm_id}; "
            f"GetFsmMode code={mode_code} value={fsm_mode}"
        )
        if fsm_code == 0 and fsm_id is not None and int(fsm_id) in target_values:
            return True
        time.sleep(0.5)
    return False


def poll_loco_stand_stable(client: LocoClient, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        fsm_code, fsm_id = client.GetFsmId()
        mode_code, fsm_mode = client.GetFsmMode()
        print(
            "Loco FSM recovery poll (stable StandUp): "
            f"GetFsmId code={fsm_code} value={fsm_id}; "
            f"GetFsmMode code={mode_code} value={fsm_mode}"
        )
        if fsm_code == 0 and mode_code == 0 and fsm_id is not None and fsm_mode is not None:
            if int(fsm_id) in {500, 801, 802}:
                return True
            if int(fsm_id) == 4 and int(fsm_mode) == 0:
                return True
        time.sleep(0.5)
    return False


def request_loco_fsm(client: LocoClient, fsm_id: int, label: str) -> None:
    code = client.SetFsmId(fsm_id)
    print(f"Loco SetFsmId({fsm_id}) recovery {label}: code={code}")
    if code != 0:
        raise RuntimeError(f"Loco SetFsmId({fsm_id}) recovery {label} failed with code {code}.")


def recover_native_on_exit(config: Config) -> None:
    print("=====================================")
    print("  Recover Native Motion Mode")
    print("=====================================")
    print(
        f"target_mode={config.native_target_mode}, service={config.native_start_service}, "
        f"recover_mode={config.native_recover_mode}, strict={config.native_recover_strict}"
    )
    select_native_motion_mode(config)
    start_native_service(config)

    client = LocoClient(service_name=config.native_loco_service)
    client.SetTimeout(5.0)
    client.Init()
    wait_loco_available(client, config.native_recover_timeout_s)
    client.SetVelocity(0.0, 0.0, 0.0, 0.2)

    if config.native_recover_mode == "service":
        print("Native recovery stopped after service/RPC availability confirmation.")
        return

    request_loco_fsm(client, 1, "Damp")
    if not poll_loco_fsm(client, [1], config.native_recover_timeout_s, "Damp FSM 1"):
        raise RuntimeError("Native recovery did not confirm Damp/FSM 1.")
    if config.native_recover_mode == "damp":
        print("Native recovery stopped at Damp/FSM 1.")
        return

    print("Requesting native StandUp/FSM 4 after AMP exit.")
    standup_start = time.monotonic()
    code = client.StandUp()
    print(f"Loco StandUp recovery: code={code}")
    if code != 0:
        raise RuntimeError(f"Loco StandUp recovery failed with code {code}.")
    if not poll_loco_fsm(client, [4] + config.native_locomotion_fsm_ids, config.native_standup_wait_s, "Stand/FSM 4"):
        raise RuntimeError("Native recovery did not confirm StandUp/FSM 4.")
    if not poll_loco_stand_stable(client, config.native_standup_wait_s):
        raise RuntimeError("Native recovery did not confirm stable StandUp/FSM 4 fsm_mode=0.")
    standup_elapsed = time.monotonic() - standup_start
    if standup_elapsed < config.native_min_standup_s:
        settle_s = config.native_min_standup_s - standup_elapsed
        print(f"Waiting {settle_s:.2f}s more to satisfy minimum StandUp settle time.")
        time.sleep(settle_s)
    if config.native_recover_mode == "stand":
        print("Native recovery stopped at StandUp/FSM 4.")
        return

    for fsm_id in config.native_start_fsm_ids:
        code = client.SetFsmId(fsm_id)
        print(f"Loco SetFsmId({fsm_id}) recovery locomotion candidate: code={code}")
        if code != 0:
            continue
        candidate_timeout_s = min(config.native_recover_timeout_s, 4.0)
        if poll_loco_fsm(
            client,
            config.native_locomotion_fsm_ids,
            candidate_timeout_s,
            "native locomotion",
        ):
            client.SetVelocity(0.0, 0.0, 0.0, 0.2)
            print("Native locomotion FSM confirmed after AMP exit.")
            return
    message = (
        "Native recovery did not confirm locomotion FSM; "
        f"expected one of {config.native_locomotion_fsm_ids}. Robot remains in the last confirmed native state."
    )
    if config.native_recover_strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Unitree G1 29DoF AMP policy on real robot.")
    parser.add_argument("net", type=str, help="Network interface for Unitree DDS, e.g. eth0.")
    parser.add_argument("config", type=str, nargs="?", default="g1_amp.yaml", help="Config filename under deploy_real/configs.")
    args = parser.parse_args()
    config = Config(str(Path(__file__).resolve().parent / "configs" / args.config))
    ChannelFactoryInitialize(0, args.net)
    print("=====================================")
    print("  Real G1 AMP Deployment")
    print("=====================================")
    print(f"Policy Path  : {config.policy_path}")
    print(f"Policy Runtime: {config.policy_runtime}")
    print(f"Command Mode : {config.command_mode}")
    if config.command_mode == "remote":
        print("Command Src  : remote joystick (initial command is zero)")
    elif config.command_mode == "fixed":
        print(f"Fixed Command: {config.cmd_init.tolist()}")
    elif config.command_mode == "joystick":
        print(
            "Joystick    : "
            f"device={config.joystick_device} axes=({config.joystick_axis_lin_x},{config.joystick_axis_lin_y},{config.joystick_axis_yaw}) "
            f"ranges=x{config.joystick_ranges['lin_vel_x']} y{config.joystick_ranges['lin_vel_y']} yaw{config.joystick_ranges['yaw_rate']}"
        )
    elif config.command_mode == "nav_mock":
        print(
            f"Nav Mock    : {config.nav_mock_rate_hz:.1f}Hz command={config.nav_mock_cmd.tolist()} "
            f"via UDP {config.nav_udp_bind_host}:{config.nav_udp_port}"
        )
    else:
        print(f"Nav UDP     : bind={config.nav_udp_bind_host}:{config.nav_udp_port}")
    print(f"Command Ramp : {config.command_ramp}")
    print(f"Handoff Mode : {config.handoff_mode}")
    print(f"Release Mode : {config.release_motion_mode}")
    print(f"Terminal Space Handoff: {config.terminal_space_handoff}")
    print("Policy Start : terminal SPACE gated" if config.terminal_space_handoff else "Policy Start : automatic/direct policy rollout")
    print(f"Default Move : {config.default_move_s:.3f}s")
    print(f"Default Hold : {config.default_hold_s:.3f}s")
    print(f"Release Hold : warmup={config.release_handoff_warmup_s:.2f}s after={config.release_handoff_after_s:.2f}s")
    print(f"Recover Native: {config.recover_native_on_exit} mode={config.native_recover_mode}")
    print(f"Run Duration : {config.run_duration}")
    print("=====================================")
    controller = Controller(config)
    stop_release_hold: Optional[Callable[[], None]] = None
    handoff_t0: Optional[float] = None
    release_call_t0: Optional[float] = None
    try:
        if config.release_motion_mode:
            handoff_t0 = time.monotonic()
            print("Handoff timer started before lowcmd pre-warm.")
            stop_release_hold = controller.start_current_pos_hold_thread()
            if config.release_handoff_warmup_s > 0.0:
                print(f"Pre-warming lowcmd current-pose hold for {config.release_handoff_warmup_s:.2f}s before ReleaseMode().")
                time.sleep(config.release_handoff_warmup_s)
            release_call_t0 = time.monotonic()
            release_motion_mode()
            print(f"ReleaseMode confirmed in {time.monotonic() - release_call_t0:.3f}s.")
            if config.release_handoff_after_s > 0.0:
                print(f"Holding current pose for {config.release_handoff_after_s:.2f}s after ReleaseMode().")
                time.sleep(config.release_handoff_after_s)
            stop_release_hold()
            stop_release_hold = None
        if config.terminal_space_handoff:
            controller.zero_torque_until_space()
            # Do not enter damping mode before policy takeover.
            if config.default_move_s > 0.0 or config.default_hold_s > 0.0:
                controller.move_to_default_pos()
                controller.default_pos_state()
        elif config.handoff_mode == "zero":
            controller.zero_torque_state()
            if config.default_move_s > 0.0 or config.default_hold_s > 0.0:
                controller.move_to_default_pos()
                controller.default_pos_state()
        else:
            print("Direct policy handoff from current posture: no zero-torque, no damping, no ReleaseMode-triggered handoff stage.")
            if config.default_move_s > 0.0 or config.default_hold_s > 0.0:
                controller.move_to_default_pos()
                controller.default_pos_state()
            else:
                print("Skipping pre-policy default-pose interpolation/hold; first outgoing LowCmd will be the policy PD target.")
        start_time = time.monotonic()
        if handoff_t0 is None:
            handoff_t0 = start_time
        print(
            "Policy rollout starting: "
            f"{start_time - handoff_t0:.3f}s since handoff timer, "
            f"{(start_time - release_call_t0):.3f}s since ReleaseMode call start."
            if release_call_t0 is not None
            else "Policy rollout starting."
        )
        first_frame = True
        while True:
            controller.run()
            if first_frame:
                first_frame = False
                now = time.monotonic()
                print(
                    "First policy frame sent: "
                    f"{now - handoff_t0:.3f}s since handoff timer"
                    + (f", {now - release_call_t0:.3f}s since ReleaseMode call start." if release_call_t0 is not None else ".")
                )
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
            if config.run_duration > 0.0 and time.monotonic() - start_time >= config.run_duration:
                print(f"Run duration reached: {config.run_duration:.2f}s")
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
    finally:
        if stop_release_hold is not None:
            stop_release_hold()
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        controller.close()
        if config.recover_native_on_exit:
            recover_native_on_exit(config)
    print("Exit")


if __name__ == "__main__":
    main()