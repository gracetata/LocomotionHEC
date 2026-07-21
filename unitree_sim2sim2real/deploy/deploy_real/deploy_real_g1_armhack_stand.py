#!/usr/bin/env python3
"""ArmHack Stand real-robot runner with operator-switched arm-only trajectories.

The actor still controls all 29 policy outputs.  At every 50 Hz control frame,
the 14 arm entries are replaced by the current minimum-jerk arm position target;
the remaining waist/leg entries stay equal to the actor output.  The composed
action is written back to the next observation's action-history block.

Real mode intentionally requires an interactive terminal.  Before ENTER it
does not initialize DDS or send LowCmd, so the robot remains in its native
damping/standby mode.  ENTER authorizes low-level handoff and policy inference;
the actor then runs continuously while an arm-only CSV performs the shared
natural-down -> flat-default -> forward -> flat-default initialization.  SPACE
is accepted only after that initialization has completed.

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


@dataclass(frozen=True)
class ArmPresetContract:
    poses: tuple[ArmPose, ...]
    damping_pose: ArmPose
    ready_pose: ArmPose
    space_cycle_poses: tuple[ArmPose, ...]
    startup_csv_path: Path
    startup_duration_s: float
    transition_s: float


class ArmCsvProgram:
    """Named 50 Hz arm-only CSV with linear sample interpolation."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        with self.path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["time_s", *ARM_JOINT_NAMES]:
                raise ValueError(
                    "Startup CSV must contain exactly time_s + canonical 14 arm joints; "
                    f"got {reader.fieldnames}."
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"Startup CSV is empty: {self.path}")
        self.times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
        self.targets = np.asarray(
            [[float(row[name]) for name in ARM_JOINT_NAMES] for row in rows], dtype=np.float32
        )
        self.times -= self.times[0]
        if not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.targets)):
            raise ValueError(f"Startup CSV contains NaN/Inf: {self.path}")
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError(f"Startup CSV time_s must be strictly increasing: {self.path}")

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    def sample(self, elapsed_s: float) -> np.ndarray:
        sample_time = float(np.clip(elapsed_s, 0.0, self.times[-1]))
        upper = min(int(np.searchsorted(self.times, sample_time, side="left")), len(self.times) - 1)
        lower = max(upper - 1, 0)
        dt = float(self.times[upper] - self.times[lower])
        if dt <= 1.0e-10:
            return self.targets[upper].copy()
        alpha = (sample_time - float(self.times[lower])) / dt
        return ((1.0 - alpha) * self.targets[lower] + alpha * self.targets[upper]).astype(np.float32)


def load_arm_presets(
    path: Path,
    verify_sources: bool = True,
) -> ArmPresetContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 2:
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
            if not lower <= float(value) <= upper:
                raise ValueError(
                    f"Pose {entry.get('id')} joint {name}={value:.6f} is outside "
                    f"hardware range [{lower:.6f}, {upper:.6f}]."
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
    if len(poses) < 4:
        raise ValueError("At least two arm poses are required for SPACE switching.")
    poses_by_id = {pose.pose_id: pose for pose in poses}
    if len(poses_by_id) != len(poses):
        raise ValueError("Arm preset ids must be unique.")

    def require_pose(pose_id: object, field: str) -> ArmPose:
        key = str(pose_id)
        if key not in poses_by_id:
            raise ValueError(f"Preset {field} references missing pose id: {key}")
        return poses_by_id[key]

    damping_pose = require_pose(payload.get("damping_pose_id"), "damping_pose_id")
    ready_pose = require_pose(payload.get("ready_pose_id"), "ready_pose_id")
    cycle_ids = payload.get("space_cycle_pose_ids")
    if not isinstance(cycle_ids, list) or len(cycle_ids) < 2:
        raise ValueError("space_cycle_pose_ids must contain at least two pose ids.")
    cycle_poses = tuple(require_pose(pose_id, "space_cycle_pose_ids") for pose_id in cycle_ids)
    if cycle_poses[0].pose_id != ready_pose.pose_id:
        raise ValueError("space_cycle_pose_ids must start from ready_pose_id.")
    if len({pose.pose_id for pose in cycle_poses}) != len(cycle_poses):
        raise ValueError("space_cycle_pose_ids must not contain duplicates.")

    startup = payload.get("startup")
    if not isinstance(startup, dict) or startup.get("policy_inference_active") is not True:
        raise ValueError("startup must require policy_inference_active=true.")
    sequence_ids = startup.get("sequence_pose_ids")
    expected_sequence_ids = [
        damping_pose.pose_id,
        ready_pose.pose_id,
        "F_forward_horizontal",
        ready_pose.pose_id,
    ]
    if sequence_ids != expected_sequence_ids:
        raise ValueError(
            "startup sequence must be natural-down -> flat-default -> forward-horizontal -> flat-default."
        )
    startup_path = (path.parent / str(startup.get("csv", ""))).resolve()
    if not startup_path.is_file():
        raise FileNotFoundError(f"Startup arm CSV does not exist: {startup_path}")
    startup_program = ArmCsvProgram(startup_path)
    startup_duration_s = float(startup.get("duration_s", -1.0))
    if not math.isclose(startup_program.duration_s, startup_duration_s, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(
            f"Startup CSV duration mismatch: json={startup_duration_s}, csv={startup_program.duration_s}."
        )
    forward_pose = require_pose("F_forward_horizontal", "startup.sequence_pose_ids")
    for label, actual, expected in (
        ("startup first row", startup_program.targets[0], damping_pose.positions),
        ("startup last row", startup_program.targets[-1], ready_pose.positions),
        ("startup forward stage", startup_program.sample(16.5), forward_pose.positions),
    ):
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{label} does not match the declared pose contract.")
    return ArmPresetContract(
        poses=tuple(poses),
        damping_pose=damping_pose,
        ready_pose=ready_pose,
        space_cycle_poses=cycle_poses,
        startup_csv_path=startup_path,
        startup_duration_s=startup_duration_s,
        transition_s=transition_s,
    )


def _verify_pose_source(preset_dir: Path, source: str, expected: np.ndarray) -> None:
    source_path_text, _, selector = source.partition(":")
    if selector not in {"", "first_row", "last_row"}:
        raise ValueError(f"Unsupported preset source selector: {selector}")
    source_path = (preset_dir / source_path_text).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Preset source CSV does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Preset source CSV is empty: {source_path}")
    row = rows[-1] if selector == "last_row" else rows[0]
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
    contract = load_arm_presets(preset_path, verify_sources=True)
    transition_s = _env_float("G1_ARMHACK_STAND_TRANSITION_S", contract.transition_s)
    if transition_s < 2.0:
        raise ValueError("G1_ARMHACK_STAND_TRANSITION_S must be >= 2.0s (the training range lower bound).")

    startup_program = ArmCsvProgram(contract.startup_csv_path)
    if not np.allclose(startup_program.sample(0.0), contract.damping_pose.positions, atol=1.0e-7):
        raise AssertionError("Post-ENTER startup does not begin at the natural-down damping pose.")
    if not np.allclose(
        startup_program.sample(startup_program.duration_s), contract.ready_pose.positions, atol=1.0e-7
    ):
        raise AssertionError("Post-ENTER startup does not finish at the flat default ready pose.")

    sequencer = ArmPresetSequencer(list(contract.space_cycle_poses), transition_s, start_time=0.0)
    before = sequencer.sample(0.0)
    sequencer.switch_next(0.0)
    start = sequencer.sample(0.0)
    middle = sequencer.sample(transition_s * 0.5)
    end = sequencer.sample(transition_s)
    if not np.array_equal(before, start) or not np.allclose(end, contract.space_cycle_poses[1].positions, atol=1.0e-7):
        raise AssertionError("Minimum-jerk switch endpoint/continuity test failed.")
    if not np.all(np.isfinite(middle)):
        raise AssertionError("Minimum-jerk transition produced non-finite values.")
    interrupted_before = sequencer.sample(transition_s * 0.5)
    sequencer.switch_next(transition_s * 0.5)
    interrupted_after = sequencer.sample(transition_s * 0.5)
    if not np.array_equal(interrupted_before, interrupted_after):
        raise AssertionError("Interrupted SPACE transition introduced a position discontinuity.")
    cycle_poses = contract.space_cycle_poses
    max_delta = max(
        float(np.max(np.abs(cycle_poses[(i + 1) % len(cycle_poses)].positions - cycle_poses[i].positions)))
        for i in range(len(cycle_poses))
    )
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
    print(f"  poses : {len(contract.poses)} arm-only presets; ids={[pose.pose_id for pose in contract.poses]}")
    print(
        "  modes : native damping/standby --ENTER--> policy-on debug; "
        "natural-down -> flat-default -> forward -> flat-default"
    )
    print(
        f"  init  : {startup_program.path}; {startup_program.duration_s:.3f}s; "
        "SPACE locked until complete"
    )
    print(f"  SPACE : {[pose.pose_id for pose in cycle_poses]}")
    print(f"  arms  : policy indices={arm_policy_indices}")
    print(f"  path  : minimum-jerk {transition_s:.3f}s; worst adjacent peak speed={max_velocity:.3f} rad/s")
    print("  target: direct default+action*scale output; no deployment-layer position clipping or slew limiting")


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
    contract = load_arm_presets(preset_path, verify_sources=True)
    startup_program = ArmCsvProgram(contract.startup_csv_path)
    transition_s = _env_float("G1_ARMHACK_STAND_TRANSITION_S", contract.transition_s)
    lowstate_timeout_s = _env_float("G1_ARMHACK_STAND_LOWSTATE_TIMEOUT_S", 0.20)
    max_tilt_rad = _env_float("G1_ARMHACK_STAND_MAX_TILT_RAD", 0.60)
    damping_exit_s = _env_float("G1_ARMHACK_STAND_DAMPING_EXIT_S", 1.0)
    joint_print_hz = _env_float("G1_ARMHACK_STAND_JOINT_PRINT_HZ", 1.0)
    damping_arm_max_error_rad = _env_float("G1_ARMHACK_STAND_DAMPING_ARM_MAX_ERROR_RAD", 0.45)
    damping_body_max_error_rad = _env_float("G1_ARMHACK_STAND_DAMPING_BODY_MAX_ERROR_RAD", 0.50)
    damping_upright_max_tilt_rad = _env_float("G1_ARMHACK_STAND_DAMPING_UPRIGHT_MAX_TILT_RAD", 0.20)
    for name, value in {
        "transition_s": transition_s,
        "lowstate_timeout_s": lowstate_timeout_s,
        "max_tilt_rad": max_tilt_rad,
        "damping_exit_s": damping_exit_s,
        "damping_arm_max_error_rad": damping_arm_max_error_rad,
        "damping_body_max_error_rad": damping_body_max_error_rad,
        "damping_upright_max_tilt_rad": damping_upright_max_tilt_rad,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    if transition_s < 2.0:
        raise ValueError("transition_s must be >= 2.0s (the training range lower bound).")
    class StandController(amp.Controller):
        def __init__(self) -> None:
            self.last_lowstate_time = time.monotonic()
            super().__init__(config)
            self.arm_policy_indices = np.asarray(
                [config.policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64
            )
            self.balance_policy_indices = np.asarray(
                [index for index, name in enumerate(config.policy_joint_names) if name not in set(ARM_JOINT_NAMES)],
                dtype=np.int64,
            )
            self.hard_lower_policy = np.asarray(
                [JOINT_LIMITS_RAD[name][0] for name in config.policy_joint_names], dtype=np.float32
            )
            self.hard_upper_policy = np.asarray(
                [JOINT_LIMITS_RAD[name][1] for name in config.policy_joint_names], dtype=np.float32
            )
            self.sequencer = ArmPresetSequencer(list(contract.space_cycle_poses), transition_s, time.monotonic())
            self.startup_start_time: Optional[float] = None
            self.startup_complete = False
            self.startup_completion_announced = False
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

        @staticmethod
        def _validate_target(target_policy: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(target_policy)):
                raise RuntimeError("Policy/arm target contains NaN/Inf.")
            return target_policy

        def damping_target_policy(self) -> np.ndarray:
            target = config.default_angles.copy()
            target[self.arm_policy_indices] = contract.damping_pose.positions
            return target

        def verify_damping_start_pose(self) -> None:
            qj_policy, _dqj_policy, quat, _ang_vel = self._update_state_arrays()
            self._check_state(qj_policy, quat)
            roll, pitch = _quat_wxyz_to_roll_pitch(quat)
            if abs(roll) > damping_upright_max_tilt_rad or abs(pitch) > damping_upright_max_tilt_rad:
                raise RuntimeError(
                    "ENTER 后检测到机器人并非直立待机："
                    f"roll={roll:.3f}, pitch={pitch:.3f}, "
                    f"limit={damping_upright_max_tilt_rad:.3f} rad。拒绝低层接管。"
                )
            body_error = np.abs(
                qj_policy[self.balance_policy_indices] - config.default_angles[self.balance_policy_indices]
            )
            worst_body_local = int(np.argmax(body_error))
            worst_body_error = float(body_error[worst_body_local])
            worst_body_policy_index = int(self.balance_policy_indices[worst_body_local])
            if worst_body_error > damping_body_max_error_rad:
                raise RuntimeError(
                    "ENTER 后检测到腰腿并非 Stand 直立默认姿态："
                    f"{config.policy_joint_names[worst_body_policy_index]} error={worst_body_error:.3f} rad, "
                    f"limit={damping_body_max_error_rad:.3f} rad。拒绝低层接管。"
                )
            arm_error = np.abs(qj_policy[self.arm_policy_indices] - contract.damping_pose.positions)
            worst_index = int(np.argmax(arm_error))
            worst_error = float(arm_error[worst_index])
            if worst_error > damping_arm_max_error_rad:
                raise RuntimeError(
                    "ENTER 后检测到双臂并非自然下垂待机姿态："
                    f"{ARM_JOINT_NAMES[worst_index]} error={worst_error:.3f} rad, "
                    f"limit={damping_arm_max_error_rad:.3f} rad。拒绝低层接管。"
                )
            print(
                "[DAMPING CHECK] 全身直立且双臂自然下垂："
                f"roll={roll:.3f}, pitch={pitch:.3f}, "
                f"腰腿最大误差={worst_body_error:.3f} rad, 双臂最大误差={worst_error:.3f} rad，通过。"
            )

        def activate_policy_initialization(self, now: float) -> None:
            self.startup_start_time = float(now)
            self.startup_complete = False
            self.startup_completion_announced = False
            damping_target = self.damping_target_policy()
            self.action_policy = (damping_target - config.default_angles) / config.action_scale
            self.sequencer = ArmPresetSequencer(
                list(contract.space_cycle_poses), transition_s, float(now) + startup_program.duration_s
            )

        def current_arm_target(self, now: float) -> np.ndarray:
            if self.startup_start_time is None:
                raise RuntimeError("Policy initialization has not been activated by ENTER.")
            elapsed = max(float(now) - self.startup_start_time, 0.0)
            if elapsed < startup_program.duration_s:
                return startup_program.sample(elapsed)
            self.startup_complete = True
            if not self.startup_completion_announced:
                self.startup_completion_announced = True
                print(
                    "[INIT COMPLETE] 双臂已完成 自然下垂→平直默认→向前伸直→收回平直默认；"
                    "SPACE 已解锁。",
                    flush=True,
                )
            return self.sequencer.sample(now)

        def run_frame(self, key: str) -> bool:
            now = time.monotonic()
            if key.lower() == "q":
                return False
            if key == " ":
                if not self.startup_complete:
                    print("[SPACE LOCKED] 自动初始化尚未完成，忽略本次空格。", flush=True)
                else:
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
            arm_target = self.current_arm_target(now)
            executed_action[self.arm_policy_indices] = (
                arm_target - config.default_angles[self.arm_policy_indices]
            ) / config.action_scale
            target_policy = config.default_angles + executed_action * config.action_scale
            target_policy = self._validate_target(target_policy)
            # Match deploy_real_g1_amp.py: action history records the direct composed target.
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

    print("============================================================")
    print("  ArmHack Stand Real Controller")
    print("============================================================")
    print(f"policy={config.policy_path}")
    print(f"startup_csv={startup_program.path} ({startup_program.duration_s:.2f}s)")
    print(f"arm presets={[pose.pose_id for pose in contract.poses]}")
    print(f"SPACE cycle={[pose.pose_id for pose in contract.space_cycle_poses]}")
    print(f"transition={transition_s:.2f}s, command=[0,0,0]")
    print("============================================================")
    print("[DAMPING / STANDBY] 尚未初始化 DDS，也未发送 LowCmd。")
    print("确认机器人全身直立、双臂自然下垂、吊架/急停/遥控器和周围环境安全。")
    answer = input("按 ENTER 进入调试模式并启动 policy；输入 q 后 ENTER 取消：")
    if answer.strip().lower() == "q":
        print("Operator cancelled before DDS initialization; no robot command was sent.")
        return

    amp.ChannelFactoryInitialize(0, net)
    controller = StandController()
    stop_release_hold = None
    lowlevel_active = False
    try:
        stop_release_hold = controller.start_current_pos_hold_thread()
        time.sleep(max(config.release_handoff_warmup_s, 0.3))
        controller.verify_damping_start_pose()
        amp.release_motion_mode()
        lowlevel_active = True
        time.sleep(max(config.release_handoff_after_s, 0.2))
        stop_release_hold()
        stop_release_hold = None
        start_time = time.monotonic()
        controller.activate_policy_initialization(start_time)
        print("[DEBUG MODE / POLICY ON] 高层 motion mode 已释放，29DoF actor 从现在起每帧推理。")
        print(
            "[AUTO INIT] 自然下垂→平直默认→向前伸直→收回平直默认；"
            f"总长 {startup_program.duration_s:.2f}s。初始化完成前 SPACE 锁定。"
        )
        print("初始化完成后 SPACE=下一双臂姿态；q/Ctrl-C/遥控器 Select=退出并进入阻尼。")
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
        try:
            if lowlevel_active:
                print(f"进入低层阻尼并保持 {damping_exit_s:.2f}s。")
                controller.send_damping_for(damping_exit_s)
            else:
                print("低层接管未完成；不发送额外 LowCmd。")
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
