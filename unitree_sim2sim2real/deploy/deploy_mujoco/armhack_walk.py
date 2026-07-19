"""ArmHack Walk adapter for the generic S3 G1 MuJoCo AMP runner.

The 96 -> 29 actor is exported unchanged.  This adapter reproduces the
IsaacLab Walk contract by fixing the 14 arm targets outside the actor and by
returning the composed 29-D raw action for both PD control and the next
observation's ``last_action`` block.  A GLFW SPACE callback can toggle the
command between zero and one deployment-contract-validated fixed velocity.
Headless behavior tests may instead select a validated time-segment schedule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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


def load_walk_pose(path: Path, pose_name: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1 or payload.get("units") != "rad":
        raise ValueError("Walk arm pose file must use schema_version=1 and radians.")
    if payload.get("joint_order_per_arm") != [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]:
        raise ValueError("Walk arm pose joint order does not match the 7-DoF-per-arm contract.")
    matches = [entry for entry in payload.get("poses", []) if entry.get("name") == pose_name]
    if len(matches) != 1:
        available = [entry.get("name") for entry in payload.get("poses", [])]
        raise ValueError(f"Unknown or duplicate Walk pose '{pose_name}'; available={available}")
    values = np.asarray(matches[0].get("left", []) + matches[0].get("right", []), dtype=np.float32)
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Walk pose '{pose_name}' must contain 14 finite radians.")
    return values


def load_command_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Walk real-deployment contract must use schema_version=1.")
    if payload.get("robot_asset") != "s3_g1_29dof" or int(payload.get("control_frequency_hz", -1)) != 50:
        raise ValueError("Walk deployment contract must describe the 50 Hz S3 G1 policy.")
    bounds = payload.get("raw_command_component_bounds", {})
    lower = np.asarray(bounds.get("min"), dtype=np.float32)
    upper = np.asarray(bounds.get("max"), dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
        raise ValueError("Walk command contract has invalid component bounds.")
    return payload


def validate_fixed_command(command: np.ndarray, contract: dict[str, Any]) -> None:
    command = np.asarray(command, dtype=np.float32)
    bounds = contract["raw_command_component_bounds"]
    lower = np.asarray(bounds["min"], dtype=np.float32)
    upper = np.asarray(bounds["max"], dtype=np.float32)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("ArmHack Walk fixed command must contain three finite values.")
    if np.any(command < lower) or np.any(command > upper):
        raise ValueError(
            "ArmHack Walk fixed command is outside the raw Nav2 CSV component envelope: "
            f"command={command.tolist()} min={lower.tolist()} max={upper.tolist()}"
        )


def load_command_schedule(
    path: Path, scenario_name: str, contract: dict[str, Any]
) -> dict[str, Any]:
    """Load one finite, positive-duration command schedule from the test corpus."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Walk behavior schedule must use schema_version=1.")
    scenarios = payload.get("scenarios", {})
    if scenario_name not in scenarios:
        raise ValueError(
            f"Unknown Walk behavior scenario '{scenario_name}'; available={sorted(scenarios)}"
        )
    scenario = dict(scenarios[scenario_name])
    raw_segments = scenario.get("segments", [])
    if not raw_segments:
        raise ValueError(f"Walk behavior scenario '{scenario_name}' has no segments.")
    segments: list[dict[str, Any]] = []
    start_time = 0.0
    for index, raw_segment in enumerate(raw_segments):
        duration_s = float(raw_segment.get("duration_s", 0.0))
        command = np.asarray(raw_segment.get("command", []), dtype=np.float32)
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"Scenario '{scenario_name}' segment {index} duration must be positive.")
        validate_fixed_command(command, contract)
        segments.append(
            {
                "index": index,
                "name": str(raw_segment.get("name", f"segment_{index}")),
                "start_time": start_time,
                "end_time": start_time + duration_s,
                "duration_s": duration_s,
                "command": command,
            }
        )
        start_time += duration_s
    return {
        "name": scenario_name,
        "description": str(scenario.get("description", "")),
        "segments": segments,
        "duration_s": start_time,
    }


class ArmHackWalkAdapter:
    """Fix one arm pose and optionally toggle zero/fixed command with SPACE."""

    def __init__(self, config: dict[str, Any], policy_joint_names: list[str], default_angles: np.ndarray):
        self.pose_path = Path(str(config["armhack_walk_pose_path"])).expanduser().resolve()
        self.contract_path = Path(str(config["armhack_walk_contract_path"])).expanduser().resolve()
        if not self.pose_path.is_file():
            raise FileNotFoundError(f"Walk pose JSON does not exist: {self.pose_path}")
        if not self.contract_path.is_file():
            raise FileNotFoundError(f"Walk deployment contract does not exist: {self.contract_path}")
        self.pose_name = str(config.get("armhack_walk_pose_name", "pos2_down"))
        self.arm_target = load_walk_pose(self.pose_path, self.pose_name)
        self.contract = load_command_contract(self.contract_path)
        self.fixed_command = np.asarray(config["armhack_walk_fixed_command"], dtype=np.float32)
        validate_fixed_command(self.fixed_command, self.contract)
        self.command_active = bool(config.get("armhack_walk_start_active", True))
        schedule_path_text = str(config.get("armhack_walk_schedule_path", "")).strip()
        self.schedule_path = Path(schedule_path_text).expanduser().resolve() if schedule_path_text else None
        self.scenario_name = str(config.get("armhack_walk_scenario_name", "")).strip()
        self.schedule = None
        if self.schedule_path is not None:
            if not self.schedule_path.is_file():
                raise FileNotFoundError(f"Walk behavior schedule JSON does not exist: {self.schedule_path}")
            if not self.scenario_name:
                raise ValueError("armhack_walk_scenario_name is required with a schedule path.")
            self.schedule = load_command_schedule(
                self.schedule_path, self.scenario_name, self.contract
            )
        missing = sorted(set(ARM_JOINT_NAMES).difference(policy_joint_names))
        if missing:
            raise ValueError(f"Walk arm joints are absent from policy_joint_names: {missing}")
        self.arm_policy_indices = np.asarray([policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64)
        self.default_angles = np.asarray(default_angles, dtype=np.float32)
        if self.default_angles.shape != (29,):
            raise ValueError("ArmHack Walk requires 29 default joint angles.")
        print(
            "[ArmHack Walk] fixed arm/command adapter: "
            f"pose={self.pose_name} command={self.current_target_command(0.0).tolist()} "
            + (
                f"schedule={self.scenario_name} duration={self.schedule['duration_s']:.3f}s"
                if self.schedule is not None
                else "(GLFW SPACE toggles zero/fixed)"
            )
        )

    def initialize_state(self, data: Any, qpos_addresses: dict[str, int]) -> None:
        for name, value in zip(ARM_JOINT_NAMES, self.arm_target):
            data.qpos[qpos_addresses[name]] = float(value)

    def compose_action(self, network_action: np.ndarray) -> np.ndarray:
        action = np.asarray(network_action, dtype=np.float32)
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Walk actor must return 29 finite actions, got shape={action.shape}")
        executed = action.copy()
        executed[self.arm_policy_indices] = (
            self.arm_target - self.default_angles[self.arm_policy_indices]
        ) / 0.25
        return executed

    @property
    def has_schedule(self) -> bool:
        return self.schedule is not None

    def current_schedule_segment(self, sim_time: float) -> dict[str, Any] | None:
        if self.schedule is None:
            return None
        clamped_time = max(float(sim_time), 0.0)
        for segment in self.schedule["segments"]:
            if clamped_time < float(segment["end_time"]):
                return segment
        return self.schedule["segments"][-1]

    def current_target_command(self, sim_time: float = 0.0) -> np.ndarray:
        segment = self.current_schedule_segment(sim_time)
        if segment is not None:
            return np.asarray(segment["command"], dtype=np.float32).copy()
        return self.fixed_command.copy() if self.command_active else np.zeros(3, dtype=np.float32)

    def key_callback(self, keycode: int) -> None:
        if int(keycode) != 32:
            return
        if self.schedule is not None:
            print("[ArmHack Walk command] SPACE ignored while a test schedule is active.", flush=True)
            return
        self.command_active = not self.command_active
        state = "FIXED" if self.command_active else "ZERO"
        print(f"[ArmHack Walk command] SPACE -> {state} {self.current_target_command().tolist()}", flush=True)

    def summary(self) -> dict[str, Any]:
        return {
            "pose_name": self.pose_name,
            "arm_target_rad": [float(value) for value in self.arm_target],
            "fixed_command": [float(value) for value in self.fixed_command],
            "final_command_state": (
                f"schedule:{self.scenario_name}"
                if self.schedule is not None
                else ("fixed" if self.command_active else "zero")
            ),
            "scenario_name": self.scenario_name,
            "schedule_path": str(self.schedule_path) if self.schedule_path is not None else "",
            "schedule_duration_s": (
                float(self.schedule["duration_s"]) if self.schedule is not None else 0.0
            ),
            "pose_path": str(self.pose_path),
            "contract_path": str(self.contract_path),
            "source_nav2_csv_sha256": self.contract["source_nav2_csv_sha256"],
        }
