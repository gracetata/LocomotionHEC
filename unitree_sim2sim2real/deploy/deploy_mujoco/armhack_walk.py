"""ArmHack Walk adapter for the generic S3 G1 MuJoCo AMP runner.

The 96 -> 29 actor is exported unchanged.  This adapter reproduces the
IsaacLab Walk contract by fixing the 14 arm targets outside the actor and by
returning the composed 29-D raw action for both PD control and the next
observation's ``last_action`` block.  Fixed-command tests retain the historical
SPACE zero/fixed toggle.  Interactive keyboard tests instead use SPACE/Z/X/C
to change arm poses while the generic keyboard reader controls velocity.
Headless behavior tests may select a validated time-segment schedule.
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


def load_walk_pose_catalog(path: Path) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
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
    names: list[str] = []
    poses: dict[str, np.ndarray] = {}
    for entry in payload.get("poses", []):
        name = str(entry.get("name", "")).strip()
        values = np.asarray(entry.get("left", []) + entry.get("right", []), dtype=np.float32)
        if not name or name in poses:
            raise ValueError(f"Walk pose names must be non-empty and unique: {name!r}")
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Walk pose '{name}' must contain 14 finite radians.")
        names.append(name)
        poses[name] = values
    if tuple(names) != ("pos1_back", "pos2_down", "pos3_front"):
        raise ValueError(
            "Walk pose order must be pos1_back -> pos2_down -> pos3_front; "
            f"got {names}"
        )
    return tuple(names), poses


def load_walk_pose(path: Path, pose_name: str) -> np.ndarray:
    names, poses = load_walk_pose_catalog(path)
    if pose_name not in poses:
        raise ValueError(f"Unknown Walk pose '{pose_name}'; available={list(names)}")
    return poses[pose_name].copy()


def minimum_jerk(alpha: float) -> float:
    """Quintic 0->1 blend with zero endpoint velocity and acceleration."""
    value = float(np.clip(alpha, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


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
                "pose_name": str(raw_segment.get("pose_name", "")).strip(),
                "policy": str(raw_segment.get("policy", "primary")).strip().lower(),
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
    """Compose arm targets and handle fixed-command or keyboard interaction."""

    def __init__(self, config: dict[str, Any], policy_joint_names: list[str], default_angles: np.ndarray):
        self.pose_path = Path(str(config["armhack_walk_pose_path"])).expanduser().resolve()
        self.contract_path = Path(str(config["armhack_walk_contract_path"])).expanduser().resolve()
        if not self.pose_path.is_file():
            raise FileNotFoundError(f"Walk pose JSON does not exist: {self.pose_path}")
        if not self.contract_path.is_file():
            raise FileNotFoundError(f"Walk deployment contract does not exist: {self.contract_path}")
        self.pose_names, self.pose_catalog = load_walk_pose_catalog(self.pose_path)
        self.pose_name = str(config.get("armhack_walk_pose_name", "pos2_down"))
        if self.pose_name not in self.pose_catalog:
            raise ValueError(
                f"Unknown Walk pose '{self.pose_name}'; available={list(self.pose_names)}"
            )
        self.arm_target = self.pose_catalog[self.pose_name].copy()
        self.keyboard_interactive = (
            str(config.get("command_mode", "independent")).lower() == "keyboard"
        )
        self.pose_transition_s = float(config.get("armhack_walk_pose_transition_s", 2.0))
        if not np.isfinite(self.pose_transition_s) or self.pose_transition_s <= 0.0:
            raise ValueError("ArmHack Walk pose transition must be positive and finite.")
        self._transition_start = self.arm_target.copy()
        self._transition_goal = self.arm_target.copy()
        self._transition_start_time = 0.0
        self._pending_pose_name: str | None = None
        self.pose_switch_count = 0
        self.contract = load_command_contract(self.contract_path)
        self.fixed_command = np.asarray(config["armhack_walk_fixed_command"], dtype=np.float32)
        validate_fixed_command(self.fixed_command, self.contract)
        self.command_active = bool(config.get("armhack_walk_start_active", True))
        schedule_path_text = str(config.get("armhack_walk_schedule_path", "")).strip()
        self.schedule_path = Path(schedule_path_text).expanduser().resolve() if schedule_path_text else None
        self.scenario_name = str(config.get("armhack_walk_scenario_name", "")).strip()
        self.schedule = None
        self._last_schedule_pose_index = -1
        if self.schedule_path is not None:
            if not self.schedule_path.is_file():
                raise FileNotFoundError(f"Walk behavior schedule JSON does not exist: {self.schedule_path}")
            if not self.scenario_name:
                raise ValueError("armhack_walk_scenario_name is required with a schedule path.")
            self.schedule = load_command_schedule(
                self.schedule_path, self.scenario_name, self.contract
            )
            for segment in self.schedule["segments"]:
                pose_name = str(segment.get("pose_name", ""))
                if pose_name and pose_name not in self.pose_catalog:
                    raise ValueError(f"Unknown scheduled Walk pose '{pose_name}'.")
                if segment.get("policy") not in {"primary", "secondary", "walk", "stand"}:
                    raise ValueError(f"Unknown scheduled policy role '{segment.get('policy')}'.")
        missing = sorted(set(ARM_JOINT_NAMES).difference(policy_joint_names))
        if missing:
            raise ValueError(f"Walk arm joints are absent from policy_joint_names: {missing}")
        self.arm_policy_indices = np.asarray([policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64)
        self.default_angles = np.asarray(default_angles, dtype=np.float32)
        if self.default_angles.shape != (29,):
            raise ValueError("ArmHack Walk requires 29 default joint angles.")
        print(
            "[ArmHack Walk] arm/command adapter: "
            f"pose={self.pose_name} "
            + (
                "command=keyboard "
                if self.keyboard_interactive
                else f"command={self.current_target_command(0.0).tolist()} "
            )
            + (
                f"schedule={self.scenario_name} duration={self.schedule['duration_s']:.3f}s"
                if self.schedule is not None
                else (
                    "(keyboard velocity; SPACE cycles arms; Z/X/C select arms)"
                    if self.keyboard_interactive
                    else "(GLFW SPACE toggles zero/fixed)"
                )
            )
        )

    def initialize_state(self, data: Any, qpos_addresses: dict[str, int]) -> None:
        for name, value in zip(ARM_JOINT_NAMES, self.arm_target):
            data.qpos[qpos_addresses[name]] = float(value)

    def _interpolated_arm_target(self, sim_time: float) -> np.ndarray:
        alpha = (float(sim_time) - self._transition_start_time) / self.pose_transition_s
        blend = minimum_jerk(alpha)
        return self._transition_start + blend * (self._transition_goal - self._transition_start)

    def _apply_pending_pose(self, sim_time: float) -> None:
        if self._pending_pose_name is None:
            return
        current = self._interpolated_arm_target(sim_time).astype(np.float32, copy=False)
        self.pose_name = self._pending_pose_name
        self._pending_pose_name = None
        self._transition_start = current.copy()
        self._transition_goal = self.pose_catalog[self.pose_name].copy()
        self._transition_start_time = float(sim_time)
        self.pose_switch_count += 1
        print(
            f"[ArmHack Walk arms] -> {self.pose_name} "
            f"(minimum-jerk {self.pose_transition_s:.2f}s)",
            flush=True,
        )

    def compose_action(self, network_action: np.ndarray, sim_time: float = 0.0) -> np.ndarray:
        action = np.asarray(network_action, dtype=np.float32)
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Walk actor must return 29 finite actions, got shape={action.shape}")
        segment = self.current_schedule_segment(sim_time)
        if segment is not None and int(segment["index"]) != self._last_schedule_pose_index:
            self._last_schedule_pose_index = int(segment["index"])
            scheduled_pose = str(segment.get("pose_name", ""))
            if scheduled_pose and scheduled_pose != self.pose_name:
                self._pending_pose_name = scheduled_pose
        self._apply_pending_pose(sim_time)
        self.arm_target = self._interpolated_arm_target(sim_time).astype(np.float32, copy=False)
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
        key = chr(keycode).upper() if 0 <= int(keycode) < 128 else ""
        if self.keyboard_interactive:
            if int(keycode) == 32 or key == "P":
                current_index = self.pose_names.index(
                    self._pending_pose_name or self.pose_name
                )
                self._pending_pose_name = self.pose_names[
                    (current_index + 1) % len(self.pose_names)
                ]
            elif key in {"Z", "X", "C"}:
                self._pending_pose_name = {
                    "Z": "pos1_back",
                    "X": "pos2_down",
                    "C": "pos3_front",
                }[key]
            return
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
                "keyboard"
                if self.keyboard_interactive
                else (
                    f"schedule:{self.scenario_name}"
                    if self.schedule is not None
                    else ("fixed" if self.command_active else "zero")
                )
            ),
            "scenario_name": self.scenario_name,
            "schedule_path": str(self.schedule_path) if self.schedule_path is not None else "",
            "schedule_duration_s": (
                float(self.schedule["duration_s"]) if self.schedule is not None else 0.0
            ),
            "pose_path": str(self.pose_path),
            "pose_switch_count": int(self.pose_switch_count),
            "pose_transition_s": float(self.pose_transition_s),
            "keyboard_interactive": bool(self.keyboard_interactive),
            "contract_path": str(self.contract_path),
            "source_nav2_csv_sha256": self.contract["source_nav2_csv_sha256"],
        }
