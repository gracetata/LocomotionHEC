"""Deterministic ArmHack Stand replay and reporting for the G1 MuJoCo runner.

The exported actor contains only the 96 -> 29 neural policy.  IsaacLab applies
the ArmHack arm targets outside the actor, then stores the composed 29-D raw
action as ``last_action``.  This module reproduces that deployment contract in
MuJoCo without changing the generic locomotion path when the feature is off.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
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

ANKLE_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

PAYLOAD_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def _quat_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _component_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": float(np.max(np.abs(values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def _norm_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
    }


def _minimum_jerk(alpha: float) -> float:
    value = float(np.clip(alpha, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


class InteractiveArmSequencer:
    """Interruptible SPACE pose cycle used after the shared startup CSV."""

    def __init__(self, pose_ids: list[str], poses: np.ndarray, transition_s: float, start_time: float) -> None:
        if len(pose_ids) < 2 or poses.shape != (len(pose_ids), len(ARM_JOINT_NAMES)):
            raise ValueError("Interactive SPACE cycle requires at least two canonical 14-DoF poses.")
        self.pose_ids = list(pose_ids)
        self.poses = np.asarray(poses, dtype=np.float64)
        self.transition_s = float(transition_s)
        if self.transition_s < 2.0:
            raise ValueError("Interactive SPACE transition must be at least 2.0 s.")
        self.active_index = 0
        self.start_time = float(start_time)
        self.start_positions = self.poses[0].copy()
        self.target_positions = self.poses[0].copy()

    def sample(self, now: float) -> np.ndarray:
        blend = _minimum_jerk((float(now) - self.start_time) / self.transition_s)
        return self.start_positions + (self.target_positions - self.start_positions) * blend

    def switch_next(self, now: float) -> str:
        current = self.sample(now)
        self.active_index = (self.active_index + 1) % len(self.poses)
        self.start_positions = current.copy()
        self.target_positions = self.poses[self.active_index].copy()
        self.start_time = float(now)
        return self.pose_ids[self.active_index]


class ArmHackStandReplay:
    """Apply deterministic arm targets and write MuJoCo Stand evaluation artifacts."""

    def __init__(self, config: dict[str, Any], policy_joint_names: list[str], default_angles: np.ndarray):
        self.config = config
        self.policy_joint_names = list(policy_joint_names)
        self.default_angles = np.asarray(default_angles, dtype=np.float64)
        self.action_scale = float(config["action_scale"])
        if self.action_scale <= 0.0:
            raise ValueError("ArmHack Stand requires action_scale > 0.")

        self.csv_path = Path(str(config["armhack_stand_csv_path"])).expanduser().resolve()
        self.manifest_path = Path(str(config["armhack_stand_manifest_path"])).expanduser().resolve()
        self.checkpoint_path = Path(str(config["armhack_stand_checkpoint_path"])).expanduser().resolve()
        self.checkpoint_sha256 = str(config.get("armhack_stand_checkpoint_sha256", "")).strip().lower()
        self.report_path = Path(str(config["armhack_stand_report_path"])).expanduser().resolve()
        self.plot_path = self.report_path.with_name(f"{self.report_path.stem}__torso_world_6d.png")
        self.trace_path = self.report_path.with_name(f"{self.report_path.stem}__trace.csv")
        self.ankle_trace_path = self.report_path.with_name(
            f"{self.report_path.stem}__ankle_diagnostics.csv"
        )
        self.ankle_plot_path = self.report_path.with_name(
            f"{self.report_path.stem}__ankle_comparison.png"
        )
        self.ankle_plot_svg_path = self.report_path.with_name(
            f"{self.report_path.stem}__ankle_comparison.svg"
        )
        self.ankle_high_frequency_svg_path = self.report_path.with_name(
            f"{self.report_path.stem}__ankle_high_frequency.svg"
        )
        self.test_id = str(config.get("armhack_stand_test_id", "all"))
        self.payload_kg = float(config.get("armhack_stand_payload_kg", 0.0))
        self.interactive = bool(config.get("armhack_stand_interactive_enable", False))
        self.interactive_direct_enter = bool(
            config.get("armhack_stand_interactive_direct_enter", False)
        )
        self.ankle_diagnostics_enabled = bool(
            config.get("armhack_stand_ankle_diagnostics_enable", True)
        )
        self.ankle_print_hz = float(config.get("armhack_stand_ankle_print_hz", 1.0))
        self.policy_settle_s = float(config.get("armhack_stand_policy_settle_s", 0.75))
        self.initial_stance_m = float(config.get("armhack_stand_initial_stance_m", -1.0))
        if self.initial_stance_m > 0.0 and not 0.05 <= self.initial_stance_m <= 0.60:
            raise ValueError("ArmHack Stand initial stance override must be within [0.05, 0.60] m.")
        if self.policy_settle_s < 0.0:
            raise ValueError("ArmHack Stand policy settle duration must be non-negative.")
        if self.ankle_print_hz < 0.0:
            raise ValueError("ArmHack Stand ankle print frequency must be >= 0 Hz.")
        self.next_ankle_print_time_s = 0.0
        if not 0.0 <= self.payload_kg <= 3.0:
            raise ValueError("ArmHack Stand payload must be within [0, 3] kg per wrist.")

        for path, label in (
            (self.csv_path, "test CSV"),
            (self.manifest_path, "manifest"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"ArmHack Stand {label} does not exist: {path}")
        if self.checkpoint_path.is_file():
            actual_checkpoint_sha = _sha256(self.checkpoint_path)
            if self.checkpoint_sha256 and actual_checkpoint_sha != self.checkpoint_sha256:
                raise ValueError("ArmHack Stand checkpoint SHA-256 does not match configured identity.")
            self.checkpoint_sha256 = actual_checkpoint_sha
        elif len(self.checkpoint_sha256) != 64:
            raise FileNotFoundError(
                "ArmHack Stand checkpoint is not packaged and no verified checkpoint SHA-256 was configured: "
                f"{self.checkpoint_path}"
            )

        missing_policy_joints = sorted(set(ARM_JOINT_NAMES).difference(self.policy_joint_names))
        if missing_policy_joints:
            raise ValueError(f"ArmHack arm joints are absent from policy_joint_names: {missing_policy_joints}")
        self.arm_policy_indices = np.asarray(
            [self.policy_joint_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64
        )
        missing_ankle_joints = sorted(set(ANKLE_JOINT_NAMES).difference(self.policy_joint_names))
        if missing_ankle_joints:
            raise ValueError(f"ArmHack ankle joints are absent from policy_joint_names: {missing_ankle_joints}")
        self.ankle_policy_indices = np.asarray(
            [self.policy_joint_names.index(name) for name in ANKLE_JOINT_NAMES], dtype=np.int64
        )
        self.balance_joint_names = [name for name in self.policy_joint_names if name not in set(ARM_JOINT_NAMES)]

        self.csv_times, self.csv_targets = self._load_csv()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != 5:
            raise ValueError("ArmHack Stand MuJoCo evaluation requires manifest schema_version=5.")
        if self.manifest.get("data_scope") != "arm_only_14_dof" or self.manifest.get("contains_full_body_state") is not False:
            raise ValueError("ArmHack Stand manifest must describe arm-only 14-DoF data.")
        self.timeline = self._load_timeline(float(self.csv_times[-1]))
        self.startup_timeline = [dict(stage) for stage in self.timeline]

        self.policy_active = not self.interactive
        self.policy_activation_time_s: float | None = None
        self.startup_complete = not self.interactive
        self.startup_completion_announced = False
        self.pending_enter = False
        self.pending_space = False
        self.stop_requested = False
        self.space_switch_count = 0
        self.auto_enter_s = float(config.get("armhack_stand_interactive_auto_enter_s", -1.0))
        self.auto_space_interval_s = float(
            config.get("armhack_stand_interactive_auto_space_interval_s", -1.0)
        )
        self.auto_space_max_switches = int(
            config.get("armhack_stand_interactive_auto_space_max_switches", 0)
        )
        self.next_auto_space_time_s: float | None = None
        self.interactive_sequencer: InteractiveArmSequencer | None = None
        self.interactive_pose_labels: dict[str, str] = {}
        self.interactive_events: list[dict[str, float | str]] = []
        if self.interactive:
            self._load_interactive_contract()
            self.timeline = [
                {
                    "kind": "damping_standby",
                    "label": "arms_natural_down_damping_standby",
                    "start_s": 0.0,
                    "end_s": float("inf"),
                }
            ]

        self.last_target = self.csv_targets[0].copy()
        self.last_target_time = 0.0
        self.torso_reference: np.ndarray | None = None
        self.sample_times: list[float] = []
        self.joint_samples: list[np.ndarray] = []
        self.arm_target_samples: list[np.ndarray] = []
        self.torso_delta_samples: list[np.ndarray] = []
        self.ankle_model_output_samples: list[np.ndarray] = []
        self.ankle_torque_command_samples: list[np.ndarray] = []
        self.ankle_actuator_torque_samples: list[np.ndarray] = []
        self.ankle_angle_samples: list[np.ndarray] = []
        self.ankle_target_angle_samples: list[np.ndarray] = []
        self.ankle_policy_active_samples: list[bool] = []
        self.payload_report: dict[str, dict[str, float]] = {}
        self._mujoco = None
        self._model = None
        self._pelvis_body_id: int | None = None
        self._torso_body_id: int | None = None
        self._foot_body_ids: tuple[int, int] | None = None
        self._foot_body_sets: tuple[set[int], set[int]] | None = None
        self._step_targets_xy: np.ndarray | None = None
        self._step_initial_foot_z: np.ndarray | None = None
        self._step_phase = 0
        self._step_lifted = np.zeros(2, dtype=bool)
        self._step_phase_start_time_s = 0.0
        self._policy_takeover_time_s = 0.0
        self._reset_root_xy: np.ndarray | None = None
        self._reset_root_yaw = 0.0
        self._phase_action_index = 27
        self._lifted_action_index = 28
        self._step_min_clearance_m = 0.035
        self._step_landing_tolerance_m = float(
            config.get("armhack_stand_step_landing_tolerance_m", 0.04)
        )
        if not 0.005 <= self._step_landing_tolerance_m <= 0.10:
            raise ValueError("ArmHack Stand landing tolerance must be within [0.005, 0.10] m.")
        self._step_initial_target_tolerance_m = 0.015
        self._step_min_duration_s = 0.40
        self._step_contract_ready = False
        self.step_action_alpha = float(config.get("armhack_stand_step_action_alpha", 1.0))
        if not 0.0 < self.step_action_alpha <= 1.0:
            raise ValueError("ArmHack Stand step action alpha must be in (0, 1].")

    def _load_interactive_contract(self) -> None:
        if self.test_id != "interactive":
            raise ValueError("Interactive ArmHack Stand requires test_id=interactive.")
        preset_path = Path(str(self.config.get("armhack_stand_preset_path", ""))).expanduser().resolve()
        if not preset_path.is_file():
            raise FileNotFoundError(f"Interactive Stand preset does not exist: {preset_path}")
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2 or payload.get("data_scope") != "arm_only_14_dof":
            raise ValueError("Interactive Stand requires arm preset schema v2 / arm_only_14_dof.")
        if payload.get("arm_joint_names") != ARM_JOINT_NAMES:
            raise ValueError("Interactive Stand preset joint order does not match the canonical 14-DoF order.")
        startup = payload.get("startup")
        if not isinstance(startup, dict) or startup.get("policy_inference_active") is not True:
            raise ValueError("Interactive Stand startup must mark policy_inference_active=true.")
        startup_path = (preset_path.parent / str(startup.get("csv", ""))).resolve()
        if startup_path != self.csv_path:
            raise ValueError(
                f"Interactive MuJoCo CSV differs from real deployment startup CSV: {self.csv_path} != {startup_path}"
            )
        if not math.isclose(float(startup.get("duration_s", -1.0)), self.csv_duration_s, abs_tol=1.0e-8):
            raise ValueError("Interactive startup duration differs between preset and CSV.")

        pose_entries = payload.get("poses")
        if not isinstance(pose_entries, list):
            raise ValueError("Interactive Stand poses must be a list.")
        poses_by_id: dict[str, np.ndarray] = {}
        for entry in pose_entries:
            pose_id = str(entry.get("id", ""))
            values = np.asarray(entry.get("positions_rad"), dtype=np.float64)
            if not pose_id or values.shape != (14,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid interactive Stand pose: {pose_id}")
            poses_by_id[pose_id] = values
            self.interactive_pose_labels[pose_id] = str(entry.get("label_zh", pose_id))
        damping_pose_id = str(payload.get("damping_pose_id"))
        ready_pose_id = str(payload.get("ready_pose_id"))
        cycle_ids = [str(value) for value in payload.get("space_cycle_pose_ids", [])]
        if damping_pose_id not in poses_by_id or ready_pose_id not in poses_by_id:
            raise ValueError("Interactive damping/ready pose id is missing from poses.")
        if len(cycle_ids) < 2 or cycle_ids[0] != ready_pose_id or any(value not in poses_by_id for value in cycle_ids):
            raise ValueError("Interactive SPACE cycle must start at ready_pose_id and contain valid poses.")
        if not np.allclose(self.csv_targets[0], poses_by_id[damping_pose_id], rtol=0.0, atol=1.0e-6):
            raise ValueError("Interactive startup first row is not the natural-down damping pose.")
        if not np.allclose(self.csv_targets[-1], poses_by_id[ready_pose_id], rtol=0.0, atol=1.0e-6):
            raise ValueError("Interactive startup last row is not the flat default ready pose.")
        transition_s = float(self.config.get("armhack_stand_interactive_transition_s", 7.5))
        self.interactive_sequencer = InteractiveArmSequencer(
            cycle_ids,
            np.stack([poses_by_id[pose_id] for pose_id in cycle_ids]),
            transition_s,
            self.csv_duration_s,
        )

    @property
    def csv_duration_s(self) -> float:
        return float(self.csv_times[-1])

    @property
    def policy_inference_enabled(self) -> bool:
        return self.policy_active

    def key_callback(self, keycode: int) -> None:
        """GLFW callback: ENTER activates policy/debug, SPACE changes arms, Q stops."""
        if not self.interactive:
            return
        if keycode in {10, 13, 257, 335}:
            self.pending_enter = True
        elif keycode == 32:
            self.pending_space = True
        elif keycode in {81, 113}:
            self.stop_requested = True

    def _trim_timeline_at(self, time_s: float) -> None:
        trimmed: list[dict[str, float | str]] = []
        for stage in self.timeline:
            start_s = float(stage["start_s"])
            end_s = float(stage["end_s"])
            if start_s >= time_s:
                continue
            copied = dict(stage)
            if end_s > time_s:
                copied["end_s"] = float(time_s)
            if float(copied["end_s"]) > start_s:
                trimmed.append(copied)
        self.timeline = trimmed

    def _activate_interactive_policy(self, sim_time: float) -> None:
        if self.policy_active:
            print("[ArmHack Stand MuJoCo] ENTER ignored: policy/debug mode is already active.", flush=True)
            return
        self.policy_active = True
        self.policy_activation_time_s = float(sim_time)
        if self.interactive_direct_enter:
            self.startup_complete = True
            self.startup_completion_announced = True
            self._trim_timeline_at(sim_time)
            assert self.interactive_sequencer is not None
            self.interactive_sequencer.active_index = 0
            self.interactive_sequencer.start_positions = self.last_target.copy()
            self.interactive_sequencer.target_positions = self.interactive_sequencer.poses[0].copy()
            self.interactive_sequencer.start_time = float(sim_time)
            # sample_target() selects the sequencer after csv_duration_s.  Shift
            # the activation clock so direct policy entry remains smooth while
            # skipping the legacy 25.5 s scripted arm startup.
            self.policy_activation_time_s = float(sim_time) - self.csv_duration_s - 1.0e-6
            transition_end = float(sim_time) + self.interactive_sequencer.transition_s
            if self.auto_space_interval_s > 0.0:
                self.next_auto_space_time_s = transition_end + self.auto_space_interval_s
            self.timeline.extend(
                [
                    {
                        "kind": "interactive_direct_enter_transition",
                        "label": self.interactive_sequencer.pose_ids[0],
                        "start_s": float(sim_time),
                        "end_s": transition_end,
                    },
                    {
                        "kind": "interactive_pose_hold",
                        "label": self.interactive_sequencer.pose_ids[0],
                        "start_s": transition_end,
                        "end_s": float("inf"),
                    },
                ]
            )
            print(
                "[ArmHack Stand MuJoCo] ENTER -> POLICY ON; direct minimum-jerk arm ready transition.",
                flush=True,
            )
            return
        self.startup_complete = False
        self._trim_timeline_at(sim_time)
        for stage in self.startup_timeline:
            shifted = dict(stage)
            shifted["start_s"] = float(sim_time) + float(stage["start_s"])
            shifted["end_s"] = float(sim_time) + float(stage["end_s"])
            self.timeline.append(shifted)
        assert self.interactive_sequencer is not None
        self.interactive_sequencer.start_time = float(sim_time) + self.csv_duration_s
        if self.auto_space_interval_s > 0.0:
            self.next_auto_space_time_s = (
                float(sim_time) + self.csv_duration_s + self.auto_space_interval_s
            )
        print(
            "[ArmHack Stand MuJoCo] ENTER -> DEBUG / POLICY ON; "
            "自然下垂→平直默认→向前伸直→收回平直默认。",
            flush=True,
        )

    def process_interaction_requests(self, sim_time: float) -> bool:
        """Apply queued/automatic keys; return False when Q requested stop."""
        if not self.interactive:
            return True
        if not self.policy_active and self.auto_enter_s >= 0.0 and sim_time >= self.auto_enter_s:
            self.pending_enter = True
        if self.pending_enter:
            self.pending_enter = False
            self._activate_interactive_policy(sim_time)

        if self.policy_active and self.policy_activation_time_s is not None:
            elapsed = float(sim_time) - self.policy_activation_time_s
            if not self.startup_complete and elapsed >= self.csv_duration_s:
                self.startup_complete = True
                if not self.startup_completion_announced:
                    self.startup_completion_announced = True
                    ready_start = self.policy_activation_time_s + self.csv_duration_s
                    self.timeline.append(
                        {
                            "kind": "interactive_pose_hold",
                            "label": "P0_symmetric_reference",
                            "start_s": ready_start,
                            "end_s": float("inf"),
                        }
                    )
                    print(
                        "[ArmHack Stand MuJoCo] INIT COMPLETE; SPACE 已解锁，当前为平直默认 P0。",
                        flush=True,
                    )
            if (
                self.startup_complete
                and self.next_auto_space_time_s is not None
                and sim_time >= self.next_auto_space_time_s
                and self.space_switch_count < self.auto_space_max_switches
            ):
                self.pending_space = True
                self.next_auto_space_time_s += self.auto_space_interval_s

        if self.pending_space:
            self.pending_space = False
            if not self.policy_active or not self.startup_complete:
                print("[ArmHack Stand MuJoCo] SPACE LOCKED until automatic initialization completes.", flush=True)
            else:
                assert self.interactive_sequencer is not None
                self._trim_timeline_at(sim_time)
                pose_id = self.interactive_sequencer.switch_next(sim_time)
                transition_end = float(sim_time) + self.interactive_sequencer.transition_s
                self.timeline.extend(
                    [
                        {
                            "kind": "interactive_space_transition",
                            "label": f"SPACE_to_{pose_id}",
                            "start_s": float(sim_time),
                            "end_s": transition_end,
                        },
                        {
                            "kind": "interactive_pose_hold",
                            "label": pose_id,
                            "start_s": transition_end,
                            "end_s": float("inf"),
                        },
                    ]
                )
                self.space_switch_count += 1
                print(
                    f"[ArmHack Stand MuJoCo] SPACE -> {pose_id} "
                    f"({self.interactive_pose_labels.get(pose_id, pose_id)}), "
                    f"minimum-jerk {self.interactive_sequencer.transition_s:.2f}s",
                    flush=True,
                )
        return not self.stop_requested

    def _load_csv(self) -> tuple[np.ndarray, np.ndarray]:
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "time_s" not in reader.fieldnames:
                raise ValueError(f"ArmHack Stand CSV has no time_s column: {self.csv_path}")
            missing = sorted(set(ARM_JOINT_NAMES).difference(reader.fieldnames))
            if missing:
                raise ValueError(f"ArmHack Stand CSV is missing arm joints: {missing}")
            unexpected_joint_columns = [
                name for name in reader.fieldnames if name != "time_s" and name not in ARM_JOINT_NAMES
            ]
            if unexpected_joint_columns:
                raise ValueError(
                    "ArmHack Stand MuJoCo CSV must contain only time_s and 14 arm joints; "
                    f"unexpected columns: {unexpected_joint_columns}"
                )
            times: list[float] = []
            targets: list[list[float]] = []
            for row in reader:
                times.append(float(row["time_s"]))
                targets.append([float(row[name]) for name in ARM_JOINT_NAMES])
        if not targets:
            raise ValueError(f"ArmHack Stand CSV contains no rows: {self.csv_path}")
        times_array = np.asarray(times, dtype=np.float64)
        targets_array = np.asarray(targets, dtype=np.float64)
        if not np.all(np.isfinite(times_array)) or not np.all(np.isfinite(targets_array)):
            raise ValueError("ArmHack Stand CSV contains non-finite values.")
        if np.any(np.diff(times_array) < 0.0):
            raise ValueError("ArmHack Stand CSV time_s must be monotonically non-decreasing.")
        times_array = times_array - times_array[0]
        return times_array, targets_array

    def _load_timeline(self, duration_s: float) -> list[dict[str, float | str]]:
        files = self.manifest.get("files", {})
        timeline: list[dict[str, Any]] = []
        collection_modes = {
            "all",
            "representative_poses",
            "synthesized_poses",
            "randomized_poses",
            "representative_trajectories",
            "synthesized_trajectories",
            "randomized_trajectories",
            "down_to_horizontal",
            "default_forward_return_down",
            "interactive",
        }
        if self.test_id.startswith("generated_random_trajectory_"):
            generated_metadata_path = self.csv_path.with_suffix(".json")
            if not generated_metadata_path.is_file():
                raise FileNotFoundError(
                    f"Generated ArmHack Stand trajectory metadata is missing: {generated_metadata_path}"
                )
            generated_metadata = json.loads(generated_metadata_path.read_text(encoding="utf-8"))
            if (
                generated_metadata.get("schema_version") != 1
                or generated_metadata.get("data_scope") != "arm_only_14_dof"
                or generated_metadata.get("contains_full_body_state") is not False
            ):
                raise ValueError(
                    f"Unsupported generated ArmHack Stand trajectory metadata: {generated_metadata_path}"
                )
            expected_csv_sha = generated_metadata.get("output", {}).get("csv_sha256")
            actual_csv_sha = _sha256(self.csv_path)
            if expected_csv_sha != actual_csv_sha:
                raise ValueError(
                    "Generated ArmHack Stand trajectory CSV SHA-256 does not match its metadata: "
                    f"expected={expected_csv_sha}, actual={actual_csv_sha}"
                )
            timeline = list(generated_metadata.get("timeline") or [])
        elif self.test_id in collection_modes:
            metadata_key = "interactive_startup" if self.test_id == "interactive" else self.test_id
            metadata = files.get(metadata_key, {})
            timeline = list(metadata.get("detailed_timeline") or metadata.get("timeline") or [])
        elif "_item" in self.test_id:
            mode, item_text = self.test_id.rsplit("_item", 1)
            item_index = int(item_text) - 1
            mapping = {
                "representative_pose": ("representative_poses", "pose_id"),
                "synthesized_pose": ("synthesized_poses", "pose_id"),
                "randomized_pose": ("randomized_poses", "pose_id"),
                "representative_trajectory": ("representative_trajectories", "trajectory_id"),
                "synthesized_trajectory": ("synthesized_trajectories", "trajectory_id"),
                "randomized_trajectory": ("randomized_trajectories", "trajectory_id"),
            }
            if mode not in mapping:
                raise ValueError(f"Unsupported ArmHack Stand test id: {self.test_id}")
            collection, label_key = mapping[mode]
            items = self.manifest.get(collection, [])
            if item_index < 0 or item_index >= len(items):
                raise ValueError(f"ArmHack Stand test item is out of range: {self.test_id}")
            item = items[item_index]
            timeline = [
                {
                    "kind": "static_hold" if "pose" in mode else "trajectory",
                    "label": str(item[label_key]),
                    "start_s": 0.0,
                    "end_s": duration_s,
                }
            ]
        cleaned: list[dict[str, float | str]] = []
        for stage in timeline:
            start_s = max(float(stage.get("start_s", 0.0)), 0.0)
            end_s = min(float(stage.get("end_s", duration_s)), duration_s)
            if end_s > start_s:
                cleaned.append(
                    {
                        "kind": str(stage.get("kind", "stage")),
                        "label": str(stage.get("label", self.test_id)),
                        "start_s": start_s,
                        "end_s": end_s,
                    }
                )
        if not cleaned:
            cleaned = [{"kind": "stage", "label": self.test_id, "start_s": 0.0, "end_s": duration_s}]
        return cleaned

    def initialize_model_and_state(
        self,
        mujoco_module,
        model,
        data,
        qpos_addresses: dict[str, int],
        torso_body_id: int,
    ) -> None:
        for name, target in zip(ARM_JOINT_NAMES, self.csv_targets[0], strict=True):
            data.qpos[qpos_addresses[name]] = float(target)
        if self.initial_stance_m > 0.0:
            roll = (self.initial_stance_m - 0.237) / 1.22
            stance_targets = {
                "left_hip_roll_joint": roll,
                "right_hip_roll_joint": -roll,
                "left_ankle_roll_joint": -roll,
                "right_ankle_roll_joint": roll,
            }
            for joint_name, target in stance_targets.items():
                data.qpos[qpos_addresses[joint_name]] = float(target)

        for body_name in PAYLOAD_BODY_NAMES:
            body_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"MuJoCo model has no ArmHack payload body: {body_name}")
            original_mass = float(model.body_mass[body_id])
            if original_mass <= 0.0:
                raise ValueError(f"MuJoCo payload body has non-positive mass: {body_name}={original_mass}")
            final_mass = original_mass + self.payload_kg
            inertia_scale = final_mass / original_mass
            model.body_mass[body_id] = final_mass
            model.body_inertia[body_id] *= inertia_scale
            self.payload_report[body_name] = {
                "original_mass_kg": original_mass,
                "added_mass_kg": self.payload_kg,
                "final_mass_kg": final_mass,
                "inertia_scale": inertia_scale,
            }

        mujoco_module.mj_forward(model, data)
        self.torso_reference = self._torso_pose(data, torso_body_id)
        self._mujoco = mujoco_module
        self._model = model
        self._torso_body_id = torso_body_id
        pelvis_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, "pelvis")
        self._pelvis_body_id = torso_body_id if pelvis_id < 0 else pelvis_id
        foot_ids = []
        for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            body_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"MuJoCo model has no ordered-step body: {body_name}")
            foot_ids.append(body_id)
        self._foot_body_ids = (foot_ids[0], foot_ids[1])
        parent_ids = np.asarray(model.body_parentid, dtype=np.int64)

        def descendants(root_id: int) -> set[int]:
            result: set[int] = set()
            for body_id in range(int(model.nbody)):
                current = body_id
                while current > 0:
                    if current == root_id:
                        result.add(body_id)
                        break
                    current = int(parent_ids[current])
            return result

        self._foot_body_sets = (descendants(foot_ids[0]), descendants(foot_ids[1]))
        self._initialize_ordered_step_state(data, 0.0)

    def _initialize_ordered_step_state(
        self, data, time_s: float, *, force_phase_zero: bool = False
    ) -> None:
        if self._pelvis_body_id is None or self._foot_body_ids is None:
            raise RuntimeError("ordered-step bodies are not initialized")
        pelvis_pos = np.asarray(data.xpos[self._pelvis_body_id], dtype=np.float64)
        pelvis_yaw = float(
            _quat_wxyz_to_rpy(np.asarray(data.xquat[self._pelvis_body_id], dtype=np.float64))[2]
        )
        self._reset_root_xy = pelvis_pos[:2].copy()
        self._reset_root_yaw = pelvis_yaw
        lateral_axis = np.asarray([-math.sin(pelvis_yaw), math.cos(pelvis_yaw)], dtype=np.float64)
        self._step_targets_xy = np.stack(
            [pelvis_pos[:2] + 0.15 * lateral_axis, pelvis_pos[:2] - 0.15 * lateral_axis]
        )
        foot_pos = np.asarray(data.xpos[list(self._foot_body_ids)], dtype=np.float64)
        self._step_initial_foot_z = foot_pos[:, 2].copy()
        contact = self._foot_contact(data)
        error = np.linalg.norm(foot_pos[:, :2] - self._step_targets_xy, axis=1)
        if force_phase_zero:
            self._step_phase = 0
            self._step_lifted[:] = False
        elif bool(np.all(contact)) and bool(np.all(error <= self._step_initial_target_tolerance_m)):
            self._step_phase = 2
            self._step_lifted[:] = True
        elif bool(contact[0]) and error[0] <= self._step_landing_tolerance_m:
            self._step_phase = 1
            self._step_lifted[:] = (True, False)
        else:
            self._step_phase = 0
            self._step_lifted[:] = False
        self._step_phase_start_time_s = float(time_s)
        self._policy_takeover_time_s = float(time_s)
        self._step_contract_ready = True
        if self._torso_body_id is not None:
            self.torso_reference = self._torso_pose(data, self._torso_body_id)
        print(
            "[ArmHack Stand MuJoCo] ordered-step observation initialized: "
            f"phase={self._step_phase} target_error_m={error.tolist()} contact={contact.tolist()}",
            flush=True,
        )

    def reset_switch_reference(self, data, time_s: float) -> None:
        """Capture the current torso SE(2) and rebuild the ordered 30 cm targets."""
        self._initialize_ordered_step_state(data, float(time_s), force_phase_zero=True)

    def should_hold_default(self, time_s: float) -> bool:
        """The same Stand actor supplies phase-two balance during settling."""
        del time_s
        return False

    def _foot_contact(self, data) -> np.ndarray:
        if self._model is None or self._foot_body_sets is None:
            return np.zeros(2, dtype=bool)
        result = np.zeros(2, dtype=bool)
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            body1 = int(self._model.geom_bodyid[int(contact.geom1)])
            body2 = int(self._model.geom_bodyid[int(contact.geom2)])
            for foot_index, body_set in enumerate(self._foot_body_sets):
                if body1 in body_set or body2 in body_set:
                    result[foot_index] = True
        return result

    def augment_last_action(self, last_action: np.ndarray, data, time_s: float) -> np.ndarray:
        """Mirror IsaacLab's phase-augmented last-action observation."""
        augmented = np.asarray(last_action, dtype=np.float32).copy()
        if float(time_s) - self._policy_takeover_time_s < self.policy_settle_s:
            # Phase two is the learned planted hold. Use the same actor (not a
            # separate damping policy) until MuJoCo contacts become valid.
            augmented[self._phase_action_index] = -1.0
            augmented[self._lifted_action_index] = 1.0
            return augmented
        if not self._step_contract_ready:
            self._initialize_ordered_step_state(data, float(time_s))
        if self._foot_body_ids is None or self._step_targets_xy is None or self._step_initial_foot_z is None:
            return augmented
        foot_pos = np.asarray(data.xpos[list(self._foot_body_ids)], dtype=np.float64)
        contact = self._foot_contact(data)
        error = np.linalg.norm(foot_pos[:, :2] - self._step_targets_xy, axis=1)
        clearance = foot_pos[:, 2] - self._step_initial_foot_z
        phase_before = self._step_phase
        if phase_before < 2:
            active = phase_before
            if clearance[active] >= self._step_min_clearance_m:
                self._step_lifted[active] = True
            elapsed = float(time_s) - self._step_phase_start_time_s
            if (
                self._step_lifted[active]
                and contact[active]
                and error[active] <= self._step_landing_tolerance_m
                and elapsed >= self._step_min_duration_s
            ):
                completed_foot = "left" if active == 0 else "right"
                self._step_phase += 1
                self._step_phase_start_time_s = float(time_s)
                print(
                    "[ArmHack Stand MuJoCo] ordered-step touchdown: "
                    f"foot={completed_foot} phase={self._step_phase} t={float(time_s):.3f}s "
                    f"target_error_m={float(error[active]):.4f}",
                    flush=True,
                )
                if self._step_phase < 2:
                    self._step_lifted[self._step_phase] = False
            elif self._step_lifted[active] and contact[active] and elapsed >= self._step_min_duration_s:
                # Wrong touchdown: require another real lift before completion.
                self._step_lifted[active] = False
        phase_signal = 0.0 if self._step_phase == 0 else (1.0 if self._step_phase == 1 else -1.0)
        active = min(self._step_phase, 1)
        lifted_signal = float(self._step_lifted[active]) if self._step_phase < 2 else 1.0
        augmented[self._phase_action_index] = phase_signal
        augmented[self._lifted_action_index] = lifted_signal
        return augmented

    def relative_pose_command(self, data) -> np.ndarray:
        """Return the reset-relative SE(2) observation used during Stand training."""
        if self._pelvis_body_id is None or self._reset_root_xy is None:
            return np.zeros(3, dtype=np.float32)
        position = np.asarray(data.xpos[self._pelvis_body_id], dtype=np.float64)
        yaw = float(_quat_wxyz_to_rpy(np.asarray(data.xquat[self._pelvis_body_id], dtype=np.float64))[2])
        delta_w = self._reset_root_xy - position[:2]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        delta_b = np.asarray(
            [cos_yaw * delta_w[0] + sin_yaw * delta_w[1], -sin_yaw * delta_w[0] + cos_yaw * delta_w[1]],
            dtype=np.float64,
        )
        yaw_error = float(_wrap_to_pi(np.asarray([self._reset_root_yaw - yaw]))[0])
        return np.asarray(
            [
                np.clip(2.0 * delta_b[0], -0.50, 0.50),
                np.clip(2.0 * delta_b[1], -0.50, 0.50),
                np.clip(1.5 * yaw_error, -0.60, 0.60),
            ],
            dtype=np.float32,
        )

    def filter_policy_action(self, previous_action: np.ndarray, next_action: np.ndarray) -> np.ndarray:
        """Jerk-limit the learned step while leaving the planted phase unfiltered."""
        current = np.asarray(previous_action, dtype=np.float32)
        proposed = np.asarray(next_action, dtype=np.float32)
        if self._step_phase >= 2:
            return proposed.copy()
        return current + self.step_action_alpha * (proposed - current)

    @staticmethod
    def _torso_pose(data, torso_body_id: int) -> np.ndarray:
        position = np.asarray(data.xpos[torso_body_id], dtype=np.float64).copy()
        rpy = _quat_wxyz_to_rpy(np.asarray(data.xquat[torso_body_id], dtype=np.float64))
        return np.concatenate((position, rpy))

    def sample_target(self, time_s: float) -> np.ndarray:
        if self.interactive:
            if not self.policy_active or self.policy_activation_time_s is None:
                return self.csv_targets[0].copy()
            elapsed = max(float(time_s) - self.policy_activation_time_s, 0.0)
            if elapsed <= self.csv_duration_s:
                sample_time = elapsed
            else:
                assert self.interactive_sequencer is not None
                return self.interactive_sequencer.sample(time_s)
        else:
            sample_time = float(time_s)
        sample_time = float(np.clip(sample_time, 0.0, self.csv_times[-1]))
        upper_index = int(np.searchsorted(self.csv_times, sample_time, side="left"))
        upper_index = min(upper_index, len(self.csv_times) - 1)
        lower_index = max(upper_index - 1, 0)
        lower_time = float(self.csv_times[lower_index])
        upper_time = float(self.csv_times[upper_index])
        if upper_time - lower_time <= 1.0e-10:
            return self.csv_targets[upper_index].copy()
        alpha = (sample_time - lower_time) / (upper_time - lower_time)
        return (1.0 - alpha) * self.csv_targets[lower_index] + alpha * self.csv_targets[upper_index]

    def compose_action(self, policy_action: np.ndarray, time_s: float) -> np.ndarray:
        composed = np.asarray(policy_action, dtype=np.float32).copy()
        target = self.sample_target(time_s)
        raw_arm_action = (target - self.default_angles[self.arm_policy_indices]) / self.action_scale
        composed[self.arm_policy_indices] = raw_arm_action.astype(np.float32)
        self.last_target = target
        if self.interactive:
            self.last_target_time = (
                0.0
                if self.policy_activation_time_s is None
                else min(max(float(time_s) - self.policy_activation_time_s, 0.0), self.csv_duration_s)
            )
        else:
            self.last_target_time = float(time_s)
        return composed

    def record_control_sample(
        self,
        data,
        qpos_addresses: dict[str, int],
        actuator_ids_by_joint: dict[str, int],
        action: np.ndarray,
        torso_body_id: int,
        time_s: float,
    ) -> None:
        if self.torso_reference is None:
            raise RuntimeError("ArmHack Stand torso reference was not initialized.")
        joint_position = np.asarray(
            [data.qpos[qpos_addresses[name]] for name in self.policy_joint_names], dtype=np.float64
        )
        torso_delta = self._torso_pose(data, torso_body_id) - self.torso_reference
        torso_delta[3:] = _wrap_to_pi(torso_delta[3:])
        self.sample_times.append(float(time_s))
        self.joint_samples.append(joint_position)
        self.arm_target_samples.append(self.last_target.copy())
        self.torso_delta_samples.append(torso_delta)
        if not self.ankle_diagnostics_enabled:
            return

        model_output = np.asarray(action, dtype=np.float64)[self.ankle_policy_indices].copy()
        torque_command = np.asarray(
            [data.ctrl[actuator_ids_by_joint[name]] for name in ANKLE_JOINT_NAMES],
            dtype=np.float64,
        )
        actuator_torque = np.asarray(
            [data.actuator_force[actuator_ids_by_joint[name]] for name in ANKLE_JOINT_NAMES],
            dtype=np.float64,
        )
        joint_angle = joint_position[self.ankle_policy_indices].copy()
        target_angle = self.default_angles[self.ankle_policy_indices] + self.action_scale * model_output
        policy_active = bool(self.policy_inference_enabled)
        self.ankle_model_output_samples.append(model_output)
        self.ankle_torque_command_samples.append(torque_command)
        self.ankle_actuator_torque_samples.append(actuator_torque)
        self.ankle_angle_samples.append(joint_angle)
        self.ankle_target_angle_samples.append(target_angle)
        self.ankle_policy_active_samples.append(policy_active)

        if self.ankle_print_hz <= 0.0 or time_s + 1.0e-9 < self.next_ankle_print_time_s:
            return
        self.next_ankle_print_time_s = float(time_s) + 1.0 / self.ankle_print_hz
        _, stage_label = self._stage_at(float(time_s))
        values = {
            name: {
                "output": float(model_output[index]),
                "torque": float(actuator_torque[index]),
                "angle": float(joint_angle[index]),
            }
            for index, name in enumerate(ANKLE_JOINT_NAMES)
        }
        print(
            f"[ANKLE] t={time_s:7.3f}s stage={stage_label} "
            f"policy={'ON' if policy_active else 'OFF'}",
            flush=True,
        )
        print(
            "  pitch "
            f"L(out={values['left_ankle_pitch_joint']['output']:+.4f}, "
            f"tau={values['left_ankle_pitch_joint']['torque']:+.3f}Nm, "
            f"q={values['left_ankle_pitch_joint']['angle']:+.4f}rad) | "
            f"R(out={values['right_ankle_pitch_joint']['output']:+.4f}, "
            f"tau={values['right_ankle_pitch_joint']['torque']:+.3f}Nm, "
            f"q={values['right_ankle_pitch_joint']['angle']:+.4f}rad)",
            flush=True,
        )
        print(
            "  roll  "
            f"L(out={values['left_ankle_roll_joint']['output']:+.4f}, "
            f"tau={values['left_ankle_roll_joint']['torque']:+.3f}Nm, "
            f"q={values['left_ankle_roll_joint']['angle']:+.4f}rad) | "
            f"R(out={values['right_ankle_roll_joint']['output']:+.4f}, "
            f"tau={values['right_ankle_roll_joint']['torque']:+.3f}Nm, "
            f"q={values['right_ankle_roll_joint']['angle']:+.4f}rad)",
            flush=True,
        )

    @staticmethod
    def _stage_color(stage: dict[str, float | str]) -> str:
        kind = str(stage["kind"])
        label = str(stage["label"])
        if "transition" in kind or "bridge" in kind:
            return "#B0BEC5"
        if kind == "damping_standby":
            return "#455A64"
        if label == "arms_down_hold":
            return "#2F4B7C"
        if label in {"arms_default_initial_hold", "arms_default_returned_hold"}:
            return "#4E79A7"
        if label == "arms_natural_down_hold":
            return "#2F4B7C"
        if label == "arms_forward_horizontal_hold":
            return "#00A087"
        if label in {"arms_flat_default_hold", "arms_flat_default_ready_hold", "P0_symmetric_reference"}:
            return "#4E79A7"
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
        if label.startswith("waypoint_"):
            return "#76B7B2"
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
                "arms_natural_down_to_flat_default": "AD→P0",
                "arms_flat_default_to_forward_horizontal": "P0→F",
                "arms_forward_horizontal_to_flat_default": "F→P0",
            }
            if label.startswith("SPACE_to_"):
                return "SPACE"
            return transition_labels.get(label, "T")
        if "bridge" in kind:
            return "B"
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
        if label == "arms_natural_down_damping_standby":
            return "DAMP"
        if label in {"arms_flat_default_hold", "arms_flat_default_ready_hold", "P0_symmetric_reference"}:
            return "P0"
        if label.startswith("waypoint_"):
            return f"W{label.split(':', 1)[0].rsplit('_', 1)[-1]}"
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

    def _stage_at(self, time_s: float) -> tuple[str, str]:
        for stage in self.timeline:
            if float(stage["start_s"]) <= time_s < float(stage["end_s"]) + 1.0e-9:
                return str(stage["kind"]), str(stage["label"])
        stage = self.timeline[-1]
        return str(stage["kind"]), str(stage["label"])

    def _write_trace(
        self,
        joint_samples: np.ndarray,
        target_samples: np.ndarray,
        torso_delta: np.ndarray,
    ) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["time_s", "stage_kind", "stage_label"]
            + [f"actual_{name}" for name in self.policy_joint_names]
            + [f"target_{name}" for name in ARM_JOINT_NAMES]
            + ["delta_x_w", "delta_y_w", "delta_z_w", "delta_roll_w", "delta_pitch_w", "delta_yaw_w"]
        )
        with self.trace_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            for sample_index, time_s in enumerate(self.sample_times):
                stage_kind, stage_label = self._stage_at(time_s)
                writer.writerow(
                    [f"{time_s:.8f}", stage_kind, stage_label]
                    + [f"{value:.9f}" for value in joint_samples[sample_index]]
                    + [f"{value:.9f}" for value in target_samples[sample_index]]
                    + [f"{value:.9f}" for value in torso_delta[sample_index]]
                )

    def _write_plot(self, torso_delta: np.ndarray) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        times = np.asarray(self.sample_times, dtype=np.float64)
        figure = plt.figure(figsize=(18.0, 10.0), layout="constrained")
        grid = figure.add_gridspec(3, 1, height_ratios=(3.0, 3.0, 1.15), hspace=0.16)
        position_axis = figure.add_subplot(grid[0, 0])
        rotation_axis = figure.add_subplot(grid[1, 0], sharex=position_axis)
        stage_axis = figure.add_subplot(grid[2, 0], sharex=position_axis)
        for index, (label, color) in enumerate(zip(("dx_w", "dy_w", "dz_w"), ("#1F77B4", "#D62728", "#2CA02C"), strict=True)):
            position_axis.plot(times, torso_delta[:, index], label=label, color=color, linewidth=1.15)
        for index, (label, color) in enumerate(zip(("droll_w", "dpitch_w", "dyaw_w"), ("#9467BD", "#FF7F0E", "#17BECF"), strict=True), start=3):
            rotation_axis.plot(times, torso_delta[:, index], label=label, color=color, linewidth=1.15)
        displayed_end = float(times[-1]) if len(times) else self.csv_duration_s
        for stage in self.timeline:
            start_s = float(stage["start_s"])
            end_s = min(float(stage["end_s"]), displayed_end)
            if end_s <= start_s:
                continue
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
            stage_axis.text(
                0.5 * (start_s + end_s),
                0.5,
                self._short_stage_label(stage),
                ha="center",
                va="center",
                rotation=90 if end_s - start_s < 2.5 else 0,
                fontsize=6.5,
                color="#111111",
                clip_on=True,
            )
        for axis in (position_axis, rotation_axis):
            axis.axhline(0.0, color="#666666", linewidth=0.6)
            axis.grid(True, alpha=0.22)
            axis.legend(loc="upper right", ncol=3)
        position_axis.set_ylabel("World translation displacement (m)")
        rotation_axis.set_ylabel("World RPY displacement (rad)")
        rotation_axis.set_xlabel("Test time (s)")
        stage_axis.set_ylim(0.0, 1.0)
        stage_axis.set_yticks([])
        stage_axis.set_xlabel("Test stage timeline (s)")
        stage_axis.set_xlim(0.0, max(displayed_end, 1.0e-6))
        figure.suptitle(f"ArmHack Stand MuJoCo torso world-frame 6D displacement — {self.test_id}", fontsize=16, fontweight="bold")
        legend = [
            Patch(facecolor="#4E79A7", label="RP"),
            Patch(facecolor="#B07AA1", label="SP"),
            Patch(facecolor="#76B7B2", label="GP"),
            Patch(facecolor="#59A14F", label="RT"),
            Patch(facecolor="#F28E2B", label="ST"),
            Patch(facecolor="#EDC948", label="GT"),
            Patch(facecolor="#B0BEC5", label="transition / bridge"),
            Patch(facecolor="#2F4B7C", label="AD"),
            Patch(facecolor="#00A087", label="AH"),
        ]
        figure.legend(handles=legend, loc="lower center", ncol=9, frameon=False)
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

    def _write_ankle_trace(
        self,
        model_output: np.ndarray,
        torque_command: np.ndarray,
        actuator_torque: np.ndarray,
        joint_angle: np.ndarray,
        target_angle: np.ndarray,
    ) -> None:
        self.ankle_trace_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["time_s", "stage_kind", "stage_label", "policy_active"]
        for name in ANKLE_JOINT_NAMES:
            fieldnames.extend(
                [
                    f"model_output_{name}",
                    f"torque_command_nm_{name}",
                    f"actuator_torque_nm_{name}",
                    f"joint_angle_rad_{name}",
                    f"target_angle_rad_{name}",
                ]
            )
        with self.ankle_trace_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            for sample_index, time_s in enumerate(self.sample_times):
                stage_kind, stage_label = self._stage_at(time_s)
                row: list[str | int] = [
                    f"{time_s:.8f}",
                    stage_kind,
                    stage_label,
                    int(self.ankle_policy_active_samples[sample_index]),
                ]
                for joint_index in range(len(ANKLE_JOINT_NAMES)):
                    row.extend(
                        [
                            f"{model_output[sample_index, joint_index]:.9f}",
                            f"{torque_command[sample_index, joint_index]:.9f}",
                            f"{actuator_torque[sample_index, joint_index]:.9f}",
                            f"{joint_angle[sample_index, joint_index]:.9f}",
                            f"{target_angle[sample_index, joint_index]:.9f}",
                        ]
                    )
                writer.writerow(row)

    def _write_ankle_plot(
        self,
        model_output: np.ndarray,
        actuator_torque: np.ndarray,
        joint_angle: np.ndarray,
        target_angle: np.ndarray,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        times = np.asarray(self.sample_times, dtype=np.float64)
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(18.0, 11.0),
            sharex=True,
            layout="constrained",
        )
        left_color = "#1F77B4"
        right_color = "#D62728"
        axis_groups = (
            (
                "Ankle pitch",
                ANKLE_JOINT_NAMES.index("left_ankle_pitch_joint"),
                ANKLE_JOINT_NAMES.index("right_ankle_pitch_joint"),
            ),
            (
                "Ankle roll",
                ANKLE_JOINT_NAMES.index("left_ankle_roll_joint"),
                ANKLE_JOINT_NAMES.index("right_ankle_roll_joint"),
            ),
        )
        for column, (title, left_index, right_index) in enumerate(axis_groups):
            axes[0, column].plot(
                times,
                model_output[:, left_index],
                color=left_color,
                linewidth=1.0,
                label="left model output",
            )
            axes[0, column].plot(
                times,
                model_output[:, right_index],
                color=right_color,
                linewidth=1.0,
                label="right model output",
            )
            axes[1, column].plot(
                times,
                actuator_torque[:, left_index],
                color=left_color,
                linewidth=1.0,
                label="left actuator torque",
            )
            axes[1, column].plot(
                times,
                actuator_torque[:, right_index],
                color=right_color,
                linewidth=1.0,
                label="right actuator torque",
            )
            axes[2, column].plot(
                times,
                joint_angle[:, left_index],
                color=left_color,
                linewidth=1.1,
                label="left actual angle",
            )
            axes[2, column].plot(
                times,
                target_angle[:, left_index],
                color=left_color,
                linewidth=0.9,
                linestyle="--",
                alpha=0.72,
                label="left target angle",
            )
            axes[2, column].plot(
                times,
                joint_angle[:, right_index],
                color=right_color,
                linewidth=1.1,
                label="right actual angle",
            )
            axes[2, column].plot(
                times,
                target_angle[:, right_index],
                color=right_color,
                linewidth=0.9,
                linestyle="--",
                alpha=0.72,
                label="right target angle",
            )
            axes[0, column].set_title(title, fontweight="bold")

        displayed_end = float(times[-1])
        for stage in self.timeline:
            start_s = float(stage["start_s"])
            end_s = min(float(stage["end_s"]), displayed_end)
            if end_s <= start_s:
                continue
            for axis in axes.flat:
                axis.axvspan(
                    start_s,
                    end_s,
                    color=self._stage_color(stage),
                    alpha=0.045,
                    linewidth=0.0,
                )
        for axis in axes.flat:
            axis.axhline(0.0, color="#666666", linewidth=0.55)
            axis.grid(True, alpha=0.22)
            axis.legend(loc="upper right", fontsize=8)
        axes[0, 0].set_ylabel("Model output (dimensionless)")
        axes[1, 0].set_ylabel("MuJoCo actuator torque (N·m)")
        axes[2, 0].set_ylabel("Joint angle (rad)")
        axes[2, 0].set_xlabel("Simulation time (s)")
        axes[2, 1].set_xlabel("Simulation time (s)")
        figure.suptitle(
            f"ArmHack Stand ankle diagnostics — left/right comparison — {self.test_id}",
            fontsize=15,
            fontweight="bold",
        )
        self.ankle_plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.ankle_plot_path, dpi=160, bbox_inches="tight")
        figure.savefig(self.ankle_plot_svg_path, format="svg", bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _high_frequency_residual(
        values: np.ndarray,
        times: np.ndarray,
        baseline_window_s: float = 0.2,
    ) -> tuple[np.ndarray, float]:
        if len(values) < 3:
            return np.zeros_like(values), 0.0
        dt_values = np.diff(times)
        positive_dt = dt_values[dt_values > 0.0]
        if not len(positive_dt):
            return np.zeros_like(values), 0.0
        dt = float(np.median(positive_dt))
        window_samples = max(int(round(float(baseline_window_s) / dt)), 3)
        if window_samples % 2 == 0:
            window_samples += 1
        if window_samples > len(values):
            window_samples = len(values) if len(values) % 2 == 1 else len(values) - 1
        if window_samples < 3:
            return np.zeros_like(values), 0.0
        half_window = window_samples // 2
        kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
        baseline = np.empty_like(values, dtype=np.float64)
        for column in range(values.shape[1]):
            padded = np.pad(values[:, column], (half_window, half_window), mode="reflect")
            baseline[:, column] = np.convolve(padded, kernel, mode="valid")
        return np.asarray(values, dtype=np.float64) - baseline, (window_samples - 1) * dt

    def _write_ankle_high_frequency_plot(
        self,
        model_output: np.ndarray,
        actuator_torque: np.ndarray,
        joint_angle: np.ndarray,
    ) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        times = np.asarray(self.sample_times, dtype=np.float64)
        output_residual, actual_window_s = self._high_frequency_residual(model_output, times)
        torque_residual, _ = self._high_frequency_residual(actuator_torque, times)
        angle_residual, _ = self._high_frequency_residual(joint_angle, times)
        settling_exclusion_s = min(1.0, 0.1 * float(times[-1])) if len(times) else 0.0
        display_mask = times >= settling_exclusion_s
        if int(np.count_nonzero(display_mask)) < 3:
            display_mask = np.ones_like(times, dtype=bool)
            settling_exclusion_s = 0.0
        display_times = times[display_mask]

        figure, axes = plt.subplots(
            3,
            2,
            figsize=(18.0, 11.0),
            sharex=True,
            layout="constrained",
        )
        left_color = "#1F77B4"
        right_color = "#D62728"
        groups = (
            (
                "Ankle pitch high-frequency residual",
                ANKLE_JOINT_NAMES.index("left_ankle_pitch_joint"),
                ANKLE_JOINT_NAMES.index("right_ankle_pitch_joint"),
            ),
            (
                "Ankle roll high-frequency residual",
                ANKLE_JOINT_NAMES.index("left_ankle_roll_joint"),
                ANKLE_JOINT_NAMES.index("right_ankle_roll_joint"),
            ),
        )
        signals = (
            (output_residual, "Model-output residual"),
            (torque_residual, "Actuator-torque residual (N·m)"),
            (angle_residual, "Joint-angle residual (rad)"),
        )
        for column, (title, left_index, right_index) in enumerate(groups):
            axes[0, column].set_title(title, fontweight="bold")
            for row, (signal, ylabel) in enumerate(signals):
                axes[row, column].plot(
                    display_times,
                    signal[display_mask, left_index],
                    color=left_color,
                    linewidth=0.85,
                    label="left",
                )
                axes[row, column].plot(
                    display_times,
                    signal[display_mask, right_index],
                    color=right_color,
                    linewidth=0.85,
                    label="right",
                )
                axes[row, column].axhline(0.0, color="#666666", linewidth=0.5)
                axes[row, column].grid(True, alpha=0.22)
                axes[row, column].legend(loc="upper right", fontsize=8)
                if column == 0:
                    axes[row, column].set_ylabel(ylabel)
        axes[2, 0].set_xlabel("Simulation time (s)")
        axes[2, 1].set_xlabel("Simulation time (s)")
        figure.suptitle(
            "ArmHack Stand ankle small-scale/high-frequency diagnostic\n"
            f"signal minus centered {actual_window_s:.3f}s moving-average baseline; "
            f"display starts at {settling_exclusion_s:.3f}s",
            fontsize=14,
            fontweight="bold",
        )
        self.ankle_high_frequency_svg_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.ankle_high_frequency_svg_path, format="svg", bbox_inches="tight")
        plt.close(figure)

        stats: dict[str, dict[str, dict[str, float]]] = {}
        for index, name in enumerate(ANKLE_JOINT_NAMES):
            stats[name] = {
                "model_output_residual": _component_stats(output_residual[display_mask, index]),
                "actuator_torque_residual_nm": _component_stats(
                    torque_residual[display_mask, index]
                ),
                "joint_angle_residual_rad": _component_stats(angle_residual[display_mask, index]),
            }
        return {
            "method": "signal_minus_centered_moving_average",
            "baseline_window_s": float(actual_window_s),
            "settling_exclusion_s": float(settling_exclusion_s),
            "statistics": stats,
            "plot_path": str(self.ankle_high_frequency_svg_path),
        }

    def finalize(self, generic_report: dict[str, Any], sim_time: float, control_dt: float) -> dict[str, Any]:
        if not self.joint_samples or not self.torso_delta_samples:
            raise RuntimeError("ArmHack Stand MuJoCo report has no control samples.")
        if self.interactive:
            self._trim_timeline_at(float(sim_time))
        joint_samples = np.stack(self.joint_samples)
        target_samples = np.stack(self.arm_target_samples)
        torso_delta = np.stack(self.torso_delta_samples)
        actual_arm = joint_samples[:, self.arm_policy_indices]
        arm_error = actual_arm - target_samples
        abs_step_delta = np.mean(np.abs(np.diff(joint_samples, axis=0)), axis=0) if len(joint_samples) > 1 else np.zeros(joint_samples.shape[1])

        joint_statistics: dict[str, dict[str, float | str]] = {}
        for index, name in enumerate(self.policy_joint_names):
            stats = _component_stats(joint_samples[:, index])
            stats["mean_abs_step_delta_rad"] = float(abs_step_delta[index])
            stats["group"] = "arm_input_joint" if name in ARM_JOINT_NAMES else "balance_policy_joint"
            joint_statistics[name] = stats

        arm_tracking: dict[str, dict[str, float]] = {}
        for index, name in enumerate(ARM_JOINT_NAMES):
            error = arm_error[:, index]
            arm_tracking[name] = {
                "mean_abs_error_rad": float(np.mean(np.abs(error))),
                "rms_error_rad": float(np.sqrt(np.mean(np.square(error)))),
                "max_abs_error_rad": float(np.max(np.abs(error))),
            }

        component_names = ("delta_x_w", "delta_y_w", "delta_z_w", "delta_roll_w", "delta_pitch_w", "delta_yaw_w")
        torso_statistics = {name: _component_stats(torso_delta[:, index]) for index, name in enumerate(component_names)}
        torso_norms = {
            "horizontal_translation_norm": np.linalg.norm(torso_delta[:, :2], axis=1),
            "translation_3d_norm": np.linalg.norm(torso_delta[:, :3], axis=1),
            "rpy_displacement_norm": np.linalg.norm(torso_delta[:, 3:], axis=1),
        }
        torso_norm_statistics = {name: _norm_stats(values) for name, values in torso_norms.items()}
        ankle_diagnostics: dict[str, Any] = {
            "enabled": self.ankle_diagnostics_enabled,
            "joint_order": ANKLE_JOINT_NAMES,
            "print_hz": self.ankle_print_hz,
            "trace_path": str(self.ankle_trace_path) if self.ankle_diagnostics_enabled else "",
            "plot_path": str(self.ankle_plot_path) if self.ankle_diagnostics_enabled else "",
            "plot_svg_path": str(self.ankle_plot_svg_path) if self.ankle_diagnostics_enabled else "",
            "high_frequency_svg_path": (
                str(self.ankle_high_frequency_svg_path) if self.ankle_diagnostics_enabled else ""
            ),
            "joint_statistics": {},
            "high_frequency": {},
        }
        if self.ankle_diagnostics_enabled:
            if not self.ankle_model_output_samples:
                raise RuntimeError("ArmHack Stand ankle diagnostics has no control samples.")
            ankle_model_output = np.stack(self.ankle_model_output_samples)
            ankle_torque_command = np.stack(self.ankle_torque_command_samples)
            ankle_actuator_torque = np.stack(self.ankle_actuator_torque_samples)
            ankle_joint_angle = np.stack(self.ankle_angle_samples)
            ankle_target_angle = np.stack(self.ankle_target_angle_samples)
            for index, name in enumerate(ANKLE_JOINT_NAMES):
                ankle_diagnostics["joint_statistics"][name] = {
                    "model_output": _component_stats(ankle_model_output[:, index]),
                    "torque_command_nm": _component_stats(ankle_torque_command[:, index]),
                    "actuator_torque_nm": _component_stats(ankle_actuator_torque[:, index]),
                    "joint_angle_rad": _component_stats(ankle_joint_angle[:, index]),
                    "target_angle_rad": _component_stats(ankle_target_angle[:, index]),
                    "tracking_error_rad": _component_stats(
                        ankle_joint_angle[:, index] - ankle_target_angle[:, index]
                    ),
                }
            self._write_ankle_trace(
                ankle_model_output,
                ankle_torque_command,
                ankle_actuator_torque,
                ankle_joint_angle,
                ankle_target_angle,
            )
            self._write_ankle_plot(
                ankle_model_output,
                ankle_actuator_torque,
                ankle_joint_angle,
                ankle_target_angle,
            )
            ankle_diagnostics["high_frequency"] = self._write_ankle_high_frequency_plot(
                ankle_model_output,
                ankle_actuator_torque,
                ankle_joint_angle,
            )
        complete = self.last_target_time >= self.csv_duration_s - 0.5 * float(control_dt)
        healthy = bool(generic_report.get("health", {}).get("healthy", False))
        checkpoint_sha = self.checkpoint_sha256
        policy_path = Path(str(self.config["policy_path"])).expanduser().resolve()

        result: dict[str, Any] = {
            "simulator": "MuJoCo",
            "test_id": self.test_id,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "policy_path": str(policy_path),
            "policy_sha256": _sha256(policy_path),
            "test_csv": str(self.csv_path),
            "test_csv_sha256": _sha256(self.csv_path),
            "manifest": str(self.manifest_path),
            "manifest_sha256": _sha256(self.manifest_path),
            "manifest_schema_version": int(self.manifest["schema_version"]),
            "control_dt_s": float(control_dt),
            "control_samples": len(self.sample_times),
            "sim_time_s": float(sim_time),
            "csv_duration_s": self.csv_duration_s,
            "complete_csv_playback": complete,
            "interactive": self.interactive,
            "policy_activation_time_s": self.policy_activation_time_s,
            "startup_complete": self.startup_complete,
            "space_switch_count": self.space_switch_count,
            "healthy": healthy,
            "health_failure_count": 0 if healthy else 1,
            "payload_kg_per_wrist": self.payload_kg,
            "payload_bodies": self.payload_report,
            "joint_statistics": joint_statistics,
            "arm_tracking": arm_tracking,
            "arm_tracking_overall": {
                "mean_abs_error_rad": float(np.mean(np.abs(arm_error))),
                "rms_error_rad": float(np.sqrt(np.mean(np.square(arm_error)))),
                "max_abs_error_rad": float(np.max(np.abs(arm_error))),
            },
            "torso_world_6d": torso_statistics,
            "torso_world_norms": torso_norm_statistics,
            "ankle_diagnostics": ankle_diagnostics,
            "timeline": self.timeline,
            "report_path": str(self.report_path),
            "plot_path": str(self.plot_path),
            "trace_path": str(self.trace_path),
            "ankle_trace_path": ankle_diagnostics["trace_path"],
            "ankle_plot_path": ankle_diagnostics["plot_path"],
            "ankle_plot_svg_path": ankle_diagnostics["plot_svg_path"],
            "ankle_high_frequency_svg_path": ankle_diagnostics["high_frequency_svg_path"],
        }

        self._write_trace(joint_samples, target_samples, torso_delta)
        self._write_plot(torso_delta)
        self._write_markdown(result, generic_report)
        return result

    def _write_markdown(self, result: dict[str, Any], generic_report: dict[str, Any]) -> None:
        health = generic_report.get("health", {})
        important = generic_report.get("important_metrics", {})
        torso = result["torso_world_6d"]
        norms = result["torso_world_norms"]
        ankle = result["ankle_diagnostics"]
        lines = [
            "# ArmHack Stand MuJoCo sim2sim 测试报告",
            "",
            "## 测试身份",
            "",
            f"- simulator：`MuJoCo`",
            f"- checkpoint：`{result['checkpoint']}`",
            f"- checkpoint SHA-256：`{result['checkpoint_sha256']}`",
            f"- exported TorchScript：`{result['policy_path']}`",
            f"- exported policy SHA-256：`{result['policy_sha256']}`",
            f"- 测试项：`{result['test_id']}`",
            f"- 测试 CSV：`{result['test_csv']}`",
            f"- 测试 CSV SHA-256：`{result['test_csv_sha256']}`",
            f"- manifest schema：`v{result['manifest_schema_version']}`，SHA-256 `{result['manifest_sha256']}`",
            f"- 控制样本：`{result['control_samples']}`，控制周期 `{result['control_dt_s']:.6f} s`",
            f"- CSV 时长：`{result['csv_duration_s']:.3f} s`，完整播放：`{result['complete_csv_playback']}`",
            f"- 交互状态机：`{result['interactive']}`；policy 激活时刻：`{result['policy_activation_time_s']}`；初始化完成：`{result['startup_complete']}`；SPACE 次数：`{result['space_switch_count']}`",
            f"- 左/右 wrist-yaw 末端附加质量：各 `{result['payload_kg_per_wrist']:.3f} kg`",
            f"- MuJoCo health：`{result['healthy']}`，health failure count：`{result['health_failure_count']}`，fall time：`{health.get('fall_time')}`",
            "- 输入范围：CSV 只覆盖 14 个双臂关节；15 个腰腿关节仍由 policy 控制。覆盖后的 29 维 raw action 会写回下一帧 `last_action`。",
            "",
            "## 结论",
            "",
            f"- 完整稳定通过：`{bool(result['complete_csv_playback'] and result['healthy'])}`。判据为完整执行初始化 CSV 且 MuJoCo health 全程有效。",
            f"- 最低 root 高度：`{float(health.get('min_root_height', 0.0)):.6f} m`；最大绝对 roll/pitch：`{float(health.get('max_abs_roll', 0.0)):.6f} / {float(health.get('max_abs_pitch', 0.0)):.6f} rad`。",
            f"- torso 水平位移 RMS/最大值：`{norms['horizontal_translation_norm']['rms']:.6f} / {norms['horizontal_translation_norm']['max']:.6f} m`。",
            f"- torso pitch 位移 RMS/最大绝对值：`{torso['delta_pitch_w']['rms']:.6f} / {torso['delta_pitch_w']['max_abs']:.6f} rad`。",
            f"- 双臂实际跟踪总体 MAE/RMS/最大误差：`{result['arm_tracking_overall']['mean_abs_error_rad']:.6f} / {result['arm_tracking_overall']['rms_error_rad']:.6f} / {result['arm_tracking_overall']['max_abs_error_rad']:.6f} rad`。",
            "",
            "## 每关节实际波动",
            "",
            "| 关节 | 分组 | 平均逐步波动 rad/step | 实际角均值 rad | 标准差 rad | 极差 rad |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name in self.policy_joint_names:
            stats = result["joint_statistics"][name]
            group = "双臂输入关节" if stats["group"] == "arm_input_joint" else "平衡策略关节"
            lines.append(
                f"| `{name}` | {group} | {stats['mean_abs_step_delta_rad']:.8f} | "
                f"{stats['mean']:.8f} | {stats['std']:.8f} | {stats['range']:.8f} |"
            )
        if ankle["enabled"]:
            lines += [
                "",
                "## 左右踝策略输出、力矩与角度",
                "",
                f"- 终端打印频率：`{float(ankle['print_hz']):.3f} Hz`。",
                "- `model output` 是 29 维 actor 原始输出中的踝关节分量；`actuator torque` 是 MuJoCo `data.actuator_force` 的实际执行器力矩。",
                "",
                "| 关节 | 输出均值 | 输出标准差 | 输出最大绝对值 | 力矩 RMS N·m | 力矩最大绝对值 N·m | 角度均值 rad | 角度标准差 rad | 跟踪误差 RMS rad |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for name in ANKLE_JOINT_NAMES:
                stats = ankle["joint_statistics"][name]
                output = stats["model_output"]
                torque = stats["actuator_torque_nm"]
                angle = stats["joint_angle_rad"]
                error = stats["tracking_error_rad"]
                lines.append(
                    f"| `{name}` | {output['mean']:.8f} | {output['std']:.8f} | "
                    f"{output['max_abs']:.8f} | {torque['rms']:.8f} | "
                    f"{torque['max_abs']:.8f} | {angle['mean']:.8f} | "
                    f"{angle['std']:.8f} | {error['rms']:.8f} |"
                )
            lines += [
                "",
                "### 左右踝对比曲线",
                "",
                f"![ArmHack Stand MuJoCo ankle comparison]({self.ankle_plot_path.name})",
                "",
                f"- 完整对比矢量图：`{ankle['plot_svg_path']}`",
                f"- 小尺度/高频残差矢量图：`{ankle['high_frequency_svg_path']}`",
                f"- 高频残差定义：原信号减去居中的 `{float(ankle['high_frequency']['baseline_window_s']):.3f} s` 移动平均；绘图跳过最初 `{float(ankle['high_frequency']['settling_exclusion_s']):.3f} s` 启动段。",
                "",
                "| 关节 | 输出高频残差 RMS | 力矩高频残差 RMS N·m | 角度高频残差 RMS rad |",
                "|---|---:|---:|---:|",
            ]
            for name in ANKLE_JOINT_NAMES:
                high_frequency = ankle["high_frequency"]["statistics"][name]
                lines.append(
                    f"| `{name}` | {high_frequency['model_output_residual']['rms']:.8f} | "
                    f"{high_frequency['actuator_torque_residual_nm']['rms']:.8f} | "
                    f"{high_frequency['joint_angle_residual_rad']['rms']:.8f} |"
                )
        lines += [
            "",
            "## 双臂目标跟踪误差",
            "",
            "| 关节 | MAE rad | RMS rad | 最大绝对误差 rad |",
            "|---|---:|---:|---:|",
        ]
        for name in ARM_JOINT_NAMES:
            stats = result["arm_tracking"][name]
            lines.append(
                f"| `{name}` | {stats['mean_abs_error_rad']:.8f} | {stats['rms_error_rad']:.8f} | {stats['max_abs_error_rad']:.8f} |"
            )
        lines += [
            "",
            "## 躯干世界坐标系 6D 位移",
            "",
            "| 分量 | 单位 | 有符号均值 | 绝对值均值 | 标准差 | RMS | 最大绝对值 | 极差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, unit in zip(("delta_x_w", "delta_y_w", "delta_z_w", "delta_roll_w", "delta_pitch_w", "delta_yaw_w"), ("m", "m", "m", "rad", "rad", "rad"), strict=True):
            stats = torso[name]
            lines.append(
                f"| `{name}` | {unit} | {stats['mean']:.8f} | {stats['mean_abs']:.8f} | "
                f"{stats['std']:.8f} | {stats['rms']:.8f} | {stats['max_abs']:.8f} | {stats['range']:.8f} |"
            )
        lines += [
            "",
            "| 综合位移 | 单位 | 均值 | 标准差 | RMS | 最大值 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, unit in (("horizontal_translation_norm", "m"), ("translation_3d_norm", "m"), ("rpy_displacement_norm", "rad")):
            stats = norms[name]
            lines.append(f"| `{name}` | {unit} | {stats['mean']:.8f} | {stats['std']:.8f} | {stats['rms']:.8f} | {stats['max']:.8f} |")
        lines += [
            "",
            "### 6D 曲线与阶段",
            "",
            f"![ArmHack Stand MuJoCo torso world-frame 6D displacement]({self.plot_path.name})",
            "",
            "| 开始 s | 结束 s | 类型 | 姿态或轨迹阶段 |",
            "|---:|---:|---|---|",
        ]
        for stage in self.timeline:
            lines.append(
                f"| {float(stage['start_s']):.3f} | {float(stage['end_s']):.3f} | "
                f"`{stage['kind']}` | `{stage['label']}` |"
            )
        lines += ["", "## MuJoCo Important Metrics", "", "| 指标 | 均值 |", "|---|---:|"]
        for name in sorted(important):
            lines.append(f"| `{name}` | {float(important[name]):.8f} |")
        lines += [
            "",
            "## 输出文件",
            "",
            f"- JSON：`{self.config.get('metrics_path', '')}`",
            f"- 逐帧 CSV：`{self.trace_path}`",
            f"- 6D PNG：`{self.plot_path}`",
            f"- 踝关节逐帧 CSV：`{ankle['trace_path']}`",
            f"- 踝关节左右对比 PNG：`{ankle['plot_path']}`",
            f"- 踝关节左右对比 SVG：`{ankle['plot_svg_path']}`",
            f"- 踝关节小尺度/高频残差 SVG：`{ankle['high_frequency_svg_path']}`",
            "",
            "## 结论边界",
            "",
            "该报告验证的是当前 MuJoCo XML、PD 参数与 schema v5 双臂输入下的 sim2sim 行为。交互模式还验证 ENTER/SPACE 状态机，但不替代真机吊架测试。若 `complete_csv_playback=False` 或 `healthy=False`，不能判定该测试项完整稳定通过。",
            "",
        ]
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")
