"""MuJoCo perturbation adapter for the full-body extreme Stand policy.

Unlike the ArmHack adapters, this class never edits the policy action.  It only
randomizes the initial MuJoCo state and applies short external wrenches to one
random body at a time, so all 29 actuator targets still come from the actor.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_PERTURB_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_knee_link",
    "right_knee_link",
)

FOOT_SPACING_JOINT_COEFFICIENTS = {
    "left_hip_roll_joint": 1.0,
    "right_hip_roll_joint": -1.0,
    "left_ankle_roll_joint": -1.0,
    "right_ankle_roll_joint": 1.0,
}

LARGE_PUSH_DIRECTIONS = (
    ("forward", np.asarray([1.0, 0.0, 0.0], dtype=np.float64)),
    ("backward", np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)),
    ("left", np.asarray([0.0, 1.0, 0.0], dtype=np.float64)),
    ("right", np.asarray([0.0, -1.0, 0.0], dtype=np.float64)),
)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": math.nan,
            "mean_abs": math.nan,
            "rms": math.nan,
            "std": math.nan,
            "p95_abs": math.nan,
            "max_abs": math.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "std": float(np.std(values)),
        "p95_abs": float(np.percentile(np.abs(values), 95.0)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _position_band_rms(
    positions: np.ndarray,
    sample_dt: float,
    low_hz: float = 20.0,
    high_hz: float = 25.0,
) -> np.ndarray:
    """Return per-joint RMS position energy in a closed frequency band."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 8 or sample_dt <= 0.0:
        return np.full(positions.shape[1] if positions.ndim == 2 else 0, math.nan)
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0) / positions.shape[0]
    power = np.square(np.abs(spectrum))
    if positions.shape[0] > 1:
        if positions.shape[0] % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
    frequencies = np.fft.rfftfreq(positions.shape[0], d=sample_dt)
    band = np.logical_and(frequencies >= low_hz, frequencies <= high_hz + 1.0e-9)
    if not np.any(band):
        return np.full(positions.shape[1], math.nan)
    return np.sqrt(np.sum(power[band], axis=0))


def _euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class ExtremeStandRecoveryPerturbation:
    """Deterministic-seed initial-state and intermittent-wrench test driver."""

    def __init__(self, config: dict, policy_joint_names: list[str]):
        self.config = config
        self.policy_joint_names = policy_joint_names
        self.rng = np.random.default_rng(int(config.get("extreme_stand_recovery_seed", 20260719)))
        self.leg_noise = float(config.get("extreme_stand_recovery_leg_noise_rad", 0.20))
        self.waist_noise = float(config.get("extreme_stand_recovery_waist_noise_rad", 0.25))
        self.arm_noise = float(config.get("extreme_stand_recovery_arm_noise_rad", 0.45))
        self.joint_velocity_noise = float(
            config.get("extreme_stand_recovery_joint_velocity_noise_rad_s", 0.75)
        )
        self.root_roll_pitch_noise = float(
            config.get("extreme_stand_recovery_root_roll_pitch_noise_rad", 0.18)
        )
        self.root_yaw_noise = float(
            config.get("extreme_stand_recovery_root_yaw_noise_rad", 0.0)
        )
        self.root_linear_velocity_noise = float(
            config.get("extreme_stand_recovery_root_linear_velocity_noise_m_s", 0.30)
        )
        self.root_angular_velocity_noise = float(
            config.get("extreme_stand_recovery_root_angular_velocity_noise_rad_s", 0.50)
        )
        self.force_max = float(config.get("extreme_stand_recovery_force_max_n", 35.0))
        self.torque_max = float(config.get("extreme_stand_recovery_torque_max_nm", 5.0))
        self.interval = float(config.get("extreme_stand_recovery_wrench_interval_s", 2.5))
        self.duration = float(config.get("extreme_stand_recovery_wrench_duration_s", 0.25))
        self.joint_limit_margin = float(
            config.get("extreme_stand_recovery_joint_limit_margin_rad", 0.02)
        )
        self.recovery_joint_mae_threshold = float(
            config.get("extreme_stand_recovery_joint_mae_threshold_rad", 0.12)
        )
        self.recovery_joint_max_threshold = float(
            config.get("extreme_stand_recovery_joint_max_threshold_rad", 0.20)
        )
        self.recovery_hold_time = float(
            config.get("extreme_stand_recovery_hold_time_s", 1.0)
        )
        self.recovery_final_window = float(
            config.get("extreme_stand_recovery_final_window_s", 1.0)
        )
        self.steady_start_time = float(
            config.get("extreme_stand_recovery_steady_start_s", 10.0)
        )
        self.feet_gaussian_variance = float(
            config.get("extreme_stand_recovery_feet_gaussian_variance_m2", 1.0e-4)
        )
        self.jerk_reward_weight = float(
            config.get("extreme_stand_recovery_joint_jerk_reward_weight", -1.0e-8)
        )
        self.foot_spacing_start_random = bool(
            config.get("extreme_stand_recovery_foot_spacing_start_random", False)
        )
        self.foot_spacing_min_delta = float(
            config.get("extreme_stand_recovery_foot_spacing_min_delta_m", 0.05)
        )
        self.foot_spacing_max_delta = float(
            config.get("extreme_stand_recovery_foot_spacing_max_delta_m", 0.12)
        )
        self.foot_spacing_max_roll_offset = float(
            config.get("extreme_stand_recovery_foot_spacing_max_roll_offset_rad", 0.35)
        )
        self.foot_spacing_search_samples = int(
            config.get("extreme_stand_recovery_foot_spacing_search_samples", 141)
        )
        self.foot_spacing_recovery_tolerance = float(
            config.get("extreme_stand_recovery_foot_spacing_recovery_tolerance_m", 0.02)
        )
        self.interactive_enabled = bool(
            config.get("extreme_stand_recovery_interactive_enable", False)
        )
        self.large_push_enabled = bool(
            config.get("extreme_stand_recovery_large_push_enable", False)
        )
        self.large_push_force = float(
            config.get("extreme_stand_recovery_large_push_force_n", 120.0)
        )
        self.large_push_duration = float(
            config.get("extreme_stand_recovery_large_push_duration_s", 0.20)
        )
        self.large_push_time = float(
            config.get("extreme_stand_recovery_large_push_time_s", 5.0)
        )
        self.large_push_direction_index = int(
            config.get("extreme_stand_recovery_large_push_direction_index", -1)
        )
        self.large_push_body_name = str(
            config.get("extreme_stand_recovery_large_push_body_name", "torso_link")
        )
        self.post_push_settle_time = float(
            config.get("extreme_stand_recovery_post_push_settle_s", 2.0)
        )
        self.target_limiter_enabled = bool(
            config.get("extreme_stand_recovery_target_limiter_enable", False)
        )
        target_velocity_limits_by_group = {
            "leg": float(
                config.get(
                    "extreme_stand_recovery_target_leg_velocity_limit_rad_s", 25.0
                )
            ),
            "waist": float(
                config.get(
                    "extreme_stand_recovery_target_waist_velocity_limit_rad_s", 10.0
                )
            ),
            "arm": float(
                config.get(
                    "extreme_stand_recovery_target_arm_velocity_limit_rad_s", 15.0
                )
            ),
        }
        target_acceleration_limits_by_group = {
            "leg": float(
                config.get(
                    "extreme_stand_recovery_target_leg_acceleration_limit_rad_s2", 600.0
                )
            ),
            "waist": float(
                config.get(
                    "extreme_stand_recovery_target_waist_acceleration_limit_rad_s2", 250.0
                )
            ),
            "arm": float(
                config.get(
                    "extreme_stand_recovery_target_arm_acceleration_limit_rad_s2", 400.0
                )
            ),
        }
        if min(
            *target_velocity_limits_by_group.values(),
            *target_acceleration_limits_by_group.values(),
        ) <= 0.0:
            raise ValueError("Extreme Stand target velocity/acceleration limits must be positive.")

        def target_group(joint_name: str) -> str:
            if joint_name.startswith("waist_"):
                return "waist"
            if any(token in joint_name for token in ("shoulder", "elbow", "wrist")):
                return "arm"
            return "leg"

        self.target_velocity_limits = np.asarray(
            [target_velocity_limits_by_group[target_group(name)] for name in policy_joint_names],
            dtype=np.float64,
        )
        self.target_acceleration_limits = np.asarray(
            [
                target_acceleration_limits_by_group[target_group(name)]
                for name in policy_joint_names
            ],
            dtype=np.float64,
        )
        self._target_limiter_position: np.ndarray | None = None
        self._target_limiter_velocity: np.ndarray | None = None
        self.latest_raw_target_position = np.full(
            len(policy_joint_names), math.nan, dtype=np.float64
        )
        self.latest_target_velocity = np.zeros(len(policy_joint_names), dtype=np.float64)
        self.latest_target_acceleration = np.zeros(
            len(policy_joint_names), dtype=np.float64
        )
        self.latest_target_velocity_clipped_count = 0
        self.latest_target_acceleration_clipped_count = 0
        self.target_limiter_velocity_clip_total = 0
        self.target_limiter_acceleration_clip_total = 0
        self.random_pose_active = (
            bool(config.get("extreme_stand_recovery_interactive_pose_start_random", False))
            if self.interactive_enabled
            else True
        )
        self.wrench_enabled = (
            bool(config.get("extreme_stand_recovery_interactive_wrench_start_enabled", False))
            if self.interactive_enabled
            else True
        )
        self.random_foot_spacing_active = self.foot_spacing_start_random
        self.body_names = tuple(
            config.get("extreme_stand_recovery_body_names", DEFAULT_PERTURB_BODY_NAMES)
        )
        if self.interval <= 0.0 or self.duration <= 0.0 or self.duration > self.interval:
            raise ValueError("Extreme Stand wrench timing requires 0 < duration <= interval.")
        if min(
            self.leg_noise,
            self.waist_noise,
            self.arm_noise,
            self.joint_velocity_noise,
            self.root_roll_pitch_noise,
            self.root_yaw_noise,
            self.root_linear_velocity_noise,
            self.root_angular_velocity_noise,
            self.force_max,
            self.torque_max,
            self.joint_limit_margin,
            self.recovery_joint_mae_threshold,
            self.recovery_joint_max_threshold,
            self.recovery_hold_time,
            self.recovery_final_window,
            self.steady_start_time,
            self.foot_spacing_min_delta,
            self.foot_spacing_max_delta,
            self.foot_spacing_max_roll_offset,
            self.foot_spacing_recovery_tolerance,
            self.large_push_force,
            self.large_push_duration,
            self.large_push_time,
            self.post_push_settle_time,
        ) < 0.0:
            raise ValueError("Extreme Stand perturbation magnitudes must be non-negative.")
        if self.large_push_enabled and self.large_push_duration <= 0.0:
            raise ValueError("Extreme Stand large-push duration must be positive.")
        if self.large_push_direction_index >= len(LARGE_PUSH_DIRECTIONS):
            raise ValueError(
                "Extreme Stand large-push direction index must be -1 or in "
                f"[0, {len(LARGE_PUSH_DIRECTIONS) - 1}]."
            )
        if self.foot_spacing_min_delta > self.foot_spacing_max_delta:
            raise ValueError(
                "Extreme Stand foot-spacing minimum delta must not exceed the maximum delta."
            )
        if self.foot_spacing_search_samples < 3:
            raise ValueError("Extreme Stand foot-spacing search requires at least 3 samples.")
        if self.feet_gaussian_variance <= 0.0:
            raise ValueError("Extreme Stand feet Gaussian variance must be positive.")
        if self.jerk_reward_weight > 0.0:
            raise ValueError("Extreme Stand jerk reward weight must be non-positive.")
        self.body_ids: list[int] = []
        self.next_wrench_time = self.interval
        self.active_until = -1.0
        self.active_body_id = -1
        self.active_body_name = ""
        self.active_wrench = np.zeros(6, dtype=np.float64)
        self.active_wrench_source = ""
        self.event_count = 0
        self.wrench_events: list[dict[str, object]] = []
        self.large_push_body_id = -1
        self.large_push_fired = False
        self.large_push_event_count = 0
        self.space_cycle_index = 0
        self.initial_joint_limit_clip_count = 0
        self.qpos_addresses: dict[str, int] = {}
        self.joint_ids: dict[str, int] = {}
        self.default_joint_positions = np.zeros(len(self.policy_joint_names), dtype=np.float64)
        self.initial_joint_abs_errors = np.zeros(len(self.policy_joint_names), dtype=np.float64)
        self.joint_error_times: list[float] = []
        self.joint_abs_error_samples: list[np.ndarray] = []
        self.qvel_addresses: dict[str, int] = {}
        self.foot_body_ids: list[int] = []
        self.default_foot_planar_distance_m = math.nan
        self.default_foot_min_z_m = math.nan
        self.initial_foot_planar_distance_m = math.nan
        self.foot_spacing_target_distance_m = math.nan
        self.foot_spacing_roll_offset_rad = 0.0
        self.foot_spacing_attainable_range_m = (math.nan, math.nan)
        self.foot_spacing_reset_count = 0
        self.foot_spacing_reset_events: list[dict[str, object]] = []
        self.current_reset_mode = "default"
        self.last_reset_time_s = 0.0
        self.default_qpos: np.ndarray | None = None
        self.default_qvel: np.ndarray | None = None
        self.motion_times: list[float] = []
        self.joint_position_samples: list[np.ndarray] = []
        self.joint_velocity_samples: list[np.ndarray] = []
        self.joint_acceleration_samples: list[np.ndarray] = []
        self.joint_jerk_samples: list[np.ndarray] = []
        self.actor_action_samples: list[np.ndarray] = []
        self.raw_target_position_samples: list[np.ndarray] = []
        self.target_position_samples: list[np.ndarray] = []
        self.target_limiter_velocity_samples: list[np.ndarray] = []
        self.target_limiter_acceleration_samples: list[np.ndarray] = []
        self.pd_torque_command_samples: list[np.ndarray] = []
        self.actuator_force_samples: list[np.ndarray] = []
        self.actuator_torque_limit_samples: list[np.ndarray] = []
        self.foot_planar_distance_samples: list[float] = []
        self._previous_joint_velocity: np.ndarray | None = None
        self._previous_joint_acceleration: np.ndarray | None = None
        self._previous_motion_time: float | None = None
        self.pending_space_cycle = False
        self.pending_wrench_toggle = False
        self.pending_foot_spacing_randomize = False
        self.pending_default_pose_reset = False
        self.pending_camera_toggle = False
        self.interaction_events: list[dict[str, object]] = []
        self.interactive_log_enabled = bool(
            config.get("extreme_stand_recovery_interactive_log_enable", False)
        )
        self.interactive_log_path = Path(
            str(config.get("extreme_stand_recovery_interactive_log_path", ""))
        ).expanduser()
        self.interactive_trials_dir = Path(
            str(config.get("extreme_stand_recovery_interactive_trials_dir", ""))
        ).expanduser()
        self.interactive_events_path = Path(
            str(config.get("extreme_stand_recovery_interactive_events_path", ""))
        ).expanduser()
        self._interactive_headers: list[str] = []
        self._interactive_stream = None
        self._interactive_writer = None
        self._interactive_events_stream = None
        self._interactive_events_writer = None
        self._interactive_trial_stream = None
        self._interactive_trial_writer = None
        self._interactive_trial_paths: list[str] = []
        self._interactive_row_count = 0
        self.current_space_trial_id = 0
        self.current_space_trial_scenario = "startup_before_first_space"
        self.current_space_trial_start_s = 0.0
        self.last_operator_event = "START"
        self.last_operator_event_time_s = 0.0
        self.foot_side_by_body_id: dict[int, str] = {}

    def _joint_noise_limit(self, joint_name: str) -> float:
        if joint_name.startswith("waist_"):
            return self.waist_noise
        if any(token in joint_name for token in ("shoulder", "elbow", "wrist")):
            return self.arm_noise
        return self.leg_noise

    def _reset_target_limiter_from_data(self, data: mujoco.MjData) -> None:
        positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        self._target_limiter_position = positions.copy()
        self._target_limiter_velocity = np.zeros_like(positions)
        self.latest_raw_target_position = positions.copy()
        self.latest_target_velocity = np.zeros_like(positions)
        self.latest_target_acceleration = np.zeros_like(positions)
        self.latest_target_velocity_clipped_count = 0
        self.latest_target_acceleration_clipped_count = 0

    def limit_target_position(
        self,
        raw_target_position: np.ndarray,
        *,
        update: bool,
        control_dt: float,
    ) -> np.ndarray:
        """Limit policy joint-position target velocity and acceleration at control rate."""

        raw_target = np.asarray(raw_target_position, dtype=np.float64)
        expected_shape = (len(self.policy_joint_names),)
        if raw_target.shape != expected_shape:
            raise ValueError(
                "Extreme Stand target limiter shape mismatch: "
                f"expected {expected_shape}, got {raw_target.shape}."
            )
        if control_dt <= 0.0:
            raise ValueError("Extreme Stand target limiter control_dt must be positive.")
        if self._target_limiter_position is None or self._target_limiter_velocity is None:
            self._target_limiter_position = raw_target.copy()
            self._target_limiter_velocity = np.zeros_like(raw_target)

        self.latest_raw_target_position = raw_target.copy()
        if not update:
            return (
                self._target_limiter_position.copy()
                if self.target_limiter_enabled
                else raw_target.copy()
            )

        previous_position = self._target_limiter_position
        previous_velocity = self._target_limiter_velocity
        requested_velocity = (raw_target - previous_position) / control_dt
        if not self.target_limiter_enabled:
            target_velocity = requested_velocity
            target_acceleration = (target_velocity - previous_velocity) / control_dt
            self._target_limiter_position = raw_target.copy()
            self._target_limiter_velocity = target_velocity.copy()
            self.latest_target_velocity = target_velocity.copy()
            self.latest_target_acceleration = target_acceleration.copy()
            self.latest_target_velocity_clipped_count = 0
            self.latest_target_acceleration_clipped_count = 0
            return raw_target.copy()

        velocity_clipped = np.abs(requested_velocity) > self.target_velocity_limits
        velocity_limited = np.clip(
            requested_velocity,
            -self.target_velocity_limits,
            self.target_velocity_limits,
        )
        requested_acceleration = (velocity_limited - previous_velocity) / control_dt
        acceleration_clipped = (
            np.abs(requested_acceleration) > self.target_acceleration_limits
        )
        target_acceleration = np.clip(
            requested_acceleration,
            -self.target_acceleration_limits,
            self.target_acceleration_limits,
        )
        target_velocity = np.clip(
            previous_velocity + target_acceleration * control_dt,
            -self.target_velocity_limits,
            self.target_velocity_limits,
        )
        target_position = previous_position + target_velocity * control_dt
        self._target_limiter_position = target_position.copy()
        self._target_limiter_velocity = target_velocity.copy()
        self.latest_target_velocity = target_velocity.copy()
        self.latest_target_acceleration = target_acceleration.copy()
        self.latest_target_velocity_clipped_count = int(np.count_nonzero(velocity_clipped))
        self.latest_target_acceleration_clipped_count = int(
            np.count_nonzero(acceleration_clipped)
        )
        self.target_limiter_velocity_clip_total += self.latest_target_velocity_clipped_count
        self.target_limiter_acceleration_clip_total += (
            self.latest_target_acceleration_clipped_count
        )
        return target_position.copy()

    def initialize_model_and_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos_addresses: dict[str, int],
        qvel_addresses: dict[str, int],
        foot_body_ids: list[int],
    ) -> None:
        """Cache the default state and select the first default/randomized pose."""

        self.body_ids = []
        valid_body_names: list[str] = []
        for body_name in self.body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self.body_ids.append(int(body_id))
                valid_body_names.append(body_name)
        self.body_names = tuple(valid_body_names)
        if not self.body_ids:
            raise RuntimeError("Extreme Stand MuJoCo test found no configured perturbation bodies.")
        self.large_push_body_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.large_push_body_name,
            )
        )
        if self.large_push_body_id < 0:
            raise RuntimeError(
                "Extreme Stand large-push body was not found in MuJoCo: "
                f"{self.large_push_body_name}"
            )

        self.qpos_addresses = {
            joint_name: int(qpos_addresses[joint_name]) for joint_name in self.policy_joint_names
        }
        self.joint_ids = {
            joint_name: int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            )
            for joint_name in self.policy_joint_names
        }
        missing_joint_ids = [name for name, joint_id in self.joint_ids.items() if joint_id < 0]
        if missing_joint_ids:
            raise RuntimeError(
                "Extreme Stand world-joint logging could not find joints: "
                + ", ".join(missing_joint_ids)
            )
        self.qvel_addresses = {
            joint_name: int(qvel_addresses[joint_name]) for joint_name in self.policy_joint_names
        }
        self.foot_body_ids = [int(body_id) for body_id in foot_body_ids]
        if len(self.foot_body_ids) != 2:
            raise RuntimeError(
                f"Extreme Stand motion-quality test requires two feet, got {len(self.foot_body_ids)}."
            )
        self.foot_side_by_body_id = {}
        for index, body_id in enumerate(self.foot_body_ids):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "left" in body_name.lower():
                side = "left"
            elif "right" in body_name.lower():
                side = "right"
            else:
                side = "left" if index == 0 else "right"
            self.foot_side_by_body_id[body_id] = side
        self.default_joint_positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        self.default_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        self.default_qvel = np.asarray(data.qvel, dtype=np.float64).copy()
        mujoco.mj_forward(model, data)
        default_foot_positions = np.asarray(data.xpos[self.foot_body_ids, :2], dtype=np.float64)
        self.default_foot_planar_distance_m = float(
            np.linalg.norm(default_foot_positions[0] - default_foot_positions[1])
        )
        self.default_foot_min_z_m = float(
            np.min(np.asarray(data.xpos[self.foot_body_ids, 2], dtype=np.float64))
        )
        self._apply_initial_pose(
            model,
            data,
            randomized=self.random_pose_active,
            randomize_foot_spacing=self.random_foot_spacing_active,
            reset_time_s=0.0,
        )
        self.next_wrench_time = self.interval
        self._initialize_interactive_data_logging()

        if self.interactive_enabled:
            print(
                "[Extreme Stand interactive] SPACE: cycle large torso pushes and reset poses; "
                "R: reset immediately to default standing pose; "
                "K: new random foot spacing; F: OFF/ON random wrench; "
                "C: FREE/FOLLOW camera. "
                f"start_pose={'RANDOM' if self.random_pose_active else 'DEFAULT'} "
                f"start_foot_spacing={'RANDOM' if self.random_foot_spacing_active else 'DEFAULT'} "
                f"start_wrench={'ON' if self.wrench_enabled else 'OFF'} "
                f"large_push={self.large_push_force:.1f}N/{self.large_push_duration:.2f}s "
                f"on {self.large_push_body_name}; "
                f"target_limiter={'ON' if self.target_limiter_enabled else 'OFF'}",
                flush=True,
            )

    def _diagnostic_headers(self) -> list[str]:
        headers = [
            "time_s",
            "space_trial_id",
            "space_trial_time_s",
            "space_trial_scenario",
            "reset_mode",
            "last_operator_event",
            "last_operator_event_time_s",
            "command/lin_vel_x_m_s",
            "command/lin_vel_y_m_s",
            "command/yaw_rate_rad_s",
            "target_command/lin_vel_x_m_s",
            "target_command/lin_vel_y_m_s",
            "target_command/yaw_rate_rad_s",
            "target_limiter/enabled",
            "target_limiter/velocity_clipped_joint_count",
            "target_limiter/acceleration_clipped_joint_count",
            "root/pos_x_m",
            "root/pos_y_m",
            "root/pos_z_m",
            "root/quat_w",
            "root/quat_x",
            "root/quat_y",
            "root/quat_z",
            "root/lin_vel_x_m_s",
            "root/lin_vel_y_m_s",
            "root/lin_vel_z_m_s",
            "root/ang_vel_x_rad_s",
            "root/ang_vel_y_rad_s",
            "root/ang_vel_z_rad_s",
            "feet/left_pos_x_world_m",
            "feet/left_pos_y_world_m",
            "feet/left_pos_z_world_m",
            "feet/right_pos_x_world_m",
            "feet/right_pos_y_world_m",
            "feet/right_pos_z_world_m",
            "feet/right_minus_left_x_world_m",
            "feet/right_minus_left_y_world_m",
            "feet/right_minus_left_z_world_m",
            "feet/planar_distance_m",
            "feet/distance_3d_m",
            "feet/default_planar_distance_m",
            "feet/planar_distance_error_m",
            "external/body_id",
            "external/body_name",
            "external/source",
            "external/force_x_world_n",
            "external/force_y_world_n",
            "external/force_z_world_n",
            "external/torque_x_world_nm",
            "external/torque_y_world_nm",
            "external/torque_z_world_nm",
        ]
        for side in ("left", "right"):
            headers.extend(
                [
                    f"ground_reaction/{side}/contact_count",
                    f"ground_reaction/{side}/force_x_world_n",
                    f"ground_reaction/{side}/force_y_world_n",
                    f"ground_reaction/{side}/force_z_world_n",
                    f"ground_reaction/{side}/moment_x_about_foot_world_nm",
                    f"ground_reaction/{side}/moment_y_about_foot_world_nm",
                    f"ground_reaction/{side}/moment_z_about_foot_world_nm",
                ]
            )
        for joint_name in self.policy_joint_names:
            headers.extend(
                [
                    f"joint_anchor_world_m/{joint_name}/x",
                    f"joint_anchor_world_m/{joint_name}/y",
                    f"joint_anchor_world_m/{joint_name}/z",
                ]
            )
        for prefix in (
            "actor_action",
            "raw_target_qpos_rad",
            "target_qpos_rad",
            "target_limiter_velocity_rad_s",
            "target_limiter_acceleration_rad_s2",
            "qpos_rad",
            "qvel_rad_s",
            "qacc_fd_rad_s2",
            "qacc_mujoco_rad_s2",
            "jerk_fd_rad_s3",
            "pd_torque_command_nm",
            "actuator_force_nm",
            "actuator_torque_limit_nm",
        ):
            headers.extend(f"{prefix}/{name}" for name in self.policy_joint_names)
        return headers

    def _initialize_interactive_data_logging(self) -> None:
        if not self.interactive_log_enabled:
            return
        if not self.interactive_enabled:
            raise ValueError("Interactive data logging requires interactive mode.")
        for label, path in (
            ("master log", self.interactive_log_path),
            ("trials directory", self.interactive_trials_dir),
            ("events log", self.interactive_events_path),
        ):
            if str(path) in ("", "."):
                raise ValueError(f"Extreme Stand interactive {label} path is empty.")
        self.interactive_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.interactive_trials_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_events_path.parent.mkdir(parents=True, exist_ok=True)
        self._interactive_headers = self._diagnostic_headers()
        self._interactive_stream = self.interactive_log_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._interactive_writer = csv.writer(self._interactive_stream)
        self._interactive_writer.writerow(self._interactive_headers)
        self._interactive_stream.flush()
        self._interactive_events_stream = self.interactive_events_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._interactive_events_writer = csv.writer(self._interactive_events_stream)
        self._interactive_events_writer.writerow(
            ["time_s", "space_trial_id", "key", "event", "details_json"]
        )
        self._record_operator_event(0.0, "START", "startup_before_first_space", {})
        print(
            "[Extreme Stand data log] 50 Hz master table: "
            f"{self.interactive_log_path.resolve()}",
            flush=True,
        )
        print(
            "[Extreme Stand data log] SPACE trial directory: "
            f"{self.interactive_trials_dir.resolve()}",
            flush=True,
        )

    def _record_operator_event(
        self,
        sim_time: float,
        key: str,
        event: str,
        details: dict[str, object],
    ) -> None:
        self.last_operator_event = f"{key}:{event}"
        self.last_operator_event_time_s = float(sim_time)
        if self._interactive_events_writer is None:
            return
        self._interactive_events_writer.writerow(
            [
                float(sim_time),
                self.current_space_trial_id,
                key,
                event,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
            ]
        )
        self._interactive_events_stream.flush()

    def _begin_space_trial(
        self,
        sim_time: float,
        scenario: str,
        details: dict[str, object],
    ) -> None:
        self.current_space_trial_id += 1
        self.current_space_trial_scenario = str(scenario)
        self.current_space_trial_start_s = float(sim_time)
        if self._interactive_trial_stream is not None:
            self._interactive_trial_stream.flush()
            self._interactive_trial_stream.close()
        self._interactive_trial_stream = None
        self._interactive_trial_writer = None
        if self.interactive_log_enabled:
            safe_scenario = "".join(
                character if character.isalnum() or character in ("-", "_") else "_"
                for character in scenario
            ).strip("_")
            trial_path = self.interactive_trials_dir / (
                f"trial_{self.current_space_trial_id:03d}_{safe_scenario}.csv"
            )
            self._interactive_trial_stream = trial_path.open(
                "w", encoding="utf-8", newline=""
            )
            self._interactive_trial_writer = csv.writer(self._interactive_trial_stream)
            self._interactive_trial_writer.writerow(self._interactive_headers)
            self._interactive_trial_stream.flush()
            self._interactive_trial_paths.append(str(trial_path.resolve()))
        self._record_operator_event(sim_time, "SPACE", scenario, details)
        print(
            "[Extreme Stand data log] "
            f"SPACE trial #{self.current_space_trial_id}: {scenario}",
            flush=True,
        )

    def _bounded_joint_position(
        self,
        model: mujoco.MjModel,
        joint_name: str,
        position: float,
        *,
        count_clip: bool = True,
    ) -> float:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Extreme Stand MuJoCo test joint not found: {joint_name}")
        if not bool(model.jnt_limited[joint_id]):
            return float(position)
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        available_margin = max(0.0, 0.5 * (upper - lower) - 1.0e-6)
        margin = min(self.joint_limit_margin, available_margin)
        bounded = float(np.clip(position, lower + margin, upper - margin))
        if count_clip and not np.isclose(bounded, position, atol=1.0e-12):
            self.initial_joint_limit_clip_count += 1
        return bounded

    def _apply_random_foot_spacing(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        reset_time_s: float,
    ) -> None:
        """Choose a feasible symmetric leg pose whose foot distance differs from default."""

        missing = [
            name
            for name in FOOT_SPACING_JOINT_COEFFICIENTS
            if name not in self.qpos_addresses
        ]
        if missing:
            raise RuntimeError(
                "Extreme Stand foot-spacing test is missing required joints: "
                + ", ".join(missing)
            )

        base_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        candidates: list[tuple[float, float, np.ndarray]] = []
        offsets = np.linspace(
            -self.foot_spacing_max_roll_offset,
            self.foot_spacing_max_roll_offset,
            self.foot_spacing_search_samples,
            dtype=np.float64,
        )
        for offset in offsets:
            candidate_qpos = base_qpos.copy()
            for joint_name, coefficient in FOOT_SPACING_JOINT_COEFFICIENTS.items():
                qpos_address = self.qpos_addresses[joint_name]
                candidate_qpos[qpos_address] = self._bounded_joint_position(
                    model,
                    joint_name,
                    float(base_qpos[qpos_address]) + coefficient * float(offset),
                    count_clip=False,
                )
            data.qpos[:] = candidate_qpos
            mujoco.mj_forward(model, data)
            foot_xy = np.asarray(data.xpos[self.foot_body_ids, :2], dtype=np.float64)
            distance = float(np.linalg.norm(foot_xy[0] - foot_xy[1]))
            candidates.append((distance, float(offset), candidate_qpos))

        distances = np.asarray([candidate[0] for candidate in candidates], dtype=np.float64)
        self.foot_spacing_attainable_range_m = (
            float(np.min(distances)),
            float(np.max(distances)),
        )
        deltas = np.abs(distances - self.default_foot_planar_distance_m)
        feasible_indices = np.flatnonzero(
            np.logical_and(
                deltas >= self.foot_spacing_min_delta,
                deltas <= self.foot_spacing_max_delta,
            )
        )
        if feasible_indices.size == 0:
            data.qpos[:] = base_qpos
            mujoco.mj_forward(model, data)
            raise RuntimeError(
                "Extreme Stand foot-spacing search found no pose in the requested delta range "
                f"[{self.foot_spacing_min_delta:.3f}, {self.foot_spacing_max_delta:.3f}] m; "
                f"attainable distance range is [{distances.min():.3f}, {distances.max():.3f}] m "
                f"around default {self.default_foot_planar_distance_m:.3f} m."
            )

        selected_index = int(self.rng.choice(feasible_indices))
        selected_distance, selected_offset, selected_qpos = candidates[selected_index]
        data.qpos[:] = selected_qpos
        mujoco.mj_forward(model, data)
        current_min_z = float(
            np.min(np.asarray(data.xpos[self.foot_body_ids, 2], dtype=np.float64))
        )
        data.qpos[2] += self.default_foot_min_z_m - current_min_z
        mujoco.mj_forward(model, data)

        foot_xy = np.asarray(data.xpos[self.foot_body_ids, :2], dtype=np.float64)
        self.foot_spacing_target_distance_m = float(selected_distance)
        self.initial_foot_planar_distance_m = float(
            np.linalg.norm(foot_xy[0] - foot_xy[1])
        )
        self.foot_spacing_roll_offset_rad = float(selected_offset)
        self.foot_spacing_reset_count += 1
        event = {
            "event_index": self.foot_spacing_reset_count,
            "time_s": float(reset_time_s),
            "default_distance_m": self.default_foot_planar_distance_m,
            "target_distance_m": self.foot_spacing_target_distance_m,
            "initial_distance_m": self.initial_foot_planar_distance_m,
            "initial_error_m": (
                self.initial_foot_planar_distance_m
                - self.default_foot_planar_distance_m
            ),
            "symmetric_hip_roll_offset_rad": self.foot_spacing_roll_offset_rad,
            "attainable_range_m": list(self.foot_spacing_attainable_range_m),
        }
        self.foot_spacing_reset_events.append(event)
        print(
            "[Extreme Stand foot spacing] "
            f"#{self.foot_spacing_reset_count} t={reset_time_s:.3f}s "
            f"default={self.default_foot_planar_distance_m:.3f}m "
            f"initial={self.initial_foot_planar_distance_m:.3f}m "
            f"error={event['initial_error_m']:+.3f}m "
            f"roll_offset={self.foot_spacing_roll_offset_rad:+.3f}rad",
            flush=True,
        )

    def _apply_initial_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomized: bool,
        randomize_foot_spacing: bool = False,
        reset_time_s: float = 0.0,
    ) -> None:
        """Reset to the cached default state, then optionally add bounded random noise."""

        if self.default_qpos is None or self.default_qvel is None:
            raise RuntimeError("Extreme Stand default state was not cached before reset.")
        data.qpos[:] = self.default_qpos
        data.qvel[:] = self.default_qvel
        data.ctrl[:] = 0.0
        data.qacc[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.initial_joint_limit_clip_count = 0

        if randomized:
            for joint_name in self.policy_joint_names:
                limit = self._joint_noise_limit(joint_name)
                qpos_address = self.qpos_addresses[joint_name]
                noisy_position = float(data.qpos[qpos_address]) + float(
                    self.rng.uniform(-limit, limit)
                )
                noisy_position = self._bounded_joint_position(
                    model, joint_name, noisy_position
                )
                data.qpos[qpos_address] = noisy_position
                data.qvel[self.qvel_addresses[joint_name]] += self.rng.uniform(
                    -self.joint_velocity_noise, self.joint_velocity_noise
                )

            roll = self.rng.uniform(-self.root_roll_pitch_noise, self.root_roll_pitch_noise)
            pitch = self.rng.uniform(-self.root_roll_pitch_noise, self.root_roll_pitch_noise)
            yaw = self.rng.uniform(-self.root_yaw_noise, self.root_yaw_noise)
            perturb_quat = _euler_xyz_to_quat_wxyz(roll, pitch, yaw)
            original_quat = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
            composed_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(composed_quat, perturb_quat, original_quat)
            data.qpos[3:7] = composed_quat / max(float(np.linalg.norm(composed_quat)), 1.0e-12)
            data.qvel[0:3] += self.rng.uniform(
                -self.root_linear_velocity_noise, self.root_linear_velocity_noise, size=3
            )
            data.qvel[3:6] += self.rng.uniform(
                -self.root_angular_velocity_noise, self.root_angular_velocity_noise, size=3
            )

        self.current_reset_mode = (
            "random_foot_spacing"
            if randomize_foot_spacing
            else ("random_pose" if randomized else "default")
        )
        self.last_reset_time_s = float(reset_time_s)
        self.foot_spacing_target_distance_m = self.default_foot_planar_distance_m
        self.foot_spacing_roll_offset_rad = 0.0
        if randomize_foot_spacing:
            self._apply_random_foot_spacing(
                model,
                data,
                reset_time_s=reset_time_s,
            )
        else:
            mujoco.mj_forward(model, data)
            foot_xy = np.asarray(data.xpos[self.foot_body_ids, :2], dtype=np.float64)
            self.initial_foot_planar_distance_m = float(
                np.linalg.norm(foot_xy[0] - foot_xy[1])
            )

        initial_positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        self.initial_joint_abs_errors = np.abs(
            initial_positions - self.default_joint_positions
        )
        self.joint_error_times.clear()
        self.joint_abs_error_samples.clear()
        self.motion_times.clear()
        self.joint_position_samples.clear()
        self.joint_velocity_samples.clear()
        self.joint_acceleration_samples.clear()
        self.joint_jerk_samples.clear()
        self.actor_action_samples.clear()
        self.raw_target_position_samples.clear()
        self.target_position_samples.clear()
        self.target_limiter_velocity_samples.clear()
        self.target_limiter_acceleration_samples.clear()
        self.pd_torque_command_samples.clear()
        self.actuator_force_samples.clear()
        self.actuator_torque_limit_samples.clear()
        self.foot_planar_distance_samples.clear()
        self._previous_joint_velocity = None
        self._previous_joint_acceleration = None
        self._previous_motion_time = None
        mujoco.mj_forward(model, data)
        self._reset_target_limiter_from_data(data)

    def key_callback(self, keycode: int) -> None:
        """Queue GUI interaction; physics state is changed only by the simulation thread."""

        if not self.interactive_enabled:
            return
        if int(keycode) == 32:
            self.pending_space_cycle = True
            print("[Extreme Stand interactive] SPACE requested next push/pose scenario.", flush=True)
        elif int(keycode) in (82, 114):
            self.pending_default_pose_reset = True
            print("[Extreme Stand interactive] R requested default standing reset.", flush=True)
        elif int(keycode) in (75, 107):
            self.pending_foot_spacing_randomize = True
            print("[Extreme Stand interactive] K requested random foot spacing.", flush=True)
        elif int(keycode) in (70, 102):
            self.pending_wrench_toggle = True
            print("[Extreme Stand interactive] F requested external-wrench switch.", flush=True)
        elif int(keycode) in (67, 99):
            self.pending_camera_toggle = True
            print("[Extreme Stand interactive] C requested camera mode switch.", flush=True)

    def _start_large_push(
        self,
        sim_time: float,
        direction_index: int,
        *,
        source: str,
    ) -> None:
        """Start one deterministic horizontal torso push without resetting policy state."""

        direction_index = int(direction_index) % len(LARGE_PUSH_DIRECTIONS)
        direction_name, direction = LARGE_PUSH_DIRECTIONS[direction_index]
        self.active_body_id = self.large_push_body_id
        self.active_body_name = self.large_push_body_name
        self.active_wrench[:] = 0.0
        self.active_wrench[:3] = direction * self.large_push_force
        self.active_until = float(sim_time) + self.large_push_duration
        self.active_wrench_source = source
        self.event_count += 1
        self.large_push_event_count += 1
        event = {
            "event_index": self.event_count,
            "large_push_index": self.large_push_event_count,
            "time_s": float(sim_time),
            "end_time_s": float(self.active_until),
            "source": source,
            "body_id": self.active_body_id,
            "body_name": self.active_body_name,
            "direction": direction_name,
            "force_world_n": [float(value) for value in self.active_wrench[:3]],
            "force_norm_n": self.large_push_force,
            "torque_world_nm": [0.0, 0.0, 0.0],
            "duration_s": self.large_push_duration,
            "impulse_n_s": self.large_push_force * self.large_push_duration,
        }
        self.wrench_events.append(event)
        print(
            "[Extreme Stand LARGE PUSH] "
            f"#{self.large_push_event_count} t={sim_time:.3f}s "
            f"body={self.large_push_body_name} direction={direction_name} "
            f"force={self.large_push_force:.1f}N duration={self.large_push_duration:.3f}s "
            f"impulse={event['impulse_n_s']:.1f}N*s source={source}",
            flush=True,
        )

    def process_interaction_requests(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
    ) -> bool:
        """Apply queued GUI requests and return whether the policy state must be reset."""

        pose_reset = False
        if self.pending_default_pose_reset:
            self.pending_default_pose_reset = False
            # A manual safety reset takes precedence over queued pose/push requests.
            self.pending_space_cycle = False
            self.pending_foot_spacing_randomize = False
            self.random_pose_active = False
            self.random_foot_spacing_active = False
            self._apply_initial_pose(
                model,
                data,
                randomized=False,
                randomize_foot_spacing=False,
                reset_time_s=sim_time,
            )
            self.active_until = -1.0
            self.active_body_id = -1
            self.active_body_name = ""
            self.active_wrench[:] = 0.0
            self.active_wrench_source = ""
            self.large_push_fired = False
            self.next_wrench_time = float(sim_time) + self.interval
            self.interaction_events.append(
                {
                    "time_s": float(sim_time),
                    "key": "R",
                    "scenario": "manual_reset_default_standing",
                    "initial_pose": "DEFAULT",
                }
            )
            self._record_operator_event(
                sim_time,
                "R",
                "manual_reset_default_standing",
                {"initial_pose": "DEFAULT"},
            )
            print(
                "[Extreme Stand interactive] R -> reset to DEFAULT standing pose; "
                "joint/root velocities, active wrench and policy action history cleared.",
                flush=True,
            )
            pose_reset = True
        if self.pending_space_cycle:
            self.pending_space_cycle = False
            scenario_index = self.space_cycle_index % 7
            self.space_cycle_index += 1
            if scenario_index < len(LARGE_PUSH_DIRECTIONS):
                direction_name = LARGE_PUSH_DIRECTIONS[scenario_index][0]
                scenario = f"large_torso_push_{direction_name}"
                self._begin_space_trial(
                    sim_time,
                    scenario,
                    {
                        "body_name": self.large_push_body_name,
                        "direction": direction_name,
                        "force_n": self.large_push_force,
                        "duration_s": self.large_push_duration,
                    },
                )
                self._start_large_push(
                    sim_time,
                    scenario_index,
                    source="interactive_space",
                )
                self.interaction_events.append(
                    {
                        "time_s": float(sim_time),
                        "key": "SPACE",
                        "scenario": scenario,
                    }
                )
            else:
                randomize_pose = scenario_index == 4
                randomize_foot_spacing = scenario_index == 5
                self.random_pose_active = randomize_pose
                self.random_foot_spacing_active = randomize_foot_spacing
                self._apply_initial_pose(
                    model,
                    data,
                    randomized=randomize_pose,
                    randomize_foot_spacing=randomize_foot_spacing,
                    reset_time_s=sim_time,
                )
                self.active_until = -1.0
                self.active_body_id = -1
                self.active_body_name = ""
                self.active_wrench_source = ""
                self.next_wrench_time = float(sim_time) + self.interval
                state = (
                    "RANDOM_FULL_BODY"
                    if randomize_pose
                    else (
                        "RANDOM_FOOT_SPACING"
                        if randomize_foot_spacing
                        else "DEFAULT"
                    )
                )
                scenario = f"reset_{state.lower()}"
                self._begin_space_trial(
                    sim_time,
                    scenario,
                    {"initial_pose": state},
                )
                self.interaction_events.append(
                    {
                        "time_s": float(sim_time),
                        "key": "SPACE",
                        "scenario": scenario,
                        "initial_pose": state,
                    }
                )
                print(
                    f"[Extreme Stand interactive] SPACE -> initial pose {state}",
                    flush=True,
                )
                pose_reset = True
        if self.pending_foot_spacing_randomize:
            self.pending_foot_spacing_randomize = False
            self.random_pose_active = False
            self.random_foot_spacing_active = True
            self._apply_initial_pose(
                model,
                data,
                randomized=False,
                randomize_foot_spacing=True,
                reset_time_s=sim_time,
            )
            self.active_until = -1.0
            self.active_body_id = -1
            self.active_body_name = ""
            self.active_wrench_source = ""
            self.next_wrench_time = float(sim_time) + self.interval
            event = {
                "time_s": float(sim_time),
                "key": "K",
                "initial_pose": "RANDOM_FOOT_SPACING",
                "initial_foot_distance_m": self.initial_foot_planar_distance_m,
                "default_foot_distance_m": self.default_foot_planar_distance_m,
            }
            self.interaction_events.append(event)
            self._record_operator_event(
                sim_time,
                "K",
                "random_foot_spacing",
                event,
            )
            print(
                "[Extreme Stand interactive] K -> random foot spacing "
                f"{self.initial_foot_planar_distance_m:.3f}m "
                f"(default {self.default_foot_planar_distance_m:.3f}m)",
                flush=True,
            )
            pose_reset = True
        if self.pending_wrench_toggle:
            self.pending_wrench_toggle = False
            self.wrench_enabled = not self.wrench_enabled
            data.xfrc_applied[:] = 0.0
            self.active_until = -1.0
            self.active_body_id = -1
            self.active_body_name = ""
            self.active_wrench_source = ""
            self.next_wrench_time = (
                float(sim_time)
                if self.wrench_enabled
                else float(sim_time) + self.interval
            )
            state = "ON" if self.wrench_enabled else "OFF"
            self.interaction_events.append(
                {"time_s": float(sim_time), "key": "F", "external_wrench": state}
            )
            self._record_operator_event(
                sim_time,
                "F",
                f"random_external_wrench_{state.lower()}",
                {"external_wrench": state},
            )
            print(f"[Extreme Stand interactive] F -> random external wrench {state}", flush=True)
        if self.pending_camera_toggle:
            self.pending_camera_toggle = False
            follow_enabled = not bool(self.config.get("follow_camera_enable", False))
            self.config["follow_camera_enable"] = follow_enabled
            state = "FOLLOW" if follow_enabled else "FREE"
            event = {
                "time_s": float(sim_time),
                "key": "C",
                "camera": state,
            }
            self.interaction_events.append(event)
            self._record_operator_event(
                sim_time,
                "C",
                f"camera_{state.lower()}",
                {"camera": state},
            )
            suffix = " (mouse orbit/zoom preserved)" if follow_enabled else ""
            print(f"[Extreme Stand interactive] C -> camera {state}{suffix}.", flush=True)
        return pose_reset

    def _ground_reaction_wrenches_world(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        floor_geom_ids: set[int],
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Aggregate each foot's floor-contact wrench in world coordinates.

        The reported moment is shifted from each contact point to the associated
        foot-body origin, making left/right values directly comparable over time.
        """

        wrenches = {
            "left": np.zeros(6, dtype=np.float64),
            "right": np.zeros(6, dtype=np.float64),
        }
        contact_counts = {"left": 0, "right": 0}
        contact_wrench = np.zeros(6, dtype=np.float64)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in floor_geom_ids:
                foot_geom = geom2
                # mj_contactForce is the wrench exerted on geom2 by geom1.
                sign_on_foot = 1.0
            elif geom2 in floor_geom_ids:
                foot_geom = geom1
                sign_on_foot = -1.0
            else:
                continue
            foot_body_id = int(model.geom_bodyid[foot_geom])
            side = self.foot_side_by_body_id.get(foot_body_id)
            if side is None:
                continue
            mujoco.mj_contactForce(model, data, contact_id, contact_wrench)
            # Contact frame axes are stored as rows; transpose maps contact-frame
            # force/torque components into the world frame.
            contact_to_world = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3).T
            force_world = sign_on_foot * (contact_to_world @ contact_wrench[:3])
            torque_at_contact_world = sign_on_foot * (
                contact_to_world @ contact_wrench[3:]
            )
            lever_world = (
                np.asarray(contact.pos, dtype=np.float64)
                - np.asarray(data.xpos[foot_body_id], dtype=np.float64)
            )
            moment_about_foot_world = torque_at_contact_world + np.cross(
                lever_world, force_world
            )
            wrenches[side][:3] += force_world
            wrenches[side][3:] += moment_about_foot_world
            contact_counts[side] += 1
        return wrenches, contact_counts

    def _write_interactive_diagnostic_row(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        floor_geom_ids: set[int],
        sim_time: float,
        command: np.ndarray,
        target_command: np.ndarray,
        actor_action: np.ndarray,
        target_position: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        simulator_accelerations: np.ndarray,
        jerks: np.ndarray,
        pd_torque_command: np.ndarray,
        actuator_force: np.ndarray,
        actuator_torque_limits: np.ndarray,
    ) -> None:
        if self._interactive_writer is None:
            return
        ground_wrenches, ground_contact_counts = self._ground_reaction_wrenches_world(
            model, data, floor_geom_ids
        )
        external_body_id = -1
        external_body_name = ""
        external_source = ""
        external_wrench = np.zeros(6, dtype=np.float64)
        nonzero_external_bodies = np.flatnonzero(
            np.linalg.norm(np.asarray(data.xfrc_applied, dtype=np.float64), axis=1) > 1.0e-12
        )
        if nonzero_external_bodies.size:
            external_body_id = int(nonzero_external_bodies[0])
            external_body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, external_body_id)
                or ""
            )
            external_wrench = np.asarray(
                data.xfrc_applied[external_body_id], dtype=np.float64
            ).copy()
            external_source = self.active_wrench_source
        root_qpos = np.asarray(data.qpos[:7], dtype=np.float64)
        root_qvel = np.asarray(data.qvel[:6], dtype=np.float64)
        foot_positions = {
            "left": np.full(3, math.nan, dtype=np.float64),
            "right": np.full(3, math.nan, dtype=np.float64),
        }
        for foot_body_id, side in self.foot_side_by_body_id.items():
            foot_positions[side] = np.asarray(
                data.xpos[foot_body_id], dtype=np.float64
            ).copy()
        foot_delta = foot_positions["right"] - foot_positions["left"]
        foot_planar_distance = float(np.linalg.norm(foot_delta[:2]))
        foot_distance_3d = float(np.linalg.norm(foot_delta))
        joint_anchors_world = np.asarray(
            [data.xanchor[self.joint_ids[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        row = [
            float(sim_time),
            self.current_space_trial_id,
            float(sim_time - self.current_space_trial_start_s),
            self.current_space_trial_scenario,
            self.current_reset_mode,
            self.last_operator_event,
            self.last_operator_event_time_s,
            *[float(value) for value in command],
            *[float(value) for value in target_command],
            int(self.target_limiter_enabled),
            self.latest_target_velocity_clipped_count,
            self.latest_target_acceleration_clipped_count,
            *[float(value) for value in root_qpos],
            *[float(value) for value in root_qvel],
            *[float(value) for value in foot_positions["left"]],
            *[float(value) for value in foot_positions["right"]],
            *[float(value) for value in foot_delta],
            foot_planar_distance,
            foot_distance_3d,
            self.default_foot_planar_distance_m,
            foot_planar_distance - self.default_foot_planar_distance_m,
            external_body_id,
            external_body_name,
            external_source,
            *[float(value) for value in external_wrench],
        ]
        for side in ("left", "right"):
            row.extend(
                [
                    ground_contact_counts[side],
                    *[float(value) for value in ground_wrenches[side]],
                ]
            )
        row.extend(float(value) for value in joint_anchors_world.reshape(-1))
        for values in (
            actor_action,
            self.latest_raw_target_position,
            target_position,
            self.latest_target_velocity,
            self.latest_target_acceleration,
            positions,
            velocities,
            accelerations,
            simulator_accelerations,
            jerks,
            pd_torque_command,
            actuator_force,
            actuator_torque_limits,
        ):
            row.extend(float(value) for value in values)
        if len(row) != len(self._interactive_headers):
            raise RuntimeError(
                "Extreme Stand interactive log schema mismatch: "
                f"row={len(row)} headers={len(self._interactive_headers)}"
            )
        self._interactive_writer.writerow(row)
        self._interactive_stream.flush()
        if self._interactive_trial_writer is not None:
            self._interactive_trial_writer.writerow(row)
            self._interactive_trial_stream.flush()
        self._interactive_row_count += 1

    def _close_interactive_data_logging(self) -> dict[str, object]:
        for stream in (self._interactive_trial_stream, self._interactive_stream):
            if stream is not None:
                stream.flush()
                stream.close()
        if self._interactive_events_stream is not None:
            self._interactive_events_stream.flush()
            self._interactive_events_stream.close()
        self._interactive_trial_stream = None
        self._interactive_trial_writer = None
        self._interactive_stream = None
        self._interactive_writer = None
        self._interactive_events_stream = None
        self._interactive_events_writer = None
        return {
            "enabled": self.interactive_log_enabled,
            "sample_rate_hz": 50.0,
            "row_count": self._interactive_row_count,
            "space_trial_count": self.current_space_trial_id,
            "master_csv": (
                str(self.interactive_log_path.resolve())
                if self.interactive_log_enabled
                else ""
            ),
            "events_csv": (
                str(self.interactive_events_path.resolve())
                if self.interactive_log_enabled
                else ""
            ),
            "space_trial_directory": (
                str(self.interactive_trials_dir.resolve())
                if self.interactive_log_enabled
                else ""
            ),
            "space_trial_csv_files": list(self._interactive_trial_paths),
        }

    def record_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        floor_geom_ids: set[int],
        sim_time: float,
        command: np.ndarray,
        target_command: np.ndarray,
        actor_action: np.ndarray,
        target_position: np.ndarray,
        pd_torque_command: np.ndarray,
        actuator_force: np.ndarray,
        actuator_torque_limits: np.ndarray,
    ) -> None:
        """Record 50 Hz state, actor and actuator diagnostics without changing state."""

        if not self.qpos_addresses:
            raise RuntimeError("Extreme Stand state recorder used before initialization.")
        positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [data.qvel[self.qvel_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        simulator_accelerations = np.asarray(
            [data.qacc[self.qvel_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        acceleration = np.full_like(velocities, math.nan)
        jerk = np.full_like(velocities, math.nan)
        if self._previous_joint_velocity is not None and self._previous_motion_time is not None:
            sample_dt = float(sim_time) - self._previous_motion_time
            if sample_dt > 0.0:
                acceleration = (velocities - self._previous_joint_velocity) / sample_dt
                if self._previous_joint_acceleration is not None:
                    jerk = (acceleration - self._previous_joint_acceleration) / sample_dt
                self._previous_joint_acceleration = acceleration.copy()
        self._previous_joint_velocity = velocities.copy()
        self._previous_motion_time = float(sim_time)

        foot_positions = np.asarray(data.xpos[self.foot_body_ids, :2], dtype=np.float64)
        foot_distance = float(np.linalg.norm(foot_positions[0] - foot_positions[1]))
        self.motion_times.append(float(sim_time))
        self.joint_position_samples.append(positions)
        self.joint_velocity_samples.append(velocities)
        self.joint_acceleration_samples.append(acceleration)
        self.joint_jerk_samples.append(jerk)
        self.actor_action_samples.append(
            np.asarray(actor_action, dtype=np.float64).copy()
        )
        self.raw_target_position_samples.append(self.latest_raw_target_position.copy())
        self.target_position_samples.append(
            np.asarray(target_position, dtype=np.float64).copy()
        )
        self.target_limiter_velocity_samples.append(self.latest_target_velocity.copy())
        self.target_limiter_acceleration_samples.append(
            self.latest_target_acceleration.copy()
        )
        self.pd_torque_command_samples.append(
            np.asarray(pd_torque_command, dtype=np.float64).copy()
        )
        self.actuator_force_samples.append(
            np.asarray(actuator_force, dtype=np.float64).copy()
        )
        self.actuator_torque_limit_samples.append(
            np.asarray(actuator_torque_limits, dtype=np.float64).copy()
        )
        self.foot_planar_distance_samples.append(foot_distance)
        self.joint_error_times.append(float(sim_time))
        self.joint_abs_error_samples.append(
            np.abs(positions - self.default_joint_positions)
        )
        self._write_interactive_diagnostic_row(
            model,
            data,
            floor_geom_ids,
            sim_time,
            command,
            target_command,
            actor_action,
            target_position,
            positions,
            velocities,
            acceleration,
            simulator_accelerations,
            jerk,
            pd_torque_command,
            actuator_force,
            actuator_torque_limits,
        )

    def _motion_quality_summary(self) -> dict:
        if not self.motion_times:
            return {
                "sample_count": 0,
                "steady_sample_count": 0,
                "steady_start_s": self.steady_start_time,
            }

        times = np.asarray(self.motion_times, dtype=np.float64)
        elapsed_times = times - self.last_reset_time_s
        positions = np.stack(self.joint_position_samples, axis=0)
        velocities = np.stack(self.joint_velocity_samples, axis=0)
        accelerations = np.stack(self.joint_acceleration_samples, axis=0)
        jerks = np.stack(self.joint_jerk_samples, axis=0)
        actor_actions = np.stack(self.actor_action_samples, axis=0)
        targets = np.stack(self.target_position_samples, axis=0)
        pd_torques = np.stack(self.pd_torque_command_samples, axis=0)
        actuator_forces = np.stack(self.actuator_force_samples, axis=0)
        torque_limits = np.stack(self.actuator_torque_limit_samples, axis=0)
        foot_distances = np.asarray(self.foot_planar_distance_samples, dtype=np.float64)
        sample_dt = (
            float(np.median(np.diff(times)))
            if times.size >= 2
            else float(self.config.get("simulation_dt", 0.002))
            * int(self.config.get("control_decimation", 10))
        )
        steady_mask = elapsed_times >= self.steady_start_time
        steady_positions = positions[steady_mask]
        steady_velocities = velocities[steady_mask]
        steady_accelerations = accelerations[steady_mask]
        steady_jerks = jerks[steady_mask]
        steady_actions = actor_actions[steady_mask]
        steady_targets = targets[steady_mask]
        steady_pd_torques = pd_torques[steady_mask]
        steady_actuator_forces = actuator_forces[steady_mask]
        steady_torque_limits = torque_limits[steady_mask]
        steady_foot_distances = foot_distances[steady_mask]
        finite_jerk_rows = np.all(np.isfinite(steady_jerks), axis=1)
        steady_jerks = steady_jerks[finite_jerk_rows]

        if steady_jerks.size:
            jerk_stats = _finite_stats(steady_jerks)
            jerk_mean_square = float(np.mean(np.square(steady_jerks)))
            jerk_per_joint_rms = np.sqrt(np.mean(np.square(steady_jerks), axis=0))
        else:
            jerk_stats = _finite_stats(np.asarray([], dtype=np.float64))
            jerk_mean_square = math.nan
            jerk_per_joint_rms = np.full(len(self.policy_joint_names), math.nan)

        high_frequency_rms = _position_band_rms(
            steady_positions,
            sample_dt,
            low_hz=20.0,
            high_hz=25.0,
        )
        broad_high_frequency_rms = _position_band_rms(
            steady_positions,
            sample_dt,
            low_hz=8.0,
            high_hz=25.0,
        )
        action_high_frequency_rms = _position_band_rms(
            steady_actions,
            sample_dt,
            low_hz=8.0,
            high_hz=25.0,
        )
        target_high_frequency_rms = _position_band_rms(
            steady_targets,
            sample_dt,
            low_hz=8.0,
            high_hz=25.0,
        )
        torque_high_frequency_rms = _position_band_rms(
            steady_pd_torques,
            sample_dt,
            low_hz=8.0,
            high_hz=25.0,
        )
        action_rates = (
            np.diff(steady_actions, axis=0) / sample_dt
            if steady_actions.shape[0] >= 2 and sample_dt > 0.0
            else np.empty((0, len(self.policy_joint_names)), dtype=np.float64)
        )
        finite_accelerations = steady_accelerations[
            np.all(np.isfinite(steady_accelerations), axis=1)
        ]
        finite_limits = np.logical_and(
            np.isfinite(steady_torque_limits),
            steady_torque_limits > 1.0e-9,
        )
        torque_saturation_fraction = (
            float(
                np.mean(
                    np.abs(steady_pd_torques[finite_limits])
                    >= 0.98 * steady_torque_limits[finite_limits]
                )
            )
            if np.any(finite_limits)
            else math.nan
        )
        foot_errors = steady_foot_distances - self.default_foot_planar_distance_m
        foot_error_stats = _finite_stats(foot_errors)
        foot_distance_stats = _finite_stats(steady_foot_distances)
        if foot_errors.size:
            gaussian_values = np.exp(
                -0.5 * np.square(foot_errors) / self.feet_gaussian_variance
            )
            within_1cm = float(np.mean(np.abs(foot_errors) <= 0.01))
            within_2cm = float(np.mean(np.abs(foot_errors) <= 0.02))
            gaussian_mean = float(np.mean(gaussian_values))
            mean_square_error = float(np.mean(np.square(foot_errors)))
        else:
            within_1cm = math.nan
            within_2cm = math.nan
            gaussian_mean = math.nan
            mean_square_error = math.nan

        return {
            "sample_count": int(times.size),
            "steady_sample_count": int(np.count_nonzero(steady_mask)),
            "steady_start_s": self.steady_start_time,
            "reset_time_s": self.last_reset_time_s,
            "reset_mode": self.current_reset_mode,
            "control_sample_dt_s": sample_dt,
            "control_sample_rate_hz": float(1.0 / sample_dt) if sample_dt > 0.0 else math.nan,
            "joint_jerk_rad_s3": {
                **jerk_stats,
                "mean_square": jerk_mean_square,
                "training_reward_weight": self.jerk_reward_weight,
                "training_weighted_mean_reward_equivalent": (
                    self.jerk_reward_weight * jerk_mean_square
                    if math.isfinite(jerk_mean_square)
                    else math.nan
                ),
                "per_joint_rms": {
                    name: float(value)
                    for name, value in zip(self.policy_joint_names, jerk_per_joint_rms)
                },
            },
            "joint_position_high_frequency_20_25hz_rms_rad": {
                "mean_across_joints": (
                    float(np.nanmean(high_frequency_rms))
                    if np.any(np.isfinite(high_frequency_rms))
                    else math.nan
                ),
                "max_across_joints": (
                    float(np.nanmax(high_frequency_rms))
                    if np.any(np.isfinite(high_frequency_rms))
                    else math.nan
                ),
                "per_joint": {
                    name: float(value)
                    for name, value in zip(self.policy_joint_names, high_frequency_rms)
                },
            },
            "joint_position_high_frequency_8_25hz_rms_rad": {
                "mean_across_joints": (
                    float(np.nanmean(broad_high_frequency_rms))
                    if np.any(np.isfinite(broad_high_frequency_rms))
                    else math.nan
                ),
                "max_across_joints": (
                    float(np.nanmax(broad_high_frequency_rms))
                    if np.any(np.isfinite(broad_high_frequency_rms))
                    else math.nan
                ),
                "per_joint": {
                    name: float(value)
                    for name, value in zip(
                        self.policy_joint_names,
                        broad_high_frequency_rms,
                    )
                },
            },
            "actor_action": {
                "value": _finite_stats(steady_actions),
                "delta_rate_per_s": _finite_stats(action_rates),
                "high_frequency_8_25hz_rms": {
                    "mean_across_joints": (
                        float(np.nanmean(action_high_frequency_rms))
                        if np.any(np.isfinite(action_high_frequency_rms))
                        else math.nan
                    ),
                    "max_across_joints": (
                        float(np.nanmax(action_high_frequency_rms))
                        if np.any(np.isfinite(action_high_frequency_rms))
                        else math.nan
                    ),
                    "per_joint": {
                        name: float(value)
                        for name, value in zip(
                            self.policy_joint_names,
                            action_high_frequency_rms,
                        )
                    },
                },
            },
            "target_joint_position_rad": {
                "high_frequency_8_25hz_rms": {
                    "mean_across_joints": (
                        float(np.nanmean(target_high_frequency_rms))
                        if np.any(np.isfinite(target_high_frequency_rms))
                        else math.nan
                    ),
                    "max_across_joints": (
                        float(np.nanmax(target_high_frequency_rms))
                        if np.any(np.isfinite(target_high_frequency_rms))
                        else math.nan
                    ),
                },
            },
            "joint_velocity_rad_s": _finite_stats(steady_velocities),
            "joint_acceleration_rad_s2": _finite_stats(finite_accelerations),
            "actuator_effort_nm": {
                "pd_command": _finite_stats(steady_pd_torques),
                "actual": _finite_stats(steady_actuator_forces),
                "pd_command_high_frequency_8_25hz_rms": {
                    "mean_across_joints": (
                        float(np.nanmean(torque_high_frequency_rms))
                        if np.any(np.isfinite(torque_high_frequency_rms))
                        else math.nan
                    ),
                    "max_across_joints": (
                        float(np.nanmax(torque_high_frequency_rms))
                        if np.any(np.isfinite(torque_high_frequency_rms))
                        else math.nan
                    ),
                    "per_joint": {
                        name: float(value)
                        for name, value in zip(
                            self.policy_joint_names,
                            torque_high_frequency_rms,
                        )
                    },
                },
                "command_saturation_fraction": torque_saturation_fraction,
            },
            "feet_planar_distance_m": {
                "default": self.default_foot_planar_distance_m,
                "mean": foot_distance_stats["mean"],
                "std": foot_distance_stats["std"],
                "error_mean": foot_error_stats["mean"],
                "error_mean_abs": foot_error_stats["mean_abs"],
                "error_rms": foot_error_stats["rms"],
                "error_p95_abs": foot_error_stats["p95_abs"],
                "error_max_abs": foot_error_stats["max_abs"],
                "mean_square_error": mean_square_error,
                "gaussian_variance_m2": self.feet_gaussian_variance,
                "gaussian_mean": gaussian_mean,
                "within_1cm_fraction": within_1cm,
                "within_2cm_fraction": within_2cm,
            },
            "trace_csv_path": "",
        }

    def _large_push_diagnostics(self) -> dict[str, object]:
        large_push_events = [
            event
            for event in self.wrench_events
            if str(event.get("source", "")).startswith(("scheduled", "interactive_space"))
        ]
        if not large_push_events or len(self.motion_times) < 8:
            return {
                "tested": False,
                "event_count": len(large_push_events),
                "diagnosis": "no_large_push_samples",
            }

        times = np.asarray(self.motion_times, dtype=np.float64)
        positions = np.stack(self.joint_position_samples, axis=0)
        jerks = np.stack(self.joint_jerk_samples, axis=0)
        actor_actions = np.stack(self.actor_action_samples, axis=0)
        pd_torques = np.stack(self.pd_torque_command_samples, axis=0)
        torque_limits = np.stack(self.actuator_torque_limit_samples, axis=0)
        sample_dt = float(np.median(np.diff(times)))
        current_events = [
            event
            for event in large_push_events
            if float(event["time_s"]) >= self.last_reset_time_s - 1.0e-9
            and float(event["time_s"]) <= float(times[-1]) + 1.0e-9
        ]
        if not current_events:
            return {
                "tested": False,
                "event_count": len(large_push_events),
                "diagnosis": "large_push_precedes_latest_pose_reset",
                "events": large_push_events,
            }

        event = current_events[-1]
        event_start = float(event["time_s"])
        event_end = float(event.get("end_time_s", event_start))
        pre_start = max(self.last_reset_time_s, event_start - 2.0)
        post_start = event_end + self.post_push_settle_time
        late_start = max(post_start, float(times[-1]) - 5.0)
        pre_mask = np.logical_and(times >= pre_start, times < event_start)
        post_mask = times >= post_start
        late_mask = times >= late_start

        def window_metrics(mask: np.ndarray) -> dict[str, object]:
            sample_count = int(np.count_nonzero(mask))
            if sample_count < 8:
                return {"sample_count": sample_count}
            window_positions = positions[mask]
            window_jerks = jerks[mask]
            window_actions = actor_actions[mask]
            window_torques = pd_torques[mask]
            window_limits = torque_limits[mask]
            finite_jerk_rows = np.all(np.isfinite(window_jerks), axis=1)
            finite_limits = np.logical_and(
                np.isfinite(window_limits),
                window_limits > 1.0e-9,
            )
            action_rates = (
                np.diff(window_actions, axis=0) / sample_dt
                if window_actions.shape[0] >= 2 and sample_dt > 0.0
                else np.empty((0, len(self.policy_joint_names)), dtype=np.float64)
            )
            position_hf = _position_band_rms(
                window_positions,
                sample_dt,
                low_hz=8.0,
                high_hz=25.0,
            )
            action_hf = _position_band_rms(
                window_actions,
                sample_dt,
                low_hz=8.0,
                high_hz=25.0,
            )
            torque_hf = _position_band_rms(
                window_torques,
                sample_dt,
                low_hz=8.0,
                high_hz=25.0,
            )
            saturation_fraction = (
                float(
                    np.mean(
                        np.abs(window_torques[finite_limits])
                        >= 0.98 * window_limits[finite_limits]
                    )
                )
                if np.any(finite_limits)
                else math.nan
            )
            return {
                "sample_count": sample_count,
                "start_time_s": float(times[mask][0]),
                "end_time_s": float(times[mask][-1]),
                "joint_position_hf_8_25hz_rms_mean_rad": float(np.nanmean(position_hf)),
                "joint_position_hf_8_25hz_rms_max_rad": float(np.nanmax(position_hf)),
                "actor_action_hf_8_25hz_rms_mean": float(np.nanmean(action_hf)),
                "actor_action_hf_8_25hz_rms_max": float(np.nanmax(action_hf)),
                "actor_action_delta_rate_rms_per_s": _finite_stats(action_rates)["rms"],
                "joint_jerk_rms_rad_s3": _finite_stats(
                    window_jerks[finite_jerk_rows]
                )["rms"],
                "pd_torque_rms_nm": _finite_stats(window_torques)["rms"],
                "pd_torque_p95_abs_nm": _finite_stats(window_torques)["p95_abs"],
                "pd_torque_max_abs_nm": _finite_stats(window_torques)["max_abs"],
                "pd_torque_hf_8_25hz_rms_mean_nm": float(np.nanmean(torque_hf)),
                "pd_torque_hf_8_25hz_rms_max_nm": float(np.nanmax(torque_hf)),
                "pd_torque_saturation_fraction": saturation_fraction,
                "highest_position_hf_joints": [
                    {
                        "joint": self.policy_joint_names[index],
                        "rms_rad": float(position_hf[index]),
                    }
                    for index in np.argsort(np.nan_to_num(position_hf, nan=-1.0))[-5:][::-1]
                ],
                "highest_action_hf_joints": [
                    {
                        "joint": self.policy_joint_names[index],
                        "rms": float(action_hf[index]),
                    }
                    for index in np.argsort(np.nan_to_num(action_hf, nan=-1.0))[-5:][::-1]
                ],
                "highest_torque_hf_joints": [
                    {
                        "joint": self.policy_joint_names[index],
                        "rms_nm": float(torque_hf[index]),
                    }
                    for index in np.argsort(np.nan_to_num(torque_hf, nan=-1.0))[-5:][::-1]
                ],
            }

        pre = window_metrics(pre_mask)
        post = window_metrics(post_mask)
        late = window_metrics(late_mask)

        def ratio(window: dict[str, object], key: str) -> float:
            pre_value = float(pre.get(key, math.nan))
            post_value = float(window.get(key, math.nan))
            if not math.isfinite(pre_value) or not math.isfinite(post_value):
                return math.nan
            return post_value / max(abs(pre_value), 1.0e-9)

        position_hf_ratio = ratio(post, "joint_position_hf_8_25hz_rms_mean_rad")
        action_hf_ratio = ratio(post, "actor_action_hf_8_25hz_rms_mean")
        action_rate_ratio = ratio(post, "actor_action_delta_rate_rms_per_s")
        torque_hf_ratio = ratio(post, "pd_torque_hf_8_25hz_rms_mean_nm")
        late_position_hf_ratio = ratio(
            late,
            "joint_position_hf_8_25hz_rms_mean_rad",
        )
        late_action_hf_ratio = ratio(late, "actor_action_hf_8_25hz_rms_mean")
        late_action_rate_ratio = ratio(late, "actor_action_delta_rate_rms_per_s")
        late_torque_hf_ratio = ratio(
            late,
            "pd_torque_hf_8_25hz_rms_mean_nm",
        )
        post_position_hf = float(
            post.get("joint_position_hf_8_25hz_rms_mean_rad", math.nan)
        )
        post_action_hf = float(post.get("actor_action_hf_8_25hz_rms_mean", math.nan))
        late_position_hf = float(
            late.get("joint_position_hf_8_25hz_rms_mean_rad", math.nan)
        )
        late_action_hf = float(
            late.get("actor_action_hf_8_25hz_rms_mean", math.nan)
        )
        late_saturation = float(
            late.get("pd_torque_saturation_fraction", math.nan)
        )
        transient_joint_vibration = bool(
            math.isfinite(position_hf_ratio)
            and position_hf_ratio >= 3.0
            and math.isfinite(post_position_hf)
            and post_position_hf >= 0.002
        )
        transient_policy_action_high_frequency = bool(
            math.isfinite(action_hf_ratio)
            and action_hf_ratio >= 3.0
            and math.isfinite(post_action_hf)
            and post_action_hf >= 0.01
        )
        persistent_joint_vibration = bool(
            math.isfinite(late_position_hf_ratio)
            and late_position_hf_ratio >= 3.0
            and math.isfinite(late_position_hf)
            and late_position_hf >= 0.002
        )
        policy_action_high_frequency = bool(
            math.isfinite(late_action_hf_ratio)
            and late_action_hf_ratio >= 3.0
            and math.isfinite(late_action_hf)
            and late_action_hf >= 0.01
        )
        pd_torque_saturation = bool(
            math.isfinite(late_saturation) and late_saturation >= 0.02
        )
        pre_action_rate = float(
            pre.get("actor_action_delta_rate_rms_per_s", math.nan)
        )
        action_rate_settle_threshold = (
            max(5.0 * pre_action_rate, 0.20)
            if math.isfinite(pre_action_rate)
            else 0.20
        )
        full_action_rates = np.vstack(
            [
                np.full((1, actor_actions.shape[1]), np.nan),
                np.diff(actor_actions, axis=0) / sample_dt,
            ]
        )
        settling_time = None
        hold_samples = max(1, int(round(2.0 / sample_dt)))
        first_post_index = int(np.searchsorted(times, event_end, side="left"))
        for index in range(first_post_index, len(times) - hold_samples + 1):
            window = full_action_rates[index : index + hold_samples]
            window_rms = _finite_stats(window)["rms"]
            if (
                math.isfinite(window_rms)
                and window_rms <= action_rate_settle_threshold
            ):
                settling_time = float(times[index] - event_end)
                break
        if pd_torque_saturation:
            diagnosis = "persistent_pd_torque_saturation_after_push"
        elif persistent_joint_vibration and policy_action_high_frequency:
            diagnosis = "persistent_policy_action_driven_high_frequency_oscillation"
        elif persistent_joint_vibration:
            diagnosis = "persistent_closed_loop_state_or_contact_oscillation"
        elif (
            transient_joint_vibration
            and transient_policy_action_high_frequency
        ):
            diagnosis = "policy_action_driven_transient_oscillation_that_settles"
        else:
            diagnosis = "transient_recovery_without_detected_persistent_jitter"
        return {
            "tested": True,
            "body_name": self.large_push_body_name,
            "configured_force_n": self.large_push_force,
            "configured_duration_s": self.large_push_duration,
            "configured_impulse_n_s": self.large_push_force * self.large_push_duration,
            "post_push_settle_s": self.post_push_settle_time,
            "event_count": len(large_push_events),
            "analyzed_event": event,
            "pre_push": pre,
            "post_push": post,
            "late_post_push": late,
            "post_over_pre": {
                "joint_position_hf_8_25hz_rms_ratio": position_hf_ratio,
                "actor_action_hf_8_25hz_rms_ratio": action_hf_ratio,
                "actor_action_delta_rate_rms_ratio": action_rate_ratio,
                "pd_torque_hf_8_25hz_rms_ratio": torque_hf_ratio,
            },
            "late_post_over_pre": {
                "joint_position_hf_8_25hz_rms_ratio": late_position_hf_ratio,
                "actor_action_hf_8_25hz_rms_ratio": late_action_hf_ratio,
                "actor_action_delta_rate_rms_ratio": late_action_rate_ratio,
                "pd_torque_hf_8_25hz_rms_ratio": late_torque_hf_ratio,
            },
            "flags": {
                "transient_joint_vibration": transient_joint_vibration,
                "transient_policy_action_high_frequency": (
                    transient_policy_action_high_frequency
                ),
                "persistent_joint_vibration": persistent_joint_vibration,
                "policy_action_high_frequency": policy_action_high_frequency,
                "pd_torque_saturation": pd_torque_saturation,
            },
            "actor_action_rate_settling": {
                "threshold_rms_per_s": action_rate_settle_threshold,
                "hold_time_s": 2.0,
                "settling_time_after_push_end_s": settling_time,
            },
            "diagnosis": diagnosis,
            "events": large_push_events,
        }

    def _foot_spacing_recovery_summary(self) -> dict[str, object]:
        if not self.motion_times:
            return {
                "tested": self.current_reset_mode == "random_foot_spacing",
                "distance_recovered": False,
                "sample_count": 0,
            }

        times = np.asarray(self.motion_times, dtype=np.float64)
        elapsed_times = times - self.last_reset_time_s
        distances = np.asarray(self.foot_planar_distance_samples, dtype=np.float64)
        errors = distances - self.default_foot_planar_distance_m
        final_start = max(float(elapsed_times[-1]) - self.recovery_final_window, 0.0)
        final_errors = errors[elapsed_times >= final_start]
        final_stats = _finite_stats(final_errors)
        recovery_time = None
        for index, start_time in enumerate(elapsed_times):
            end_index = int(
                np.searchsorted(
                    elapsed_times,
                    start_time + self.recovery_hold_time,
                    side="left",
                )
            )
            if end_index >= len(elapsed_times):
                break
            if np.all(
                np.abs(errors[index : end_index + 1])
                <= self.foot_spacing_recovery_tolerance
            ):
                recovery_time = float(start_time)
                break
        tested = self.current_reset_mode == "random_foot_spacing"
        perturbation_applied = bool(
            tested
            and math.isfinite(self.initial_foot_planar_distance_m)
            and abs(
                self.initial_foot_planar_distance_m
                - self.default_foot_planar_distance_m
            )
            >= self.foot_spacing_min_delta - 1.0e-6
        )
        distance_recovered = bool(
            tested
            and perturbation_applied
            and final_errors.size
            and np.all(
                np.abs(final_errors) <= self.foot_spacing_recovery_tolerance
            )
        )
        return {
            "tested": tested,
            "perturbation_applied": perturbation_applied,
            "default_distance_m": self.default_foot_planar_distance_m,
            "target_initial_distance_m": self.foot_spacing_target_distance_m,
            "actual_initial_distance_m": self.initial_foot_planar_distance_m,
            "initial_error_m": (
                self.initial_foot_planar_distance_m
                - self.default_foot_planar_distance_m
            ),
            "symmetric_hip_roll_offset_rad": self.foot_spacing_roll_offset_rad,
            "attainable_distance_range_m": list(self.foot_spacing_attainable_range_m),
            "recovery_tolerance_m": self.foot_spacing_recovery_tolerance,
            "hold_time_s": self.recovery_hold_time,
            "final_window_s": self.recovery_final_window,
            "final_error_mean_m": final_stats["mean"],
            "final_error_mean_abs_m": final_stats["mean_abs"],
            "final_error_max_abs_m": final_stats["max_abs"],
            "minimum_error_abs_m": float(np.min(np.abs(errors))),
            "recovery_time_s": recovery_time,
            "distance_recovered": distance_recovered,
            "sample_count": int(times.size),
            "reset_time_s": self.last_reset_time_s,
            "events": list(self.foot_spacing_reset_events),
        }

    def write_motion_quality_trace_csv(self, output_path: str | Path) -> str:
        if not str(output_path) or not self.motion_times:
            return ""
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "time_s",
            "elapsed_since_reset_s",
            "reset_mode",
            "foot_planar_distance_m",
            "foot_distance_error_m",
        ]
        for prefix in (
            "qpos_rad",
            "qvel_rad_s",
            "qacc_fd_rad_s2",
            "jerk_fd_rad_s3",
            "actor_action",
            "raw_target_qpos_rad",
            "target_qpos_rad",
            "target_limiter_velocity_rad_s",
            "target_limiter_acceleration_rad_s2",
            "pd_torque_command_nm",
            "actuator_force_nm",
            "actuator_torque_limit_nm",
        ):
            headers.extend(f"{prefix}/{name}" for name in self.policy_joint_names)
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for (
                time_s,
                foot_distance,
                positions,
                velocities,
                accelerations,
                jerks,
                actor_actions,
                raw_target_positions,
                target_positions,
                target_limiter_velocities,
                target_limiter_accelerations,
                pd_torques,
                actuator_forces,
                torque_limits,
            ) in zip(
                self.motion_times,
                self.foot_planar_distance_samples,
                self.joint_position_samples,
                self.joint_velocity_samples,
                self.joint_acceleration_samples,
                self.joint_jerk_samples,
                self.actor_action_samples,
                self.raw_target_position_samples,
                self.target_position_samples,
                self.target_limiter_velocity_samples,
                self.target_limiter_acceleration_samples,
                self.pd_torque_command_samples,
                self.actuator_force_samples,
                self.actuator_torque_limit_samples,
            ):
                writer.writerow(
                    [
                        float(time_s),
                        float(time_s - self.last_reset_time_s),
                        self.current_reset_mode,
                        float(foot_distance),
                        float(foot_distance - self.default_foot_planar_distance_m),
                        *[float(value) for value in positions],
                        *[float(value) for value in velocities],
                        *[float(value) for value in accelerations],
                        *[float(value) for value in jerks],
                        *[float(value) for value in actor_actions],
                        *[float(value) for value in raw_target_positions],
                        *[float(value) for value in target_positions],
                        *[float(value) for value in target_limiter_velocities],
                        *[float(value) for value in target_limiter_accelerations],
                        *[float(value) for value in pd_torques],
                        *[float(value) for value in actuator_forces],
                        *[float(value) for value in torque_limits],
                    ]
                )
        return str(output_path.resolve())

    def update_external_wrench(self, data: mujoco.MjData, sim_time: float) -> None:
        """Apply a short random wrench to one random body at a time."""

        data.xfrc_applied[:] = 0.0
        if (
            self.large_push_enabled
            and not self.large_push_fired
            and sim_time >= self.large_push_time
        ):
            direction_index = self.large_push_direction_index
            if direction_index < 0:
                direction_index = int(self.rng.integers(0, len(LARGE_PUSH_DIRECTIONS)))
            self._start_large_push(
                sim_time,
                direction_index,
                source="scheduled_large_push",
            )
            self.large_push_fired = True
        if (
            sim_time < self.active_until
            and self.active_body_id >= 0
            and self.active_wrench_source in ("scheduled_large_push", "interactive_space")
        ):
            data.xfrc_applied[self.active_body_id] = self.active_wrench
            return
        if not self.wrench_enabled or (self.force_max <= 0.0 and self.torque_max <= 0.0):
            return
        if sim_time >= self.next_wrench_time:
            index = int(self.rng.integers(0, len(self.body_ids)))
            self.active_body_id = self.body_ids[index]
            self.active_body_name = self.body_names[index]
            self.active_wrench[:3] = self.rng.uniform(-self.force_max, self.force_max, size=3)
            self.active_wrench[3:] = self.rng.uniform(-self.torque_max, self.torque_max, size=3)
            self.active_until = sim_time + self.duration
            self.active_wrench_source = "periodic_random"
            self.next_wrench_time += self.interval
            self.event_count += 1
            event = {
                "event_index": self.event_count,
                "time_s": float(sim_time),
                "end_time_s": float(self.active_until),
                "source": self.active_wrench_source,
                "body_id": self.active_body_id,
                "body_name": self.active_body_name,
                "force_world_n": [float(value) for value in self.active_wrench[:3]],
                "torque_world_nm": [float(value) for value in self.active_wrench[3:]],
                "duration_s": self.duration,
            }
            self.wrench_events.append(event)
            print(
                "[Extreme Stand wrench] "
                f"#{self.event_count} t={sim_time:.3f}s body={self.active_body_name} "
                "force_world_n=("
                + ", ".join(f"{value:.2f}" for value in self.active_wrench[:3])
                + ") torque_world_nm=("
                + ", ".join(f"{value:.2f}" for value in self.active_wrench[3:])
                + ")",
                flush=True,
            )
        if sim_time < self.active_until and self.active_body_id >= 0:
            data.xfrc_applied[self.active_body_id] = self.active_wrench

    def summary(self) -> dict:
        interactive_data_log = self._close_interactive_data_logging()
        body_event_counts = {
            body_name: sum(
                event["body_name"] == body_name for event in self.wrench_events
            )
            for body_name in self.body_names
        }
        initial_mae = float(np.mean(self.initial_joint_abs_errors))
        initial_max = float(np.max(self.initial_joint_abs_errors))
        if self.joint_abs_error_samples:
            errors = np.stack(self.joint_abs_error_samples, axis=0)
            times = np.asarray(self.joint_error_times, dtype=np.float64)
            joint_mae = np.mean(errors, axis=1)
            joint_max = np.max(errors, axis=1)
            final_start = max(float(times[-1]) - self.recovery_final_window, 0.0)
            final_mask = times >= final_start
            final_errors = errors[final_mask]
            final_mae = float(np.mean(final_errors))
            final_max = float(np.mean(np.max(final_errors, axis=1)))
            final_per_joint = np.mean(final_errors, axis=0)
            minimum_mae = float(np.min(joint_mae))
            recovery_time = None
            for index, start_time in enumerate(times):
                end_index = int(
                    np.searchsorted(times, start_time + self.recovery_hold_time, side="left")
                )
                if end_index >= len(times):
                    break
                if np.all(
                    joint_mae[index : end_index + 1]
                    <= self.recovery_joint_mae_threshold
                ) and np.all(
                    joint_max[index : end_index + 1]
                    <= self.recovery_joint_max_threshold
                ):
                    recovery_time = float(start_time - self.last_reset_time_s)
                    break
        else:
            final_mae = math.nan
            final_max = math.nan
            final_per_joint = np.full(len(self.policy_joint_names), math.nan)
            minimum_mae = math.nan
            recovery_time = None
        recovery_ratio = (
            float((initial_mae - final_mae) / initial_mae)
            if initial_mae > 1.0e-12 and math.isfinite(final_mae)
            else 0.0
        )
        return {
            "action_override": False,
            "seed": int(self.config.get("extreme_stand_recovery_seed", 20260719)),
            "initial_noise": {
                "leg_rad": self.leg_noise,
                "waist_rad": self.waist_noise,
                "arm_rad": self.arm_noise,
                "joint_velocity_rad_s": self.joint_velocity_noise,
                "root_roll_pitch_rad": self.root_roll_pitch_noise,
                "root_yaw_rad": self.root_yaw_noise,
                "root_linear_velocity_m_s": self.root_linear_velocity_noise,
                "root_angular_velocity_rad_s": self.root_angular_velocity_noise,
                "joint_limit_margin_rad": self.joint_limit_margin,
                "joint_limit_clip_count": self.initial_joint_limit_clip_count,
            },
            "foot_spacing_randomization": {
                "start_random": self.foot_spacing_start_random,
                "active_at_end": self.random_foot_spacing_active,
                "minimum_delta_m": self.foot_spacing_min_delta,
                "maximum_delta_m": self.foot_spacing_max_delta,
                "maximum_roll_offset_rad": self.foot_spacing_max_roll_offset,
                "search_samples": self.foot_spacing_search_samples,
                "reset_count": self.foot_spacing_reset_count,
                "events": list(self.foot_spacing_reset_events),
            },
            "wrench": {
                "force_max_n": self.force_max,
                "torque_max_nm": self.torque_max,
                "interval_s": self.interval,
                "duration_s": self.duration,
                "event_count": self.event_count,
                "body_names": list(self.body_names),
                "body_event_counts": body_event_counts,
                "events": list(self.wrench_events),
                "enabled_at_end": self.wrench_enabled,
            },
            "large_push": {
                "enabled": self.large_push_enabled,
                "body_name": self.large_push_body_name,
                "force_n": self.large_push_force,
                "duration_s": self.large_push_duration,
                "impulse_n_s": self.large_push_force * self.large_push_duration,
                "scheduled_time_s": self.large_push_time,
                "direction_index": self.large_push_direction_index,
                "event_count": self.large_push_event_count,
                "post_push_diagnostics": self._large_push_diagnostics(),
            },
            "interactive": {
                "enabled": self.interactive_enabled,
                "initial_pose_at_end": self.current_reset_mode,
                "wrench_enabled_at_end": self.wrench_enabled,
                "controls": {
                    "SPACE": (
                        "cycle_torso_push_forward_backward_left_right_"
                        "random_pose_random_foot_spacing_default"
                    ),
                    "K": "new_random_foot_spacing_initial_pose",
                    "F": "toggle_random_wrench",
                    "R": "manual_reset_default_standing",
                    "C": "toggle_free_follow_camera",
                },
                "events": list(self.interaction_events),
                "data_logging": interactive_data_log,
            },
            "default_pose_recovery": {
                "joint_count": len(self.policy_joint_names),
                "joint_mae_threshold_rad": self.recovery_joint_mae_threshold,
                "joint_max_threshold_rad": self.recovery_joint_max_threshold,
                "hold_time_s": self.recovery_hold_time,
                "final_window_s": self.recovery_final_window,
                "initial_joint_mae_rad": initial_mae,
                "initial_joint_max_abs_error_rad": initial_max,
                "minimum_joint_mae_rad": minimum_mae,
                "final_joint_mae_rad": final_mae,
                "final_joint_max_abs_error_rad": final_max,
                "recovery_ratio": recovery_ratio,
                "recovery_time_s": recovery_time,
                "pose_recovered": bool(
                    math.isfinite(final_mae)
                    and final_mae <= self.recovery_joint_mae_threshold
                    and math.isfinite(final_max)
                    and final_max <= self.recovery_joint_max_threshold
                ),
                "initial_abs_error_by_joint_rad": {
                    name: float(value)
                    for name, value in zip(
                        self.policy_joint_names, self.initial_joint_abs_errors
                    )
                },
                "final_mean_abs_error_by_joint_rad": {
                    name: float(value)
                    for name, value in zip(self.policy_joint_names, final_per_joint)
                },
            },
            "foot_spacing_recovery": self._foot_spacing_recovery_summary(),
            "motion_quality": self._motion_quality_summary(),
        }
