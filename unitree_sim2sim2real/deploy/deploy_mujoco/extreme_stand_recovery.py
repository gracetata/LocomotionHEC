"""MuJoCo perturbation adapter for the full-body extreme Stand policy.

Unlike the ArmHack adapters, this class never edits the policy action.  It only
randomizes the initial MuJoCo state and applies short external wrenches to one
random body at a time, so all 29 actuator targets still come from the actor.
"""

from __future__ import annotations

import math

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
        self.interactive_enabled = bool(
            config.get("extreme_stand_recovery_interactive_enable", False)
        )
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
        ) < 0.0:
            raise ValueError("Extreme Stand perturbation magnitudes must be non-negative.")
        self.body_ids: list[int] = []
        self.next_wrench_time = self.interval
        self.active_until = -1.0
        self.active_body_id = -1
        self.active_body_name = ""
        self.active_wrench = np.zeros(6, dtype=np.float64)
        self.event_count = 0
        self.wrench_events: list[dict[str, object]] = []
        self.initial_joint_limit_clip_count = 0
        self.qpos_addresses: dict[str, int] = {}
        self.default_joint_positions = np.zeros(len(self.policy_joint_names), dtype=np.float64)
        self.initial_joint_abs_errors = np.zeros(len(self.policy_joint_names), dtype=np.float64)
        self.joint_error_times: list[float] = []
        self.joint_abs_error_samples: list[np.ndarray] = []
        self.qvel_addresses: dict[str, int] = {}
        self.default_qpos: np.ndarray | None = None
        self.default_qvel: np.ndarray | None = None
        self.pending_pose_toggle = False
        self.pending_wrench_toggle = False
        self.interaction_events: list[dict[str, object]] = []

    def _joint_noise_limit(self, joint_name: str) -> float:
        if joint_name.startswith("waist_"):
            return self.waist_noise
        if any(token in joint_name for token in ("shoulder", "elbow", "wrist")):
            return self.arm_noise
        return self.leg_noise

    def initialize_model_and_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos_addresses: dict[str, int],
        qvel_addresses: dict[str, int],
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

        self.qpos_addresses = {
            joint_name: int(qpos_addresses[joint_name]) for joint_name in self.policy_joint_names
        }
        self.qvel_addresses = {
            joint_name: int(qvel_addresses[joint_name]) for joint_name in self.policy_joint_names
        }
        self.default_joint_positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        self.default_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        self.default_qvel = np.asarray(data.qvel, dtype=np.float64).copy()
        self._apply_initial_pose(model, data, randomized=self.random_pose_active)
        self.next_wrench_time = self.interval

        if self.interactive_enabled:
            print(
                "[Extreme Stand interactive] SPACE: DEFAULT/RANDOM initial pose; "
                "F: OFF/ON random wrench. "
                f"start_pose={'RANDOM' if self.random_pose_active else 'DEFAULT'} "
                f"start_wrench={'ON' if self.wrench_enabled else 'OFF'}",
                flush=True,
            )

    def _apply_initial_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomized: bool,
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
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise RuntimeError(f"Extreme Stand MuJoCo test joint not found: {joint_name}")
                if bool(model.jnt_limited[joint_id]):
                    lower, upper = (float(value) for value in model.jnt_range[joint_id])
                    available_margin = max(0.0, 0.5 * (upper - lower) - 1.0e-6)
                    margin = min(self.joint_limit_margin, available_margin)
                    clipped_position = float(
                        np.clip(noisy_position, lower + margin, upper - margin)
                    )
                    if not np.isclose(clipped_position, noisy_position, atol=1.0e-12):
                        self.initial_joint_limit_clip_count += 1
                    noisy_position = clipped_position
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

        initial_positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        self.initial_joint_abs_errors = np.abs(
            initial_positions - self.default_joint_positions
        )
        self.joint_error_times.clear()
        self.joint_abs_error_samples.clear()
        mujoco.mj_forward(model, data)

    def key_callback(self, keycode: int) -> None:
        """Queue GUI interaction; physics state is changed only by the simulation thread."""

        if not self.interactive_enabled:
            return
        if int(keycode) == 32:
            self.pending_pose_toggle = True
            print("[Extreme Stand interactive] SPACE requested initial-pose switch.", flush=True)
        elif int(keycode) in (70, 102):
            self.pending_wrench_toggle = True
            print("[Extreme Stand interactive] F requested external-wrench switch.", flush=True)

    def process_interaction_requests(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
    ) -> bool:
        """Apply queued GUI requests and return whether the policy state must be reset."""

        pose_reset = False
        if self.pending_pose_toggle:
            self.pending_pose_toggle = False
            self.random_pose_active = not self.random_pose_active
            self._apply_initial_pose(model, data, randomized=self.random_pose_active)
            self.active_until = -1.0
            self.active_body_id = -1
            self.active_body_name = ""
            self.next_wrench_time = float(sim_time) + self.interval
            state = "RANDOM" if self.random_pose_active else "DEFAULT"
            self.interaction_events.append(
                {"time_s": float(sim_time), "key": "SPACE", "initial_pose": state}
            )
            print(f"[Extreme Stand interactive] SPACE -> initial pose {state}", flush=True)
            pose_reset = True
        if self.pending_wrench_toggle:
            self.pending_wrench_toggle = False
            self.wrench_enabled = not self.wrench_enabled
            data.xfrc_applied[:] = 0.0
            self.active_until = -1.0
            self.active_body_id = -1
            self.active_body_name = ""
            self.next_wrench_time = (
                float(sim_time)
                if self.wrench_enabled
                else float(sim_time) + self.interval
            )
            state = "ON" if self.wrench_enabled else "OFF"
            self.interaction_events.append(
                {"time_s": float(sim_time), "key": "F", "external_wrench": state}
            )
            print(f"[Extreme Stand interactive] F -> random external wrench {state}", flush=True)
        return pose_reset

    def record_state(self, data: mujoco.MjData, sim_time: float) -> None:
        """Record full-body joint error without changing policy actions or simulation state."""

        if not self.qpos_addresses:
            raise RuntimeError("Extreme Stand state recorder used before initialization.")
        positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.policy_joint_names],
            dtype=np.float64,
        )
        self.joint_error_times.append(float(sim_time))
        self.joint_abs_error_samples.append(
            np.abs(positions - self.default_joint_positions)
        )

    def update_external_wrench(self, data: mujoco.MjData, sim_time: float) -> None:
        """Apply a short random wrench to one random body at a time."""

        data.xfrc_applied[:] = 0.0
        if not self.wrench_enabled or (self.force_max <= 0.0 and self.torque_max <= 0.0):
            return
        if sim_time >= self.next_wrench_time:
            index = int(self.rng.integers(0, len(self.body_ids)))
            self.active_body_id = self.body_ids[index]
            self.active_body_name = self.body_names[index]
            self.active_wrench[:3] = self.rng.uniform(-self.force_max, self.force_max, size=3)
            self.active_wrench[3:] = self.rng.uniform(-self.torque_max, self.torque_max, size=3)
            self.active_until = sim_time + self.duration
            self.next_wrench_time += self.interval
            self.event_count += 1
            event = {
                "event_index": self.event_count,
                "time_s": float(sim_time),
                "body_id": self.active_body_id,
                "body_name": self.active_body_name,
                "force_world_n": [float(value) for value in self.active_wrench[:3]],
                "torque_world_nm": [float(value) for value in self.active_wrench[3:]],
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
                    recovery_time = float(start_time)
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
            "interactive": {
                "enabled": self.interactive_enabled,
                "initial_pose_at_end": "random" if self.random_pose_active else "default",
                "wrench_enabled_at_end": self.wrench_enabled,
                "controls": {
                    "SPACE": "toggle_default_random_initial_pose",
                    "F": "toggle_random_wrench",
                },
                "events": list(self.interaction_events),
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
        }
