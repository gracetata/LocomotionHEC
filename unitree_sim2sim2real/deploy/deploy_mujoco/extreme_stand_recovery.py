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
            self.root_linear_velocity_noise,
            self.root_angular_velocity_noise,
            self.force_max,
            self.torque_max,
        ) < 0.0:
            raise ValueError("Extreme Stand perturbation magnitudes must be non-negative.")
        self.body_ids: list[int] = []
        self.next_wrench_time = self.interval
        self.active_until = -1.0
        self.active_body_id = -1
        self.active_body_name = ""
        self.active_wrench = np.zeros(6, dtype=np.float64)
        self.event_count = 0

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
        """Apply bounded initial pose/velocity noise without touching actions."""

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

        for joint_name in self.policy_joint_names:
            limit = self._joint_noise_limit(joint_name)
            data.qpos[qpos_addresses[joint_name]] += self.rng.uniform(-limit, limit)
            data.qvel[qvel_addresses[joint_name]] += self.rng.uniform(
                -self.joint_velocity_noise, self.joint_velocity_noise
            )

        roll = self.rng.uniform(-self.root_roll_pitch_noise, self.root_roll_pitch_noise)
        pitch = self.rng.uniform(-self.root_roll_pitch_noise, self.root_roll_pitch_noise)
        perturb_quat = _euler_xyz_to_quat_wxyz(roll, pitch, 0.0)
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

    def update_external_wrench(self, data: mujoco.MjData, sim_time: float) -> None:
        """Apply a short random wrench to one random body at a time."""

        data.xfrc_applied[:] = 0.0
        if sim_time >= self.next_wrench_time:
            index = int(self.rng.integers(0, len(self.body_ids)))
            self.active_body_id = self.body_ids[index]
            self.active_body_name = self.body_names[index]
            self.active_wrench[:3] = self.rng.uniform(-self.force_max, self.force_max, size=3)
            self.active_wrench[3:] = self.rng.uniform(-self.torque_max, self.torque_max, size=3)
            self.active_until = sim_time + self.duration
            self.next_wrench_time += self.interval
            self.event_count += 1
        if sim_time < self.active_until and self.active_body_id >= 0:
            data.xfrc_applied[self.active_body_id] = self.active_wrench

    def summary(self) -> dict:
        return {
            "action_override": False,
            "seed": int(self.config.get("extreme_stand_recovery_seed", 20260719)),
            "initial_noise": {
                "leg_rad": self.leg_noise,
                "waist_rad": self.waist_noise,
                "arm_rad": self.arm_noise,
                "joint_velocity_rad_s": self.joint_velocity_noise,
                "root_roll_pitch_rad": self.root_roll_pitch_noise,
                "root_linear_velocity_m_s": self.root_linear_velocity_noise,
                "root_angular_velocity_rad_s": self.root_angular_velocity_noise,
            },
            "wrench": {
                "force_max_n": self.force_max,
                "torque_max_nm": self.torque_max,
                "interval_s": self.interval,
                "duration_s": self.duration,
                "event_count": self.event_count,
                "body_names": list(self.body_names),
            },
        }
