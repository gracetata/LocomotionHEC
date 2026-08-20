"""MuJoCo sim2sim runner for IsaacLab-exported Unitree G1 29DoF AMP policies.

Core functions:
    load_config reads g1_amp.yaml and environment overrides. build_observation
    reconstructs the 96-D IsaacLab AMP policy observation in Lab joint order.
    run_mujoco applies policy position targets to the MuJoCo actuator order.
    ensure_floor_xml writes a temporary scene when the robot XML needs a floor,
    light, or a non-invasive Unitree mesh-name remap. The GLFW path can draw
    both the robot torso fixed-point trace and the command-integrated task trace
    in world coordinates. summarize_rollout_metrics reports command tracking,
    health, Important Metrics analogs, and scores.

Inputs/outputs:
    Input is a TorchScript or ONNX policy exported by scripts/export_g1_amp_policy.sh
    and a Unitree G1 29DoF MuJoCo XML. Output is a GLFW visualization or a
    headless MuJoCo rollout plus scalar evaluation metrics. The source XML is
    never modified; any missing floor is written to a generated temporary scene.

Usage:
    python deploy/deploy_mujoco/deploy_mujoco_g1_amp.py deploy/deploy_mujoco/configs/g1_amp.yaml
    G1_AMP_USE_GLFW=False G1_AMP_SIMULATION_DURATION=5 python deploy/deploy_mujoco/deploy_mujoco_g1_amp.py deploy/deploy_mujoco/configs/g1_amp.yaml
    G1_AMP_COMMAND_MODE=joystick G1_AMP_USE_GLFW=True python deploy/deploy_mujoco/deploy_mujoco_g1_amp.py deploy/deploy_mujoco/configs/g1_amp.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml  # type: ignore[reportMissingImports]

from armhack_stand import ARM_JOINT_NAMES, ArmHackStandReplay


UNITREE_ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT_DIR = UNITREE_ROOT_DIR.parent
LEGGED_LAB_ROOT_DIR = PROJECT_ROOT_DIR / "legged_lab"
DEFAULT_MESH_FILE_REMAPS = {
    "waist_yaw_link.STL": "waist_yaw_link_rev_1_0.STL",
    "waist_roll_link.STL": "waist_roll_link_rev_1_0.STL",
    "torso_link.STL": "torso_link_rev_1_0.STL",
}
ROBOT_XML_PATHS = {
    "g1_29dof": UNITREE_ROOT_DIR / "resources/robots/g1_description/g1_29dof.xml",
    "s3_g1_29dof": LEGGED_LAB_ROOT_DIR
    / "source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/g1_29dof.xml",
}
ROBOT_ASSET_ALIASES = {
    "g1": "g1_29dof",
    "g1_29dof": "g1_29dof",
    "g1_29dof_mjcf": "g1_29dof",
    "g1_mjcf": "g1_29dof",
    "original_g1": "g1_29dof",
    "s3": "s3_g1_29dof",
    "s3_g1_29dof": "s3_g1_29dof",
    "s3_g1_29dof_mjcf": "s3_g1_29dof",
    "s3_mjcf": "s3_g1_29dof",
}


def _resolve_path(value: str) -> str:
    return (
        str(value)
        .replace("{PROJECT_ROOT_DIR}", str(PROJECT_ROOT_DIR))
        .replace("{UNITREE_ROOT_DIR}", str(UNITREE_ROOT_DIR))
        .replace("{LEGGED_LAB_ROOT_DIR}", str(LEGGED_LAB_ROOT_DIR))
    )


def _canonical_robot_asset(value: str) -> str:
    robot_asset = value.strip().lower()
    if robot_asset not in ROBOT_ASSET_ALIASES:
        valid_values = ", ".join(sorted(ROBOT_ASSET_ALIASES))
        raise ValueError(f"Unknown robot asset '{value}'. Valid values: {valid_values}")
    return ROBOT_ASSET_ALIASES[robot_asset]


def _resolve_robot_xml(config: dict) -> tuple[str, str]:
    robot_asset_value = os.environ.get(
        "G1_AMP_ROBOT_ASSET", os.environ.get("ROBOT_ASSET", str(config.get("robot_asset", "")))
    ).strip()
    xml_path_value = os.environ.get("G1_AMP_XML_PATH", os.environ.get("XML_PATH", "")).strip()

    if robot_asset_value:
        robot_asset = _canonical_robot_asset(robot_asset_value)
    else:
        robot_asset = ""

    if xml_path_value:
        return robot_asset or "custom", _resolve_path(xml_path_value)
    if robot_asset:
        return robot_asset, _resolve_path(str(ROBOT_XML_PATHS[robot_asset]))
    return str(config.get("robot_asset", "custom")), _resolve_path(config["xml_path"])


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return float(raw_value) if raw_value is not None else default


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return int(raw_value) if raw_value is not None else default


def _env_yaml_list(name: str, default: list[float]) -> list[float]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    values = yaml.safe_load(raw_value)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must be a two-value list, got: {raw_value}")
    return [float(values[0]), float(values[1])]


def _env_yaml_vector(name: str, default: list[float], length: int) -> list[float]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    values = yaml.safe_load(raw_value)
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{name} must be a {length}-value list, got: {raw_value}")
    return [float(value) for value in values]


class PolicyRunner:
    def __init__(self, policy_path: str, runtime: str) -> None:
        self.runtime = runtime.strip().lower()
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
                    "onnxruntime is required for ONNX MuJoCo rollout. "
                    "Install it in the UNITREE_PYTHON environment."
                ) from exc
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.policy = None
            self.session = ort.InferenceSession(
                policy_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(
                    f"Policy must have one input and one output, got {len(inputs)} inputs and {len(outputs)} outputs."
                )
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
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
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Policy output contains non-finite values.")
        return action


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["policy_path"] = _resolve_path(os.environ.get("G1_AMP_POLICY_PATH", config["policy_path"]))
    config["policy_runtime"] = os.environ.get("G1_AMP_POLICY_RUNTIME", str(config.get("policy_runtime", "auto")))
    config["robot_asset"], config["xml_path"] = _resolve_robot_xml(config)
    config["simulation_duration"] = float(os.environ.get("G1_AMP_SIMULATION_DURATION", config["simulation_duration"]))
    config["use_glfw"] = _env_bool("G1_AMP_USE_GLFW", bool(config.get("use_glfw", True)))
    config["real_time"] = _env_bool("G1_AMP_REAL_TIME", bool(config.get("real_time", True)))
    config["add_floor"] = _env_bool("G1_AMP_ADD_FLOOR", bool(config.get("add_floor", True)))
    config["ensure_lighting"] = _env_bool("G1_AMP_ENSURE_LIGHTING", bool(config.get("ensure_lighting", True)))
    config["repair_missing_meshes"] = _env_bool(
        "G1_AMP_REPAIR_MISSING_MESHES", bool(config.get("repair_missing_meshes", True))
    )
    config["drop_missing_mesh_geoms"] = _env_bool(
        "G1_AMP_DROP_MISSING_MESH_GEOMS", bool(config.get("drop_missing_mesh_geoms", True))
    )
    config["apply_joint_passive_params"] = _env_bool(
        "G1_AMP_APPLY_JOINT_PASSIVE_PARAMS", bool(config.get("apply_joint_passive_params", True))
    )
    config["joint_damping"] = _env_float("G1_AMP_JOINT_DAMPING", float(config.get("joint_damping", 0.05)))
    config["joint_armature"] = _env_float("G1_AMP_JOINT_ARMATURE", float(config.get("joint_armature", 0.01)))
    config["joint_frictionloss"] = _env_float("G1_AMP_JOINT_FRICTIONLOSS", float(config.get("joint_frictionloss", 0.2)))
    config["wrist_frictionloss"] = _env_float("G1_AMP_WRIST_FRICTIONLOSS", float(config.get("wrist_frictionloss", 0.1)))
    config["metrics_path"] = os.environ.get("G1_AMP_METRICS_PATH", config.get("metrics_path", ""))
    config["torso_body_name"] = os.environ.get("G1_AMP_TORSO_BODY_NAME", config.get("torso_body_name", "torso_link"))
    config["torso_trace_enable"] = _env_bool(
        "G1_AMP_TORSO_TRACE_ENABLE", bool(config.get("torso_trace_enable", True))
    )
    config["torso_trace_path"] = os.environ.get("G1_AMP_TORSO_TRACE_PATH", config.get("torso_trace_path", ""))
    config["torso_trace_local_point"] = _env_yaml_vector(
        "G1_AMP_TORSO_TRACE_LOCAL_POINT", list(config.get("torso_trace_local_point", [0.0, 0.0, 0.18])), 3
    )
    config["torso_trace_stride"] = _env_int("G1_AMP_TORSO_TRACE_STRIDE", int(config.get("torso_trace_stride", 10)))
    config["torso_trace_max_points"] = _env_int(
        "G1_AMP_TORSO_TRACE_MAX_POINTS", int(config.get("torso_trace_max_points", 300))
    )
    config["follow_camera_enable"] = _env_bool(
        "G1_AMP_FOLLOW_CAMERA_ENABLE", bool(config.get("follow_camera_enable", True))
    )
    config["follow_camera_distance"] = _env_float(
        "G1_AMP_FOLLOW_CAMERA_DISTANCE", float(config.get("follow_camera_distance", 3.2))
    )
    config["follow_camera_azimuth_deg"] = _env_float(
        "G1_AMP_FOLLOW_CAMERA_AZIMUTH_DEG", float(config.get("follow_camera_azimuth_deg", 145.0))
    )
    config["follow_camera_elevation_deg"] = _env_float(
        "G1_AMP_FOLLOW_CAMERA_ELEVATION_DEG", float(config.get("follow_camera_elevation_deg", -20.0))
    )
    config["follow_camera_lookat_local_offset"] = _env_yaml_vector(
        "G1_AMP_FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET",
        list(config.get("follow_camera_lookat_local_offset", [-0.35, -0.20, 0.20])),
        3,
    )
    config["task_trace_enable"] = _env_bool("G1_AMP_TASK_TRACE_ENABLE", bool(config.get("task_trace_enable", True)))
    config["task_trace_path"] = os.environ.get("G1_AMP_TASK_TRACE_PATH", config.get("task_trace_path", ""))
    config["task_trace_height"] = _env_float("G1_AMP_TASK_TRACE_HEIGHT", float(config.get("task_trace_height", 0.05)))
    config["task_trace_stride"] = _env_int("G1_AMP_TASK_TRACE_STRIDE", int(config.get("task_trace_stride", 10)))
    config["task_trace_max_points"] = _env_int(
        "G1_AMP_TASK_TRACE_MAX_POINTS", int(config.get("task_trace_max_points", 300))
    )
    config["healthy_min_root_height"] = _env_float(
        "G1_AMP_HEALTHY_MIN_ROOT_HEIGHT", float(config.get("healthy_min_root_height", 0.45))
    )
    config["healthy_max_roll_pitch"] = _env_float(
        "G1_AMP_HEALTHY_MAX_ROLL_PITCH", float(config.get("healthy_max_roll_pitch", 1.0))
    )
    if "G1_AMP_CMD_INIT" in os.environ:
        config["cmd_init"] = yaml.safe_load(os.environ["G1_AMP_CMD_INIT"])
    config["random_commands"] = _env_bool("G1_AMP_RANDOM_COMMANDS", bool(config.get("random_commands", False)))
    config["command_mode"] = os.environ.get("G1_AMP_COMMAND_MODE", config.get("command_mode", "independent"))
    config["command_seed"] = _env_int("G1_AMP_COMMAND_SEED", int(config.get("command_seed", 1)))
    config["command_interval"] = _env_float("G1_AMP_COMMAND_INTERVAL", float(config.get("command_interval", 2.0)))
    config["command_ramp"] = _env_bool("G1_AMP_COMMAND_RAMP", bool(config.get("command_ramp", False)))
    config["command_smoothing_tau"] = _env_float(
        "G1_AMP_COMMAND_SMOOTHING_TAU", float(config.get("command_smoothing_tau", 0.25))
    )
    config["command_max_linear_accel"] = _env_float(
        "G1_AMP_COMMAND_MAX_LINEAR_ACCEL", float(config.get("command_max_linear_accel", 1.2))
    )
    config["command_max_yaw_accel"] = _env_float(
        "G1_AMP_COMMAND_MAX_YAW_ACCEL", float(config.get("command_max_yaw_accel", 1.5))
    )
    config["nav2_data_path"] = _resolve_path(os.environ.get("G1_AMP_NAV2_DATA_PATH", config.get("nav2_data_path", "")))
    config["nav2_augmentation_filter"] = os.environ.get(
        "G1_AMP_NAV2_AUGMENTATION_FILTER", config.get("nav2_augmentation_filter", "none,mirror_lr")
    )
    config["nav2_scenario_family_filter"] = os.environ.get(
        "G1_AMP_NAV2_SCENARIO_FAMILY_FILTER", config.get("nav2_scenario_family_filter", "")
    )
    config["nav2_combo_filter"] = os.environ.get("G1_AMP_NAV2_COMBO_FILTER", config.get("nav2_combo_filter", ""))
    config["nav2_controller_filter"] = os.environ.get(
        "G1_AMP_NAV2_CONTROLLER_FILTER", config.get("nav2_controller_filter", "")
    )
    config["nav2_planner_filter"] = os.environ.get("G1_AMP_NAV2_PLANNER_FILTER", config.get("nav2_planner_filter", ""))
    config["nav2_dataset_sample_dt"] = _env_float(
        "G1_AMP_NAV2_DATASET_SAMPLE_DT", float(config.get("nav2_dataset_sample_dt", 0.05))
    )
    config["nav2_window_duration_s"] = _env_float(
        "G1_AMP_NAV2_WINDOW_DURATION_S", float(config.get("nav2_window_duration_s", 0.0))
    )
    config["nav2_command_scale"] = _env_yaml_vector(
        "G1_AMP_NAV2_COMMAND_SCALE", list(config.get("nav2_command_scale", [0.70, 0.55, 0.55])), 3
    )
    config["nav2_command_clip_min"] = _env_yaml_vector(
        "G1_AMP_NAV2_COMMAND_CLIP_MIN", list(config.get("nav2_command_clip_min", [-0.6, -0.3, -0.6])), 3
    )
    config["nav2_command_clip_max"] = _env_yaml_vector(
        "G1_AMP_NAV2_COMMAND_CLIP_MAX", list(config.get("nav2_command_clip_max", [0.6, 0.3, 0.6])), 3
    )
    config["early_motion_enable"] = _env_bool(
        "G1_AMP_EARLY_MOTION_ENABLE", bool(config.get("early_motion_enable", True))
    )
    config["early_motion_window_s"] = _env_float(
        "G1_AMP_EARLY_MOTION_WINDOW_S", float(config.get("early_motion_window_s", 1.0))
    )
    config["joint_random_enable"] = _env_bool(
        "G1_AMP_JOINT_RANDOM_ENABLE", bool(config.get("joint_random_enable", False))
    )
    config["joint_random_seed"] = _env_int("G1_AMP_JOINT_RANDOM_SEED", int(config.get("joint_random_seed", 20260718)))
    config["joint_pos_noise_rad"] = _env_float(
        "G1_AMP_JOINT_POS_NOISE_RAD", float(config.get("joint_pos_noise_rad", 0.0))
    )
    config["joint_vel_noise_rad_per_s"] = _env_float(
        "G1_AMP_JOINT_VEL_NOISE_RAD_PER_S", float(config.get("joint_vel_noise_rad_per_s", 0.0))
    )
    config["non_arm_joint_target_noise_enable"] = _env_bool(
        "G1_AMP_NON_ARM_JOINT_TARGET_NOISE_ENABLE",
        bool(config.get("non_arm_joint_target_noise_enable", False)),
    )
    config["non_arm_joint_target_noise_seed"] = _env_int(
        "G1_AMP_NON_ARM_JOINT_TARGET_NOISE_SEED",
        int(config.get("non_arm_joint_target_noise_seed", 20260719)),
    )
    config["non_arm_joint_target_noise_rad"] = _env_float(
        "G1_AMP_NON_ARM_JOINT_TARGET_NOISE_RAD",
        float(config.get("non_arm_joint_target_noise_rad", 0.0)),
    )
    config["foot_recovery_enable"] = _env_bool(
        "G1_AMP_FOOT_RECOVERY_ENABLE", bool(config.get("foot_recovery_enable", False))
    )
    config["ordered_step_observation_enable"] = _env_bool(
        "G1_AMP_ORDERED_STEP_OBSERVATION_ENABLE",
        bool(config.get("ordered_step_observation_enable", False)),
    )
    config["ordered_step_mirror_policy_enable"] = _env_bool(
        "G1_AMP_ORDERED_STEP_MIRROR_POLICY_ENABLE",
        bool(config.get("ordered_step_mirror_policy_enable", False)),
    )
    config["ordered_step_hold_last_action_enable"] = _env_bool(
        "G1_AMP_ORDERED_STEP_HOLD_LAST_ACTION_ENABLE",
        bool(config.get("ordered_step_hold_last_action_enable", False)),
    )
    config["ordered_step_hold_policy_path"] = os.environ.get(
        "G1_AMP_ORDERED_STEP_HOLD_POLICY_PATH",
        str(config.get("ordered_step_hold_policy_path", "")),
    )
    config["ordered_step_hold_blend_duration_s"] = _env_float(
        "G1_AMP_ORDERED_STEP_HOLD_BLEND_DURATION_S",
        float(config.get("ordered_step_hold_blend_duration_s", 1.0)),
    )
    config["ordered_step_transition_tolerance_m"] = _env_float(
        "G1_AMP_ORDERED_STEP_TRANSITION_TOLERANCE_M",
        float(config.get("ordered_step_transition_tolerance_m", 0.055)),
    )
    config["ordered_step_min_clearance_m"] = _env_float(
        "G1_AMP_ORDERED_STEP_MIN_CLEARANCE_M",
        float(config.get("ordered_step_min_clearance_m", 0.035)),
    )
    config["ordered_step_min_duration_s"] = _env_float(
        "G1_AMP_ORDERED_STEP_MIN_DURATION_S",
        float(config.get("ordered_step_min_duration_s", 0.0)),
    )
    config["ordered_step_action_smoothing_alpha"] = _env_float(
        "G1_AMP_ORDERED_STEP_ACTION_SMOOTHING_ALPHA",
        float(config.get("ordered_step_action_smoothing_alpha", 1.0)),
    )
    config["initial_ankle_distance_m"] = _env_float(
        "G1_AMP_INITIAL_ANKLE_DISTANCE_M", float(config.get("initial_ankle_distance_m", 0.30))
    )
    config["interactive_stance_reset"] = _env_bool(
        "G1_AMP_INTERACTIVE_STANCE_RESET", bool(config.get("interactive_stance_reset", False))
    )
    config["interactive_stance_distance_range_m"] = _env_yaml_list(
        "G1_AMP_INTERACTIVE_STANCE_DISTANCE_RANGE_M",
        list(config.get("interactive_stance_distance_range_m", [0.08, 0.48])),
    )
    config["interactive_stance_seed"] = _env_int(
        "G1_AMP_INTERACTIVE_STANCE_SEED", int(config.get("interactive_stance_seed", 20260814))
    )
    config["position_recovery_command_enable"] = _env_bool(
        "G1_AMP_POSITION_RECOVERY_COMMAND_ENABLE",
        bool(config.get("position_recovery_command_enable", False)),
    )
    config["position_recovery_command_xy_clip_m"] = _env_float(
        "G1_AMP_POSITION_RECOVERY_COMMAND_XY_CLIP_M",
        float(config.get("position_recovery_command_xy_clip_m", 0.50)),
    )
    config["position_recovery_command_yaw_clip_rad"] = _env_float(
        "G1_AMP_POSITION_RECOVERY_COMMAND_YAW_CLIP_RAD",
        float(config.get("position_recovery_command_yaw_clip_rad", 0.60)),
    )
    config["position_recovery_command_xy_gain"] = _env_float(
        "G1_AMP_POSITION_RECOVERY_COMMAND_XY_GAIN",
        float(config.get("position_recovery_command_xy_gain", 2.0)),
    )
    config["position_recovery_command_yaw_gain"] = _env_float(
        "G1_AMP_POSITION_RECOVERY_COMMAND_YAW_GAIN",
        float(config.get("position_recovery_command_yaw_gain", 1.5)),
    )
    config["target_ankle_distance_m"] = _env_float(
        "G1_AMP_TARGET_ANKLE_DISTANCE_M", float(config.get("target_ankle_distance_m", 0.30))
    )
    config["ankle_distance_tolerance_m"] = _env_float(
        "G1_AMP_ANKLE_DISTANCE_TOLERANCE_M", float(config.get("ankle_distance_tolerance_m", 0.03))
    )
    config["ankle_convergence_hold_s"] = _env_float(
        "G1_AMP_ANKLE_CONVERGENCE_HOLD_S", float(config.get("ankle_convergence_hold_s", 0.50))
    )
    config["recovery_settle_time_s"] = _env_float(
        "G1_AMP_RECOVERY_SETTLE_TIME_S", float(config.get("recovery_settle_time_s", 5.0))
    )
    config["mujoco_push_enable"] = _env_bool(
        "G1_AMP_MUJOCO_PUSH_ENABLE", bool(config.get("mujoco_push_enable", False))
    )
    config["mujoco_push_seed"] = _env_int(
        "G1_AMP_MUJOCO_PUSH_SEED", int(config.get("mujoco_push_seed", 20260814))
    )
    config["mujoco_push_first_time_s"] = _env_float(
        "G1_AMP_MUJOCO_PUSH_FIRST_TIME_S", float(config.get("mujoco_push_first_time_s", 6.0))
    )
    config["mujoco_push_interval_range_s"] = _env_yaml_list(
        "G1_AMP_MUJOCO_PUSH_INTERVAL_RANGE_S", list(config.get("mujoco_push_interval_range_s", [3.0, 6.0]))
    )
    config["mujoco_push_duration_s"] = _env_float(
        "G1_AMP_MUJOCO_PUSH_DURATION_S", float(config.get("mujoco_push_duration_s", 0.12))
    )
    config["mujoco_push_force_range_n"] = _env_yaml_list(
        "G1_AMP_MUJOCO_PUSH_FORCE_RANGE_N", list(config.get("mujoco_push_force_range_n", [80.0, 120.0]))
    )
    config["mujoco_push_yaw_torque_range_nm"] = _env_yaml_list(
        "G1_AMP_MUJOCO_PUSH_YAW_TORQUE_RANGE_NM",
        list(config.get("mujoco_push_yaw_torque_range_nm", [-8.0, 8.0])),
    )
    config["armhack_stand_enable"] = _env_bool(
        "G1_AMP_ARMHACK_STAND_ENABLE", bool(config.get("armhack_stand_enable", False))
    )
    config["armhack_stand_csv_path"] = _resolve_path(
        os.environ.get("G1_AMP_ARMHACK_STAND_CSV_PATH", config.get("armhack_stand_csv_path", ""))
    )
    config["armhack_stand_manifest_path"] = _resolve_path(
        os.environ.get("G1_AMP_ARMHACK_STAND_MANIFEST_PATH", config.get("armhack_stand_manifest_path", ""))
    )
    config["armhack_stand_checkpoint_path"] = _resolve_path(
        os.environ.get("G1_AMP_ARMHACK_STAND_CHECKPOINT_PATH", config.get("armhack_stand_checkpoint_path", ""))
    )
    config["armhack_stand_report_path"] = _resolve_path(
        os.environ.get("G1_AMP_ARMHACK_STAND_REPORT_PATH", config.get("armhack_stand_report_path", ""))
    )
    config["armhack_stand_test_id"] = os.environ.get(
        "G1_AMP_ARMHACK_STAND_TEST_ID", config.get("armhack_stand_test_id", "all")
    )
    config["armhack_stand_payload_kg"] = _env_float(
        "G1_AMP_ARMHACK_STAND_PAYLOAD_KG", float(config.get("armhack_stand_payload_kg", 0.0))
    )
    command_ranges = dict(config.get("command_ranges", {}))
    command_ranges["lin_vel_x"] = _env_yaml_list(
        "G1_AMP_CMD_LIN_X_RANGE", list(command_ranges.get("lin_vel_x", [-0.2, 1.0]))
    )
    command_ranges["lin_vel_y"] = _env_yaml_list(
        "G1_AMP_CMD_LIN_Y_RANGE", list(command_ranges.get("lin_vel_y", [-0.25, 0.25]))
    )
    command_ranges["yaw_rate"] = _env_yaml_list(
        "G1_AMP_CMD_YAW_RANGE", list(command_ranges.get("yaw_rate", [-0.6, 0.6]))
    )
    command_ranges["curvature"] = _env_yaml_list(
        "G1_AMP_CMD_CURVATURE_RANGE", list(command_ranges.get("curvature", [-0.7, 0.7]))
    )
    command_ranges["low_speed_lin_vel_x"] = _env_yaml_list(
        "G1_AMP_CMD_LOW_SPEED_LIN_X_RANGE", list(command_ranges.get("low_speed_lin_vel_x", [-0.15, 0.30]))
    )
    command_ranges["low_speed_lin_vel_y"] = _env_yaml_list(
        "G1_AMP_CMD_LOW_SPEED_LIN_Y_RANGE", list(command_ranges.get("low_speed_lin_vel_y", [-0.20, 0.20]))
    )
    command_ranges["low_speed_yaw_rate"] = _env_yaml_list(
        "G1_AMP_CMD_LOW_SPEED_YAW_RANGE", list(command_ranges.get("low_speed_yaw_rate", [-0.50, 0.50]))
    )
    command_ranges["yaw_noise"] = _env_yaml_list(
        "G1_AMP_CMD_YAW_NOISE_RANGE", list(command_ranges.get("yaw_noise", [-0.05, 0.05]))
    )
    config["command_ranges"] = command_ranges
    joystick_ranges = dict(config.get("joystick_ranges", {}))
    joystick_ranges["lin_vel_x"] = _env_yaml_list(
        "G1_AMP_JOYSTICK_LIN_X_RANGE", list(joystick_ranges.get("lin_vel_x", command_ranges["lin_vel_x"]))
    )
    joystick_ranges["lin_vel_y"] = _env_yaml_list(
        "G1_AMP_JOYSTICK_LIN_Y_RANGE", list(joystick_ranges.get("lin_vel_y", command_ranges["lin_vel_y"]))
    )
    joystick_ranges["yaw_rate"] = _env_yaml_list(
        "G1_AMP_JOYSTICK_YAW_RANGE", list(joystick_ranges.get("yaw_rate", command_ranges["yaw_rate"]))
    )
    config["joystick_ranges"] = joystick_ranges
    config["joystick_device"] = os.environ.get("G1_AMP_JOYSTICK_DEVICE", config.get("joystick_device", "/dev/input/js0"))
    config["joystick_axis_lin_x"] = _env_int("G1_AMP_JOYSTICK_AXIS_LIN_X", int(config.get("joystick_axis_lin_x", 1)))
    config["joystick_axis_lin_y"] = _env_int("G1_AMP_JOYSTICK_AXIS_LIN_Y", int(config.get("joystick_axis_lin_y", 0)))
    config["joystick_axis_yaw"] = _env_int("G1_AMP_JOYSTICK_AXIS_YAW", int(config.get("joystick_axis_yaw", 3)))
    config["joystick_sign_lin_x"] = _env_float("G1_AMP_JOYSTICK_SIGN_LIN_X", float(config.get("joystick_sign_lin_x", -1.0)))
    config["joystick_sign_lin_y"] = _env_float("G1_AMP_JOYSTICK_SIGN_LIN_Y", float(config.get("joystick_sign_lin_y", -1.0)))
    config["joystick_sign_yaw"] = _env_float("G1_AMP_JOYSTICK_SIGN_YAW", float(config.get("joystick_sign_yaw", -1.0)))
    config["joystick_axis_max"] = _env_float("G1_AMP_JOYSTICK_AXIS_MAX", float(config.get("joystick_axis_max", 32768.0)))
    config["joystick_deadzone"] = _env_float("G1_AMP_JOYSTICK_DEADZONE", float(config.get("joystick_deadzone", 0.05)))
    config["command_rel_low_speed"] = _env_float(
        "G1_AMP_CMD_REL_LOW_SPEED", float(config.get("command_rel_low_speed", 0.25))
    )
    config["command_max_curvature"] = _env_float(
        "G1_AMP_CMD_MAX_CURVATURE", float(config.get("command_max_curvature", 0.7))
    )
    config["command_high_speed_lateral_vel"] = _env_float(
        "G1_AMP_CMD_HIGH_SPEED_LATERAL_VEL", float(config.get("command_high_speed_lateral_vel", 0.06))
    )
    config["command_lateral_decay_start_speed"] = _env_float(
        "G1_AMP_CMD_LATERAL_DECAY_START_SPEED", float(config.get("command_lateral_decay_start_speed", 0.25))
    )
    config["command_lateral_decay_end_speed"] = _env_float(
        "G1_AMP_CMD_LATERAL_DECAY_END_SPEED", float(config.get("command_lateral_decay_end_speed", 0.80))
    )
    return config


def _format_array(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(str(float(value)) for value in values)


def _mesh_file_ok(mesh_path: Path) -> bool:
    return mesh_path.is_file() and mesh_path.stat().st_size > 0


def _remove_mesh_geoms(parent: ET.Element, missing_mesh_names: set[str]) -> int:
    removed = 0
    for child in list(parent):
        if child.tag == "geom" and child.get("mesh") in missing_mesh_names:
            parent.remove(child)
            removed += 1
            continue
        removed += _remove_mesh_geoms(child, missing_mesh_names)
    return removed


def _prepare_mesh_assets(root: ET.Element, source_path: Path, config: dict, scene_report: dict) -> bool:
    if not bool(config.get("repair_missing_meshes", True)):
        return False

    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
    mesh_dir = meshdir if meshdir.is_absolute() else source_path.parent / meshdir
    mesh_file_remaps = dict(DEFAULT_MESH_FILE_REMAPS)
    mesh_file_remaps.update(config.get("mesh_file_remaps", {}) or {})
    missing_mesh_names: set[str] = set()
    changed = False

    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            mesh_file = mesh.get("file")
            mesh_name = mesh.get("name")
            if not mesh_file or not mesh_name:
                continue
            if _mesh_file_ok(mesh_dir / mesh_file):
                continue
            remap_file = mesh_file_remaps.get(Path(mesh_file).name)
            if remap_file and _mesh_file_ok(mesh_dir / remap_file):
                mesh.set("file", remap_file)
                scene_report["mesh_file_remaps"].append({"mesh": mesh_name, "from": mesh_file, "to": remap_file})
                changed = True
                continue
            missing_mesh_names.add(mesh_name)
            scene_report["dropped_missing_mesh_assets"].append({"mesh": mesh_name, "file": mesh_file})
            asset.remove(mesh)
            changed = True

    if not missing_mesh_names:
        return changed

    if not bool(config.get("drop_missing_mesh_geoms", True)):
        missing_files = ", ".join(item["file"] for item in scene_report["dropped_missing_mesh_assets"])
        raise FileNotFoundError(f"Missing Unitree mesh files and dropping is disabled: {missing_files}")

    removed_geoms = _remove_mesh_geoms(root, missing_mesh_names)
    scene_report["dropped_missing_mesh_geom_count"] = removed_geoms
    return True


def _ensure_compiler_meshdir_absolute(root: ET.Element, source_path: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None or not compiler.get("meshdir"):
        return
    meshdir = Path(compiler.get("meshdir", ""))
    if not meshdir.is_absolute():
        compiler.set("meshdir", str((source_path.parent / meshdir).resolve()))


def _worldbodies(root: ET.Element, source_path: Path) -> list[ET.Element]:
    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError(f"MuJoCo XML has no <worldbody>: {source_path}")
    return worldbodies


def _has_floor(worldbodies: list[ET.Element]) -> bool:
    for worldbody in worldbodies:
        for geom in worldbody.findall("geom"):
            if geom.get("type") == "plane" or geom.get("name", "").lower() in {"floor", "ground"}:
                return True
    return False


def _ensure_lighting(root: ET.Element, worldbody: ET.Element, config: dict, scene_report: dict) -> bool:
    if not bool(config.get("ensure_lighting", True)):
        return False
    changed = False
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
        changed = True
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {"diffuse": "0.7 0.7 0.7", "ambient": "0.2 0.2 0.2", "specular": "0.3 0.3 0.3"},
        )
        scene_report["added_headlight"] = True
        changed = True
    if not root.findall(".//light"):
        ET.SubElement(
            worldbody,
            "light",
            {"name": "key_light", "pos": "1 -3 4", "dir": "-0.2 0.4 -1", "directional": "true"},
        )
        scene_report["added_world_light"] = True
        changed = True
    return changed


def _ensure_joint_passive_params(root: ET.Element, config: dict, scene_report: dict) -> bool:
    if not bool(config.get("apply_joint_passive_params", True)):
        return False
    changed = False
    updated = 0
    for joint in root.findall(".//joint"):
        if joint.get("type") == "free" or joint.get("name") == "floating_base_joint":
            continue
        joint_name = joint.get("name", "")
        frictionloss = float(config.get("joint_frictionloss", 0.2))
        if joint_name.endswith("wrist_pitch_joint") or joint_name.endswith("wrist_yaw_joint"):
            frictionloss = float(config.get("wrist_frictionloss", 0.1))
        desired = {
            "damping": float(config.get("joint_damping", 0.05)),
            "armature": float(config.get("joint_armature", 0.01)),
            "frictionloss": frictionloss,
        }
        joint_changed = False
        for attr, value in desired.items():
            value_text = f"{value:g}"
            if joint.get(attr) != value_text:
                joint.set(attr, value_text)
                joint_changed = True
        if joint_changed:
            updated += 1
            changed = True
    scene_report["joint_passive_params_updated"] = updated
    return changed


def ensure_floor_xml(xml_path: str, config: dict) -> str:
    source_path = Path(xml_path).expanduser().resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()
    worldbodies = _worldbodies(root, source_path)
    scene_report = {
        "source_xml": str(source_path),
        "generated_xml": None,
        "mesh_file_remaps": [],
        "dropped_missing_mesh_assets": [],
        "dropped_missing_mesh_geom_count": 0,
        "added_floor": False,
        "added_headlight": False,
        "added_world_light": False,
        "joint_passive_params_updated": 0,
    }
    changed = _prepare_mesh_assets(root, source_path, config, scene_report)
    changed = _ensure_lighting(root, worldbodies[0], config, scene_report) or changed
    changed = _ensure_joint_passive_params(root, config, scene_report) or changed

    if not bool(config.get("add_floor", True)) or _has_floor(worldbodies):
        if changed:
            _ensure_compiler_meshdir_absolute(root, source_path)
            generated_path = Path(tempfile.gettempdir()) / f"{source_path.stem}_sim2sim_scene.xml"
            tree.write(generated_path, encoding="utf-8", xml_declaration=True)
            scene_report["generated_xml"] = str(generated_path)
            config["_scene_report"] = scene_report
            print(f"[INFO] Generated MuJoCo scene: {generated_path}")
            return str(generated_path)
        config["_scene_report"] = scene_report
        return str(source_path)

    floor_size = config.get("floor_size", [20.0, 20.0, 0.05])
    floor_rgba = config.get("floor_rgba", [0.25, 0.25, 0.25, 1.0])
    floor_friction = config.get("floor_friction", [1.0, 0.005, 0.0001])
    ET.SubElement(
        worldbodies[0],
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": _format_array(floor_size),
            "rgba": _format_array(floor_rgba),
            "friction": _format_array(floor_friction),
            "condim": "3",
        },
    )
    scene_report["added_floor"] = True
    _ensure_compiler_meshdir_absolute(root, source_path)

    generated_path = Path(tempfile.gettempdir()) / f"{source_path.stem}_sim2sim_scene.xml"
    tree.write(generated_path, encoding="utf-8", xml_declaration=True)
    scene_report["generated_xml"] = str(generated_path)
    config["_scene_report"] = scene_report
    print(f"[INFO] Generated MuJoCo scene: {generated_path}")
    return str(generated_path)


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def quat_to_roll_pitch_yaw(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=np.float32)


def quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_w, first_x, first_y, first_z = first
    second_w, second_x, second_y, second_z = second
    return np.array(
        [
            first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
            first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
            first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
            first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
        ],
        dtype=np.float32,
    )


def quat_delta_to_ang_vel_w(previous_quat: np.ndarray, current_quat: np.ndarray, dt: float) -> np.ndarray:
    delta_quat = quat_multiply(current_quat, quat_conjugate(previous_quat))
    if delta_quat[0] < 0.0:
        delta_quat *= -1.0
    vector_norm = float(np.linalg.norm(delta_quat[1:4]))
    if vector_norm < 1.0e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(delta_quat[0]))
    axis = delta_quat[1:4] / vector_norm
    return (axis * (angle / max(dt, 1.0e-8))).astype(np.float32)


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def torso_fixed_point_w(data: mujoco.MjData, torso_body_id: int, local_point: np.ndarray) -> np.ndarray:
    torso_rotation = data.xmat[torso_body_id].reshape(3, 3)
    return data.xpos[torso_body_id].copy() + torso_rotation @ local_point


def update_follow_camera(viewer, data: mujoco.MjData, torso_body_id: int, config: dict) -> None:
    if not bool(config.get("follow_camera_enable", True)):
        return
    torso_pos_w = data.xpos[torso_body_id].copy().astype(np.float32)
    local_offset = np.asarray(config.get("follow_camera_lookat_local_offset", [-0.35, -0.20, 0.20]), dtype=np.float32)
    lookat_w = torso_pos_w + local_offset
    viewer.cam.lookat[:] = lookat_w.astype(np.float64)
    viewer.cam.distance = float(config.get("follow_camera_distance", 3.2))
    viewer.cam.azimuth = float(config.get("follow_camera_azimuth_deg", 145.0))
    viewer.cam.elevation = float(config.get("follow_camera_elevation_deg", -20.0))


def make_joint_address_maps(model: mujoco.MjModel, joint_names: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    qpos_addresses: dict[str, int] = {}
    qvel_addresses: dict[str, int] = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
        qpos_addresses[joint_name] = int(model.jnt_qposadr[joint_id])
        qvel_addresses[joint_name] = int(model.jnt_dofadr[joint_id])
    return qpos_addresses, qvel_addresses


def apply_initial_joint_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    qpos_addresses: dict[str, int],
    qvel_addresses: dict[str, int],
    config: dict,
) -> dict:
    if not bool(config.get("joint_random_enable", False)):
        return {"enabled": False}

    pos_noise_rad = float(config.get("joint_pos_noise_rad", 0.0))
    vel_noise_rad_per_s = float(config.get("joint_vel_noise_rad_per_s", 0.0))
    if pos_noise_rad < 0.0 or vel_noise_rad_per_s < 0.0:
        raise ValueError("Joint randomization noise magnitudes must be non-negative.")
    if pos_noise_rad == 0.0 and vel_noise_rad_per_s == 0.0:
        return {"enabled": False}

    rng = np.random.default_rng(int(config.get("joint_random_seed", 20260718)))
    pos_noise_by_joint: dict[str, float] = {}
    vel_noise_by_joint: dict[str, float] = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found for randomization.")

        pos_noise = float(rng.uniform(-pos_noise_rad, pos_noise_rad)) if pos_noise_rad > 0.0 else 0.0
        qpos_address = qpos_addresses[joint_name]
        randomized_qpos = float(data.qpos[qpos_address]) + pos_noise
        if int(model.jnt_limited[joint_id]):
            lower, upper = (float(value) for value in model.jnt_range[joint_id])
            randomized_qpos = float(np.clip(randomized_qpos, lower, upper))
            pos_noise = randomized_qpos - float(data.qpos[qpos_address])
        data.qpos[qpos_address] = randomized_qpos
        pos_noise_by_joint[joint_name] = pos_noise

        vel_noise = float(rng.uniform(-vel_noise_rad_per_s, vel_noise_rad_per_s)) if vel_noise_rad_per_s > 0.0 else 0.0
        data.qvel[qvel_addresses[joint_name]] = vel_noise
        vel_noise_by_joint[joint_name] = vel_noise

    return {
        "enabled": True,
        "seed": int(config.get("joint_random_seed", 20260718)),
        "joint_count": len(joint_names),
        "pos_noise_rad": pos_noise_rad,
        "vel_noise_rad_per_s": vel_noise_rad_per_s,
        "max_abs_applied_pos_noise_rad": float(max(abs(value) for value in pos_noise_by_joint.values())),
        "max_abs_applied_vel_noise_rad_per_s": float(max(abs(value) for value in vel_noise_by_joint.values())),
        "pos_noise_by_joint_rad": pos_noise_by_joint,
        "vel_noise_by_joint_rad_per_s": vel_noise_by_joint,
    }


def apply_initial_foot_recovery_stance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: dict[str, int],
    config: dict,
) -> dict:
    """Set a symmetric G1 stance corresponding to a requested ankle distance."""
    if not bool(config.get("foot_recovery_enable", False)):
        return {"enabled": False}

    requested_distance = float(config.get("initial_ankle_distance_m", 0.30))
    if not 0.05 <= requested_distance <= 0.60:
        raise ValueError("initial_ankle_distance_m must be within [0.05, 0.60].")
    nominal_distance = 0.237
    distance_per_rad = 1.22
    roll_angle = (requested_distance - nominal_distance) / distance_per_rad
    signed_targets = {
        "left_hip_roll_joint": roll_angle,
        "right_hip_roll_joint": -roll_angle,
        "left_ankle_roll_joint": -roll_angle,
        "right_ankle_roll_joint": roll_angle,
    }
    applied_targets: dict[str, float] = {}
    for joint_name, target in signed_targets.items():
        if joint_name not in qpos_addresses:
            raise ValueError(f"MuJoCo foot recovery is missing joint: {joint_name}")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo foot recovery is missing joint model entry: {joint_name}")
        if int(model.jnt_limited[joint_id]):
            target = float(np.clip(target, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1]))
        data.qpos[qpos_addresses[joint_name]] = target
        applied_targets[joint_name] = float(target)

    mujoco.mj_forward(model, data)
    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if left_body_id < 0 or right_body_id < 0:
        raise ValueError("MuJoCo foot recovery requires both ankle_roll_link bodies.")
    actual_distance = float(np.linalg.norm(data.xpos[left_body_id, :2] - data.xpos[right_body_id, :2]))
    return {
        "enabled": True,
        "requested_distance_m": requested_distance,
        "actual_distance_m": actual_distance,
        "roll_angle_rad": float(roll_angle),
        "joint_targets_rad": applied_targets,
    }


class MujocoPushScheduler:
    """Apply deterministic, finite-duration horizontal pushes to the torso."""

    def __init__(self, config: dict, torso_body_id: int):
        self.enabled = bool(config.get("mujoco_push_enable", False))
        self.torso_body_id = int(torso_body_id)
        self.rng = np.random.default_rng(int(config.get("mujoco_push_seed", 20260814)))
        self.interval_range = tuple(float(value) for value in config.get("mujoco_push_interval_range_s", [3.0, 6.0]))
        self.force_range = tuple(float(value) for value in config.get("mujoco_push_force_range_n", [80.0, 120.0]))
        self.yaw_torque_range = tuple(
            float(value) for value in config.get("mujoco_push_yaw_torque_range_nm", [-8.0, 8.0])
        )
        self.duration_s = float(config.get("mujoco_push_duration_s", 0.12))
        self.next_start_s = float(config.get("mujoco_push_first_time_s", 6.0))
        self.active_until_s = -1.0
        self.active_wrench = np.zeros(6, dtype=np.float64)
        self.events: list[dict] = []
        if self.interval_range[0] <= 0.0 or self.interval_range[1] < self.interval_range[0]:
            raise ValueError("mujoco_push_interval_range_s must satisfy 0 < min <= max.")
        if self.force_range[0] < 0.0 or self.force_range[1] < self.force_range[0]:
            raise ValueError("mujoco_push_force_range_n must satisfy 0 <= min <= max.")
        if self.duration_s <= 0.0:
            raise ValueError("mujoco_push_duration_s must be positive.")

    def apply(self, data: mujoco.MjData, sim_time: float) -> None:
        data.xfrc_applied[self.torso_body_id, :] = 0.0
        if not self.enabled:
            return
        if sim_time >= self.next_start_s and sim_time >= self.active_until_s:
            direction = float(self.rng.uniform(-math.pi, math.pi))
            magnitude = float(self.rng.uniform(self.force_range[0], self.force_range[1]))
            yaw_torque = float(self.rng.uniform(self.yaw_torque_range[0], self.yaw_torque_range[1]))
            self.active_wrench[:] = [
                magnitude * math.cos(direction),
                magnitude * math.sin(direction),
                0.0,
                0.0,
                0.0,
                yaw_torque,
            ]
            self.active_until_s = sim_time + self.duration_s
            self.next_start_s = self.active_until_s + float(
                self.rng.uniform(self.interval_range[0], self.interval_range[1])
            )
            self.events.append(
                {
                    "start_time_s": float(sim_time),
                    "duration_s": self.duration_s,
                    "force_w_n": [float(value) for value in self.active_wrench[:3]],
                    "torque_w_nm": [float(value) for value in self.active_wrench[3:]],
                }
            )
        if sim_time < self.active_until_s:
            data.xfrc_applied[self.torso_body_id, :] = self.active_wrench


def apply_non_arm_target_noise(
    action: np.ndarray,
    non_arm_policy_indices: np.ndarray,
    rng: np.random.Generator,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    noise_raw = np.zeros_like(action, dtype=np.float32)
    if not bool(config.get("non_arm_joint_target_noise_enable", False)):
        return action, noise_raw

    noise_rad = float(config.get("non_arm_joint_target_noise_rad", 0.0))
    if noise_rad < 0.0:
        raise ValueError("non_arm_joint_target_noise_rad must be non-negative.")
    if noise_rad == 0.0 or non_arm_policy_indices.size == 0:
        return action, noise_raw

    action_scale = float(config["action_scale"])
    if action_scale <= 0.0:
        raise ValueError("action_scale must be positive for target noise.")
    noise_raw[non_arm_policy_indices] = (
        rng.uniform(-noise_rad, noise_rad, size=non_arm_policy_indices.size).astype(np.float32) / action_scale
    )
    return (action + noise_raw).astype(np.float32), noise_raw


def init_policy_application_metrics(policy_joint_names: list[str], non_arm_policy_indices: np.ndarray) -> dict:
    return {
        "sample_count": 0,
        "non_arm_joint_names": [policy_joint_names[int(index)] for index in non_arm_policy_indices],
        "network_non_arm_mean_abs": [],
        "executed_non_arm_mean_abs": [],
        "target_noise_non_arm_mean_abs": [],
        "ctrl_non_arm_mean_abs": [],
        "network_all_max_abs": [],
        "executed_all_max_abs": [],
        "first_network_action": None,
        "first_executed_action": None,
        "first_target_noise_raw": None,
        "first_ctrl_by_policy_order": None,
    }


def record_policy_application(
    metrics: dict,
    data: mujoco.MjData,
    actuator_ids_by_joint: dict[str, int],
    policy_joint_names: list[str],
    non_arm_policy_indices: np.ndarray,
    network_action: np.ndarray,
    executed_action: np.ndarray,
    target_noise_raw: np.ndarray,
) -> None:
    policy_metrics = metrics.setdefault(
        "policy_application", init_policy_application_metrics(policy_joint_names, non_arm_policy_indices)
    )
    ctrl_by_policy_order = np.asarray(
        [float(data.ctrl[actuator_ids_by_joint[joint_name]]) for joint_name in policy_joint_names], dtype=np.float32
    )
    non_arm_network = network_action[non_arm_policy_indices]
    non_arm_executed = executed_action[non_arm_policy_indices]
    non_arm_noise = target_noise_raw[non_arm_policy_indices]
    non_arm_ctrl = ctrl_by_policy_order[non_arm_policy_indices]

    policy_metrics["sample_count"] += 1
    policy_metrics["network_non_arm_mean_abs"].append(float(np.mean(np.abs(non_arm_network))))
    policy_metrics["executed_non_arm_mean_abs"].append(float(np.mean(np.abs(non_arm_executed))))
    policy_metrics["target_noise_non_arm_mean_abs"].append(float(np.mean(np.abs(non_arm_noise))))
    policy_metrics["ctrl_non_arm_mean_abs"].append(float(np.mean(np.abs(non_arm_ctrl))))
    policy_metrics["network_all_max_abs"].append(float(np.max(np.abs(network_action))))
    policy_metrics["executed_all_max_abs"].append(float(np.max(np.abs(executed_action))))
    if policy_metrics["first_network_action"] is None:
        policy_metrics["first_network_action"] = [float(value) for value in network_action]
        policy_metrics["first_executed_action"] = [float(value) for value in executed_action]
        policy_metrics["first_target_noise_raw"] = [float(value) for value in target_noise_raw]
        policy_metrics["first_ctrl_by_policy_order"] = [float(value) for value in ctrl_by_policy_order]


def make_actuator_id_map(model: mujoco.MjModel, joint_names: list[str]) -> dict[str, int]:
    actuator_ids: dict[str, int] = {}
    requested_joint_names = set(joint_names)
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = _name_from_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name not in requested_joint_names:
            continue
        if joint_name in actuator_ids:
            first_actuator_name = _name_from_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_ids[joint_name])
            second_actuator_name = _name_from_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            raise ValueError(
                f"Joint '{joint_name}' is driven by multiple actuators: "
                f"{first_actuator_name or actuator_ids[joint_name]} and {second_actuator_name or actuator_id}."
            )
        actuator_ids[joint_name] = actuator_id

    missing_joint_names = [joint_name for joint_name in joint_names if joint_name not in actuator_ids]
    if missing_joint_names:
        raise ValueError(f"MuJoCo model has no joint actuator for: {missing_joint_names}")
    return actuator_ids


def _name_from_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, int(object_id))
    return name or ""


def find_floor_geom_ids(model: mujoco.MjModel) -> set[int]:
    floor_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        geom_name = _name_from_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE or geom_name in {"floor", "ground"}:
            floor_geom_ids.add(geom_id)
    return floor_geom_ids


def find_foot_body_ids(model: mujoco.MjModel) -> set[int]:
    foot_body_ids: set[int] = set()
    for body_id in range(model.nbody):
        body_name = _name_from_id(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name.endswith("ankle_roll_link") or body_name.endswith("foot_link"):
            foot_body_ids.add(body_id)
    return foot_body_ids


def contact_force_with_floor(model: mujoco.MjModel, data: mujoco.MjData, floor_geom_ids: set[int], foot_body_ids: set[int]) -> tuple[float, int]:
    total_force = 0.0
    foot_contact_count = 0
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in floor_geom_ids:
            other_geom = geom2
        elif geom2 in floor_geom_ids:
            other_geom = geom1
        else:
            continue

        other_body = int(model.geom_bodyid[other_geom])
        if other_body not in foot_body_ids:
            continue

        mujoco.mj_contactForce(model, data, contact_id, contact_force)
        total_force += float(np.linalg.norm(contact_force[:3]))
        foot_contact_count += 1
    return total_force, foot_contact_count


def foot_bodies_in_contact_with_floor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_ids: set[int],
    foot_body_ids: set[int],
) -> set[int]:
    """Return foot body IDs that currently have at least one floor contact."""
    contact_body_ids: set[int] = set()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in floor_geom_ids:
            other_geom = geom2
        elif geom2 in floor_geom_ids:
            other_geom = geom1
        else:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        if other_body in foot_body_ids:
            contact_body_ids.add(other_body)
    return contact_body_ids


def init_rollout_metrics(
    data: mujoco.MjData,
    torso_body_id: int,
    pelvis_body_id: int,
    ankle_body_ids: tuple[int, int],
    config: dict,
) -> dict:
    torso_pos_w = data.xpos[torso_body_id].copy().astype(np.float32)
    torso_quat_w = data.xquat[torso_body_id].copy().astype(np.float32)
    _, _, initial_yaw = quat_to_roll_pitch_yaw(data.qpos[3:7].copy())
    local_point = np.asarray(config.get("torso_trace_local_point", [0.0, 0.0, 0.18]), dtype=np.float32)
    torso_trace_point_w = torso_fixed_point_w(data, torso_body_id, local_point).astype(np.float32)
    task_trace_point_w = np.array(
        [float(torso_trace_point_w[0]), float(torso_trace_point_w[1]), float(config.get("task_trace_height", 0.05))],
        dtype=np.float32,
    )
    pelvis_pos_w = data.xpos[pelvis_body_id].copy().astype(np.float64)
    _, _, pelvis_yaw = quat_to_roll_pitch_yaw(data.xquat[pelvis_body_id].copy())
    lateral_axis_xy = np.asarray([-math.sin(pelvis_yaw), math.cos(pelvis_yaw)], dtype=np.float64)
    ordered_targets_xy = np.stack(
        [pelvis_pos_w[:2] + 0.15 * lateral_axis_xy, pelvis_pos_w[:2] - 0.15 * lateral_axis_xy]
    )
    initial_ankle_z = np.asarray([data.xpos[body_id, 2] for body_id in ankle_body_ids], dtype=np.float64)
    return {
        "root_heights": [],
        "torso_heights": [],
        "roll_abs": [],
        "pitch_abs": [],
        "lin_vel_xy_errors": [],
        "lateral_vel_errors": [],
        "yaw_rate_errors": [],
        "vertical_vel_errors": [],
        "height_errors": [],
        "ang_vel_xy_errors": [],
        "ang_acc_xy_errors": [],
        "specific_force_xy_errors": [],
        "specific_force_z_errors": [],
        "lin_vel_x": [],
        "lin_vel_y": [],
        "yaw_rate": [],
        "command_samples": [],
        "segment_ids": [],
        "command_segments": [],
        "foot_contact_steps": 0,
        "fallen": False,
        "fall_time": None,
        "prev_torso_pos_w": torso_pos_w,
        "prev_torso_quat_w": torso_quat_w,
        "prev_torso_lin_vel_w": np.zeros(3, dtype=np.float32),
        "prev_torso_ang_vel_b": np.zeros(3, dtype=np.float32),
        "forward_path": 0.0,
        "lateral_path": 0.0,
        "torso_height_target_m": float(torso_pos_w[2]),
        "torso_trace_local_point_m": [float(value) for value in local_point],
        "torso_trace_points": [[0.0, float(torso_trace_point_w[0]), float(torso_trace_point_w[1]), float(torso_trace_point_w[2])]],
        "torso_trace_path_length_m": 0.0,
        "prev_torso_trace_point_w": torso_trace_point_w,
        "task_trace_points": [[0.0, float(task_trace_point_w[0]), float(task_trace_point_w[1]), float(task_trace_point_w[2]), float(initial_yaw)]],
        "task_trace_path_length_m": 0.0,
        "task_trace_point_w": task_trace_point_w,
        "task_trace_yaw": float(initial_yaw),
        "initial_root_height": float(data.qpos[2]),
        "initial_torso_height": float(torso_pos_w[2]),
        "early_motion": {
            "time_s": [],
            "root_height": [],
            "torso_height": [],
            "roll_abs": [],
            "pitch_abs": [],
            "vertical_vel_abs": [],
            "yaw_rate_abs": [],
            "lin_vel_xy_error": [],
            "yaw_rate_error": [],
            "foot_contact_count": [],
            "command": [],
        },
        "step_count": 0,
        "initial_torso_pos_w": torso_pos_w.copy(),
        "initial_torso_yaw_rad": float(initial_yaw),
        "foot_recovery_time_s": [],
        "ankle_distance_m": [],
        "ankle_distance_error_m": [],
        "torso_xy_displacement_m": [],
        "torso_yaw_error_rad": [],
        "ankle_torque_nm": [],
        "lower_body_joint_names": [],
        "lower_body_joint_vel_rad_s": [],
        "lower_body_joint_acc_rad_s2": [],
        "prev_lower_body_joint_vel_rad_s": None,
        "foot_ground_force_n": [],
        "foot_ground_force_rate_n_per_s": [],
        "prev_foot_ground_force_n": None,
        "ordered_foot_steps": {
            "pelvis_reference_xy_m": pelvis_pos_w[:2].tolist(),
            "pelvis_reference_yaw_rad": float(pelvis_yaw),
            "target_xy_m": ordered_targets_xy.tolist(),
            "initial_ankle_z_m": initial_ankle_z.tolist(),
            "max_clearance_m": [0.0, 0.0],
            "left_lifted": False,
            "right_lifted": False,
            "left_airborne_after_lift": False,
            "right_airborne_after_lift": False,
            "phase_start_time_s": 0.0,
            "left_completion_time_s": None,
            "right_completion_time_s": None,
            "right_lifted_before_left": False,
            "final_target_error_m": [0.0, 0.0],
        },
    }


def update_early_motion_metrics(
    metrics: dict,
    config: dict,
    sim_time: float,
    root_height: float,
    torso_height: float,
    roll_abs: float,
    pitch_abs: float,
    vertical_vel_abs: float,
    yaw_rate_abs: float,
    lin_vel_xy_error: float,
    yaw_rate_error: float,
    foot_contact_count: int,
    command: np.ndarray,
) -> None:
    if not bool(config.get("early_motion_enable", True)):
        return
    if sim_time > float(config.get("early_motion_window_s", 1.0)):
        return
    early = metrics["early_motion"]
    early["time_s"].append(float(sim_time))
    early["root_height"].append(float(root_height))
    early["torso_height"].append(float(torso_height))
    early["roll_abs"].append(float(roll_abs))
    early["pitch_abs"].append(float(pitch_abs))
    early["vertical_vel_abs"].append(float(vertical_vel_abs))
    early["yaw_rate_abs"].append(float(yaw_rate_abs))
    early["lin_vel_xy_error"].append(float(lin_vel_xy_error))
    early["yaw_rate_error"].append(float(yaw_rate_error))
    early["foot_contact_count"].append(int(foot_contact_count))
    early["command"].append([float(command[0]), float(command[1]), float(command[2])])


def update_rollout_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metrics: dict,
    qvel_addresses: dict[str, int],
    policy_joint_names: list[str],
    command: np.ndarray,
    floor_geom_ids: set[int],
    foot_body_ids: set[int],
    torso_body_id: int,
    ankle_body_ids: tuple[int, int],
    ankle_actuator_ids: tuple[int, int, int, int],
    config: dict,
    sim_time: float,
    segment_id: int,
) -> None:
    dt = float(model.opt.timestep)
    torso_pos_w = data.xpos[torso_body_id].copy().astype(np.float32)
    torso_quat_w = data.xquat[torso_body_id].copy().astype(np.float32)
    torso_rotation_w = data.xmat[torso_body_id].reshape(3, 3).copy().astype(np.float32)
    roll, pitch, yaw = quat_to_roll_pitch_yaw(torso_quat_w)
    torso_lin_vel_w = (torso_pos_w - metrics["prev_torso_pos_w"]) / dt
    torso_ang_vel_w = quat_delta_to_ang_vel_w(metrics["prev_torso_quat_w"], torso_quat_w, dt)
    torso_lin_vel_yaw_b = yaw_rotation_matrix(float(yaw)).T @ torso_lin_vel_w
    torso_ang_vel_b = torso_rotation_w.T @ torso_ang_vel_w
    torso_lin_acc_w = (torso_lin_vel_w - metrics["prev_torso_lin_vel_w"]) / dt
    torso_ang_acc_b = (torso_ang_vel_b - metrics["prev_torso_ang_vel_b"]) / dt
    specific_force_b = torso_rotation_w.T @ (torso_lin_acc_w - np.array([0.0, 0.0, -9.80665], dtype=np.float32))

    root_height = float(data.qpos[2])
    torso_height = float(torso_pos_w[2])
    roll_abs = abs(float(roll))
    pitch_abs = abs(float(pitch))
    lin_vel_xy_error = float(np.linalg.norm(torso_lin_vel_yaw_b[:2] - command[:2]))
    yaw_rate_error = abs(float(torso_ang_vel_b[2] - command[2]))
    vertical_vel_abs = abs(float(torso_lin_vel_w[2]))
    yaw_rate_abs = abs(float(torso_ang_vel_b[2]))
    metrics["root_heights"].append(root_height)
    metrics["torso_heights"].append(torso_height)
    metrics["roll_abs"].append(roll_abs)
    metrics["pitch_abs"].append(pitch_abs)
    metrics["lin_vel_xy_errors"].append(lin_vel_xy_error)
    metrics["lateral_vel_errors"].append(abs(float(torso_lin_vel_yaw_b[1] - command[1])))
    metrics["yaw_rate_errors"].append(yaw_rate_error)
    metrics["vertical_vel_errors"].append(vertical_vel_abs)
    metrics["height_errors"].append(abs(float(torso_height - metrics["torso_height_target_m"])))
    metrics["ang_vel_xy_errors"].append(float(np.linalg.norm(torso_ang_vel_b[:2])))
    metrics["ang_acc_xy_errors"].append(float(np.linalg.norm(torso_ang_acc_b[:2])))
    metrics["specific_force_xy_errors"].append(float(np.linalg.norm(specific_force_b[:2])))
    metrics["specific_force_z_errors"].append(abs(float(specific_force_b[2] - 9.80665)))
    metrics["lin_vel_x"].append(float(torso_lin_vel_yaw_b[0]))
    metrics["lin_vel_y"].append(float(torso_lin_vel_yaw_b[1]))
    metrics["yaw_rate"].append(float(torso_ang_vel_b[2]))
    metrics["command_samples"].append([float(command[0]), float(command[1]), float(command[2])])
    metrics["segment_ids"].append(int(segment_id))

    ankle_distance = float(
        np.linalg.norm(data.xpos[ankle_body_ids[0], :2] - data.xpos[ankle_body_ids[1], :2])
    )
    target_ankle_distance = float(config.get("target_ankle_distance_m", 0.30))
    metrics["foot_recovery_time_s"].append(float(sim_time))
    metrics["ankle_distance_m"].append(ankle_distance)
    metrics["ankle_distance_error_m"].append(abs(ankle_distance - target_ankle_distance))
    metrics["torso_xy_displacement_m"].append(
        float(np.linalg.norm(torso_pos_w[:2] - metrics["initial_torso_pos_w"][:2]))
    )
    metrics["torso_yaw_error_rad"].append(
        abs(
            math.atan2(
                math.sin(float(yaw - metrics["initial_torso_yaw_rad"])),
                math.cos(float(yaw - metrics["initial_torso_yaw_rad"])),
            )
        )
    )
    metrics["ankle_torque_nm"].append(
        [float(data.actuator_force[actuator_id]) for actuator_id in ankle_actuator_ids]
    )

    lower_body_joint_names = metrics["lower_body_joint_names"]
    if not lower_body_joint_names:
        lower_body_joint_names.extend(
            name
            for name in policy_joint_names
            if any(token in name for token in ("hip_", "knee_", "ankle_"))
        )
    lower_body_joint_vel = np.asarray(
        [float(data.qvel[qvel_addresses[name]]) for name in lower_body_joint_names], dtype=np.float64
    )
    prev_lower_body_joint_vel = metrics["prev_lower_body_joint_vel_rad_s"]
    if prev_lower_body_joint_vel is None:
        lower_body_joint_acc = np.zeros_like(lower_body_joint_vel)
    else:
        lower_body_joint_acc = (lower_body_joint_vel - prev_lower_body_joint_vel) / dt
    metrics["lower_body_joint_vel_rad_s"].append(lower_body_joint_vel.tolist())
    metrics["lower_body_joint_acc_rad_s2"].append(lower_body_joint_acc.tolist())
    metrics["prev_lower_body_joint_vel_rad_s"] = lower_body_joint_vel

    if (
        not metrics["fallen"]
        and (root_height < float(config["healthy_min_root_height"]) or abs(roll) > float(config["healthy_max_roll_pitch"]) or abs(pitch) > float(config["healthy_max_roll_pitch"]))
    ):
        metrics["fallen"] = True
        metrics["fall_time"] = sim_time

    torso_delta_xy = torso_pos_w[:2] - metrics["prev_torso_pos_w"][:2]
    command_norm = float(np.linalg.norm(command[:2]))
    if command_norm > 0.05:
        task_forward_xy = command[:2] / command_norm
    else:
        task_forward_xy = torso_rotation_w[:, 0][:2]
        task_forward_xy = task_forward_xy / max(float(np.linalg.norm(task_forward_xy)), 1.0e-6)
    task_lateral_xy = np.array([-task_forward_xy[1], task_forward_xy[0]], dtype=np.float32)
    metrics["forward_path"] += abs(float(np.dot(torso_delta_xy, task_forward_xy)))
    metrics["lateral_path"] += abs(float(np.dot(torso_delta_xy, task_lateral_xy)))

    foot_ground_force, foot_contact_count = contact_force_with_floor(
        model, data, floor_geom_ids, foot_body_ids
    )
    contacted_foot_body_ids = foot_bodies_in_contact_with_floor(
        model, data, floor_geom_ids, foot_body_ids
    )
    prev_foot_ground_force = metrics["prev_foot_ground_force_n"]
    foot_ground_force_rate = (
        0.0
        if prev_foot_ground_force is None
        else abs(float(foot_ground_force) - float(prev_foot_ground_force)) / dt
    )
    metrics["foot_ground_force_n"].append(float(foot_ground_force))
    metrics["foot_ground_force_rate_n_per_s"].append(float(foot_ground_force_rate))
    metrics["prev_foot_ground_force_n"] = float(foot_ground_force)
    ordered_steps = metrics["ordered_foot_steps"]
    ordered_targets_xy = np.asarray(ordered_steps["target_xy_m"], dtype=np.float64)
    initial_ankle_z = np.asarray(ordered_steps["initial_ankle_z_m"], dtype=np.float64)
    ankle_xy = np.stack([data.xpos[body_id, :2] for body_id in ankle_body_ids])
    ankle_z = np.asarray([data.xpos[body_id, 2] for body_id in ankle_body_ids], dtype=np.float64)
    clearance = ankle_z - initial_ankle_z
    target_error = np.linalg.norm(ankle_xy - ordered_targets_xy, axis=1)
    ordered_steps["max_clearance_m"] = [
        max(float(ordered_steps["max_clearance_m"][index]), float(clearance[index])) for index in range(2)
    ]
    min_clearance_m = float(config.get("ordered_step_min_clearance_m", 0.035))
    min_duration_s = float(config.get("ordered_step_min_duration_s", 0.0))
    if min_clearance_m <= 0.0:
        raise ValueError("ordered_step_min_clearance_m must be positive.")
    if min_duration_s < 0.0:
        raise ValueError("ordered_step_min_duration_s must be non-negative.")
    if float(clearance[0]) >= min_clearance_m:
        ordered_steps["left_lifted"] = True
    if ordered_steps["left_completion_time_s"] is None:
        if ordered_steps["left_lifted"] and ankle_body_ids[0] not in contacted_foot_body_ids:
            ordered_steps["left_airborne_after_lift"] = True
        if float(clearance[1]) >= min_clearance_m:
            ordered_steps["right_lifted_before_left"] = True
        if (
            ordered_steps["left_lifted"]
            and ordered_steps["left_airborne_after_lift"]
            and float(target_error[0]) <= float(config.get("ordered_step_transition_tolerance_m", 0.055))
            and ankle_body_ids[0] in contacted_foot_body_ids
            and float(sim_time - ordered_steps["phase_start_time_s"]) >= min_duration_s
        ):
            ordered_steps["left_completion_time_s"] = float(sim_time)
            ordered_steps["right_lifted"] = False
            ordered_steps["right_airborne_after_lift"] = False
            ordered_steps["phase_start_time_s"] = float(sim_time)
    elif ordered_steps["right_completion_time_s"] is None:
        if float(clearance[1]) >= min_clearance_m:
            ordered_steps["right_lifted"] = True
        if ordered_steps["right_lifted"] and ankle_body_ids[1] not in contacted_foot_body_ids:
            ordered_steps["right_airborne_after_lift"] = True
        if (
            ordered_steps["right_lifted"]
            and ordered_steps["right_airborne_after_lift"]
            and float(target_error[1]) <= float(config.get("ordered_step_transition_tolerance_m", 0.055))
            and ankle_body_ids[1] in contacted_foot_body_ids
            and float(sim_time - ordered_steps["phase_start_time_s"]) >= min_duration_s
        ):
            ordered_steps["right_completion_time_s"] = float(sim_time)
    ordered_steps["final_target_error_m"] = [float(value) for value in target_error]
    if foot_contact_count > 0:
        metrics["foot_contact_steps"] += 1
    update_early_motion_metrics(
        metrics,
        config,
        sim_time,
        root_height,
        torso_height,
        roll_abs,
        pitch_abs,
        vertical_vel_abs,
        yaw_rate_abs,
        lin_vel_xy_error,
        yaw_rate_error,
        foot_contact_count,
        command,
    )

    local_point = np.asarray(config.get("torso_trace_local_point", [0.0, 0.0, 0.18]), dtype=np.float32)
    torso_trace_point_w = torso_fixed_point_w(data, torso_body_id, local_point).astype(np.float32)
    trace_delta = torso_trace_point_w - metrics["prev_torso_trace_point_w"]
    metrics["torso_trace_path_length_m"] += float(np.linalg.norm(trace_delta))
    if bool(config.get("torso_trace_enable", True)) and metrics["step_count"] % max(int(config.get("torso_trace_stride", 10)), 1) == 0:
        metrics["torso_trace_points"].append(
            [float(sim_time), float(torso_trace_point_w[0]), float(torso_trace_point_w[1]), float(torso_trace_point_w[2])]
        )

    previous_task_trace_point = metrics["task_trace_point_w"].copy()
    metrics["task_trace_yaw"] = float(metrics["task_trace_yaw"] + float(command[2]) * dt)
    command_xy_w = yaw_rotation_matrix(float(metrics["task_trace_yaw"]))[:2, :2] @ command[:2]
    metrics["task_trace_point_w"][:2] += command_xy_w.astype(np.float32) * dt
    metrics["task_trace_point_w"][2] = float(config.get("task_trace_height", 0.05))
    metrics["task_trace_path_length_m"] += float(np.linalg.norm(metrics["task_trace_point_w"] - previous_task_trace_point))
    if bool(config.get("task_trace_enable", True)) and metrics["step_count"] % max(int(config.get("task_trace_stride", 10)), 1) == 0:
        point = metrics["task_trace_point_w"]
        metrics["task_trace_points"].append(
            [float(sim_time), float(point[0]), float(point[1]), float(point[2]), float(metrics["task_trace_yaw"])]
        )

    metrics["prev_torso_pos_w"] = torso_pos_w.copy()
    metrics["prev_torso_quat_w"] = torso_quat_w.copy()
    metrics["prev_torso_lin_vel_w"] = torso_lin_vel_w.copy()
    metrics["prev_torso_ang_vel_b"] = torso_ang_vel_b.copy()
    metrics["prev_torso_trace_point_w"] = torso_trace_point_w.copy()
    metrics["step_count"] += 1


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _safe_min(values: list[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _bounded_exp_score(error: float, scale: float) -> float:
    return float(100.0 * np.exp(-max(float(error), 0.0) / max(float(scale), 1.0e-6)))


def summarize_early_motion(metrics: dict, config: dict) -> dict:
    early = metrics.get("early_motion", {})
    time_s = early.get("time_s", [])
    if not time_s:
        return {
            "enabled": bool(config.get("early_motion_enable", True)),
            "window_s": float(config.get("early_motion_window_s", 1.0)),
            "samples": 0,
        }

    contact_counts = np.asarray(early["foot_contact_count"], dtype=np.int32)
    time_values = np.asarray(time_s, dtype=np.float32)
    commands = np.asarray(early["command"], dtype=np.float32)
    first_any_contact = None
    first_no_contact = None
    if np.any(contact_counts > 0):
        first_any_contact = float(time_values[np.argmax(contact_counts > 0)])
    if np.any(contact_counts == 0):
        first_no_contact = float(time_values[np.argmax(contact_counts == 0)])

    return {
        "enabled": bool(config.get("early_motion_enable", True)),
        "window_s": float(config.get("early_motion_window_s", 1.0)),
        "samples": int(len(time_s)),
        "initial_command": [float(value) for value in commands[0]],
        "mean_command": [float(value) for value in np.mean(commands, axis=0)],
        "min_root_height": _safe_min(early["root_height"]),
        "min_torso_height": _safe_min(early["torso_height"]),
        "root_height_drop_m": float(metrics["initial_root_height"] - _safe_min(early["root_height"])),
        "torso_height_drop_m": float(metrics["initial_torso_height"] - _safe_min(early["torso_height"])),
        "max_abs_roll_rad": _safe_max(early["roll_abs"]),
        "max_abs_pitch_rad": _safe_max(early["pitch_abs"]),
        "max_vertical_vel_abs_m_per_s": _safe_max(early["vertical_vel_abs"]),
        "max_yaw_rate_abs_rad_per_s": _safe_max(early["yaw_rate_abs"]),
        "lin_vel_xy_mae_m_per_s": _safe_mean(early["lin_vel_xy_error"]),
        "yaw_rate_mae_rad_per_s": _safe_mean(early["yaw_rate_error"]),
        "foot_contact_duty": float(np.count_nonzero(contact_counts > 0) / max(len(contact_counts), 1)),
        "first_any_foot_contact_time_s": first_any_contact,
        "first_no_foot_contact_time_s": first_no_contact,
    }


def summarize_command_segments(metrics: dict) -> list[dict]:
    segments: list[dict] = []
    segment_ids = np.asarray(metrics["segment_ids"], dtype=np.int32)
    lin_errors = np.asarray(metrics["lin_vel_xy_errors"], dtype=np.float32)
    yaw_errors = np.asarray(metrics["yaw_rate_errors"], dtype=np.float32)
    lin_vel_x = np.asarray(metrics["lin_vel_x"], dtype=np.float32)
    lin_vel_y = np.asarray(metrics["lin_vel_y"], dtype=np.float32)
    yaw_rate = np.asarray(metrics["yaw_rate"], dtype=np.float32)
    command_samples = np.asarray(metrics["command_samples"], dtype=np.float32)
    for segment in metrics["command_segments"]:
        mask = segment_ids == int(segment["id"])
        if not np.any(mask):
            continue
        lin_mae = float(np.mean(lin_errors[mask]))
        yaw_mae = float(np.mean(yaw_errors[mask]))
        tracking_score = 0.7 * _bounded_exp_score(lin_mae, 0.35) + 0.3 * _bounded_exp_score(yaw_mae, 0.50)
        item = {
                "id": int(segment["id"]),
                "start_time": float(segment["start_time"]),
                "command": [float(value) for value in segment["command"]],
                "mean_command": [float(value) for value in np.mean(command_samples[mask], axis=0)],
                "samples": int(np.count_nonzero(mask)),
                "mean_lin_vel_x": float(np.mean(lin_vel_x[mask])),
                "mean_lin_vel_y": float(np.mean(lin_vel_y[mask])),
                "mean_yaw_rate": float(np.mean(yaw_rate[mask])),
                "lin_vel_xy_mae": lin_mae,
                "yaw_rate_mae": yaw_mae,
                "tracking_score": float(tracking_score),
            }
        if "nav2_metadata" in segment:
            item["nav2_metadata"] = dict(segment["nav2_metadata"])
            item["nav2_group_index"] = int(segment.get("nav2_group_index", -1))
            item["nav2_row_start"] = int(segment.get("nav2_row_start", -1))
            item["nav2_row_end"] = int(segment.get("nav2_row_end", -1))
        segments.append(item)
    return segments


def score_rollout(health: dict, tracking: dict, important_metrics: dict, sim_time: float) -> dict:
    if bool(health["healthy"]):
        health_score = 100.0
    elif health["fall_time"] is None:
        health_score = 0.0
    else:
        health_score = 100.0 * max(0.0, min(float(health["fall_time"]) / max(sim_time, 1.0e-6), 1.0))
    lin_score = _bounded_exp_score(float(tracking["lin_vel_xy_mae"]), 0.35)
    yaw_score = _bounded_exp_score(float(tracking["yaw_rate_mae"]), 0.50)
    tracking_score = 0.7 * lin_score + 0.3 * yaw_score
    posture_score = 0.5 * _bounded_exp_score(float(important_metrics["torso_roll_error_rad"]), 0.20) + 0.5 * _bounded_exp_score(
        float(important_metrics["torso_pitch_error_rad"]), 0.20
    )
    imu_score = 0.4 * _bounded_exp_score(float(important_metrics["torso_ang_vel_xy_error_rad_per_s"]), 1.0)
    imu_score += 0.3 * _bounded_exp_score(float(important_metrics["torso_specific_force_xy_error_m_per_s2"]), 8.0)
    imu_score += 0.3 * _bounded_exp_score(float(important_metrics["torso_vertical_vel_error_m_per_s"]), 0.35)
    path_score = _bounded_exp_score(float(important_metrics["torso_lateral_path_ratio"]), 0.5)
    torso_score = 0.4 * posture_score + 0.4 * imu_score + 0.2 * path_score
    contact_score = 100.0 * max(0.0, min(float(health["foot_contact_duty"]) / 0.2, 1.0))
    total_score = 0.30 * health_score + 0.35 * tracking_score + 0.25 * torso_score + 0.10 * contact_score
    return {
        "total_score": float(total_score),
        "health_score": float(health_score),
        "tracking_score": float(tracking_score),
        "torso_score": float(torso_score),
        "contact_score": float(contact_score),
        "lin_vel_score": float(lin_score),
        "yaw_rate_score": float(yaw_score),
    }


def summarize_foot_recovery(metrics: dict, config: dict) -> dict:
    times = np.asarray(metrics.get("foot_recovery_time_s", []), dtype=np.float64)
    distances = np.asarray(metrics.get("ankle_distance_m", []), dtype=np.float64)
    errors = np.asarray(metrics.get("ankle_distance_error_m", []), dtype=np.float64)
    torso_xy = np.asarray(metrics.get("torso_xy_displacement_m", []), dtype=np.float64)
    torso_yaw = np.asarray(metrics.get("torso_yaw_error_rad", []), dtype=np.float64)
    torques = np.asarray(metrics.get("ankle_torque_nm", []), dtype=np.float64)
    lower_body_joint_vel = np.asarray(metrics.get("lower_body_joint_vel_rad_s", []), dtype=np.float64)
    lower_body_joint_acc = np.asarray(metrics.get("lower_body_joint_acc_rad_s2", []), dtype=np.float64)
    foot_ground_force = np.asarray(metrics.get("foot_ground_force_n", []), dtype=np.float64)
    foot_ground_force_rate = np.asarray(metrics.get("foot_ground_force_rate_n_per_s", []), dtype=np.float64)
    enabled = bool(config.get("foot_recovery_enable", False))
    if times.size == 0:
        return {"enabled": enabled, "sample_count": 0, "passed": False}

    tolerance = float(config.get("ankle_distance_tolerance_m", 0.03))
    hold_s = float(config.get("ankle_convergence_hold_s", 0.50))
    settle_s = float(config.get("recovery_settle_time_s", 5.0))
    dt = float(config.get("simulation_dt", 0.002))
    hold_steps = max(int(math.ceil(hold_s / max(dt, 1.0e-9))), 1)
    within = errors <= tolerance
    convergence_time = None
    if hold_steps <= within.size:
        consecutive = np.convolve(within.astype(np.int32), np.ones(hold_steps, dtype=np.int32), mode="valid")
        indices = np.flatnonzero(consecutive == hold_steps)
        if indices.size:
            convergence_time = float(times[int(indices[0])])

    settled_mask = times >= settle_s
    if not np.any(settled_mask):
        settled_mask = np.ones_like(times, dtype=bool)
    torque_abs = np.abs(torques) if torques.size else np.zeros((1, 4), dtype=np.float64)
    smoothness_window_s = float(config.get("recovery_smoothness_window_s", 2.0))
    smoothness_mask = times <= smoothness_window_s
    if not np.any(smoothness_mask):
        smoothness_mask = np.ones_like(times, dtype=bool)
    smooth_vel = lower_body_joint_vel[smoothness_mask]
    smooth_acc = lower_body_joint_acc[smoothness_mask]
    smooth_force = foot_ground_force[smoothness_mask]
    smooth_force_rate = foot_ground_force_rate[smoothness_mask]
    ankle_distance_velocity = np.gradient(distances, times) if times.size > 1 else np.zeros_like(times)
    ankle_distance_acceleration = (
        np.gradient(ankle_distance_velocity, times) if times.size > 2 else np.zeros_like(times)
    )
    smooth_times = times[smoothness_mask]
    velocity_peak_sample, velocity_peak_joint = np.unravel_index(
        np.argmax(np.abs(smooth_vel)), smooth_vel.shape
    )
    acceleration_peak_sample, acceleration_peak_joint = np.unravel_index(
        np.argmax(np.abs(smooth_acc)), smooth_acc.shape
    )
    velocity_peak_sample = int(velocity_peak_sample)
    velocity_peak_joint = int(velocity_peak_joint)
    acceleration_peak_sample = int(acceleration_peak_sample)
    acceleration_peak_joint = int(acceleration_peak_joint)
    ankle_speed_peak_sample = int(np.argmax(np.abs(ankle_distance_velocity[smoothness_mask])))
    force_peak_sample = int(np.argmax(smooth_force))
    max_torso_xy = float(np.max(torso_xy))
    max_torso_yaw = float(np.max(torso_yaw))
    settled_error = float(np.mean(errors[settled_mask]))
    passed = bool(
        enabled
        and convergence_time is not None
        and settled_error <= tolerance
        and max_torso_xy <= float(config.get("foot_recovery_max_torso_xy_m", 0.30))
        and max_torso_yaw <= float(config.get("foot_recovery_max_torso_yaw_rad", 0.35))
    )
    return {
        "enabled": enabled,
        "sample_count": int(times.size),
        "target_distance_m": float(config.get("target_ankle_distance_m", 0.30)),
        "tolerance_m": tolerance,
        "initial_distance_m": float(distances[0]),
        "final_distance_m": float(distances[-1]),
        "final_abs_error_m": float(errors[-1]),
        "mean_abs_error_m": float(np.mean(errors)),
        "settled_mean_abs_error_m": settled_error,
        "within_tolerance_fraction": float(np.mean(within)),
        "convergence_hold_s": hold_s,
        "convergence_time_s": convergence_time,
        "torso_xy_displacement_final_m": float(torso_xy[-1]),
        "torso_xy_displacement_max_m": max_torso_xy,
        "torso_yaw_error_final_rad": float(torso_yaw[-1]),
        "torso_yaw_error_max_rad": max_torso_yaw,
        "ankle_torque_mean_abs_nm": float(np.mean(torque_abs)),
        "ankle_torque_rms_nm": float(np.sqrt(np.mean(np.square(torques)))) if torques.size else 0.0,
        "ankle_torque_max_abs_nm": float(np.max(torque_abs)),
        "smoothness_window_s": smoothness_window_s,
        "lower_body_joint_names": list(metrics.get("lower_body_joint_names", [])),
        "lower_body_joint_velocity_mean_abs_rad_per_s": float(np.mean(np.abs(smooth_vel))),
        "lower_body_joint_velocity_rms_rad_per_s": float(np.sqrt(np.mean(np.square(smooth_vel)))),
        "lower_body_joint_velocity_p95_abs_rad_per_s": float(np.percentile(np.abs(smooth_vel), 95.0)),
        "lower_body_joint_velocity_max_abs_rad_per_s": float(np.max(np.abs(smooth_vel))),
        "lower_body_joint_velocity_peak_time_s": float(smooth_times[velocity_peak_sample]),
        "lower_body_joint_velocity_peak_joint": metrics["lower_body_joint_names"][velocity_peak_joint],
        "lower_body_joint_acceleration_mean_abs_rad_per_s2": float(np.mean(np.abs(smooth_acc))),
        "lower_body_joint_acceleration_rms_rad_per_s2": float(np.sqrt(np.mean(np.square(smooth_acc)))),
        "lower_body_joint_acceleration_p95_abs_rad_per_s2": float(np.percentile(np.abs(smooth_acc), 95.0)),
        "lower_body_joint_acceleration_max_abs_rad_per_s2": float(np.max(np.abs(smooth_acc))),
        "lower_body_joint_acceleration_peak_time_s": float(smooth_times[acceleration_peak_sample]),
        "lower_body_joint_acceleration_peak_joint": metrics["lower_body_joint_names"][acceleration_peak_joint],
        "ankle_distance_velocity_max_abs_m_per_s": float(np.max(np.abs(ankle_distance_velocity[smoothness_mask]))),
        "ankle_distance_velocity_peak_time_s": float(smooth_times[ankle_speed_peak_sample]),
        "ankle_distance_acceleration_max_abs_m_per_s2": float(np.max(np.abs(ankle_distance_acceleration[smoothness_mask]))),
        "foot_ground_force_mean_n": float(np.mean(smooth_force)),
        "foot_ground_force_rms_n": float(np.sqrt(np.mean(np.square(smooth_force)))),
        "foot_ground_force_p95_n": float(np.percentile(smooth_force, 95.0)),
        "foot_ground_force_max_n": float(np.max(smooth_force)),
        "foot_ground_force_peak_time_s": float(smooth_times[force_peak_sample]),
        "foot_ground_force_rate_p95_n_per_s": float(np.percentile(smooth_force_rate, 95.0)),
        "foot_ground_force_rate_max_n_per_s": float(np.max(smooth_force_rate)),
        "passed": passed,
    }


def summarize_ordered_step_dynamics(metrics: dict) -> dict[str, dict[str, float]]:
    """Report impact and torso motion separately for each commanded step."""
    times = np.asarray(metrics.get("foot_recovery_time_s", []), dtype=np.float64)
    if times.size == 0:
        return {}
    torso_xy = np.asarray(metrics.get("torso_xy_displacement_m", []), dtype=np.float64)
    torques = np.asarray(metrics.get("ankle_torque_nm", []), dtype=np.float64)
    joint_vel = np.asarray(metrics.get("lower_body_joint_vel_rad_s", []), dtype=np.float64)
    joint_acc = np.asarray(metrics.get("lower_body_joint_acc_rad_s2", []), dtype=np.float64)
    foot_force = np.asarray(metrics.get("foot_ground_force_n", []), dtype=np.float64)
    ordered = metrics.get("ordered_foot_steps", {})
    left_end = ordered.get("left_completion_time_s")
    right_end = ordered.get("right_completion_time_s")

    def summarize_window(mask: np.ndarray, start_s: float, end_s: float) -> dict[str, float]:
        if not np.any(mask):
            return {}
        window_torso = torso_xy[mask]
        window_torque = torques[mask]
        window_vel = joint_vel[mask]
        window_acc = joint_acc[mask]
        window_force = foot_force[mask]
        return {
            "start_time_s": float(start_s),
            "end_time_s": float(end_s),
            "duration_s": float(max(end_s - start_s, 0.0)),
            "torso_xy_displacement_end_m": float(window_torso[-1]),
            "torso_xy_displacement_max_m": float(np.max(window_torso)),
            "lower_body_joint_velocity_p95_abs_rad_per_s": float(
                np.percentile(np.abs(window_vel), 95.0)
            ),
            "lower_body_joint_velocity_max_abs_rad_per_s": float(np.max(np.abs(window_vel))),
            "lower_body_joint_acceleration_p95_abs_rad_per_s2": float(
                np.percentile(np.abs(window_acc), 95.0)
            ),
            "lower_body_joint_acceleration_max_abs_rad_per_s2": float(np.max(np.abs(window_acc))),
            "foot_ground_force_p95_n": float(np.percentile(window_force, 95.0)),
            "foot_ground_force_max_n": float(np.max(window_force)),
            "ankle_torque_rms_nm": float(np.sqrt(np.mean(np.square(window_torque)))),
            "ankle_torque_max_abs_nm": float(np.max(np.abs(window_torque))),
        }

    result: dict[str, dict[str, float]] = {}
    if left_end is not None:
        left_end = float(left_end)
        result["left_step"] = summarize_window(times <= left_end, 0.0, left_end)
    if left_end is not None and right_end is not None:
        right_end = float(right_end)
        result["right_step"] = summarize_window(
            (times > float(left_end)) & (times <= right_end), float(left_end), right_end
        )
        result["two_step_transition"] = summarize_window(times <= right_end, 0.0, right_end)
    return result


def summarize_rollout_metrics(metrics: dict, sim_time: float, command: np.ndarray, config: dict) -> dict:
    policy_application_metrics = metrics.get("policy_application", {})
    policy_application = {
        "sample_count": int(policy_application_metrics.get("sample_count", 0)),
        "non_arm_joint_names": list(policy_application_metrics.get("non_arm_joint_names", [])),
        "network_non_arm_mean_abs": _safe_mean(policy_application_metrics.get("network_non_arm_mean_abs", [])),
        "executed_non_arm_mean_abs": _safe_mean(policy_application_metrics.get("executed_non_arm_mean_abs", [])),
        "target_noise_non_arm_mean_abs": _safe_mean(
            policy_application_metrics.get("target_noise_non_arm_mean_abs", [])
        ),
        "ctrl_non_arm_mean_abs": _safe_mean(policy_application_metrics.get("ctrl_non_arm_mean_abs", [])),
        "network_all_max_abs": _safe_max(policy_application_metrics.get("network_all_max_abs", [])),
        "executed_all_max_abs": _safe_max(policy_application_metrics.get("executed_all_max_abs", [])),
        "first_network_action": policy_application_metrics.get("first_network_action"),
        "first_executed_action": policy_application_metrics.get("first_executed_action"),
        "first_target_noise_raw": policy_application_metrics.get("first_target_noise_raw"),
        "first_ctrl_by_policy_order": policy_application_metrics.get("first_ctrl_by_policy_order"),
        "non_arm_target_noise_enabled": bool(config.get("non_arm_joint_target_noise_enable", False)),
        "non_arm_target_noise_rad": float(config.get("non_arm_joint_target_noise_rad", 0.0)),
    }
    important_metrics = {
        "torso_roll_error_rad": _safe_mean(metrics["roll_abs"]),
        "torso_pitch_error_rad": _safe_mean(metrics["pitch_abs"]),
        "torso_lin_vel_xy_cmd_error_m_per_s": _safe_mean(metrics["lin_vel_xy_errors"]),
        "torso_lateral_vel_cmd_error_m_per_s": _safe_mean(metrics["lateral_vel_errors"]),
        "torso_yaw_rate_cmd_error_rad_per_s": _safe_mean(metrics["yaw_rate_errors"]),
        "torso_vertical_vel_error_m_per_s": _safe_mean(metrics["vertical_vel_errors"]),
        "torso_height_error_m": _safe_mean(metrics["height_errors"]),
        "torso_ang_vel_xy_error_rad_per_s": _safe_mean(metrics["ang_vel_xy_errors"]),
        "torso_ang_acc_xy_error_rad_per_s2": _safe_mean(metrics["ang_acc_xy_errors"]),
        "torso_specific_force_xy_error_m_per_s2": _safe_mean(metrics["specific_force_xy_errors"]),
        "torso_specific_force_z_error_m_per_s2": _safe_mean(metrics["specific_force_z_errors"]),
        "torso_lateral_path_ratio": float(metrics["lateral_path"] / max(metrics["forward_path"], 1.0e-4)),
    }
    task_tracking = {
        "command_lin_vel_x": float(command[0]),
        "command_lin_vel_y": float(command[1]),
        "command_yaw_rate": float(command[2]),
        "mean_lin_vel_x": _safe_mean(metrics["lin_vel_x"]),
        "mean_lin_vel_y": _safe_mean(metrics["lin_vel_y"]),
        "mean_yaw_rate": _safe_mean(metrics["yaw_rate"]),
        "lin_vel_xy_mae": _safe_mean(metrics["lin_vel_xy_errors"]),
        "lin_vel_xy_max_error": _safe_max(metrics["lin_vel_xy_errors"]),
        "yaw_rate_mae": _safe_mean(metrics["yaw_rate_errors"]),
        "yaw_rate_max_error": _safe_max(metrics["yaw_rate_errors"]),
        "forward_distance": float(metrics["forward_path"]),
        "lateral_distance": float(metrics["lateral_path"]),
    }
    health = {
        "healthy": not bool(metrics["fallen"]),
        "fallen": bool(metrics["fallen"]),
        "fall_time": metrics["fall_time"],
        "min_root_height": _safe_min(metrics["root_heights"]),
        "mean_root_height": _safe_mean(metrics["root_heights"]),
        "min_torso_height": _safe_min(metrics["torso_heights"]),
        "mean_torso_height": _safe_mean(metrics["torso_heights"]),
        "max_abs_roll": _safe_max(metrics["roll_abs"]),
        "max_abs_pitch": _safe_max(metrics["pitch_abs"]),
        "foot_contact_duty": float(metrics["foot_contact_steps"] / max(len(metrics["root_heights"]), 1)),
        "healthy_min_root_height": float(config["healthy_min_root_height"]),
        "healthy_max_roll_pitch": float(config["healthy_max_roll_pitch"]),
    }
    command_samples = np.asarray(metrics["command_samples"], dtype=np.float32)
    if command_samples.size:
        command_mean = np.mean(command_samples, axis=0)
        command_min = np.min(command_samples, axis=0)
        command_max = np.max(command_samples, axis=0)
    else:
        command_mean = command_min = command_max = command
    task_tracking.update(
        {
            "command_mode": str(config.get("command_mode", "independent"))
            if bool(config.get("random_commands", False)) or str(config.get("command_mode", "")).lower() in {"joystick", "nav2"}
            else "fixed",
            "command_ramp": bool(config.get("command_ramp", False)),
            "command_interval": float(config.get("command_interval", 0.0)),
            "mean_command_lin_vel_x": float(command_mean[0]),
            "mean_command_lin_vel_y": float(command_mean[1]),
            "mean_command_yaw_rate": float(command_mean[2]),
            "min_command_lin_vel_x": float(command_min[0]),
            "max_command_lin_vel_x": float(command_max[0]),
            "min_command_lin_vel_y": float(command_min[1]),
            "max_command_lin_vel_y": float(command_max[1]),
            "min_command_yaw_rate": float(command_min[2]),
            "max_command_yaw_rate": float(command_max[2]),
        }
    )
    score = score_rollout(health, task_tracking, important_metrics, sim_time)
    ordered_foot_steps = dict(metrics["ordered_foot_steps"])
    ordered_foot_steps["target_offset_m"] = 0.15
    ordered_foot_steps["min_clearance_m"] = float(config.get("ordered_step_min_clearance_m", 0.035))
    ordered_foot_steps["landing_tolerance_m"] = float(
        config.get("ordered_step_transition_tolerance_m", 0.055)
    )
    ordered_foot_steps["final_target_tolerance_m"] = 0.035
    ordered_foot_steps["left_completed"] = ordered_foot_steps["left_completion_time_s"] is not None
    ordered_foot_steps["right_completed"] = ordered_foot_steps["right_completion_time_s"] is not None
    ordered_foot_steps["order_valid"] = not bool(ordered_foot_steps["right_lifted_before_left"])
    ordered_foot_steps["passed"] = bool(
        ordered_foot_steps["left_completed"]
        and ordered_foot_steps["right_completed"]
        and ordered_foot_steps["order_valid"]
        and max(ordered_foot_steps["final_target_error_m"]) <= 0.035
    )
    ordered_foot_steps["transition_dynamics"] = summarize_ordered_step_dynamics(metrics)
    return {
        "sim_time": float(sim_time),
        "task_tracking": task_tracking,
        "command_segments": summarize_command_segments(metrics),
        "early_motion": summarize_early_motion(metrics, config),
        "important_metrics": important_metrics,
        "health": health,
        "score": score,
        "torso_trace": {
            "body_name": str(config.get("torso_body_name", "torso_link")),
            "local_point_m": metrics["torso_trace_local_point_m"],
            "sample_count": len(metrics["torso_trace_points"]),
            "path_length_m": float(metrics["torso_trace_path_length_m"]),
            "csv_path": "",
        },
        "task_trace": {
            "enabled": bool(config.get("task_trace_enable", True)),
            "sample_count": len(metrics["task_trace_points"]),
            "path_length_m": float(metrics["task_trace_path_length_m"]),
            "height_m": float(config.get("task_trace_height", 0.05)),
            "csv_path": "",
        },
        "joystick": {
            "enabled": str(config.get("command_mode", "")).lower() == "joystick",
            "device": str(config.get("joystick_device", "/dev/input/js0")),
            "axis_lin_x": int(config.get("joystick_axis_lin_x", 1)),
            "axis_lin_y": int(config.get("joystick_axis_lin_y", 0)),
            "axis_yaw": int(config.get("joystick_axis_yaw", 3)),
            "ranges": config.get("joystick_ranges", {}),
            "deadzone": float(config.get("joystick_deadzone", 0.05)),
        },
        "policy_application": policy_application,
        "foot_recovery": summarize_foot_recovery(metrics, config),
        "ordered_foot_steps": ordered_foot_steps,
        "scene": config.get("_scene_report", {}),
    }


def print_rollout_report(report: dict) -> None:
    health = report["health"]
    tracking = report["task_tracking"]
    important = report["important_metrics"]
    score = report["score"]
    print("[METRIC] MuJoCo health:")
    print(
        "  healthy={healthy} fallen={fallen} fall_time={fall_time} min_root_height={min_root_height:.3f} "
        "min_torso_height={min_torso_height:.3f} max_abs_roll={max_abs_roll:.3f} "
        "max_abs_pitch={max_abs_pitch:.3f} foot_contact_duty={foot_contact_duty:.3f}".format(**health)
    )
    print("[METRIC] MuJoCo task tracking:")
    print(
        "  cmd=({command_lin_vel_x:.3f}, {command_lin_vel_y:.3f}, {command_yaw_rate:.3f}) "
        "mean=({mean_lin_vel_x:.3f}, {mean_lin_vel_y:.3f}, {mean_yaw_rate:.3f}) "
        "lin_vel_xy_mae={lin_vel_xy_mae:.3f} yaw_rate_mae={yaw_rate_mae:.3f}".format(**tracking)
    )
    early = report.get("early_motion", {})
    if early.get("samples", 0):
        print("[METRIC] MuJoCo early motion:")
        print(
            "  window={window_s:.2f}s torso_drop={torso_height_drop_m:.4f} root_drop={root_height_drop_m:.4f} "
            "max_roll={max_abs_roll_rad:.4f} max_pitch={max_abs_pitch_rad:.4f} "
            "max_yaw_rate={max_yaw_rate_abs_rad_per_s:.4f}".format(**early)
        )
        print(
            "  lin_mae={lin_vel_xy_mae_m_per_s:.4f} yaw_mae={yaw_rate_mae_rad_per_s:.4f} "
            "contact_duty={foot_contact_duty:.3f} first_no_contact={first_no_foot_contact_time_s}".format(**early)
        )
    print("[METRIC] MuJoCo Important Metrics:")
    print(
        "  torso_roll_error_rad={torso_roll_error_rad:.4f} torso_pitch_error_rad={torso_pitch_error_rad:.4f} "
        "torso_height_error_m={torso_height_error_m:.4f}".format(**important)
    )
    print(
        "  torso_lin_vel_xy_cmd_error_m_per_s={torso_lin_vel_xy_cmd_error_m_per_s:.4f} "
        "torso_lateral_vel_cmd_error_m_per_s={torso_lateral_vel_cmd_error_m_per_s:.4f} "
        "torso_yaw_rate_cmd_error_rad_per_s={torso_yaw_rate_cmd_error_rad_per_s:.4f}".format(**important)
    )
    print(
        "  torso_vertical_vel_error_m_per_s={torso_vertical_vel_error_m_per_s:.4f} "
        "torso_ang_vel_xy_error_rad_per_s={torso_ang_vel_xy_error_rad_per_s:.4f} "
        "torso_ang_acc_xy_error_rad_per_s2={torso_ang_acc_xy_error_rad_per_s2:.4f}".format(**important)
    )
    print(
        "  torso_specific_force_xy_error_m_per_s2={torso_specific_force_xy_error_m_per_s2:.4f} "
        "torso_specific_force_z_error_m_per_s2={torso_specific_force_z_error_m_per_s2:.4f} "
        "torso_lateral_path_ratio={torso_lateral_path_ratio:.4f}".format(**important)
    )
    if report.get("torso_trace"):
        trace = report["torso_trace"]
        print(
            "[METRIC] MuJoCo torso trace: body={body_name} local_point_m={local_point_m} "
            "samples={sample_count} path_length_m={path_length_m:.3f} csv={csv_path}".format(**trace)
        )
    if report.get("task_trace"):
        trace = report["task_trace"]
        print(
            "[METRIC] MuJoCo task trace: enabled={enabled} samples={sample_count} "
            "path_length_m={path_length_m:.3f} height_m={height_m:.3f} csv={csv_path}".format(**trace)
        )
    if report.get("policy_application"):
        policy_application = report["policy_application"]
        print("[METRIC] MuJoCo policy application:")
        print(
            "  samples={sample_count} network_nonarm_abs={network_non_arm_mean_abs:.4f} "
            "executed_nonarm_abs={executed_non_arm_mean_abs:.4f} noise_nonarm_abs={target_noise_non_arm_mean_abs:.4f} "
            "ctrl_nonarm_abs={ctrl_non_arm_mean_abs:.4f}".format(**policy_application)
        )
    if report.get("foot_recovery", {}).get("enabled", False):
        recovery = report["foot_recovery"]
        print("[METRIC] MuJoCo foot recovery:")
        print(
            "  initial={initial_distance_m:.3f}m target={target_distance_m:.3f}m "
            "final={final_distance_m:.3f}m settled_error={settled_mean_abs_error_m:.3f}m "
            "convergence={convergence_time_s} passed={passed}".format(**recovery)
        )
        print(
            "  torso_xy_max={torso_xy_displacement_max_m:.3f}m torso_yaw_max={torso_yaw_error_max_rad:.3f}rad "
            "ankle_torque_mean_abs={ankle_torque_mean_abs_nm:.3f}Nm "
            "ankle_torque_max_abs={ankle_torque_max_abs_nm:.3f}Nm".format(**recovery)
        )
        print(
            "  smooth[{smoothness_window_s:.1f}s] lower_qvel_mean={lower_body_joint_velocity_mean_abs_rad_per_s:.3f} "
            "max={lower_body_joint_velocity_max_abs_rad_per_s:.3f}rad/s@"
            "{lower_body_joint_velocity_peak_time_s:.3f}s({lower_body_joint_velocity_peak_joint}) lower_qacc_p95="
            "{lower_body_joint_acceleration_p95_abs_rad_per_s2:.1f} max="
            "{lower_body_joint_acceleration_max_abs_rad_per_s2:.1f}rad/s2@"
            "{lower_body_joint_acceleration_peak_time_s:.3f}s({lower_body_joint_acceleration_peak_joint})".format(
                **recovery
            )
        )
        print(
            "  ankle_speed_max={ankle_distance_velocity_max_abs_m_per_s:.3f}m/s@"
            "{ankle_distance_velocity_peak_time_s:.3f}s "
            "foot_force_mean={foot_ground_force_mean_n:.1f} p95={foot_ground_force_p95_n:.1f} "
            "max={foot_ground_force_max_n:.1f}N@{foot_ground_force_peak_time_s:.3f}s force_rate_p95="
            "{foot_ground_force_rate_p95_n_per_s:.1f}N/s".format(**recovery)
        )
    if report.get("ordered_foot_steps"):
        ordered = report["ordered_foot_steps"]
        print("[METRIC] MuJoCo ordered foot steps:")
        print(
            "  left_done={left_completed} t_left={left_completion_time_s} "
            "right_done={right_completed} t_right={right_completion_time_s} "
            "order_valid={order_valid} passed={passed}".format(**ordered)
        )
        print(
            "  clearance_left={left:.3f}m clearance_right={right:.3f}m "
            "target_error_left={err_left:.3f}m target_error_right={err_right:.3f}m".format(
                left=float(ordered["max_clearance_m"][0]),
                right=float(ordered["max_clearance_m"][1]),
                err_left=float(ordered["final_target_error_m"][0]),
                err_right=float(ordered["final_target_error_m"][1]),
            )
        )
        transition = ordered.get("transition_dynamics", {}).get("two_step_transition", {})
        if transition:
            print(
                "  two_step[{duration_s:.3f}s] torso_end={torso_xy_displacement_end_m:.3f}m "
                "qvel_p95={lower_body_joint_velocity_p95_abs_rad_per_s:.2f}rad/s "
                "qacc_p95={lower_body_joint_acceleration_p95_abs_rad_per_s2:.1f}rad/s2 "
                "force_max={foot_ground_force_max_n:.1f}N ankle_tau_rms={ankle_torque_rms_nm:.2f}Nm".format(
                    **transition
                )
            )
    print("[METRIC] MuJoCo score:")
    print(
        "  total={total_score:.1f} health={health_score:.1f} tracking={tracking_score:.1f} "
        "torso={torso_score:.1f} contact={contact_score:.1f}".format(**score)
    )
    if report.get("command_segments"):
        print("[METRIC] MuJoCo command segments:")
        for segment in report["command_segments"]:
            command = segment["command"]
            print(
                "  #{id} t={start_time:.1f}s cmd=({cmd_x:.2f}, {cmd_y:.2f}, {cmd_yaw:.2f}) "
                "mean=({mean_lin_vel_x:.2f}, {mean_lin_vel_y:.2f}, {mean_yaw_rate:.2f}) "
                "lin_mae={lin_vel_xy_mae:.3f} yaw_mae={yaw_rate_mae:.3f} score={tracking_score:.1f}".format(
                    id=segment["id"],
                    start_time=segment["start_time"],
                    cmd_x=command[0],
                    cmd_y=command[1],
                    cmd_yaw=command[2],
                    mean_lin_vel_x=segment["mean_lin_vel_x"],
                    mean_lin_vel_y=segment["mean_lin_vel_y"],
                    mean_yaw_rate=segment["mean_yaw_rate"],
                    lin_vel_xy_mae=segment["lin_vel_xy_mae"],
                    yaw_rate_mae=segment["yaw_rate_mae"],
                    tracking_score=segment["tracking_score"],
                )
            )


def _append_trace_spheres(scene, trace_points: np.ndarray, max_points: int, radius: float, color_rgb: np.ndarray) -> None:
    if trace_points.size == 0:
        return
    trace_points = trace_points[-max_points:]
    remaining_geoms = len(scene.geoms) - int(scene.ngeom)
    max_geoms = min(remaining_geoms, len(trace_points))
    if max_geoms <= 0:
        return
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    for point_index in range(max_geoms):
        point = trace_points[point_index, 1:4].astype(np.float64)
        alpha = 0.25 + 0.75 * (point_index + 1) / max(max_geoms, 1)
        rgba = np.array([float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]), alpha], dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            point,
            identity,
            rgba,
        )
        scene.ngeom += 1


def draw_rollout_traces(viewer, metrics: dict, config: dict) -> None:
    try:
        scene = viewer.user_scn
        scene.ngeom = 0
        if bool(config.get("task_trace_enable", True)) and metrics.get("task_trace_points"):
            task_points = np.asarray(metrics["task_trace_points"], dtype=np.float32)
            _append_trace_spheres(
                scene,
                task_points,
                int(config.get("task_trace_max_points", 300)),
                0.022,
                np.array([1.0, 0.55, 0.05], dtype=np.float32),
            )
        if bool(config.get("torso_trace_enable", True)) and metrics.get("torso_trace_points"):
            torso_points = np.asarray(metrics["torso_trace_points"], dtype=np.float32)
            _append_trace_spheres(
                scene,
                torso_points,
                int(config.get("torso_trace_max_points", 300)),
                0.018,
                np.array([0.1, 0.7, 1.0], dtype=np.float32),
            )
    except Exception:
        return


def write_torso_trace_csv(metrics: dict, config: dict) -> str:
    trace_path = str(config.get("torso_trace_path", ""))
    if not trace_path or not metrics.get("torso_trace_points"):
        return ""
    output_path = Path(trace_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["time_s,x_m,y_m,z_m"]
    rows.extend(
        f"{sample[0]:.6f},{sample[1]:.9f},{sample[2]:.9f},{sample[3]:.9f}"
        for sample in metrics["torso_trace_points"]
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(output_path)


def write_task_trace_csv(metrics: dict, config: dict) -> str:
    trace_path = str(config.get("task_trace_path", ""))
    if not trace_path or not metrics.get("task_trace_points"):
        return ""
    output_path = Path(trace_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["time_s,x_m,y_m,z_m,yaw_rad"]
    rows.extend(
        f"{sample[0]:.6f},{sample[1]:.9f},{sample[2]:.9f},{sample[3]:.9f},{sample[4]:.9f}"
        for sample in metrics["task_trace_points"]
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(output_path)


def build_observation(
    data: mujoco.MjData,
    policy_joint_names: list[str],
    qpos_addresses: dict[str, int],
    qvel_addresses: dict[str, int],
    default_angles: np.ndarray,
    action: np.ndarray,
    command: np.ndarray,
    config: dict,
) -> np.ndarray:
    num_actions = len(policy_joint_names)
    obs = np.zeros(9 + 3 * num_actions, dtype=np.float32)

    joint_pos = np.array([data.qpos[qpos_addresses[name]] for name in policy_joint_names], dtype=np.float32)
    joint_vel = np.array([data.qvel[qvel_addresses[name]] for name in policy_joint_names], dtype=np.float32)
    quat = data.qpos[3:7].copy()
    omega = data.qvel[3:6].copy().astype(np.float32)

    obs[0:3] = omega * float(config["ang_vel_scale"])
    obs[3:6] = get_gravity_orientation(quat)
    obs[6:9] = command * np.asarray(config["cmd_scale"], dtype=np.float32)
    obs[9 : 9 + num_actions] = (joint_pos - default_angles) * float(config["dof_pos_scale"])
    obs[9 + num_actions : 9 + 2 * num_actions] = joint_vel * float(config["dof_vel_scale"])
    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
    return obs


def mirror_g1_joint_vector_left_right(values: np.ndarray) -> np.ndarray:
    """Apply the G1 29-DoF left/right reflection used by Isaac training."""
    mirrored = np.zeros_like(values)
    left = np.asarray([0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27], dtype=np.int64)
    right = np.asarray([1, 4, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=np.int64)
    mirrored[left] = values[right]
    mirrored[right] = values[left]
    mirrored[[2, 5, 8]] = values[[2, 5, 8]]
    mirrored[[3, 4, 15, 16, 17, 18, 23, 24]] *= -1.0
    mirrored[[6, 7, 19, 20, 27, 28]] *= -1.0
    mirrored[[2, 5]] *= -1.0
    return mirrored


def mirror_g1_policy_observation_left_right(obs: np.ndarray) -> np.ndarray:
    """Mirror a 96-D G1 policy observation without changing its schema."""
    if obs.shape != (96,):
        raise ValueError(f"Expected a 96-D G1 observation, got shape {obs.shape}.")
    mirrored = obs.copy()
    mirrored[0:3] *= np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
    mirrored[3:6] *= np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
    mirrored[6:9] *= np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
    mirrored[9:38] = mirror_g1_joint_vector_left_right(obs[9:38])
    mirrored[38:67] = mirror_g1_joint_vector_left_right(obs[38:67])
    mirrored[67:96] = mirror_g1_joint_vector_left_right(obs[67:96])
    return mirrored


def apply_pd_control(
    data: mujoco.MjData,
    actuator_joint_names: list[str],
    actuator_ids_by_joint: dict[str, int],
    qpos_addresses: dict[str, int],
    qvel_addresses: dict[str, int],
    target_by_joint: dict[str, float],
    kp_by_joint: dict[str, float],
    kd_by_joint: dict[str, float],
) -> None:
    for joint_name in actuator_joint_names:
        qpos = data.qpos[qpos_addresses[joint_name]]
        qvel = data.qvel[qvel_addresses[joint_name]]
        actuator_id = actuator_ids_by_joint[joint_name]
        data.ctrl[actuator_id] = kp_by_joint[joint_name] * (target_by_joint[joint_name] - qpos) - kd_by_joint[joint_name] * qvel


class JoystickCommandReader:
    """Non-blocking Linux joystick reader for `/dev/input/js*` axis events."""

    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80

    def __init__(self, config: dict):
        self.device_path = str(config.get("joystick_device", "/dev/input/js0"))
        self.axis_max = max(float(config.get("joystick_axis_max", 32768.0)), 1.0)
        self.deadzone = max(float(config.get("joystick_deadzone", 0.05)), 0.0)
        self.axis_values: dict[int, int] = {}
        self.fd = os.open(self.device_path, os.O_RDONLY | os.O_NONBLOCK)
        self.axis_lin_x = int(config.get("joystick_axis_lin_x", 1))
        self.axis_lin_y = int(config.get("joystick_axis_lin_y", 0))
        self.axis_yaw = int(config.get("joystick_axis_yaw", 3))
        self.sign_lin_x = float(config.get("joystick_sign_lin_x", -1.0))
        self.sign_lin_y = float(config.get("joystick_sign_lin_y", 1.0))
        self.sign_yaw = float(config.get("joystick_sign_yaw", 1.0))
        self.ranges = dict(config.get("joystick_ranges", {}))
        print(
            "[INFO] Joystick command mode opened: "
            f"device={self.device_path} axes=(x:{self.axis_lin_x}, y:{self.axis_lin_y}, yaw:{self.axis_yaw})"
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
        return np.array(
            [
                self._map_signed_range(lin_x_unit, self.ranges.get("lin_vel_x", [-0.2, 1.0])),
                self._map_signed_range(lin_y_unit, self.ranges.get("lin_vel_y", [-0.25, 0.25])),
                self._map_signed_range(yaw_unit, self.ranges.get("yaw_rate", [-0.6, 0.6])),
            ],
            dtype=np.float32,
        )


def sample_independent_random_command(rng: np.random.Generator, config: dict) -> np.ndarray:
    ranges = config["command_ranges"]
    return np.array(
        [
            rng.uniform(float(ranges["lin_vel_x"][0]), float(ranges["lin_vel_x"][1])),
            rng.uniform(float(ranges["lin_vel_y"][0]), float(ranges["lin_vel_y"][1])),
            rng.uniform(float(ranges["yaw_rate"][0]), float(ranges["yaw_rate"][1])),
        ],
        dtype=np.float32,
    )


def sample_curvature_random_command(rng: np.random.Generator, config: dict) -> np.ndarray:
    ranges = config["command_ranges"]
    if rng.uniform(0.0, 1.0) <= float(config.get("command_rel_low_speed", 0.25)):
        return np.array(
            [
                rng.uniform(float(ranges["low_speed_lin_vel_x"][0]), float(ranges["low_speed_lin_vel_x"][1])),
                rng.uniform(float(ranges["low_speed_lin_vel_y"][0]), float(ranges["low_speed_lin_vel_y"][1])),
                rng.uniform(float(ranges["low_speed_yaw_rate"][0]), float(ranges["low_speed_yaw_rate"][1])),
            ],
            dtype=np.float32,
        )

    lin_vel_x = rng.uniform(float(ranges["lin_vel_x"][0]), float(ranges["lin_vel_x"][1]))
    curvature = rng.uniform(float(ranges["curvature"][0]), float(ranges["curvature"][1]))
    curvature = float(np.clip(curvature, -float(config.get("command_max_curvature", 0.7)), float(config.get("command_max_curvature", 0.7))))
    yaw_noise = rng.uniform(float(ranges["yaw_noise"][0]), float(ranges["yaw_noise"][1]))
    yaw_rate = np.clip(lin_vel_x * curvature + yaw_noise, float(ranges["yaw_rate"][0]), float(ranges["yaw_rate"][1]))

    lateral_low = max(abs(float(ranges["lin_vel_y"][0])), abs(float(ranges["lin_vel_y"][1])))
    lateral_high = abs(float(config.get("command_high_speed_lateral_vel", 0.06)))
    decay_start = float(config.get("command_lateral_decay_start_speed", 0.25))
    decay_end = float(config.get("command_lateral_decay_end_speed", 0.80))
    blend = np.clip((abs(lin_vel_x) - decay_start) / max(decay_end - decay_start, 1.0e-6), 0.0, 1.0)
    lateral_limit = lateral_low * (1.0 - blend) + lateral_high * blend
    lin_vel_y = np.clip(rng.uniform(-lateral_limit, lateral_limit), float(ranges["lin_vel_y"][0]), float(ranges["lin_vel_y"][1]))
    return np.array([lin_vel_x, lin_vel_y, yaw_rate], dtype=np.float32)


def parse_csv_filter(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "*":
        return None
    parsed = {part.strip() for part in text.split(",") if part.strip()}
    return parsed or None


def estimate_nav2_sample_dt(group_times: list[list[float]], fallback_dt: float) -> float:
    deltas: list[float] = []
    for times in group_times:
        previous = None
        for value in times:
            if previous is not None:
                delta = float(value) - float(previous)
                if 1.0e-4 <= delta <= 0.25:
                    deltas.append(delta)
                    if len(deltas) >= 20000:
                        break
            previous = value
        if len(deltas) >= 20000:
            break
    if not deltas:
        return float(fallback_dt)
    deltas.sort()
    return float(deltas[len(deltas) // 2])


class Nav2CommandReplay:
    """Replay continuous command windows from the Nav2 loopback cmd_vel dataset."""

    def __init__(self, config: dict, rng: np.random.Generator):
        self.rng = rng
        data_path = Path(str(config.get("nav2_data_path", ""))).expanduser()
        if not data_path.is_absolute():
            data_path = UNITREE_ROOT_DIR.parent / data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"Nav2 command dataset not found: {data_path}")
        self.data_path = data_path
        self.command_scale = np.asarray(config.get("nav2_command_scale", [0.70, 0.55, 0.55]), dtype=np.float32)
        self.command_min = np.asarray(config.get("nav2_command_clip_min", [-0.6, -0.3, -0.6]), dtype=np.float32)
        self.command_max = np.asarray(config.get("nav2_command_clip_max", [0.6, 0.3, 0.6]), dtype=np.float32)
        self.groups = self._load_groups(config)
        self.sample_dt = estimate_nav2_sample_dt(
            [group["times"] for group in self.groups], float(config.get("nav2_dataset_sample_dt", 0.05))
        )
        window_duration = float(config.get("nav2_window_duration_s", 0.0))
        if window_duration <= 0.0:
            window_duration = float(config.get("simulation_duration", 20.0))
        self.window_rows = max(1, int(math.ceil(window_duration / max(self.sample_dt, 1.0e-6))))
        self.group_index = 0
        self.row_index = 0
        self.row_end = 1
        self.row_elapsed = 0.0
        self.active_group = self.groups[0]

    def _load_groups(self, config: dict) -> list[dict]:
        augmentation_set = parse_csv_filter(str(config.get("nav2_augmentation_filter", "none,mirror_lr")))
        family_set = parse_csv_filter(str(config.get("nav2_scenario_family_filter", "")))
        combo_set = parse_csv_filter(str(config.get("nav2_combo_filter", "")))
        controller_set = parse_csv_filter(str(config.get("nav2_controller_filter", "")))
        planner_set = parse_csv_filter(str(config.get("nav2_planner_filter", "")))
        grouped_rows: dict[tuple[str, str, str, str, str, str, str], list[tuple[float, tuple[float, float, float]]]] = {}
        metadata_by_key: dict[tuple[str, str, str, str, str, str, str], dict[str, str]] = {}
        raw_rows = 0
        kept_rows = 0
        with open(self.data_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"vx", "vy", "wz", "combo", "planner", "controller", "scenario", "goal_id"}
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"Nav2 command dataset missing columns: {sorted(missing_columns)}")
            for row in reader:
                raw_rows += 1
                augmentation = row.get("augmentation", "none") or "none"
                family = row.get("scenario_family", "unknown") or "unknown"
                combo = row.get("combo", "unknown") or "unknown"
                controller = row.get("controller", "unknown") or "unknown"
                planner = row.get("planner", "unknown") or "unknown"
                if augmentation_set is not None and augmentation not in augmentation_set:
                    continue
                if family_set is not None and family not in family_set:
                    continue
                if combo_set is not None and combo not in combo_set:
                    continue
                if controller_set is not None and controller not in controller_set:
                    continue
                if planner_set is not None and planner not in planner_set:
                    continue
                key = (combo, planner, controller, row.get("scenario", "unknown"), row.get("goal_id", "unknown"), augmentation, family)
                group = grouped_rows.setdefault(key, [])
                if key not in metadata_by_key:
                    metadata_by_key[key] = {
                        "combo": combo,
                        "planner": planner,
                        "controller": controller,
                        "scenario": row.get("scenario", "unknown"),
                        "goal_id": row.get("goal_id", "unknown"),
                        "scenario_family": family,
                        "augmentation": augmentation,
                    }
                time_text = row.get("t", "")
                time_value = float(time_text) if time_text else float(len(group)) * float(config.get("nav2_dataset_sample_dt", 0.05))
                group.append((time_value, (float(row["vx"]), float(row["vy"]), float(row["wz"]))))
                kept_rows += 1
        groups: list[dict] = []
        for key in sorted(grouped_rows):
            rows = sorted(grouped_rows[key], key=lambda item: item[0])
            if not rows:
                continue
            groups.append(
                {
                    "commands": np.asarray([item[1] for item in rows], dtype=np.float32),
                    "times": [float(item[0]) for item in rows],
                    "metadata": metadata_by_key[key],
                }
            )
        if not groups:
            raise ValueError(f"Nav2 command dataset has no rows after filtering: {self.data_path}")
        lengths = [int(group["commands"].shape[0]) for group in groups]
        print(
            "[INFO] Nav2 command replay loaded: "
            f"path={self.data_path} rows={kept_rows}/{raw_rows} groups={len(groups)} "
            f"group_len[min/median/max]={min(lengths)}/{int(np.median(lengths))}/{max(lengths)}"
        )
        return groups

    def _scaled_command(self, command: np.ndarray) -> np.ndarray:
        return np.clip(command * self.command_scale, self.command_min, self.command_max).astype(np.float32)

    def sample_window(self, sim_time: float) -> tuple[np.ndarray, dict]:
        self.group_index = int(self.rng.integers(0, len(self.groups)))
        self.active_group = self.groups[self.group_index]
        length = int(self.active_group["commands"].shape[0])
        max_offset = max(length - self.window_rows + 1, 1)
        self.row_index = int(self.rng.integers(0, max_offset)) if max_offset > 1 else 0
        self.row_end = min(length, self.row_index + self.window_rows)
        self.row_elapsed = 0.0
        command = self.current_command()
        segment_info = {
            "start_time": float(sim_time),
            "command": [float(command[0]), float(command[1]), float(command[2])],
            "nav2_group_index": int(self.group_index),
            "nav2_row_start": int(self.row_index),
            "nav2_row_end": int(self.row_end),
            "nav2_metadata": dict(self.active_group["metadata"]),
        }
        return command, segment_info

    def current_command(self) -> np.ndarray:
        return self._scaled_command(self.active_group["commands"][self.row_index])

    def update(self, dt: float, sim_time: float):
        self.row_elapsed += float(dt)
        advance = int(math.floor(self.row_elapsed / max(self.sample_dt, 1.0e-6)))
        if advance <= 0:
            return self.current_command(), None
        self.row_elapsed -= float(advance) * self.sample_dt
        next_row = self.row_index + advance
        if next_row >= self.row_end:
            return self.sample_window(sim_time)
        self.row_index = next_row
        return self.current_command(), None


def sample_random_command(rng: np.random.Generator, config: dict) -> np.ndarray:
    if str(config.get("command_mode", "independent")).lower() == "curvature":
        return sample_curvature_random_command(rng, config)
    return sample_independent_random_command(rng, config)


def smooth_command(current_command: np.ndarray, target_command: np.ndarray, dt: float, config: dict) -> np.ndarray:
    if not bool(config.get("command_ramp", False)):
        return target_command.astype(np.float32)
    tau = float(config.get("command_smoothing_tau", 0.25))
    if tau > 0.0:
        alpha = min(max(float(dt) / tau, 0.0), 1.0)
        delta = (target_command - current_command) * alpha
    else:
        delta = target_command - current_command
    max_linear_delta = max(float(config.get("command_max_linear_accel", 1.2)) * float(dt), 0.0)
    max_yaw_delta = max(float(config.get("command_max_yaw_accel", 1.5)) * float(dt), 0.0)
    if max_linear_delta > 0.0:
        delta[:2] = np.clip(delta[:2], -max_linear_delta, max_linear_delta)
    if max_yaw_delta > 0.0:
        delta[2] = np.clip(delta[2], -max_yaw_delta, max_yaw_delta)
    return (current_command + delta).astype(np.float32)


def reset_pose_recovery_command(
    data: mujoco.MjData,
    reference_xy_w: np.ndarray,
    reference_yaw_w: float,
    config: dict,
) -> np.ndarray:
    """Return reset-relative [dx_b, dy_b, dyaw] with Isaac command semantics."""
    _, _, current_yaw = quat_to_roll_pitch_yaw(data.qpos[3:7].copy())
    delta_w = np.asarray(reference_xy_w, dtype=np.float64) - data.qpos[:2]
    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)
    command = np.asarray(
        [
            cos_yaw * delta_w[0] + sin_yaw * delta_w[1],
            -sin_yaw * delta_w[0] + cos_yaw * delta_w[1],
            (float(reference_yaw_w) - current_yaw + math.pi) % (2.0 * math.pi) - math.pi,
        ],
        dtype=np.float32,
    )
    command[:2] *= float(config.get("position_recovery_command_xy_gain", 2.0))
    command[2] *= float(config.get("position_recovery_command_yaw_gain", 1.5))
    xy_clip = max(float(config.get("position_recovery_command_xy_clip_m", 0.50)), 0.0)
    yaw_clip = max(float(config.get("position_recovery_command_yaw_clip_rad", 0.60)), 0.0)
    command[:2] = np.clip(command[:2], -xy_clip, xy_clip)
    command[2] = np.clip(command[2], -yaw_clip, yaw_clip)
    return command


def run_mujoco(config: dict) -> None:
    policy_joint_names = list(config["policy_joint_names"])
    actuator_joint_names = list(config["actuator_joint_names"])
    if set(policy_joint_names) != set(actuator_joint_names):
        raise ValueError("policy_joint_names and actuator_joint_names must contain the same joint names.")

    default_angles = np.asarray(config["default_angles"], dtype=np.float32)
    kps = np.asarray(config["kps"], dtype=np.float32)
    kds = np.asarray(config["kds"], dtype=np.float32)
    armhack_stand = (
        ArmHackStandReplay(config, policy_joint_names, default_angles)
        if bool(config.get("armhack_stand_enable", False))
        else None
    )
    if armhack_stand is not None:
        if bool(config.get("random_commands", False)) or str(config.get("command_mode", "independent")).lower() != "independent":
            raise ValueError("ArmHack Stand MuJoCo replay requires independent commands.")
        if not np.allclose(np.asarray(config["cmd_init"], dtype=np.float32), 0.0, atol=1.0e-8):
            raise ValueError("ArmHack Stand MuJoCo replay requires cmd_init=[0, 0, 0].")
    rng = np.random.default_rng(int(config.get("command_seed", 1)))
    interactive_stance_rng = np.random.default_rng(int(config.get("interactive_stance_seed", 20260814)))
    interactive_reset_requested = threading.Event()
    interactive_arm_pose_requested = threading.Event()
    target_noise_rng = np.random.default_rng(int(config.get("non_arm_joint_target_noise_seed", 20260719)))
    command_mode = str(config.get("command_mode", "independent")).lower()
    joystick = JoystickCommandReader(config) if command_mode == "joystick" else None
    nav2_replay = Nav2CommandReplay(config, rng) if command_mode == "nav2" else None
    target_command = np.asarray(config["cmd_init"], dtype=np.float32)
    nav2_segment_info = None
    if joystick is not None:
        target_command = joystick.read_command()
    elif nav2_replay is not None:
        target_command, nav2_segment_info = nav2_replay.sample_window(0.0)
    elif bool(config.get("random_commands", False)):
        target_command = sample_random_command(rng, config)
    command = np.zeros(3, dtype=np.float32) if bool(config.get("command_ramp", False)) else target_command.copy()
    action = np.zeros(len(policy_joint_names), dtype=np.float32)
    last_network_action = np.zeros(len(policy_joint_names), dtype=np.float32)
    phase_one_canonical_action_history = np.zeros(len(policy_joint_names), dtype=np.float32)
    hold_policy_action_history = np.zeros(len(policy_joint_names), dtype=np.float32)
    completion_latched_action = None
    last_target_noise_raw = np.zeros(len(policy_joint_names), dtype=np.float32)
    arm_joint_names = set(ARM_JOINT_NAMES)
    non_arm_policy_indices = np.asarray(
        [index for index, joint_name in enumerate(policy_joint_names) if joint_name not in arm_joint_names],
        dtype=np.int64,
    )

    kp_by_joint = dict(zip(policy_joint_names, kps))
    kd_by_joint = dict(zip(policy_joint_names, kds))
    default_by_joint = dict(zip(policy_joint_names, default_angles))

    scene_xml_path = ensure_floor_xml(config["xml_path"], config)
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = float(config["simulation_dt"])
    qpos_addresses, qvel_addresses = make_joint_address_maps(model, actuator_joint_names)
    actuator_ids_by_joint = make_actuator_id_map(model, actuator_joint_names)
    floor_geom_ids = find_floor_geom_ids(model)
    foot_body_ids = find_foot_body_ids(model)
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(config.get("torso_body_name", "torso_link")))
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_ankle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    ankle_joint_names = (
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    )
    ankle_actuator_ids = tuple(actuator_ids_by_joint[name] for name in ankle_joint_names)
    if not floor_geom_ids:
        raise RuntimeError("MuJoCo model has no floor/ground plane; set add_floor: true or provide a scene XML with a floor.")
    if not foot_body_ids:
        raise RuntimeError("MuJoCo model has no ankle_roll_link/foot_link bodies for contact metrics.")
    if torso_body_id < 0:
        raise RuntimeError(f"MuJoCo model has no torso body named '{config.get('torso_body_name', 'torso_link')}'.")
    if pelvis_body_id < 0:
        raise RuntimeError("MuJoCo model has no pelvis body for ordered foot-step targets.")
    if left_ankle_body_id < 0 or right_ankle_body_id < 0:
        raise RuntimeError("MuJoCo model is missing left/right ankle_roll_link bodies.")

    data.qpos[0:3] = np.asarray(config["root_pos_init"], dtype=np.float32)
    data.qpos[3:7] = np.asarray(config["root_quat_init"], dtype=np.float32)
    for joint_name, default_angle in default_by_joint.items():
        data.qpos[qpos_addresses[joint_name]] = default_angle
    if armhack_stand is not None:
        armhack_stand.initialize_model_and_state(mujoco, model, data, qpos_addresses, torso_body_id)
    else:
        mujoco.mj_forward(model, data)
    foot_recovery_initialization = apply_initial_foot_recovery_stance(
        model, data, qpos_addresses, config
    )
    joint_randomization_report = apply_initial_joint_randomization(
        model,
        data,
        policy_joint_names,
        qpos_addresses,
        qvel_addresses,
        config,
    )
    if joint_randomization_report.get("enabled", False):
        mujoco.mj_forward(model, data)
        print(
            "[INFO] Applied initial joint randomization: "
            f"seed={joint_randomization_report['seed']} "
            f"pos=+/-{joint_randomization_report['pos_noise_rad']}rad "
            f"vel=+/-{joint_randomization_report['vel_noise_rad_per_s']}rad/s "
            f"joints={joint_randomization_report['joint_count']}"
        )
    rollout_metrics = init_rollout_metrics(
        data,
        torso_body_id,
        pelvis_body_id,
        (left_ankle_body_id, right_ankle_body_id),
        config,
    )
    recovery_reference_xy_w = data.qpos[:2].copy()
    _, _, recovery_reference_yaw_w = quat_to_roll_pitch_yaw(data.qpos[3:7].copy())
    rollout_metrics["policy_application"] = init_policy_application_metrics(policy_joint_names, non_arm_policy_indices)
    current_segment_id = 0
    next_command_time = max(float(config.get("command_interval", 2.0)), float(model.opt.timestep))
    initial_segment = {
        "id": current_segment_id,
        "start_time": 0.0,
        "command": [float(target_command[0]), float(target_command[1]), float(target_command[2])],
    }
    if nav2_segment_info is not None:
        initial_segment.update(nav2_segment_info)
    rollout_metrics["command_segments"].append(initial_segment)

    policy = PolicyRunner(config["policy_path"], str(config.get("policy_runtime", "auto")))
    hold_policy_path = str(config.get("ordered_step_hold_policy_path", "")).strip()
    hold_policy = (
        PolicyRunner(hold_policy_path, str(config.get("policy_runtime", "auto")))
        if hold_policy_path
        else None
    )
    push_scheduler = MujocoPushScheduler(config, torso_body_id)

    def sample_interactive_stance_distance() -> float:
        distance_range = np.asarray(config.get("interactive_stance_distance_range_m", [0.08, 0.48]), dtype=float)
        if distance_range.shape != (2,) or distance_range[0] < 0.05 or distance_range[1] > 0.60:
            raise ValueError("interactive_stance_distance_range_m must be [min, max] within [0.05, 0.60].")
        if distance_range[0] >= distance_range[1]:
            raise ValueError("interactive_stance_distance_range_m requires min < max.")
        draw = float(interactive_stance_rng.random())
        if draw < 0.50:
            low = max(float(distance_range[0]), 0.08)
            high = min(float(distance_range[1]), 0.14)
        elif draw < 0.60:
            low = max(float(distance_range[0]), 0.28)
            high = min(float(distance_range[1]), 0.32)
        else:
            low, high = (float(value) for value in distance_range)
        if low >= high:
            low, high = (float(value) for value in distance_range)
        return float(interactive_stance_rng.uniform(low, high))

    def reset_to_interactive_stance(requested_distance_m: float) -> dict:
        nonlocal action, last_network_action, last_target_noise_raw, phase_one_canonical_action_history
        nonlocal hold_policy_action_history, completion_latched_action
        nonlocal recovery_reference_xy_w, recovery_reference_yaw_w
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = np.asarray(config["root_pos_init"], dtype=np.float32)
        data.qpos[3:7] = np.asarray(config["root_quat_init"], dtype=np.float32)
        for joint_name, default_angle in default_by_joint.items():
            data.qpos[qpos_addresses[joint_name]] = default_angle
        if armhack_stand is not None:
            for joint_name, target in zip(ARM_JOINT_NAMES, armhack_stand.csv_targets[0], strict=True):
                data.qpos[qpos_addresses[joint_name]] = float(target)
        stance_config = dict(config)
        stance_config["initial_ankle_distance_m"] = requested_distance_m
        result = apply_initial_foot_recovery_stance(model, data, qpos_addresses, stance_config)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        action = np.zeros(len(policy_joint_names), dtype=np.float32)
        last_network_action = np.zeros(len(policy_joint_names), dtype=np.float32)
        phase_one_canonical_action_history = np.zeros(len(policy_joint_names), dtype=np.float32)
        hold_policy_action_history = np.zeros(len(policy_joint_names), dtype=np.float32)
        completion_latched_action = None
        last_target_noise_raw = np.zeros(len(policy_joint_names), dtype=np.float32)
        mujoco.mj_forward(model, data)
        recovery_reference_xy_w = data.qpos[:2].copy()
        _, _, recovery_reference_yaw_w = quat_to_roll_pitch_yaw(data.qpos[3:7].copy())
        pelvis_pos_w = data.xpos[pelvis_body_id].copy()
        _, _, pelvis_yaw = quat_to_roll_pitch_yaw(data.xquat[pelvis_body_id].copy())
        lateral_axis = np.asarray([-math.sin(pelvis_yaw), math.cos(pelvis_yaw)], dtype=np.float64)
        ordered_targets_xy = np.stack(
            [pelvis_pos_w[:2] + 0.15 * lateral_axis, pelvis_pos_w[:2] - 0.15 * lateral_axis]
        )
        initial_ankle_z = [float(data.xpos[left_ankle_body_id, 2]), float(data.xpos[right_ankle_body_id, 2])]
        rollout_metrics["ordered_foot_steps"] = {
            "pelvis_reference_xy_m": pelvis_pos_w[:2].tolist(),
            "pelvis_reference_yaw_rad": float(pelvis_yaw),
            "target_xy_m": ordered_targets_xy.tolist(),
            "initial_ankle_z_m": initial_ankle_z,
            "max_clearance_m": [0.0, 0.0],
            "left_lifted": False,
            "right_lifted": False,
            "left_completion_time_s": None,
            "right_completion_time_s": None,
            "right_lifted_before_left": False,
            "final_target_error_m": [0.0, 0.0],
        }
        if armhack_stand is not None:
            armhack_stand.torso_reference = armhack_stand._torso_pose(data, torso_body_id)
            armhack_stand.reset_interactive_timebase(0.0)
        return result

    def step_policy_if_needed(counter: int, sim_time: float) -> np.ndarray:
        nonlocal last_network_action, phase_one_canonical_action_history
        nonlocal hold_policy_action_history, completion_latched_action
        if counter % int(config["control_decimation"]) != 0:
            return action
        ordered = rollout_metrics["ordered_foot_steps"]
        if hold_policy is not None and ordered["right_completion_time_s"] is not None:
            hold_obs = build_observation(
                data,
                policy_joint_names,
                qpos_addresses,
                qvel_addresses,
                default_angles,
                action,
                command,
                config,
            )
            hold_action_obs_start = 9 + 2 * len(policy_joint_names)
            hold_obs[hold_action_obs_start : hold_action_obs_start + len(policy_joint_names)] = (
                hold_policy_action_history
            )
            hold_obs[hold_action_obs_start + 27] = -1.0
            hold_obs[hold_action_obs_start + 28] = 0.0
            hold_action = hold_policy.infer(hold_obs)
            hold_policy_action_history = hold_action.copy()
            if completion_latched_action is None:
                completion_latched_action = action.copy()
            blend_duration_s = max(float(config.get("ordered_step_hold_blend_duration_s", 1.0)), 1.0e-6)
            blend_alpha = float(
                np.clip(
                    (sim_time - float(ordered["right_completion_time_s"])) / blend_duration_s,
                    0.0,
                    1.0,
                )
            )
            next_action = (
                (1.0 - blend_alpha) * completion_latched_action + blend_alpha * hold_action
            ).astype(np.float32)
            last_network_action = next_action.copy()
            if armhack_stand is not None:
                next_action = armhack_stand.compose_action(next_action, sim_time)
            return next_action
        if (
            bool(config.get("ordered_step_hold_last_action_enable", False))
            and ordered["right_completion_time_s"] is not None
        ):
            # The phase-zero stepping skill has no trained phase-two input.
            # Latch the successful landing target instead of invoking it again
            # with an out-of-distribution completion signal.
            next_action = action.copy()
            last_network_action = next_action.copy()
            if armhack_stand is not None:
                next_action = armhack_stand.compose_action(next_action, sim_time)
            return next_action
        obs = build_observation(
            data,
            policy_joint_names,
            qpos_addresses,
            qvel_addresses,
            default_angles,
            action,
            command,
            config,
        )
        phase_one_active = (
            ordered["left_completion_time_s"] is not None
            and ordered["right_completion_time_s"] is None
        )
        mirror_phase_one = bool(config.get("ordered_step_mirror_policy_enable", False)) and phase_one_active
        if mirror_phase_one:
            obs = mirror_g1_policy_observation_left_right(obs)
            # Phase one reuses a phase-zero skill as a fresh mirrored episode.
            # Do not leak the just-completed left-step action history into the
            # right-step policy state; maintain its own canonical history.
            action_obs_start = 9 + 2 * len(policy_joint_names)
            obs[action_obs_start : action_obs_start + len(policy_joint_names)] = (
                phase_one_canonical_action_history
            )
            if bool(config.get("ordered_step_observation_enable", False)):
                # The mirrored controller deliberately reuses the learned
                # phase-zero (active-left-foot) skill for the physical right
                # foot.  Keep its two phase-conditioning slots in the same
                # distribution as phase-zero training after mirroring.
                action_obs_start = 9 + 2 * len(policy_joint_names)
                obs[action_obs_start + 27] = 0.0
                obs[action_obs_start + 28] = float(bool(ordered["right_lifted"]))
        elif bool(config.get("ordered_step_observation_enable", False)):
            if ordered["right_completion_time_s"] is not None:
                phase_signal = -1.0
                lifted_signal = 0.0
            elif ordered["left_completion_time_s"] is not None:
                phase_signal = 1.0
                lifted_signal = float(bool(ordered["right_lifted"]))
            else:
                phase_signal = 0.0
                lifted_signal = float(bool(ordered["left_lifted"]))
            action_obs_start = 9 + 2 * len(policy_joint_names)
            obs[action_obs_start + 27] = phase_signal
            obs[action_obs_start + 28] = lifted_signal
        next_action = policy.infer(obs)
        if mirror_phase_one:
            phase_one_canonical_action_history = next_action.copy()
            next_action = mirror_g1_joint_vector_left_right(next_action)
        smoothing_alpha = float(config.get("ordered_step_action_smoothing_alpha", 1.0))
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("ordered_step_action_smoothing_alpha must be in (0, 1].")
        if smoothing_alpha < 1.0:
            smoothed_action = next_action.copy()
            smoothed_action[non_arm_policy_indices] = (
                action[non_arm_policy_indices]
                + smoothing_alpha
                * (next_action[non_arm_policy_indices] - action[non_arm_policy_indices])
            )
            next_action = smoothed_action
        last_network_action = next_action.copy()
        if armhack_stand is not None:
            next_action = armhack_stand.compose_action(next_action, sim_time)
        return next_action

    def simulate_loop(viewer=None) -> None:
        nonlocal action, command, target_command, current_segment_id, next_command_time, last_target_noise_raw
        counter = 0
        sim_time = 0.0
        wall_start = time.time()
        try:
            while sim_time < float(config["simulation_duration"]):
                if viewer is not None and not viewer.is_running():
                    break
                if interactive_reset_requested.is_set():
                    interactive_reset_requested.clear()
                    requested_distance = sample_interactive_stance_distance()
                    reset_report = reset_to_interactive_stance(requested_distance)
                    counter = 0
                    sim_time = 0.0
                    print(
                        "[INTERACTIVE] Reset stance: "
                        f"requested={reset_report['requested_distance_m']:.3f}m "
                        f"actual={reset_report['actual_distance_m']:.3f}m"
                    )
                if interactive_arm_pose_requested.is_set() and armhack_stand is not None:
                    interactive_arm_pose_requested.clear()
                    pose_report = armhack_stand.cycle_interactive_pose(sim_time)
                    print(
                        "[INTERACTIVE] Arm pose: "
                        f"{pose_report['pose_number']}/{pose_report['pose_count']} "
                        f"source_t={pose_report['csv_time_s']:.2f}s "
                        f"transition={pose_report['transition_duration_s']:.2f}s"
                    )
                step_start = time.time()
                if joystick is not None:
                    target_command = joystick.read_command()
                elif nav2_replay is not None:
                    target_command, nav2_segment_info = nav2_replay.update(float(model.opt.timestep), sim_time)
                    if nav2_segment_info is not None:
                        current_segment_id += 1
                        segment = {"id": current_segment_id}
                        segment.update(nav2_segment_info)
                        rollout_metrics["command_segments"].append(segment)
                elif bool(config.get("random_commands", False)) and sim_time >= next_command_time:
                    target_command = sample_random_command(rng, config)
                    current_segment_id += 1
                    rollout_metrics["command_segments"].append(
                        {
                            "id": current_segment_id,
                            "start_time": float(sim_time),
                            "command": [float(target_command[0]), float(target_command[1]), float(target_command[2])],
                        }
                    )
                    next_command_time += max(float(config.get("command_interval", 2.0)), float(model.opt.timestep))
                if bool(config.get("position_recovery_command_enable", False)):
                    target_command = reset_pose_recovery_command(
                        data, recovery_reference_xy_w, recovery_reference_yaw_w, config
                    )
                command = smooth_command(command, target_command, float(model.opt.timestep), config)
                control_step = counter % int(config["control_decimation"]) == 0
                action = step_policy_if_needed(counter, sim_time)
                if control_step:
                    action, last_target_noise_raw = apply_non_arm_target_noise(
                        action,
                        non_arm_policy_indices,
                        target_noise_rng,
                        config,
                    )
                target_policy = default_angles + action * float(config["action_scale"])
                target_by_joint = dict(zip(policy_joint_names, target_policy))
                apply_pd_control(
                    data,
                    actuator_joint_names,
                    actuator_ids_by_joint,
                    qpos_addresses,
                    qvel_addresses,
                    target_by_joint,
                    kp_by_joint,
                    kd_by_joint,
                )
                if control_step:
                    record_policy_application(
                        rollout_metrics,
                        data,
                        actuator_ids_by_joint,
                        policy_joint_names,
                        non_arm_policy_indices,
                        last_network_action,
                        action,
                        last_target_noise_raw,
                    )
                push_scheduler.apply(data, sim_time)
                mujoco.mj_step(model, data)
                counter += 1
                sim_time += model.opt.timestep
                if armhack_stand is not None and control_step:
                    armhack_stand.record_control_sample(
                        data,
                        qpos_addresses,
                        torso_body_id,
                        sim_time,
                    )
                update_rollout_metrics(
                    model,
                    data,
                    rollout_metrics,
                    qvel_addresses,
                    policy_joint_names,
                    command,
                    floor_geom_ids,
                    foot_body_ids,
                    torso_body_id,
                    (left_ankle_body_id, right_ankle_body_id),
                    ankle_actuator_ids,
                    config,
                    sim_time,
                    current_segment_id,
                )
                if viewer is not None:
                    update_follow_camera(viewer, data, torso_body_id, config)
                    draw_rollout_traces(viewer, rollout_metrics, config)
                    viewer.sync()
                if bool(config.get("real_time", True)):
                    sleep_time = model.opt.timestep - (time.time() - step_start)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
        finally:
            if joystick is not None:
                joystick.close()
        print(f"[INFO] MuJoCo rollout finished: sim_time={sim_time:.3f}s wall_time={time.time() - wall_start:.3f}s")
        report = summarize_rollout_metrics(rollout_metrics, sim_time, command, config)
        trace_csv_path = write_torso_trace_csv(rollout_metrics, config)
        if trace_csv_path:
            report["torso_trace"]["csv_path"] = trace_csv_path
        task_trace_csv_path = write_task_trace_csv(rollout_metrics, config)
        if task_trace_csv_path:
            report["task_trace"]["csv_path"] = task_trace_csv_path
        if armhack_stand is not None:
            report["armhack_stand"] = armhack_stand.finalize(
                report,
                sim_time,
                float(config["simulation_dt"]) * int(config["control_decimation"]),
            )
            print(f"[REPORT] ArmHack Stand MuJoCo report: {armhack_stand.report_path}")
            print(f"[REPORT] ArmHack Stand MuJoCo torso plot: {armhack_stand.plot_path}")
            print(f"[REPORT] ArmHack Stand MuJoCo trace: {armhack_stand.trace_path}")
        report["initial_joint_randomization"] = joint_randomization_report
        report["foot_recovery_initialization"] = foot_recovery_initialization
        report["push_disturbances"] = {
            "enabled": push_scheduler.enabled,
            "event_count": len(push_scheduler.events),
            "events": push_scheduler.events,
        }
        print_rollout_report(report)
        metrics_path = str(config.get("metrics_path", ""))
        if metrics_path:
            output_path = Path(metrics_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[INFO] MuJoCo metrics written to: {output_path}")

    if bool(config.get("use_glfw", True)):
        from mujoco import viewer as mujoco_viewer

        def on_viewer_key(keycode: int) -> None:
            if keycode == ord(" ") and bool(config.get("interactive_stance_reset", False)):
                interactive_reset_requested.set()
            elif keycode == ord("1") and armhack_stand is not None:
                interactive_arm_pose_requested.set()

        key_callback = (
            on_viewer_key
            if bool(config.get("interactive_stance_reset", False)) or armhack_stand is not None
            else None
        )
        active_viewer = mujoco_viewer.launch_passive(model, data, key_callback=key_callback)
        try:
            if bool(config.get("interactive_stance_reset", False)):
                print("[INTERACTIVE] Press SPACE to reset with a new randomized initial ankle distance.")
            if armhack_stand is not None:
                print("[INTERACTIVE] Press 1 to cycle arm poses with a 1.0s minimum-jerk transition.")
            update_follow_camera(active_viewer, data, torso_body_id, config)
            simulate_loop(active_viewer)
        finally:
            active_viewer.close()
    else:
        simulate_loop(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Unitree G1 AMP policy in MuJoCo.")
    parser.add_argument("config_file", type=str, help="Path to g1_amp.yaml config file.")
    args = parser.parse_args()
    run_mujoco(load_config(args.config_file))


if __name__ == "__main__":
    main()
