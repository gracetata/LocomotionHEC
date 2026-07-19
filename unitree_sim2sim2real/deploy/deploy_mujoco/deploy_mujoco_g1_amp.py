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
    Input is a TorchScript policy exported by scripts/export_g1_amp_policy.sh
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
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml  # type: ignore[reportMissingImports]

from armhack_stand import ArmHackStandReplay
from armhack_walk import ArmHackWalkAdapter
from extreme_stand_recovery import ExtremeStandRecoveryPerturbation


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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["policy_path"] = _resolve_path(os.environ.get("G1_AMP_POLICY_PATH", config["policy_path"]))
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
    config["armhack_walk_enable"] = _env_bool(
        "G1_AMP_ARMHACK_WALK_ENABLE", bool(config.get("armhack_walk_enable", False))
    )
    config["armhack_walk_pose_path"] = _resolve_path(
        os.environ.get("G1_AMP_ARMHACK_WALK_POSE_PATH", config.get("armhack_walk_pose_path", ""))
    )
    config["armhack_walk_contract_path"] = _resolve_path(
        os.environ.get("G1_AMP_ARMHACK_WALK_CONTRACT_PATH", config.get("armhack_walk_contract_path", ""))
    )
    config["armhack_walk_pose_name"] = os.environ.get(
        "G1_AMP_ARMHACK_WALK_POSE_NAME", config.get("armhack_walk_pose_name", "pos2_down")
    )
    config["armhack_walk_fixed_command"] = _env_yaml_vector(
        "G1_AMP_ARMHACK_WALK_FIXED_COMMAND",
        list(config.get("armhack_walk_fixed_command", [0.35, 0.0, 0.0])),
        3,
    )
    config["armhack_walk_start_active"] = _env_bool(
        "G1_AMP_ARMHACK_WALK_START_ACTIVE", bool(config.get("armhack_walk_start_active", True))
    )
    config["armhack_walk_schedule_path"] = _resolve_path(
        os.environ.get(
            "G1_AMP_ARMHACK_WALK_SCHEDULE_PATH",
            config.get("armhack_walk_schedule_path", ""),
        )
    )
    config["armhack_walk_scenario_name"] = os.environ.get(
        "G1_AMP_ARMHACK_WALK_SCENARIO_NAME",
        config.get("armhack_walk_scenario_name", ""),
    )
    config["armhack_walk_hard_zero_command"] = _env_bool(
        "G1_AMP_ARMHACK_WALK_HARD_ZERO_COMMAND",
        bool(config.get("armhack_walk_hard_zero_command", False)),
    )
    config["armhack_walk_zero_epsilon"] = _env_float(
        "G1_AMP_ARMHACK_WALK_ZERO_EPSILON",
        float(config.get("armhack_walk_zero_epsilon", 1.0e-6)),
    )
    config["behavior_settle_time_s"] = _env_float(
        "G1_AMP_BEHAVIOR_SETTLE_TIME_S", float(config.get("behavior_settle_time_s", 0.75))
    )
    config["extreme_stand_recovery_enable"] = _env_bool(
        "G1_AMP_EXTREME_STAND_RECOVERY_ENABLE",
        bool(config.get("extreme_stand_recovery_enable", False)),
    )
    config["extreme_stand_recovery_seed"] = _env_int(
        "G1_AMP_EXTREME_STAND_RECOVERY_SEED",
        int(config.get("extreme_stand_recovery_seed", 20260719)),
    )
    for key, env_name, default in (
        ("extreme_stand_recovery_leg_noise_rad", "G1_AMP_EXTREME_STAND_LEG_NOISE_RAD", 0.20),
        ("extreme_stand_recovery_waist_noise_rad", "G1_AMP_EXTREME_STAND_WAIST_NOISE_RAD", 0.25),
        ("extreme_stand_recovery_arm_noise_rad", "G1_AMP_EXTREME_STAND_ARM_NOISE_RAD", 0.45),
        (
            "extreme_stand_recovery_joint_velocity_noise_rad_s",
            "G1_AMP_EXTREME_STAND_JOINT_VEL_NOISE_RAD_S",
            0.75,
        ),
        (
            "extreme_stand_recovery_root_roll_pitch_noise_rad",
            "G1_AMP_EXTREME_STAND_ROOT_RP_NOISE_RAD",
            0.18,
        ),
        (
            "extreme_stand_recovery_root_linear_velocity_noise_m_s",
            "G1_AMP_EXTREME_STAND_ROOT_LIN_VEL_NOISE_M_S",
            0.30,
        ),
        (
            "extreme_stand_recovery_root_angular_velocity_noise_rad_s",
            "G1_AMP_EXTREME_STAND_ROOT_ANG_VEL_NOISE_RAD_S",
            0.50,
        ),
        ("extreme_stand_recovery_force_max_n", "G1_AMP_EXTREME_STAND_FORCE_MAX_N", 35.0),
        ("extreme_stand_recovery_torque_max_nm", "G1_AMP_EXTREME_STAND_TORQUE_MAX_NM", 5.0),
        (
            "extreme_stand_recovery_wrench_interval_s",
            "G1_AMP_EXTREME_STAND_WRENCH_INTERVAL_S",
            2.5,
        ),
        (
            "extreme_stand_recovery_wrench_duration_s",
            "G1_AMP_EXTREME_STAND_WRENCH_DURATION_S",
            0.25,
        ),
    ):
        config[key] = _env_float(env_name, float(config.get(key, default)))
    config["sole_min_clearance_m"] = _env_float(
        "G1_AMP_SOLE_MIN_CLEARANCE_M", float(config.get("sole_min_clearance_m", 0.025))
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


def foot_contact_states_with_floor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_ids: set[int],
    foot_body_ids: list[int] | set[int],
) -> np.ndarray:
    """Return one floor-contact flag per ordered foot body."""
    ordered_ids = list(foot_body_ids)
    index_by_body = {body_id: index for index, body_id in enumerate(ordered_ids)}
    states = np.zeros(len(ordered_ids), dtype=bool)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in floor_geom_ids:
            other_geom = geom2
        elif geom2 in floor_geom_ids:
            other_geom = geom1
        else:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        if other_body in index_by_body:
            states[index_by_body[other_body]] = True
    return states


def oriented_sole_signed_clearance(
    data: mujoco.MjData,
    foot_body_ids: list[int] | set[int],
    center_offset_x: float = 0.035,
    half_length: float = 0.090,
    half_width: float = 0.035,
) -> float:
    """2-D SAT signed clearance between the two oriented S3 G1 sole rectangles."""
    ordered_ids = list(foot_body_ids)
    if len(ordered_ids) != 2:
        raise ValueError("oriented sole clearance requires exactly two foot bodies")
    local_corners = np.asarray(
        [
            [center_offset_x - half_length, -half_width, 0.0],
            [center_offset_x + half_length, -half_width, 0.0],
            [center_offset_x + half_length, half_width, 0.0],
            [center_offset_x - half_length, half_width, 0.0],
        ],
        dtype=np.float64,
    )
    footprints = []
    for body_id in ordered_ids:
        rotation = data.xmat[body_id].reshape(3, 3)
        footprints.append((local_corners @ rotation.T + data.xpos[body_id])[:, :2])
    left, right = footprints
    raw_axes = [
        left[1] - left[0],
        left[3] - left[0],
        right[1] - right[0],
        right[3] - right[0],
    ]
    separations = []
    for raw_axis in raw_axes:
        axis = raw_axis / max(float(np.linalg.norm(raw_axis)), 1.0e-9)
        left_projection = left @ axis
        right_projection = right @ axis
        separations.append(
            max(
                float(np.min(right_projection) - np.max(left_projection)),
                float(np.min(left_projection) - np.max(right_projection)),
            )
        )
    # Positive means separated; negative is overlap depth along the least-overlapping axis.
    return float(max(separations))


def init_rollout_metrics(
    data: mujoco.MjData,
    torso_body_id: int,
    config: dict,
    foot_body_ids: list[int] | set[int],
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
        "sample_times": [],
        "torso_pos_xy": [],
        "foot_signed_clearances_m": [],
        "foot_planar_speeds_m_per_s": [],
        "foot_touchdown_events": [],
        "foot_contact_steps": 0,
        "foot_touchdown_count": 0,
        "fallen": False,
        "fall_time": None,
        "prev_torso_pos_w": torso_pos_w,
        "prev_torso_quat_w": torso_quat_w,
        "prev_torso_lin_vel_w": np.zeros(3, dtype=np.float32),
        "prev_torso_ang_vel_b": np.zeros(3, dtype=np.float32),
        "prev_foot_pos_w": data.xpos[list(foot_body_ids)].copy().astype(np.float32),
        "prev_foot_contact_states": np.zeros(len(list(foot_body_ids)), dtype=bool),
        "simulation_dt": float(config.get("simulation_dt", 0.001)),
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
    foot_body_ids: list[int] | set[int],
    torso_body_id: int,
    config: dict,
    sim_time: float,
    segment_id: int,
) -> None:
    del qvel_addresses, policy_joint_names
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
    metrics["sample_times"].append(float(sim_time))
    metrics["torso_pos_xy"].append([float(torso_pos_w[0]), float(torso_pos_w[1])])

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

    _, foot_contact_count = contact_force_with_floor(model, data, floor_geom_ids, foot_body_ids)
    if foot_contact_count > 0:
        metrics["foot_contact_steps"] += 1
    ordered_foot_ids = list(foot_body_ids)
    current_foot_pos_w = data.xpos[ordered_foot_ids].copy().astype(np.float32)
    foot_planar_speed = np.linalg.norm(
        (current_foot_pos_w[:, :2] - metrics["prev_foot_pos_w"][:, :2]) / dt,
        axis=1,
    )
    contact_states = foot_contact_states_with_floor(
        model, data, floor_geom_ids, ordered_foot_ids
    )
    touchdown_events = np.logical_and(contact_states, ~metrics["prev_foot_contact_states"])
    touchdown_count = int(np.count_nonzero(touchdown_events))
    signed_clearance = oriented_sole_signed_clearance(data, ordered_foot_ids)
    metrics["foot_planar_speeds_m_per_s"].append(float(np.mean(foot_planar_speed)))
    metrics["foot_touchdown_events"].append(touchdown_count)
    metrics["foot_signed_clearances_m"].append(float(signed_clearance))
    metrics["foot_touchdown_count"] += touchdown_count
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
    metrics["prev_foot_pos_w"] = current_foot_pos_w
    metrics["prev_foot_contact_states"] = contact_states
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


def summarize_command_segments(metrics: dict, config: dict) -> list[dict]:
    segments: list[dict] = []
    segment_ids = np.asarray(metrics["segment_ids"], dtype=np.int32)
    lin_errors = np.asarray(metrics["lin_vel_xy_errors"], dtype=np.float32)
    yaw_errors = np.asarray(metrics["yaw_rate_errors"], dtype=np.float32)
    lin_vel_x = np.asarray(metrics["lin_vel_x"], dtype=np.float32)
    lin_vel_y = np.asarray(metrics["lin_vel_y"], dtype=np.float32)
    yaw_rate = np.asarray(metrics["yaw_rate"], dtype=np.float32)
    command_samples = np.asarray(metrics["command_samples"], dtype=np.float32)
    sample_times = np.asarray(metrics["sample_times"], dtype=np.float32)
    torso_pos_xy = np.asarray(metrics["torso_pos_xy"], dtype=np.float32)
    roll_abs = np.asarray(metrics["roll_abs"], dtype=np.float32)
    pitch_abs = np.asarray(metrics["pitch_abs"], dtype=np.float32)
    sole_clearance = np.asarray(metrics["foot_signed_clearances_m"], dtype=np.float32)
    foot_speed = np.asarray(metrics["foot_planar_speeds_m_per_s"], dtype=np.float32)
    touchdown_events = np.asarray(metrics["foot_touchdown_events"], dtype=np.int32)
    settle_time = max(float(config.get("behavior_settle_time_s", 0.75)), 0.0)
    minimum_clearance = float(config.get("sole_min_clearance_m", 0.025))
    simulation_dt = max(float(metrics.get("simulation_dt", 0.001)), 1.0e-6)
    for segment in metrics["command_segments"]:
        mask = segment_ids == int(segment["id"])
        if not np.any(mask):
            continue
        steady_mask = mask & (sample_times >= float(segment["start_time"]) + settle_time)
        if not np.any(steady_mask):
            steady_mask = mask
        lin_mae = float(np.mean(lin_errors[mask]))
        yaw_mae = float(np.mean(yaw_errors[mask]))
        tracking_score = 0.7 * _bounded_exp_score(lin_mae, 0.35) + 0.3 * _bounded_exp_score(yaw_mae, 0.50)
        segment_positions = torso_pos_xy[steady_mask]
        displacement = (
            float(np.linalg.norm(segment_positions[-1] - segment_positions[0]))
            if len(segment_positions) >= 2
            else 0.0
        )
        steady_duration = max(float(np.count_nonzero(steady_mask)) * simulation_dt, simulation_dt)
        steady_touchdowns = int(np.sum(touchdown_events[steady_mask]))
        item = {
                "id": int(segment["id"]),
                "name": str(segment.get("name", f"segment_{segment['id']}")),
                "start_time": float(segment["start_time"]),
                "command": [float(value) for value in segment["command"]],
                "mean_command": [float(value) for value in np.mean(command_samples[mask], axis=0)],
                "samples": int(np.count_nonzero(mask)),
                "duration_s": float(np.count_nonzero(mask) * simulation_dt),
                "steady_settle_time_s": settle_time,
                "steady_samples": int(np.count_nonzero(steady_mask)),
                "mean_lin_vel_x": float(np.mean(lin_vel_x[mask])),
                "mean_lin_vel_y": float(np.mean(lin_vel_y[mask])),
                "mean_yaw_rate": float(np.mean(yaw_rate[mask])),
                "steady_mean_lin_vel_x": float(np.mean(lin_vel_x[steady_mask])),
                "steady_mean_lin_vel_y": float(np.mean(lin_vel_y[steady_mask])),
                "steady_mean_yaw_rate": float(np.mean(yaw_rate[steady_mask])),
                "lin_vel_xy_mae": lin_mae,
                "yaw_rate_mae": yaw_mae,
                "steady_lin_vel_xy_mae": float(np.mean(lin_errors[steady_mask])),
                "steady_yaw_rate_mae": float(np.mean(yaw_errors[steady_mask])),
                "steady_planar_displacement_m": displacement,
                "steady_mean_foot_planar_speed_m_per_s": float(np.mean(foot_speed[steady_mask])),
                "steady_touchdown_count": steady_touchdowns,
                "steady_step_frequency_hz": float(steady_touchdowns / steady_duration),
                "steady_min_signed_sole_clearance_m": float(np.min(sole_clearance[steady_mask])),
                "steady_sole_clearance_violation_fraction": float(
                    np.count_nonzero(sole_clearance[steady_mask] < minimum_clearance)
                    / max(np.count_nonzero(steady_mask), 1)
                ),
                "steady_max_abs_roll_rad": float(np.max(roll_abs[steady_mask])),
                "steady_max_abs_pitch_rad": float(np.max(pitch_abs[steady_mask])),
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


def summarize_rollout_metrics(metrics: dict, sim_time: float, command: np.ndarray, config: dict) -> dict:
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
        "foot_touchdown_count": int(metrics["foot_touchdown_count"]),
        "step_frequency_hz": float(metrics["foot_touchdown_count"] / max(float(sim_time), 1.0e-6)),
        "min_signed_sole_clearance_m": _safe_min(metrics["foot_signed_clearances_m"]),
        "sole_clearance_violation_fraction": float(
            np.count_nonzero(
                np.asarray(metrics["foot_signed_clearances_m"], dtype=np.float32)
                < float(config.get("sole_min_clearance_m", 0.025))
            )
            / max(len(metrics["foot_signed_clearances_m"]), 1)
        ),
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
    return {
        "sim_time": float(sim_time),
        "task_tracking": task_tracking,
        "command_segments": summarize_command_segments(metrics, config),
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
    armhack_walk = (
        ArmHackWalkAdapter(config, policy_joint_names, default_angles)
        if bool(config.get("armhack_walk_enable", False))
        else None
    )
    extreme_stand_recovery = (
        ExtremeStandRecoveryPerturbation(config, policy_joint_names)
        if bool(config.get("extreme_stand_recovery_enable", False))
        else None
    )
    if armhack_stand is not None and armhack_walk is not None:
        raise ValueError("ArmHack Stand and Walk adapters cannot be enabled together.")
    if extreme_stand_recovery is not None and (armhack_stand is not None or armhack_walk is not None):
        raise ValueError("Extreme Stand recovery cannot be combined with an ArmHack action adapter.")
    if armhack_stand is not None:
        if bool(config.get("random_commands", False)) or str(config.get("command_mode", "independent")).lower() != "independent":
            raise ValueError("ArmHack Stand MuJoCo replay requires fixed independent zero commands.")
        if not np.allclose(np.asarray(config["cmd_init"], dtype=np.float32), 0.0, atol=1.0e-8):
            raise ValueError("ArmHack Stand MuJoCo replay requires cmd_init=[0, 0, 0].")
    if armhack_walk is not None:
        if bool(config.get("random_commands", False)) or str(config.get("command_mode", "independent")).lower() != "independent":
            raise ValueError("ArmHack Walk MuJoCo requires fixed independent commands.")
        if not bool(config.get("command_ramp", False)):
            raise ValueError("ArmHack Walk MuJoCo requires command_ramp=True for zero/fixed transitions.")
    if extreme_stand_recovery is not None:
        if bool(config.get("random_commands", False)) or str(config.get("command_mode", "independent")).lower() != "independent":
            raise ValueError("Extreme Stand recovery MuJoCo test requires fixed independent zero commands.")
        if not np.allclose(np.asarray(config["cmd_init"], dtype=np.float32), 0.0, atol=1.0e-8):
            raise ValueError("Extreme Stand recovery MuJoCo test requires cmd_init=[0, 0, 0].")
    rng = np.random.default_rng(int(config.get("command_seed", 1)))
    command_mode = str(config.get("command_mode", "independent")).lower()
    joystick = JoystickCommandReader(config) if command_mode == "joystick" else None
    nav2_replay = Nav2CommandReplay(config, rng) if command_mode == "nav2" else None
    target_command = (
        armhack_walk.current_target_command(0.0)
        if armhack_walk is not None
        else np.asarray(config["cmd_init"], dtype=np.float32)
    )
    nav2_segment_info = None
    if joystick is not None:
        target_command = joystick.read_command()
    elif nav2_replay is not None:
        target_command, nav2_segment_info = nav2_replay.sample_window(0.0)
    elif bool(config.get("random_commands", False)):
        target_command = sample_random_command(rng, config)
    command = np.zeros(3, dtype=np.float32) if bool(config.get("command_ramp", False)) else target_command.copy()
    action = np.zeros(len(policy_joint_names), dtype=np.float32)

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
    if not floor_geom_ids:
        raise RuntimeError("MuJoCo model has no floor/ground plane; set add_floor: true or provide a scene XML with a floor.")
    if not foot_body_ids:
        raise RuntimeError("MuJoCo model has no ankle_roll_link/foot_link bodies for contact metrics.")
    foot_body_ids = sorted(
        foot_body_ids,
        key=lambda body_id: _name_from_id(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
    )
    if len(foot_body_ids) != 2:
        names = [
            _name_from_id(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in foot_body_ids
        ]
        raise RuntimeError(f"ArmHack Walk behavior metrics require exactly two feet, got {names}")
    if torso_body_id < 0:
        raise RuntimeError(f"MuJoCo model has no torso body named '{config.get('torso_body_name', 'torso_link')}'.")

    data.qpos[0:3] = np.asarray(config["root_pos_init"], dtype=np.float32)
    data.qpos[3:7] = np.asarray(config["root_quat_init"], dtype=np.float32)
    for joint_name, default_angle in default_by_joint.items():
        data.qpos[qpos_addresses[joint_name]] = default_angle
    if extreme_stand_recovery is not None:
        extreme_stand_recovery.initialize_model_and_state(
            model, data, qpos_addresses, qvel_addresses
        )
    if armhack_stand is not None:
        armhack_stand.initialize_model_and_state(mujoco, model, data, qpos_addresses, torso_body_id)
    else:
        if armhack_walk is not None:
            armhack_walk.initialize_state(data, qpos_addresses)
            # IsaacLab writes the selected arm pose and its composed raw action
            # before the first policy observation.  Match that reset contract so
            # obs[67:96] never claims zero arms while MuJoCo already holds the
            # fixed pose.
            action = armhack_walk.compose_action(action)
        mujoco.mj_forward(model, data)
    rollout_metrics = init_rollout_metrics(data, torso_body_id, config, foot_body_ids)
    current_segment_id = 0
    current_adapter_segment_index = (
        int(armhack_walk.current_schedule_segment(0.0)["index"])
        if armhack_walk is not None and armhack_walk.has_schedule
        else -1
    )
    next_command_time = max(float(config.get("command_interval", 2.0)), float(model.opt.timestep))
    initial_segment = {
        "id": current_segment_id,
        "start_time": 0.0,
        "command": [float(target_command[0]), float(target_command[1]), float(target_command[2])],
    }
    if armhack_walk is not None and armhack_walk.has_schedule:
        adapter_segment = armhack_walk.current_schedule_segment(0.0)
        initial_segment["name"] = str(adapter_segment["name"])
        initial_segment["scenario_name"] = armhack_walk.scenario_name
    if nav2_segment_info is not None:
        initial_segment.update(nav2_segment_info)
    rollout_metrics["command_segments"].append(initial_segment)

    policy = torch.jit.load(config["policy_path"], map_location="cpu")
    policy.eval()

    def step_policy_if_needed(counter: int, sim_time: float) -> np.ndarray:
        if counter % int(config["control_decimation"]) != 0:
            return action
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
        with torch.inference_mode():
            next_action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().cpu().numpy().squeeze().astype(np.float32)
        if armhack_stand is not None:
            next_action = armhack_stand.compose_action(next_action, sim_time)
        elif armhack_walk is not None:
            next_action = armhack_walk.compose_action(next_action)
        return next_action

    def simulate_loop(viewer=None) -> None:
        nonlocal action, command, target_command, current_segment_id, current_adapter_segment_index, next_command_time
        counter = 0
        sim_time = 0.0
        wall_start = time.time()
        try:
            while sim_time < float(config["simulation_duration"]):
                if viewer is not None and not viewer.is_running():
                    break
                step_start = time.time()
                if armhack_walk is not None:
                    previous_target_command = target_command.copy()
                    target_command = armhack_walk.current_target_command(sim_time)
                    adapter_segment = armhack_walk.current_schedule_segment(sim_time)
                    if adapter_segment is not None:
                        adapter_index = int(adapter_segment["index"])
                        if adapter_index != current_adapter_segment_index:
                            current_adapter_segment_index = adapter_index
                            current_segment_id += 1
                            rollout_metrics["command_segments"].append(
                                {
                                    "id": current_segment_id,
                                    "name": str(adapter_segment["name"]),
                                    "scenario_name": armhack_walk.scenario_name,
                                    "start_time": float(sim_time),
                                    "command": [float(value) for value in target_command],
                                }
                            )
                    elif not np.allclose(
                        target_command, previous_target_command, rtol=0.0, atol=1.0e-8
                    ):
                        current_segment_id += 1
                        rollout_metrics["command_segments"].append(
                            {
                                "id": current_segment_id,
                                "name": "manual_fixed" if np.linalg.norm(target_command) > 0.0 else "manual_zero",
                                "start_time": float(sim_time),
                                "command": [float(value) for value in target_command],
                            }
                        )
                elif joystick is not None:
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
                hard_zero = (
                    armhack_walk is not None
                    and bool(config.get("armhack_walk_hard_zero_command", False))
                    and float(np.linalg.norm(target_command))
                    <= float(config.get("armhack_walk_zero_epsilon", 1.0e-6))
                )
                command = (
                    np.zeros(3, dtype=np.float32)
                    if hard_zero
                    else smooth_command(command, target_command, float(model.opt.timestep), config)
                )
                control_step = counter % int(config["control_decimation"]) == 0
                action = step_policy_if_needed(counter, sim_time)
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
                if extreme_stand_recovery is not None:
                    extreme_stand_recovery.update_external_wrench(data, sim_time)
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
        if armhack_walk is not None:
            report["armhack_walk"] = armhack_walk.summary()
        if extreme_stand_recovery is not None:
            report["extreme_stand_recovery"] = extreme_stand_recovery.summary()
        print_rollout_report(report)
        metrics_path = str(config.get("metrics_path", ""))
        if metrics_path:
            output_path = Path(metrics_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[INFO] MuJoCo metrics written to: {output_path}")

    if bool(config.get("use_glfw", True)):
        from mujoco import viewer as mujoco_viewer

        key_callback = armhack_walk.key_callback if armhack_walk is not None else None
        with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as active_viewer:
            update_follow_camera(active_viewer, data, torso_body_id, config)
            simulate_loop(active_viewer)
    else:
        simulate_loop(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Unitree G1 AMP policy in MuJoCo.")
    parser.add_argument("config_file", type=str, help="Path to g1_amp.yaml config file.")
    args = parser.parse_args()
    run_mujoco(load_config(args.config_file))


if __name__ == "__main__":
    main()
