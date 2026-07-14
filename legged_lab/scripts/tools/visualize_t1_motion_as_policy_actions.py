#!/usr/bin/env python3
"""Visualize a T1 motion by feeding it through the policy action path.

The motion joint targets are converted to the same raw 27-D action vector a policy
would output for the T1 AMP task, then applied with ``env.step(action)``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


DEFAULT_MOTION = (
    "source/legged_lab/legged_lab/data/MotionData/"
    "t1_29dof_accad_g1used_50hz_amp_official/B10_-__Walk_turn_left_45_stageii.pkl"
)

parser = argparse.ArgumentParser(description="Visualize a T1 AMP motion through the current action path.")
parser.add_argument("--motion", type=str, default=DEFAULT_MOTION, help="Path to a converted 29-DoF T1 AMP pickle.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--real-time", action="store_true", default=True, help="Throttle playback to simulation time.")
parser.add_argument("--no-real-time", dest="real_time", action="store_false", help="Run as fast as possible.")
parser.add_argument("--free-root", action="store_true", help="Do not fix the root link during visualization.")
parser.add_argument("--loop", action="store_true", default=True, help="Loop the motion until the viewer is closed.")
parser.add_argument("--once", dest="loop", action="store_false", help="Exit after one motion pass.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import joblib  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from legged_lab.envs import ManagerBasedAmpEnv  # noqa: E402
from legged_lab.tasks.locomotion.amp.config.t1.t1_amp_env_cfg import T1AmpEnvCfg  # noqa: E402


LAB_DOF_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def install_numpy_core_pickle_shim() -> None:
    if hasattr(np, "_core"):
        return

    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule_name in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        full_name = f"numpy._core.{submodule_name}"
        if full_name not in sys.modules:
            sys.modules[full_name] = importlib.import_module(f"numpy.core.{submodule_name}")


def disable_randomization_and_collisions(env_cfg: T1AmpEnvCfg, fix_root: bool) -> None:
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True  # type: ignore[union-attr]
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False  # type: ignore[union-attr]
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = fix_root  # type: ignore[union-attr]
    env_cfg.scene.robot.spawn.activate_contact_sensors = False  # type: ignore[union-attr]
    env_cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)  # type: ignore[union-attr]

    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.randomize_rigid_body_com = None
    env_cfg.events.scale_link_mass = None
    env_cfg.events.scale_actuator_gains = None
    env_cfg.events.scale_joint_parameters = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    env_cfg.events.push_robot = None

    env_cfg.scene.contact_forces = None

    env_cfg.terminations.time_out = None
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.base_height = None
    env_cfg.terminations.bad_orientation = None
    env_cfg.commands.base_velocity.debug_vis = False

    for reward_name in list(env_cfg.rewards.__dict__.keys()):
        if not reward_name.startswith("__"):
            setattr(env_cfg.rewards, reward_name, None)


def main() -> None:
    install_numpy_core_pickle_shim()
    repo_root = Path(__file__).resolve().parents[2]
    motion_path = Path(args_cli.motion)
    if not motion_path.is_absolute():
        motion_path = repo_root / motion_path

    motion = joblib.load(motion_path)
    motion_joint_pos = torch.as_tensor(motion["dof_pos"], dtype=torch.float32)
    if motion_joint_pos.shape[-1] != len(LAB_DOF_NAMES):
        raise ValueError(f"Expected 29-DoF motion, got shape {tuple(motion_joint_pos.shape)}")

    env_cfg = T1AmpEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.motion_data.motion_dataset.motion_data_dir = str(motion_path.parent)
    env_cfg.motion_data.motion_dataset.motion_data_weights = {motion_path.stem: 1.0}
    env_cfg.animation.animation.random_initialize = False
    env_cfg.animation.animation.random_fetch = False
    env_cfg.animation.animation.enable_visualization = False
    disable_randomization_and_collisions(env_cfg, fix_root=not args_cli.free_root)

    env = ManagerBasedAmpEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names)
    action_joint_ids = action_term._joint_ids
    motion_name_to_index = {name: index for index, name in enumerate(LAB_DOF_NAMES)}
    motion_action_indices = [motion_name_to_index[name] for name in action_joint_names]

    print("[T1 Motion Action Visualizer] Motion:", motion_path)
    print("[T1 Motion Action Visualizer] FPS:", motion.get("fps"), "frames:", motion_joint_pos.shape[0])
    print("[T1 Motion Action Visualizer] Action joint ids:", action_joint_ids)
    print("[T1 Motion Action Visualizer] Action joint names:")
    for index, name in enumerate(action_joint_names):
        print(f"  {index:2d}: {name}")
    print("[T1 Motion Action Visualizer] Collisions off, gravity off, root fixed:", not args_cli.free_root)

    env.reset()
    target_joint_pos = motion_joint_pos[:, motion_action_indices].to(env.device)
    num_frames = target_joint_pos.shape[0]

    initial_joint_pos = target_joint_pos[0].unsqueeze(0).repeat(env.num_envs, 1)
    initial_joint_vel = torch.zeros_like(initial_joint_pos)
    robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel, joint_ids=action_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    frame_index = 0
    while simulation_app.is_running():
        step_start = time.time()
        with torch.inference_mode():
            desired_joint_pos = target_joint_pos[frame_index].unsqueeze(0).repeat(env.num_envs, 1)
            offset = action_term._offset
            scale = action_term._scale
            if isinstance(offset, torch.Tensor):
                offset = offset[: env.num_envs]
            raw_action = (desired_joint_pos - offset) / scale
            env.step(raw_action)

        frame_index += 1
        if frame_index >= num_frames:
            if args_cli.loop:
                frame_index = 0
            else:
                break

        if args_cli.real_time:
            sleep_time = env.step_dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()