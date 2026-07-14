#!/usr/bin/env python3
"""Visualize a G1 29-DoF CSV joint trajectory and optionally record video.

This script is intentionally standalone: it does not modify task registration,
``play.py``, or any environment config file. By default it replays the CSV as
an exact kinematic state sequence so the recorded video shows the original
q0..q28 full-body joint trajectory.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_CSV = (
    "Reference Data/ArmHack/StandPerturb/raw/"
    "g1_full_body_motion_sdk_50hz.csv"
)


parser = argparse.ArgumentParser(description="Replay a G1 29-DoF CSV joint trajectory in Isaac Sim.")
parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="CSV path with time_s and q0..q28 columns.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of synchronized robots to spawn.")
parser.add_argument("--start_frame", type=int, default=0, help="First CSV frame to replay.")
parser.add_argument("--end_frame", type=int, default=None, help="Exclusive end CSV frame. Defaults to all frames.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum replay steps. Defaults to selected CSV length.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop when max_steps exceeds selected CSV length.")
parser.add_argument(
    "--csv_joint_order",
    type=str,
    choices=("sdk", "lab"),
    default="sdk",
    help=(
        "Order of q0..q28 in the CSV. sdk matches Unitree/GMR motor order; "
        "lab matches the IsaacLab AMP policy/action order."
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    choices=("state", "action"),
    default="state",
    help=(
        "state writes the CSV q values directly to simulator joint state; "
        "action converts q values to JointPositionAction raw actions and steps physics."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle playback to env step time.")
parser.add_argument("--video", action="store_true", default=True, help="Record an mp4 video.")
parser.add_argument("--no-video", dest="video", action="store_false", help="Disable video recording.")
parser.add_argument("--video_length", type=int, default=None, help="Recorded video length in env steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Video output directory. Defaults to logs/csv_full_body_playback/<timestamp>/videos.",
)
parser.add_argument("--camera_distance", type=float, default=3.0, help="Camera distance from robot.")
parser.add_argument("--camera_height", type=float, default=1.35, help="Camera eye height.")
parser.add_argument("--camera_target_height", type=float, default=0.85, help="Camera target height.")
parser.add_argument("--free-root", action="store_true", help="Do not fix the root link in the visualizer config.")
parser.add_argument(
    "--keep-gravity",
    action="store_true",
    help="Keep gravity/collisions enabled. Default disables them for clean CSV pose playback.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import legged_lab.tasks  # noqa: F401, E402
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG  # noqa: E402
from legged_lab.envs.g1_perturb_env import G1PerturbAmpEnv  # noqa: E402
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1_LOCOMOTION_JOINT_NAMES  # noqa: E402
from legged_lab.tasks.locomotion.amp.config.g1_perturb.g1_stand_perturb_env_cfg import (  # noqa: E402
    G1StandPerturbEnvCfg_PLAY,
)


class CsvTrajectory:
    """In-memory G1 q-column trajectory."""

    def __init__(self, times_s: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor, path: Path):
        self.times_s = times_s
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.path = path

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def dt(self) -> float:
        if self.times_s.numel() < 2:
            return 0.02
        deltas = self.times_s[1:] - self.times_s[:-1]
        positive = deltas[deltas > 0.0]
        if positive.numel() == 0:
            return 0.02
        return float(torch.median(positive).item())


class CsvFullBodyPlaybackWrapper(gym.Wrapper):
    """Replay CSV frames either as exact simulator state or as raw actions."""

    def __init__(self, env, trajectory: CsvTrajectory, mode: str, loop: bool):
        super().__init__(env)
        self.trajectory = trajectory
        self.mode = mode
        self.loop = loop
        self.frame_index = 0

        base_env = self.unwrapped
        self.robot = base_env.scene["robot"]
        action_term = base_env.action_manager.get_term("joint_pos")
        if not isinstance(action_term, JointPositionAction):
            raise TypeError(f"Expected joint_pos to be JointPositionAction, got {type(action_term).__name__}.")
        self.action_term = action_term

        action_joint_names = list(action_term._joint_names)
        if action_joint_names != G1_LOCOMOTION_JOINT_NAMES:
            raise RuntimeError(
                "The action joint order does not match G1_LOCOMOTION_JOINT_NAMES. "
                f"got={action_joint_names}"
            )
        resolved_joint_ids = (
            list(range(action_term.action_dim))
            if isinstance(action_term._joint_ids, slice)
            else list(action_term._joint_ids)
        )
        self.joint_ids = resolved_joint_ids
        self._fixed_root_state: torch.Tensor | None = None

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.frame_index = 0
        self._fixed_root_state = self.robot.data.root_state_w.clone()
        self._fixed_root_state[:, 7:13] = 0.0
        self._write_state_frame(0)
        _set_camera(self)
        return result

    def step(self, action):
        base_env = self.unwrapped
        frame = self._frame_for_step()
        if self.mode == "action":
            result = self.env.step(self._raw_action_from_frame(frame))
        else:
            zero_action = torch.zeros(
                (base_env.num_envs, self.action_term.action_dim), dtype=torch.float32, device=base_env.device
            )
            result = self.env.step(zero_action)
            self._write_state_frame(frame)
        self.frame_index += 1
        return result

    def _frame_for_step(self) -> int:
        if self.loop:
            return self.frame_index % self.trajectory.num_frames
        return min(self.frame_index, self.trajectory.num_frames - 1)

    def _write_state_frame(self, frame: int) -> None:
        base_env = self.unwrapped
        if self._fixed_root_state is None:
            self._fixed_root_state = self.robot.data.root_state_w.clone()
            self._fixed_root_state[:, 7:13] = 0.0

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel)
        target_pos = self.trajectory.joint_pos[frame].to(base_env.device).unsqueeze(0).repeat(base_env.num_envs, 1)
        target_vel = self.trajectory.joint_vel[frame].to(base_env.device).unsqueeze(0).repeat(base_env.num_envs, 1)
        joint_pos[:, self.joint_ids] = target_pos
        joint_vel[:, self.joint_ids] = target_vel

        self.robot.write_root_state_to_sim(self._fixed_root_state)
        self.robot.write_joint_state_to_sim(joint_pos[:, self.joint_ids], joint_vel[:, self.joint_ids], joint_ids=self.joint_ids)
        base_env.scene.write_data_to_sim()
        base_env.sim.forward()
        if base_env.sim.has_gui() and not base_env.sim.has_rtx_sensors():
            base_env.sim.render()

    def _raw_action_from_frame(self, frame: int) -> torch.Tensor:
        base_env = self.unwrapped
        desired_joint_pos = self.trajectory.joint_pos[frame].to(base_env.device).unsqueeze(0).repeat(base_env.num_envs, 1)

        offset = self.action_term._offset
        scale = self.action_term._scale
        if isinstance(offset, torch.Tensor):
            offset = offset[: base_env.num_envs]
        else:
            offset = torch.full_like(desired_joint_pos, float(offset))
        if isinstance(scale, torch.Tensor):
            scale = scale[: base_env.num_envs]
        else:
            scale = torch.full_like(desired_joint_pos, float(scale))

        raw_action = (desired_joint_pos - offset) / torch.clamp(scale, min=1.0e-6)
        if self.action_term.cfg.clip is not None:
            raw_action = torch.clamp(raw_action, min=self.action_term._clip[..., 0], max=self.action_term._clip[..., 1])
        return raw_action


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return (_repo_root() / path).resolve()


def _source_joint_names(csv_joint_order: str) -> list[str]:
    if csv_joint_order == "sdk":
        return list(UNITREE_G1_29DOF_CFG.joint_sdk_names)
    return list(G1_LOCOMOTION_JOINT_NAMES)


def _load_csv_trajectory(
    csv_path: Path, start_frame: int, end_frame: int | None, csv_joint_order: str
) -> CsvTrajectory:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        num_joints = len(G1_LOCOMOTION_JOINT_NAMES)
        required = ["time_s"] + [f"q{i}" for i in range(num_joints)]
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")

        source_joint_names = _source_joint_names(csv_joint_order)
        source_index_by_name = {name: index for index, name in enumerate(source_joint_names)}
        missing_target_names = [name for name in G1_LOCOMOTION_JOINT_NAMES if name not in source_index_by_name]
        if missing_target_names:
            raise ValueError(f"CSV source joint order is missing target joints: {missing_target_names}")
        source_to_lab_indices = [source_index_by_name[name] for name in G1_LOCOMOTION_JOINT_NAMES]
        q_column_indices = [header.index(f"q{i}") for i in range(num_joints)]

        times: list[float] = []
        rows: list[list[float]] = []
        skipped_rows = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) < num_joints + 1:
                skipped_rows += 1
                continue
            try:
                if len(row) == len(header):
                    raw_joint_values = [float(row[column_index]) for column_index in q_column_indices]
                else:
                    # Some rows contain an extra natural-language command column between
                    # stage and q0. The final 29 columns are still the G1 q0..q28 values.
                    raw_joint_values = [float(value) for value in row[-num_joints:]]
                times.append(float(row[0]))
                rows.append([raw_joint_values[index] for index in source_to_lab_indices])
            except ValueError as exc:
                skipped_rows += 1
                print(f"[G1 CSV Playback] Skipping non-numeric row {line_number}: {exc}")
        if skipped_rows > 0:
            print(f"[G1 CSV Playback] Skipped {skipped_rows} malformed/non-numeric CSV rows.")

    if not rows:
        raise ValueError(f"CSV contains no trajectory rows: {csv_path}")

    start = max(int(start_frame), 0)
    stop = len(rows) if end_frame is None else min(max(int(end_frame), start), len(rows))
    if stop <= start:
        raise ValueError(f"Empty frame range start_frame={start_frame}, end_frame={end_frame}, csv_frames={len(rows)}")

    selected_times = times[start:stop]
    initial_time = selected_times[0]
    times_tensor = torch.tensor([time_value - initial_time for time_value in selected_times], dtype=torch.float32)
    joint_pos = torch.tensor(rows[start:stop], dtype=torch.float32)
    dt = 0.02
    if times_tensor.numel() >= 2:
        positive = (times_tensor[1:] - times_tensor[:-1])[(times_tensor[1:] - times_tensor[:-1]) > 0.0]
        if positive.numel() > 0:
            dt = float(torch.median(positive).item())
    joint_vel = torch.zeros_like(joint_pos)
    if joint_pos.shape[0] > 1:
        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / max(dt, 1.0e-6)
        joint_vel[-1] = joint_vel[-2]
    return CsvTrajectory(times_s=times_tensor, joint_pos=joint_pos, joint_vel=joint_vel, path=csv_path)


def _disable_visualizer_noise(env_cfg: G1StandPerturbEnvCfg_PLAY, keep_gravity: bool, fix_root: bool) -> None:
    if hasattr(env_cfg, "upper_body_perturbation"):
        env_cfg.upper_body_perturbation.enabled = False

    if not keep_gravity:
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True  # type: ignore[union-attr]
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False  # type: ignore[union-attr]
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = fix_root  # type: ignore[union-attr]
        env_cfg.scene.robot.spawn.activate_contact_sensors = False  # type: ignore[union-attr]
        env_cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)  # type: ignore[union-attr]
        env_cfg.scene.contact_forces = None

    for name in (
        "physics_material",
        "add_base_mass",
        "randomize_rigid_body_com",
        "scale_link_mass",
        "scale_actuator_gains",
        "scale_joint_parameters",
        "base_external_force_torque",
        "reset_base",
        "reset_robot_joints",
        "reset_from_ref",
        "push_robot",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    for name in list(env_cfg.terminations.__dict__.keys()):
        if not name.startswith("__"):
            setattr(env_cfg.terminations, name, None)

    for name in list(env_cfg.rewards.__dict__.keys()):
        if not name.startswith("__"):
            setattr(env_cfg.rewards, name, None)

    env_cfg.commands.base_velocity.debug_vis = False


def _set_camera(env) -> None:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu()
    eye = [
        float(root_pos[0] + args_cli.camera_distance),
        float(root_pos[1] - 0.35 * args_cli.camera_distance),
        float(root_pos[2] + args_cli.camera_height),
    ]
    target = [float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + args_cli.camera_target_height)]
    base_env.sim.set_camera_view(eye=eye, target=target)


def main() -> None:
    csv_path = _resolve_path(args_cli.csv)
    trajectory = _load_csv_trajectory(
        csv_path, args_cli.start_frame, args_cli.end_frame, args_cli.csv_joint_order
    )

    env_cfg = G1StandPerturbEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    _disable_visualizer_noise(env_cfg, keep_gravity=args_cli.keep_gravity, fix_root=not args_cli.free_root)

    print("[G1 CSV Playback] CSV:", trajectory.path)
    print(
        f"[G1 CSV Playback] frames={trajectory.num_frames} dt~{trajectory.dt:.6f}s "
        f"mode={args_cli.mode} csv_joint_order={args_cli.csv_joint_order}"
    )
    print("[G1 CSV Playback] CSV q-column source order:")
    for index, name in enumerate(_source_joint_names(args_cli.csv_joint_order)):
        print(f"  q{index:02d}: {name}")
    if args_cli.csv_joint_order != "lab":
        print("[G1 CSV Playback] Reordered CSV q-columns to IsaacLab policy/action order before replay.")

    env = G1PerturbAmpEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = CsvFullBodyPlaybackWrapper(env, trajectory=trajectory, mode=args_cli.mode, loop=args_cli.loop)

    steps_to_run = args_cli.max_steps if args_cli.max_steps is not None else trajectory.num_frames
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_dir = (
            Path(args_cli.video_dir).expanduser().resolve()
            if args_cli.video_dir is not None
            else (_repo_root() / "logs" / "csv_full_body_playback" / timestamp / "videos").resolve()
        )
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length if args_cli.video_length is not None else steps_to_run,
            "disable_logger": True,
        }
        print("[G1 CSV Playback] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env.reset()
    _set_camera(env)

    dummy_action = torch.zeros(
        (env.unwrapped.num_envs, env.unwrapped.action_manager.get_term("joint_pos").action_dim),
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    for step_index in range(steps_to_run):
        if not simulation_app.is_running():
            break
        step_start = time.time()
        with torch.inference_mode():
            env.step(dummy_action)
        if args_cli.real_time:
            sleep_time = env.unwrapped.step_dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        if not args_cli.loop and step_index + 1 >= trajectory.num_frames:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
