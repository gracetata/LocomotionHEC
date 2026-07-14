#!/usr/bin/env python3
"""Visualize G1 AMP reference motions directly in Isaac Sim.

This tool is intentionally separate from policy playback.  It writes reference
root and joint states straight into the official G1 articulation so dataset
conversion issues can be inspected without policy, reward, contact, or gravity
effects in the loop.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import time
import types
from dataclasses import dataclass

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION_DIR = (
    REPO_ROOT
    / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz_task_core"
)
VIEW_MODES = ("training_current", "name_aligned", "name_aligned_xyzw")
MODE_SELECTIONS = (
    "balanced_modes",
    "mode_forward_slow",
    "mode_forward_normal",
    "mode_backward",
    "mode_lateral_left",
    "mode_lateral_right",
    "mode_turn_left",
    "mode_turn_right",
    "mode_stand",
)


parser = argparse.ArgumentParser(description="Visualize G1 AMP reference motion pickles in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of reference clips to show at once.")
parser.add_argument("--motion_dir", type=str, default=str(DEFAULT_MOTION_DIR), help="Directory of Lab/GMR pkl motions.")
parser.add_argument("--motion_name", type=str, default="", help="Specific motion stem or .pkl file name to replay.")
parser.add_argument("--start_index", type=int, default=0, help="First sorted motion index when motion_name is empty.")
parser.add_argument(
    "--motion_selection",
    choices=("sorted", "spread", "random", "yaw", "path_turn", "misaligned", "cycle_all", *MODE_SELECTIONS),
    default="cycle_all",
    help="How to select clips when motion_name is empty.",
)
parser.add_argument("--view_mode", choices=VIEW_MODES, default="name_aligned_xyzw", help="Reference interpretation mode.")
parser.add_argument("--robot_asset", type=str, default="g1_29dof", help="Robot asset preset: g1_29dof, s3_g1_29dof, s3_g1_29dof_mjcf.")
parser.add_argument("--height_offset", type=float, default=0.15, help="Extra visual root height offset in meters.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many frames; 0 runs until the window closes.")
parser.add_argument("--real_time", action="store_true", help="Pace playback against wall-clock time.")
parser.add_argument("--loop", action="store_true", help="Loop clips instead of holding their final frame.")
parser.add_argument("--no_loop", action="store_true", help="Disable clip looping.")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--history_steps", type=int, default=4, help="Number of current/future AMP ghost frames to show.")
parser.add_argument("--trail_length", type=int, default=160, help="Maximum root/foot trail length in frames.")
parser.add_argument("--print_interval", type=int, default=25, help="Frames between status log updates.")
parser.add_argument("--zero_world_gravity", action="store_true", help="Set world gravity to zero in addition to disabling robot gravity.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np


def _install_numpy_pickle_shim() -> None:
    """Allow numpy>=2 pickles to load in numpy<2 runtimes."""

    if hasattr(np, "_core"):
        return

    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        full_name = f"numpy._core.{submodule}"
        if full_name not in sys.modules:
            try:
                sys.modules[full_name] = importlib.import_module(f"numpy.core.{submodule}")
            except ImportError:
                pass


_install_numpy_pickle_shim()

import joblib
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except Exception:  # pragma: no cover - unavailable in some headless contexts
    omni_debug_draw = None

from legged_lab.assets.unitree import (
    UNITREE_G1_29DOF_CFG,
    UNITREE_S3_G1_29DOF_CFG,
    UNITREE_S3_G1_29DOF_MJCF_CFG,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1_LOCOMOTION_JOINT_NAMES


@configclass
class ReferenceSceneCfg(InteractiveSceneCfg):
    """Minimal scene for reference motion playback."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=850.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = None


@dataclass
class MotionClip:
    name: str
    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray
    source_dof_names: list[str]
    link_body_list: list[str]
    local_body_pos: np.ndarray | None
    root_vel_w: np.ndarray
    root_vel_b_xy: np.ndarray
    yaw_rate: np.ndarray
    heading_yaw_error: np.ndarray
    roll_pitch_yaw: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


@dataclass
class CycleAllState:
    clips: list[MotionClip]
    active_clip_ids: list[int]
    frame_cursors: np.ndarray
    next_clip_id: int


class StatusPanel:
    """Tiny Isaac UI panel used when running with a GUI."""

    def __init__(self, enabled: bool):
        self._label = None
        if not enabled:
            return
        try:
            import omni.ui as ui

            self._window = ui.Window("G1 AMP Reference Motion", width=520, height=300)
            with self._window.frame:
                with ui.VStack():
                    self._label = ui.Label("", word_wrap=True)
        except Exception as exc:  # pragma: no cover - UI is optional
            print(f"[Reference Viewer] Could not create status UI panel: {exc}")

    def update(self, text: str) -> None:
        if self._label is not None:
            self._label.text = text


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _select_robot_cfg(robot_asset: str) -> ArticulationCfg:
    preset = robot_asset.strip().lower()
    if preset in ("g1", "g1_29dof", "original_g1"):
        cfg = UNITREE_G1_29DOF_CFG
    elif preset in ("s3", "s3_g1_29dof"):
        cfg = UNITREE_S3_G1_29DOF_CFG
    elif preset in ("s3_g1_29dof_mjcf", "s3_mjcf"):
        cfg = UNITREE_S3_G1_29DOF_MJCF_CFG
    else:
        raise ValueError(f"Unsupported robot_asset={robot_asset!r}. Use g1_29dof, s3_g1_29dof, or s3_g1_29dof_mjcf.")

    cfg = cfg.replace(prim_path="{ENV_REGEX_NS}/Robot_ref")
    cfg.spawn.rigid_props.disable_gravity = True  # type: ignore[union-attr]
    cfg.spawn.articulation_props.enabled_self_collisions = False  # type: ignore[union-attr]
    cfg.spawn.activate_contact_sensors = False  # type: ignore[union-attr]
    cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)  # type: ignore[union-attr]
    return cfg


def _select_motion_paths(motion_dir: Path, motion_name: str, start_index: int, num_envs: int) -> list[Path]:
    motion_paths = sorted(motion_dir.glob("*.pkl"))
    if not motion_paths:
        raise FileNotFoundError(f"No .pkl files found in motion_dir={motion_dir}")

    # Keep this function's old sorted behavior as the default. Other selection
    # modes are diagnostic shortcuts for datasets whose sorted prefix is not
    # representative of the full task distribution.
    selection_mode = args_cli.motion_selection
    if motion_name:
        target = motion_name if motion_name.endswith(".pkl") else f"{motion_name}.pkl"
        matches = [path for path in motion_paths if path.name == target or path.stem == motion_name]
        if not matches:
            raise FileNotFoundError(f"Motion {motion_name!r} not found in {motion_dir}")
        selected = matches
    elif selection_mode == "spread":
        indices = np.linspace(0, len(motion_paths) - 1, num=max(num_envs, 1), dtype=int)
        selected = [motion_paths[index] for index in indices]
    elif selection_mode == "random":
        rng = np.random.default_rng(start_index)
        indices = rng.choice(len(motion_paths), size=min(num_envs, len(motion_paths)), replace=False)
        selected = [motion_paths[index] for index in indices]
    elif selection_mode in ("yaw", "path_turn", "misaligned"):
        scored = [(_motion_selection_score(path, selection_mode), path) for path in motion_paths]
        selected = [path for _, path in sorted(scored, key=lambda item: item[0], reverse=True)[:num_envs]]
    elif selection_mode == "balanced_modes":
        selected = _select_balanced_mode_paths(motion_paths, start_index, num_envs)
    elif selection_mode.startswith("mode_"):
        mode_name = selection_mode.removeprefix("mode_")
        mode_paths = [path for path in motion_paths if _motion_mode_name(path) == mode_name]
        if not mode_paths:
            raise FileNotFoundError(f"No motions with mode {mode_name!r} found in {motion_dir}")
        start = start_index % len(mode_paths)
        selected = mode_paths[start:] + mode_paths[:start]
    elif selection_mode == "cycle_all":
        start = start_index % len(motion_paths)
        selected = motion_paths[start:] + motion_paths[:start]
    else:
        selected = motion_paths[start_index : start_index + num_envs]
        if len(selected) < num_envs:
            selected += motion_paths[: num_envs - len(selected)]

    while selection_mode != "cycle_all" and len(selected) < num_envs:
        selected.append(selected[len(selected) % len(selected)])
    return selected if selection_mode == "cycle_all" else selected[:num_envs]


def _motion_mode_name(path: Path) -> str:
    mode_names = (
        "forward_slow",
        "forward_normal",
        "backward",
        "lateral_left",
        "lateral_right",
        "turn_left",
        "turn_right",
        "stand",
    )
    for mode_name in mode_names:
        if path.stem.endswith(f"_{mode_name}"):
            return mode_name
    try:
        raw = joblib.load(path)
        task_scope = raw.get("task_scope", {})
        if isinstance(task_scope, dict) and task_scope.get("mode_name"):
            return str(task_scope["mode_name"])
        task = raw.get("task", {})
        if isinstance(task, dict) and task.get("mode_name"):
            return str(task["mode_name"])
    except Exception:
        pass
    return ""


def _select_balanced_mode_paths(motion_paths: list[Path], start_index: int, num_envs: int) -> list[Path]:
    mode_order = (
        "forward_slow",
        "forward_normal",
        "backward",
        "lateral_left",
        "lateral_right",
        "turn_left",
        "turn_right",
        "stand",
    )
    paths_by_mode: dict[str, list[Path]] = {mode_name: [] for mode_name in mode_order}
    for path in motion_paths:
        mode_name = _motion_mode_name(path)
        if mode_name in paths_by_mode:
            paths_by_mode[mode_name].append(path)
    selected: list[Path] = []
    offset = max(start_index, 0)
    while len(selected) < num_envs:
        added = False
        for mode_name in mode_order:
            mode_paths = paths_by_mode[mode_name]
            if not mode_paths:
                continue
            selected.append(mode_paths[(offset + len(selected)) % len(mode_paths)])
            added = True
            if len(selected) >= num_envs:
                break
        if not added:
            raise FileNotFoundError("No mode-labeled motions found for balanced_modes selection.")
    return selected


def _motion_selection_score(path: Path, selection_mode: str) -> float:
    raw = joblib.load(path)
    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    raw_quat = np.asarray(raw["root_rot"], dtype=np.float32)
    quat = _normalize_quat_wxyz(_convert_xyzw_to_wxyz(raw_quat))
    fps = float(raw["fps"])
    root_vel = _compute_root_velocity(root_pos, fps)
    yaw_rate = _compute_yaw_rate(quat, fps)
    if selection_mode == "yaw":
        return float(np.mean(np.abs(yaw_rate)))
    if selection_mode == "path_turn":
        displacement = np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2])
        path_length = float(np.sum(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1)))
        return path_length / max(displacement, 1.0e-4)
    if selection_mode == "misaligned":
        alignment = _root_velocity_yaw_alignment(root_vel, quat)
        return float(alignment["abs_heading_yaw_error_mean_rad"])
    raise ValueError(f"Unsupported motion selection mode: {selection_mode}")


def _convert_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw[..., [3, 0, 1, 2]]


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1.0e-8)
    quat = quat / norm
    sign = np.where(quat[..., 0:1] < 0.0, -1.0, 1.0)
    return (quat * sign).astype(np.float32)


def _quat_to_rpy_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _compute_root_velocity(root_pos: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(root_pos, dtype=np.float32)
    if root_pos.shape[0] > 1:
        vel[:-1] = (root_pos[1:] - root_pos[:-1]) * fps
        vel[-1] = vel[-2]
    return vel


def _compute_yaw_rate(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    yaw = np.unwrap(_quat_to_rpy_wxyz(quat_wxyz)[:, 2])
    yaw_rate = np.zeros_like(yaw, dtype=np.float32)
    if yaw.shape[0] > 1:
        yaw_rate[:-1] = (yaw[1:] - yaw[:-1]) * fps
        yaw_rate[-1] = yaw_rate[-2]
    return yaw_rate


def _root_velocity_yaw_details(root_vel_w: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw = _quat_to_rpy_wxyz(quat_wxyz)[:, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    root_vel_b_xy = np.stack(
        [
            cos_yaw * root_vel_w[:, 0] + sin_yaw * root_vel_w[:, 1],
            -sin_yaw * root_vel_w[:, 0] + cos_yaw * root_vel_w[:, 1],
        ],
        axis=-1,
    ).astype(np.float32)
    heading = np.arctan2(root_vel_w[:, 1], root_vel_w[:, 0])
    heading_yaw_error = (heading - yaw + np.pi) % (2.0 * np.pi) - np.pi
    return root_vel_b_xy, heading_yaw_error.astype(np.float32)


def _root_velocity_yaw_alignment(root_vel_w: np.ndarray, quat_wxyz: np.ndarray) -> dict[str, float]:
    speed_xy = np.linalg.norm(root_vel_w[:, :2], axis=-1)
    moving = speed_xy > 0.15
    if not np.any(moving):
        return {
            "moving_frames": 0,
            "abs_heading_yaw_error_mean_rad": 0.0,
            "abs_heading_yaw_error_p95_rad": 0.0,
            "aligned_ratio_30deg": 0.0,
            "sideways_ratio_60deg": 0.0,
            "backward_ratio_120deg": 0.0,
        }
    _, error = _root_velocity_yaw_details(root_vel_w, quat_wxyz)
    abs_error = np.abs(error[moving])
    return {
        "moving_frames": int(np.sum(moving)),
        "abs_heading_yaw_error_mean_rad": float(np.mean(abs_error)),
        "abs_heading_yaw_error_p95_rad": float(np.quantile(abs_error, 0.95)),
        "aligned_ratio_30deg": float(np.mean(abs_error < np.deg2rad(30.0))),
        "sideways_ratio_60deg": float(np.mean(abs_error > np.deg2rad(60.0))),
        "backward_ratio_120deg": float(np.mean(abs_error > np.deg2rad(120.0))),
    }


def _rotate_vectors_wxyz(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q_vec = quat[1:4]
    uv = np.cross(q_vec, vectors)
    uuv = np.cross(q_vec, uv)
    return vectors + 2.0 * (quat[0] * uv + uuv)


def _align_dof_pos(raw_dof_pos: np.ndarray, source_dof_names: list[str], view_mode: str) -> tuple[np.ndarray, list[int] | None]:
    if view_mode == "training_current":
        return raw_dof_pos.astype(np.float32), None

    if not source_dof_names:
        raise ValueError("Selected view mode requires pkl['dof_names'], but the motion file does not contain it.")

    name_to_idx = {name: idx for idx, name in enumerate(source_dof_names)}
    missing = [name for name in G1_LOCOMOTION_JOINT_NAMES if name not in name_to_idx]
    if missing:
        raise ValueError(f"Selected view mode requires all target joints in pkl['dof_names']; missing={missing}")

    permutation = [name_to_idx[name] for name in G1_LOCOMOTION_JOINT_NAMES]
    return raw_dof_pos[:, permutation].astype(np.float32), permutation


def _load_motion_clip(path: Path, view_mode: str) -> MotionClip:
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Motion file {path} did not load as a dictionary.")

    for key in ("fps", "root_pos", "root_rot", "dof_pos"):
        if key not in raw:
            raise KeyError(f"Motion file {path} is missing required key {key!r}.")

    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    raw_quat = np.asarray(raw["root_rot"], dtype=np.float32)
    raw_dof_pos = np.asarray(raw["dof_pos"], dtype=np.float32)
    source_dof_names = list(raw.get("dof_names", []))
    link_body_list = list(raw.get("link_body_list", []))
    local_body_pos = raw.get("local_body_pos", None)
    if local_body_pos is not None:
        local_body_pos = np.asarray(local_body_pos, dtype=np.float32)

    dof_pos, _ = _align_dof_pos(raw_dof_pos, source_dof_names, view_mode)
    root_quat = _convert_xyzw_to_wxyz(raw_quat) if view_mode == "name_aligned_xyzw" else raw_quat
    root_quat = _normalize_quat_wxyz(root_quat)
    root_vel_w = _compute_root_velocity(root_pos, float(raw["fps"]))
    root_vel_b_xy, heading_yaw_error = _root_velocity_yaw_details(root_vel_w, root_quat)
    roll_pitch_yaw = _quat_to_rpy_wxyz(root_quat)

    return MotionClip(
        name=path.stem,
        fps=float(raw["fps"]),
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        dof_pos=dof_pos,
        source_dof_names=source_dof_names,
        link_body_list=link_body_list,
        local_body_pos=local_body_pos,
        root_vel_w=root_vel_w,
        root_vel_b_xy=root_vel_b_xy,
        yaw_rate=_compute_yaw_rate(root_quat, float(raw["fps"])),
        heading_yaw_error=heading_yaw_error,
        roll_pitch_yaw=roll_pitch_yaw,
    )


def _make_cycle_all_state(clips: list[MotionClip], num_envs: int) -> CycleAllState:
    active_clip_ids = [env_idx % len(clips) for env_idx in range(num_envs)]
    return CycleAllState(
        clips=clips,
        active_clip_ids=active_clip_ids,
        frame_cursors=np.zeros((num_envs,), dtype=np.float32),
        next_clip_id=num_envs % len(clips),
    )


def _cycle_clip_and_frame(state: CycleAllState, env_idx: int, speed: float) -> tuple[MotionClip, int, bool]:
    changed = False
    clip = state.clips[state.active_clip_ids[env_idx]]
    while state.frame_cursors[env_idx] >= clip.num_frames:
        state.active_clip_ids[env_idx] = state.next_clip_id
        state.next_clip_id = (state.next_clip_id + 1) % len(state.clips)
        state.frame_cursors[env_idx] = 0.0
        clip = state.clips[state.active_clip_ids[env_idx]]
        changed = True

    frame_idx = min(int(state.frame_cursors[env_idx]), clip.num_frames - 1)
    state.frame_cursors[env_idx] += max(float(speed), 0.0)
    return clip, frame_idx, changed


def _active_cycle_clips(state: CycleAllState) -> list[MotionClip]:
    return [state.clips[clip_id] for clip_id in state.active_clip_ids]


def _print_startup_validation(
    clips: list[MotionClip], view_mode: str, total_motion_count: int | None = None, cycle_all: bool = False
) -> None:
    clip = clips[0]
    print("=" * 80)
    print("[Reference Viewer] G1 AMP reference motion diagnostics")
    print(f"Motion count in view : {len(clips)}")
    if total_motion_count is not None:
        print(f"Motion count loaded  : {total_motion_count}")
    print(f"First motion         : {clip.name}")
    print(f"Playback mode        : {'cycle_all' if cycle_all else 'fixed selection'}")
    names = [item.name for item in clips[:8]]
    suffix = " ..." if len(clips) > len(names) else ""
    print(f"Selected motions     : {', '.join(names)}{suffix}")
    print(f"Frames/fps/duration : {clip.num_frames} / {clip.fps:g} / {clip.duration:.3f}s")
    print(f"View mode           : {view_mode}")

    if clip.source_dof_names:
        mismatches = [
            (idx, target, source)
            for idx, (target, source) in enumerate(zip(G1_LOCOMOTION_JOINT_NAMES, clip.source_dof_names))
            if target != source
        ]
        print(f"Joint mismatch count: {len(mismatches)} / {len(G1_LOCOMOTION_JOINT_NAMES)}")
        for idx, target, source in mismatches[:12]:
            print(f"  mismatch[{idx:02d}]: training={target:<32} pkl={source}")
    else:
        print("Joint mismatch count: unavailable because pkl['dof_names'] is absent")

    selected_quat = "xyzw -> wxyz" if view_mode == "name_aligned_xyzw" else "raw interpreted as wxyz"
    print(f"Quaternion handling : {selected_quat}")
    print(f"First root quat     : {clip.root_quat_wxyz[0].tolist()} (wxyz after selected handling)")
    print(f"Root z range        : {clip.root_pos[:, 2].min():.3f} .. {clip.root_pos[:, 2].max():.3f} m")
    print(f"Root speed mean/max : {np.linalg.norm(clip.root_vel_w[:, :2], axis=-1).mean():.3f} / {np.linalg.norm(clip.root_vel_w[:, :2], axis=-1).max():.3f} m/s")
    path_length = float(np.sum(np.linalg.norm(np.diff(clip.root_pos[:, :2], axis=0), axis=1)))
    displacement = float(np.linalg.norm(clip.root_pos[-1, :2] - clip.root_pos[0, :2]))
    print(
        "Root path           : "
        f"displacement={displacement:.3f}m path={path_length:.3f}m "
        f"path/disp={path_length / max(displacement, 1.0e-4):.3f}"
    )
    print(
        "Root local velocity : "
        f"vx_mean={clip.root_vel_b_xy[:, 0].mean():+.3f}m/s "
        f"|vy|_mean={np.abs(clip.root_vel_b_xy[:, 1]).mean():.3f}m/s"
    )
    alignment = _root_velocity_yaw_alignment(clip.root_vel_w, clip.root_quat_wxyz)
    print(
        "Root vel/yaw align : "
        f"moving={alignment['moving_frames']} "
        f"mean_abs_err={alignment['abs_heading_yaw_error_mean_rad']:.3f}rad "
        f"p95={alignment['abs_heading_yaw_error_p95_rad']:.3f}rad "
        f"aligned<30deg={alignment['aligned_ratio_30deg']:.3f} "
        f"sideways>60deg={alignment['sideways_ratio_60deg']:.3f} "
        f"backward>120deg={alignment['backward_ratio_120deg']:.3f}"
    )
    print("=" * 80)


def _make_marker(name: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=f"/Visuals/G1Reference/{name}",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            },
        )
    )


def _create_marker_groups(robot: Articulation) -> list[tuple[str, list[int], VisualizationMarkers]]:
    body_names = robot.data.body_names
    specs = [
        ("feet", ["left_ankle_roll_link", "right_ankle_roll_link", "left_toe_link", "right_toe_link"], (0.0, 0.95, 0.25), 0.035),
        ("wrists", ["left_wrist_yaw_link", "right_wrist_yaw_link"], (0.95, 0.2, 0.95), 0.03),
        ("shoulders", ["left_shoulder_roll_link", "right_shoulder_roll_link"], (1.0, 0.82, 0.1), 0.03),
        ("torso", ["pelvis", "torso_link", "head_link"], (0.1, 0.55, 1.0), 0.04),
    ]
    groups: list[tuple[str, list[int], VisualizationMarkers]] = []
    for marker_name, names, color, radius in specs:
        ids = [body_names.index(name) for name in names if name in body_names]
        missing = [name for name in names if name not in body_names]
        if missing:
            print(f"[Reference Viewer] Marker group {marker_name!r} missing bodies: {missing}")
        if ids:
            groups.append((marker_name, ids, _make_marker(marker_name, color, radius)))
    return groups


def _create_ghost_markers(history_steps: int) -> list[VisualizationMarkers]:
    markers = []
    colors = [
        (1.0, 1.0, 1.0),
        (0.65, 0.9, 1.0),
        (0.35, 0.65, 1.0),
        (0.2, 0.35, 0.9),
    ]
    for idx in range(max(history_steps, 0)):
        color = colors[idx] if idx < len(colors) else colors[-1]
        radius = max(0.014, 0.026 - 0.003 * idx)
        markers.append(_make_marker(f"amp_ghost_{idx}", color, radius))
    return markers


def _frame_index(clip: MotionClip, frame_counter: int, loop: bool, speed: float) -> int:
    scaled = int(frame_counter * speed)
    if loop:
        return scaled % clip.num_frames
    return min(scaled, clip.num_frames - 1)


def _update_reference_state(
    scene: InteractiveScene,
    robot: Articulation,
    clips: list[MotionClip],
    frame_counter: int,
    joint_ids: list[int],
    loop: bool,
    speed: float,
    height_offset: float,
    cycle_state: CycleAllState | None = None,
) -> tuple[list[MotionClip], list[int], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root_states = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    active_clips: list[MotionClip] = []
    frame_indices: list[int] = []
    changed_env_ids: list[int] = []
    root_velocities = []
    yaw_rates = []
    root_forward_vectors = []
    root_velocities_b_xy = []
    heading_yaw_errors = []

    for env_idx in range(scene.num_envs):
        if cycle_state is None:
            clip = clips[env_idx]
            frame_idx = _frame_index(clip, frame_counter, loop, speed)
            changed = False
        else:
            clip, frame_idx, changed = _cycle_clip_and_frame(cycle_state, env_idx, speed)
        active_clips.append(clip)
        frame_indices.append(frame_idx)
        if changed:
            changed_env_ids.append(env_idx)
        root_pos = clip.root_pos[frame_idx].copy()
        root_pos[2] += height_offset
        root_pos += scene.env_origins[env_idx].detach().cpu().numpy()
        root_quat = clip.root_quat_wxyz[frame_idx]

        root_states[env_idx, :3] = torch.as_tensor(root_pos, device=scene.device)
        root_states[env_idx, 3:7] = torch.as_tensor(root_quat, device=scene.device)
        root_states[env_idx, 7:10] = torch.as_tensor(clip.root_vel_w[frame_idx], device=scene.device)
        root_states[env_idx, 10:13] = 0.0
        root_states[env_idx, 12] = float(clip.yaw_rate[frame_idx])
        joint_pos[env_idx, joint_ids] = torch.as_tensor(clip.dof_pos[frame_idx], device=scene.device)
        root_velocities.append(clip.root_vel_w[frame_idx])
        yaw_rates.append(clip.yaw_rate[frame_idx])
        root_forward_vectors.append(_rotate_vectors_wxyz(root_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        root_velocities_b_xy.append(clip.root_vel_b_xy[frame_idx])
        heading_yaw_errors.append(clip.heading_yaw_error[frame_idx])

    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(joint_pos[:, joint_ids], joint_vel[:, joint_ids], joint_ids=joint_ids)
    return (
        active_clips,
        frame_indices,
        changed_env_ids,
        np.asarray(root_velocities),
        np.asarray(yaw_rates),
        np.asarray(root_forward_vectors),
        np.asarray(root_velocities_b_xy),
        np.asarray(heading_yaw_errors),
    )


def _visualize_body_markers(robot: Articulation, marker_groups: list[tuple[str, list[int], VisualizationMarkers]]) -> None:
    for _, body_ids, marker in marker_groups:
        positions = robot.data.body_pos_w[:, body_ids, :].reshape(-1, 3)
        marker.visualize(translations=positions)


def _ghost_key_body_indices(clip: MotionClip) -> list[int]:
    if not clip.link_body_list:
        return []
    desired = [
        "pelvis",
        "torso_link",
        "head_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_toe_link",
        "right_toe_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    ]
    return [clip.link_body_list.index(name) for name in desired if name in clip.link_body_list]


def _visualize_ghosts(
    scene: InteractiveScene,
    clips: list[MotionClip],
    frame_indices: list[int],
    loop: bool,
    height_offset: float,
    ghost_markers: list[VisualizationMarkers],
    ghost_body_indices: list[int],
) -> None:
    if not ghost_markers or not ghost_body_indices:
        return
    for ghost_idx, marker in enumerate(ghost_markers):
        translations = []
        for env_idx, clip in enumerate(clips):
            if clip.local_body_pos is None:
                continue
            frame = frame_indices[env_idx] + ghost_idx
            if loop:
                frame = frame % clip.num_frames
            elif frame >= clip.num_frames:
                frame = clip.num_frames - 1
            root_pos = clip.root_pos[frame].copy()
            root_pos[2] += height_offset
            root_pos += scene.env_origins[env_idx].detach().cpu().numpy()
            local = clip.local_body_pos[frame, ghost_body_indices, :]
            rotated = _rotate_vectors_wxyz(clip.root_quat_wxyz[frame], local)
            translations.append(root_pos[None, :] + rotated)
        if translations:
            marker.visualize(translations=torch.as_tensor(np.concatenate(translations, axis=0), device=scene.device))


def _clear_trails_for_envs(trail_buffers: dict[str, list[list[list[float]]]], env_ids: list[int]) -> None:
    for env_idx in env_ids:
        for env_trails in trail_buffers.values():
            env_trails[env_idx].clear()


def _append_trails(
    robot: Articulation,
    trail_buffers: dict[str, list[list[list[float]]]],
    body_name_to_id: dict[str, int],
    trail_length: int,
) -> None:
    root_pos = robot.data.root_pos_w.detach().cpu().numpy()
    body_pos = robot.data.body_pos_w.detach().cpu().numpy()
    sources = {
        "root": root_pos,
    }
    for key, body_name in (
        ("left_foot", "left_ankle_roll_link"),
        ("right_foot", "right_ankle_roll_link"),
    ):
        if body_name in body_name_to_id:
            sources[key] = body_pos[:, body_name_to_id[body_name], :]

    for key, values in sources.items():
        for env_idx, pos in enumerate(values):
            trail_buffers[key][env_idx].append(pos.tolist())
            if len(trail_buffers[key][env_idx]) > trail_length:
                trail_buffers[key][env_idx].pop(0)


def _draw_debug_lines(
    draw_interface,
    trail_buffers: dict[str, list[list[list[float]]]],
    root_velocities: np.ndarray,
    yaw_rates: np.ndarray,
    root_forward_vectors: np.ndarray,
) -> None:
    if draw_interface is None:
        return
    draw_interface.clear_lines()
    colors = {
        "root": [0.1, 0.75, 1.0, 1.0],
        "left_foot": [0.1, 1.0, 0.25, 1.0],
        "right_foot": [1.0, 0.35, 0.15, 1.0],
    }
    for key, env_trails in trail_buffers.items():
        for trail in env_trails:
            if len(trail) < 2:
                continue
            starts = trail[:-1]
            ends = trail[1:]
            draw_interface.draw_lines(starts, ends, [colors[key]] * len(starts), [2.0] * len(starts))

    root_trails = trail_buffers["root"]
    arrow_starts = []
    arrow_ends = []
    arrow_colors = []
    forward_starts = []
    forward_ends = []
    forward_colors = []
    yaw_starts = []
    yaw_ends = []
    yaw_colors = []
    for env_idx, trail in enumerate(root_trails):
        if not trail:
            continue
        start = np.asarray(trail[-1], dtype=np.float32)
        vel = root_velocities[env_idx].copy()
        vel[2] = 0.0
        arrow_starts.append(start.tolist())
        arrow_ends.append((start + 0.35 * vel).tolist())
        arrow_colors.append([0.0, 1.0, 1.0, 1.0])
        forward = root_forward_vectors[env_idx].copy()
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm > 1.0e-6:
            forward = forward / norm
        forward_starts.append((start + np.array([0.0, 0.0, 0.025])).tolist())
        forward_ends.append((start + np.array([0.0, 0.0, 0.025]) + 0.45 * forward).tolist())
        forward_colors.append([1.0, 0.15, 0.15, 1.0])
        yaw_starts.append((start + np.array([0.0, 0.0, 0.05])).tolist())
        yaw_height = np.clip(yaw_rates[env_idx], -2.0, 2.0) * 0.15
        yaw_ends.append((start + np.array([0.0, 0.0, 0.05 + yaw_height])).tolist())
        yaw_colors.append([1.0, 0.7, 0.0, 1.0] if yaw_height >= 0.0 else [0.6, 0.4, 1.0, 1.0])
    if arrow_starts:
        draw_interface.draw_lines(arrow_starts, arrow_ends, arrow_colors, [4.0] * len(arrow_starts))
        draw_interface.draw_lines(forward_starts, forward_ends, forward_colors, [4.0] * len(forward_starts))
        draw_interface.draw_lines(yaw_starts, yaw_ends, yaw_colors, [5.0] * len(yaw_starts))


def _clearance_and_contact(
    robot: Articulation,
    body_name_to_id: dict[str, int],
    height_offset: float,
) -> tuple[float, float, bool, bool]:
    body_pos = robot.data.body_pos_w.detach().cpu().numpy()
    env_origin = robot.data.root_pos_w.detach().cpu().numpy()
    # The visual root is lifted; subtract that lift so contact inference uses the source motion height.
    def body_clearance(name: str) -> float:
        if name not in body_name_to_id:
            return float("nan")
        return float(body_pos[0, body_name_to_id[name], 2] - height_offset)

    left = body_clearance("left_ankle_roll_link")
    right = body_clearance("right_ankle_roll_link")
    return left, right, bool(left < 0.035), bool(right < 0.035)


def _status_text(
    clips: list[MotionClip],
    frame_indices: list[int],
    root_velocities: np.ndarray,
    yaw_rates: np.ndarray,
    root_velocities_b_xy: np.ndarray,
    heading_yaw_errors: np.ndarray,
    robot: Articulation,
    body_name_to_id: dict[str, int],
    height_offset: float,
    view_mode: str,
) -> str:
    clip = clips[0]
    frame = frame_indices[0]
    rpy = clip.roll_pitch_yaw[frame]
    speed_xy = float(np.linalg.norm(root_velocities[0, :2]))
    left_clearance, right_clearance, left_contact, right_contact = _clearance_and_contact(
        robot, body_name_to_id, height_offset
    )
    phase = frame / max(clip.num_frames - 1, 1)
    return (
        f"mode={view_mode} motion={clip.name}\n"
        f"frame={frame}/{clip.num_frames - 1} time={frame / clip.fps:.3f}s phase={phase:.3f}\n"
        f"root_z={clip.root_pos[frame, 2]:.3f} rpy=({rpy[0]:+.3f}, {rpy[1]:+.3f}, {rpy[2]:+.3f}) rad\n"
        f"speed_xy={speed_xy:.3f} m/s local_v=({root_velocities_b_xy[0, 0]:+.3f}, {root_velocities_b_xy[0, 1]:+.3f}) m/s\n"
        f"yaw_rate={yaw_rates[0]:+.3f} rad/s heading-yaw={heading_yaw_errors[0]:+.3f} rad\n"
        f"foot_clearance_raw L/R={left_clearance:.3f}/{right_clearance:.3f} m "
        f"contact L/R={left_contact}/{right_contact}"
    )


def main() -> None:
    motion_dir = _resolve_path(args_cli.motion_dir)
    motion_paths = _select_motion_paths(motion_dir, args_cli.motion_name, args_cli.start_index, args_cli.num_envs)
    loaded_clips = [_load_motion_clip(path, args_cli.view_mode) for path in motion_paths]
    cycle_all = args_cli.motion_selection == "cycle_all" and not args_cli.motion_name
    cycle_state = _make_cycle_all_state(loaded_clips, args_cli.num_envs) if cycle_all else None
    clips = _active_cycle_clips(cycle_state) if cycle_state is not None else loaded_clips
    _print_startup_validation(clips, args_cli.view_mode, total_motion_count=len(loaded_clips), cycle_all=cycle_all)

    loop = args_cli.loop or not args_cli.no_loop
    sim_dt = 1.0 / clips[0].fps
    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt,
        device=args_cli.device,
        gravity=(0.0, 0.0, 0.0) if args_cli.zero_world_gravity else (0.0, 0.0, -9.81),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, -3.0, 1.6], [0.0, 0.0, 0.85])

    scene_cfg = ReferenceSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.robot = _select_robot_cfg(args_cli.robot_asset)
    print("[Reference Viewer] Creating IsaacLab InteractiveScene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[Reference Viewer] Resetting simulation context...", flush=True)
    sim.reset()
    print("[Reference Viewer] Simulation reset complete; updating scene buffers...", flush=True)
    scene.update(sim_dt)
    print("[Reference Viewer] Scene buffers updated.", flush=True)

    robot: Articulation = scene["robot"]
    body_name_to_id = {name: idx for idx, name in enumerate(robot.data.body_names)}
    joint_name_to_id = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    missing_joints = [name for name in G1_LOCOMOTION_JOINT_NAMES if name not in joint_name_to_id]
    if missing_joints:
        raise RuntimeError(f"Robot asset is missing G1 locomotion joints: {missing_joints}")
    joint_ids = [joint_name_to_id[name] for name in G1_LOCOMOTION_JOINT_NAMES]

    print(f"[Reference Viewer] Robot joint count={len(robot.data.joint_names)} body count={len(robot.data.body_names)}")
    print(f"[Reference Viewer] Writing motion columns to joint ids={joint_ids}")
    if args_cli.zero_world_gravity:
        print("[Reference Viewer] World gravity is zero; robot gravity/collisions/contact are also disabled.")
    else:
        print("[Reference Viewer] Robot gravity/collisions/contact are disabled; world gravity is left at default.")

    marker_groups = _create_marker_groups(robot)
    ghost_markers = _create_ghost_markers(args_cli.history_steps)
    ghost_body_indices = _ghost_key_body_indices(clips[0])
    status_panel = StatusPanel(enabled=not args_cli.headless)
    draw_interface = None
    if omni_debug_draw is not None and not args_cli.headless:
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    trail_buffers: dict[str, list[list[list[float]]]] = {
        "root": [[] for _ in range(args_cli.num_envs)],
        "left_foot": [[] for _ in range(args_cli.num_envs)],
        "right_foot": [[] for _ in range(args_cli.num_envs)],
    }

    frame_counter = 0
    next_frame_wall_time = time.perf_counter()
    max_steps = args_cli.max_steps if args_cli.max_steps > 0 else None

    while simulation_app.is_running():
        with torch.inference_mode():
            (
                active_clips,
                frame_indices,
                changed_env_ids,
                root_velocities,
                yaw_rates,
                root_forward_vectors,
                root_velocities_b_xy,
                heading_yaw_errors,
            ) = _update_reference_state(
                scene,
                robot,
                clips,
                frame_counter,
                joint_ids,
                loop,
                args_cli.speed,
                args_cli.height_offset,
                cycle_state,
            )
            sim.render()
            scene.update(sim_dt)
            _visualize_body_markers(robot, marker_groups)
            _visualize_ghosts(
                scene,
                active_clips,
                frame_indices,
                loop,
                args_cli.height_offset,
                ghost_markers,
                ghost_body_indices,
            )
            _clear_trails_for_envs(trail_buffers, changed_env_ids)
            _append_trails(robot, trail_buffers, body_name_to_id, args_cli.trail_length)
            _draw_debug_lines(draw_interface, trail_buffers, root_velocities, yaw_rates, root_forward_vectors)

            if frame_counter % max(args_cli.print_interval, 1) == 0:
                text = _status_text(
                    active_clips,
                    frame_indices,
                    root_velocities,
                    yaw_rates,
                    root_velocities_b_xy,
                    heading_yaw_errors,
                    robot,
                    body_name_to_id,
                    args_cli.height_offset,
                    args_cli.view_mode,
                )
                print("[Reference Viewer]\n" + text)
                status_panel.update(text)

            frame_counter += 1
            if max_steps is not None and frame_counter >= max_steps:
                break

        if args_cli.real_time:
            next_frame_wall_time += sim_dt / max(args_cli.speed, 1.0e-6)
            sleep_time = next_frame_wall_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    if draw_interface is not None:
        draw_interface.clear_lines()
    if hasattr(scene, "close"):
        scene.close()
    print(f"[Reference Viewer] Finished after {frame_counter} frames.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
