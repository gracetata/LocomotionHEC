#!/usr/bin/env python3
"""Build deterministic, arm-only ArmHack Stand test data from the arm CSV.

The source trajectory is traversed offline. Representative poses cover the
normalized 14-DoF arm posture space, representative motion windows cover
different high-motion regions, and synthesized samples are generated once with
a fixed seed. Isaac Sim playback therefore contains no runtime sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "Reference Data" / "ArmHack" / "StandPerturb" / "g1_arm_trajectory_named_50hz.csv"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent / "TestData" / "ArmOnly"


@dataclass(frozen=True)
class MotionWindow:
    start_s: float
    end_s: float
    max_joint_span_rad: float
    dominant_joint: str
    motion_score: float
    descriptor: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "time_s" not in reader.fieldnames:
            raise ValueError(f"CSV must contain time_s and named arm joints: {path}")
        joint_names = [name for name in reader.fieldnames if name != "time_s"]
        rows = list(reader)

    if len(joint_names) != 14 or not rows:
        raise ValueError(f"Expected 14 arm joints and at least one row, got {len(joint_names)} joints.")
    non_arm_joint_names = [
        name
        for name in joint_names
        if not name.startswith(("left_", "right_"))
        or not any(part in name for part in ("shoulder", "elbow", "wrist"))
    ]
    if non_arm_joint_names:
        raise ValueError(f"Arm-only source contains non-arm joints: {non_arm_joint_names}")
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    positions = np.asarray([[float(row[name]) for name in joint_names] for row in rows], dtype=np.float64)
    times -= times[0]
    if not np.all(np.isfinite(positions)) or not np.all(np.diff(times) >= 0.0):
        raise ValueError("Source trajectory contains non-finite values or non-monotonic time.")
    return joint_names, times, positions


def _write_trajectory(path: Path, joint_names: list[str], positions: np.ndarray, fps: float) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", *joint_names])
        for index, pose in enumerate(positions):
            writer.writerow([f"{index / fps:.8f}", *(f"{value:.8f}" for value in pose)])
    return (len(positions) - 1) / fps if len(positions) > 1 else 0.0


def _min_jerk(start: np.ndarray, end: np.ndarray, duration_s: float, fps: float) -> np.ndarray:
    count = max(int(round(duration_s * fps)), 1)
    phase = np.arange(1, count + 1, dtype=np.float64) / count
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return start[None, :] + blend[:, None] * (end - start)[None, :]


def _pose_sequence(
    poses: np.ndarray,
    labels: list[str],
    fps: float,
    hold_s: float,
    transition_s: float,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    hold_count = max(int(round(hold_s * fps)), 1)
    frames: list[np.ndarray] = []
    timeline: list[dict[str, float | str]] = []
    cursor = 0
    for index, (pose, label) in enumerate(zip(poses, labels, strict=True)):
        if index > 0:
            transition = _min_jerk(poses[index - 1], pose, transition_s, fps)
            transition_start = cursor / fps
            frames.extend(transition)
            cursor += len(transition)
            timeline.append(
                {
                    "kind": "smooth_transition",
                    "label": f"{labels[index - 1]}_to_{label}",
                    "start_s": transition_start,
                    "end_s": cursor / fps,
                }
            )
        hold_start = cursor / fps
        frames.extend(np.repeat(pose[None, :], hold_count, axis=0))
        cursor += hold_count
        timeline.append({"kind": "static_hold", "label": label, "start_s": hold_start, "end_s": cursor / fps})
    return np.asarray(frames), timeline


def _concat_sequences(sequences: list[np.ndarray], fps: float, bridge_s: float) -> np.ndarray:
    if not sequences:
        raise ValueError("At least one sequence is required.")
    frames = [sequences[0]]
    for sequence in sequences[1:]:
        frames.append(_min_jerk(frames[-1][-1], sequence[0], bridge_s, fps))
        frames.append(sequence[1:] if len(sequence) > 1 else sequence)
    return np.concatenate(frames, axis=0)


def _concat_labeled_sequences(
    sequences: list[tuple[str, np.ndarray]], fps: float, bridge_s: float
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    if not sequences:
        raise ValueError("At least one labeled sequence is required.")
    first_label, first_sequence = sequences[0]
    frames = [first_sequence]
    cursor = len(first_sequence)
    timeline: list[dict[str, float | str]] = [
        {
            "kind": "section",
            "label": first_label,
            "start_s": 0.0,
            "end_s": (cursor - 1) / fps,
        }
    ]
    for label, sequence in sequences[1:]:
        bridge = _min_jerk(frames[-1][-1], sequence[0], bridge_s, fps)
        bridge_start = (cursor - 1) / fps
        frames.append(bridge)
        cursor += len(bridge)
        bridge_end = (cursor - 1) / fps
        timeline.append(
            {
                "kind": "smooth_section_bridge",
                "label": f"to_{label}",
                "start_s": bridge_start,
                "end_s": bridge_end,
            }
        )
        frames.append(sequence[1:] if len(sequence) > 1 else sequence)
        cursor += max(len(sequence) - 1, 1)
        timeline.append(
            {
                "kind": "section",
                "label": label,
                "start_s": bridge_end,
                "end_s": (cursor - 1) / fps,
            }
        )
    return np.concatenate(frames, axis=0), timeline


def _select_representative_poses(
    times: np.ndarray,
    positions: np.ndarray,
    count: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.percentile(positions, 1.0, axis=0)
    upper = np.percentile(positions, 99.0, axis=0)
    center = np.median(positions, axis=0)
    scale = np.maximum(np.percentile(positions, 95.0, axis=0) - np.percentile(positions, 5.0, axis=0), 0.05)

    candidate_indices = np.arange(0, len(times), max(stride, 1), dtype=np.int64)
    normalized = np.clip((positions[candidate_indices] - center) / scale, -3.0, 3.0)
    first = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    selected_local = [first]
    min_distance = np.linalg.norm(normalized - normalized[first], axis=1)
    while len(selected_local) < count:
        next_index = int(np.argmax(min_distance))
        selected_local.append(next_index)
        distance = np.linalg.norm(normalized - normalized[next_index], axis=1)
        min_distance = np.minimum(min_distance, distance)

    selected_indices = candidate_indices[np.asarray(selected_local, dtype=np.int64)]
    return selected_indices, positions[selected_indices], lower, upper


def _motion_candidates(
    joint_names: list[str],
    times: np.ndarray,
    positions: np.ndarray,
    joint_scale: np.ndarray,
    window_s: float,
    stride_s: float,
) -> list[MotionWindow]:
    candidates: list[MotionWindow] = []
    for start_s in np.arange(0.0, times[-1] - window_s + 1.0e-9, stride_s):
        lower_index = int(np.searchsorted(times, start_s, side="left"))
        upper_index = int(np.searchsorted(times, start_s + window_s, side="right"))
        window = positions[lower_index:upper_index]
        window_times = times[lower_index:upper_index]
        if len(window) < 3:
            continue
        span = np.ptp(window, axis=0)
        if float(np.max(span)) < 0.20:
            continue
        delta_t = np.maximum(np.diff(window_times), 1.0e-6)
        velocity = np.diff(window, axis=0) / delta_t[:, None]
        velocity_rms = np.sqrt(np.mean(velocity**2, axis=0))
        mean_pose = np.mean(window, axis=0)
        endpoint_delta = window[-1] - window[0]
        descriptor = np.concatenate(
            (
                0.30 * mean_pose / joint_scale,
                span / joint_scale,
                0.20 * window_s * velocity_rms / joint_scale,
                0.40 * endpoint_delta / joint_scale,
            )
        )
        motion_score = float(np.linalg.norm(span / joint_scale) + 0.25 * np.linalg.norm(endpoint_delta / joint_scale))
        dominant_index = int(np.argmax(span))
        candidates.append(
            MotionWindow(
                start_s=float(start_s),
                end_s=float(start_s + window_s),
                max_joint_span_rad=float(span[dominant_index]),
                dominant_joint=joint_names[dominant_index],
                motion_score=motion_score,
                descriptor=descriptor,
            )
        )
    return candidates


def _select_motion_windows(candidates: list[MotionWindow], count: int, minimum_separation_s: float) -> list[MotionWindow]:
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} motion windows passed the filter; need {count}.")
    descriptors = np.stack([candidate.descriptor for candidate in candidates])
    descriptors = (descriptors - np.mean(descriptors, axis=0)) / np.maximum(np.std(descriptors, axis=0), 1.0e-6)
    scores = np.asarray([candidate.motion_score for candidate in candidates])
    selected = [int(np.argmax(scores))]
    while len(selected) < count:
        distance = np.min(
            np.stack([np.linalg.norm(descriptors - descriptors[index], axis=1) for index in selected]), axis=0
        )
        valid = np.ones(len(candidates), dtype=bool)
        for index, candidate in enumerate(candidates):
            if any(abs(candidate.start_s - candidates[chosen].start_s) < minimum_separation_s for chosen in selected):
                valid[index] = False
        objective = distance * (0.5 + scores / max(float(np.max(scores)), 1.0e-6))
        objective[~valid] = -np.inf
        next_index = int(np.argmax(objective))
        if not np.isfinite(objective[next_index]):
            raise ValueError("Unable to find enough separated representative motion windows.")
        selected.append(next_index)
    return sorted((candidates[index] for index in selected), key=lambda item: item.start_s)


def _resample_source_window(
    times: np.ndarray,
    positions: np.ndarray,
    start_s: float,
    end_s: float,
    output_duration_s: float,
    fps: float,
) -> np.ndarray:
    count = int(round(output_duration_s * fps)) + 1
    query_times = np.linspace(start_s, end_s, count)
    return np.stack([np.interp(query_times, times, positions[:, joint]) for joint in range(positions.shape[1])], axis=1)


def _synthesize_poses(
    representative_poses: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Create arm-only poses by interpolating between two real arm poses.

    Pairwise convex interpolation stays inside the measured 14-DoF arm pose
    manifold envelope.  No leg, waist, root, or full-body values are created.
    """
    synthesized: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for index in range(count):
        parent_indices = rng.choice(len(representative_poses), size=2, replace=False)
        alpha = float(rng.uniform(0.25, 0.75))
        pose = (1.0 - alpha) * representative_poses[parent_indices[0]] + alpha * representative_poses[
            parent_indices[1]
        ]
        synthesized.append(pose)
        metadata.append(
            {
                "pose_id": f"synth_pose_{index + 1:02d}",
                "parent_representative_pose_indices": [int(value + 1) for value in parent_indices],
                "interpolation_alpha": alpha,
                "parent_weights": [1.0 - alpha, alpha],
                "data_scope": "arm_only_14_dof",
            }
        )
    return np.asarray(synthesized), metadata


def _synthesize_trajectories(
    representative_poses: np.ndarray,
    count: int,
    rng: np.random.Generator,
    fps: float,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Create 20 s arm-only trajectories between valid interpolated arm poses."""
    trajectories: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for trajectory_index in range(count):
        anchors: list[np.ndarray] = []
        anchor_metadata: list[dict[str, object]] = []
        for _ in range(4):
            parent_indices = rng.choice(len(representative_poses), size=2, replace=False)
            alpha = float(rng.uniform(0.15, 0.85))
            anchor = (1.0 - alpha) * representative_poses[parent_indices[0]] + alpha * representative_poses[
                parent_indices[1]
            ]
            anchors.append(anchor)
            anchor_metadata.append(
                {
                    "parents": [int(value + 1) for value in parent_indices],
                    "interpolation_alpha": alpha,
                    "weights": [1.0 - alpha, alpha],
                }
            )
        hold_start = np.repeat(anchors[0][None, :], int(round(2.0 * fps)) + 1, axis=0)
        transitions = [_min_jerk(anchors[index], anchors[index + 1], 5.0, fps) for index in range(3)]
        hold_end = np.repeat(anchors[-1][None, :], int(round(3.0 * fps)), axis=0)
        trajectory = np.concatenate([hold_start, *transitions, hold_end], axis=0)
        trajectories.append(trajectory)
        metadata.append(
            {
                "trajectory_id": f"synth_trajectory_{trajectory_index + 1:02d}",
                "duration_s": (len(trajectory) - 1) / fps,
                "anchor_generation": anchor_metadata,
                "profile": "quintic minimum-jerk, 2 s initial hold, three 5 s transitions, 3 s final hold",
                "data_scope": "arm_only_14_dof",
            }
        )
    return trajectories, metadata


def _write_pose_catalog(
    path: Path,
    joint_names: list[str],
    representative_indices: np.ndarray,
    times: np.ndarray,
    representative_poses: np.ndarray,
    synthesized_poses: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pose_id", "kind", "source_time_s", *joint_names])
        for index, (source_index, pose) in enumerate(
            zip(representative_indices, representative_poses, strict=True), start=1
        ):
            writer.writerow(
                [f"representative_pose_{index:02d}", "source", f"{times[source_index]:.8f}", *(f"{v:.8f}" for v in pose)]
            )
        for index, pose in enumerate(synthesized_poses, start=1):
            writer.writerow([f"synth_pose_{index:02d}", "synthesized", "", *(f"{v:.8f}" for v in pose)])


def build_suite(args: argparse.Namespace) -> dict[str, object]:
    joint_names, times, positions = _load_source(args.source)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    representative_indices, representative_poses, _lower, _upper = _select_representative_poses(
        times, positions, args.representative_pose_count, args.pose_stride
    )
    joint_scale = np.maximum(np.percentile(positions, 95.0, axis=0) - np.percentile(positions, 5.0, axis=0), 0.05)
    synthesized_poses, synthesized_pose_metadata = _synthesize_poses(
        representative_poses, args.synthesized_pose_count, rng
    )

    pose_catalog_path = output / "poses" / "arm_pose_catalog.csv"
    pose_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pose_catalog(
        pose_catalog_path,
        joint_names,
        representative_indices,
        times,
        representative_poses,
        synthesized_poses,
    )

    representative_pose_labels = [f"representative_pose_{index + 1:02d}" for index in range(len(representative_poses))]
    representative_pose_sequence, representative_pose_timeline = _pose_sequence(
        representative_poses,
        representative_pose_labels,
        args.fps,
        args.pose_hold_s,
        args.pose_transition_s,
    )
    synthesized_pose_labels = [f"synth_pose_{index + 1:02d}" for index in range(len(synthesized_poses))]
    synthesized_pose_sequence, synthesized_pose_timeline = _pose_sequence(
        synthesized_poses,
        synthesized_pose_labels,
        args.fps,
        args.pose_hold_s,
        args.pose_transition_s,
    )

    files: dict[str, dict[str, object]] = {}
    for name, sequence, timeline in (
        ("representative_poses", representative_pose_sequence, representative_pose_timeline),
        ("synthesized_poses", synthesized_pose_sequence, synthesized_pose_timeline),
    ):
        path = output / "sequences" / f"{name}_arm_only_sequence_50hz.csv"
        duration = _write_trajectory(path, joint_names, sequence, args.fps)
        files[name] = {
            "path": path.relative_to(output).as_posix(),
            "duration_s": duration,
            "timeline": timeline,
        }

    candidates = _motion_candidates(
        joint_names,
        times,
        positions,
        joint_scale,
        args.source_window_s,
        args.window_stride_s,
    )
    selected_windows = _select_motion_windows(
        candidates, args.representative_trajectory_count, args.minimum_window_separation_s
    )
    representative_trajectories: list[np.ndarray] = []
    representative_trajectory_metadata: list[dict[str, object]] = []
    for index, window in enumerate(selected_windows, start=1):
        trajectory = _resample_source_window(
            times,
            positions,
            window.start_s,
            window.end_s,
            args.playback_window_s,
            args.fps,
        )
        representative_trajectories.append(trajectory)
        path = (
            output
            / "trajectories"
            / "representative"
            / f"representative_arm_trajectory_{index:02d}_source_{int(window.start_s):03d}_{int(window.end_s):03d}s_025x_50hz.csv"
        )
        duration = _write_trajectory(path, joint_names, trajectory, args.fps)
        representative_trajectory_metadata.append(
            {
                "trajectory_id": f"representative_trajectory_{index:02d}",
                "path": path.relative_to(output).as_posix(),
                "source_start_s": window.start_s,
                "source_end_s": window.end_s,
                "source_duration_s": args.source_window_s,
                "playback_duration_s": duration,
                "time_stretch": args.playback_window_s / args.source_window_s,
                "equivalent_source_speed": args.source_window_s / args.playback_window_s,
                "max_joint_span_rad": window.max_joint_span_rad,
                "dominant_joint": window.dominant_joint,
                "motion_score": window.motion_score,
            }
        )
    representative_trajectory_sequence = _concat_sequences(
        representative_trajectories, args.fps, args.trajectory_bridge_s
    )
    representative_trajectory_path = (
        output / "sequences" / "representative_trajectories_arm_only_sequence_50hz.csv"
    )
    files["representative_trajectories"] = {
        "path": representative_trajectory_path.relative_to(output).as_posix(),
        "duration_s": _write_trajectory(
            representative_trajectory_path, joint_names, representative_trajectory_sequence, args.fps
        ),
        "items": representative_trajectory_metadata,
    }

    synthesized_trajectories, synthesized_trajectory_metadata = _synthesize_trajectories(
        representative_poses,
        args.synthesized_trajectory_count,
        rng,
        args.fps,
    )
    for index, trajectory in enumerate(synthesized_trajectories, start=1):
        path = (
            output
            / "trajectories"
            / "synthesized"
            / f"synthesized_arm_trajectory_{index:02d}_seed{args.seed}_minimum_jerk_50hz.csv"
        )
        duration = _write_trajectory(path, joint_names, trajectory, args.fps)
        synthesized_trajectory_metadata[index - 1]["path"] = path.relative_to(output).as_posix()
        synthesized_trajectory_metadata[index - 1]["duration_s"] = duration
    synthesized_trajectory_sequence = _concat_sequences(
        synthesized_trajectories, args.fps, args.trajectory_bridge_s
    )
    synthesized_trajectory_path = (
        output / "sequences" / f"synthesized_trajectories_arm_only_sequence_seed{args.seed}_50hz.csv"
    )
    files["synthesized_trajectories"] = {
        "path": synthesized_trajectory_path.relative_to(output).as_posix(),
        "duration_s": _write_trajectory(
            synthesized_trajectory_path, joint_names, synthesized_trajectory_sequence, args.fps
        ),
        "items": synthesized_trajectory_metadata,
    }

    all_sequence, all_timeline = _concat_labeled_sequences(
        [
            ("representative_poses", representative_pose_sequence),
            ("synthesized_poses", synthesized_pose_sequence),
            ("representative_trajectories", representative_trajectory_sequence),
            ("synthesized_trajectories", synthesized_trajectory_sequence),
        ],
        args.fps,
        args.section_bridge_s,
    )
    all_path = output / "sequences" / f"all_arm_only_evaluation_sequence_seed{args.seed}_50hz.csv"
    files["all"] = {
        "path": all_path.relative_to(output).as_posix(),
        "duration_s": _write_trajectory(all_path, joint_names, all_sequence, args.fps),
        "section_order": [
            "representative_poses",
            "synthesized_poses",
            "representative_trajectories",
            "synthesized_trajectories",
        ],
        "timeline": all_timeline,
    }

    static_pose_frames = int(round(args.static_pose_test_duration_s * args.fps)) + 1
    representative_pose_file_metadata: list[dict[str, object]] = []
    for index, pose in enumerate(representative_poses, start=1):
        path = (
            output
            / "poses"
            / "representative"
            / f"representative_arm_pose_{index:02d}_hold20s_50hz.csv"
        )
        duration = _write_trajectory(
            path, joint_names, np.repeat(pose[None, :], static_pose_frames, axis=0), args.fps
        )
        representative_pose_file_metadata.append(
            {
                "pose_id": f"representative_pose_{index:02d}",
                "path": path.relative_to(output).as_posix(),
                "duration_s": duration,
            }
        )
    for index, pose in enumerate(synthesized_poses, start=1):
        path = (
            output
            / "poses"
            / "synthesized"
            / f"synthesized_arm_pose_{index:02d}_seed{args.seed}_hold20s_50hz.csv"
        )
        duration = _write_trajectory(
            path, joint_names, np.repeat(pose[None, :], static_pose_frames, axis=0), args.fps
        )
        synthesized_pose_metadata[index - 1]["path"] = path.relative_to(output).as_posix()
        synthesized_pose_metadata[index - 1]["duration_s"] = duration

    try:
        source_path = str(args.source.relative_to(PROJECT_ROOT))
    except ValueError:
        source_path = str(args.source)

    manifest: dict[str, object] = {
        "schema_version": 2,
        "data_scope": "arm_only_14_dof",
        "contains_full_body_state": False,
        "source": {
            "path": source_path,
            "sha256": _sha256(args.source),
            "rows": len(times),
            "duration_s": float(times[-1]),
            "joint_names": joint_names,
        },
        "generation": {
            "seed": args.seed,
            "fps": args.fps,
            "controlled_joint_count": len(joint_names),
            "controlled_joint_names": joint_names,
            "representative_pose_method": "normalized farthest-point coverage from median pose",
            "representative_trajectory_method": "diverse non-overlapping 5 s high-motion windows",
            "synthesis_method": "pairwise convex interpolation of measured arm poses plus quintic minimum-jerk arm trajectories",
            "runtime_random_sampling": False,
        },
        "representative_poses": [
            {
                **representative_pose_file_metadata[index],
                "source_row": int(source_index),
                "source_time_s": float(times[source_index]),
                "data_scope": "arm_only_14_dof",
            }
            for index, source_index in enumerate(representative_indices)
        ],
        "synthesized_poses": synthesized_pose_metadata,
        "representative_trajectories": representative_trajectory_metadata,
        "synthesized_trajectories": synthesized_trajectory_metadata,
        "files": files,
    }
    generated_paths = [path for path in output.rglob("*.csv")]
    manifest["generated_file_sha256"] = {
        path.relative_to(output).as_posix(): _sha256(path) for path in sorted(generated_paths)
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--representative-pose-count", type=int, default=6)
    parser.add_argument("--synthesized-pose-count", type=int, default=3)
    parser.add_argument("--representative-trajectory-count", type=int, default=4)
    parser.add_argument("--synthesized-trajectory-count", type=int, default=3)
    parser.add_argument("--pose-stride", type=int, default=10)
    parser.add_argument("--pose-hold-s", type=float, default=4.0)
    parser.add_argument("--pose-transition-s", type=float, default=2.0)
    parser.add_argument("--static-pose-test-duration-s", type=float, default=20.0)
    parser.add_argument("--source-window-s", type=float, default=5.0)
    parser.add_argument("--playback-window-s", type=float, default=20.0)
    parser.add_argument("--window-stride-s", type=float, default=1.0)
    parser.add_argument("--minimum-window-separation-s", type=float, default=10.0)
    parser.add_argument("--trajectory-bridge-s", type=float, default=2.0)
    parser.add_argument("--section-bridge-s", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    manifest = build_suite(args)
    print(f"Built deterministic ArmHack Stand arm-only test data: {args.output}")
    print(f"Source SHA-256: {manifest['source']['sha256']}")
    print("Representative poses:")
    for pose in manifest["representative_poses"]:
        print(f"  {pose['pose_id']}: source_time_s={pose['source_time_s']:.6f}")
    print("Representative trajectories:")
    for trajectory in manifest["representative_trajectories"]:
        print(
            f"  {trajectory['trajectory_id']}: "
            f"source={trajectory['source_start_s']:.1f}..{trajectory['source_end_s']:.1f}s, "
            f"dominant={trajectory['dominant_joint']}, span={trajectory['max_joint_span_rad']:.4f}rad"
        )
    print(f"All-sequence duration: {manifest['files']['all']['duration_s']:.3f}s")


if __name__ == "__main__":
    main()
