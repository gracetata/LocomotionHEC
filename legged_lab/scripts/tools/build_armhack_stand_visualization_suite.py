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
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "Reference Data" / "ArmHack" / "StandPerturb" / "g1_arm_trajectory_named_50hz.csv"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent / "TestData" / "ArmOnly"
DEFAULT_RANDOM_POSE_BANK = (
    DEFAULT_SOURCE.parent / "RandomizedTraining" / "random_arm_pose_bank_seed20260715.json"
)
DEFAULT_TRAINING_EPISODE_HORIZON_S = 20.0
DEFAULT_TRAINING_CSV_END_MARGIN_S = 0.25
REMOVED_ARMS_DOWN_SOURCE_TIME_S = 404.897585


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


def _load_random_pose_bank(
    path: Path,
    expected_joint_names: list[str],
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("data_scope") != "arm_only_14_dof":
        raise ValueError(f"Unsupported randomized ArmHack pose-bank schema: {path}")
    bank_joint_names = payload.get("joint_names")
    if not isinstance(bank_joint_names, list) or set(bank_joint_names) != set(expected_joint_names):
        raise ValueError(f"Random pose-bank joints differ from the Stand CSV: {path}")
    entries = payload.get("poses")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Random pose bank is empty: {path}")
    bank_poses = np.asarray([entry["positions_rad"] for entry in entries], dtype=np.float64)
    bank_velocity_limits = np.asarray(payload.get("interpolation_velocity_limits_rad_s"), dtype=np.float64)
    reorder = [bank_joint_names.index(name) for name in expected_joint_names]
    poses = bank_poses[:, reorder]
    velocity_limits = bank_velocity_limits[reorder]
    if poses.ndim != 2 or poses.shape[1] != len(expected_joint_names):
        raise ValueError(f"Random pose bank must have shape (N, {len(expected_joint_names)}): {path}")
    if velocity_limits.shape != (len(expected_joint_names),) or np.any(velocity_limits <= 0.0):
        raise ValueError(f"Random pose-bank velocity limits are invalid: {path}")
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(velocity_limits)):
        raise ValueError(f"Random pose bank contains non-finite values: {path}")
    return poses, entries, velocity_limits, payload


def _write_trajectory(path: Path, joint_names: list[str], positions: np.ndarray, fps: float) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["time_s", *joint_names])
        for index, pose in enumerate(positions):
            writer.writerow([f"{index / fps:.8f}", *(f"{value:.8f}" for value in pose)])
    return (len(positions) - 1) / fps if len(positions) > 1 else 0.0


def _min_jerk(start: np.ndarray, end: np.ndarray, duration_s: float, fps: float) -> np.ndarray:
    count = max(int(round(duration_s * fps)), 1)
    phase = np.arange(1, count + 1, dtype=np.float64) / count
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return start[None, :] + blend[:, None] * (end - start)[None, :]


def _coverage_indices(positions: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or count > len(positions):
        raise ValueError(f"Cannot select {count} coverage poses from {len(positions)} candidates.")
    center = np.median(positions, axis=0)
    scale = np.maximum(np.percentile(positions, 95.0, axis=0) - np.percentile(positions, 5.0, axis=0), 0.05)
    normalized = np.clip((positions - center) / scale, -4.0, 4.0)
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    min_distance = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    while len(selected) < count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(min_distance, np.linalg.norm(normalized - normalized[next_index], axis=1))
    return np.asarray(selected, dtype=np.int64)


def _velocity_safe_transition_duration_s(
    start: np.ndarray,
    end: np.ndarray,
    velocity_limits: np.ndarray,
    nominal_duration_s: float,
) -> float:
    # Quintic minimum-jerk interpolation has max d(blend)/du = 1.875.
    required_duration_s = float(np.max(1.875 * np.abs(end - start) / velocity_limits))
    return max(float(nominal_duration_s), required_duration_s)


def _velocity_safe_pose_sequence(
    poses: np.ndarray,
    labels: list[str],
    velocity_limits: np.ndarray,
    fps: float,
    hold_s: float,
    nominal_transition_s: float,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    hold_count = max(int(round(hold_s * fps)), 1)
    frames: list[np.ndarray] = []
    timeline: list[dict[str, float | str]] = []
    cursor = 0
    for index, (pose, label) in enumerate(zip(poses, labels, strict=True)):
        if index > 0:
            transition_duration_s = _velocity_safe_transition_duration_s(
                poses[index - 1], pose, velocity_limits, nominal_transition_s
            )
            transition = _min_jerk(poses[index - 1], pose, transition_duration_s, fps)
            transition_start = cursor / fps
            frames.extend(transition)
            cursor += len(transition)
            timeline.append(
                {
                    "kind": "velocity_safe_transition",
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


def _random_pose_interpolation_trajectories(
    poses: np.ndarray,
    pose_labels: list[str],
    velocity_limits: np.ndarray,
    count: int,
    nominal_duration_s: float,
    fps: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    if len(poses) < 2:
        raise ValueError("At least two randomized poses are required for interpolation trajectories.")
    trajectories: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    used_pairs: set[tuple[int, int]] = set()
    while len(trajectories) < count:
        start_index, end_index = (int(value) for value in rng.choice(len(poses), size=2, replace=False))
        pair = (start_index, end_index)
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        duration_s = _velocity_safe_transition_duration_s(
            poses[start_index], poses[end_index], velocity_limits, nominal_duration_s
        )
        trajectory = np.concatenate(
            (
                poses[start_index][None, :],
                _min_jerk(poses[start_index], poses[end_index], duration_s, fps),
            ),
            axis=0,
        )
        trajectory_id = f"randomized_trajectory_{len(trajectories) + 1:02d}"
        trajectories.append(trajectory)
        metadata.append(
            {
                "trajectory_id": trajectory_id,
                "start_pose_id": pose_labels[start_index],
                "end_pose_id": pose_labels[end_index],
                "duration_s": (len(trajectory) - 1) / fps,
                "nominal_duration_s": nominal_duration_s,
                "interpolation": "quintic minimum jerk",
                "velocity_limited": True,
                "data_scope": "arm_only_14_dof",
            }
        )
    return trajectories, metadata


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


def _expand_group_timelines(
    section_timeline: list[dict[str, float | str]],
    child_timelines: dict[str, list[dict[str, float | str]]],
) -> list[dict[str, float | str]]:
    """Expand top-level sections into plot-ready pose/trajectory stages."""
    detailed: list[dict[str, float | str]] = []
    for section in section_timeline:
        section_kind = str(section["kind"])
        section_label = str(section["label"])
        section_start = float(section["start_s"])
        section_end = float(section["end_s"])
        if section_kind != "section":
            detailed.append(dict(section))
            continue

        child_timeline = child_timelines.get(section_label, [])
        if not child_timeline:
            detailed.append(dict(section))
            continue
        for child in child_timeline:
            child_start = section_start + float(child["start_s"])
            child_end = min(section_start + float(child["end_s"]), section_end)
            if child_end <= child_start:
                continue
            detailed.append(
                {
                    "kind": str(child["kind"]),
                    "label": str(child["label"]),
                    "group": section_label,
                    "start_s": child_start,
                    "end_s": child_end,
                }
            )
    return detailed


def _select_representative_poses(
    times: np.ndarray,
    positions: np.ndarray,
    count: int,
    stride: int,
    maximum_source_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eligible_indices = np.flatnonzero(times <= maximum_source_time_s + 1.0e-9)
    if len(eligible_indices) < count:
        raise ValueError(
            "Training-reachable static-pose interval contains too few samples: "
            f"{len(eligible_indices)} available, {count} requested."
        )
    eligible_positions = positions[eligible_indices]
    lower = np.percentile(eligible_positions, 1.0, axis=0)
    upper = np.percentile(eligible_positions, 99.0, axis=0)
    center = np.median(eligible_positions, axis=0)
    scale = np.maximum(
        np.percentile(eligible_positions, 95.0, axis=0)
        - np.percentile(eligible_positions, 5.0, axis=0),
        0.05,
    )

    candidate_indices = eligible_indices[:: max(stride, 1)]
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
    maximum_source_time_s: float,
) -> list[MotionWindow]:
    candidates: list[MotionWindow] = []
    candidate_end_s = min(float(times[-1]), maximum_source_time_s)
    for start_s in np.arange(0.0, candidate_end_s - window_s + 1.0e-9, stride_s):
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
    representative_trajectories: list[np.ndarray],
    representative_windows: list[MotionWindow],
    count: int,
    rng: np.random.Generator,
    speed_scale: float,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Blend pairs of measured 1x arm trajectories without creating full-body data."""
    if len(representative_trajectories) < 2:
        raise ValueError("At least two representative trajectories are required for synthesis.")
    if any(trajectory.shape != representative_trajectories[0].shape for trajectory in representative_trajectories):
        raise ValueError("Representative trajectories must share the same shape before synthesis.")

    trajectories: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for trajectory_index in range(count):
        parent_indices = rng.choice(len(representative_trajectories), size=2, replace=False)
        alpha = float(rng.uniform(0.25, 0.75))
        first_index, second_index = (int(value) for value in parent_indices)
        trajectory = (
            (1.0 - alpha) * representative_trajectories[first_index]
            + alpha * representative_trajectories[second_index]
        )
        trajectories.append(trajectory)
        metadata.append(
            {
                "trajectory_id": f"synth_trajectory_{trajectory_index + 1:02d}",
                "parent_representative_trajectory_indices": [first_index + 1, second_index + 1],
                "parent_source_windows_s": [
                    [representative_windows[first_index].start_s, representative_windows[first_index].end_s],
                    [representative_windows[second_index].start_s, representative_windows[second_index].end_s],
                ],
                "interpolation_alpha": alpha,
                "parent_weights": [1.0 - alpha, alpha],
                "equivalent_source_speed": speed_scale,
                "profile": "frame-aligned convex blend of two measured arm-only trajectories",
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
    randomized_poses: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pose_id", "kind", "source_time_s", *joint_names])
        for index, (source_index, pose) in enumerate(
            zip(representative_indices, representative_poses, strict=True), start=1
        ):
            writer.writerow(
                [f"representative_pose_{index:02d}", "source", f"{times[source_index]:.8f}", *(f"{v:.8f}" for v in pose)]
            )
        for index, pose in enumerate(synthesized_poses, start=1):
            writer.writerow([f"synth_pose_{index:02d}", "synthesized", "", *(f"{v:.8f}" for v in pose)])
        for index, pose in enumerate(randomized_poses, start=1):
            writer.writerow([f"randomized_pose_{index:02d}", "randomized_bank", "", *(f"{v:.8f}" for v in pose)])


def build_suite(args: argparse.Namespace) -> dict[str, object]:
    joint_names, times, positions = _load_source(args.source)
    random_bank_poses, random_bank_entries, random_velocity_limits, random_bank_payload = (
        _load_random_pose_bank(args.random_pose_bank, joint_names)
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not np.isclose(args.trajectory_speed_scale, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Stand evaluation trajectories are fixed to the original 1.0x source speed.")
    for generated_directory in ("poses", "trajectories", "sequences"):
        shutil.rmtree(output / generated_directory, ignore_errors=True)
    (output / "manifest.json").unlink(missing_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.training_episode_horizon_s <= 0.0:
        raise ValueError("training_episode_horizon_s must be positive.")
    if args.training_csv_end_margin_s < 0.0:
        raise ValueError("training_csv_end_margin_s must be non-negative.")
    maximum_motion_source_time_s = float(times[-1]) - args.training_csv_end_margin_s
    maximum_static_pose_source_time_s = (
        maximum_motion_source_time_s
        - args.training_episode_horizon_s * args.trajectory_speed_scale
    )
    if maximum_static_pose_source_time_s <= 0.0:
        raise ValueError(
            "Training sampling contract leaves no valid static-pose start interval: "
            f"source_duration={times[-1]:.6f}s, "
            f"episode_horizon={args.training_episode_horizon_s:.6f}s, "
            f"end_margin={args.training_csv_end_margin_s:.6f}s."
        )

    representative_indices, representative_poses, _lower, _upper = _select_representative_poses(
        times,
        positions,
        args.representative_pose_count,
        args.pose_stride,
        maximum_static_pose_source_time_s,
    )
    joint_scale = np.maximum(np.percentile(positions, 95.0, axis=0) - np.percentile(positions, 5.0, axis=0), 0.05)
    synthesized_poses, synthesized_pose_metadata = _synthesize_poses(
        representative_poses, args.synthesized_pose_count, rng
    )
    generated_bank_indices = np.asarray(
        [
            index
            for index, entry in enumerate(random_bank_entries)
            if entry.get("kind") == "convex_synthesized"
        ],
        dtype=np.int64,
    )
    if len(generated_bank_indices) < args.randomized_pose_count:
        raise ValueError(
            f"Random pose bank has only {len(generated_bank_indices)} synthesized poses; "
            f"{args.randomized_pose_count} requested."
        )
    selected_generated_local = _coverage_indices(
        random_bank_poses[generated_bank_indices], args.randomized_pose_count
    )
    randomized_bank_indices = generated_bank_indices[selected_generated_local]
    randomized_poses = random_bank_poses[randomized_bank_indices]
    randomized_pose_metadata: list[dict[str, object]] = [
        {
            "pose_id": f"randomized_pose_{index + 1:02d}",
            "pose_bank_index": int(bank_index),
            "pose_bank_pose_id": str(random_bank_entries[bank_index]["pose_id"]),
            "data_scope": "arm_only_14_dof",
        }
        for index, bank_index in enumerate(randomized_bank_indices)
    ]

    pose_catalog_path = output / "poses" / "arm_pose_catalog.csv"
    pose_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pose_catalog(
        pose_catalog_path,
        joint_names,
        representative_indices,
        times,
        representative_poses,
        synthesized_poses,
        randomized_poses,
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
    randomized_pose_labels = [metadata["pose_id"] for metadata in randomized_pose_metadata]
    randomized_pose_sequence, randomized_pose_timeline = _velocity_safe_pose_sequence(
        randomized_poses,
        randomized_pose_labels,
        random_velocity_limits,
        args.fps,
        args.pose_hold_s,
        args.pose_transition_s,
    )

    files: dict[str, dict[str, object]] = {}
    for name, sequence, timeline in (
        ("representative_poses", representative_pose_sequence, representative_pose_timeline),
        ("synthesized_poses", synthesized_pose_sequence, synthesized_pose_timeline),
        ("randomized_poses", randomized_pose_sequence, randomized_pose_timeline),
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
        maximum_motion_source_time_s,
    )
    selected_windows = _select_motion_windows(
        candidates, args.representative_trajectory_count, args.minimum_window_separation_s
    )
    representative_trajectories: list[np.ndarray] = []
    representative_trajectory_metadata: list[dict[str, object]] = []
    playback_duration_s = args.source_window_s / args.trajectory_speed_scale
    for index, window in enumerate(selected_windows, start=1):
        trajectory = _resample_source_window(
            times,
            positions,
            window.start_s,
            window.end_s,
            playback_duration_s,
            args.fps,
        )
        representative_trajectories.append(trajectory)
        path = (
            output
            / "trajectories"
            / "representative"
            / f"representative_arm_trajectory_{index:02d}_source_{int(window.start_s):03d}_{int(window.end_s):03d}s_1x_50hz.csv"
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
                "time_stretch": playback_duration_s / args.source_window_s,
                "equivalent_source_speed": args.trajectory_speed_scale,
                "max_joint_span_rad": window.max_joint_span_rad,
                "dominant_joint": window.dominant_joint,
                "motion_score": window.motion_score,
            }
        )
    representative_trajectory_sequence, representative_trajectory_timeline = _concat_labeled_sequences(
        [
            (metadata["trajectory_id"], trajectory)
            for metadata, trajectory in zip(
                representative_trajectory_metadata, representative_trajectories, strict=True
            )
        ],
        args.fps,
        args.trajectory_bridge_s,
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
        "timeline": representative_trajectory_timeline,
    }

    synthesized_trajectories, synthesized_trajectory_metadata = _synthesize_trajectories(
        representative_trajectories,
        selected_windows,
        args.synthesized_trajectory_count,
        rng,
        args.trajectory_speed_scale,
    )
    for index, trajectory in enumerate(synthesized_trajectories, start=1):
        path = (
            output
            / "trajectories"
            / "synthesized"
            / f"synthesized_arm_trajectory_{index:02d}_seed{args.seed}_measured_blend_1x_50hz.csv"
        )
        duration = _write_trajectory(path, joint_names, trajectory, args.fps)
        synthesized_trajectory_metadata[index - 1]["path"] = path.relative_to(output).as_posix()
        synthesized_trajectory_metadata[index - 1]["duration_s"] = duration
    synthesized_trajectory_sequence, synthesized_trajectory_timeline = _concat_labeled_sequences(
        [
            (metadata["trajectory_id"], trajectory)
            for metadata, trajectory in zip(
                synthesized_trajectory_metadata, synthesized_trajectories, strict=True
            )
        ],
        args.fps,
        args.trajectory_bridge_s,
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
        "timeline": synthesized_trajectory_timeline,
    }

    randomized_trajectories, randomized_trajectory_metadata = _random_pose_interpolation_trajectories(
        randomized_poses,
        [str(label) for label in randomized_pose_labels],
        random_velocity_limits,
        args.randomized_trajectory_count,
        args.randomized_trajectory_nominal_duration_s,
        args.fps,
        rng,
    )
    for index, trajectory in enumerate(randomized_trajectories, start=1):
        path = (
            output
            / "trajectories"
            / "randomized"
            / f"randomized_arm_trajectory_{index:02d}_seed20260715_minjerk_50hz.csv"
        )
        duration = _write_trajectory(path, joint_names, trajectory, args.fps)
        randomized_trajectory_metadata[index - 1]["path"] = path.relative_to(output).as_posix()
        randomized_trajectory_metadata[index - 1]["duration_s"] = duration
    randomized_trajectory_sequence, randomized_trajectory_timeline = _concat_labeled_sequences(
        [
            (str(metadata["trajectory_id"]), trajectory)
            for metadata, trajectory in zip(
                randomized_trajectory_metadata, randomized_trajectories, strict=True
            )
        ],
        args.fps,
        args.trajectory_bridge_s,
    )
    randomized_trajectory_path = (
        output / "sequences" / "randomized_trajectories_arm_only_sequence_seed20260715_50hz.csv"
    )
    files["randomized_trajectories"] = {
        "path": randomized_trajectory_path.relative_to(output).as_posix(),
        "duration_s": _write_trajectory(
            randomized_trajectory_path, joint_names, randomized_trajectory_sequence, args.fps
        ),
        "items": randomized_trajectory_metadata,
        "timeline": randomized_trajectory_timeline,
    }

    all_sequence, all_timeline = _concat_labeled_sequences(
        [
            ("representative_poses", representative_pose_sequence),
            ("synthesized_poses", synthesized_pose_sequence),
            ("randomized_poses", randomized_pose_sequence),
            ("representative_trajectories", representative_trajectory_sequence),
            ("synthesized_trajectories", synthesized_trajectory_sequence),
            ("randomized_trajectories", randomized_trajectory_sequence),
        ],
        args.fps,
        args.section_bridge_s,
    )
    all_detailed_timeline = _expand_group_timelines(
        all_timeline,
        {
            "representative_poses": representative_pose_timeline,
            "synthesized_poses": synthesized_pose_timeline,
            "randomized_poses": randomized_pose_timeline,
            "representative_trajectories": representative_trajectory_timeline,
            "synthesized_trajectories": synthesized_trajectory_timeline,
            "randomized_trajectories": randomized_trajectory_timeline,
        },
    )
    all_path = output / "sequences" / f"all_arm_only_evaluation_sequence_seed{args.seed}_50hz.csv"
    files["all"] = {
        "path": all_path.relative_to(output).as_posix(),
        "duration_s": _write_trajectory(all_path, joint_names, all_sequence, args.fps),
        "section_order": [
            "representative_poses",
            "synthesized_poses",
            "randomized_poses",
            "representative_trajectories",
            "synthesized_trajectories",
            "randomized_trajectories",
        ],
        "timeline": all_timeline,
        "detailed_timeline": all_detailed_timeline,
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
    for index, pose in enumerate(randomized_poses, start=1):
        path = (
            output
            / "poses"
            / "randomized"
            / f"randomized_arm_pose_{index:02d}_seed20260715_hold20s_50hz.csv"
        )
        duration = _write_trajectory(
            path, joint_names, np.repeat(pose[None, :], static_pose_frames, axis=0), args.fps
        )
        randomized_pose_metadata[index - 1]["path"] = path.relative_to(output).as_posix()
        randomized_pose_metadata[index - 1]["duration_s"] = duration

    try:
        source_path = str(args.source.relative_to(PROJECT_ROOT))
    except ValueError:
        source_path = str(args.source)

    manifest: dict[str, object] = {
        "schema_version": 5,
        "data_scope": "arm_only_14_dof",
        "contains_full_body_state": False,
        "source": {
            "path": source_path,
            "sha256": _sha256(args.source),
            "rows": len(times),
            "duration_s": float(times[-1]),
            "joint_names": joint_names,
        },
        "random_pose_bank": {
            "path": args.random_pose_bank.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(args.random_pose_bank),
            "schema_version": random_bank_payload["schema_version"],
            "bank_size": random_bank_payload["generation"]["bank_size"],
            "seed": random_bank_payload["generation"]["seed"],
        },
        "generation": {
            "seed": args.seed,
            "fps": args.fps,
            "controlled_joint_count": len(joint_names),
            "controlled_joint_names": joint_names,
            "trajectory_speed_scale": args.trajectory_speed_scale,
            "training_sampling_contract": {
                "csv_loop": False,
                "episode_horizon_s": args.training_episode_horizon_s,
                "csv_end_margin_s": args.training_csv_end_margin_s,
                "static_pose_source_time_range_s": [0.0, maximum_static_pose_source_time_s],
                "moving_trajectory_source_time_range_s": [0.0, maximum_motion_source_time_s],
            },
            "representative_pose_method": (
                "normalized farthest-point coverage from median pose within the "
                "training-reachable static-start interval"
            ),
            "representative_trajectory_method": (
                "diverse non-overlapping 5 s high-motion windows within the "
                "training-reachable moving interval"
            ),
            "synthesis_method": "pairwise convex interpolation of measured arm poses and frame-aligned measured 1x arm trajectories",
            "randomized_pose_method": (
                "farthest-point coverage of the new convex-synthesized entries in the Stand random pose bank"
            ),
            "randomized_trajectory_method": (
                "velocity-limited quintic minimum-jerk interpolation between deterministic randomized poses"
            ),
            "runtime_random_sampling": False,
        },
        "excluded_test_postures": [
            {
                "label": "former_representative_pose_02_arms_down",
                "source_time_s": REMOVED_ARMS_DOWN_SOURCE_TIME_S,
                "reason": (
                    "explicitly removed from the Stand pose test set; this source time is also "
                    "outside the training-reachable static-start interval"
                ),
            }
        ],
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
        "randomized_poses": randomized_pose_metadata,
        "representative_trajectories": representative_trajectory_metadata,
        "synthesized_trajectories": synthesized_trajectory_metadata,
        "randomized_trajectories": randomized_trajectory_metadata,
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
    parser.add_argument("--random-pose-bank", type=Path, default=DEFAULT_RANDOM_POSE_BANK)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--representative-pose-count", type=int, default=6)
    parser.add_argument("--synthesized-pose-count", type=int, default=3)
    parser.add_argument("--representative-trajectory-count", type=int, default=4)
    parser.add_argument("--synthesized-trajectory-count", type=int, default=3)
    parser.add_argument("--randomized-pose-count", type=int, default=8)
    parser.add_argument("--randomized-trajectory-count", type=int, default=6)
    parser.add_argument("--randomized-trajectory-nominal-duration-s", type=float, default=5.0)
    parser.add_argument("--pose-stride", type=int, default=10)
    parser.add_argument("--pose-hold-s", type=float, default=4.0)
    parser.add_argument("--pose-transition-s", type=float, default=2.0)
    parser.add_argument("--static-pose-test-duration-s", type=float, default=20.0)
    parser.add_argument("--source-window-s", type=float, default=5.0)
    parser.add_argument("--trajectory-speed-scale", type=float, default=1.0)
    parser.add_argument("--window-stride-s", type=float, default=1.0)
    parser.add_argument("--minimum-window-separation-s", type=float, default=10.0)
    parser.add_argument("--trajectory-bridge-s", type=float, default=2.0)
    parser.add_argument("--section-bridge-s", type=float, default=3.0)
    parser.add_argument(
        "--training-episode-horizon-s",
        type=float,
        default=DEFAULT_TRAINING_EPISODE_HORIZON_S,
        help="Stand training episode horizon used to bound static-pose reset starts.",
    )
    parser.add_argument(
        "--training-csv-end-margin-s",
        type=float,
        default=DEFAULT_TRAINING_CSV_END_MARGIN_S,
        help="Stand training CSV end margin excluded from all evaluation candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.random_pose_bank = args.random_pose_bank.expanduser().resolve()
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
    print(
        f"Randomized evaluation: {len(manifest['randomized_poses'])} poses + "
        f"{len(manifest['randomized_trajectories'])} pose-interpolation trajectories"
    )
    print(f"All-sequence duration: {manifest['files']['all']['duration_s']:.3f}s")


if __name__ == "__main__":
    main()
