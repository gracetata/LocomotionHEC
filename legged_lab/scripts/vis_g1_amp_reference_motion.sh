#!/usr/bin/env bash
# Visualize G1 AMP reference motion pickles directly in Isaac Sim.
# Usage:
#   bash scripts/vis_g1_amp_reference_motion.sh
#   VIEW_MODE=training_current NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
#
# Environment variables:
#   ISAACLAB_PYTHON     : Python executable from .envrc conda env.
#   CONDA_ENV_NAME      : Conda env name used when ISAACLAB_PYTHON is unset (default: env_isaaclab).
#   CONDA_BASE          : Conda installation root (default: /home/hecggdz/miniconda3).
#   ROBOT_ASSET         : Robot asset preset: g1_29dof, s3_g1_29dof, s3_g1_29dof_mjcf (default: g1_29dof).
#   MOTION_DIR          : Directory containing AMP reference .pkl files.
#   MOTION_NAME         : Optional specific motion stem or .pkl file name.
#   MOTION_SELECTION    : sorted, spread, random, yaw, path_turn, misaligned, cycle_all,
#                         balanced_modes, mode_forward_slow, mode_forward_normal, mode_backward,
#                         mode_lateral_left, mode_lateral_right, mode_turn_left, mode_turn_right, mode_stand.
#   VIEW_MODE           : training_current, name_aligned, name_aligned_xyzw (default: name_aligned_xyzw).
#   NUM_ENVS            : Number of clips to visualize at once (default: 4).
#   START_INDEX         : First sorted motion index when MOTION_NAME is empty (default: 0).
#   DEVICE              : Isaac Sim device (default: cuda:0).
#   HEADLESS            : True/False for Isaac Sim UI (default: False).
#   MAX_STEPS           : Stop after this many frames; empty means run until window closes.
#   REAL_TIME           : True/False real-time pacing (default: True).
#   LOOP                : True/False loop clips instead of holding final frame (default: True).
#   SPEED               : Playback speed multiplier (default: 1.0).
#   HEIGHT_OFFSET       : Extra root z offset for visual inspection (default: 0.15).
#   HISTORY_STEPS       : Current/future AMP ghost frames to show (default: 4).
#   TRAIL_LENGTH        : Root/foot trail length in frames (default: 160).
#   PRINT_INTERVAL      : Frames between status logs (default: 25).
#   ZERO_WORLD_GRAVITY  : True/False set world gravity to zero too (default: True).
#   EXTRA_ARGS          : Additional raw CLI arguments for the Python viewer.
# ============================================================
: <<'BLOCK'
# [Reference motion viewer shortcuts]

# 1. Primary correctness check: joint-name aligned + xyzw->wxyz root quaternion conversion.
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_washed_50hz VIEW_MODE=name_aligned_xyzw HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=cycle_all HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh

# 1b. Diagnose root velocity/yaw alignment on the full dataset.
ROBOT_ASSET=g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=path_turn HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=yaw HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=misaligned HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh

# 2. SOLVED_BUG: Compare with what the current CMU adaptive training task likely consumed.
ROBOT_ASSET=g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz_task_core VIEW_MODE=training_current HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh

# 3. Command-balanced directional dataset: inspect one clip from each locomotion mode.
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=balanced_modes HEIGHT_OFFSET=0.15 NUM_ENVS=8 bash scripts/vis_g1_amp_reference_motion.sh

# 3b. Command-balanced directional dataset: inspect specific modes.
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_forward_slow HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_forward_normal HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_backward HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_lateral_left HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_lateral_right HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_turn_left HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_turn_right HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh
ROBOT_ASSET=s3_g1_29dof MOTION_DIR=source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz VIEW_MODE=name_aligned_xyzw MOTION_SELECTION=mode_stand HEIGHT_OFFSET=0.15 NUM_ENVS=4 bash scripts/vis_g1_amp_reference_motion.sh

# 4. Headless smoke test.
HEADLESS=True MAX_STEPS=20 VIEW_MODE=name_aligned_xyzw NUM_ENVS=2 bash scripts/vis_g1_amp_reference_motion.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_leglab}
CONDA_BASE=${CONDA_BASE:-/home/hecggdz/miniconda3}
if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
        ISAACLAB_PYTHON="${CONDA_PREFIX}/bin/python"
    else
        ISAACLAB_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
    fi
fi

ROBOT_ASSET=${ROBOT_ASSET:-g1_29dof}
MOTION_DIR=${MOTION_DIR:-source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz_task_core}
MOTION_NAME=${MOTION_NAME:-}
MOTION_SELECTION=${MOTION_SELECTION:-cycle_all}
VIEW_MODE=${VIEW_MODE:-name_aligned_xyzw}
NUM_ENVS=${NUM_ENVS:-4}
START_INDEX=${START_INDEX:-0}
DEVICE=${DEVICE:-cuda:0}
HEADLESS=${HEADLESS:-False}
MAX_STEPS=${MAX_STEPS:-}
REAL_TIME=${REAL_TIME:-True}
LOOP=${LOOP:-True}
SPEED=${SPEED:-1.0}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.15}
HISTORY_STEPS=${HISTORY_STEPS:-4}
TRAIL_LENGTH=${TRAIL_LENGTH:-160}
PRINT_INTERVAL=${PRINT_INTERVAL:-25}
ZERO_WORLD_GRAVITY=${ZERO_WORLD_GRAVITY:-True}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi

if [[ "${MOTION_DIR}" != /* ]]; then
    MOTION_DIR="${ROOT_DIR}/${MOTION_DIR}"
fi
if [[ ! -d "${MOTION_DIR}" ]]; then
    echo "Error: MOTION_DIR does not exist: ${MOTION_DIR}" >&2
    exit 1
fi

ISAACLAB_CONDA_PREFIX=$(cd "$(dirname "${ISAACLAB_PYTHON}")/.." && pwd)
export LD_LIBRARY_PATH="${ISAACLAB_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${LEGGED_LAB_DIR}/source/legged_lab${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_LOG_DEFAULT_LEVEL=${OMNI_LOG_DEFAULT_LEVEL:-error}
export OMNI_KIT_QUIET=${OMNI_KIT_QUIET:-1}

args=(
    scripts/tools/vis_g1_amp_reference_motion.py
    --num_envs "${NUM_ENVS}"
    --motion_dir "${MOTION_DIR}"
    --start_index "${START_INDEX}"
    --motion_selection "${MOTION_SELECTION}"
    --view_mode "${VIEW_MODE}"
    --robot_asset "${ROBOT_ASSET}"
    --device "${DEVICE}"
    --height_offset "${HEIGHT_OFFSET}"
    --speed "${SPEED}"
    --history_steps "${HISTORY_STEPS}"
    --trail_length "${TRAIL_LENGTH}"
    --print_interval "${PRINT_INTERVAL}"
)

if [[ -n "${MOTION_NAME}" ]]; then
    args+=(--motion_name "${MOTION_NAME}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
    args+=(--max_steps "${MAX_STEPS}")
fi
if [[ "${HEADLESS}" == "True" || "${HEADLESS}" == "true" || "${HEADLESS}" == "1" ]]; then
    export HEADLESS=1
    args+=(--headless)
else
    export HEADLESS=0
fi
if [[ "${REAL_TIME}" == "True" || "${REAL_TIME}" == "true" || "${REAL_TIME}" == "1" ]]; then
    args+=(--real_time)
fi
if [[ "${LOOP}" == "False" || "${LOOP}" == "false" || "${LOOP}" == "0" ]]; then
    args+=(--no_loop)
else
    args+=(--loop)
fi
if [[ "${ZERO_WORLD_GRAVITY}" == "True" || "${ZERO_WORLD_GRAVITY}" == "true" || "${ZERO_WORLD_GRAVITY}" == "1" ]]; then
    args+=(--zero_world_gravity)
fi

echo "==============================================="
echo "  Visualizing G1 AMP Reference Motion Dataset"
echo "==============================================="
echo "Robot Asset       : ${ROBOT_ASSET}"
echo "Motion Dir        : ${MOTION_DIR}"
echo "Motion Name       : ${MOTION_NAME:-<selected batch>}"
echo "Motion Selection  : ${MOTION_SELECTION}"
echo "View Mode         : ${VIEW_MODE}"
echo "Num Envs          : ${NUM_ENVS}"
echo "Start Index       : ${START_INDEX}"
echo "Device            : ${DEVICE}"
echo "Headless          : ${HEADLESS}"
echo "Max Steps         : ${MAX_STEPS:-<window>}"
echo "Real Time / Loop  : ${REAL_TIME} / ${LOOP}"
echo "Speed             : ${SPEED}"
echo "Height Offset     : ${HEIGHT_OFFSET}"
echo "History / Trail   : ${HISTORY_STEPS} / ${TRAIL_LENGTH}"
echo "Zero World Gravity: ${ZERO_WORLD_GRAVITY}"
echo "Extra Args        : ${EXTRA_ARGS}"
echo "==============================================="

cd "${LEGGED_LAB_DIR}"
"${ISAACLAB_PYTHON}" "${args[@]}" ${EXTRA_ARGS}
