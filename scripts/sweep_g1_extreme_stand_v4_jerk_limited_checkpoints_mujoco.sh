#!/usr/bin/env bash
# Screen V4-derived checkpoints with the deployment target limiter enabled.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RUN_DIR=${RUN_DIR:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_16-37-50_g1_extreme_stand_v4_jerk_limited_from_model2999_full_20260807"} \
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_jerk_limited_sweep_20260807"} \
CHECKPOINT_IDS=${CHECKPOINT_IDS:-0,20,25,50,100,150,199} \
CANDIDATE_PREFIX=v4jl \
CANDIDATE_DISPLAY_NAME="V4 state-gated jerk limiter" \
CANDIDATE_TARGET_LIMITER_ENABLE=True \
PROFILES=${PROFILES:-nominal,pose_recovery,feet_distance_recovery} \
SEEDS=${SEEDS:-20260806} \
DURATION=${DURATION:-40.0} \
STEADY_START_S=${STEADY_START_S:-10.0} \
CPU_AFFINITY=${CPU_AFFINITY:-0-7,10-31} \
SKIP_EXISTING=${SKIP_EXISTING:-True} \
    bash "${ROOT_DIR}/scripts/sweep_g1_extreme_stand_support_lock_v8_checkpoints_mujoco.sh"
