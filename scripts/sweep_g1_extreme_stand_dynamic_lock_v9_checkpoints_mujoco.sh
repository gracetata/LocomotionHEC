#!/usr/bin/env bash
# Strict three-profile screen for the complete V9 checkpoints produced before
# the frozen-V4 KL safety stop.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RUN_DIR=${RUN_DIR:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_16-11-40_g1_extreme_stand_recovery_dynamic_lock_v9_from_v4_model2999_full_20260807"} \
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_v9_dynamic_lock_sweep_20260807"} \
CHECKPOINT_IDS=${CHECKPOINT_IDS:-0,10} \
CANDIDATE_PREFIX=v9 \
CANDIDATE_DISPLAY_NAME="Dynamic-Lock V9" \
PROFILES=${PROFILES:-nominal,pose_recovery,feet_distance_recovery} \
SEEDS=${SEEDS:-20260806} \
DURATION=${DURATION:-40.0} \
STEADY_START_S=${STEADY_START_S:-10.0} \
CPU_AFFINITY=${CPU_AFFINITY:-0-7,10-31} \
SKIP_EXISTING=${SKIP_EXISTING:-True} \
    bash "${ROOT_DIR}/scripts/sweep_g1_extreme_stand_support_lock_v8_checkpoints_mujoco.sh"
