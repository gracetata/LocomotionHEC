#!/usr/bin/env bash
# Fixed 68-run V4 versus V4-derived state-gated target-jerk candidate gate.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=${RUN_DIR:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_16-37-50_g1_extreme_stand_v4_jerk_limited_from_model2999_full_20260807"}
CANDIDATE_ID=${CANDIDATE_ID:-20}
CANDIDATE_CHECKPOINT=${CANDIDATE_CHECKPOINT:-"${RUN_DIR}/model_${CANDIDATE_ID}.pt"}

V5_CHECKPOINT="${CANDIDATE_CHECKPOINT}" \
BASELINE_LABEL=v4_baseline \
CANDIDATE_LABEL="v4_jerk_limited_model${CANDIDATE_ID}" \
BASELINE_TARGET_LIMITER_ENABLE=False \
CANDIDATE_TARGET_LIMITER_ENABLE=True \
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_jerk_limited_full_gate_20260807"} \
    bash "${ROOT_DIR}/scripts/test_g1_extreme_stand_smooth_settle_v5_mujoco.sh"
