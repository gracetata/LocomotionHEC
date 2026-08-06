#!/usr/bin/env bash
# Continue V4 with target settling, normalized Top-K smoothing and one-at-a-time pushes.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_smooth_settle_v5_from_v4_model2999}
MAX_ITERATIONS=${MAX_ITERATIONS:-1500}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-1.0e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}

# Wider reset noise is retained from V4.  Forces are controlled exclusively by
# the V5 task's 10 -> 20 -> 36 -> 45 N non-overlapping curriculum.
LEG_NOISE_RAD=${LEG_NOISE_RAD:-0.30}
WAIST_NOISE_RAD=${WAIST_NOISE_RAD:-0.40}
ARM_NOISE_RAD=${ARM_NOISE_RAD:-0.65}

[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: SAVE_INTERVAL must be a positive integer, got ${SAVE_INTERVAL}." >&2
    exit 1
}

echo "============================================================"
echo " Extreme Stand Recovery Smooth-Settle V5"
echo "============================================================"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Policy contract : 96 observations -> 29 full-body actions"
echo "Push curriculum : 10, 20, 36, 45 N; one body/direction at a time"
echo "Impulse / quiet : 0.10--0.30 s / 6--10 s"
echo "Direction prob. : forward=.15, backward=.30, left=.20, right=.35"
echo "Training        : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Save interval   : ${SAVE_INTERVAL} iterations"
echo "Run name        : ${RUN_NAME}"
echo "============================================================"

TASK=LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-SmoothSettleV5-v0 \
DISTURBANCE_MODE=single \
BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256}" \
VERIFY_BASE_SHA256=True \
RUN_NAME="${RUN_NAME}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
NUM_ENVS="${NUM_ENVS}" \
LEARNING_RATE="${LEARNING_RATE}" \
LEG_NOISE_RAD="${LEG_NOISE_RAD}" \
WAIST_NOISE_RAD="${WAIST_NOISE_RAD}" \
ARM_NOISE_RAD="${ARM_NOISE_RAD}" \
bash "${SCRIPT_DIR}/train_g1_extreme_stand_recovery.sh" \
    "agent.save_interval=${SAVE_INTERVAL}" \
    "$@"
