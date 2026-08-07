#!/usr/bin/env bash
# Correct V5 target drift and train against torso-equivalent impulses up to 72 N.s.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-08-07_12-25-12_g1_extreme_stand_smooth_settle_v5_resume900_to1499_20260807/model_1499.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-13538475518be2a323dfedff230949b3c6b8057c8f4f9af000adbbfd90c7ee7c}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_target_lock_v6_from_v5_model1499}
MAX_ITERATIONS=${MAX_ITERATIONS:-1000}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-5.0e-6}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}

LEG_NOISE_RAD=${LEG_NOISE_RAD:-0.30}
WAIST_NOISE_RAD=${WAIST_NOISE_RAD:-0.40}
ARM_NOISE_RAD=${ARM_NOISE_RAD:-0.65}

[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: SAVE_INTERVAL must be a positive integer, got ${SAVE_INTERVAL}." >&2
    exit 1
}

echo "============================================================"
echo " Extreme Stand Recovery Target-Lock V6"
echo "============================================================"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Policy contract : 96 observations -> 29 full-body actions"
echo "Push curriculum : 45, 90, 150, 240 N torso-equivalent"
echo "Impulse / quiet : 0.10--0.30 s / 8--12 s"
echo "V6 correction   : rational settle reward + near-default target lock"
echo "Training        : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Learning rate   : ${LEARNING_RATE}"
echo "Run name        : ${RUN_NAME}"
echo "============================================================"

TASK=LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-TargetLockV6-v0 \
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
