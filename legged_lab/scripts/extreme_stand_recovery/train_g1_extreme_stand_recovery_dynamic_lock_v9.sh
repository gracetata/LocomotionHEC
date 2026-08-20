#!/usr/bin/env bash
# Preserve V4 with frozen-policy KL and a joint-independent stable-state lock.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_dynamic_lock_v9_from_v4_model2999}
MAX_ITERATIONS=${MAX_ITERATIONS:-200}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-2.0e-7}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.08}
BASELINE_KL_HARD_LIMIT=${BASELINE_KL_HARD_LIMIT:-0.15}

LEG_NOISE_RAD=${LEG_NOISE_RAD:-0.30}
WAIST_NOISE_RAD=${WAIST_NOISE_RAD:-0.40}
ARM_NOISE_RAD=${ARM_NOISE_RAD:-0.65}

echo "============================================================"
echo " Extreme Stand Recovery Dynamic-Lock V9"
echo "============================================================"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Policy contract : 96 observations -> 29 full-body actions"
echo "Retention       : frozen V4 mean-policy KL scale ${BASELINE_KL_SCALE}"
echo "KL safety limit : ${BASELINE_KL_HARD_LIMIT} (abort before further forgetting)"
echo "Target lock     : root upright + low velocity -> full-body Top-K"
echo "Environment mix : 75% clean anchors + 25% one-at-a-time pushes"
echo "Push curriculum : 20, 45, 90, 180 N"
echo "Training        : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Learning rate   : ${LEARNING_RATE}; save every ${SAVE_INTERVAL}"
echo "Run name        : ${RUN_NAME}"
echo "============================================================"

TASK=LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-DynamicLockV9-v0 \
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
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${SCRIPT_DIR}/train_g1_extreme_stand_recovery.sh" \
    "agent.save_interval=${SAVE_INTERVAL}" \
    agent.algorithm.baseline_kl_cfg.mean_only=True \
    agent.algorithm.baseline_kl_cfg.command_conditioned=False \
    "agent.algorithm.baseline_kl_cfg.hard_limit=${BASELINE_KL_HARD_LIMIT}" \
    "$@"
