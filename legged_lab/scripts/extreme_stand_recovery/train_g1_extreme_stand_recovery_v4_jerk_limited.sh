#!/usr/bin/env bash
# Continue the immutable V4 model under state-gated stable-region damping.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_v4_jerk_limited_from_model2999}
MAX_ITERATIONS=${MAX_ITERATIONS:-25}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-5.0e-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
ISAAC_KIT_ARGS=${ISAAC_KIT_ARGS:---/UJITSO/enabled=false --/app/extensions/registryEnabled=false}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.20}
BASELINE_KL_HARD_LIMIT=${BASELINE_KL_HARD_LIMIT:-0.02}
V4_KL_CHECKPOINT=${V4_KL_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}

echo "============================================================"
echo " Extreme Stand V4 state-gated stable-damping continuation"
echo "============================================================"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Base SHA-256    : ${EXPECTED_BASE_SHA256}"
echo "Action path     : exact V4 recovery targets; dissipative velocity feedback when stable"
echo "Actor update    : 25 critic-only iterations, then output layer only"
echo "Retention       : V4 mean-policy KL scale=${BASELINE_KL_SCALE}, hard=${BASELINE_KL_HARD_LIMIT}"
echo "Training        : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Learning rate   : ${LEARNING_RATE}; save every ${SAVE_INTERVAL}"
echo "Isaac Kit args  : ${ISAAC_KIT_ARGS}"
echo "Run name        : ${RUN_NAME}"
echo "============================================================"

TASK=LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-V4JerkLimited-v0 \
DISTURBANCE_MODE=legacy \
BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256}" \
VERIFY_BASE_SHA256=True \
RUN_NAME="${RUN_NAME}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
NUM_ENVS="${NUM_ENVS}" \
LEARNING_RATE="${LEARNING_RATE}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${V4_KL_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${SCRIPT_DIR}/train_g1_extreme_stand_recovery.sh" \
    --kit_args "${ISAAC_KIT_ARGS}" \
    "agent.save_interval=${SAVE_INTERVAL}" \
    agent.algorithm.baseline_kl_cfg.mean_only=True \
    agent.algorithm.baseline_kl_cfg.command_conditioned=False \
    "agent.algorithm.baseline_kl_cfg.hard_limit=${BASELINE_KL_HARD_LIMIT}" \
    env.rewards.dof_torques_l2.weight=-1.0e-5 \
    env.rewards.joint_torque_rate_l2.weight=-2.0e-7 \
    env.rewards.dof_acc_l2.weight=-5.0e-7 \
    env.rewards.joint_jerk_l2.weight=-5.0e-8 \
    env.rewards.action_rate_l2.weight=-5.0e-2 \
    env.rewards.action_second_difference_l2.weight=-1.0e-1 \
    "$@"
