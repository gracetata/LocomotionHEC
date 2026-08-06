#!/usr/bin/env bash
# Full-state continuation after the model_600 checkpoint from the first Nav2 behavior run.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

SOURCE_RUN="2026-07-31_00-06-03_nav2_behavior_model10990_full_3000_20260731"
SOURCE_CHECKPOINT="${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/${SOURCE_RUN}/model_600.pt"
SOURCE_CHECKPOINT_SHA256="1166a35a504e6f7fee6eb0c00247f10dec55309ee3731ed17e7995883387e119"
SOURCE_CHECKPOINT_SIZE=16202319
BASE_CHECKPOINT="${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"
BASE_SHA256="1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
BASE_SIZE=14826139

NUM_ENVS=${NUM_ENVS:-4096}
REMAINING_ITERATIONS=${REMAINING_ITERATIONS:-2400}
RUN_NAME=${RUN_NAME:-nav2_behavior_resume600_to2999_20260731}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/train_${RUN_NAME}.log"}

verify_file() {
    local path=$1 expected_size=$2 expected_sha=$3
    [[ -f "${path}" ]]
    [[ "$(stat -c '%s' "${path}")" == "${expected_size}" ]]
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    if ! verify_file "${BASE_CHECKPOINT}" "${BASE_SIZE}" "${BASE_SHA256}"; then
        echo "Error: protected model_10990 failed its post-training integrity check." >&2
        status=1
    fi
    exit "${status}"
}
trap verify_on_exit EXIT

verify_file "${BASE_CHECKPOINT}" "${BASE_SIZE}" "${BASE_SHA256}" || {
    echo "Error: protected model_10990 failed its integrity check." >&2
    exit 1
}
verify_file \
    "${SOURCE_CHECKPOINT}" \
    "${SOURCE_CHECKPOINT_SIZE}" \
    "${SOURCE_CHECKPOINT_SHA256}" || {
    echo "Error: model_600 failed its full-state continuation contract." >&2
    exit 1
}

TASK=LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0 \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${REMAINING_ITERATIONS}" \
SEED=42 \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${SOURCE_RUN}$" \
CHECKPOINT='^model_600.pt$' \
HEADLESS=True \
QUIET_TERMINAL=True \
TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=True \
RANDOMIZATION_STRENGTH=1 \
STYLE_REWARD_SCALE=5.0 \
TASK_STYLE_LERP=0.4 \
AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE=0.003 \
EXTRA_HYDRA_ARGS="" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.load_actor_only=False \
    agent.load_policy_only=False \
    agent.reset_iteration_on_policy_only_load=False
