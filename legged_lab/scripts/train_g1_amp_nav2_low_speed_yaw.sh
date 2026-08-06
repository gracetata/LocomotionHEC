#!/usr/bin/env bash
# Full-state continuation of model_12995 for low-speed and in-place-yaw specialization.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0"
EXPERIMENT_NAME="g1_amp_nav2_behavior"
SOURCE_RUN="2026-08-04_14-12-30_nav2_behavior_from_model9996_fullstate_3000_20260804"
SOURCE_CHECKPOINT="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${SOURCE_RUN}/model_12995.pt"
SOURCE_SIZE=16202843
SOURCE_SHA256="6862627cdfe5cc95a1c0916c17bbde50d320c0a551da0ab8312bfbce05f09a70"
PROFILE_PATH="${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/nav2_behavior_50hz/task_sampling_low_speed_yaw_config.json"

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
RUN_NAME=${RUN_NAME:-nav2_low_speed_yaw_from_model12995_fullstate_3000_20260804}
SEED=${SEED:-43}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}
RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-1}
MODE_PROBABILITY=${MODE_PROBABILITY:-0.80}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.003}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}

die() {
    echo "Error: $*" >&2
    exit 1
}

verify_source() {
    [[ -f "${SOURCE_CHECKPOINT}" ]] || {
        echo "Error: source model is missing: ${SOURCE_CHECKPOINT}" >&2
        return 1
    }
    [[ "$(stat -c '%s' "${SOURCE_CHECKPOINT}")" == "${SOURCE_SIZE}" ]] || {
        echo "Error: protected model_12995 size changed." >&2
        return 1
    }
    [[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] || {
        echo "Error: protected model_12995 SHA-256 changed." >&2
        return 1
    }
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    verify_source || status=1
    exit "${status}"
}
trap verify_on_exit EXIT

verify_source
[[ -f "${PROFILE_PATH}" ]] || die "specialization profile is missing: ${PROFILE_PATH}"
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be positive"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be positive"
[[ "${RUN_NAME}" != */* ]] || die "RUN_NAME must not contain a path separator"

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|\
        --run_name|--run_name=*|agent.load_actor_only=*|agent.load_policy_only=*|\
        env.commands.base_velocity.mode_sampling_config_path=*|env.commands.base_velocity.mode_probability=*)
            die "Protected specialization setting cannot be overridden: ${arg}"
            ;;
    esac
done

echo "=================================================="
echo " G1 Nav2 low-speed / in-place-yaw specialization"
echo "=================================================="
echo "Source checkpoint : ${SOURCE_CHECKPOINT}"
echo "Source SHA-256    : ${SOURCE_SHA256}"
echo "Continuation      : full policy/critic/PPO/AMP state"
echo "Training          : ${NUM_ENVS} envs × ${MAX_ITERATIONS} iterations"
echo "Mode mix          : ${MODE_PROBABILITY} specialized / recorded Nav2 remainder"
echo "Profile           : 38% micro translation, 52% in-place yaw, 10% stand"
echo "Run               : ${RUN_NAME}"
echo "Log               : ${TRAIN_LOG_FILE}"
echo "=================================================="

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${SOURCE_RUN}$" \
CHECKPOINT="^model_12995.pt$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=True \
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
STYLE_REWARD_SCALE=5.0 \
TASK_STYLE_LERP=0.4 \
AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${SOURCE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.load_actor_only=False \
    agent.load_policy_only=False \
    agent.reset_iteration_on_policy_only_load=False \
    env.commands.base_velocity.mode_sampling_config_path="${PROFILE_PATH}" \
    env.commands.base_velocity.mode_probability="${MODE_PROBABILITY}" \
    "$@"

verify_source
echo "Protected model_12995 verified unchanged after training."
