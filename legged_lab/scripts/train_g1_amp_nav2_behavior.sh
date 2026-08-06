#!/usr/bin/env bash
# Actor-only refinement of the generic full-body G1 Nav2 velocity policy.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0"
EXPERIMENT_NAME="g1_amp_nav2_behavior"
BASE_CHECKPOINT="${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"
EXPECTED_BASE_SHA256="1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
EXPECTED_BASE_SIZE=14826139
STAGING_RUN="_baseline_model10990_actor_only"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"
OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2BehaviorFinetune"

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
RUN_NAME=${RUN_NAME:-nav2_behavior_from_model10990}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}
RSI_ENABLE=${RSI_ENABLE:-True}
RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-1}
STYLE_REWARD_SCALE=${STYLE_REWARD_SCALE:-5.0}
TASK_STYLE_LERP=${TASK_STYLE_LERP:-0.4}
AMP_GRAD_PENALTY_SCALE=${AMP_GRAD_PENALTY_SCALE:-20.0}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.003}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"}

die() {
    echo "Error: $*" >&2
    exit 1
}

verify_baseline() {
    [[ -f "${BASE_CHECKPOINT}" ]] || {
        echo "Error: protected baseline is missing: ${BASE_CHECKPOINT}" >&2
        return 1
    }
    local actual_size actual_sha
    actual_size=$(stat -c '%s' "${BASE_CHECKPOINT}")
    actual_sha=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
    [[ "${actual_size}" == "${EXPECTED_BASE_SIZE}" ]] || {
        echo "Error: protected model_10990 size changed: ${actual_size}" >&2
        return 1
    }
    [[ "${actual_sha}" == "${EXPECTED_BASE_SHA256}" ]] || {
        echo "Error: protected model_10990 SHA-256 changed: ${actual_sha}" >&2
        return 1
    }
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    if ! verify_baseline; then
        status=1
    fi
    exit "${status}"
}
trap verify_on_exit EXIT

verify_baseline
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be a positive integer"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be a positive integer"
[[ "${RUN_NAME}" != */* ]] || die "RUN_NAME must not contain a path separator"

mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}"
BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")
PROTECTED_DIR=$(realpath "${PROJECT_ROOT}/checkpoint/walk")
case "${OUTPUT_DIR}/" in
    "${PROTECTED_DIR}/"*) die "Nav2 behavior output must not be inside checkpoint/walk" ;;
esac
TRAIN_LOG_FILE=$(realpath -m "${TRAIN_LOG_FILE}")
case "${TRAIN_LOG_FILE}" in
    "${PROTECTED_DIR}"/*) die "training log must not be inside checkpoint/walk" ;;
esac
ln -sfn "${BASE_CHECKPOINT}" "${STAGING_DIR}/model_10990.pt"

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|\
        --run_name|--run_name=*|\
        agent.experiment_name=*|agent.checkpoint_output_dir=*|agent.load_actor_only=*|\
        agent.load_policy_only=*|agent.reset_iteration_on_policy_only_load=*|\
        env.upper_body_perturbation*|env.actions.joint_pos.joint_names=*)
            die "Protected generic Nav2 setting cannot be overridden: ${arg}"
            ;;
    esac
done

echo "=================================================="
echo " Generic full-body G1 Nav2 behavior refinement"
echo "=================================================="
echo "Task             : ${TASK}"
echo "Protected base   : ${BASE_CHECKPOINT}"
echo "Base SHA-256     : ${EXPECTED_BASE_SHA256}"
echo "Training         : ${NUM_ENVS} envs × ${MAX_ITERATIONS} iterations"
echo "Run              : ${RUN_NAME}"
echo "Experiment       : ${EXPERIMENT_NAME}"
echo "Dedicated output : ${OUTPUT_DIR}"
echo "Load contract    : actor-only, fresh critic/PPO/AMP, iteration 0"
echo "=================================================="

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${STAGING_RUN}$" \
CHECKPOINT="^model_10990.pt$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE="${RSI_ENABLE}" \
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
STYLE_REWARD_SCALE="${STYLE_REWARD_SCALE}" \
TASK_STYLE_LERP="${TASK_STYLE_LERP}" \
AMP_GRAD_PENALTY_SCALE="${AMP_GRAD_PENALTY_SCALE}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
EXTRA_HYDRA_ARGS="" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.load_actor_only=True \
    agent.load_policy_only=False \
    agent.reset_iteration_on_policy_only_load=True \
    "$@"

verify_baseline
echo "Protected model_10990 verified unchanged after training."
