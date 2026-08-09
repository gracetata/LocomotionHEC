#!/usr/bin/env bash
# Continue a full-capacity lateral expert while keeping model_9996 protected.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODEL9996=$(realpath "${LEGGED_LAB_DIR}/../checkpoint/nav2_behavior_model9996_source/model_9996.pt")
MODEL9996_SIZE=16202421
MODEL9996_SHA256="bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"
STAGE=${STAGE:-spacing}
if [[ "${STAGE}" == "spacing" ]]; then
    TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActorLateral-v0"
elif [[ "${STAGE}" == "final" ]]; then
    TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActorLateralFinal-v0"
elif [[ "${STAGE}" == "robust" ]]; then
    TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActorLateralRobust-v0"
else
    echo "Error: STAGE must be spacing, final, or robust." >&2
    exit 1
fi
if [[ "${STAGE}" == "robust" ]]; then
    EXPERIMENT_NAME="g1_amp_nav2_two_goal_model9996_full_actor_lateral_robust"
    OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2TwoGoalModel9996FullActorLateralRobust"
    RANDOMIZATION_STRENGTH=1
else
    EXPERIMENT_NAME="g1_amp_nav2_two_goal_model9996_full_actor_lateral"
    OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2TwoGoalModel9996FullActorLateral"
    RANDOMIZATION_STRENGTH=0
fi

: "${SOURCE_CHECKPOINT:?set SOURCE_CHECKPOINT to a finite full-actor lateral candidate}"
: "${SOURCE_SIZE:?set SOURCE_SIZE}"
: "${SOURCE_SHA256:?set SOURCE_SHA256}"
SOURCE_CHECKPOINT=$(realpath "${SOURCE_CHECKPOINT}")
STAGING_RUN="_source_model9996_full_actor_lateral"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-40}
RUN_NAME=${RUN_NAME:-model9996_full_actor_lateral}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}

die() { echo "Error: $*" >&2; exit 1; }
verify_file() {
    local path=$1 size=$2 sha=$3
    [[ -f "${path}" ]] || return 1
    [[ "$(stat -c '%s' "${path}")" == "${size}" ]] || return 1
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${sha}" ]]
}
verify_all() {
    verify_file "${MODEL9996}" "${MODEL9996_SIZE}" "${MODEL9996_SHA256}" &&
    verify_file "${SOURCE_CHECKPOINT}" "${SOURCE_SIZE}" "${SOURCE_SHA256}"
}
verify_on_exit() {
    local status=$?
    trap - EXIT
    verify_all || status=1
    exit "${status}"
}
trap verify_on_exit EXIT

verify_all || die "protected model_9996 or stage source failed its immutable contract"
[[ "${SOURCE_CHECKPOINT}" != "${MODEL9996}" ]] || die "lateral continuation needs a learned full actor"
mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_source.pt"

echo "=================================================="
echo " model_9996 gated full-actor lateral expert"
echo "=================================================="
echo "Protected source  : ${MODEL9996_SHA256}"
echo "Stage source      : ${SOURCE_SHA256}"
echo "Commands          : vx=wz=0, vy=+/-[0.20,0.35]"
echo "Deployment intent : this actor is gated to strict lateral only"
echo "Stage             : ${STAGE}"
if [[ "${STAGE}" == "spacing" ]]; then
    echo "Foot barrier      : hard -150 below 0.025 m; ordering -30"
    echo "Response          : signed +100; stationary shortfall -300"
else
    if [[ "${STAGE}" == "final" ]]; then
        echo "Foot barrier      : hard -150; safe-set shaping reduced"
        echo "Response/leak     : signed +80; shortfall -400; leak -20"
    else
        echo "Foot barrier      : hard -300 below 0.030 m; overlap x16"
        echo "Randomization     : moderate friction/mass/COM/actuator/joint/push"
    fi
fi
echo "Training          : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "=================================================="

TASK="${TASK}" NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" \
RUN_NAME="${RUN_NAME}" RESUME=True LOAD_RUN="^${STAGING_RUN}$" CHECKPOINT="^model_source.pt$" \
HEADLESS=True QUIET_TERMINAL=True TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" STYLE_REWARD_SCALE=5.0 TASK_STYLE_LERP=1.0 \
AMP_GRAD_PENALTY_SCALE=20.0 BASELINE_KL_ENABLE=False \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    agent.checkpoint_output_dir="${OUTPUT_DIR}" \
    agent.load_actor_only=False \
    agent.load_actor_amp_only=False \
    agent.load_policy_only=True \
    agent.reset_iteration_on_policy_only_load=True \
    agent.freeze_base_actor=False \
    agent.freeze_actor_hidden_layers=1 \
    agent.actor_warmup_iterations=0 \
    agent.algorithm.baseline_kl_cfg.enabled=False \
    agent.algorithm.amp_cfg.freeze_discriminator=True \
    "$@"

grep -q 'Learning iteration' "${TRAIN_LOG_FILE}" || die "training completed no PPO iteration"
if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|NaN|nan detected|TypeError:' \
    "${TRAIN_LOG_FILE}"; then
    die "training log contains a fatal Python/CUDA/numerical error"
fi
verify_all || die "immutable source changed during training"
echo "Protected model_9996 and lateral source verified unchanged."
