#!/usr/bin/env bash
# Two-stage refinement for safe lateral stepping and zero-linear pure yaw.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

STAGE=${STAGE:-1}
if [[ "${STAGE}" == "1" ]]; then
    TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalFinetune-v0"
elif [[ "${STAGE}" == "2" ]]; then
    TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalStage2Finetune-v0"
else
    echo "Error: STAGE must be 1 or 2." >&2
    exit 1
fi
EXPERIMENT_NAME="g1_amp_nav2_two_goal"
PROTECTED_BASELINE="${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/2026-08-04_14-12-30_nav2_behavior_from_model9996_fullstate_3000_20260804/model_12995.pt"
PROTECTED_BASELINE_SIZE=16202843
PROTECTED_BASELINE_SHA256="6862627cdfe5cc95a1c0916c17bbde50d320c0a551da0ab8312bfbce05f09a70"
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"${PROTECTED_BASELINE}"}
SOURCE_SIZE=${SOURCE_SIZE:-${PROTECTED_BASELINE_SIZE}}
SOURCE_SHA256=${SOURCE_SHA256:-"${PROTECTED_BASELINE_SHA256}"}
STAGING_RUN="_source_two_goal_stage${STAGE}"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"
OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2TwoGoalFinetune"

NUM_ENVS=${NUM_ENVS:-4096}
# Each curriculum stage is deliberately short and evaluated every ten updates.
MAX_ITERATIONS=${MAX_ITERATIONS:-60}
RUN_NAME=${RUN_NAME:-nav2_two_goal_stage${STAGE}}
SEED=${SEED:-44}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.08}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}

die() {
    echo "Error: $*" >&2
    exit 1
}

verify_source() {
    [[ -f "${SOURCE_CHECKPOINT}" ]] || {
        echo "Error: protected model_12995 is missing: ${SOURCE_CHECKPOINT}" >&2
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

verify_baseline() {
    [[ -f "${PROTECTED_BASELINE}" ]] || {
        echo "Error: protected model_12995 is missing: ${PROTECTED_BASELINE}" >&2
        return 1
    }
    [[ "$(stat -c '%s' "${PROTECTED_BASELINE}")" == "${PROTECTED_BASELINE_SIZE}" ]] || {
        echo "Error: protected model_12995 size changed." >&2
        return 1
    }
    [[ "$(sha256sum "${PROTECTED_BASELINE}" | awk '{print $1}')" == "${PROTECTED_BASELINE_SHA256}" ]] || {
        echo "Error: protected model_12995 SHA-256 changed." >&2
        return 1
    }
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    verify_source || status=1
    verify_baseline || status=1
    exit "${status}"
}
trap verify_on_exit EXIT

verify_source
verify_baseline
if [[ "${STAGE}" == "2" && "${SOURCE_CHECKPOINT}" == "${PROTECTED_BASELINE}" ]]; then
    die "STAGE=2 requires SOURCE_CHECKPOINT, SOURCE_SIZE, and SOURCE_SHA256 from an accepted stage-1 checkpoint"
fi
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be positive"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be positive"
[[ "${RUN_NAME}" != */* ]] || die "RUN_NAME must not contain a path separator"

mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_source.pt"

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|\
        --run_name|--run_name=*|agent.experiment_name=*|agent.checkpoint_output_dir=*|\
        agent.load_actor_only=*|agent.load_actor_amp_only=*|agent.load_policy_only=*|\
        agent.freeze_actor_hidden_layers=*|agent.algorithm.amp_cfg.freeze_discriminator=*|\
        env.commands.base_velocity.mode_sampling_config_path=*|\
        env.commands.base_velocity.mode_probability=*)
            die "Protected two-goal setting cannot be overridden: ${arg}"
            ;;
    esac
done

echo "=================================================="
echo " G1 Nav2 two-goal conservative refinement"
echo "=================================================="
echo "Source checkpoint : ${SOURCE_CHECKPOINT}"
echo "Source SHA-256    : ${SOURCE_SHA256}"
echo "Goals             : safe pure lateral + zero-linear pure yaw"
echo "Distribution      : 80% balanced goals + 20% recorded Nav2 anchor"
if [[ "${STAGE}" == "1" ]]; then
    echo "Load contract     : actor + frozen AMP; fresh critic/optimizers"
else
    echo "Load contract     : full-state continuation from accepted stage 1"
fi
echo "Optimization      : lr=7.5e-6, PPO epochs=2, clip=0.12, actor first layer frozen"
echo "Actor schedule    : stage 1 critic-only iterations 0-11, then actor enabled"
echo "Baseline KL       : specialization=0.005, retention=${BASELINE_KL_SCALE}, hard=0.15"
echo "Training          : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Run               : ${RUN_NAME}"
echo "Output            : ${OUTPUT_DIR}"
echo "=================================================="

if [[ "${STAGE}" == "1" ]]; then
    LOAD_ACTOR_AMP_ONLY=True
else
    LOAD_ACTOR_AMP_ONLY=False
fi

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${STAGING_RUN}$" \
CHECKPOINT="^model_source.pt$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
STYLE_REWARD_SCALE=5.0 \
TASK_STYLE_LERP=0.85 \
AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${PROTECTED_BASELINE}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    agent.checkpoint_output_dir="${OUTPUT_DIR}" \
    agent.load_actor_only=False \
    agent.load_actor_amp_only="${LOAD_ACTOR_AMP_ONLY}" \
    agent.load_policy_only=False \
    agent.reset_iteration_on_policy_only_load=True \
    "$@"

verify_source
verify_baseline
echo "Protected model_12995 verified unchanged after training."
