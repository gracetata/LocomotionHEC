#!/usr/bin/env bash
# Train only safe strict lateral motion and zero-linear-velocity pure yaw from model_9996.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

STAGE=${STAGE:-bootstrap}
PROTECTED_MODEL9996="${LEGGED_LAB_DIR}/../checkpoint/nav2_behavior_model9996_source/model_9996.pt"
PROTECTED_MODEL9996_SIZE=16202421
PROTECTED_MODEL9996_SHA256="bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"

case "${STAGE}" in
    bootstrap)
        TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996Bootstrap-v0"
        SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"${PROTECTED_MODEL9996}"}
        SOURCE_SIZE=${SOURCE_SIZE:-${PROTECTED_MODEL9996_SIZE}}
        SOURCE_SHA256=${SOURCE_SHA256:-"${PROTECTED_MODEL9996_SHA256}"}
        LOAD_ACTOR_AMP_ONLY=True
        LOAD_POLICY_ONLY=False
        COMMAND_BRIDGE_ENABLE=True
        COMMAND_BRIDGE_SCALE=${COMMAND_BRIDGE_SCALE:-0.20}
        COMMAND_BRIDGE_RESIDUAL_LR=${COMMAND_BRIDGE_RESIDUAL_LR:-3.0e-4}
        MAX_ITERATIONS=${MAX_ITERATIONS:-20}
        ;;
    corrective)
        TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996Corrective-v0"
        : "${SOURCE_CHECKPOINT:?corrective requires SOURCE_CHECKPOINT from an accepted bootstrap run}"
        : "${SOURCE_SIZE:?corrective requires SOURCE_SIZE}"
        : "${SOURCE_SHA256:?corrective requires SOURCE_SHA256}"
        [[ "${SOURCE_CHECKPOINT}" != "${PROTECTED_MODEL9996}" ]] || {
            echo "Error: corrective must load a residual-policy checkpoint, not bare model_9996." >&2
            exit 1
        }
        LOAD_ACTOR_AMP_ONLY=False
        LOAD_POLICY_ONLY=True
        COMMAND_BRIDGE_ENABLE=False
        COMMAND_BRIDGE_SCALE=0.0
        COMMAND_BRIDGE_RESIDUAL_LR=0.0
        MAX_ITERATIONS=${MAX_ITERATIONS:-30}
        ;;
    *)
        echo "Error: STAGE must be bootstrap or corrective." >&2
        exit 1
        ;;
esac

EXPERIMENT_NAME="g1_amp_nav2_two_goal_model9996"
OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2TwoGoalModel9996"
STAGING_RUN="_source_model9996_${STAGE}"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"

NUM_ENVS=${NUM_ENVS:-4096}
RUN_NAME=${RUN_NAME:-nav2_two_goal_model9996_${STAGE}}
SEED=${SEED:-44}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.08}
COMMAND_BRIDGE_RESIDUAL_UPDATES=${COMMAND_BRIDGE_RESIDUAL_UPDATES:-1}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}

die() {
    echo "Error: $*" >&2
    exit 1
}

verify_file() {
    local path=$1
    local expected_size=$2
    local expected_sha=$3
    local label=$4
    [[ -f "${path}" ]] || die "${label} is missing: ${path}"
    [[ "$(stat -c '%s' "${path}")" == "${expected_size}" ]] || \
        die "${label} size changed: ${path}"
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]] || \
        die "${label} SHA-256 changed: ${path}"
}

verify_all() {
    verify_file \
        "${PROTECTED_MODEL9996}" \
        "${PROTECTED_MODEL9996_SIZE}" \
        "${PROTECTED_MODEL9996_SHA256}" \
        "protected model_9996"
    verify_file "${SOURCE_CHECKPOINT}" "${SOURCE_SIZE}" "${SOURCE_SHA256}" "stage source"
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    verify_all || status=1
    exit "${status}"
}
trap verify_on_exit EXIT

verify_all
if [[ "${STAGE}" == "bootstrap" ]]; then
    [[ "${SOURCE_CHECKPOINT}" == "${PROTECTED_MODEL9996}" ]] || \
        die "bootstrap source must be the protected model_9996"
fi
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be positive"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be positive"
[[ "${RUN_NAME}" != */* ]] || die "RUN_NAME must not contain a path separator"
[[ "${OUTPUT_DIR}" != "$(dirname "${PROTECTED_MODEL9996}")" ]] || \
    die "output directory must not equal the protected source directory"

mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_source.pt"

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|\
        --run_name|--run_name=*|agent.experiment_name=*|agent.checkpoint_output_dir=*|\
        agent.load_actor_only=*|agent.load_actor_amp_only=*|agent.load_policy_only=*|\
        agent.freeze_base_actor=*|agent.freeze_pure_yaw_residual=*|\
        agent.policy.fixed_command_bridge_fraction=*|\
        agent.algorithm.command_bridge_cfg.enabled=*|agent.algorithm.command_bridge_cfg.scale=*|\
        agent.algorithm.command_bridge_cfg.residual_learning_rate=*|\
        env.commands.base_velocity.mode_sampling_config_path=*|\
        env.commands.base_velocity.mode_probability=*)
            die "protected model_9996 two-goal setting cannot be overridden: ${arg}"
            ;;
    esac
done

echo "=================================================="
echo " G1 Nav2 model_9996 two-goal training"
echo "=================================================="
echo "Stage             : ${STAGE}"
echo "Protected source  : ${PROTECTED_MODEL9996}"
echo "Protected SHA-256 : ${PROTECTED_MODEL9996_SHA256}"
echo "Stage source      : ${SOURCE_CHECKPOINT}"
echo "Stage SHA-256     : ${SOURCE_SHA256}"
echo "Goals             : safe strict lateral + vx=vy=0 pure yaw"
echo "Distribution      : 80% balanced goals + 20% Nav2 retention"
echo "Base actor        : frozen model_9996, strict residual gates"
echo "Deployed carrier  : disabled (fixed fraction is exactly zero)"
echo "Teacher carrier   : ${COMMAND_BRIDGE_ENABLE}, training loss only"
echo "Foot barrier      : soft 0.040 m; hard 0.025 m; weight -12"
echo "Training          : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Run               : ${RUN_NAME}"
echo "Output            : ${OUTPUT_DIR}"
echo "=================================================="

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
TASK_STYLE_LERP=1.0 \
AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${PROTECTED_MODEL9996}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    agent.checkpoint_output_dir="${OUTPUT_DIR}" \
    agent.load_actor_only=False \
    agent.load_actor_amp_only="${LOAD_ACTOR_AMP_ONLY}" \
    agent.load_policy_only="${LOAD_POLICY_ONLY}" \
    agent.reset_iteration_on_policy_only_load=True \
    agent.freeze_base_actor=True \
    agent.freeze_pure_yaw_residual=False \
    agent.policy.fixed_command_bridge_fraction=0.0 \
    agent.algorithm.command_bridge_cfg.enabled="${COMMAND_BRIDGE_ENABLE}" \
    agent.algorithm.command_bridge_cfg.scale="${COMMAND_BRIDGE_SCALE}" \
    agent.algorithm.command_bridge_cfg.residual_learning_rate="${COMMAND_BRIDGE_RESIDUAL_LR}" \
    agent.algorithm.command_bridge_cfg.residual_updates_per_batch="${COMMAND_BRIDGE_RESIDUAL_UPDATES}" \
    "$@"

verify_all
echo "Protected model_9996 and stage source verified unchanged after training."
