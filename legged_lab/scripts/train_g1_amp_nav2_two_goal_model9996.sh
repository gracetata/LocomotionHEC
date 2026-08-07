#!/usr/bin/env bash
# Train only safe strict lateral motion and zero-linear-velocity pure yaw from model_9996.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

STAGE=${STAGE:-bootstrap}
PROTECTED_MODEL9996=$(realpath \
    "${LEGGED_LAB_DIR}/../checkpoint/nav2_behavior_model9996_source/model_9996.pt")
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
        FOOT_BARRIER_DESCRIPTION="soft 0.040 m; hard 0.025 m; weight -12"
        MAX_ITERATIONS=${MAX_ITERATIONS:-20}
        FREEZE_LATERAL_RESIDUAL=False
        FREEZE_PURE_YAW_RESIDUAL=False
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
        FOOT_BARRIER_DESCRIPTION="soft 0.040 m; hard 0.025 m; weight -12"
        MAX_ITERATIONS=${MAX_ITERATIONS:-30}
        FREEZE_LATERAL_RESIDUAL=False
        FREEZE_PURE_YAW_RESIDUAL=False
        ;;
    barrier_corrective)
        TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996BarrierCorrective-v0"
        : "${SOURCE_CHECKPOINT:?barrier_corrective requires an accepted moving policy}"
        : "${SOURCE_SIZE:?barrier_corrective requires SOURCE_SIZE}"
        : "${SOURCE_SHA256:?barrier_corrective requires SOURCE_SHA256}"
        [[ "${SOURCE_CHECKPOINT}" != "${PROTECTED_MODEL9996}" ]] || {
            echo "Error: barrier_corrective must load a residual-policy checkpoint." >&2
            exit 1
        }
        LOAD_ACTOR_AMP_ONLY=False
        LOAD_POLICY_ONLY=True
        COMMAND_BRIDGE_ENABLE=False
        COMMAND_BRIDGE_SCALE=0.0
        COMMAND_BRIDGE_RESIDUAL_LR=0.0
        FOOT_BARRIER_DESCRIPTION="soft 0.080 m; hard 0.040 m; weight -50; accept 0.025 m"
        MAX_ITERATIONS=${MAX_ITERATIONS:-30}
        FREEZE_LATERAL_RESIDUAL=False
        FREEZE_PURE_YAW_RESIDUAL=False
        ;;
    lateral_direct)
        TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996LateralSpecialist-v0"
        SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"${PROTECTED_MODEL9996}"}
        SOURCE_SIZE=${SOURCE_SIZE:-${PROTECTED_MODEL9996_SIZE}}
        SOURCE_SHA256=${SOURCE_SHA256:-"${PROTECTED_MODEL9996_SHA256}"}
        [[ "${SOURCE_CHECKPOINT}" == "${PROTECTED_MODEL9996}" ]] || {
            echo "Error: lateral_direct must start with zero residuals from protected model_9996." >&2
            exit 1
        }
        LOAD_ACTOR_AMP_ONLY=True
        LOAD_POLICY_ONLY=False
        COMMAND_BRIDGE_ENABLE=False
        COMMAND_BRIDGE_SCALE=0.0
        COMMAND_BRIDGE_RESIDUAL_LR=0.0
        FREEZE_LATERAL_RESIDUAL=False
        FREEZE_PURE_YAW_RESIDUAL=True
        FOOT_BARRIER_DESCRIPTION="direct model_9996 lateral: soft 0.100 m; hard 0.045 m; weight -100"
        MAX_ITERATIONS=${MAX_ITERATIONS:-40}
        ;;
    lateral_specialist)
        TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996LateralSpecialist-v0"
        : "${SOURCE_CHECKPOINT:?lateral_specialist requires a moving residual policy}"
        : "${SOURCE_SIZE:?lateral_specialist requires SOURCE_SIZE}"
        : "${SOURCE_SHA256:?lateral_specialist requires SOURCE_SHA256}"
        LOAD_ACTOR_AMP_ONLY=False
        LOAD_POLICY_ONLY=True
        COMMAND_BRIDGE_ENABLE=False
        COMMAND_BRIDGE_SCALE=0.0
        COMMAND_BRIDGE_RESIDUAL_LR=0.0
        FREEZE_LATERAL_RESIDUAL=False
        FREEZE_PURE_YAW_RESIDUAL=True
        FOOT_BARRIER_DESCRIPTION="lateral-only: soft 0.100 m; hard 0.045 m; weight -100"
        MAX_ITERATIONS=${MAX_ITERATIONS:-40}
        ;;
    yaw_specialist)
        TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996YawSpecialist-v0"
        : "${SOURCE_CHECKPOINT:?yaw_specialist requires the accepted lateral checkpoint}"
        : "${SOURCE_SIZE:?yaw_specialist requires SOURCE_SIZE}"
        : "${SOURCE_SHA256:?yaw_specialist requires SOURCE_SHA256}"
        LOAD_ACTOR_AMP_ONLY=False
        LOAD_POLICY_ONLY=True
        COMMAND_BRIDGE_ENABLE=False
        COMMAND_BRIDGE_SCALE=0.0
        COMMAND_BRIDGE_RESIDUAL_LR=0.0
        FREEZE_LATERAL_RESIDUAL=True
        FREEZE_PURE_YAW_RESIDUAL=False
        FOOT_BARRIER_DESCRIPTION="yaw-only: zero linear command; drift constraint unsaturated"
        MAX_ITERATIONS=${MAX_ITERATIONS:-40}
        ;;
    *)
        echo "Error: invalid STAGE for model_9996 two-goal training." >&2
        exit 1
        ;;
esac

# Symlink targets must remain valid from the staging directory even when the
# caller supplied a repository-relative checkpoint path.
SOURCE_CHECKPOINT=$(realpath "${SOURCE_CHECKPOINT}")

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
if [[ "${STAGE}" == "bootstrap" || "${STAGE}" == "lateral_direct" ]]; then
    [[ "${SOURCE_CHECKPOINT}" == "${PROTECTED_MODEL9996}" ]] || \
        die "${STAGE} source must be the protected model_9996"
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
        agent.freeze_base_actor=*|agent.freeze_lateral_residual=*|agent.freeze_pure_yaw_residual=*|\
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
echo "Foot barrier      : ${FOOT_BARRIER_DESCRIPTION}"
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
    agent.freeze_lateral_residual="${FREEZE_LATERAL_RESIDUAL}" \
    agent.freeze_pure_yaw_residual="${FREEZE_PURE_YAW_RESIDUAL}" \
    agent.policy.fixed_command_bridge_fraction=0.0 \
    agent.algorithm.command_bridge_cfg.enabled="${COMMAND_BRIDGE_ENABLE}" \
    agent.algorithm.command_bridge_cfg.scale="${COMMAND_BRIDGE_SCALE}" \
    agent.algorithm.command_bridge_cfg.residual_learning_rate="${COMMAND_BRIDGE_RESIDUAL_LR}" \
    agent.algorithm.command_bridge_cfg.residual_updates_per_batch="${COMMAND_BRIDGE_RESIDUAL_UPDATES}" \
    "$@"

if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|NaN|nan detected|TypeError:' \
    "${TRAIN_LOG_FILE}"; then
    die "training log contains a fatal Python/CUDA/numerical error: ${TRAIN_LOG_FILE}"
fi
grep -q 'Learning iteration' "${TRAIN_LOG_FILE}" || \
    die "training did not complete a PPO iteration: ${TRAIN_LOG_FILE}"
grep -q 'Saved dedicated AMP checkpoint copy' "${TRAIN_LOG_FILE}" || \
    die "training produced no dedicated checkpoint: ${TRAIN_LOG_FILE}"

verify_all
echo "Protected model_9996 and stage source verified unchanged after training."
