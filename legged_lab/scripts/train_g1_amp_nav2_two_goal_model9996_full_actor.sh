#!/usr/bin/env bash
# Full-actor two-goal refinement from the exact protected model_9996 actor.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODEL9996=$(realpath "${LEGGED_LAB_DIR}/../checkpoint/nav2_behavior_model9996_source/model_9996.pt")
MODEL9996_SIZE=16202421
MODEL9996_SHA256="bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"
TASK="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActor-v0"
EXPERIMENT_NAME="g1_amp_nav2_two_goal_model9996_full_actor"
OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2TwoGoalModel9996FullActor"
STAGING_RUN="_source_model9996_full_actor"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-80}
RUN_NAME=${RUN_NAME:-model9996_full_actor_two_goal}
SEED=${SEED:-44}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}

die() { echo "Error: $*" >&2; exit 1; }

verify_model9996() {
    [[ -f "${MODEL9996}" ]] || return 1
    [[ "$(stat -c '%s' "${MODEL9996}")" == "${MODEL9996_SIZE}" ]] || return 1
    [[ "$(sha256sum "${MODEL9996}" | awk '{print $1}')" == "${MODEL9996_SHA256}" ]]
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    verify_model9996 || status=1
    exit "${status}"
}
trap verify_on_exit EXIT

verify_model9996 || die "protected model_9996 failed its size/SHA-256 contract"
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be positive"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be positive"
[[ "${RUN_NAME}" != */* ]] || die "RUN_NAME must not contain a path separator"

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|\
        agent.experiment_name=*|agent.checkpoint_output_dir=*|agent.load_actor_only=*|\
        agent.load_actor_amp_only=*|agent.load_policy_only=*|agent.freeze_base_actor=*|\
        agent.freeze_actor_hidden_layers=*|agent.actor_warmup_iterations=*|\
        agent.algorithm.baseline_kl_cfg.*|agent.algorithm.amp_cfg.freeze_discriminator=*|\
        env.commands.base_velocity.mode_sampling_config_path=*|\
        env.commands.base_velocity.mode_probability=*)
            die "protected full-actor setting cannot be overridden: ${arg}"
            ;;
    esac
done

mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${MODEL9996}" "${STAGING_DIR}/model_source.pt"

echo "=================================================="
echo " G1 Nav2 model_9996 full-actor two-goal refinement"
echo "=================================================="
echo "Source            : ${MODEL9996}"
echo "Source SHA-256    : ${MODEL9996_SHA256}"
echo "Goals             : safe strict lateral + vx=vy=0 pure yaw"
echo "Distribution      : 80% balanced goals + 20% Nav2 retention"
echo "Capacity          : first actor layer frozen; remaining actor trainable"
echo "Actor warmup      : 8 critic-only iterations"
echo "Baseline KL       : specialization=0.005; retention=0.08; hard=0.15"
echo "Foot safety       : shape-aware hard barrier -100 below 0.025 m"
echo "Training          : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Run               : ${RUN_NAME}"
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
TASK_STYLE_LERP=0.85 \
AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${MODEL9996}" \
BASELINE_KL_SCALE=0.08 \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    agent.checkpoint_output_dir="${OUTPUT_DIR}" \
    agent.load_actor_only=False \
    agent.load_actor_amp_only=True \
    agent.load_policy_only=False \
    agent.freeze_base_actor=False \
    agent.freeze_actor_hidden_layers=1 \
    agent.actor_warmup_iterations=8 \
    agent.algorithm.amp_cfg.freeze_discriminator=True \
    "$@"

grep -q 'Learning iteration' "${TRAIN_LOG_FILE}" || die "training completed no PPO iteration"
if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|NaN|nan detected|TypeError:' \
    "${TRAIN_LOG_FILE}"; then
    die "training log contains a fatal Python/CUDA/numerical error"
fi
verify_model9996 || die "protected model_9996 changed during training"
echo "Protected model_9996 verified unchanged after training."
