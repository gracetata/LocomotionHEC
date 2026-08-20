#!/usr/bin/env bash
# Train one ArmHack Walk gated expert.  GPU training is intentionally locked to future5090.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

EXPERT=${EXPERT:-lateral}
STAGE=${STAGE:-learn}
BASE_CHECKPOINT="${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"
BASE_SIZE=14826139
BASE_SHA256="1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${BASE_CHECKPOINT}}
SOURCE_SIZE=${SOURCE_SIZE:-${BASE_SIZE}}
SOURCE_SHA256=${SOURCE_SHA256:-${BASE_SHA256}}
NUM_ENVS=${NUM_ENVS:-4096}
SEED=${SEED:-42}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
export PATH="${HOME}/.local/bin:${PATH}"

die() { echo "Error: $*" >&2; exit 1; }
verify_file() {
    local path=$1 size=$2 sha=$3
    [[ -f "${path}" ]] || return 1
    [[ "$(stat -c '%s' "${path}")" == "${size}" ]] || return 1
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${sha}" ]]
}

[[ "$(hostname)" == "tata-futurelab" ]] || die "training is locked to future5090 (tata-futurelab)"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 5090' || die "future5090 RTX 5090 not detected"
[[ -x "${ISAACLAB_PYTHON}" ]] || die "IsaacLab Python is not executable: ${ISAACLAB_PYTHON}"
case "${EXPERT}" in lateral|yaw) ;; *) die "EXPERT must be lateral or yaw" ;; esac
case "${STAGE}" in learn|robust) ;; *) die "STAGE must be learn or robust" ;; esac
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be positive"

if pgrep -af 'train.py --task LeggedLab-Isaac-AMP-G1-.*Stand' >/dev/null; then
    die "Stand training is active; refusing to start Walk training"
fi
if pgrep -af 'train.py --task LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoal' >/dev/null; then
    die "another ArmHack Walk two-goal training process is active"
fi

SOURCE_CHECKPOINT=$(realpath "${SOURCE_CHECKPOINT}")
BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
verify_file "${BASE_CHECKPOINT}" "${BASE_SIZE}" "${BASE_SHA256}" || die "protected model_10990 contract failed"
verify_file "${SOURCE_CHECKPOINT}" "${SOURCE_SIZE}" "${SOURCE_SHA256}" || die "expert source contract failed"

if [[ "${EXPERT}" == "lateral" ]]; then
    if [[ "${STAGE}" == "robust" ]]; then
        TASK="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalLateralRobust-v0"
    else
        TASK="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalLateral-v0"
    fi
else
    if [[ "${STAGE}" == "robust" ]]; then
        TASK="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalYawRobust-v0"
    else
        TASK="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalYaw-v0"
    fi
fi

EXPERIMENT_NAME="g1_armhack_walk_two_goal_${EXPERT}_${STAGE}"
OUTPUT_DIR="${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkTwoGoalFinetune/${EXPERT}/${STAGE}"
RUN_NAME=${RUN_NAME:-armhack_walk_two_goal_${EXPERT}_${STAGE}}
if [[ -z "${MAX_ITERATIONS:-}" ]]; then
    if [[ "${STAGE}" == "robust" ]]; then MAX_ITERATIONS=16
    elif [[ "${EXPERT}" == "yaw" ]]; then MAX_ITERATIONS=80
    else MAX_ITERATIONS=60
    fi
fi
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be positive"

STAGING_RUN="_source_${EXPERT}_${STAGE}"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}
mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_source.pt"

if [[ "${STAGE}" == "robust" ]]; then RANDOMIZATION_STRENGTH=1; else RANDOMIZATION_STRENGTH=0; fi

echo "=================================================="
echo " ArmHack Walk two-goal ${EXPERT} expert"
echo "=================================================="
echo "Host / GPU       : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Task / stage     : ${TASK} / ${STAGE}"
echo "Protected base   : ${BASE_SHA256}"
echo "Source           : ${SOURCE_CHECKPOINT} (${SOURCE_SHA256})"
echo "Arm poses        : random per episode; fixed throughout episode"
echo "Training         : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Output           : ${OUTPUT_DIR}"
echo "=================================================="

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
TASK="${TASK}" NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" \
RUN_NAME="${RUN_NAME}" RESUME=True LOAD_RUN="^${STAGING_RUN}$" CHECKPOINT='^model_source.pt$' \
HEADLESS=True QUIET_TERMINAL=True TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
STYLE_REWARD_SCALE=5.0 TASK_STYLE_LERP=1.0 AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=False \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    agent.checkpoint_output_dir="${OUTPUT_DIR}" \
    agent.load_policy_only=True \
    agent.reset_iteration_on_policy_only_load=True \
    agent.freeze_base_actor=False \
    agent.freeze_actor_hidden_layers=1 \
    agent.actor_warmup_iterations=0 \
    agent.algorithm.baseline_kl_cfg.enabled=False \
    agent.algorithm.amp_cfg.freeze_discriminator=True \
    "$@"

grep -q 'Learning iteration' "${TRAIN_LOG_FILE}" || die "training produced no PPO iteration"
if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|nan detected|TypeError:' "${TRAIN_LOG_FILE}"; then
    die "training log contains a fatal Python/CUDA/numerical error"
fi
verify_file "${BASE_CHECKPOINT}" "${BASE_SIZE}" "${BASE_SHA256}" || die "protected model_10990 changed"
verify_file "${SOURCE_CHECKPOINT}" "${SOURCE_SIZE}" "${SOURCE_SHA256}" || die "expert source changed"
echo "Training completed; protected source hashes unchanged."
