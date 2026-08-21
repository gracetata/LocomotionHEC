#!/usr/bin/env bash
# Fine-tune one actor of the ArmHack gated Walk policy on the 30-cm ankle kernel.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

BRANCH=${BRANCH:-base}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:?set SOURCE_CHECKPOINT to one split actor checkpoint}
SOURCE_SHA256=${SOURCE_SHA256:?set SOURCE_SHA256 to the exact source hash}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-160}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-armhack_walk_ankle30_${BRANCH}}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
KL_SCALE=${KL_SCALE:-1.00}
RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-1}
export PATH="${HOME}/.local/bin:${PATH}"

die() { echo "Error: $*" >&2; exit 1; }
[[ "$(hostname)" == "tata-futurelab" ]] || die "training is locked to future5090"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 5090' || die "RTX 5090 not detected"
case "${BRANCH}" in base|lateral|yaw) ;; *) die "BRANCH must be base, lateral, or yaw" ;; esac
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "source checkpoint missing: ${SOURCE_CHECKPOINT}"
SOURCE_CHECKPOINT=$(realpath "${SOURCE_CHECKPOINT}")
[[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] \
    || die "source checkpoint SHA-256 mismatch"
if pgrep -af 'train.py --task .*Stand' >/dev/null; then
    die "Stand training is active; refusing to start Walk training"
fi
if pgrep -af 'train.py --task .*ArmHackWalkAnkleSpacing' >/dev/null; then
    die "another ankle-spacing Walk training is active"
fi

case "${BRANCH}" in
    base) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkAnkleSpacingBase-v0 ;;
    lateral) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkAnkleSpacingLateral-v0 ;;
    yaw) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkAnkleSpacingYaw-v0 ;;
esac
TASK=${TASK_OVERRIDE:-${TASK}}
EXPERIMENT_NAME="g1_armhack_walk_ankle_spacing_${BRANCH}"
OUTPUT_DIR="${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkAnkleSpacingFinetune/${BRANCH}"
STAGING_RUN="_source_${BRANCH}_${SOURCE_SHA256:0:12}"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${STAGING_RUN}"
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/train_${RUN_NAME}.log"}
mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_source.pt"

echo "ArmHack ankle-spacing branch=${BRANCH} target=0.30m sigma=0.06m weight=500"
echo "source=${SOURCE_CHECKPOINT} sha256=${SOURCE_SHA256}"
echo "training=${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
TASK="${TASK}" NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" \
RUN_NAME="${RUN_NAME}" RESUME=True LOAD_RUN="^${STAGING_RUN}$" CHECKPOINT='^model_source.pt$' \
HEADLESS=True QUIET_TERMINAL=True TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" STYLE_REWARD_SCALE=5.0 TASK_STYLE_LERP=1.0 \
AMP_GRAD_PENALTY_SCALE=20.0 BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${SOURCE_CHECKPOINT}" BASELINE_KL_SCALE="${KL_SCALE}" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    agent.checkpoint_output_dir="${OUTPUT_DIR}" \
    agent.load_policy_only=True \
    agent.reset_iteration_on_policy_only_load=True \
    agent.freeze_base_actor=False \
    agent.freeze_actor_hidden_layers=0 \
    agent.actor_warmup_iterations=0 \
    agent.restore_configured_learning_rate_on_load=True \
    agent.algorithm.baseline_kl_cfg.enabled=True \
    agent.algorithm.baseline_kl_cfg.checkpoint_path="${SOURCE_CHECKPOINT}" \
    agent.algorithm.baseline_kl_cfg.scale="${KL_SCALE}" \
    agent.algorithm.baseline_kl_cfg.hard_limit=0.25 \
    agent.algorithm.amp_cfg.freeze_discriminator=True \
    "$@"

grep -q 'Learning iteration' "${TRAIN_LOG_FILE}" || die "no PPO iteration in training log"
if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|nan detected|TypeError:' "${TRAIN_LOG_FILE}"; then
    die "fatal error detected in training log"
fi
[[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] \
    || die "source checkpoint changed during training"
echo "Training completed: ${OUTPUT_DIR}"
