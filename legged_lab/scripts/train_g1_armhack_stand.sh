#!/usr/bin/env bash
# Fine-tune the ArmHack static-balance policy from the verified S3 G1 model_9996 checkpoint.
#
# Curriculum (24 control steps per PPO iteration):
#   1. Hold a random CSV arm pose for STATIC_ITERATIONS.
#   2. Ramp continuously from zero speed to SLOW_MOTION_SCALE over RAMP_ITERATIONS.
#   3. Continue at SLOW_MOTION_SCALE for the remaining iterations.
#
# The policy observes current robot state only. No trajectory phase, future arm
# target, or look-ahead is added to the observation.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-StandPerturb-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_DIR}/ArmHack Checkpoints/StandPerturb/BaselineModel9996/model_9996.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-16202421}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
STATIC_ITERATIONS=${STATIC_ITERATIONS:-500}
RAMP_ITERATIONS=${RAMP_ITERATIONS:-1000}
SLOW_MOTION_SCALE=${SLOW_MOTION_SCALE:-0.25}
RUN_NAME=${RUN_NAME:-armhack_stand_curriculum_from_model9996}
SEED=${SEED:-42}

HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}
RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-0}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.003}
ENTROPY_COEF=${ENTROPY_COEF:-0.002}

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Error: model_9996 checkpoint not found: ${BASE_CHECKPOINT}" >&2
    echo "Copy the verified S3 G1 checkpoint to that path or set BASE_CHECKPOINT." >&2
    exit 1
fi

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
if [[ "${ACTUAL_BASE_SHA256}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "Error: model_9996 SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SHA256}" >&2
    echo "Actual:   ${ACTUAL_BASE_SHA256}" >&2
    exit 1
fi
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
if [[ "${ACTUAL_BASE_SIZE}" != "${EXPECTED_BASE_SIZE}" ]]; then
    echo "Error: model_9996 size mismatch. Expected ${EXPECTED_BASE_SIZE}, got ${ACTUAL_BASE_SIZE}." >&2
    exit 1
fi

if (( STATIC_ITERATIONS < 0 || RAMP_ITERATIONS < 0 || MAX_ITERATIONS <= 0 )); then
    echo "Error: iteration counts must be non-negative and MAX_ITERATIONS must be positive." >&2
    exit 1
fi

if ! awk -v value="${SLOW_MOTION_SCALE}" 'BEGIN { exit !(value >= 0.0 && value <= 1.0) }'; then
    echo "Error: SLOW_MOTION_SCALE must be within [0, 1]." >&2
    exit 1
fi

STEPS_PER_ITERATION=24
STATIC_STEPS=$((STATIC_ITERATIONS * STEPS_PER_ITERATION))
RAMP_STEPS=$((RAMP_ITERATIONS * STEPS_PER_ITERATION))

# IsaacLab resolves resume checkpoints under the selected experiment log root.
# Stage the external baseline as a symlink so the generic loader can resolve it
# without duplicating the 15 MB checkpoint.
BASE_RUN_NAME="_armhack_stand_baseline_model9996"
BASE_RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_9996.pt"

echo "=============================================="
echo " ArmHack Stand curriculum fine-tune"
echo "=============================================="
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Base SHA-256    : ${ACTUAL_BASE_SHA256}"
echo "Base iteration  : 9996 (actor 96 -> 29, critic input 297)"
echo "Static stage    : ${STATIC_ITERATIONS} iterations (${STATIC_STEPS} steps)"
echo "Slow ramp       : ${RAMP_ITERATIONS} iterations (${RAMP_STEPS} steps)"
echo "Final arm speed : ${SLOW_MOTION_SCALE} x source trajectory"
echo "Total training  : ${MAX_ITERATIONS} iterations, ${NUM_ENVS} envs"
echo "Randomization   : ${RANDOMIZATION_STRENGTH} (0 recommended for this curriculum)"
echo "Baseline KL     : ${BASELINE_KL_SCALE}"
echo "Run name        : ${RUN_NAME}"
echo "=============================================="

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${BASE_RUN_NAME}$" \
CHECKPOINT="^model_9996.pt$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
ENTROPY_COEF="${ENTROPY_COEF}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
  env.upper_body_perturbation.csv_curriculum_enabled=True \
  env.upper_body_perturbation.csv_curriculum_static_steps="${STATIC_STEPS}" \
  env.upper_body_perturbation.csv_curriculum_ramp_steps="${RAMP_STEPS}" \
  env.upper_body_perturbation.csv_curriculum_motion_scale="${SLOW_MOTION_SCALE}" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
