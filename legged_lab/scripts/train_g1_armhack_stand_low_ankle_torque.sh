#!/usr/bin/env bash
# Continue the completed natural-down -> P0 Stand policy while explicitly
# reducing unnecessary applied torque at left/right ankle pitch and roll.
#
# This launcher is Stand-only.  It keeps the learned full-speed arm transition,
# wrist payload, torso wrench, actuator-gain, friction, and armature
# randomization inherited by StandDownToDefault.  The actor still observes no
# future arm target, transition duration, payload, wrench, or DR parameters.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_DIR=$(cd "${PROJECT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-StandDownToDefault-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-21_13-53-24_armhack_stand_down_to_flat_default_from_robust_model2999_full_20260721/model_1999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"9a54ac40985a56b13ba0327bfd9f1db8a6e65a59dfcf3e866041b85f344066bb"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-14825781}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-1000}
RUN_NAME=${RUN_NAME:-armhack_stand_low_ankle_roll_pitch_torque_wide_stance_from_model1999}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}

# This is an additional soft penalty on the four selected ankle joints. The
# inherited 15-DoF lower-body torque penalty remains -2e-6.  Survival, support,
# posture, anti-slide, and the robust -500 fall penalty remain active; necessary
# ankle effort is therefore always preferable to losing balance.
ANKLE_TORQUE_PENALTY=${ANKLE_TORQUE_PENALTY:-5.0e-5}
LEARNING_RATE=${LEARNING_RATE:-3.0e-5}
DESIRED_KL=${DESIRED_KL:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.001}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.001}

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Error: completed Stand model_1999 not found: ${BASE_CHECKPOINT}" >&2
    exit 1
fi
BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
if [[ "${ACTUAL_BASE_SHA256}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "Error: Stand model_1999 SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SHA256}" >&2
    echo "Actual:   ${ACTUAL_BASE_SHA256}" >&2
    exit 1
fi
if [[ "${ACTUAL_BASE_SIZE}" != "${EXPECTED_BASE_SIZE}" ]]; then
    echo "Error: expected ${EXPECTED_BASE_SIZE} bytes; got ${ACTUAL_BASE_SIZE}." >&2
    exit 1
fi
if (( NUM_ENVS <= 0 || MAX_ITERATIONS <= 0 )); then
    echo "Error: NUM_ENVS and MAX_ITERATIONS must be positive integers." >&2
    exit 1
fi
if ! awk -v value="${ANKLE_TORQUE_PENALTY}" \
    'BEGIN { exit !(value >= 1.0e-7 && value <= 1.0e-3) }'; then
    echo "Error: ANKLE_TORQUE_PENALTY must be within [1e-7, 1e-3]." >&2
    exit 1
fi

BASE_RUN_NAME="_armhack_stand_low_ankle_torque_base_model1999_9a54ac40985a"
BASE_RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_1999.pt"

echo "============================================================"
echo " ArmHack Stand low ankle roll/pitch torque + wide stance"
echo "============================================================"
echo "Task             : ${TASK} (Stand only)"
echo "Base checkpoint  : ${BASE_CHECKPOINT}"
echo "Base SHA-256     : ${ACTUAL_BASE_SHA256}"
echo "Selected joints  : left/right ankle pitch + left/right ankle roll"
echo "Torque source    : Isaac applied_torque (Nm), not raw actor output"
echo "Ankle penalty    : -${ANKLE_TORQUE_PENALTY} * sum(tau_ankle^2), soft constraint"
echo "Default stance   : hip roll +/-0.06 rad, ankle counter-roll -/+0.06 rad"
echo "Foot separation  : target 0.30 m (slightly wider than ~0.281 m shoulders)"
echo "Arm motion       : full AD -> P0 minimum-jerk, random delay/speed"
echo "Prior protection : baseline KL ${BASELINE_KL_SCALE}, lr ${LEARNING_RATE}"
echo "Training         : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Run name         : ${RUN_NAME}"
echo "============================================================"

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${BASE_RUN_NAME}$" \
CHECKPOINT="^model_1999.pt$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=1 \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
ENTROPY_COEF="${ENTROPY_COEF}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
  env.upper_body_perturbation.pose_transition_curriculum_enabled=False \
  env.upper_body_perturbation.pose_transition_curriculum_motion_scale=1.0 \
  env.rewards.ankle_roll_pitch_torques_l2.weight="-${ANKLE_TORQUE_PENALTY}" \
  agent.algorithm.learning_rate="${LEARNING_RATE}" \
  agent.algorithm.desired_kl="${DESIRED_KL}" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
