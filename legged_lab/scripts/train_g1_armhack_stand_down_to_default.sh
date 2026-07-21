#!/usr/bin/env bash
# Continue the latest robust Stand policy on the real deployment startup motion:
# upright + natural-down arms -> simultaneous minimum-jerk lift -> flat P0 arms.
#
# Every reset writes the exact AD arm pose into simulation.  A random 2..5 s
# natural-down hold and random 3..9 s lift duration are sampled independently
# per environment.  Future pose, delay, duration, and phase are not observed by
# the policy.  Walk tasks and files are not referenced by this launcher.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_DIR=$(cd "${PROJECT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-StandDownToDefault-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${REPO_DIR}/checkpoint/stand/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-14825781}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-2000}
RUN_NAME=${RUN_NAME:-armhack_stand_down_to_flat_default_from_robust_model2999}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}

START_DELAY_MIN_S=${START_DELAY_MIN_S:-2.0}
START_DELAY_MAX_S=${START_DELAY_MAX_S:-5.0}
LIFT_DURATION_MIN_S=${LIFT_DURATION_MIN_S:-3.0}
LIFT_DURATION_MAX_S=${LIFT_DURATION_MAX_S:-9.0}
STATIC_STEPS=${STATIC_STEPS:-12000}
RAMP_STEPS=${RAMP_STEPS:-12000}
FINAL_MOTION_SCALE=${FINAL_MOTION_SCALE:-1.0}

LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
DESIRED_KL=${DESIRED_KL:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.002}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.001}

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Error: latest robust Stand checkpoint not found: ${BASE_CHECKPOINT}" >&2
    exit 1
fi
BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
if [[ "${ACTUAL_BASE_SHA256}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "Error: Stand base checkpoint SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SHA256}" >&2
    echo "Actual:   ${ACTUAL_BASE_SHA256}" >&2
    exit 1
fi
if [[ "${ACTUAL_BASE_SIZE}" != "${EXPECTED_BASE_SIZE}" ]]; then
    echo "Error: Stand base checkpoint size mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SIZE}; actual: ${ACTUAL_BASE_SIZE}" >&2
    exit 1
fi
if (( NUM_ENVS <= 0 || MAX_ITERATIONS <= 0 || STATIC_STEPS < 0 || RAMP_STEPS < 0 )); then
    echo "Error: NUM_ENVS/MAX_ITERATIONS must be positive and curriculum steps non-negative." >&2
    exit 1
fi

validate_ordered_range() {
    local label=$1
    local lower=$2
    local upper=$3
    local minimum=$4
    local maximum=$5
    if ! awk -v lo="${lower}" -v hi="${upper}" -v min="${minimum}" -v max="${maximum}" \
        'BEGIN { exit !(lo >= min && hi >= lo && hi <= max) }'; then
        echo "Error: ${label} must satisfy ${minimum} <= min <= max <= ${maximum}; got ${lower}..${upper}." >&2
        exit 1
    fi
}

validate_ordered_range "natural-down start delay" "${START_DELAY_MIN_S}" "${START_DELAY_MAX_S}" 0.0 20.0
validate_ordered_range "arm lift duration" "${LIFT_DURATION_MIN_S}" "${LIFT_DURATION_MAX_S}" 0.1 20.0
if ! awk -v value="${FINAL_MOTION_SCALE}" 'BEGIN { exit !(value >= 0.0 && value <= 1.0) }'; then
    echo "Error: FINAL_MOTION_SCALE must be within [0, 1]." >&2
    exit 1
fi

# The checkpoint resolver reads from the experiment log namespace.  Keep the
# source checkpoint immutable and expose it through a deterministic symlink.
BASE_RUN_NAME="_armhack_stand_down_to_default_base_model2999_146aca1f547c"
BASE_RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_2999.pt"

echo "============================================================"
echo " ArmHack Stand natural-down -> flat-default continuation"
echo "============================================================"
echo "Task             : ${TASK} (Stand only)"
echo "Base checkpoint  : ${BASE_CHECKPOINT}"
echo "Base SHA-256     : ${ACTUAL_BASE_SHA256}"
echo "Reset arm pose   : AD_natural_down from shared stand_arm_presets.json"
echo "Goal arm pose    : P0_symmetric_reference from the same shared file"
echo "Start hold       : U(${START_DELAY_MIN_S}, ${START_DELAY_MAX_S}) s per env/reset"
echo "Lift duration    : U(${LIFT_DURATION_MIN_S}, ${LIFT_DURATION_MAX_S}) s per env/reset"
echo "Arm interpolation: one shared minimum-jerk phase for both arms"
echo "Curriculum       : ${STATIC_STEPS} static steps -> ${RAMP_STEPS} ramp steps -> ${FINAL_MOTION_SCALE}x"
echo "Observation      : no future target/delay/duration/phase"
echo "Robustness       : inherits payload, torso wrench, actuator and joint DR"
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
CHECKPOINT="^model_2999.pt$" \
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
  env.upper_body_perturbation.source=pose_transition \
  env.upper_body_perturbation.pose_transition_initialize_joint_state_on_reset=True \
  "env.upper_body_perturbation.pose_transition_start_delay_range_s=[${START_DELAY_MIN_S},${START_DELAY_MAX_S}]" \
  "env.upper_body_perturbation.pose_transition_duration_range_s=[${LIFT_DURATION_MIN_S},${LIFT_DURATION_MAX_S}]" \
  env.upper_body_perturbation.pose_transition_curriculum_enabled=True \
  env.upper_body_perturbation.pose_transition_curriculum_static_steps="${STATIC_STEPS}" \
  env.upper_body_perturbation.pose_transition_curriculum_ramp_steps="${RAMP_STEPS}" \
  env.upper_body_perturbation.pose_transition_curriculum_motion_scale="${FINAL_MOTION_SCALE}" \
  agent.algorithm.learning_rate="${LEARNING_RATE}" \
  agent.algorithm.desired_kl="${DESIRED_KL}" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
