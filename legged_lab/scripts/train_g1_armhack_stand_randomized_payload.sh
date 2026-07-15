#!/usr/bin/env bash
# Continue ArmHack Stand training from the verified 1.0x model_2999 policy.
#
# New distribution:
#   1. Random arm-only poses from the measured-range pose bank.
#   2. Smooth minimum-jerk interpolation between random poses.
#   3. Independent 0..PAYLOAD_MAX_KG added mass on left/right wrist-yaw links.
#
# The policy receives current robot observations only.  The next pose, segment
# duration, interpolation phase, and randomized payload are not observations.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-14_20-34-20_armhack_stand_curriculum_1x_from_model9996_full_20260714/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"2c87cc2cc3706c1024594d14d85a34e7bf468b54f6b66e49b6155ef72a2dbd16"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-14825781}
POSE_BANK=${POSE_BANK:-"${PROJECT_DIR}/Reference Data/ArmHack/StandPerturb/RandomizedTraining/random_arm_pose_bank_seed20260715.json"}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
STATIC_ITERATIONS=${STATIC_ITERATIONS:-500}
RAMP_ITERATIONS=${RAMP_ITERATIONS:-1000}
FINAL_MOTION_SCALE=${FINAL_MOTION_SCALE:-1.0}
TRANSITION_MIN_S=${TRANSITION_MIN_S:-2.0}
TRANSITION_MAX_S=${TRANSITION_MAX_S:-6.0}
PAYLOAD_MAX_KG=${PAYLOAD_MAX_KG:-1.0}
RUN_NAME=${RUN_NAME:-armhack_stand_randomized_poses_payload_from_model2999}
SEED=${SEED:-42}

HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.003}
ENTROPY_COEF=${ENTROPY_COEF:-0.002}

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Error: model_2999 checkpoint not found: ${BASE_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${POSE_BANK}" ]]; then
    echo "Error: randomized arm-pose bank not found: ${POSE_BANK}" >&2
    echo "Build it with: python scripts/tools/build_armhack_stand_randomized_training_data.py" >&2
    exit 1
fi

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
POSE_BANK=$(realpath "${POSE_BANK}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
if [[ "${ACTUAL_BASE_SHA256}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "Error: model_2999 SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SHA256}" >&2
    echo "Actual:   ${ACTUAL_BASE_SHA256}" >&2
    exit 1
fi
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
if [[ "${ACTUAL_BASE_SIZE}" != "${EXPECTED_BASE_SIZE}" ]]; then
    echo "Error: model_2999 size mismatch. Expected ${EXPECTED_BASE_SIZE}, got ${ACTUAL_BASE_SIZE}." >&2
    exit 1
fi

if (( STATIC_ITERATIONS < 0 || RAMP_ITERATIONS < 0 || MAX_ITERATIONS <= 0 )); then
    echo "Error: iteration counts must be non-negative and MAX_ITERATIONS must be positive." >&2
    exit 1
fi
if ! awk -v value="${FINAL_MOTION_SCALE}" 'BEGIN { exit !(value >= 0.0 && value <= 1.0) }'; then
    echo "Error: FINAL_MOTION_SCALE must be within [0, 1]." >&2
    exit 1
fi
if ! awk -v lo="${TRANSITION_MIN_S}" -v hi="${TRANSITION_MAX_S}" \
    'BEGIN { exit !(lo > 0.0 && hi >= lo) }'; then
    echo "Error: transition duration must satisfy 0 < TRANSITION_MIN_S <= TRANSITION_MAX_S." >&2
    exit 1
fi
if ! awk -v value="${PAYLOAD_MAX_KG}" 'BEGIN { exit !(value >= 0.0 && value <= 3.0) }'; then
    echo "Error: PAYLOAD_MAX_KG must be within [0, 3]." >&2
    exit 1
fi

STEPS_PER_ITERATION=24
STATIC_STEPS=$((STATIC_ITERATIONS * STEPS_PER_ITERATION))
RAMP_STEPS=$((RAMP_ITERATIONS * STEPS_PER_ITERATION))

# The generic loader resolves checkpoints inside the experiment log tree.  A
# symlink preserves the external model and avoids copying it into the repo.
BASE_RUN_NAME="_armhack_stand_randomized_payload_base_model2999"
BASE_RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_2999.pt"

echo "============================================================"
echo " ArmHack Stand randomized-pose + payload continuation"
echo "============================================================"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Base SHA-256    : ${ACTUAL_BASE_SHA256}"
echo "Pose bank       : ${POSE_BANK}"
echo "Static stage    : ${STATIC_ITERATIONS} iterations (${STATIC_STEPS} steps)"
echo "Motion ramp     : ${RAMP_ITERATIONS} iterations (${RAMP_STEPS} steps)"
echo "Final speed     : ${FINAL_MOTION_SCALE}"
echo "Transition      : nominal ${TRANSITION_MIN_S}..${TRANSITION_MAX_S} s; extended when velocity-safe"
echo "End payload     : independent 0..${PAYLOAD_MAX_KG} kg per wrist-yaw link"
echo "Total training  : ${MAX_ITERATIONS} iterations, ${NUM_ENVS} envs"
echo "Baseline KL     : ${BASELINE_KL_SCALE} against input model_2999"
echo "Run name        : ${RUN_NAME}"
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
RANDOMIZATION_STRENGTH=0 \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
ENTROPY_COEF="${ENTROPY_COEF}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
  "env.upper_body_perturbation.random_pose_bank_path='${POSE_BANK}'" \
  env.upper_body_perturbation.random_curriculum_enabled=True \
  env.upper_body_perturbation.random_curriculum_static_steps="${STATIC_STEPS}" \
  env.upper_body_perturbation.random_curriculum_ramp_steps="${RAMP_STEPS}" \
  env.upper_body_perturbation.random_curriculum_motion_scale="${FINAL_MOTION_SCALE}" \
  "env.upper_body_perturbation.random_transition_duration_range_s=[${TRANSITION_MIN_S},${TRANSITION_MAX_S}]" \
  "env.events.randomize_end_effector_payload.params.mass_distribution_params=[0.0,${PAYLOAD_MAX_KG}]" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
