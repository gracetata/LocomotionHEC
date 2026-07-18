#!/usr/bin/env bash
# Continue ArmHack Stand from the verified randomized-payload model_2999.
#
# Stand-only robustness distribution:
#   1. Full-speed (1.0x) random 14-DoF arm-pose interpolation from reset.
#   2. Independent 0..PAYLOAD_MAX_KG mass on each wrist-yaw link.
#   3. Resampled torso external force/torque while the arms keep moving.
#   4. Per-environment actuator gain, joint friction, and armature scaling.
#   5. Increased one-step non-timeout termination penalty.
#
# The policy observation remains unchanged: none of the future arm target,
# payload, wrench, or randomized joint parameters is exposed to the actor.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-StandRobust-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"877e929d516cffe9131cc235477ceef4b226ec69e41c0f1c23e48816cfa28821"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-14825781}
POSE_BANK=${POSE_BANK:-"${PROJECT_DIR}/Reference Data/ArmHack/StandPerturb/RandomizedTraining/random_arm_pose_bank_seed20260715.json"}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
RUN_NAME=${RUN_NAME:-armhack_stand_robust_wrench_joint_dr_from_model2999}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}

TRANSITION_MIN_S=${TRANSITION_MIN_S:-2.0}
TRANSITION_MAX_S=${TRANSITION_MAX_S:-6.0}
PAYLOAD_MAX_KG=${PAYLOAD_MAX_KG:-1.0}
TERMINATION_PENALTY_MAG=${TERMINATION_PENALTY_MAG:-500.0}
EXTERNAL_FORCE_MAX_N=${EXTERNAL_FORCE_MAX_N:-20.0}
EXTERNAL_TORQUE_MAX_NM=${EXTERNAL_TORQUE_MAX_NM:-3.0}
FORCE_INTERVAL_MIN_S=${FORCE_INTERVAL_MIN_S:-2.0}
FORCE_INTERVAL_MAX_S=${FORCE_INTERVAL_MAX_S:-5.0}
ACTUATOR_GAIN_MIN_SCALE=${ACTUATOR_GAIN_MIN_SCALE:-0.90}
ACTUATOR_GAIN_MAX_SCALE=${ACTUATOR_GAIN_MAX_SCALE:-1.10}
JOINT_FRICTION_MIN_SCALE=${JOINT_FRICTION_MIN_SCALE:-0.80}
JOINT_FRICTION_MAX_SCALE=${JOINT_FRICTION_MAX_SCALE:-1.20}
JOINT_ARMATURE_MIN_SCALE=${JOINT_ARMATURE_MIN_SCALE:-0.90}
JOINT_ARMATURE_MAX_SCALE=${JOINT_ARMATURE_MAX_SCALE:-1.10}

LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
DESIRED_KL=${DESIRED_KL:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.002}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.001}

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Error: robust Stand base checkpoint not found: ${BASE_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${POSE_BANK}" ]]; then
    echo "Error: randomized arm-pose bank not found: ${POSE_BANK}" >&2
    exit 1
fi

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
POSE_BANK=$(realpath "${POSE_BANK}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
if [[ "${ACTUAL_BASE_SHA256}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "Error: robust Stand model_2999 SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SHA256}" >&2
    echo "Actual:   ${ACTUAL_BASE_SHA256}" >&2
    exit 1
fi
if [[ "${ACTUAL_BASE_SIZE}" != "${EXPECTED_BASE_SIZE}" ]]; then
    echo "Error: robust Stand model_2999 size mismatch." >&2
    echo "Expected: ${EXPECTED_BASE_SIZE}; actual: ${ACTUAL_BASE_SIZE}" >&2
    exit 1
fi

if (( NUM_ENVS <= 0 || MAX_ITERATIONS <= 0 )); then
    echo "Error: NUM_ENVS and MAX_ITERATIONS must be positive integers." >&2
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

validate_ordered_range "transition duration" "${TRANSITION_MIN_S}" "${TRANSITION_MAX_S}" 0.1 20.0
validate_ordered_range "force interval" "${FORCE_INTERVAL_MIN_S}" "${FORCE_INTERVAL_MAX_S}" 0.02 20.0
validate_ordered_range "actuator gain scale" "${ACTUATOR_GAIN_MIN_SCALE}" "${ACTUATOR_GAIN_MAX_SCALE}" 0.5 1.5
validate_ordered_range "joint friction scale" "${JOINT_FRICTION_MIN_SCALE}" "${JOINT_FRICTION_MAX_SCALE}" 0.5 1.5
validate_ordered_range "joint armature scale" "${JOINT_ARMATURE_MIN_SCALE}" "${JOINT_ARMATURE_MAX_SCALE}" 0.5 1.5

if ! awk -v value="${PAYLOAD_MAX_KG}" 'BEGIN { exit !(value >= 0.0 && value <= 3.0) }'; then
    echo "Error: PAYLOAD_MAX_KG must be within [0, 3]." >&2
    exit 1
fi
if ! awk -v value="${TERMINATION_PENALTY_MAG}" 'BEGIN { exit !(value >= 200.0 && value <= 2000.0) }'; then
    echo "Error: TERMINATION_PENALTY_MAG must be within [200, 2000]." >&2
    exit 1
fi
if ! awk -v value="${EXTERNAL_FORCE_MAX_N}" 'BEGIN { exit !(value >= 0.0 && value <= 100.0) }'; then
    echo "Error: EXTERNAL_FORCE_MAX_N must be within [0, 100]." >&2
    exit 1
fi
if ! awk -v value="${EXTERNAL_TORQUE_MAX_NM}" 'BEGIN { exit !(value >= 0.0 && value <= 20.0) }'; then
    echo "Error: EXTERNAL_TORQUE_MAX_NM must be within [0, 20]." >&2
    exit 1
fi

# The generic checkpoint resolver loads from the experiment log namespace.
# Use a read-only symlink rather than copying or modifying the input model.
BASE_RUN_NAME="_armhack_stand_robust_base_model2999_877e929d516c"
BASE_RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_2999.pt"

echo "============================================================"
echo " ArmHack Stand robust sim-to-real continuation"
echo "============================================================"
echo "Task             : ${TASK} (Stand only)"
echo "Base checkpoint  : ${BASE_CHECKPOINT}"
echo "Base SHA-256     : ${ACTUAL_BASE_SHA256}"
echo "Arm distribution : random pose bank, continuous 1.0x, ${TRANSITION_MIN_S}..${TRANSITION_MAX_S} s"
echo "Wrist payload    : independent U(0, ${PAYLOAD_MAX_KG}) kg per side"
echo "Torso wrench     : force U(-${EXTERNAL_FORCE_MAX_N}, ${EXTERNAL_FORCE_MAX_N}) N per axis"
echo "                   torque U(-${EXTERNAL_TORQUE_MAX_NM}, ${EXTERNAL_TORQUE_MAX_NM}) Nm per axis"
echo "Wrench interval  : independent ${FORCE_INTERVAL_MIN_S}..${FORCE_INTERVAL_MAX_S} s per env"
echo "Actuator gains   : stiffness/damping U(${ACTUATOR_GAIN_MIN_SCALE}, ${ACTUATOR_GAIN_MAX_SCALE}) x nominal"
echo "Joint friction   : U(${JOINT_FRICTION_MIN_SCALE}, ${JOINT_FRICTION_MAX_SCALE}) x nominal"
echo "Joint armature   : U(${JOINT_ARMATURE_MIN_SCALE}, ${JOINT_ARMATURE_MAX_SCALE}) x nominal"
echo "Fall penalty     : -${TERMINATION_PENALTY_MAG} (previous stage: -200)"
echo "Optimizer        : lr=${LEARNING_RATE}, desired_kl=${DESIRED_KL}, baseline_kl=${BASELINE_KL_SCALE}"
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
  "env.upper_body_perturbation.random_pose_bank_path='${POSE_BANK}'" \
  env.upper_body_perturbation.random_curriculum_enabled=False \
  env.upper_body_perturbation.random_curriculum_static_steps=0 \
  env.upper_body_perturbation.random_curriculum_ramp_steps=0 \
  env.upper_body_perturbation.random_curriculum_motion_scale=1.0 \
  "env.upper_body_perturbation.random_transition_duration_range_s=[${TRANSITION_MIN_S},${TRANSITION_MAX_S}]" \
  "env.events.randomize_end_effector_payload.params.mass_distribution_params=[0.0,${PAYLOAD_MAX_KG}]" \
  "env.events.random_torso_external_wrench.interval_range_s=[${FORCE_INTERVAL_MIN_S},${FORCE_INTERVAL_MAX_S}]" \
  "env.events.random_torso_external_wrench.params.force_range=[-${EXTERNAL_FORCE_MAX_N},${EXTERNAL_FORCE_MAX_N}]" \
  "env.events.random_torso_external_wrench.params.torque_range=[-${EXTERNAL_TORQUE_MAX_NM},${EXTERNAL_TORQUE_MAX_NM}]" \
  "env.events.scale_actuator_gains.params.stiffness_distribution_params=[${ACTUATOR_GAIN_MIN_SCALE},${ACTUATOR_GAIN_MAX_SCALE}]" \
  "env.events.scale_actuator_gains.params.damping_distribution_params=[${ACTUATOR_GAIN_MIN_SCALE},${ACTUATOR_GAIN_MAX_SCALE}]" \
  "env.events.scale_joint_parameters.params.friction_distribution_params=[${JOINT_FRICTION_MIN_SCALE},${JOINT_FRICTION_MAX_SCALE}]" \
  "env.events.scale_joint_parameters.params.armature_distribution_params=[${JOINT_ARMATURE_MIN_SCALE},${JOINT_ARMATURE_MAX_SCALE}]" \
  env.rewards.termination_penalty.weight="-${TERMINATION_PENALTY_MAG}" \
  agent.algorithm.learning_rate="${LEARNING_RATE}" \
  agent.algorithm.desired_kl="${DESIRED_KL}" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
