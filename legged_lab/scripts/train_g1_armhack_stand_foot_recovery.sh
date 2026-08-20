#!/usr/bin/env bash
# Fine-tune the latest robust ArmHack Stand model for 30 cm stance recovery,
# random pushes, world-frame torso stability, and low ankle torque.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-StandFootRecovery-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-18_15-53-17_armhack_stand_robust_wrench_joint_dr_from_model2999_full_hec5090_20260718/model_2999.pt"}
BASELINE_KL_CHECKPOINT=${BASELINE_KL_CHECKPOINT:-${BASE_CHECKPOINT}}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-14825781}
POSE_BANK=${POSE_BANK:-"${PROJECT_DIR}/Reference Data/ArmHack/StandPerturb/RandomizedTraining/random_arm_pose_bank_seed20260715.json"}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
RUN_NAME=${RUN_NAME:-armhack_stand_foot_recovery_push_low_ankle_torque_from_robust2999}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
AGENT_DEVICE=${AGENT_DEVICE:-${DEVICE}}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/miniconda3/envs/env_leglab/bin/python}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}

STANCE_MIN_M=${STANCE_MIN_M:-0.08}
STANCE_MAX_M=${STANCE_MAX_M:-0.32}
CLOSE_STANCE_MIN_M=${CLOSE_STANCE_MIN_M:-0.08}
CLOSE_STANCE_MAX_M=${CLOSE_STANCE_MAX_M:-0.14}
CLOSE_STANCE_PROB=${CLOSE_STANCE_PROB:-0.60}
POSITION_SCALE_MIN=${POSITION_SCALE_MIN:-0.95}
POSITION_SCALE_MAX=${POSITION_SCALE_MAX:-1.05}
INITIAL_JOINT_VEL_MIN=${INITIAL_JOINT_VEL_MIN:-0.0}
INITIAL_JOINT_VEL_MAX=${INITIAL_JOINT_VEL_MAX:-0.0}
TARGET_STANCE_M=${TARGET_STANCE_M:-0.30}
STANCE_SUCCESS_TOLERANCE_M=${STANCE_SUCCESS_TOLERANCE_M:-0.015}
PUSH_MAX_MPS=${PUSH_MAX_MPS:-0.35}
PUSH_YAW_MAX_RADPS=${PUSH_YAW_MAX_RADPS:-0.45}
PUSH_INTERVAL_MIN_S=${PUSH_INTERVAL_MIN_S:-3.0}
PUSH_INTERVAL_MAX_S=${PUSH_INTERVAL_MAX_S:-7.0}
EXTERNAL_FORCE_MAX_N=${EXTERNAL_FORCE_MAX_N:-20.0}
EXTERNAL_TORQUE_MAX_NM=${EXTERNAL_TORQUE_MAX_NM:-3.0}
ANKLE_TORQUE_WEIGHT=${ANKLE_TORQUE_WEIGHT:--1.0e-3}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
DESIRED_KL=${DESIRED_KL:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.002}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.0005}
CURRICULUM_STEP_OFFSET=${CURRICULUM_STEP_OFFSET:-0}

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi
for required_file in "${BASE_CHECKPOINT}" "${BASELINE_KL_CHECKPOINT}" "${POSE_BANK}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Error: required training input not found: ${required_file}" >&2
        exit 1
    fi
done

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
BASELINE_KL_CHECKPOINT=$(realpath "${BASELINE_KL_CHECKPOINT}")
POSE_BANK=$(realpath "${POSE_BANK}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
if [[ "${ACTUAL_BASE_SHA256}" != "${EXPECTED_BASE_SHA256}" || "${ACTUAL_BASE_SIZE}" != "${EXPECTED_BASE_SIZE}" ]]; then
    echo "Error: latest robust base checkpoint identity mismatch." >&2
    echo "Expected sha/size: ${EXPECTED_BASE_SHA256} / ${EXPECTED_BASE_SIZE}" >&2
    echo "Actual sha/size  : ${ACTUAL_BASE_SHA256} / ${ACTUAL_BASE_SIZE}" >&2
    exit 1
fi

validate_range() {
    local label=$1 lower=$2 upper=$3 minimum=$4 maximum=$5
    if ! awk -v lo="${lower}" -v hi="${upper}" -v min="${minimum}" -v max="${maximum}" \
        'BEGIN { exit !(lo >= min && hi >= lo && hi <= max) }'; then
        echo "Error: ${label} must satisfy ${minimum} <= min <= max <= ${maximum}; got ${lower}..${upper}." >&2
        exit 1
    fi
}

validate_range "stance distance" "${STANCE_MIN_M}" "${STANCE_MAX_M}" 0.05 0.60
validate_range "close stance distance" "${CLOSE_STANCE_MIN_M}" "${CLOSE_STANCE_MAX_M}" "${STANCE_MIN_M}" "${STANCE_MAX_M}"
validate_range "joint position scale" "${POSITION_SCALE_MIN}" "${POSITION_SCALE_MAX}" 0.50 1.50
validate_range "initial joint velocity" "${INITIAL_JOINT_VEL_MIN}" "${INITIAL_JOINT_VEL_MAX}" -5.0 5.0
validate_range "push interval" "${PUSH_INTERVAL_MIN_S}" "${PUSH_INTERVAL_MAX_S}" 0.2 30.0
if ! awk -v value="${CLOSE_STANCE_PROB}" 'BEGIN { exit !(value >= 0.0 && value <= 1.0) }'; then
    echo "Error: CLOSE_STANCE_PROB must be in [0, 1]." >&2
    exit 1
fi
if ! awk -v value="${STANCE_SUCCESS_TOLERANCE_M}" 'BEGIN { exit !(value > 0.0 && value <= 0.10) }'; then
    echo "Error: STANCE_SUCCESS_TOLERANCE_M must be in (0, 0.10]." >&2
    exit 1
fi
if [[ ! "${CURRICULUM_STEP_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "Error: CURRICULUM_STEP_OFFSET must be a non-negative integer." >&2
    exit 1
fi

BASE_RUN_NAME="_armhack_stand_foot_recovery_base_robust2999_146aca1f547c"
BASE_RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_2999.pt"
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${PROJECT_DIR}/logs/monitoring/${RUN_NAME}.log"}

echo "============================================================"
echo " ArmHack Stand 30 cm foot-recovery continuation"
echo "============================================================"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Base SHA-256    : ${ACTUAL_BASE_SHA256}"
echo "Frozen KL actor : ${BASELINE_KL_CHECKPOINT} (scale=${BASELINE_KL_SCALE})"
echo "Initial stance : U(${STANCE_MIN_M}, ${STANCE_MAX_M}) m; close ${CLOSE_STANCE_MIN_M}..${CLOSE_STANCE_MAX_M} with p=${CLOSE_STANCE_PROB}"
echo "Initial joints : position scale ${POSITION_SCALE_MIN}..${POSITION_SCALE_MAX}; velocity ${INITIAL_JOINT_VEL_MIN}..${INITIAL_JOINT_VEL_MAX} rad/s"
echo "Target stance  : ${TARGET_STANCE_M} m ankle-to-ankle, success +/-${STANCE_SUCCESS_TOLERANCE_M} m"
echo "Impulse push   : xy +/-${PUSH_MAX_MPS} m/s, yaw +/-${PUSH_YAW_MAX_RADPS} rad/s every ${PUSH_INTERVAL_MIN_S}..${PUSH_INTERVAL_MAX_S} s"
echo "Torso wrench   : +/-${EXTERNAL_FORCE_MAX_N} N, +/-${EXTERNAL_TORQUE_MAX_NM} Nm"
echo "Ankle torque w : ${ANKLE_TORQUE_WEIGHT}"
echo "Curriculum step: local step + ${CURRICULUM_STEP_OFFSET}"
echo "Training       : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations on ${DEVICE}"
echo "Run/log        : ${RUN_NAME} / ${TRAIN_LOG_FILE}"
echo "============================================================"

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
DEVICE="${DEVICE}" \
AGENT_DEVICE="${AGENT_DEVICE}" \
ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${BASE_RUN_NAME}$" \
CHECKPOINT="^model_2999.pt$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=1 \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
ENTROPY_COEF="${ENTROPY_COEF}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASELINE_KL_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
  "env.upper_body_perturbation.random_pose_bank_path='${POSE_BANK}'" \
  "env.events.reset_robot_joints.params.distance_range=[${STANCE_MIN_M},${STANCE_MAX_M}]" \
  "env.events.reset_robot_joints.params.close_distance_range=[${CLOSE_STANCE_MIN_M},${CLOSE_STANCE_MAX_M}]" \
  env.events.reset_robot_joints.params.close_stance_probability="${CLOSE_STANCE_PROB}" \
  "env.events.reset_robot_joints.params.position_scale_range=[${POSITION_SCALE_MIN},${POSITION_SCALE_MAX}]" \
  "env.events.reset_robot_joints.params.velocity_range=[${INITIAL_JOINT_VEL_MIN},${INITIAL_JOINT_VEL_MAX}]" \
  "env.events.push_robot.interval_range_s=[${PUSH_INTERVAL_MIN_S},${PUSH_INTERVAL_MAX_S}]" \
  "env.events.push_robot.params.velocity_range.x=[-${PUSH_MAX_MPS},${PUSH_MAX_MPS}]" \
  "env.events.push_robot.params.velocity_range.y=[-${PUSH_MAX_MPS},${PUSH_MAX_MPS}]" \
  "env.events.push_robot.params.velocity_range.yaw=[-${PUSH_YAW_MAX_RADPS},${PUSH_YAW_MAX_RADPS}]" \
  "env.events.random_torso_external_wrench.params.force_range=[-${EXTERNAL_FORCE_MAX_N},${EXTERNAL_FORCE_MAX_N}]" \
  "env.events.random_torso_external_wrench.params.torque_range=[-${EXTERNAL_TORQUE_MAX_NM},${EXTERNAL_TORQUE_MAX_NM}]" \
  env.rewards.ankle_distance_l1.params.target_distance="${TARGET_STANCE_M}" \
  env.rewards.ankle_distance_exp.params.target_distance="${TARGET_STANCE_M}" \
  env.rewards.ankle_distance_success.params.target_distance="${TARGET_STANCE_M}" \
  env.rewards.ankle_distance_success.params.tolerance="${STANCE_SUCCESS_TOLERANCE_M}" \
  env.rewards.ankle_torques_l2.params.target_distance="${TARGET_STANCE_M}" \
  env.rewards.torso_xy_position_near_stance_l2.params.target_distance="${TARGET_STANCE_M}" \
  env.rewards.torso_yaw_near_stance_l2.params.target_distance="${TARGET_STANCE_M}" \
  env.rewards.ankle_torques_l2.weight="${ANKLE_TORQUE_WEIGHT}" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_torques_l2=[[0,${ANKLE_TORQUE_WEIGHT}],[12000,${ANKLE_TORQUE_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.wrench_force_abs_schedule=[[0,5.0],[4000,10.0],[12000,${EXTERNAL_FORCE_MAX_N}]]" \
  "env.curriculum.stance_recovery.params.wrench_torque_abs_schedule=[[0,0.75],[4000,1.5],[12000,${EXTERNAL_TORQUE_MAX_NM}]]" \
  "env.curriculum.stance_recovery.params.push_xy_abs_schedule=[[0,0.10],[4000,0.15],[12000,${PUSH_MAX_MPS}]]" \
  "env.curriculum.stance_recovery.params.push_yaw_abs_schedule=[[0,0.12],[4000,0.20],[12000,${PUSH_YAW_MAX_RADPS}]]" \
  env.curriculum.stance_recovery.params.step_offset="${CURRICULUM_STEP_OFFSET}" \
  agent.algorithm.learning_rate="${LEARNING_RATE}" \
  agent.algorithm.desired_kl="${DESIRED_KL}" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
