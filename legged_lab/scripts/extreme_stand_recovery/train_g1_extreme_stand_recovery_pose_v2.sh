#!/usr/bin/env bash
# Continue final model_4999 with stronger generalized/default Cartesian pose recovery
# and a symmetric default foot-distance objective.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-19_14-34-18_g1_extreme_stand_recovery_full_20260719_1433/model_4999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-16af8b298fe4789194b6f798ee5591a3cc61edab307724a82906cc5e9a038fe7}
VERIFY_BASE_SHA256=${VERIFY_BASE_SHA256:-True}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_pose_v2_from_model4999_20260720}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-3.0e-5}

DEFAULT_JOINT_POSE_WEIGHT=${DEFAULT_JOINT_POSE_WEIGHT:-5.0}
DEFAULT_JOINT_POSE_STD=${DEFAULT_JOINT_POSE_STD:-0.25}
DEFAULT_LEG_POSE_WEIGHT=${DEFAULT_LEG_POSE_WEIGHT:-3.0}
DEFAULT_LEG_POSE_STD=${DEFAULT_LEG_POSE_STD:-0.18}
DEFAULT_CARTESIAN_POSE_WEIGHT=${DEFAULT_CARTESIAN_POSE_WEIGHT:-2.5}
DEFAULT_CARTESIAN_POSE_STD=${DEFAULT_CARTESIAN_POSE_STD:-0.12}
DEFAULT_FEET_DISTANCE_PENALTY=${DEFAULT_FEET_DISTANCE_PENALTY:-8.0}

validate_positive() {
    local name=$1 value=$2
    awk -v value="${value}" 'BEGIN { exit !(value > 0.0) }' || {
        echo "Error: ${name} must be positive, got ${value}." >&2
        exit 1
    }
}

for item in \
    "DEFAULT_JOINT_POSE_WEIGHT:${DEFAULT_JOINT_POSE_WEIGHT}" \
    "DEFAULT_JOINT_POSE_STD:${DEFAULT_JOINT_POSE_STD}" \
    "DEFAULT_LEG_POSE_WEIGHT:${DEFAULT_LEG_POSE_WEIGHT}" \
    "DEFAULT_LEG_POSE_STD:${DEFAULT_LEG_POSE_STD}" \
    "DEFAULT_CARTESIAN_POSE_WEIGHT:${DEFAULT_CARTESIAN_POSE_WEIGHT}" \
    "DEFAULT_CARTESIAN_POSE_STD:${DEFAULT_CARTESIAN_POSE_STD}" \
    "DEFAULT_FEET_DISTANCE_PENALTY:${DEFAULT_FEET_DISTANCE_PENALTY}"; do
    validate_positive "${item%%:*}" "${item#*:}"
done

echo "============================================================"
echo " Extreme Stand Recovery Pose V2 continuation"
echo "============================================================"
echo "Base checkpoint       : ${BASE_CHECKPOINT}"
echo "Full joint pose       : +${DEFAULT_JOINT_POSE_WEIGHT}, std=${DEFAULT_JOINT_POSE_STD} rad"
echo "Leg joint pose        : +${DEFAULT_LEG_POSE_WEIGHT}, std=${DEFAULT_LEG_POSE_STD} rad"
echo "Cartesian key bodies  : +${DEFAULT_CARTESIAN_POSE_WEIGHT}, std=${DEFAULT_CARTESIAN_POSE_STD} m"
echo "Default foot distance : -${DEFAULT_FEET_DISTANCE_PENALTY} * squared_error_m2"
echo "Learning rate         : ${LEARNING_RATE}"
echo "Training              : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "Run name              : ${RUN_NAME}"
echo "============================================================"

BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256}" \
VERIFY_BASE_SHA256="${VERIFY_BASE_SHA256}" \
RUN_NAME="${RUN_NAME}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
NUM_ENVS="${NUM_ENVS}" \
LEARNING_RATE="${LEARNING_RATE}" \
bash "${SCRIPT_DIR}/train_g1_extreme_stand_recovery.sh" \
  "env.rewards.default_joint_pose_exp.weight=${DEFAULT_JOINT_POSE_WEIGHT}" \
  "env.rewards.default_joint_pose_exp.params.std=${DEFAULT_JOINT_POSE_STD}" \
  "env.rewards.default_leg_joint_pose_exp.weight=${DEFAULT_LEG_POSE_WEIGHT}" \
  "env.rewards.default_leg_joint_pose_exp.params.std=${DEFAULT_LEG_POSE_STD}" \
  "env.rewards.default_key_body_pose_exp.weight=${DEFAULT_CARTESIAN_POSE_WEIGHT}" \
  "env.rewards.default_key_body_pose_exp.params.std=${DEFAULT_CARTESIAN_POSE_STD}" \
  "env.rewards.default_feet_distance_l2.weight=-${DEFAULT_FEET_DISTANCE_PENALTY}" \
  "$@"
