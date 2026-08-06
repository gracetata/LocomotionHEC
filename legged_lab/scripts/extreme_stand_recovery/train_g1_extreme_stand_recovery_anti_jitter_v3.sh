#!/usr/bin/env bash
# Continue Pose V2 model_2999 with joint-jerk suppression and narrow
# asset-default Cartesian/foot-distance Gaussian rewards.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264}
VERIFY_BASE_SHA256=${VERIFY_BASE_SHA256:-True}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_anti_jitter_v3_from_pose_v2_model2999}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-2.0e-5}

JOINT_JERK_PENALTY=${JOINT_JERK_PENALTY:-1.0e-8}
DEFAULT_CARTESIAN_POSE_WEIGHT=${DEFAULT_CARTESIAN_POSE_WEIGHT:-8.0}
DEFAULT_CARTESIAN_POSE_VARIANCE=${DEFAULT_CARTESIAN_POSE_VARIANCE:-4.0e-4}
DEFAULT_FEET_DISTANCE_PENALTY=${DEFAULT_FEET_DISTANCE_PENALTY:-8.0}
DEFAULT_FEET_GAUSSIAN_WEIGHT=${DEFAULT_FEET_GAUSSIAN_WEIGHT:-3.0}
DEFAULT_FEET_GAUSSIAN_VARIANCE=${DEFAULT_FEET_GAUSSIAN_VARIANCE:-1.0e-4}

validate_positive() {
    local name=$1 value=$2
    awk -v value="${value}" 'BEGIN { exit !(value > 0.0) }' || {
        echo "Error: ${name} must be positive, got ${value}." >&2
        exit 1
    }
}

for item in \
    "JOINT_JERK_PENALTY:${JOINT_JERK_PENALTY}" \
    "DEFAULT_CARTESIAN_POSE_WEIGHT:${DEFAULT_CARTESIAN_POSE_WEIGHT}" \
    "DEFAULT_CARTESIAN_POSE_VARIANCE:${DEFAULT_CARTESIAN_POSE_VARIANCE}" \
    "DEFAULT_FEET_DISTANCE_PENALTY:${DEFAULT_FEET_DISTANCE_PENALTY}" \
    "DEFAULT_FEET_GAUSSIAN_WEIGHT:${DEFAULT_FEET_GAUSSIAN_WEIGHT}" \
    "DEFAULT_FEET_GAUSSIAN_VARIANCE:${DEFAULT_FEET_GAUSSIAN_VARIANCE}"; do
    validate_positive "${item%%:*}" "${item#*:}"
done

echo "============================================================"
echo " Extreme Stand Recovery anti-jitter Pose V3 continuation"
echo "============================================================"
echo "Base checkpoint       : ${BASE_CHECKPOINT}"
echo "Joint jerk             : -${JOINT_JERK_PENALTY} * mean(jerk^2)"
echo "Cartesian key bodies  : +${DEFAULT_CARTESIAN_POSE_WEIGHT}, variance=${DEFAULT_CARTESIAN_POSE_VARIANCE} m^2"
echo "Default foot distance : -${DEFAULT_FEET_DISTANCE_PENALTY} * squared_error_m2"
echo "Foot-distance peak    : +${DEFAULT_FEET_GAUSSIAN_WEIGHT}, variance=${DEFAULT_FEET_GAUSSIAN_VARIANCE} m^2"
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
  "env.rewards.joint_jerk_l2.weight=-${JOINT_JERK_PENALTY}" \
  "env.rewards.default_key_body_pose_gaussian.weight=${DEFAULT_CARTESIAN_POSE_WEIGHT}" \
  "env.rewards.default_key_body_pose_gaussian.params.variance=${DEFAULT_CARTESIAN_POSE_VARIANCE}" \
  "env.rewards.default_feet_distance_l2.weight=-${DEFAULT_FEET_DISTANCE_PENALTY}" \
  "env.rewards.default_feet_distance_gaussian.weight=${DEFAULT_FEET_GAUSSIAN_WEIGHT}" \
  "env.rewards.default_feet_distance_gaussian.params.variance=${DEFAULT_FEET_GAUSSIAN_VARIANCE}" \
  "$@"
