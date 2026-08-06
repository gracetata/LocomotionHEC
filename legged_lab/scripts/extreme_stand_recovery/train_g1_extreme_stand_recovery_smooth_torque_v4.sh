#!/usr/bin/env bash
# Continue the final Anti-Jitter V3 policy with stronger smoothness costs.
#
# V4 keeps the 96-observation -> 29-action full-body Stand contract and all
# posture/upright/survival rewards.  It adds explicit policy-action curvature
# and applied-torque-rate penalties, strengthens torque/acceleration/jerk
# regularization, and widens the initial-state/disturbance distribution.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_resume1400_to2999_full_20260724/model_2999.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-e2c694d2d7710315f41f1c6c75849ffb95b53d0fb29e612aa211e1525a7cb1e4}
VERIFY_BASE_SHA256=${VERIFY_BASE_SHA256:-True}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999}
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
NUM_ENVS=${NUM_ENVS:-4096}
LEARNING_RATE=${LEARNING_RATE:-1.0e-5}

# Smoothness costs.  The torque-rate and jerk terms use physical derivatives;
# their coefficients are consequently much smaller than the action terms.
JOINT_TORQUE_PENALTY=${JOINT_TORQUE_PENALTY:-1.0e-5}
JOINT_TORQUE_RATE_PENALTY=${JOINT_TORQUE_RATE_PENALTY:-2.0e-7}
JOINT_ACCELERATION_PENALTY=${JOINT_ACCELERATION_PENALTY:-5.0e-7}
JOINT_JERK_PENALTY=${JOINT_JERK_PENALTY:-5.0e-8}
ACTION_RATE_PENALTY=${ACTION_RATE_PENALTY:-5.0e-2}
ACTION_SECOND_DIFFERENCE_PENALTY=${ACTION_SECOND_DIFFERENCE_PENALTY:-1.0e-1}

# V4 deliberately covers somewhat wider initial states and disturbances than
# V3.  These remain within the existing task launcher's validated safe ranges.
LEG_NOISE_RAD=${LEG_NOISE_RAD:-0.30}
WAIST_NOISE_RAD=${WAIST_NOISE_RAD:-0.40}
ARM_NOISE_RAD=${ARM_NOISE_RAD:-0.65}
TORSO_FORCE_MAX_N=${TORSO_FORCE_MAX_N:-45.0}
TORSO_TORQUE_MAX_NM=${TORSO_TORQUE_MAX_NM:-6.0}
PELVIS_FORCE_MAX_N=${PELVIS_FORCE_MAX_N:-40.0}
PELVIS_TORQUE_MAX_NM=${PELVIS_TORQUE_MAX_NM:-5.0}
LIMB_FORCE_MAX_N=${LIMB_FORCE_MAX_N:-15.0}
LIMB_TORQUE_MAX_NM=${LIMB_TORQUE_MAX_NM:-2.5}

validate_positive() {
    local name=$1 value=$2
    awk -v value="${value}" 'BEGIN { exit !(value > 0.0) }' || {
        echo "Error: ${name} must be positive, got ${value}." >&2
        exit 1
    }
}

for item in \
    "JOINT_TORQUE_PENALTY:${JOINT_TORQUE_PENALTY}" \
    "JOINT_TORQUE_RATE_PENALTY:${JOINT_TORQUE_RATE_PENALTY}" \
    "JOINT_ACCELERATION_PENALTY:${JOINT_ACCELERATION_PENALTY}" \
    "JOINT_JERK_PENALTY:${JOINT_JERK_PENALTY}" \
    "ACTION_RATE_PENALTY:${ACTION_RATE_PENALTY}" \
    "ACTION_SECOND_DIFFERENCE_PENALTY:${ACTION_SECOND_DIFFERENCE_PENALTY}"; do
    validate_positive "${item%%:*}" "${item#*:}"
done

echo "============================================================"
echo " Extreme Stand Recovery Smooth-Torque V4 continuation"
echo "============================================================"
echo "Base checkpoint       : ${BASE_CHECKPOINT}"
echo "Applied torque L2     : -${JOINT_TORQUE_PENALTY}"
echo "Applied torque rate   : -${JOINT_TORQUE_RATE_PENALTY}"
echo "Joint acceleration    : -${JOINT_ACCELERATION_PENALTY}"
echo "Joint jerk            : -${JOINT_JERK_PENALTY}"
echo "Action rate           : -${ACTION_RATE_PENALTY}"
echo "Action 2nd difference : -${ACTION_SECOND_DIFFERENCE_PENALTY}"
echo "Joint reset noise     : leg=${LEG_NOISE_RAD}, waist=${WAIST_NOISE_RAD}, arm=${ARM_NOISE_RAD} rad"
echo "Torso wrench          : +/-${TORSO_FORCE_MAX_N} N, +/-${TORSO_TORQUE_MAX_NM} Nm"
echo "Pelvis wrench         : +/-${PELVIS_FORCE_MAX_N} N, +/-${PELVIS_TORQUE_MAX_NM} Nm"
echo "Limb wrench           : +/-${LIMB_FORCE_MAX_N} N, +/-${LIMB_TORQUE_MAX_NM} Nm"
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
LEG_NOISE_RAD="${LEG_NOISE_RAD}" \
WAIST_NOISE_RAD="${WAIST_NOISE_RAD}" \
ARM_NOISE_RAD="${ARM_NOISE_RAD}" \
TORSO_FORCE_MAX_N="${TORSO_FORCE_MAX_N}" \
TORSO_TORQUE_MAX_NM="${TORSO_TORQUE_MAX_NM}" \
PELVIS_FORCE_MAX_N="${PELVIS_FORCE_MAX_N}" \
PELVIS_TORQUE_MAX_NM="${PELVIS_TORQUE_MAX_NM}" \
LIMB_FORCE_MAX_N="${LIMB_FORCE_MAX_N}" \
LIMB_TORQUE_MAX_NM="${LIMB_TORQUE_MAX_NM}" \
bash "${SCRIPT_DIR}/train_g1_extreme_stand_recovery.sh" \
  "env.rewards.dof_torques_l2.weight=-${JOINT_TORQUE_PENALTY}" \
  "env.rewards.joint_torque_rate_l2.weight=-${JOINT_TORQUE_RATE_PENALTY}" \
  "env.rewards.dof_acc_l2.weight=-${JOINT_ACCELERATION_PENALTY}" \
  "env.rewards.joint_jerk_l2.weight=-${JOINT_JERK_PENALTY}" \
  "env.rewards.action_rate_l2.weight=-${ACTION_RATE_PENALTY}" \
  "env.rewards.action_second_difference_l2.weight=-${ACTION_SECOND_DIFFERENCE_PENALTY}" \
  "$@"
