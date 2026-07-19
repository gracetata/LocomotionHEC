#!/usr/bin/env bash
# Export a checkpoint and test full-body recovery under initial noise and wrenches in MuJoCo.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

CHECKPOINT=${CHECKPOINT:-"${PROJECT_ROOT}/checkpoint/stand/model_2999.pt"}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
UNITREE_PYTHON=${UNITREE_PYTHON:-"${HOME}/anaconda3/envs/gmr/bin/python"}
USE_GLFW=${USE_GLFW:-True}
SIMULATION_DURATION=${SIMULATION_DURATION:-30.0}
SEED=${SEED:-20260719}
FORCE_MAX_N=${FORCE_MAX_N:-35.0}
TORQUE_MAX_NM=${TORQUE_MAX_NM:-5.0}
WRENCH_INTERVAL_S=${WRENCH_INTERVAL_S:-2.5}
WRENCH_DURATION_S=${WRENCH_DURATION_S:-0.25}

[[ -f "${CHECKPOINT}" ]] || { echo "Error: CHECKPOINT not found: ${CHECKPOINT}" >&2; exit 1; }
CHECKPOINT=$(realpath "${CHECKPOINT}")
EXPORT_DIR=${EXPORT_DIR:-"$(dirname "${CHECKPOINT}")/exported_extreme_stand_recovery"}
POLICY_PATH=${POLICY_PATH:-"${EXPORT_DIR}/policy.pt"}
METRICS_PATH=${METRICS_PATH:-"${EXPORT_DIR}/mujoco_extreme_stand_recovery_metrics.json"}

if [[ ! -f "${POLICY_PATH}" || "${FORCE_EXPORT:-False}" =~ ^([Tt]rue|1)$ ]]; then
    CHECKPOINT="${CHECKPOINT}" \
    ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" \
    EXPORT_DIR="${EXPORT_DIR}" \
    bash "${PROJECT_ROOT}/scripts/export_g1_amp_policy.sh"
fi

echo "MuJoCo policy    : ${POLICY_PATH}"
echo "Initial noise    : full-body joint/root state enabled"
echo "Random wrench    : +/-${FORCE_MAX_N} N, +/-${TORQUE_MAX_NM} Nm every ${WRENCH_INTERVAL_S} s"
echo "Action contract  : actor output is used for all 29 joints without overwrite"

G1_AMP_ARMHACK_STAND_ENABLE=False \
G1_AMP_ARMHACK_WALK_ENABLE=False \
G1_AMP_EXTREME_STAND_RECOVERY_ENABLE=True \
G1_AMP_EXTREME_STAND_RECOVERY_SEED="${SEED}" \
G1_AMP_EXTREME_STAND_FORCE_MAX_N="${FORCE_MAX_N}" \
G1_AMP_EXTREME_STAND_TORQUE_MAX_NM="${TORQUE_MAX_NM}" \
G1_AMP_EXTREME_STAND_WRENCH_INTERVAL_S="${WRENCH_INTERVAL_S}" \
G1_AMP_EXTREME_STAND_WRENCH_DURATION_S="${WRENCH_DURATION_S}" \
UNITREE_PYTHON="${UNITREE_PYTHON}" \
POLICY_PATH="${POLICY_PATH}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" \
REAL_TIME="${USE_GLFW}" \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
CMD_INIT='[0.0,0.0,0.0]' \
RANDOM_COMMANDS=False \
COMMAND_MODE=independent \
COMMAND_RAMP=False \
METRICS_PATH="${METRICS_PATH}" \
bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh" "$@"
