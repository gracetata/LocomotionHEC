#!/usr/bin/env bash
# Interactive MuJoCo handoff test for the original usable Stand and current Walk actor.
# No real-robot transport or DDS process is started by this script.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAND_DIR=${STAND_DIR:-${ROOT_DIR}/checkpoint/stand/armhack_step_stabilize_20260815}
STAND_CHECKPOINT=${STAND_CHECKPOINT:-${STAND_DIR}/model_1999.pt}
STAND_POLICY=${STAND_POLICY:-${STAND_DIR}/policy.onnx}
STAND_METADATA=${STAND_METADATA:-${STAND_DIR}/policy.deploy.json}
STAND_CHECKPOINT_SHA256=${STAND_CHECKPOINT_SHA256:-dc87b7f4e1fb451556cfc81ac2af926896bebdd41c1199e7b5236687b5952c0e}
STAND_POLICY_SHA256=${STAND_POLICY_SHA256:-c9a028ced244f7b62c87c9dba8e5497d852afd67c113bccdba67f9a11a85349c}

# Current best single-actor Walk after micro-backward training and deployment calibration.
WALK_POLICY=${WALK_POLICY:-${ROOT_DIR}/legged_lab/evaluations/first_principles_single/final_models/walk_v17_calibrated/policy.onnx}
WALK_POLICY_SHA256=${WALK_POLICY_SHA256:-bc02dc4055ed418753cdb0cc7cbd8ae332739d5d4cbbd20295c7a2f7025b3ebd}

TEST_DATA_DIR="${ROOT_DIR}/legged_lab/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly"
STAND_CSV=${STAND_CSV:-${TEST_DATA_DIR}/special/arms_down_flat_forward_return_flat_25p5s_50hz.csv}
STAND_MANIFEST=${STAND_MANIFEST:-${TEST_DATA_DIR}/manifest.json}
STAND_PRESETS=${STAND_PRESETS:-${ROOT_DIR}/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/legged_lab/outputs/armhack_stand_walk_switch}
UNITREE_PYTHON=${UNITREE_PYTHON:-${HOME}/anaconda3/envs/gmr/bin/python}
SIMULATION_DURATION=${SIMULATION_DURATION:-600.0}
USE_GLFW=${USE_GLFW:-True}
REAL_TIME=${REAL_TIME:-True}
INITIAL_STANCE_M=${INITIAL_STANCE_M:--1.0}
ARM_TRANSITION_S=${ARM_TRANSITION_S:-3.0}
WALK_INITIAL_COMMAND=${WALK_INITIAL_COMMAND:-'[0.25, 0.0, 0.0]'}

check_sha() {
    local path=$1 expected=$2 label=$3 actual
    [[ -f "${path}" ]] || { echo "Error: ${label} missing: ${path}" >&2; exit 1; }
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]] || {
        echo "Error: ${label} SHA-256 mismatch: expected=${expected} actual=${actual}" >&2
        exit 1
    }
}

[[ -x "${UNITREE_PYTHON}" ]] || { echo "Error: MuJoCo Python missing: ${UNITREE_PYTHON}" >&2; exit 1; }
for required in "${STAND_METADATA}" "${STAND_CSV}" "${STAND_MANIFEST}" "${STAND_PRESETS}"; do
    [[ -f "${required}" ]] || { echo "Error: required file missing: ${required}" >&2; exit 1; }
done
check_sha "${STAND_CHECKPOINT}" "${STAND_CHECKPOINT_SHA256}" "Stand checkpoint"
check_sha "${STAND_POLICY}" "${STAND_POLICY_SHA256}" "Stand ONNX"
check_sha "${WALK_POLICY}" "${WALK_POLICY_SHA256}" "Walk ONNX"
mkdir -p "${OUTPUT_DIR}"

export G1_AMP_POLICY_SWITCH_ENABLE=True
export G1_AMP_POLICY_SWITCH_START_MODE=stand
export G1_AMP_POLICY_SWITCH_WALK_POLICY_PATH="${WALK_POLICY}"
export G1_AMP_POLICY_SWITCH_WALK_POLICY_SHA256="${WALK_POLICY_SHA256}"
export G1_AMP_POLICY_SWITCH_AUTO_TOGGLE_INTERVAL_S=${POLICY_SWITCH_AUTO_TOGGLE_INTERVAL_S:--1.0}
export G1_AMP_POLICY_SWITCH_AUTO_TOGGLE_MAX=${POLICY_SWITCH_AUTO_TOGGLE_MAX:-0}
export G1_AMP_DEPLOY_METADATA_PATH="${STAND_METADATA}"

export G1_AMP_ARMHACK_STAND_ENABLE=True
export G1_AMP_ARMHACK_WALK_ENABLE=False
export G1_AMP_ARMHACK_STAND_CSV_PATH="${STAND_CSV}"
export G1_AMP_ARMHACK_STAND_MANIFEST_PATH="${STAND_MANIFEST}"
export G1_AMP_ARMHACK_STAND_CHECKPOINT_PATH="${STAND_CHECKPOINT}"
export G1_AMP_ARMHACK_STAND_CHECKPOINT_SHA256="${STAND_CHECKPOINT_SHA256}"
export G1_AMP_ARMHACK_STAND_REPORT_PATH="${OUTPUT_DIR}/interactive_switch_report.md"
export G1_AMP_ARMHACK_STAND_TEST_ID=interactive
export G1_AMP_ARMHACK_STAND_PAYLOAD_KG=0.0
export G1_AMP_ARMHACK_STAND_INTERACTIVE_ENABLE=True
export G1_AMP_ARMHACK_STAND_INTERACTIVE_DIRECT_ENTER=True
export G1_AMP_ARMHACK_STAND_PRESET_PATH="${STAND_PRESETS}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_TRANSITION_S="${ARM_TRANSITION_S}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_ENTER_S=${G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_ENTER_S:--1.0}
export G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_SPACE_INTERVAL_S=${G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_SPACE_INTERVAL_S:--1.0}
export G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_SPACE_MAX_SWITCHES=${G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_SPACE_MAX_SWITCHES:-0}
export G1_AMP_ARMHACK_STAND_INITIAL_STANCE_M="${INITIAL_STANCE_M}"

echo "============================================================"
echo " ArmHack original Stand <-> current Walk interactive MuJoCo"
echo "============================================================"
echo "Startup : STAND inference is active immediately; no initial key is required"
echo "ENTER   : first press STAND -> WALK; subsequent presses toggle WALK <-> STAND"
echo "SPACE/P : cycle shared arm poses with minimum-jerk motion"
echo "W/S     : Walk vx +/- ${KEYBOARD_LINEAR_STEP:-0.01} m/s"
echo "A/D     : Walk vy +/- ${KEYBOARD_LINEAR_STEP:-0.01} m/s"
echo "Q/E     : Walk yaw-rate +/- ${KEYBOARD_YAW_STEP:-0.05} rad/s"
echo "0       : set stored Walk command to exact zero"
echo "1..7    : Walk command presets (3=pure yaw)"
echo "ESC     : stop; closing the viewer also stops"
echo "Stand   : ${STAND_POLICY}"
echo "Walk    : ${WALK_POLICY}"
echo "============================================================"

UNITREE_PYTHON="${UNITREE_PYTHON}" \
POLICY_PATH="${STAND_POLICY}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" REAL_TIME="${REAL_TIME}" SIMULATION_DURATION="${SIMULATION_DURATION}" \
COMMAND_MODE=keyboard COMMAND_RAMP=True CMD_INIT="${WALK_INITIAL_COMMAND}" \
COMMAND_SMOOTHING_TAU=${COMMAND_SMOOTHING_TAU:-0.30} \
COMMAND_MAX_LINEAR_ACCEL=${COMMAND_MAX_LINEAR_ACCEL:-0.60} \
COMMAND_MAX_YAW_ACCEL=${COMMAND_MAX_YAW_ACCEL:-0.80} \
RANDOM_COMMANDS=False \
TORSO_TRACE_ENABLE=True TORSO_TRACE_PATH="${OUTPUT_DIR}/torso_trace.csv" \
TASK_TRACE_ENABLE=True TASK_TRACE_PATH="${OUTPUT_DIR}/task_trace.csv" \
METRICS_PATH="${OUTPUT_DIR}/metrics.json" \
FOLLOW_CAMERA_ENABLE=${FOLLOW_CAMERA_ENABLE:-True} \
bash "${ROOT_DIR}/scripts/sim2sim_g1_amp_mujoco.sh"
