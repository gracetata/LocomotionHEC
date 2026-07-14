#!/usr/bin/env bash
# Deploy a previously exported G1 AMP locomotion.onnx on the real robot.
# This launcher only reads an ONNX file; it never reads a checkpoint or exports a model.
# Direct handoff: do not proactively switch zero-torque/damping/native modes before policy rollout.
#
# Usage:
#   DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_amp_onnx.sh checkpoint/model_9996/locomotion.onnx
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=joystick \
#     bash scripts/deploy_real_g1_amp_onnx.sh checkpoint/model_9996/locomotion.onnx
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=nav_mock \
#     bash scripts/deploy_real_g1_amp_onnx.sh checkpoint/model_9996/locomotion.onnx
#
# Command modes:
#   fixed    : use CMD_INIT='[vx, vy, yaw_rate]'.
#   joystick : read /dev/input/js* with the same axis/range mapping as scripts/sim2sim_g1_amp_mujoco.sh.
#   nav_udp  : receive 50Hz UDP commands from the navigation module.
#   nav_mock : publish [0.7, 0.0, 0.0] at 50Hz through the same UDP receiver.
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# !!运行后如果发生抖振:这个版本没有提前进入调试模式,待更新,先运行scripts/deploy_real_g1_amp_loco_policy.sh中的策略即可进入
# !!如果静默退出,插拔网线即可
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=joystick \
    bash scripts/deploy_real_g1_amp_onnx.sh checkpoint/model_8997/locomotion.onnx

CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=joystick \
    bash scripts/deploy_real_g1_amp_onnx.sh checkpoint/model_9996/locomotion.onnx

CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=nav_mock NAV_MOCK_CMD='[0.4,0.0,0.0]'\
    bash scripts/deploy_real_g1_amp_onnx.sh checkpoint/model_8997/locomotion.onnx
BLOCK
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
DEFAULT_ONNX="${ROOT_DIR}/checkpoint/model_9996/locomotion.onnx"
SOURCE_ARG=${1:-}
ONNX_PATH=${SOURCE_ARG:-${ONNX_PATH:-${DEFAULT_ONNX}}}
if [[ "${ONNX_PATH}" != /* ]]; then
    ONNX_PATH="${ROOT_DIR}/${ONNX_PATH}"
fi

NET=${NET:-}
ROBOT_IP=${ROBOT_IP:-192.168.123.161}
CONFIG=${CONFIG:-g1_amp.yaml}
COMMAND_MODE=${COMMAND_MODE:-remote}
CMD_INIT=${CMD_INIT:-'[0.0, 0.0, 0.0]'}
RUN_DURATION=${RUN_DURATION:-0.0}
if [[ -z "${COMMAND_RAMP+x}" ]]; then
    if [[ "${COMMAND_MODE}" == "fixed" || "${COMMAND_MODE}" == nav* ]]; then
        COMMAND_RAMP=True
    else
        COMMAND_RAMP=False
    fi
fi
COMMAND_MAX_LINEAR_ACCEL=${COMMAND_MAX_LINEAR_ACCEL:-0.5}
COMMAND_MAX_YAW_ACCEL=${COMMAND_MAX_YAW_ACCEL:-0.8}
COMMAND_PRINT_PERIOD_S=${COMMAND_PRINT_PERIOD_S:-0.5}

# Direct policy flow: no ReleaseMode(), no zero-torque gate, no damping gate.
# With HANDOFF_MODE=stand and DEFAULT_MOVE/HOLD at 0, deploy_real_g1_amp.py skips
# pre-policy default-pose interpolation; the first outgoing LowCmd is the policy PD target.
RELEASE_MOTION_MODE=${RELEASE_MOTION_MODE:-False}
HANDOFF_MODE=${HANDOFF_MODE:-stand}
TERMINAL_SPACE_HANDOFF=${TERMINAL_SPACE_HANDOFF:-False}
DEFAULT_MOVE_S=${DEFAULT_MOVE_S:-0.0}
DEFAULT_HOLD_S=${DEFAULT_HOLD_S:-0.0}
RELEASE_HANDOFF_WARMUP_S=${RELEASE_HANDOFF_WARMUP_S:-0.0}
RELEASE_HANDOFF_AFTER_S=${RELEASE_HANDOFF_AFTER_S:-0.0}

JOYSTICK_DEVICE=${JOYSTICK_DEVICE:-/dev/input/js0}
JOYSTICK_AXIS_LIN_X=${JOYSTICK_AXIS_LIN_X:-1}
JOYSTICK_AXIS_LIN_Y=${JOYSTICK_AXIS_LIN_Y:-0}
JOYSTICK_AXIS_YAW=${JOYSTICK_AXIS_YAW:-3}
JOYSTICK_SIGN_LIN_X=${JOYSTICK_SIGN_LIN_X:--1.0}
JOYSTICK_SIGN_LIN_Y=${JOYSTICK_SIGN_LIN_Y:--1.0}
JOYSTICK_SIGN_YAW=${JOYSTICK_SIGN_YAW:--1.0}
JOYSTICK_AXIS_MAX=${JOYSTICK_AXIS_MAX:-32768.0}
JOYSTICK_DEADZONE=${JOYSTICK_DEADZONE:-0.05}
JOYSTICK_LIN_X_RANGE=${JOYSTICK_LIN_X_RANGE:-'[-0.2,1.5]'}
JOYSTICK_LIN_Y_RANGE=${JOYSTICK_LIN_Y_RANGE:-'[-0.25,0.25]'}
JOYSTICK_YAW_RANGE=${JOYSTICK_YAW_RANGE:-'[-0.6,0.6]'}

NAV_UDP_BIND_HOST=${NAV_UDP_BIND_HOST:-0.0.0.0}
NAV_UDP_PORT=${NAV_UDP_PORT:-15050}
NAV_COMMAND_TIMEOUT_S=${NAV_COMMAND_TIMEOUT_S:-0.25}
NAV_STALE_BEHAVIOR=${NAV_STALE_BEHAVIOR:-zero}
NAV_COMMAND_CLIP_MIN=${NAV_COMMAND_CLIP_MIN:-'[-0.8,-0.5,-1.57]'}
NAV_COMMAND_CLIP_MAX=${NAV_COMMAND_CLIP_MAX:-'[0.8,0.5,1.57]'}
NAV_MOCK_CMD=${NAV_MOCK_CMD:-'[0.4,0.0,0.0]'}
NAV_MOCK_RATE_HZ=${NAV_MOCK_RATE_HZ:-50.0}
NAV_MOCK_TARGET_HOST=${NAV_MOCK_TARGET_HOST:-127.0.0.1}

RECOVER_NATIVE_ON_EXIT=${RECOVER_NATIVE_ON_EXIT:-False}
NATIVE_RECOVER_MODE=${NATIVE_RECOVER_MODE:-damp}
NATIVE_RECOVER_STRICT=${NATIVE_RECOVER_STRICT:-False}
NATIVE_TARGET_MODE=${NATIVE_TARGET_MODE:-ai}
NATIVE_START_SERVICE=${NATIVE_START_SERVICE:-ai_sport}
NATIVE_LOCO_SERVICE=${NATIVE_LOCO_SERVICE:-sport}
NATIVE_RECOVER_TIMEOUT_S=${NATIVE_RECOVER_TIMEOUT_S:-20.0}
NATIVE_STANDUP_WAIT_S=${NATIVE_STANDUP_WAIT_S:-8.0}
NATIVE_MIN_STANDUP_S=${NATIVE_MIN_STANDUP_S:-6.0}
NATIVE_START_FSM_IDS=${NATIVE_START_FSM_IDS:-801,802,500}
NATIVE_LOCOMOTION_FSM_IDS=${NATIVE_LOCOMOTION_FSM_IDS:-802,801,500}

SMOKE_TEST_POLICY=${SMOKE_TEST_POLICY:-True}
PING_ROBOT=${PING_ROBOT:-True}
CONFIRM_REAL_ROBOT=${CONFIRM_REAL_ROBOT:-}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

case "${COMMAND_MODE}" in
    fixed|joystick|nav_udp|nav_mock|nav|navigation|mock_nav|nav_loopback)
        ;;
    *)
        echo "Error: COMMAND_MODE must be fixed, joystick, nav_udp, or nav_mock; got ${COMMAND_MODE}" >&2
        exit 1
        ;;
esac

if ! is_true "${DRY_RUN}" && [[ "${CONFIRM_REAL_ROBOT}" != "I_UNDERSTAND" ]]; then
    echo "Refusing to start real-robot deployment without CONFIRM_REAL_ROBOT=I_UNDERSTAND." >&2
    exit 2
fi
if [[ -z "${NET}" ]]; then
    echo "Error: set NET to the Unitree network interface, e.g. NET=enp11s0." >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "Error: ONNX_PATH does not exist: ${ONNX_PATH}" >&2
    echo "Run: CHECKPOINT=<model_*.pt> bash scripts/export_g1_amp_locomotion_onnx.sh" >&2
    exit 1
fi
if [[ "${ONNX_PATH}" != *.onnx ]]; then
    echo "Error: this launcher only accepts .onnx files: ${ONNX_PATH}" >&2
    exit 1
fi
if ! is_true "${DRY_RUN}" && [[ "${COMMAND_MODE}" == "joystick" && ! -r "${JOYSTICK_DEVICE}" ]]; then
    echo "Error: JOYSTICK_DEVICE is not readable: ${JOYSTICK_DEVICE}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/user/anaconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export G1_AMP_POLICY_PATH="${ONNX_PATH}"
export G1_AMP_POLICY_RUNTIME=onnx
export G1_AMP_COMMAND_MODE="${COMMAND_MODE}"
export G1_AMP_CMD_INIT="${CMD_INIT}"
export G1_AMP_RUN_DURATION="${RUN_DURATION}"
export G1_AMP_COMMAND_RAMP="${COMMAND_RAMP}"
export G1_AMP_COMMAND_MAX_LINEAR_ACCEL="${COMMAND_MAX_LINEAR_ACCEL}"
export G1_AMP_COMMAND_MAX_YAW_ACCEL="${COMMAND_MAX_YAW_ACCEL}"
export G1_AMP_COMMAND_PRINT_PERIOD_S="${COMMAND_PRINT_PERIOD_S}"
export G1_AMP_RELEASE_MOTION_MODE="${RELEASE_MOTION_MODE}"
export G1_AMP_HANDOFF_MODE="${HANDOFF_MODE}"
export G1_AMP_TERMINAL_SPACE_HANDOFF="${TERMINAL_SPACE_HANDOFF}"
export G1_AMP_WAIT_FOR_BUTTON_A=False
export G1_AMP_DEFAULT_MOVE_S="${DEFAULT_MOVE_S}"
export G1_AMP_DEFAULT_HOLD_S="${DEFAULT_HOLD_S}"
export G1_AMP_RELEASE_HANDOFF_WARMUP_S="${RELEASE_HANDOFF_WARMUP_S}"
export G1_AMP_RELEASE_HANDOFF_AFTER_S="${RELEASE_HANDOFF_AFTER_S}"
export G1_AMP_JOYSTICK_DEVICE="${JOYSTICK_DEVICE}"
export G1_AMP_JOYSTICK_AXIS_LIN_X="${JOYSTICK_AXIS_LIN_X}"
export G1_AMP_JOYSTICK_AXIS_LIN_Y="${JOYSTICK_AXIS_LIN_Y}"
export G1_AMP_JOYSTICK_AXIS_YAW="${JOYSTICK_AXIS_YAW}"
export G1_AMP_JOYSTICK_SIGN_LIN_X="${JOYSTICK_SIGN_LIN_X}"
export G1_AMP_JOYSTICK_SIGN_LIN_Y="${JOYSTICK_SIGN_LIN_Y}"
export G1_AMP_JOYSTICK_SIGN_YAW="${JOYSTICK_SIGN_YAW}"
export G1_AMP_JOYSTICK_AXIS_MAX="${JOYSTICK_AXIS_MAX}"
export G1_AMP_JOYSTICK_DEADZONE="${JOYSTICK_DEADZONE}"
export G1_AMP_JOYSTICK_LIN_X_RANGE="${JOYSTICK_LIN_X_RANGE}"
export G1_AMP_JOYSTICK_LIN_Y_RANGE="${JOYSTICK_LIN_Y_RANGE}"
export G1_AMP_JOYSTICK_YAW_RANGE="${JOYSTICK_YAW_RANGE}"
export G1_AMP_NAV_UDP_BIND_HOST="${NAV_UDP_BIND_HOST}"
export G1_AMP_NAV_UDP_PORT="${NAV_UDP_PORT}"
export G1_AMP_NAV_COMMAND_TIMEOUT_S="${NAV_COMMAND_TIMEOUT_S}"
export G1_AMP_NAV_STALE_BEHAVIOR="${NAV_STALE_BEHAVIOR}"
export G1_AMP_NAV_COMMAND_CLIP_MIN="${NAV_COMMAND_CLIP_MIN}"
export G1_AMP_NAV_COMMAND_CLIP_MAX="${NAV_COMMAND_CLIP_MAX}"
export G1_AMP_NAV_MOCK_CMD="${NAV_MOCK_CMD}"
export G1_AMP_NAV_MOCK_RATE_HZ="${NAV_MOCK_RATE_HZ}"
export G1_AMP_NAV_MOCK_TARGET_HOST="${NAV_MOCK_TARGET_HOST}"
export G1_AMP_RECOVER_NATIVE_ON_EXIT="${RECOVER_NATIVE_ON_EXIT}"
export G1_AMP_NATIVE_RECOVER_MODE="${NATIVE_RECOVER_MODE}"
export G1_AMP_NATIVE_RECOVER_STRICT="${NATIVE_RECOVER_STRICT}"
export G1_AMP_NATIVE_TARGET_MODE="${NATIVE_TARGET_MODE}"
export G1_AMP_NATIVE_START_SERVICE="${NATIVE_START_SERVICE}"
export G1_AMP_NATIVE_LOCO_SERVICE="${NATIVE_LOCO_SERVICE}"
export G1_AMP_NATIVE_RECOVER_TIMEOUT_S="${NATIVE_RECOVER_TIMEOUT_S}"
export G1_AMP_NATIVE_STANDUP_WAIT_S="${NATIVE_STANDUP_WAIT_S}"
export G1_AMP_NATIVE_MIN_STANDUP_S="${NATIVE_MIN_STANDUP_S}"
export G1_AMP_NATIVE_START_FSM_IDS="${NATIVE_START_FSM_IDS}"
export G1_AMP_NATIVE_LOCOMOTION_FSM_IDS="${NATIVE_LOCOMOTION_FSM_IDS}"

echo "====================================="
echo "  Real G1 AMP ONNX Deploy Launcher"
echo "====================================="
echo "Python       : ${UNITREE_PYTHON}"
echo "Net          : ${NET}"
echo "Config       : ${CONFIG}"
echo "ONNX         : ${ONNX_PATH}"
echo "Command Mode : ${COMMAND_MODE}"
if [[ "${COMMAND_MODE}" == "fixed" ]]; then
    echo "Command      : ${CMD_INIT}"
elif [[ "${COMMAND_MODE}" == "joystick" ]]; then
    echo "Joystick     : device=${JOYSTICK_DEVICE} axes=(${JOYSTICK_AXIS_LIN_X},${JOYSTICK_AXIS_LIN_Y},${JOYSTICK_AXIS_YAW})"
    echo "Joy Ranges   : x=${JOYSTICK_LIN_X_RANGE} y=${JOYSTICK_LIN_Y_RANGE} yaw=${JOYSTICK_YAW_RANGE}"
elif [[ "${COMMAND_MODE}" == "nav_mock" || "${COMMAND_MODE}" == "mock_nav" || "${COMMAND_MODE}" == "nav_loopback" ]]; then
    echo "Nav Mock     : ${NAV_MOCK_RATE_HZ}Hz ${NAV_MOCK_CMD} via UDP port ${NAV_UDP_PORT}"
else
    echo "Nav UDP      : bind=${NAV_UDP_BIND_HOST}:${NAV_UDP_PORT} timeout=${NAV_COMMAND_TIMEOUT_S}s stale=${NAV_STALE_BEHAVIOR}"
fi
echo "Duration     : ${RUN_DURATION}"
echo "Ramp         : ${COMMAND_RAMP} lin_acc=${COMMAND_MAX_LINEAR_ACCEL} yaw_acc=${COMMAND_MAX_YAW_ACCEL}"
echo "Release Mode : ${RELEASE_MOTION_MODE}"
echo "Handoff Mode : ${HANDOFF_MODE}"
echo "Terminal Gate: ${TERMINAL_SPACE_HANDOFF}"
echo "Terminal Flow: direct -> ONNX policy (no zero torque / damping / ReleaseMode)"
echo "Default Move : ${DEFAULT_MOVE_S}s"
echo "Recover Nat. : ${RECOVER_NATIVE_ON_EXIT} mode=${NATIVE_RECOVER_MODE} target=${NATIVE_TARGET_MODE}"
echo "Smoke Test   : ${SMOKE_TEST_POLICY}"
echo "Dry Run      : ${DRY_RUN}"
echo "====================================="

RUN_COMMAND=("${UNITREE_PYTHON}" deploy_real_g1_amp.py "${NET}" "${CONFIG}")

if is_true "${DRY_RUN}"; then
    printf 'Dry-run deploy command:'
    printf ' %q' env G1_AMP_POLICY_PATH="${ONNX_PATH}" G1_AMP_POLICY_RUNTIME=onnx \
        G1_AMP_COMMAND_MODE="${COMMAND_MODE}" G1_AMP_CMD_INIT="${CMD_INIT}" \
        G1_AMP_RELEASE_MOTION_MODE="${RELEASE_MOTION_MODE}" G1_AMP_HANDOFF_MODE="${HANDOFF_MODE}" \
        G1_AMP_TERMINAL_SPACE_HANDOFF="${TERMINAL_SPACE_HANDOFF}" \
        G1_AMP_DEFAULT_MOVE_S="${DEFAULT_MOVE_S}" G1_AMP_DEFAULT_HOLD_S="${DEFAULT_HOLD_S}" \
        "${RUN_COMMAND[@]}"
    printf '\n'
    exit 0
fi

if is_true "${PING_ROBOT}"; then
    ping -c 1 -W 1 "${ROBOT_IP}" >/dev/null
    echo "Ping ${ROBOT_IP}: ok"
fi

if is_true "${SMOKE_TEST_POLICY}"; then
    "${UNITREE_PYTHON}" - "${ONNX_PATH}" <<'PY'
import sys
import numpy as np
import onnxruntime as ort

onnx_path = sys.argv[1]
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
obs = np.zeros((1, 96), dtype=np.float32)
action = np.asarray(session.run([output_name], {input_name: obs})[0])
shape = tuple(action.shape)
print(f"ONNX smoke: {onnx_path} -> {shape}")
if shape not in {(1, 29), (29,)}:
    raise SystemExit(f"Unexpected ONNX output shape: {shape}, expected (1, 29)")
PY
fi

cd "${UNITREE_DIR}/deploy/deploy_real"
"${RUN_COMMAND[@]}"