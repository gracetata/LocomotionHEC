#!/usr/bin/env bash
# ArmHack Walk 真机入口：支持固定速度/Joystick，SPACE 平滑切换双臂姿态。
#
# 离线自检（不初始化 DDS）：
#   DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_armhack_walk.sh
# 真机（仅限吊架、现场急停和操作员就绪）：
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 \
#     bash scripts/deploy_real_g1_armhack_walk.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"

NET=${NET:-}
CONFIG=${CONFIG:-g1_amp.yaml}
POLICY_PATH=${POLICY_PATH:-${ROOT_DIR}/use/armhack_walk_model_10990.onnx}
POSE_PATH=${POSE_PATH:-${ROOT_DIR}/legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json}
CONTRACT_PATH=${CONTRACT_PATH:-${ROOT_DIR}/legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json}
POSE_NAME=${POSE_NAME:-pos2_down}
FIXED_COMMAND=${FIXED_COMMAND:-'[0.35,0.0,0.0]'}
COMMAND_MODE=${COMMAND_MODE:-fixed}
EXPECTED_POLICY_SHA256=${EXPECTED_POLICY_SHA256:-b052c3b0583834a742ea59e736d55c3c9bafabb75f1d4fae65980166d4a895aa}
EXPECTED_POSE_SHA256=${EXPECTED_POSE_SHA256:-deeeff8c8dd4ef00ac3e5740c45521522a737905ccb332cb973fcc5b15bb5f17}
EXPECTED_CONTRACT_SHA256=${EXPECTED_CONTRACT_SHA256:-8978e1aa7e9eefa742ac81d5230c18d2b9cd1546ea600a973da8fa02d8537c95}
STARTUP_MOVE_S=${STARTUP_MOVE_S:-5.0}
RUN_DURATION=${RUN_DURATION:-0.0}
JOINT_PRINT_HZ=${JOINT_PRINT_HZ:-1.0}
LOWSTATE_TIMEOUT_S=${LOWSTATE_TIMEOUT_S:-0.20}
MAX_TILT_RAD=${MAX_TILT_RAD:-0.60}
DAMPING_EXIT_S=${DAMPING_EXIT_S:-1.0}
ARM_POSE_SWITCH_S=${ARM_POSE_SWITCH_S:-2.0}
COMMAND_MAX_LINEAR_ACCEL=${COMMAND_MAX_LINEAR_ACCEL:-0.5}
COMMAND_MAX_YAW_ACCEL=${COMMAND_MAX_YAW_ACCEL:-0.8}
JOYSTICK_DEVICE=${JOYSTICK_DEVICE:-/dev/input/js0}
JOYSTICK_AXIS_LIN_X=${JOYSTICK_AXIS_LIN_X:-1}
JOYSTICK_AXIS_LIN_Y=${JOYSTICK_AXIS_LIN_Y:-0}
JOYSTICK_AXIS_YAW=${JOYSTICK_AXIS_YAW:-3}
JOYSTICK_SIGN_LIN_X=${JOYSTICK_SIGN_LIN_X:--1.0}
JOYSTICK_SIGN_LIN_Y=${JOYSTICK_SIGN_LIN_Y:--1.0}
JOYSTICK_SIGN_YAW=${JOYSTICK_SIGN_YAW:--1.0}
JOYSTICK_AXIS_MAX=${JOYSTICK_AXIS_MAX:-32768.0}
JOYSTICK_DEADZONE=${JOYSTICK_DEADZONE:-0.05}
JOYSTICK_LIN_X_RANGE=${JOYSTICK_LIN_X_RANGE:-'[-0.2,0.6]'}
JOYSTICK_LIN_Y_RANGE=${JOYSTICK_LIN_Y_RANGE:-'[-0.3,0.3]'}
JOYSTICK_YAW_RANGE=${JOYSTICK_YAW_RANGE:-'[-0.5187280216217041,0.6]'}
CONFIRM_REAL_ROBOT=${CONFIRM_REAL_ROBOT:-}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

if [[ -z "${UNITREE_PYTHON:-}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/env_leglab/bin/python" \
        "${HOME}/miniconda3/envs/armhack-real/bin/python" \
        "${HOME}/anaconda3/envs/armhack-real/bin/python" \
        "${HOME}/anaconda3/envs/unitree-rl/bin/python" \
        "${HOME}/anaconda3/envs/gmr/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            UNITREE_PYTHON="${candidate}"
            break
        fi
    done
fi

[[ -n "${UNITREE_PYTHON:-}" && -x "${UNITREE_PYTHON}" ]] || {
    echo "Error: 请设置 UNITREE_PYTHON 为安装了 Unitree SDK/ONNX Runtime 的 Python。" >&2
    exit 1
}
[[ -n "${NET}" ]] || { echo "Error: 必须设置连接 G1 的网卡，例如 NET=enp11s0。" >&2; exit 1; }
[[ "${COMMAND_MODE}" == "fixed" || "${COMMAND_MODE}" == "joystick" ]] || {
    echo "Error: COMMAND_MODE 只支持 fixed 或 joystick。" >&2
    exit 1
}
for path_var in POLICY_PATH POSE_PATH CONTRACT_PATH; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${ROOT_DIR}/${value}"
    fi
done
for path in "${POLICY_PATH}" "${POSE_PATH}" "${CONTRACT_PATH}"; do
    [[ -f "${path}" ]] || { echo "Error: 部署文件不存在: ${path}" >&2; exit 1; }
done

check_sha() {
    local path=$1 expected=$2 label=$3 actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ -n "${expected}" && "${actual}" != "${expected}" ]]; then
        echo "Error: ${label} SHA-256 不匹配。expected=${expected} actual=${actual}" >&2
        exit 1
    fi
    printf '%s' "${actual}"
}

policy_sha=$(check_sha "${POLICY_PATH}" "${EXPECTED_POLICY_SHA256}" policy)
pose_sha=$(check_sha "${POSE_PATH}" "${EXPECTED_POSE_SHA256}" pose_data)
contract_sha=$(check_sha "${CONTRACT_PATH}" "${EXPECTED_CONTRACT_SHA256}" deployment_contract)

read -r CMD_VX CMD_VY CMD_WZ <<<"$("${UNITREE_PYTHON}" - "${FIXED_COMMAND}" <<'PY'
import sys
import yaml
values = yaml.safe_load(sys.argv[1])
if not isinstance(values, list) or len(values) != 3:
    raise SystemExit("FIXED_COMMAND must be [vx,vy,wz]")
print(*(float(value) for value in values))
PY
)"

export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
UNITREE_ENV_PREFIX=$(cd "$(dirname "${UNITREE_PYTHON}")/.." && pwd)
export LD_LIBRARY_PATH="${UNITREE_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export G1_AMP_POLICY_PATH="${POLICY_PATH}"
export G1_AMP_POLICY_RUNTIME=onnx
export G1_AMP_COMMAND_MODE="${COMMAND_MODE}"
export G1_AMP_CMD_INIT="${FIXED_COMMAND}"
if [[ -z "${COMMAND_RAMP:-}" ]]; then
    if [[ "${COMMAND_MODE}" == "joystick" ]]; then
        COMMAND_RAMP=False
    else
        COMMAND_RAMP=True
    fi
fi
export G1_AMP_COMMAND_RAMP="${COMMAND_RAMP}"
export G1_AMP_COMMAND_MAX_LINEAR_ACCEL="${COMMAND_MAX_LINEAR_ACCEL}"
export G1_AMP_COMMAND_MAX_YAW_ACCEL="${COMMAND_MAX_YAW_ACCEL}"
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
export G1_AMP_RUN_DURATION="${RUN_DURATION}"
export G1_AMP_RELEASE_MOTION_MODE=True
export G1_AMP_HANDOFF_MODE=stand
export G1_AMP_WAIT_FOR_BUTTON_A=False
export G1_ARMHACK_WALK_POSE_PATH="${POSE_PATH}"
export G1_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_PATH}"
export G1_ARMHACK_WALK_POSE_NAME="${POSE_NAME}"
export G1_ARMHACK_WALK_STARTUP_MOVE_S="${STARTUP_MOVE_S}"
export G1_ARMHACK_WALK_JOINT_PRINT_HZ="${JOINT_PRINT_HZ}"
export G1_ARMHACK_WALK_LOWSTATE_TIMEOUT_S="${LOWSTATE_TIMEOUT_S}"
export G1_ARMHACK_WALK_MAX_TILT_RAD="${MAX_TILT_RAD}"
export G1_ARMHACK_WALK_DAMPING_EXIT_S="${DAMPING_EXIT_S}"
export G1_ARMHACK_WALK_ARM_POSE_SWITCH_S="${ARM_POSE_SWITCH_S}"
export G1_ARMHACK_WALK_CONFIRM="${CONFIRM_REAL_ROBOT}"

echo "============================================================"
echo " ArmHack Walk model_10990 Real-Robot Deployment"
echo "============================================================"
echo "Python       : ${UNITREE_PYTHON}"
echo "Net/config   : ${NET} / ${CONFIG}"
echo "Policy       : ${POLICY_PATH}"
echo "Policy SHA   : ${policy_sha}"
echo "Pose         : ${POSE_NAME} (${pose_sha})"
echo "Contract SHA : ${contract_sha}"
echo "Command mode : ${COMMAND_MODE}"
if [[ "${COMMAND_MODE}" == "fixed" ]]; then
    echo "Initial cmd  : [0,0,0]"
    echo "V cmd        : [${CMD_VX},${CMD_VY},${CMD_WZ}] <-> [0,0,0]"
else
    echo "Joystick     : ${JOYSTICK_DEVICE}; axes=(${JOYSTICK_AXIS_LIN_X},${JOYSTICK_AXIS_LIN_Y},${JOYSTICK_AXIS_YAW})"
    echo "Joy ranges   : x=${JOYSTICK_LIN_X_RANGE} y=${JOYSTICK_LIN_Y_RANGE} yaw=${JOYSTICK_YAW_RANGE}"
fi
echo "Cmd ramp     : ${COMMAND_RAMP}"
echo "SPACE arms   : pos1_back -> pos2_down -> pos3_front (minimum-jerk ${ARM_POSE_SWITCH_S}s)"
echo "Stop         : q / Ctrl-C / remote Select -> damping"
echo "Dry run      : ${DRY_RUN}"
echo "============================================================"

SELF_TEST_COMMAND=(
    "${UNITREE_PYTHON}"
    "${UNITREE_DIR}/deploy/deploy_real/deploy_real_g1_armhack_walk.py"
    --self-test
    --policy "${POLICY_PATH}"
    --poses "${POSE_PATH}"
    --contract "${CONTRACT_PATH}"
    --config "${UNITREE_DIR}/deploy/deploy_real/configs/${CONFIG}"
    --pose-name "${POSE_NAME}"
    --command-mode "${COMMAND_MODE}"
    --fixed-command "${CMD_VX}" "${CMD_VY}" "${CMD_WZ}"
    --joystick-lin-x-range ${JOYSTICK_LIN_X_RANGE//[\[\],]/ }
    --joystick-lin-y-range ${JOYSTICK_LIN_Y_RANGE//[\[\],]/ }
    --joystick-yaw-range ${JOYSTICK_YAW_RANGE//[\[\],]/ }
)

if is_true "${DRY_RUN}"; then
    "${SELF_TEST_COMMAND[@]}"
    printf 'Dry-run real command:'
    printf ' %q' "${UNITREE_PYTHON}" deploy_real_g1_armhack_walk.py "${NET}" "${CONFIG}"
    printf '\n'
    echo "DRY_RUN 完成：未初始化 DDS，也未向机器人发送命令。"
    exit 0
fi

[[ "${CONFIRM_REAL_ROBOT}" == "I_UNDERSTAND" ]] || {
    echo "拒绝启动：真机必须设置 CONFIRM_REAL_ROBOT=I_UNDERSTAND。" >&2
    exit 2
}
[[ -t 0 ]] || { echo "Error: 真机模式必须从交互终端启动。" >&2; exit 1; }
if [[ "${COMMAND_MODE}" == "joystick" && ! -r "${JOYSTICK_DEVICE}" ]]; then
    echo "Error: Joystick 设备不存在或不可读: ${JOYSTICK_DEVICE}" >&2
    exit 1
fi
if ! "${UNITREE_PYTHON}" -c \
    'import cyclonedds, numpy, onnxruntime, torch, yaml; from unitree_sdk2py.core.channel import ChannelFactoryInitialize' \
    >/dev/null 2>&1; then
    echo "Error: 真机 Python 缺少 cyclonedds、unitree_sdk2py、onnxruntime、torch、numpy 或 PyYAML。" >&2
    exit 1
fi
"${SELF_TEST_COMMAND[@]}"

echo "安全前提：必须使用吊架，现场人员持有急停/遥控器，周围无人。"
echo "两次 ENTER 分别授权移动双臂和启动零速度 policy；SPACE 切换双臂姿态，fixed 模式按 V 切换速度。"
cd "${UNITREE_DIR}/deploy/deploy_real"
exec "${UNITREE_PYTHON}" deploy_real_g1_armhack_walk.py "${NET}" "${CONFIG}"
