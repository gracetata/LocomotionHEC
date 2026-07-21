#!/usr/bin/env bash
# ArmHack Stand 真机入口：ENTER 从原生阻尼/待机进入 policy 调试模式，初始化后 SPACE 切换双臂。
#
# 离线自检（不初始化 DDS，不连接机器人）：
#   DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_armhack_stand.sh
#
# 真机（必须在有吊架、急停和现场操作员时执行）：
#   UNITREE_PYTHON=/path/to/unitree-sdk-env/bin/python \
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 \
#   bash scripts/deploy_real_g1_armhack_stand.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"

NET=${NET:-}
CONFIG=${CONFIG:-g1_amp.yaml}
POLICY_PATH=${POLICY_PATH:-${ROOT_DIR}/use/armhack_stand_model_2999.onnx}
POLICY_METADATA_PATH=${POLICY_METADATA_PATH:-${ROOT_DIR}/use/armhack_stand_model_2999.deploy.json}
PRESET_PATH=${PRESET_PATH:-${ROOT_DIR}/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json}
EXPECTED_POLICY_SHA256=${EXPECTED_POLICY_SHA256:-354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f}
TRANSITION_S=${TRANSITION_S:-7.5}
RUN_DURATION=${RUN_DURATION:-0.0}
JOINT_PRINT_HZ=${JOINT_PRINT_HZ:-1.0}
LOWSTATE_TIMEOUT_S=${LOWSTATE_TIMEOUT_S:-0.20}
MAX_TILT_RAD=${MAX_TILT_RAD:-0.60}
DAMPING_EXIT_S=${DAMPING_EXIT_S:-1.0}
DAMPING_ARM_MAX_ERROR_RAD=${DAMPING_ARM_MAX_ERROR_RAD:-0.45}
DAMPING_BODY_MAX_ERROR_RAD=${DAMPING_BODY_MAX_ERROR_RAD:-0.50}
DAMPING_UPRIGHT_MAX_TILT_RAD=${DAMPING_UPRIGHT_MAX_TILT_RAD:-0.20}
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

for path_var in POLICY_PATH POLICY_METADATA_PATH PRESET_PATH; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${ROOT_DIR}/${value}"
    fi
done

if [[ -z "${UNITREE_PYTHON:-}" || ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON 未设置或不可执行: ${UNITREE_PYTHON:-<unset>}" >&2
    exit 1
fi
if [[ ! -f "${POLICY_PATH}" ]]; then
    echo "Error: Stand policy 不存在: ${POLICY_PATH}" >&2
    exit 1
fi
if [[ ! -f "${PRESET_PATH}" ]]; then
    echo "Error: 双臂预设不存在: ${PRESET_PATH}" >&2
    exit 1
fi
if [[ ! -f "${POLICY_METADATA_PATH}" ]]; then
    echo "Error: 最新 Stand policy 元数据不存在: ${POLICY_METADATA_PATH}" >&2
    exit 1
fi
if [[ -z "${NET}" ]]; then
    echo "Error: 必须设置连接 G1 的网卡，例如 NET=enp11s0。" >&2
    exit 1
fi

metadata_checkpoint_sha=$("${UNITREE_PYTHON}" - "${POLICY_METADATA_PATH}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("obs_dim") != 96 or payload.get("action_dim") != 29:
    raise SystemExit("metadata actor contract must be obs_dim=96 and action_dim=29")
print(payload.get("checkpoint_sha256", ""))
PY
)
if [[ -n "${metadata_checkpoint_sha}" && "${metadata_checkpoint_sha}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "Error: policy 元数据中的 checkpoint SHA-256 不匹配。" >&2
    echo "  expected: ${EXPECTED_CHECKPOINT_SHA256}" >&2
    echo "  actual  : ${metadata_checkpoint_sha}" >&2
    exit 1
fi

actual_sha256=$(sha256sum "${POLICY_PATH}" | awk '{print $1}')
if [[ -n "${EXPECTED_POLICY_SHA256}" && "${actual_sha256}" != "${EXPECTED_POLICY_SHA256}" ]]; then
    echo "Error: policy SHA-256 不匹配。" >&2
    echo "  expected: ${EXPECTED_POLICY_SHA256}" >&2
    echo "  actual  : ${actual_sha256}" >&2
    echo "若有意使用新模型，请同时显式设置 EXPECTED_POLICY_SHA256。" >&2
    exit 1
fi

export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
UNITREE_ENV_PREFIX=$(cd "$(dirname "${UNITREE_PYTHON}")/.." && pwd)
export LD_LIBRARY_PATH="${UNITREE_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# 50 Hz 单策略推理不需要 ORT/BLAS 建立大线程池；固定单线程也避免与 DDS 控制线程争抢。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export G1_AMP_POLICY_PATH="${POLICY_PATH}"
export G1_AMP_POLICY_RUNTIME=onnx
export G1_AMP_COMMAND_MODE=fixed
export G1_AMP_CMD_INIT='[0.0, 0.0, 0.0]'
export G1_AMP_RUN_DURATION="${RUN_DURATION}"
export G1_AMP_RELEASE_MOTION_MODE=True
export G1_AMP_HANDOFF_MODE=stand
export G1_AMP_WAIT_FOR_BUTTON_A=False
export G1_ARMHACK_STAND_PRESET_PATH="${PRESET_PATH}"
export G1_ARMHACK_STAND_TRANSITION_S="${TRANSITION_S}"
export G1_ARMHACK_STAND_JOINT_PRINT_HZ="${JOINT_PRINT_HZ}"
export G1_ARMHACK_STAND_LOWSTATE_TIMEOUT_S="${LOWSTATE_TIMEOUT_S}"
export G1_ARMHACK_STAND_MAX_TILT_RAD="${MAX_TILT_RAD}"
export G1_ARMHACK_STAND_DAMPING_EXIT_S="${DAMPING_EXIT_S}"
export G1_ARMHACK_STAND_DAMPING_ARM_MAX_ERROR_RAD="${DAMPING_ARM_MAX_ERROR_RAD}"
export G1_ARMHACK_STAND_DAMPING_BODY_MAX_ERROR_RAD="${DAMPING_BODY_MAX_ERROR_RAD}"
export G1_ARMHACK_STAND_DAMPING_UPRIGHT_MAX_TILT_RAD="${DAMPING_UPRIGHT_MAX_TILT_RAD}"
export G1_ARMHACK_STAND_CONFIRM="${CONFIRM_REAL_ROBOT}"

echo "============================================================"
echo "  ArmHack Stand Real-Robot Deployment"
echo "============================================================"
echo "Python       : ${UNITREE_PYTHON}"
echo "Net          : ${NET}"
echo "Config       : ${CONFIG}"
echo "Policy       : ${POLICY_PATH}"
echo "Policy SHA256: ${actual_sha256}"
echo "Policy meta  : ${POLICY_METADATA_PATH}"
echo "Source ckpt  : ${EXPECTED_CHECKPOINT_SHA256}"
echo "Arm presets  : ${PRESET_PATH}"
echo "Mode switch  : ENTER, native damping/standby -> policy-on debug"
echo "Auto init    : natural-down -> flat-default -> forward -> flat-default"
echo "Arm switch   : SPACE after auto init, minimum-jerk ${TRANSITION_S}s"
echo "Policy cmd   : [0, 0, 0] (forced)"
echo "Debug entry  : ENTER -> current-pose hold + MotionSwitcher.ReleaseMode() + actor"
echo "Damping gate : tilt<=${DAMPING_UPRIGHT_MAX_TILT_RAD} rad, body_err<=${DAMPING_BODY_MAX_ERROR_RAD} rad, arm_err<=${DAMPING_ARM_MAX_ERROR_RAD} rad"
echo "Target path  : direct default+action*scale (no position clipping / per-frame slew limit)"
echo "Inference CPU: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "Stop         : q / Ctrl-C / remote Select -> damping"
echo "Dry run      : ${DRY_RUN}"
echo "============================================================"

SELF_TEST_COMMAND=(
    "${UNITREE_PYTHON}"
    "${UNITREE_DIR}/deploy/deploy_real/deploy_real_g1_armhack_stand.py"
    --self-test
    --policy "${POLICY_PATH}"
    --presets "${PRESET_PATH}"
    --config "${UNITREE_DIR}/deploy/deploy_real/configs/${CONFIG}"
)

if is_true "${DRY_RUN}"; then
    "${SELF_TEST_COMMAND[@]}"
    printf 'Dry-run real command:'
    printf ' %q' "${UNITREE_PYTHON}" deploy_real_g1_armhack_stand.py "${NET}" "${CONFIG}"
    printf '\n'
    echo "DRY_RUN 完成：未初始化 DDS，也未向机器人发送任何命令。"
    exit 0
fi

if [[ "${CONFIRM_REAL_ROBOT}" != "I_UNDERSTAND" ]]; then
    echo "拒绝启动：真机模式必须设置 CONFIRM_REAL_ROBOT=I_UNDERSTAND。" >&2
    exit 2
fi
if [[ ! -t 0 ]]; then
    echo "Error: 真机模式必须从交互式终端启动，空格/q 急停键依赖 TTY。" >&2
    exit 1
fi

if ! "${UNITREE_PYTHON}" -c \
    'import cyclonedds, numpy, onnxruntime, torch, yaml; from unitree_sdk2py.core.channel import ChannelFactoryInitialize' \
    >/dev/null 2>&1; then
    echo "Error: UNITREE_PYTHON 缺少 cyclonedds、unitree_sdk2py、onnxruntime、torch、numpy 或 PyYAML。" >&2
    echo "请使用已经按 unitree_sim2sim2real/doc/setup_zh.md 安装完成的 Unitree 环境。" >&2
    exit 1
fi

"${SELF_TEST_COMMAND[@]}"

echo
echo "安全前提：机器人必须处于吊架保护，现场人员持有急停/遥控器，周围无人。"
echo "程序将先保持当前姿态，再调用 ReleaseMode 释放高层运控并确认进入低层调试控制状态。"
echo "启动后尚未初始化 DDS；确认自然下垂待机姿态后按一次 ENTER。"
echo "ENTER 后 actor 持续推理并自动完成初始化；完成后 SPACE 切换双臂，q 退出。"
echo

cd "${UNITREE_DIR}/deploy/deploy_real"
exec "${UNITREE_PYTHON}" deploy_real_g1_armhack_stand.py "${NET}" "${CONFIG}"
