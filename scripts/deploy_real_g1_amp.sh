#!/usr/bin/env bash
# Launch Unitree G1 29DoF AMP real-robot deployment after explicit operator confirmation.
# Usage:
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 bash scripts/deploy_real_g1_amp.sh
#   DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_amp.sh
#
# Environment variables:
#   UNITREE_PYTHON          : Python executable from unitree_sim2sim2real/.envrc conda env.
#   NET                     : Network interface passed to Unitree DDS, e.g. eth0.
#   CONFIG                  : Real deployment YAML filename (default: g1_amp.yaml).
#   POLICY_PATH             : TorchScript policy exported by scripts/export_g1_amp_policy.sh.
#   COMMAND_MODE            : remote or fixed. remote reads the wireless controller; fixed uses CMD_INIT.
#   CMD_INIT                : Fixed physical command [lin_x, lin_y, yaw_rate], e.g. '[0.3, 0.0, 0.0]'.
#   RUN_DURATION            : Seconds to run the policy after automatic rollout. 0 means unlimited.
#   COMMAND_RAMP            : True/False. Defaults to True for fixed mode and False for remote mode.
#   COMMAND_MAX_LINEAR_ACCEL: Fixed/remote command ramp limit in m/s^2.
#   COMMAND_MAX_YAW_ACCEL   : Fixed/remote command ramp limit in rad/s^2.
#   RELEASE_MOTION_MODE     : True releases Unitree high-level motion mode before low-level commands.
#   HANDOFF_MODE            : zero or stand. stand skips zero-torque wait and moves from current posture.
#   RECOVER_NATIVE_ON_EXIT  : True switches back to Unitree native ai_sport after AMP exits.
#   NATIVE_RECOVER_MODE     : service/damp/stand/locomotion. locomotion requests Damp -> StandUp -> 801/802.
#   CONFIRM_REAL_ROBOT      : Must equal I_UNDERSTAND for non-dry-run deployment.
#   DRY_RUN                 : True prints the resolved command without sending DDS commands.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 只打印将要执行的命令，不连接真机：
DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_amp.sh

# 2. 真机遥控器模式部署，需要操作员确认：
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 bash scripts/deploy_real_g1_amp.sh

# 3. 固定 0.3m/s 前进短测：
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=fixed CMD_INIT='[0.3, 0.0, 0.0]' RUN_DURATION=10 bash scripts/deploy_real_g1_amp.sh

# 4. 从 Unitree high-level 站立姿态平滑切到 AMP 默认姿态，再自动进入固定策略：
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 HANDOFF_MODE=stand COMMAND_MODE=fixed CMD_INIT='[0.3, 0.0, 0.0]' RUN_DURATION=10 bash scripts/deploy_real_g1_amp.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
NET=${NET:-}
CONFIG=${CONFIG:-g1_amp.yaml}
POLICY_PATH=${POLICY_PATH:-${ROOT_DIR}/outputs/baseline/exported/policy.pt}
COMMAND_MODE=${COMMAND_MODE:-remote}
CMD_INIT=${CMD_INIT:-'[0.0, 0.0, 0.0]'}
RUN_DURATION=${RUN_DURATION:-0.0}
if [[ -z "${COMMAND_RAMP+x}" ]]; then
    if [[ "${COMMAND_MODE}" == "fixed" ]]; then
        COMMAND_RAMP=True
    else
        COMMAND_RAMP=False
    fi
fi
COMMAND_MAX_LINEAR_ACCEL=${COMMAND_MAX_LINEAR_ACCEL:-0.5}
COMMAND_MAX_YAW_ACCEL=${COMMAND_MAX_YAW_ACCEL:-0.8}
RELEASE_MOTION_MODE=${RELEASE_MOTION_MODE:-True}
HANDOFF_MODE=${HANDOFF_MODE:-zero}
DEFAULT_MOVE_S=${DEFAULT_MOVE_S:-2.0}
DEFAULT_HOLD_S=${DEFAULT_HOLD_S:-0.0}
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
CONFIRM_REAL_ROBOT=${CONFIRM_REAL_ROBOT:-}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" ]]
}

if [[ "${POLICY_PATH}" != /* ]]; then
    POLICY_PATH="${ROOT_DIR}/${POLICY_PATH}"
fi

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
if [[ ! -f "${POLICY_PATH}" ]]; then
    echo "Error: POLICY_PATH does not exist: ${POLICY_PATH}" >&2
    echo "Run: CHECKPOINT=<checkpoint> bash scripts/export_g1_amp_policy.sh" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export G1_AMP_POLICY_PATH="${POLICY_PATH}"
export G1_AMP_COMMAND_MODE="${COMMAND_MODE}"
export G1_AMP_CMD_INIT="${CMD_INIT}"
export G1_AMP_RUN_DURATION="${RUN_DURATION}"
export G1_AMP_COMMAND_RAMP="${COMMAND_RAMP}"
export G1_AMP_COMMAND_MAX_LINEAR_ACCEL="${COMMAND_MAX_LINEAR_ACCEL}"
export G1_AMP_COMMAND_MAX_YAW_ACCEL="${COMMAND_MAX_YAW_ACCEL}"
export G1_AMP_RELEASE_MOTION_MODE="${RELEASE_MOTION_MODE}"
export G1_AMP_HANDOFF_MODE="${HANDOFF_MODE}"
export G1_AMP_WAIT_FOR_BUTTON_A=False
export G1_AMP_DEFAULT_MOVE_S="${DEFAULT_MOVE_S}"
export G1_AMP_DEFAULT_HOLD_S="${DEFAULT_HOLD_S}"
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
echo "  Real G1 AMP Deploy Launcher"
echo "====================================="
echo "Python      : ${UNITREE_PYTHON}"
echo "Net         : ${NET}"
echo "Config      : ${CONFIG}"
echo "Policy      : ${POLICY_PATH}"
echo "Command Mode: ${COMMAND_MODE}"
if [[ "${COMMAND_MODE}" == "remote" ]]; then
    echo "Command     : remote joystick (initial command is zero)"
else
    echo "Command     : ${CMD_INIT}"
fi
echo "Duration    : ${RUN_DURATION}"
echo "Ramp        : ${COMMAND_RAMP} lin_acc=${COMMAND_MAX_LINEAR_ACCEL} yaw_acc=${COMMAND_MAX_YAW_ACCEL}"
echo "Release Mode: ${RELEASE_MOTION_MODE}"
echo "Handoff Mode: ${HANDOFF_MODE}"
echo "Policy Start: automatic (no Button A wait)"
echo "Default Move: ${DEFAULT_MOVE_S}s"
echo "Default Hold: ${DEFAULT_HOLD_S}s"
echo "Recover Nat.: ${RECOVER_NATIVE_ON_EXIT} mode=${NATIVE_RECOVER_MODE} target=${NATIVE_TARGET_MODE}"
echo "RecoverStrict: ${NATIVE_RECOVER_STRICT}"
echo "Dry Run     : ${DRY_RUN}"
echo "====================================="

RUN_COMMAND=("${UNITREE_PYTHON}" deploy_real_g1_amp.py "${NET}" "${CONFIG}")

if is_true "${DRY_RUN}"; then
    printf 'Dry-run command:'
    printf ' %q' "${RUN_COMMAND[@]}"
    printf '\n'
    exit 0
fi

cd "${UNITREE_DIR}/deploy/deploy_real"
"${RUN_COMMAND[@]}"
