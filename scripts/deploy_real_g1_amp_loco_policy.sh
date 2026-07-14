#!/usr/bin/env bash
# Deploy a G1 AMP policy from Unitree high-level locomotion into low-level rt/lowcmd control.
# Preferred handoff is MotionSwitcherClient.ReleaseMode(): pre-warm lowcmd current-pose hold,
# release high-level motion mode, interpolate to AMP default pose, then start the policy.
#
# Usage:
#   DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_amp_loco_policy.sh
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 bash scripts/deploy_real_g1_amp_loco_policy.sh \
#       legged_lab/logs/rsl_rl/g1_amp/2026-06-01_02-23-30_finetune_nav2_stage2_realwindows_follow/model_5997.pt
#
# Environment variables:
#   NET                       : Unitree DDS network interface, e.g. enp11s0.
#   CHECKPOINT / MODEL_PATH   : RSL-RL checkpoint model_*.pt. Positional arg has highest priority.
#   POLICY_PATH               : TorchScript policy.pt. If unset and CHECKPOINT is given, uses <run>/exported/policy.pt.
#   EXPORT_POLICY             : auto/True/False. auto exports only when policy.pt is missing.
#   FORCE_EXPORT              : True forces checkpoint -> TorchScript export.
#   HANDOFF_ROUTE             : release or zero. Default release; zero uses legacy low-level zero-torque start flow.
#   COMMAND_MODE              : fixed or remote. Default remote; remote reads the wireless joystick and starts from zero.
#   CMD_INIT                  : Fixed physical command [lin_x, lin_y, yaw_rate]. Only used when COMMAND_MODE=fixed; default '[0.0, 0.0, 0.0]'.
#   RUN_DURATION              : Seconds to run after policy start. Default 3.0; 0 means unlimited.
#   DEFAULT_MOVE_S            : Interpolation time from current posture to AMP default. Default 0.5.
#   DEFAULT_HOLD_S            : Hold default pose before policy rollout. Default 0.0.
#   RELEASE_HANDOFF_WARMUP_S  : Current-pose lowcmd pre-warm before ReleaseMode. Default 0.3.
#   RELEASE_HANDOFF_AFTER_S   : Current-pose lowcmd hold after ReleaseMode. Default 0.2.
#   RECOVER_NATIVE_ON_EXIT    : True switches back to Unitree native mode after AMP exits. Default False.
#   NATIVE_RECOVER_MODE       : service/damp/stand/locomotion. Default damp; locomotion waits for stable StandUp, then requests 802/801/500.
#   NATIVE_RECOVER_STRICT     : True fails if requested native locomotion is not confirmed. Default False.
#   CONFIRM_REAL_ROBOT        : Must equal I_UNDERSTAND for non-dry-run deployment.
#   DRY_RUN                   : True prints resolved export/deploy commands without sending robot commands.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 检查默认 stage2 checkpoint 的导出/部署命令，不连接真机。
DRY_RUN=True NET=enp11s0 bash scripts/deploy_real_g1_amp_loco_policy.sh

# 2. 从当前高层运控/站立姿态 ReleaseMode 接入 stage2 policy；到 AMP default 后不等按键，直接进入策略；速度由遥控器摇杆给定，初始为零。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 RUN_DURATION=3000 \
    bash scripts/deploy_real_g1_amp_loco_policy.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500/model_8997.pt

# 3. 同一路径，固定 0.1m/s 短测。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=fixed CMD_INIT='[0.1, 0.0, 0.0]' RUN_DURATION=3 \
    bash scripts/deploy_real_g1_amp_loco_policy.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-01_02-23-30_finetune_nav2_stage2_realwindows_follow/model_5997.pt

CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=fixed CMD_INIT='[0.0, 0.0, 0.0]' RUN_DURATION=50 \
    bash scripts/deploy_real_g1_amp_loco_policy.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500/model_8997.pt

CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=fixed CMD_INIT='[0.9, 0.0, 0.0]' RUN_DURATION=50 \
    bash scripts/deploy_real_g1_amp_loco_policy.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500/model_8997.pt

# 4. AMP 策略退出后自动切回宇树本体链路；若 locomotion 未确认，默认停在最后确认的本体状态并打印 warning。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 RUN_DURATION=30 RECOVER_NATIVE_ON_EXIT=True NATIVE_RECOVER_MODE=locomotion \
    bash scripts/deploy_real_g1_amp_loco_policy.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-01_02-23-30_finetune_nav2_stage2_realwindows_follow/model_5997.pt
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}

DEFAULT_CHECKPOINT="${ROOT_DIR}/legged_lab/logs/rsl_rl/g1_amp/2026-06-01_02-23-30_finetune_nav2_stage2_realwindows_follow/model_5997.pt"

NET=${NET:-}
ROBOT_IP=${ROBOT_IP:-192.168.123.161}
CONFIG=${CONFIG:-g1_amp.yaml}
SOURCE_ARG=${1:-}
MODEL_SOURCE=${SOURCE_ARG:-${MODEL_PATH:-${CHECKPOINT:-${DEFAULT_CHECKPOINT}}}}
POLICY_PATH=${POLICY_PATH:-}
EXPORT_POLICY=${EXPORT_POLICY:-auto}
FORCE_EXPORT=${FORCE_EXPORT:-False}
HANDOFF_ROUTE=${HANDOFF_ROUTE:-release}
COMMAND_MODE=${COMMAND_MODE:-remote}
CMD_INIT=${CMD_INIT:-'[0.0, 0.0, 0.0]'}
RUN_DURATION=${RUN_DURATION:-3.0}
COMMAND_RAMP=${COMMAND_RAMP:-True}
COMMAND_MAX_LINEAR_ACCEL=${COMMAND_MAX_LINEAR_ACCEL:-0.3}
COMMAND_MAX_YAW_ACCEL=${COMMAND_MAX_YAW_ACCEL:-0.5}
WAIT_FOR_BUTTON_A=False
DEFAULT_MOVE_S=${DEFAULT_MOVE_S:-0.5}
DEFAULT_HOLD_S=${DEFAULT_HOLD_S:-0.0}
RELEASE_HANDOFF_WARMUP_S=${RELEASE_HANDOFF_WARMUP_S:-0.3}
RELEASE_HANDOFF_AFTER_S=${RELEASE_HANDOFF_AFTER_S:-0.2}
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

resolve_repo_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s' "${path}"
    else
        printf '%s/%s' "${ROOT_DIR}" "${path}"
    fi
}

if [[ -z "${NET}" ]]; then
    echo "Error: set NET to the Unitree network interface, e.g. NET=enp11s0." >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi
if ! is_true "${DRY_RUN}" && [[ "${CONFIRM_REAL_ROBOT}" != "I_UNDERSTAND" ]]; then
    echo "Refusing to start real-robot deployment without CONFIRM_REAL_ROBOT=I_UNDERSTAND." >&2
    exit 2
fi

SOURCE_KIND=checkpoint
CHECKPOINT_PATH=""
if [[ -n "${POLICY_PATH}" ]]; then
    POLICY_PATH=$(resolve_repo_path "${POLICY_PATH}")
    SOURCE_KIND=policy
else
    MODEL_SOURCE=$(resolve_repo_path "${MODEL_SOURCE}")
    if [[ "$(basename "${MODEL_SOURCE}")" == "policy.pt" || "${MODEL_SOURCE}" == */exported/*.pt ]]; then
        POLICY_PATH="${MODEL_SOURCE}"
        SOURCE_KIND=policy
    else
        CHECKPOINT_PATH="${MODEL_SOURCE}"
        POLICY_PATH="$(dirname "${CHECKPOINT_PATH}")/exported/policy.pt"
    fi
fi

if [[ "${SOURCE_KIND}" == "checkpoint" && ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Error: checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ "${HANDOFF_ROUTE}" == "release" ]]; then
    RELEASE_MOTION_MODE=True
    HANDOFF_MODE=stand
elif [[ "${HANDOFF_ROUTE}" == "zero" ]]; then
    RELEASE_MOTION_MODE=False
    HANDOFF_MODE=zero
else
    echo "Error: HANDOFF_ROUTE must be release or zero, got: ${HANDOFF_ROUTE}" >&2
    exit 1
fi

NEED_EXPORT=False
if [[ "${SOURCE_KIND}" == "checkpoint" ]]; then
    if is_true "${FORCE_EXPORT}"; then
        NEED_EXPORT=True
    elif [[ "${EXPORT_POLICY,,}" == "auto" && ! -f "${POLICY_PATH}" ]]; then
        NEED_EXPORT=True
    elif is_true "${EXPORT_POLICY}"; then
        NEED_EXPORT=True
    fi
fi

if [[ "${SOURCE_KIND}" == "policy" && ! -f "${POLICY_PATH}" ]]; then
    echo "Error: POLICY_PATH does not exist: ${POLICY_PATH}" >&2
    exit 1
fi
if [[ "${SOURCE_KIND}" == "checkpoint" && "${NEED_EXPORT}" == "False" && ! -f "${POLICY_PATH}" ]]; then
    echo "Error: exported policy does not exist: ${POLICY_PATH}" >&2
    echo "Set EXPORT_POLICY=True or FORCE_EXPORT=True." >&2
    exit 1
fi

EXPORT_DIR=$(dirname "${POLICY_PATH}")
ONNX_OUT=${ONNX_OUT:-${EXPORT_DIR}/policy.onnx}
METADATA_OUT=${METADATA_OUT:-${EXPORT_DIR}/policy.deploy.json}

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "====================================="
echo "  G1 AMP Loco -> Policy Handoff"
echo "====================================="
echo "Net             : ${NET}"
echo "Robot IP        : ${ROBOT_IP}"
echo "Config          : ${CONFIG}"
echo "Source Kind     : ${SOURCE_KIND}"
echo "Checkpoint      : ${CHECKPOINT_PATH:-<none>}"
echo "Policy          : ${POLICY_PATH}"
echo "Need Export     : ${NEED_EXPORT}"
echo "Handoff Route   : ${HANDOFF_ROUTE} (release=${RELEASE_MOTION_MODE}, mode=${HANDOFF_MODE})"
echo "Command Mode    : ${COMMAND_MODE}"
if [[ "${COMMAND_MODE}" == "remote" ]]; then
    echo "Command         : remote joystick (initial command is zero)"
else
    echo "Command         : ${CMD_INIT}"
fi
echo "Run Duration    : ${RUN_DURATION}"
echo "Ramp            : ${COMMAND_RAMP} lin_acc=${COMMAND_MAX_LINEAR_ACCEL} yaw_acc=${COMMAND_MAX_YAW_ACCEL}"
echo "Policy Start    : automatic (no Button A wait)"
echo "Default Move    : ${DEFAULT_MOVE_S}s"
echo "Default Hold    : ${DEFAULT_HOLD_S}s"
echo "Release Hold    : warmup=${RELEASE_HANDOFF_WARMUP_S}s after=${RELEASE_HANDOFF_AFTER_S}s"
echo "Recover Native  : ${RECOVER_NATIVE_ON_EXIT} mode=${NATIVE_RECOVER_MODE} target=${NATIVE_TARGET_MODE} strict=${NATIVE_RECOVER_STRICT}"
echo "Smoke Test      : ${SMOKE_TEST_POLICY}"
echo "Dry Run         : ${DRY_RUN}"
echo "====================================="

if [[ "${NEED_EXPORT}" == "True" ]]; then
    EXPORT_COMMAND=(
        CHECKPOINT="${CHECKPOINT_PATH}"
        JIT_OUT="${POLICY_PATH}"
        ONNX_OUT="${ONNX_OUT}"
        METADATA_OUT="${METADATA_OUT}"
        ISAACLAB_PYTHON="${ISAACLAB_PYTHON}"
        bash "${ROOT_DIR}/scripts/export_g1_amp_policy.sh"
    )
    if is_true "${DRY_RUN}"; then
        printf 'Dry-run export command:'
        printf ' %q' "${EXPORT_COMMAND[@]}"
        printf '\n'
    else
        env CHECKPOINT="${CHECKPOINT_PATH}" JIT_OUT="${POLICY_PATH}" ONNX_OUT="${ONNX_OUT}" \
            METADATA_OUT="${METADATA_OUT}" ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" \
            bash "${ROOT_DIR}/scripts/export_g1_amp_policy.sh"
    fi
fi

if is_true "${DRY_RUN}"; then
    printf 'Dry-run deploy command:'
    printf ' %q' env \
        POLICY_PATH="${POLICY_PATH}" COMMAND_MODE="${COMMAND_MODE}" CMD_INIT="${CMD_INIT}" RUN_DURATION="${RUN_DURATION}" \
        COMMAND_RAMP="${COMMAND_RAMP}" COMMAND_MAX_LINEAR_ACCEL="${COMMAND_MAX_LINEAR_ACCEL}" COMMAND_MAX_YAW_ACCEL="${COMMAND_MAX_YAW_ACCEL}" \
        DEFAULT_MOVE_S="${DEFAULT_MOVE_S}" DEFAULT_HOLD_S="${DEFAULT_HOLD_S}" \
        RELEASE_MOTION_MODE="${RELEASE_MOTION_MODE}" HANDOFF_MODE="${HANDOFF_MODE}" \
        G1_AMP_WAIT_FOR_BUTTON_A="${WAIT_FOR_BUTTON_A}" G1_AMP_DEFAULT_MOVE_S="${DEFAULT_MOVE_S}" G1_AMP_DEFAULT_HOLD_S="${DEFAULT_HOLD_S}" \
        G1_AMP_RELEASE_HANDOFF_WARMUP_S="${RELEASE_HANDOFF_WARMUP_S}" G1_AMP_RELEASE_HANDOFF_AFTER_S="${RELEASE_HANDOFF_AFTER_S}" \
        G1_AMP_RECOVER_NATIVE_ON_EXIT="${RECOVER_NATIVE_ON_EXIT}" G1_AMP_NATIVE_RECOVER_MODE="${NATIVE_RECOVER_MODE}" \
        G1_AMP_NATIVE_RECOVER_STRICT="${NATIVE_RECOVER_STRICT}" \
        G1_AMP_NATIVE_TARGET_MODE="${NATIVE_TARGET_MODE}" G1_AMP_NATIVE_START_SERVICE="${NATIVE_START_SERVICE}" \
        G1_AMP_NATIVE_LOCO_SERVICE="${NATIVE_LOCO_SERVICE}" G1_AMP_NATIVE_RECOVER_TIMEOUT_S="${NATIVE_RECOVER_TIMEOUT_S}" \
        G1_AMP_NATIVE_STANDUP_WAIT_S="${NATIVE_STANDUP_WAIT_S}" G1_AMP_NATIVE_MIN_STANDUP_S="${NATIVE_MIN_STANDUP_S}" \
        G1_AMP_NATIVE_START_FSM_IDS="${NATIVE_START_FSM_IDS}" \
        G1_AMP_NATIVE_LOCOMOTION_FSM_IDS="${NATIVE_LOCOMOTION_FSM_IDS}" \
        CONFIRM_REAL_ROBOT="${CONFIRM_REAL_ROBOT}" NET="${NET}" CONFIG="${CONFIG}" \
        bash "${ROOT_DIR}/scripts/deploy_real_g1_amp.sh"
    printf '\n'
    exit 0
fi

if is_true "${PING_ROBOT}"; then
    ping -c 1 -W 1 "${ROBOT_IP}" >/dev/null
    echo "Ping ${ROBOT_IP}: ok"
fi

if is_true "${SMOKE_TEST_POLICY}"; then
    "${UNITREE_PYTHON}" - "${POLICY_PATH}" <<'PY'
import sys
import torch

policy_path = sys.argv[1]
policy = torch.jit.load(policy_path, map_location="cpu")
policy.eval()
obs = torch.zeros(1, 96)
with torch.inference_mode():
    action = policy(obs)
shape = tuple(action.shape)
print(f"TorchScript smoke: {policy_path} -> {shape}")
if shape != (1, 29):
    raise SystemExit(f"Unexpected policy output shape: {shape}, expected (1, 29)")
PY
fi

export POLICY_PATH
export COMMAND_MODE
export CMD_INIT
export RUN_DURATION
export COMMAND_RAMP
export COMMAND_MAX_LINEAR_ACCEL
export COMMAND_MAX_YAW_ACCEL
export DEFAULT_MOVE_S
export DEFAULT_HOLD_S
export RELEASE_MOTION_MODE
export HANDOFF_MODE
export G1_AMP_WAIT_FOR_BUTTON_A="${WAIT_FOR_BUTTON_A}"
export G1_AMP_DEFAULT_MOVE_S="${DEFAULT_MOVE_S}"
export G1_AMP_DEFAULT_HOLD_S="${DEFAULT_HOLD_S}"
export G1_AMP_RELEASE_HANDOFF_WARMUP_S="${RELEASE_HANDOFF_WARMUP_S}"
export G1_AMP_RELEASE_HANDOFF_AFTER_S="${RELEASE_HANDOFF_AFTER_S}"
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
export CONFIRM_REAL_ROBOT
export NET
export CONFIG

bash "${ROOT_DIR}/scripts/deploy_real_g1_amp.sh"
