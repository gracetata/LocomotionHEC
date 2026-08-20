#!/usr/bin/env bash
# Real-time interactive MuJoCo visualization for the final ArmHack two-goal Walk policy.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POLICY_PATH=${POLICY_PATH:-"${ROOT_DIR}/checkpoint/walk/armhack_two_goal_20260811/policy.onnx"}
POSE_PATH=${POSE_PATH:-"${ROOT_DIR}/legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"}
CONTRACT_PATH=${CONTRACT_PATH:-"${ROOT_DIR}/legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"}
POSE_NAME=${POSE_NAME:-pos2_down}
ARM_POSE_TRANSITION_S=${ARM_POSE_TRANSITION_S:-2.0}
SIMULATION_DURATION=${SIMULATION_DURATION:-600.0}
UNITREE_PYTHON=${UNITREE_PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/outputs/armhack_two_goal_keyboard"}
EXPECTED_POLICY_SHA256="011c4bbf47846285328045967e45b78274d3c81c7cff315fc19b5c9aae095d5b"

die() { echo "Error: $*" >&2; exit 1; }
[[ -x "${UNITREE_PYTHON}" ]] || die "gmr Python 不可执行: ${UNITREE_PYTHON}"
for path in "${POLICY_PATH}" "${POSE_PATH}" "${CONTRACT_PATH}"; do
    [[ -f "${path}" ]] || die "文件不存在: ${path}"
done
actual_sha=$(sha256sum "${POLICY_PATH}" | awk '{print $1}')
[[ "${actual_sha}" == "${EXPECTED_POLICY_SHA256}" ]] || die "policy.pt SHA-256 不匹配: ${actual_sha}"
PYTHONNOUSERSITE=1 "${UNITREE_PYTHON}" -c 'import glfw, mujoco, numpy, onnxruntime, yaml' \
    || die "env_isaaclab 环境缺少 MuJoCo/ONNX Runtime 依赖"
mkdir -p "${RESULTS_ROOT}"

export PYTHONNOUSERSITE=1
export G1_AMP_ARMHACK_WALK_ENABLE=True
export G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_PATH}"
export G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_PATH}"
export G1_AMP_ARMHACK_WALK_POSE_NAME="${POSE_NAME}"
export G1_AMP_ARMHACK_WALK_POSE_TRANSITION_S="${ARM_POSE_TRANSITION_S}"
export G1_AMP_ARMHACK_WALK_FIXED_COMMAND='[0.0,0.0,0.0]'
export G1_AMP_ARMHACK_WALK_START_ACTIVE=False

echo "============================================================"
echo " ArmHack Walk 双目标策略：MuJoCo 键盘实时可视化"
echo "============================================================"
echo "速度: W/S 前后, A/D 左右, Q/E 转向, 0 立即零速"
echo "预设: 1 慢速前进, 2 左移, 3 左转, 4 斜移, 5 右移, 6 右转, 7 正常前进"
echo "双臂: SPACE/P 循环; Z=靠后, X=下垂, C=靠前"
echo "时间: 强制真实 1.0x；终端每 2 秒输出一次 RTF，未启用慢放"
echo "============================================================"

UNITREE_PYTHON="${UNITREE_PYTHON}" \
POLICY_PATH="${POLICY_PATH}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW=True REAL_TIME=True RENDER_FPS=60 REALTIME_STATUS_INTERVAL_S=2 \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
METRICS_PATH="${RESULTS_ROOT}/mujoco_metrics.json" \
CMD_INIT='[0.0,0.0,0.0]' RANDOM_COMMANDS=False COMMAND_MODE=keyboard \
COMMAND_RAMP=True COMMAND_SMOOTHING_TAU=0.15 \
COMMAND_MAX_LINEAR_ACCEL=1.5 COMMAND_MAX_YAW_ACCEL=2.0 \
CMD_LIN_X_RANGE='[-0.2,0.6]' CMD_LIN_Y_RANGE='[-0.3,0.3]' \
CMD_YAW_RANGE='[-0.5187280216217041,0.6]' \
KEYBOARD_LINEAR_STEP=0.05 KEYBOARD_YAW_STEP=0.05 \
TASK_TRACE_ENABLE=False TORSO_TRACE_ENABLE=False \
bash "${ROOT_DIR}/scripts/sim2sim_g1_amp_mujoco.sh"
