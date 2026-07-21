#!/usr/bin/env bash
# 使用 use/ 中的 ArmHack Walk model_10990 ONNX 进行可移植 MuJoCo sim2sim。

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POLICY_PATH=${POLICY_PATH:-"${ROOT_DIR}/use/armhack_walk_model_10990.onnx"}
POSE_PATH=${POSE_PATH:-"${ROOT_DIR}/legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"}
CONTRACT_PATH=${CONTRACT_PATH:-"${ROOT_DIR}/legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"}
POSE_NAME=${POSE_NAME:-pos2_down}
FIXED_COMMAND=${FIXED_COMMAND:-'[0.35,0.0,0.0]'}
START_ACTIVE=${START_ACTIVE:-True}
USE_GLFW=${USE_GLFW:-False}
REAL_TIME=${REAL_TIME:-${USE_GLFW}}
SIMULATION_DURATION=${SIMULATION_DURATION:-5.0}
SEED=${SEED:-20260721}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/outputs/use_models/armhack_walk"}
METRICS_PATH=${METRICS_PATH:-"${RESULTS_ROOT}/seed_${SEED}/metrics.json"}
EXPECTED_POLICY_SHA256=${EXPECTED_POLICY_SHA256:-b052c3b0583834a742ea59e736d55c3c9bafabb75f1d4fae65980166d4a895aa}

if [[ -z "${UNITREE_PYTHON:-}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/gmr/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            UNITREE_PYTHON="${candidate}"
            break
        fi
    done
fi

[[ -n "${UNITREE_PYTHON:-}" && -x "${UNITREE_PYTHON}" ]] || {
    echo "Error: UNITREE_PYTHON 未设置或不可执行: ${UNITREE_PYTHON:-<unset>}" >&2
    exit 1
}
for path in "${POLICY_PATH}" "${POSE_PATH}" "${CONTRACT_PATH}"; do
    [[ -f "${path}" ]] || { echo "Error: 文件不存在: ${path}" >&2; exit 1; }
done
actual_sha=$(sha256sum "${POLICY_PATH}" | awk '{print $1}')
[[ "${actual_sha}" == "${EXPECTED_POLICY_SHA256}" ]] || {
    echo "Error: Walk ONNX SHA-256 不匹配: ${actual_sha}" >&2
    exit 1
}
"${UNITREE_PYTHON}" -c 'import mujoco, numpy, onnxruntime, torch, yaml' >/dev/null
mkdir -p "$(dirname "${METRICS_PATH}")"

export G1_AMP_ARMHACK_WALK_ENABLE=True
export G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_PATH}"
export G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_PATH}"
export G1_AMP_ARMHACK_WALK_POSE_NAME="${POSE_NAME}"
export G1_AMP_ARMHACK_WALK_FIXED_COMMAND="${FIXED_COMMAND}"
export G1_AMP_ARMHACK_WALK_START_ACTIVE="${START_ACTIVE}"

echo "[ArmHack Walk use/ ONNX] policy=${POLICY_PATH} pose=${POSE_NAME} duration=${SIMULATION_DURATION}s"
UNITREE_PYTHON="${UNITREE_PYTHON}" \
POLICY_PATH="${POLICY_PATH}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" REAL_TIME="${REAL_TIME}" \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
METRICS_PATH="${METRICS_PATH}" \
CMD_INIT='[0.0,0.0,0.0]' RANDOM_COMMANDS=False \
COMMAND_MODE=independent COMMAND_RAMP=True \
COMMAND_MAX_LINEAR_ACCEL=0.5 COMMAND_MAX_YAW_ACCEL=0.8 \
TASK_TRACE_ENABLE=False TORSO_TRACE_ENABLE=False \
bash "${ROOT_DIR}/scripts/sim2sim_g1_amp_mujoco.sh"

echo "[ArmHack Walk use/ ONNX] PASS metrics=${METRICS_PATH}"
