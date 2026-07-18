#!/usr/bin/env bash
# Export/test/visualize the ArmHack Walk model_3999 policy in S3 G1 MuJoCo.
#
# Headless deterministic test:
#   USE_GLFW=False REAL_TIME=False SIMULATION_DURATION=20 \
#     bash scripts/val_mujoco_g1_armhack_walk.sh
# GUI visualization (SPACE toggles zero/fixed velocity):
#   USE_GLFW=True REAL_TIME=True bash scripts/val_mujoco_g1_armhack_walk.sh

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
DEFAULT_CHECKPOINT="${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkPerturbFinetune/2026-07-18_imported_walk_model3999_454c9bc0b5e3/model_3999.pt"
DEPLOY_DIR="${LEGGED_LAB_DIR}/deployment/armhack_walk/model_3999"
POSE_DATA="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"
CONTRACT_DATA="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"

CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-454c9bc0b5e38b2a9800c6faaa9e8ba6995f7d99bd3844155929a10a4fb8e2ff}
ONNX_PATH=${ONNX_PATH:-${DEPLOY_DIR}/walk_model3999.onnx}
TORCHSCRIPT_PATH=${TORCHSCRIPT_PATH:-${DEPLOY_DIR}/walk_model3999.pt}
METADATA_PATH=${METADATA_PATH:-${DEPLOY_DIR}/walk_model3999.deploy.json}
POSE_NAME=${POSE_NAME:-pos2_down}
FIXED_COMMAND=${FIXED_COMMAND:-'[0.35,0.0,0.0]'}
START_ACTIVE=${START_ACTIVE:-True}
SIMULATION_DURATION=${SIMULATION_DURATION:-20.0}
USE_GLFW=${USE_GLFW:-True}
REAL_TIME=${REAL_TIME:-True}
FORCE_EXPORT=${FORCE_EXPORT:-False}
ISAAC_PYTHON=${ISAAC_PYTHON:-/home/user/anaconda3/envs/env_isaaclab/bin/python}
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/user/anaconda3/envs/gmr/bin/python}
MUJOCO_CPU_THREADS=${MUJOCO_CPU_THREADS:-1}
REPORT_DIR=${REPORT_DIR:-${DEPLOY_DIR}/Local Test Reports}
METRICS_PATH=${METRICS_PATH:-${REPORT_DIR}/walk_model3999_${POSE_NAME}_metrics.json}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

for path_var in CHECKPOINT ONNX_PATH TORCHSCRIPT_PATH METADATA_PATH POSE_DATA CONTRACT_DATA REPORT_DIR METRICS_PATH; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${LEGGED_LAB_DIR}/${value}"
    fi
done

[[ -x "${UNITREE_PYTHON}" ]] || { echo "Error: UNITREE_PYTHON 不可执行: ${UNITREE_PYTHON}" >&2; exit 1; }
[[ -f "${POSE_DATA}" ]] || { echo "Error: Walk pose JSON 不存在: ${POSE_DATA}" >&2; exit 1; }
[[ -f "${CONTRACT_DATA}" ]] || { echo "Error: Walk deployment contract 不存在: ${CONTRACT_DATA}" >&2; exit 1; }

need_export=False
if is_true "${FORCE_EXPORT}" || [[ ! -f "${ONNX_PATH}" || ! -f "${TORCHSCRIPT_PATH}" || ! -f "${METADATA_PATH}" ]]; then
    need_export=True
fi
if is_true "${need_export}"; then
    [[ -x "${ISAAC_PYTHON}" ]] || { echo "Error: ISAAC_PYTHON 不可执行: ${ISAAC_PYTHON}" >&2; exit 1; }
    [[ -f "${CHECKPOINT}" ]] || {
        echo "Error: 缺少导出所需 checkpoint，且部署模型尚不完整: ${CHECKPOINT}" >&2
        exit 1
    }
    actual_checkpoint_sha=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
    [[ "${actual_checkpoint_sha}" == "${EXPECTED_CHECKPOINT_SHA256}" ]] || {
        echo "Error: model_3999 checkpoint SHA-256 不匹配。" >&2
        echo "expected=${EXPECTED_CHECKPOINT_SHA256}" >&2
        echo "actual=${actual_checkpoint_sha}" >&2
        exit 1
    }
    mkdir -p "${DEPLOY_DIR}"
    "${ISAAC_PYTHON}" "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
        --robot g1 \
        --checkpoint "${CHECKPOINT}" \
        --output "${ONNX_PATH}" \
        --jit-output "${TORCHSCRIPT_PATH}" \
        --metadata "${METADATA_PATH}" \
        --default-command 0.35 0.0 0.0
fi

for artifact in "${ONNX_PATH}" "${TORCHSCRIPT_PATH}" "${METADATA_PATH}"; do
    [[ -f "${artifact}" ]] || { echo "Error: Walk deployment artifact 不存在: ${artifact}" >&2; exit 1; }
done

"${UNITREE_PYTHON}" - "${ONNX_PATH}" "${TORCHSCRIPT_PATH}" "${POSE_DATA}" "${CONTRACT_DATA}" "${POSE_NAME}" "${FIXED_COMMAND}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

onnx_path, jit_path, pose_path, contract_path, pose_name, command_text = sys.argv[1:]
contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
command = np.asarray(yaml.safe_load(command_text), dtype=np.float32)
lower = np.asarray(contract["raw_command_component_bounds"]["min"], dtype=np.float32)
upper = np.asarray(contract["raw_command_component_bounds"]["max"], dtype=np.float32)
if command.shape != (3,) or not np.all(np.isfinite(command)) or np.any(command < lower) or np.any(command > upper):
    raise SystemExit(f"FIXED_COMMAND outside raw Nav2 CSV range: {command.tolist()} not in [{lower.tolist()}, {upper.tolist()}]")
poses = json.loads(Path(pose_path).read_text(encoding="utf-8"))["poses"]
if sum(entry["name"] == pose_name for entry in poses) != 1:
    raise SystemExit(f"POSE_NAME must select exactly one pose: {pose_name}")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
jit = torch.jit.load(jit_path, map_location="cpu").eval()
rng = np.random.default_rng(20260718)
for obs in [np.zeros((1, 96), dtype=np.float32)] + [rng.normal(0, 0.5, (1, 96)).astype(np.float32) for _ in range(4)]:
    onnx_action = session.run(["actions"], {"obs": obs})[0]
    with torch.inference_mode():
        jit_action = jit(torch.from_numpy(obs)).numpy()
    if onnx_action.shape != (1, 29) or not np.all(np.isfinite(onnx_action)):
        raise SystemExit("ONNX actor failed finite [1,29] check")
    if not np.allclose(onnx_action, jit_action, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"ONNX/TorchScript mismatch: max_abs={np.max(np.abs(onnx_action-jit_action))}")
print(f"[SELF-TEST PASS] actor 96->29; pose={pose_name}; command={command.tolist()} within raw Nav2 range")
PY

mkdir -p "${REPORT_DIR}"
echo "============================================================"
echo " ArmHack Walk model_3999 MuJoCo sim2sim"
echo "============================================================"
echo "Checkpoint   : ${CHECKPOINT}"
echo "ONNX         : ${ONNX_PATH}"
echo "TorchScript  : ${TORCHSCRIPT_PATH}"
echo "Pose         : ${POSE_NAME}"
echo "Fixed command: ${FIXED_COMMAND}"
echo "Start active : ${START_ACTIVE}"
echo "GLFW/RT      : ${USE_GLFW}/${REAL_TIME}"
echo "Metrics      : ${METRICS_PATH}"
echo "GUI key      : SPACE toggles [0,0,0] <-> fixed command"
echo "============================================================"

export G1_AMP_ARMHACK_WALK_ENABLE=True
export G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_DATA}"
export G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_DATA}"
export G1_AMP_ARMHACK_WALK_POSE_NAME="${POSE_NAME}"
export G1_AMP_ARMHACK_WALK_FIXED_COMMAND="${FIXED_COMMAND}"
export G1_AMP_ARMHACK_WALK_START_ACTIVE="${START_ACTIVE}"

UNITREE_PYTHON="${UNITREE_PYTHON}" \
OMP_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
MKL_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
OPENBLAS_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
NUMEXPR_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
POLICY_PATH="${TORCHSCRIPT_PATH}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" \
REAL_TIME="${REAL_TIME}" \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
METRICS_PATH="${METRICS_PATH}" \
CMD_INIT='[0.0,0.0,0.0]' \
RANDOM_COMMANDS=False \
COMMAND_MODE=independent \
COMMAND_RAMP=True \
COMMAND_MAX_LINEAR_ACCEL=0.5 \
COMMAND_MAX_YAW_ACCEL=0.8 \
bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh"

echo "[ArmHack Walk MuJoCo] metrics: ${METRICS_PATH}"
