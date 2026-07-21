#!/usr/bin/env bash
# HEC-5090 / 本机通用：三套 use 模型的入口、MuJoCo 和真机 dry-run 总检。
# 绝不初始化 DDS，也不向实体机器人发送 LowCmd。

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NET=${NET:-enp11s0}
SMOKE_DURATION=${SMOKE_DURATION:-3.0}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/outputs/hec5090_use_models_smoke"}

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
    echo "Error: 找不到 env_leglab/gmr Python；请设置 UNITREE_PYTHON。" >&2
    exit 1
}

echo "============================================================"
echo " HEC-5090 三模型安全总检（MuJoCo + real dry-run）"
echo "============================================================"
echo "Root   : ${ROOT_DIR}"
echo "Python : ${UNITREE_PYTHON}"
echo "Net    : ${NET}（只传给 dry-run，不初始化 DDS）"
echo "Results: ${RESULTS_ROOT}"
echo "============================================================"

"${UNITREE_PYTHON}" - <<'PY'
import importlib.util
required = ["numpy", "torch", "onnxruntime", "mujoco", "yaml", "cyclonedds", "unitree_sdk2py"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python modules: {missing}")
print("[PASS] Python dependencies:", ", ".join(required))
PY

check_model() {
    local path=$1 expected=$2 label=$3 actual
    [[ -f "${path}" ]] || { echo "Error: ${label} 不存在: ${path}" >&2; exit 1; }
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]] || {
        echo "Error: ${label} SHA-256 不匹配: ${actual}" >&2
        exit 1
    }
    echo "[PASS] ${label}: ${actual}"
}

check_model "${ROOT_DIR}/use/armhack_walk_model_10990.onnx" \
    b052c3b0583834a742ea59e736d55c3c9bafabb75f1d4fae65980166d4a895aa "ArmHack Walk ONNX"
check_model "${ROOT_DIR}/use/armhack_stand_model_2999.onnx" \
    354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf "ArmHack Stand ONNX"
check_model "${ROOT_DIR}/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
    0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1 "Extreme Stand ONNX"

echo "[1/6] ArmHack Walk MuJoCo"
UNITREE_PYTHON="${UNITREE_PYTHON}" USE_GLFW=False REAL_TIME=False \
SIMULATION_DURATION="${SMOKE_DURATION}" RESULTS_ROOT="${RESULTS_ROOT}/walk" \
bash "${ROOT_DIR}/scripts/sim2sim_g1_armhack_walk_use_onnx.sh"

echo "[2/6] ArmHack Stand MuJoCo"
UNITREE_PYTHON="${UNITREE_PYTHON}" MODE=representative_poses \
USE_GLFW=False REAL_TIME=False SIMULATION_DURATION="${SMOKE_DURATION}" \
POLICY_PATH="${ROOT_DIR}/use/armhack_stand_model_2999.torchscript.pt" \
bash "${ROOT_DIR}/legged_lab/scripts/val_mujoco_g1_armhack_stand.sh"

echo "[3/6] Extreme Stand MuJoCo"
UNITREE_PYTHON="${UNITREE_PYTHON}" PROFILE=nominal \
USE_GLFW=False REAL_TIME=False SIMULATION_DURATION="${SMOKE_DURATION}" \
RESULTS_ROOT="${RESULTS_ROOT}/extreme" \
bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"

echo "[4/6] ArmHack Walk real dry-run"
UNITREE_PYTHON="${UNITREE_PYTHON}" DRY_RUN=True NET="${NET}" COMMAND_MODE=fixed \
bash "${ROOT_DIR}/scripts/deploy_real_g1_armhack_walk.sh"

echo "[5/6] ArmHack Stand real dry-run"
UNITREE_PYTHON="${UNITREE_PYTHON}" DRY_RUN=True NET="${NET}" \
bash "${ROOT_DIR}/scripts/deploy_real_g1_armhack_stand.sh"

echo "[6/6] Extreme Stand real dry-run"
UNITREE_PYTHON="${UNITREE_PYTHON}" DRY_RUN=True NET="${NET}" PING_ROBOT=False \
bash "${ROOT_DIR}/scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh"

echo "[PASS] 三套模型 MuJoCo 与真机 dry-run 全部通过；未初始化 DDS，未发送 LowCmd。"
