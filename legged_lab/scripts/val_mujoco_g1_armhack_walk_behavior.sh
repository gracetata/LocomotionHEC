#!/usr/bin/env bash
# Export and run one scheduled ArmHack Walk behavior scenario in S3 G1 MuJoCo.
# The chain follows the HEC-5090 launcher: hash -> export -> 96/29 self-test -> rollout.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

CHECKPOINT=${CHECKPOINT:-${PROJECT_ROOT}/checkpoint/walk/model_10990.pt}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-}
POSE_DATA=${POSE_DATA:-"${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"}
CONTRACT_DATA=${CONTRACT_DATA:-"${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"}
SCHEDULE_DATA=${SCHEDULE_DATA:-"${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/behavior_test_scenarios.json"}
SCENARIO_NAME=${SCENARIO_NAME:-smoke_walk_to_zero}
POSE_NAME=${POSE_NAME:-pos2_down}
SIMULATION_DURATION=${SIMULATION_DURATION:-}
USE_GLFW=${USE_GLFW:-False}
REAL_TIME=${REAL_TIME:-False}
FORCE_EXPORT=${FORCE_EXPORT:-False}
ISAAC_PYTHON=${ISAAC_PYTHON:-/home/user/anaconda3/envs/env_isaaclab/bin/python}
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/user/anaconda3/envs/gmr/bin/python}
MUJOCO_CPU_THREADS=${MUJOCO_CPU_THREADS:-1}
BEHAVIOR_SETTLE_TIME_S=${BEHAVIOR_SETTLE_TIME_S:-0.75}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

for path_var in CHECKPOINT POSE_DATA CONTRACT_DATA SCHEDULE_DATA; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${LEGGED_LAB_DIR}/${value}"
    fi
done
[[ -x "${UNITREE_PYTHON}" ]] || { echo "Error: UNITREE_PYTHON 不可执行: ${UNITREE_PYTHON}" >&2; exit 1; }
for required in "${CHECKPOINT}" "${POSE_DATA}" "${CONTRACT_DATA}" "${SCHEDULE_DATA}"; do
    [[ -f "${required}" ]] || { echo "Error: 缺少文件: ${required}" >&2; exit 1; }
done

CHECKPOINT=$(realpath "${CHECKPOINT}")
CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
if [[ -n "${EXPECTED_CHECKPOINT_SHA256}" && "${CHECKPOINT_SHA256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "Error: checkpoint SHA-256 不匹配" >&2
    echo "expected=${EXPECTED_CHECKPOINT_SHA256}" >&2
    echo "actual=${CHECKPOINT_SHA256}" >&2
    exit 1
fi
CHECKPOINT_STEM=$(basename "${CHECKPOINT}" .pt)
DEPLOY_DIR=${DEPLOY_DIR:-"${LEGGED_LAB_DIR}/deployment/armhack_walk/behavior/${CHECKPOINT_STEM}_${CHECKPOINT_SHA256:0:12}"}
ONNX_PATH=${ONNX_PATH:-"${DEPLOY_DIR}/${CHECKPOINT_STEM}.onnx"}
TORCHSCRIPT_PATH=${TORCHSCRIPT_PATH:-"${DEPLOY_DIR}/${CHECKPOINT_STEM}.torchscript.pt"}
METADATA_PATH=${METADATA_PATH:-"${DEPLOY_DIR}/${CHECKPOINT_STEM}.deploy.json"}
REPORT_DIR=${REPORT_DIR:-"${DEPLOY_DIR}/Local Test Reports"}
METRICS_PATH=${METRICS_PATH:-"${REPORT_DIR}/${SCENARIO_NAME}_${POSE_NAME}.json"}

SIMULATION_DURATION_FROM_SCHEDULE=$("${UNITREE_PYTHON}" - "${SCHEDULE_DATA}" "${SCENARIO_NAME}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
scenario = payload.get("scenarios", {}).get(sys.argv[2])
if scenario is None:
    raise SystemExit(f"unknown scenario: {sys.argv[2]}")
segments = scenario.get("segments", [])
if not segments:
    raise SystemExit("scenario has no segments")
print(sum(float(segment["duration_s"]) for segment in segments))
PY
)
SIMULATION_DURATION=${SIMULATION_DURATION:-${SIMULATION_DURATION_FROM_SCHEDULE}}

need_export=False
if is_true "${FORCE_EXPORT}" || [[ ! -f "${ONNX_PATH}" || ! -f "${TORCHSCRIPT_PATH}" || ! -f "${METADATA_PATH}" ]]; then
    need_export=True
fi
if is_true "${need_export}"; then
    [[ -x "${ISAAC_PYTHON}" ]] || { echo "Error: ISAAC_PYTHON 不可执行: ${ISAAC_PYTHON}" >&2; exit 1; }
    mkdir -p "${DEPLOY_DIR}"
    "${ISAAC_PYTHON}" "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
        --robot g1 \
        --checkpoint "${CHECKPOINT}" \
        --output "${ONNX_PATH}" \
        --jit-output "${TORCHSCRIPT_PATH}" \
        --metadata "${METADATA_PATH}" \
        --default-command 0.35 0.0 0.0
fi

"${UNITREE_PYTHON}" - "${ONNX_PATH}" "${TORCHSCRIPT_PATH}" "${POSE_DATA}" "${SCHEDULE_DATA}" "${SCENARIO_NAME}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

onnx_path, jit_path, pose_path, schedule_path, scenario_name = sys.argv[1:]
poses = json.loads(Path(pose_path).read_text(encoding="utf-8"))["poses"]
if len(poses) != 3:
    raise SystemExit("Walk pose data must contain three named poses")
scenario = json.loads(Path(schedule_path).read_text(encoding="utf-8"))["scenarios"][scenario_name]
for segment in scenario["segments"]:
    command = np.asarray(segment["command"], dtype=np.float32)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise SystemExit("scenario command must be finite [vx,vy,wz]")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
jit = torch.jit.load(jit_path, map_location="cpu").eval()
rng = np.random.default_rng(20260719)
for obs in [np.zeros((1, 96), dtype=np.float32)] + [rng.normal(0, 0.5, (1, 96)).astype(np.float32) for _ in range(4)]:
    onnx_action = session.run(["actions"], {"obs": obs})[0]
    with torch.inference_mode():
        jit_action = jit(torch.from_numpy(obs)).numpy()
    if onnx_action.shape != (1, 29) or not np.all(np.isfinite(onnx_action)):
        raise SystemExit("ONNX actor failed finite 96->29 check")
    if not np.allclose(onnx_action, jit_action, rtol=1e-5, atol=1e-6):
        raise SystemExit("ONNX/TorchScript actor mismatch")
print(f"[SELF-TEST PASS] actor 96->29; scenario={scenario_name}; segments={len(scenario['segments'])}")
PY

mkdir -p "${REPORT_DIR}"
echo "============================================================"
echo " ArmHack Walk behavior MuJoCo"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "SHA-256  : ${CHECKPOINT_SHA256}"
echo "Pose      : ${POSE_NAME}"
echo "Scenario  : ${SCENARIO_NAME} (${SIMULATION_DURATION}s)"
echo "Metrics   : ${METRICS_PATH}"
echo "Hard zero : True (policy-facing command bypasses ramp at exact zero)"
echo "============================================================"

export G1_AMP_ARMHACK_WALK_ENABLE=True
export G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_DATA}"
export G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_DATA}"
export G1_AMP_ARMHACK_WALK_POSE_NAME="${POSE_NAME}"
export G1_AMP_ARMHACK_WALK_FIXED_COMMAND='[0.35,0.0,0.0]'
export G1_AMP_ARMHACK_WALK_START_ACTIVE=True
export G1_AMP_ARMHACK_WALK_SCHEDULE_PATH="${SCHEDULE_DATA}"
export G1_AMP_ARMHACK_WALK_SCENARIO_NAME="${SCENARIO_NAME}"
export G1_AMP_ARMHACK_WALK_HARD_ZERO_COMMAND=True
export G1_AMP_ARMHACK_WALK_ZERO_EPSILON=1e-6
export G1_AMP_BEHAVIOR_SETTLE_TIME_S="${BEHAVIOR_SETTLE_TIME_S}"
export G1_AMP_SOLE_MIN_CLEARANCE_M=0.025

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

echo "[ArmHack Walk behavior] metrics: ${METRICS_PATH}"
