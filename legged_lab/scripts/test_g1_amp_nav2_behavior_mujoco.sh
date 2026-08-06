#!/usr/bin/env bash
# Export and headless-smoke a generic full-body G1 Nav2 behavior checkpoint.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

BASE_CHECKPOINT="${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"
EXPECTED_BASE_SHA256="1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
EXPECTED_BASE_SIZE=14826139
CHECKPOINT=${CHECKPOINT:-${1:-}}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
SIMULATION_DURATION=${SIMULATION_DURATION:-2.0}
CMD_INIT=${CMD_INIT:-'[0.06,0.0,0.0]'}

die() {
    echo "Error: $*" >&2
    exit 1
}

verify_baseline() {
    [[ -f "${BASE_CHECKPOINT}" ]] || return 1
    [[ "$(stat -c '%s' "${BASE_CHECKPOINT}")" == "${EXPECTED_BASE_SIZE}" ]] || return 1
    [[ "$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_BASE_SHA256}" ]]
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    if ! verify_baseline; then
        echo "Error: protected model_10990 changed during MuJoCo smoke." >&2
        status=1
    fi
    exit "${status}"
}
trap verify_on_exit EXIT

verify_baseline || die "protected model_10990 failed its size/SHA-256 contract"
[[ -n "${CHECKPOINT}" ]] || die "set CHECKPOINT to a new Nav2Behavior model_*.pt"
[[ -x "${ISAACLAB_PYTHON}" ]] || die "ISAACLAB_PYTHON is not executable"
CHECKPOINT=$(realpath "${CHECKPOINT}")
[[ -f "${CHECKPOINT}" ]] || die "checkpoint does not exist: ${CHECKPOINT}"
PROTECTED_DIR=$(realpath "${PROJECT_ROOT}/checkpoint/walk")
case "${CHECKPOINT}" in
    "${PROTECTED_DIR}"/*) die "refusing to export or test a protected baseline path" ;;
esac

EXPORT_DIR=${EXPORT_DIR:-"$(dirname "${CHECKPOINT}")/exported/nav2_behavior_mujoco_smoke"}
mkdir -p "${EXPORT_DIR}"
EXPORT_DIR=$(realpath "${EXPORT_DIR}")
POLICY_ONNX="${EXPORT_DIR}/policy.onnx"
POLICY_JIT="${EXPORT_DIR}/policy.pt"
POLICY_METADATA="${EXPORT_DIR}/policy.deploy.json"
METRICS_PATH="${EXPORT_DIR}/metrics.json"

PYTHONNOUSERSITE=1 "${ISAACLAB_PYTHON}" "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
    --robot g1 \
    --checkpoint "${CHECKPOINT}" \
    --output "${POLICY_ONNX}" \
    --jit-output "${POLICY_JIT}" \
    --metadata "${POLICY_METADATA}" \
    --default-command 0.06 0.0 0.0

PYTHONNOUSERSITE=1 "${ISAACLAB_PYTHON}" - "${POLICY_JIT}" "${POLICY_METADATA}" <<'PY'
import json
import sys

import torch

policy_path, metadata_path = sys.argv[1:]
policy = torch.jit.load(policy_path, map_location="cpu").eval()
with torch.inference_mode():
    actions = policy(torch.zeros(1, 96))
if actions.shape != (1, 29):
    raise RuntimeError(f"Expected 96->29 policy, got output shape {tuple(actions.shape)}")
if not torch.isfinite(actions).all():
    raise RuntimeError("TorchScript inference produced a non-finite action.")
metadata = json.loads(open(metadata_path, encoding="utf-8").read())
if metadata.get("obs_dim") != 96 or metadata.get("action_dim") != 29:
    raise RuntimeError(f"Invalid deployment metadata interface: {metadata}")
print("[nav2-behavior-smoke] TorchScript interface 96->29 is finite.")
PY

POLICY_PATH="${POLICY_JIT}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW=False \
REAL_TIME=False \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
CMD_INIT="${CMD_INIT}" \
RANDOM_COMMANDS=False \
COMMAND_MODE=independent \
COMMAND_RAMP=False \
METRICS_PATH="${METRICS_PATH}" \
TORSO_TRACE_PATH="${EXPORT_DIR}/torso_trace.csv" \
TASK_TRACE_PATH="${EXPORT_DIR}/task_trace.csv" \
bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh"

PYTHONNOUSERSITE=1 "${ISAACLAB_PYTHON}" - "${METRICS_PATH}" <<'PY'
import json
import math
import sys

metrics = json.loads(open(sys.argv[1], encoding="utf-8").read())

def validate_finite(value, path="metrics"):
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric at {path}: {value}")

validate_finite(metrics)
if float(metrics.get("sim_time", 0.0)) <= 0.0:
    raise RuntimeError("MuJoCo smoke did not advance simulation time.")
health = metrics.get("health", {})
if not health.get("healthy") or health.get("fallen"):
    raise RuntimeError(f"MuJoCo health check failed: {health}")
print(
    "[nav2-behavior-smoke] healthy=True "
    f"sim_time={metrics['sim_time']:.3f}s "
    f"total_score={metrics['score']['total_score']:.1f}"
)
PY

verify_baseline || die "protected model_10990 failed its post-smoke contract"
echo "Generic G1 Nav2 behavior MuJoCo smoke passed: ${METRICS_PATH}"
