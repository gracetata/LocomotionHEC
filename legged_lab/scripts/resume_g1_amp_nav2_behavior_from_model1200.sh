#!/usr/bin/env bash
# Full-state continuation after the model_1200 checkpoint from the Nav2 behavior run.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

SOURCE_RUN="2026-07-31_16-16-20_nav2_behavior_resume600_to2999_20260731"
SOURCE_CHECKPOINT="${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/${SOURCE_RUN}/model_1200.pt"
SOURCE_CHECKPOINT_SHA256="61eb7f3d46382ad5ed89ab00dcc440be1420d4fe87e88e726def46dd2d85d5d5"
SOURCE_CHECKPOINT_SIZE=16202741
BASE_CHECKPOINT="${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"
BASE_SHA256="1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
BASE_SIZE=14826139
ISAACLAB_PYTHON="/home/user/anaconda3/envs/env_isaaclab/bin/python"

NUM_ENVS=${NUM_ENVS:-4096}
REMAINING_ITERATIONS=${REMAINING_ITERATIONS:-1800}
RUN_NAME=${RUN_NAME:-nav2_behavior_resume1200_to2999_20260804}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/train_${RUN_NAME}.log"}
CPU_AFFINITY=${CPU_AFFINITY:-16-31}

verify_file() {
    local path=$1 expected_size=$2 expected_sha=$3
    [[ -f "${path}" ]]
    [[ "$(stat -c '%s' "${path}")" == "${expected_size}" ]]
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    if ! verify_file "${BASE_CHECKPOINT}" "${BASE_SIZE}" "${BASE_SHA256}"; then
        echo "Error: protected model_10990 failed its post-training integrity check." >&2
        status=1
    fi
    exit "${status}"
}
trap verify_on_exit EXIT

verify_file "${BASE_CHECKPOINT}" "${BASE_SIZE}" "${BASE_SHA256}" || {
    echo "Error: protected model_10990 failed its integrity check." >&2
    exit 1
}
verify_file \
    "${SOURCE_CHECKPOINT}" \
    "${SOURCE_CHECKPOINT_SIZE}" \
    "${SOURCE_CHECKPOINT_SHA256}" || {
    echo "Error: model_1200 failed its full-state continuation contract." >&2
    exit 1
}

# Keep torch and TensorBoard inside the IsaacLab Conda environment.  The previous
# continuation mixed this environment with ~/.local TensorBoard and its async
# event writer terminated training at iteration 1389.
export PYTHONNOUSERSITE=1
taskset -c "${CPU_AFFINITY}" "${ISAACLAB_PYTHON}" - <<'PY'
import importlib
import tempfile
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

module = importlib.import_module("tensorboard.compat.tensorflow_stub.pywrap_tensorflow")
expected_root = Path("/home/user/anaconda3/envs/env_isaaclab").resolve()
module_path = Path(module.__file__).resolve()
if expected_root not in module_path.parents:
    raise RuntimeError(f"TensorBoard escaped the IsaacLab environment: {module_path}")
if not isinstance(module.masked_crc32c(b"nav2-event-writer-preflight"), int):
    raise RuntimeError("TensorBoard masked_crc32c preflight returned a non-integer")
with tempfile.TemporaryDirectory(prefix="g1-nav2-tensorboard-preflight-") as log_dir:
    writer = SummaryWriter(log_dir=log_dir, max_queue=16, flush_secs=1)
    for step in range(1024):
        writer.add_scalar("preflight/value", step / 1024.0, step)
    writer.flush()
    writer.close()
print(f"TensorBoard preflight passed: {module_path}")
PY

# The host's high-frequency P-cores produced intermittent Python object/type
# corruption during repeated torch imports.  Restrict this continuation to the
# lower-frequency E-cores, which passed repeated full checkpoint validation.
TASK=LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0 \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${REMAINING_ITERATIONS}" \
SEED=42 \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${SOURCE_RUN}$" \
CHECKPOINT='^model_1200.pt$' \
HEADLESS=True \
QUIET_TERMINAL=True \
TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=True \
RANDOMIZATION_STRENGTH=1 \
STYLE_REWARD_SCALE=5.0 \
TASK_STYLE_LERP=0.4 \
AMP_GRAD_PENALTY_SCALE=20.0 \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE=0.003 \
EXTRA_HYDRA_ARGS="" \
taskset -c "${CPU_AFFINITY}" bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.load_actor_only=False \
    agent.load_policy_only=False \
    agent.reset_iteration_on_policy_only_load=False
