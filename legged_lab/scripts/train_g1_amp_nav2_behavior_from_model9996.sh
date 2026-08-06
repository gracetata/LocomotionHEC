#!/usr/bin/env bash
# Full-state Nav2 behavior continuation from the archived HEC-5090 model_9996.pt.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

SOURCE_CHECKPOINT="${PROJECT_ROOT}/checkpoint/nav2_behavior_model9996_source/model_9996.pt"
SOURCE_SHA256="bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"
SOURCE_SIZE=16202421
SOURCE_ITERATION=9996
STAGING_RUN="_source_model9996_full_state"
STAGING_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/${STAGING_RUN}"
ISAACLAB_PYTHON="/home/user/anaconda3/envs/env_isaaclab/bin/python"

NUM_ENVS=${NUM_ENVS:-4096}
REMAINING_ITERATIONS=${REMAINING_ITERATIONS:-3000}
RUN_NAME=${RUN_NAME:-nav2_behavior_from_model9996_fullstate_3000_20260804}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp_nav2_behavior/train_${RUN_NAME}.log"}
CPU_AFFINITY=${CPU_AFFINITY:-16-31}

verify_source() {
    [[ -f "${SOURCE_CHECKPOINT}" ]]
    [[ "$(stat -c '%s' "${SOURCE_CHECKPOINT}")" == "${SOURCE_SIZE}" ]]
    [[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]]
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    if ! verify_source; then
        echo "Error: protected model_9996 failed its post-training integrity check." >&2
        status=1
    fi
    exit "${status}"
}
trap verify_on_exit EXIT

verify_source || {
    echo "Error: archived HEC-5090 model_9996 failed its integrity check." >&2
    exit 1
}
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: NUM_ENVS must be a positive integer." >&2
    exit 1
}
[[ "${REMAINING_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: REMAINING_ITERATIONS must be a positive integer." >&2
    exit 1
}
[[ "${RUN_NAME}" != */* ]] || {
    echo "Error: RUN_NAME must not contain a path separator." >&2
    exit 1
}

mkdir -p "${STAGING_DIR}"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_9996.pt"

# The local i9 host has shown intermittent object corruption on its high-clock
# P-cores.  Keep the validation and training process on the repeatedly verified
# E-core set, and keep Python packages inside the IsaacLab Conda environment.
export PYTHONNOUSERSITE=1
taskset -c "${CPU_AFFINITY}" "${ISAACLAB_PYTHON}" - "${SOURCE_CHECKPOINT}" <<'PY'
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if checkpoint.get("iter") != 9996:
    raise RuntimeError(f"Expected checkpoint iteration 9996, got {checkpoint.get('iter')}")
model = checkpoint["model_state_dict"]
discriminator = checkpoint["amp_discriminator_state_dict"]
actor_weights = [
    value
    for name, value in model.items()
    if name.startswith("actor.") and name.endswith(".weight")
]
critic_weights = [
    value
    for name, value in model.items()
    if name.startswith("critic.") and name.endswith(".weight")
]
disc_weights = [value for name, value in discriminator.items() if name.endswith("weight")]
if actor_weights[0].shape[1] != 96 or actor_weights[-1].shape[0] != 29:
    raise RuntimeError("model_9996 actor interface is not 96->29")
if critic_weights[0].shape[1] != 297 or critic_weights[-1].shape[0] != 1:
    raise RuntimeError("model_9996 critic interface is not 297->1")
if disc_weights[0].shape[1] != 280:
    raise RuntimeError("model_9996 discriminator input is not 280")
if not checkpoint.get("optimizer_state_dict"):
    raise RuntimeError("model_9996 has no PPO optimizer state")
if not checkpoint.get("amp_discriminator_optimizer_state_dict"):
    raise RuntimeError("model_9996 has no AMP optimizer state")

nonfinite = []
def visit(value, path="checkpoint"):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            nonfinite.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            visit(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            visit(item, f"{path}[{index}]")
visit(checkpoint)
if nonfinite:
    raise RuntimeError(f"model_9996 contains non-finite tensors: {nonfinite}")

with tempfile.TemporaryDirectory(prefix="g1-nav2-model9996-tensorboard-") as log_dir:
    writer = SummaryWriter(log_dir=log_dir, max_queue=16, flush_secs=1)
    for step in range(1024):
        writer.add_scalar("preflight/value", step / 1024.0, step)
    writer.flush()
    writer.close()
print("model_9996 full-state and TensorBoard preflight passed")
PY

FINAL_ITERATION=$((SOURCE_ITERATION + REMAINING_ITERATIONS - 1))
echo "=================================================="
echo " Generic full-body G1 Nav2 behavior continuation"
echo "=================================================="
echo "Source checkpoint : ${SOURCE_CHECKPOINT}"
echo "Source SHA-256    : ${SOURCE_SHA256}"
echo "Load contract     : full actor/critic/PPO/AMP/normalizer/optimizers"
echo "Iterations        : ${SOURCE_ITERATION}..${FINAL_ITERATION}"
echo "Run               : ${RUN_NAME}"
echo "CPU affinity      : ${CPU_AFFINITY}"
echo "Baseline KL       : frozen actor from the same model_9996 source"
echo "=================================================="

TASK=LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0 \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${REMAINING_ITERATIONS}" \
SEED=42 \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${STAGING_RUN}$" \
CHECKPOINT='^model_9996.pt$' \
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
BASELINE_KL_CHECKPOINT="${SOURCE_CHECKPOINT}" \
BASELINE_KL_SCALE=0.003 \
EXTRA_HYDRA_ARGS="" \
taskset -c "${CPU_AFFINITY}" bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.load_actor_only=False \
    agent.load_policy_only=False \
    agent.reset_iteration_on_policy_only_load=False \
    agent.reset_amp_on_load=False
