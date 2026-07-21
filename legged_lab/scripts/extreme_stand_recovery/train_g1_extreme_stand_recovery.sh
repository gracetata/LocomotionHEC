#!/usr/bin/env bash
# Train the independent 96-observation -> 29-action full-body Stand recovery policy.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-v0"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_ROOT}/checkpoint/stand/model_2999.pt"}
VERIFY_BASE_SHA256=${VERIFY_BASE_SHA256:-True}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f"}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-5000}
RUN_NAME=${RUN_NAME:-g1_extreme_stand_recovery_from_stand_model2999}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}

LEG_NOISE_RAD=${LEG_NOISE_RAD:-0.25}
WAIST_NOISE_RAD=${WAIST_NOISE_RAD:-0.35}
ARM_NOISE_RAD=${ARM_NOISE_RAD:-0.60}
TORSO_FORCE_MAX_N=${TORSO_FORCE_MAX_N:-35.0}
TORSO_TORQUE_MAX_NM=${TORSO_TORQUE_MAX_NM:-5.0}
PELVIS_FORCE_MAX_N=${PELVIS_FORCE_MAX_N:-30.0}
PELVIS_TORQUE_MAX_NM=${PELVIS_TORQUE_MAX_NM:-4.0}
LIMB_FORCE_MAX_N=${LIMB_FORCE_MAX_N:-12.0}
LIMB_TORQUE_MAX_NM=${LIMB_TORQUE_MAX_NM:-2.0}
TERMINATION_PENALTY_MAG=${TERMINATION_PENALTY_MAG:-1000.0}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}

die() {
    echo "Error: $*" >&2
    exit 1
}

[[ -f "${BASE_CHECKPOINT}" ]] || die "base Stand checkpoint not found: ${BASE_CHECKPOINT}"
[[ -x "${ISAACLAB_PYTHON}" ]] || die "IsaacLab Python is not executable: ${ISAACLAB_PYTHON}"
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be a positive integer."
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be a positive integer."

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
if [[ "${VERIFY_BASE_SHA256,,}" == "true" || "${VERIFY_BASE_SHA256}" == "1" ]]; then
    [[ "${ACTUAL_BASE_SHA256}" == "${EXPECTED_BASE_SHA256}" ]] || \
        die "base checkpoint SHA-256 mismatch: expected ${EXPECTED_BASE_SHA256}, got ${ACTUAL_BASE_SHA256}"
fi

validate_nonnegative() {
    local name=$1 value=$2 maximum=$3
    awk -v value="${value}" -v maximum="${maximum}" \
        'BEGIN { exit !(value >= 0.0 && value <= maximum) }' || \
        die "${name} must be within [0, ${maximum}], got ${value}."
}

validate_nonnegative LEG_NOISE_RAD "${LEG_NOISE_RAD}" 1.0
validate_nonnegative WAIST_NOISE_RAD "${WAIST_NOISE_RAD}" 1.0
validate_nonnegative ARM_NOISE_RAD "${ARM_NOISE_RAD}" 1.5
validate_nonnegative TORSO_FORCE_MAX_N "${TORSO_FORCE_MAX_N}" 100.0
validate_nonnegative TORSO_TORQUE_MAX_NM "${TORSO_TORQUE_MAX_NM}" 20.0
validate_nonnegative PELVIS_FORCE_MAX_N "${PELVIS_FORCE_MAX_N}" 100.0
validate_nonnegative PELVIS_TORQUE_MAX_NM "${PELVIS_TORQUE_MAX_NM}" 20.0
validate_nonnegative LIMB_FORCE_MAX_N "${LIMB_FORCE_MAX_N}" 50.0
validate_nonnegative LIMB_TORQUE_MAX_NM "${LIMB_TORQUE_MAX_NM}" 10.0

# The RSL-RL resolver expects a checkpoint inside the selected experiment log
# namespace.  Normalize storages to CPU so a model saved on remote cuda:1 also
# loads on a single-GPU cuda:0 workstation.  The source checkpoint is immutable.
BASE_RUN_NAME="_extreme_stand_base_${ACTUAL_BASE_SHA256:0:12}"
BASE_RUN_DIR="${LEGGED_LAB_DIR}/logs/rsl_rl/g1_extreme_stand_recovery/${BASE_RUN_NAME}"
mkdir -p "${BASE_RUN_DIR}"
BASE_CHECKPOINT_NAME=$(basename "${BASE_CHECKPOINT}")
[[ "${BASE_CHECKPOINT_NAME}" =~ ^model_[A-Za-z0-9_-]+\.pt$ ]] || \
    die "base checkpoint filename must match model_<id>.pt, got ${BASE_CHECKPOINT_NAME}"
PORTABLE_BASE_CHECKPOINT="${BASE_RUN_DIR}/${BASE_CHECKPOINT_NAME}"
PORTABLE_BASE_MARKER="${BASE_RUN_DIR}/source.sha256"
if [[ ! -f "${PORTABLE_BASE_CHECKPOINT}" || ! -f "${PORTABLE_BASE_MARKER}" || \
      "$(<"${PORTABLE_BASE_MARKER}")" != "${ACTUAL_BASE_SHA256}" ]]; then
    "${ISAACLAB_PYTHON}" - "${BASE_CHECKPOINT}" "${PORTABLE_BASE_CHECKPOINT}" \
        "${PORTABLE_BASE_MARKER}" "${ACTUAL_BASE_SHA256}" <<'PY'
import os
import sys
from pathlib import Path

import torch

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
marker = Path(sys.argv[3])
source_sha256 = sys.argv[4]
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(destination.suffix + ".tmp")
checkpoint = torch.load(source, weights_only=False, map_location="cpu")
torch.save(checkpoint, temporary)
os.replace(temporary, destination)
marker.write_text(source_sha256, encoding="utf-8")
PY
fi

echo "============================================================"
echo " G1 extreme full-body Stand recovery training"
echo "============================================================"
echo "Task            : ${TASK} (ManagerBasedAmpEnv; no arm override)"
echo "Policy contract : 96 observations -> 29 full-body joint actions"
echo "Base checkpoint : ${BASE_CHECKPOINT}"
echo "Base SHA-256    : ${ACTUAL_BASE_SHA256}"
echo "Joint noise     : leg=${LEG_NOISE_RAD}, waist=${WAIST_NOISE_RAD}, arm=${ARM_NOISE_RAD} rad"
echo "Torso wrench    : +/-${TORSO_FORCE_MAX_N} N, +/-${TORSO_TORQUE_MAX_NM} Nm"
echo "Pelvis wrench   : +/-${PELVIS_FORCE_MAX_N} N, +/-${PELVIS_TORQUE_MAX_NM} Nm"
echo "Limb wrench     : +/-${LIMB_FORCE_MAX_N} N, +/-${LIMB_TORQUE_MAX_NM} Nm"
echo "Fall penalty    : -${TERMINATION_PENALTY_MAG}"
echo "Training        : ${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations on ${DEVICE}"
echo "Run name        : ${RUN_NAME}"
echo "============================================================"

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
DEVICE="${DEVICE}" \
AGENT_DEVICE="${DEVICE}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="^${BASE_RUN_NAME}$" \
CHECKPOINT="^${BASE_CHECKPOINT_NAME}$" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=1 \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
ENTROPY_COEF=0.003 \
BASELINE_KL_ENABLE=False \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
  "env.events.reset_leg_joints_with_noise.params.position_range=[-${LEG_NOISE_RAD},${LEG_NOISE_RAD}]" \
  "env.events.reset_waist_joints_with_noise.params.position_range=[-${WAIST_NOISE_RAD},${WAIST_NOISE_RAD}]" \
  "env.events.reset_arm_joints_with_noise.params.position_range=[-${ARM_NOISE_RAD},${ARM_NOISE_RAD}]" \
  "env.events.random_torso_external_wrench.params.force_range=[-${TORSO_FORCE_MAX_N},${TORSO_FORCE_MAX_N}]" \
  "env.events.random_torso_external_wrench.params.torque_range=[-${TORSO_TORQUE_MAX_NM},${TORSO_TORQUE_MAX_NM}]" \
  "env.events.random_pelvis_external_wrench.params.force_range=[-${PELVIS_FORCE_MAX_N},${PELVIS_FORCE_MAX_N}]" \
  "env.events.random_pelvis_external_wrench.params.torque_range=[-${PELVIS_TORQUE_MAX_NM},${PELVIS_TORQUE_MAX_NM}]" \
  "env.events.random_arm_external_wrench.params.force_range=[-${LIMB_FORCE_MAX_N},${LIMB_FORCE_MAX_N}]" \
  "env.events.random_arm_external_wrench.params.torque_range=[-${LIMB_TORQUE_MAX_NM},${LIMB_TORQUE_MAX_NM}]" \
  "env.events.random_leg_external_wrench.params.force_range=[-${LIMB_FORCE_MAX_N},${LIMB_FORCE_MAX_N}]" \
  "env.events.random_leg_external_wrench.params.torque_range=[-${LIMB_TORQUE_MAX_NM},${LIMB_TORQUE_MAX_NM}]" \
  "env.rewards.termination_penalty.weight=-${TERMINATION_PENALTY_MAG}" \
  "agent.algorithm.learning_rate=${LEARNING_RATE}" \
  agent.load_policy_only=True \
  agent.reset_iteration_on_policy_only_load=True \
  "$@"
