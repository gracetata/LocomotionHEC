#!/usr/bin/env bash
# Export a selected Unitree G1 AMP checkpoint to checkpoint/<weight-name>/locomotion.onnx.
#
# Usage:
#   bash scripts/export_g1_amp_locomotion_onnx.sh \
#     legged_lab/logs/rsl_rl/g1_amp/<run>/model_9996.pt
#   CHECKPOINT=legged_lab/logs/rsl_rl/g1_amp/<run>/model_9996.pt bash scripts/export_g1_amp_locomotion_onnx.sh
#
# Environment variables:
#   ISAACLAB_PYTHON : Python executable from the IsaacLab environment.
#   CHECKPOINT      : RSL-RL model_*.pt checkpoint. Positional arg has priority.
#   OUTPUT_ROOT     : Root folder for exported ONNX packages (default: ./checkpoint).
#   OUTPUT_NAME     : Output folder name under OUTPUT_ROOT (default: checkpoint basename without .pt).
#   ONNX_OUT        : Explicit ONNX path override.
#   METADATA_OUT    : Explicit metadata JSON path override.
#   DRY_RUN         : True prints the export command without writing files.
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
bash scripts/export_g1_amp_locomotion_onnx.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-17_03-48-38_s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000/model_9996.pt
    
bash scripts/export_g1_amp_locomotion_onnx.sh \
    legged_lab/logs/rsl_rl/g1_amp/2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500/model_8997.pt

BLOCK

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}/legged_lab"

ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
DEFAULT_CHECKPOINT="${ROOT_DIR}/legged_lab/logs/rsl_rl/g1_amp/2026-06-17_03-48-38_s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000/model_9996.pt"

if [[ $# -gt 0 && "${1}" != --* ]]; then
    CHECKPOINT="$1"
    shift
else
    CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
fi

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${ROOT_DIR}/${CHECKPOINT}"
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/checkpoint}
CHECKPOINT_BASENAME=$(basename "${CHECKPOINT}")
DEFAULT_OUTPUT_NAME="${CHECKPOINT_BASENAME%.*}"
OUTPUT_NAME=${OUTPUT_NAME:-${DEFAULT_OUTPUT_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${OUTPUT_NAME}}
ONNX_OUT=${ONNX_OUT:-${OUTPUT_DIR}/locomotion.onnx}
METADATA_OUT=${METADATA_OUT:-${OUTPUT_DIR}/locomotion.deploy.json}
DRY_RUN=${DRY_RUN:-False}

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_ROOT}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi
if [[ "${ONNX_OUT}" != /* ]]; then
    ONNX_OUT="${ROOT_DIR}/${ONNX_OUT}"
fi
if [[ "${METADATA_OUT}" != /* ]]; then
    METADATA_OUT="${ROOT_DIR}/${METADATA_OUT}"
fi

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/user/anaconda3/envs/env_isaaclab/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

EXPORT_COMMAND=(
    "${ISAACLAB_PYTHON}"
    "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py"
    --robot g1
    --checkpoint "${CHECKPOINT}"
    --output "${ONNX_OUT}"
    --metadata "${METADATA_OUT}"
    "$@"
)

echo "====================================="
echo "  Export G1 AMP Locomotion ONNX"
echo "====================================="
echo "Python     : ${ISAACLAB_PYTHON}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "ONNX       : ${ONNX_OUT}"
echo "Metadata   : ${METADATA_OUT}"
echo "Dry Run    : ${DRY_RUN}"
echo "====================================="

if is_true "${DRY_RUN}"; then
    printf 'Dry-run export command:'
    printf ' %q' "${EXPORT_COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${EXPORT_COMMAND[@]}"

echo "====================================="
echo "ONNX export complete:"
echo "  ${ONNX_OUT}"
echo "Deploy example:"
echo "  CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=<iface> ONNX_PATH=${ONNX_OUT} COMMAND_MODE=nav_mock bash scripts/deploy_real_g1_amp_onnx.sh"
echo "====================================="
