#!/usr/bin/env bash
# Export a Unitree G1 AMP checkpoint to TorchScript and ONNX without starting Isaac Sim.
# Usage:
#   CHECKPOINT=legged_lab/logs/rsl_rl/g1_amp/<run>/model_2999.pt bash scripts/export_g1_amp_policy.sh
#
# Environment variables:
#   ISAACLAB_PYTHON : Python executable from legged_lab/.envrc conda env.
#   CHECKPOINT      : RSL-RL checkpoint to export.
#   JIT_OUT         : TorchScript output path consumed by unitree_sim2sim2real.
#   ONNX_OUT        : ONNX output path for inspection / external runtimes.
#   METADATA_OUT    : JSON deployment metadata path.
#   DRY_RUN         : True/False. Print the offline export command without writing files.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 导出默认已跑通 G1 AMP checkpoint 到 exported/policy.pt 与 exported/policy.onnx
bash scripts/export_g1_amp_policy.sh

# 指定 checkpoint 导出
CHECKPOINT=legged_lab/logs/rsl_rl/g1_amp/2026-05-03_16-19-07/model_1400.pt bash scripts/export_g1_amp_policy.sh

# 只检查将要执行的离线命令，不写文件
DRY_RUN=True bash scripts/export_g1_amp_policy.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}/legged_lab"

ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
CHECKPOINT=${CHECKPOINT:-${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp/2026-05-20_23-50-16_baseline_3000_accad50hz_rsi_20260520/model_2999.pt}
EXPORT_DIR=${EXPORT_DIR:-$(dirname "${CHECKPOINT}")/exported}
JIT_OUT=${JIT_OUT:-${EXPORT_DIR}/policy.pt}
ONNX_OUT=${ONNX_OUT:-${EXPORT_DIR}/policy.onnx}
METADATA_OUT=${METADATA_OUT:-${EXPORT_DIR}/policy.deploy.json}
DRY_RUN=${DRY_RUN:-False}

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
	echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
	exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
	echo "Error: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
	exit 1
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/env_isaaclab/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "====================================="
echo "  Exporting G1 AMP Actor Offline"
echo "====================================="
echo "Python      : ${ISAACLAB_PYTHON}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "TorchScript : ${JIT_OUT}"
echo "ONNX        : ${ONNX_OUT}"
echo "Metadata    : ${METADATA_OUT}"
echo "Dry Run     : ${DRY_RUN}"
echo "====================================="

EXPORT_COMMAND=(
	"${ISAACLAB_PYTHON}"
	"${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py"
	--robot g1
	--checkpoint "${CHECKPOINT}"
	--output "${ONNX_OUT}"
	--jit-output "${JIT_OUT}"
	--metadata "${METADATA_OUT}"
	"$@"
)

if [[ "${DRY_RUN,,}" == "true" || "${DRY_RUN}" == "1" ]]; then
	printf 'Dry-run command:'
	printf ' %q' "${EXPORT_COMMAND[@]}"
	printf '\n'
	exit 0
fi

"${EXPORT_COMMAND[@]}"
