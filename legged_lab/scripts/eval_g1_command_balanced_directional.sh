#!/usr/bin/env bash
# Run fixed-command directional diagnostics for a command-balanced G1 AMP checkpoint.
#
# Usage:
#   CHECKPOINT=logs/rsl_rl/g1_amp/<run>/model_7498.pt bash scripts/eval_g1_command_balanced_directional.sh
#   EVAL_MODES="normal_walk backward lateral_left lateral_right" bash scripts/eval_g1_command_balanced_directional.sh
#
# Environment variables:
#   ISAACLAB_PYTHON : Python executable from .envrc conda env.
#   CONDA_ENV_NAME  : Conda env name used when ISAACLAB_PYTHON is unset (default: env_isaaclab).
#   CONDA_BASE      : Conda installation root (default: /home/hecggdz/miniconda3).
#   CHECKPOINT      : Policy checkpoint to evaluate.
#   OUTPUT_DIR      : Directory for *_directional_gait_report.json files.
#   TASK            : Gym task id.
#   MOTION_DIR      : Reference motion directory for matched-demo comparison.
#   ROBOT_ASSET     : Robot asset preset (default: s3_g1_29dof).
#   NUM_ENVS        : Number of rollout envs per mode (default: 32).
#   MAX_STEPS       : Policy steps per mode (default: 700).
#   WARMUP_STEPS    : Warmup steps ignored in metrics (default: 120).
#   EVAL_MODES      : Space-separated modes.
#   SUMMARIZE       : True/False write directional_gait_summary.{json,md} after eval (default: True).
#   EXTRA_ARGS      : Additional raw arguments for analyze_g1_amp_directional_gait.py.
# ============================================================
: <<'BLOCK'
# [Command-balanced diagnostic shortcuts]

# 1. Evaluate the current 2500-iter command-balanced finetune after it finishes.
CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500/model_7498.pt bash scripts/eval_g1_command_balanced_directional.sh

# 2. Faster focused pass for the problem modes.
CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500/model_7498.pt EVAL_MODES="backward lateral_left lateral_right" NUM_ENVS=24 bash scripts/eval_g1_command_balanced_directional.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_leglab}
CONDA_BASE=${CONDA_BASE:-/home/hecggdz/miniconda3}
if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
        ISAACLAB_PYTHON="${CONDA_PREFIX}/bin/python"
    else
        ISAACLAB_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
    fi
fi

TASK=${TASK:-LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-v0}
RUN_DIR=${RUN_DIR:-logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500}
CHECKPOINT=${CHECKPOINT:-${RUN_DIR}/model_7498.pt}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_DIR}/directional_gait_analysis}
MOTION_DIR=${MOTION_DIR:-source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz}
ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
NUM_ENVS=${NUM_ENVS:-32}
MAX_STEPS=${MAX_STEPS:-700}
WARMUP_STEPS=${WARMUP_STEPS:-120}
EVAL_MODES=${EVAL_MODES:-"normal_walk slow_walk backward lateral_left lateral_right turn_left turn_right"}
SUMMARIZE=${SUMMARIZE:-True}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${ROOT_DIR}/${CHECKPOINT}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi
if [[ "${MOTION_DIR}" != /* ]]; then
    MOTION_DIR="${ROOT_DIR}/${MOTION_DIR}"
fi
if [[ ! -d "${MOTION_DIR}" ]]; then
    echo "Error: MOTION_DIR does not exist: ${MOTION_DIR}" >&2
    exit 1
fi

ISAACLAB_CONDA_PREFIX=$(cd "$(dirname "${ISAACLAB_PYTHON}")/.." && pwd)
export LD_LIBRARY_PATH="${ISAACLAB_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${LEGGED_LAB_DIR}/source/legged_lab${PYTHONPATH:+:${PYTHONPATH}}"
export LEGGED_LAB_G1_AMP_ROBOT_ASSET="${ROBOT_ASSET}"
export OMNI_LOG_DEFAULT_LEVEL=${OMNI_LOG_DEFAULT_LEVEL:-error}
export OMNI_KIT_QUIET=${OMNI_KIT_QUIET:-1}

echo "==============================================="
echo "  Evaluating G1 Command-Balanced Directional"
echo "==============================================="
echo "Task        : ${TASK}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Motion Dir  : ${MOTION_DIR}"
echo "Output Dir  : ${OUTPUT_DIR}"
echo "Robot Asset : ${ROBOT_ASSET}"
echo "Num Envs    : ${NUM_ENVS}"
echo "Steps       : ${MAX_STEPS} warmup=${WARMUP_STEPS}"
echo "Modes       : ${EVAL_MODES}"
echo "Summarize   : ${SUMMARIZE}"
echo "Extra Args  : ${EXTRA_ARGS}"
echo "==============================================="

cd "${LEGGED_LAB_DIR}"
mkdir -p "${OUTPUT_DIR}"

for mode in ${EVAL_MODES}; do
    case "${mode}" in
        normal_walk)
            vx=0.82
            vy=0.0
            wz=0.0
            ;;
        slow_walk)
            vx=0.42
            vy=0.0
            wz=0.0
            ;;
        backward)
            vx=-0.40
            vy=0.0
            wz=0.0
            ;;
        lateral_left)
            vx=0.0
            vy=0.30
            wz=0.0
            ;;
        lateral_right)
            vx=0.0
            vy=-0.30
            wz=0.0
            ;;
        turn_left)
            vx=0.35
            vy=0.0
            wz=0.50
            ;;
        turn_right)
            vx=0.35
            vy=0.0
            wz=-0.50
            ;;
        *)
            echo "Error: unknown eval mode: ${mode}" >&2
            echo "Valid modes: normal_walk slow_walk backward lateral_left lateral_right turn_left turn_right" >&2
            exit 1
            ;;
    esac

    echo
    echo "[Eval] ${mode}: command=(${vx}, ${vy}, ${wz})"
    "${ISAACLAB_PYTHON}" scripts/tools/analyze_g1_amp_directional_gait.py \
        --task "${TASK}" \
        --checkpoint "${CHECKPOINT}" \
        --motion_dir "${MOTION_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --label "${mode}" \
        --command_lin_x "${vx}" \
        --command_lin_y "${vy}" \
        --command_yaw "${wz}" \
        --num_envs "${NUM_ENVS}" \
        --max_steps "${MAX_STEPS}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --robot_asset "${ROBOT_ASSET}" \
        ${EXTRA_ARGS}
done

if [[ "${SUMMARIZE}" == "True" || "${SUMMARIZE}" == "true" || "${SUMMARIZE}" == "1" ]]; then
    echo
    "${ISAACLAB_PYTHON}" scripts/tools/summarize_g1_directional_gait_reports.py \
        --report_dir "${OUTPUT_DIR}"
fi

echo
echo "Reports written to: ${OUTPUT_DIR}"
