#!/usr/bin/env bash
# Conservative continuation from the validated model_159 checkpoint.
# The mature curriculum starts immediately, with modestly harder disturbances
# and a stronger near-target ankle-torque penalty.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BASE_RUN="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-14_14-42-29_armhack_stand_exact_30cm_tight_success_final_20260814"
BASE_MODEL="${BASE_RUN}/model_159.pt"

BASE_CHECKPOINT="${BASE_MODEL}" \
BASELINE_KL_CHECKPOINT="${BASE_MODEL}" \
EXPECTED_BASE_SHA256="7d7343a55c68abe67c1779158eb0d8ad209a5969e38acfe8844887367a8318a1" \
EXPECTED_BASE_SIZE=14825679 \
NUM_ENVS=${NUM_ENVS:-4096} \
MAX_ITERATIONS=${MAX_ITERATIONS:-240} \
RUN_NAME=${RUN_NAME:-armhack_stand_low_torque_robust_stage1_from159_20260814} \
SEED=${SEED:-20260814} \
DEVICE=${DEVICE:-cuda:1} \
AGENT_DEVICE=${AGENT_DEVICE:-cuda:1} \
STANCE_MIN_M=0.08 \
STANCE_MAX_M=0.32 \
CLOSE_STANCE_MIN_M=0.08 \
CLOSE_STANCE_MAX_M=0.14 \
CLOSE_STANCE_PROB=0.50 \
POSITION_SCALE_MIN=0.92 \
POSITION_SCALE_MAX=1.08 \
INITIAL_JOINT_VEL_MIN=-0.10 \
INITIAL_JOINT_VEL_MAX=0.10 \
TARGET_STANCE_M=0.30 \
STANCE_SUCCESS_TOLERANCE_M=0.015 \
PUSH_MAX_MPS=0.42 \
PUSH_YAW_MAX_RADPS=0.55 \
PUSH_INTERVAL_MIN_S=2.0 \
PUSH_INTERVAL_MAX_S=4.0 \
EXTERNAL_FORCE_MAX_N=25.0 \
EXTERNAL_TORQUE_MAX_NM=4.0 \
ANKLE_TORQUE_WEIGHT=-1.5e-3 \
LEARNING_RATE=1.5e-5 \
DESIRED_KL=0.008 \
ENTROPY_COEF=0.0004 \
BASELINE_KL_SCALE=0.0003 \
CURRICULUM_STEP_OFFSET=12000 \
bash "${SCRIPT_DIR}/train_g1_armhack_stand_foot_recovery.sh" \
  agent.save_interval=80 \
  "$@"
