#!/usr/bin/env bash
# Low-torque robustness/pose-return polish from stage-1 model_239.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BASE_RUN="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-14_18-12-29_armhack_stand_low_torque_robust_stage1_from159_20260814"
BASE_MODEL="${BASE_RUN}/model_239.pt"

BASE_CHECKPOINT="${BASE_MODEL}" \
BASELINE_KL_CHECKPOINT="${BASE_MODEL}" \
EXPECTED_BASE_SHA256="7650a78a5436c3b7cf2243544c4b01b1ee1a85b5cae8eb2024de84d65eb7ad6a" \
EXPECTED_BASE_SIZE=14825679 \
NUM_ENVS=${NUM_ENVS:-4096} \
MAX_ITERATIONS=${MAX_ITERATIONS:-160} \
RUN_NAME=${RUN_NAME:-armhack_stand_low_torque_robust_pose_return_stage2_from239_20260814} \
SEED=${SEED:-20260815} \
DEVICE=${DEVICE:-cuda:1} \
AGENT_DEVICE=${AGENT_DEVICE:-cuda:1} \
STANCE_MIN_M=0.08 \
STANCE_MAX_M=0.32 \
CLOSE_STANCE_MIN_M=0.08 \
CLOSE_STANCE_MAX_M=0.14 \
CLOSE_STANCE_PROB=0.55 \
POSITION_SCALE_MIN=0.90 \
POSITION_SCALE_MAX=1.10 \
INITIAL_JOINT_VEL_MIN=-0.15 \
INITIAL_JOINT_VEL_MAX=0.15 \
TARGET_STANCE_M=0.29 \
STANCE_SUCCESS_TOLERANCE_M=0.012 \
PUSH_MAX_MPS=0.45 \
PUSH_YAW_MAX_RADPS=0.60 \
PUSH_INTERVAL_MIN_S=2.0 \
PUSH_INTERVAL_MAX_S=4.0 \
EXTERNAL_FORCE_MAX_N=28.0 \
EXTERNAL_TORQUE_MAX_NM=4.5 \
ANKLE_TORQUE_WEIGHT=-1.8e-3 \
LEARNING_RATE=8.0e-6 \
DESIRED_KL=0.006 \
ENTROPY_COEF=0.0003 \
BASELINE_KL_SCALE=0.0004 \
CURRICULUM_STEP_OFFSET=12000 \
bash "${SCRIPT_DIR}/train_g1_armhack_stand_foot_recovery.sh" \
  agent.save_interval=40 \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_xy_position_l2=[[0,-7.0],[12000,-7.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_yaw_l2=[[0,-4.0],[12000,-4.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.root_xy_position_l2=[[0,-3.5],[12000,-3.5]]" \
  env.rewards.torso_xy_position_near_stance_l2.weight=-16.0 \
  env.rewards.torso_yaw_near_stance_l2.weight=-8.0 \
  "$@"
