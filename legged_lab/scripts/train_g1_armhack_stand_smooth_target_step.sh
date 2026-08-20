#!/usr/bin/env bash
# Second-stage continuation that suppresses the reset-time lower-body PD target step.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SOURCE_MODEL="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-15_00-47-00_armhack_stand_smooth_impact_gpu1_from1999_20260815/model_1199.pt"
TARGET_STEP_PROFILE=${TARGET_STEP_PROFILE:-moderate}

case "${TARGET_STEP_PROFILE}" in
    moderate)
        RUN_NAME_DEFAULT=armhack_stand_smooth_target_step_moderate_gpu1_from_impact1199_20260815
        TARGET_ERROR_WEIGHT=-2.0
        TRANSITION_ANKLE_TORQUE_WEIGHT=-2.0e-4
        LOWER_JOINT_VEL_WEIGHT=-4.0e-3
        LOWER_JOINT_ACC_WEIGHT=-4.0e-6
        LOWER_ACTION_RATE_WEIGHT=-4.0e-2
        ANKLE_SEPARATION_SPEED_WEIGHT=-2.0
        FOOT_FORCE_EXCESS_WEIGHT=-2.0
        TORSO_NEAR_WEIGHT=-18.0
        ;;
    strong)
        RUN_NAME_DEFAULT=armhack_stand_smooth_target_step_strong_gpu0_from_impact1199_20260815
        TARGET_ERROR_WEIGHT=-5.0
        TRANSITION_ANKLE_TORQUE_WEIGHT=-3.0e-4
        LOWER_JOINT_VEL_WEIGHT=-5.0e-3
        LOWER_JOINT_ACC_WEIGHT=-6.0e-6
        LOWER_ACTION_RATE_WEIGHT=-5.0e-2
        ANKLE_SEPARATION_SPEED_WEIGHT=-3.0
        FOOT_FORCE_EXCESS_WEIGHT=-2.5
        TORSO_NEAR_WEIGHT=-20.0
        ;;
    *)
        echo "Error: TARGET_STEP_PROFILE must be moderate or strong; got ${TARGET_STEP_PROFILE}." >&2
        exit 1
        ;;
esac

BACK_POSE='[0.91,0.91,0.52,-0.52,0.11,-0.11,0.01,0.01,-0.12,0.12,-1.03,-1.03,0.01,-0.01]'
DOWN_POSE='[0.2504,0.2504,0.265,-0.265,-0.0919,0.0919,0.8356,0.8356,0.0031,-0.0031,0.0104,0.0104,-0.0102,0.0102]'
FRONT_POSE='[0.27,0.27,0.79,-0.79,-0.22,0.22,-0.49,-0.49,0.85,-0.85,0.4,0.4,0.05,-0.05]'

echo "Target-step profile: ${TARGET_STEP_PROFILE}"
echo "Target/current gap : ${TARGET_ERROR_WEIGHT}"
echo "Transition ankle   : ${TRANSITION_ANKLE_TORQUE_WEIGHT}"

BASE_CHECKPOINT="${SOURCE_MODEL}" \
BASELINE_KL_CHECKPOINT="${SOURCE_MODEL}" \
EXPECTED_BASE_SHA256=9319458446629356c8a268bd3994b2e003f49775da7a4100f4e9161b62525d08 \
EXPECTED_BASE_SIZE=14825781 \
NUM_ENVS=${NUM_ENVS:-4096} \
MAX_ITERATIONS=${MAX_ITERATIONS:-300} \
RUN_NAME=${RUN_NAME:-${RUN_NAME_DEFAULT}} \
SEED=${SEED:-20260818} \
DEVICE=${DEVICE:-cuda:1} \
AGENT_DEVICE=${AGENT_DEVICE:-${DEVICE}} \
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
LEARNING_RATE=6.0e-6 \
DESIRED_KL=0.008 \
ENTROPY_COEF=0.0001 \
BASELINE_KL_SCALE=0.0004 \
CURRICULUM_STEP_OFFSET=12000 \
bash "${SCRIPT_DIR}/train_g1_armhack_stand_foot_recovery.sh" \
  agent.save_interval=100 \
  "env.upper_body_perturbation.random_extra_pose_names=[back,down,front]" \
  "env.upper_body_perturbation.random_extra_pose_set=[${BACK_POSE},${DOWN_POSE},${FRONT_POSE}]" \
  "env.upper_body_perturbation.random_extra_pose_weights=[1.0,1.0,1.0]" \
  env.upper_body_perturbation.random_extra_pose_probability=0.30 \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_distance_l1=[[0,-10.0],[12000,-10.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_distance_exp=[[0,7.0],[12000,7.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_distance_success=[[0,7.0],[12000,7.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_xy_position_l2=[[0,-8.0],[12000,-8.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_yaw_l2=[[0,-4.5],[12000,-4.5]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.root_xy_position_l2=[[0,-3.5],[12000,-3.5]]" \
  env.rewards.torso_xy_position_near_stance_l2.weight="${TORSO_NEAR_WEIGHT}" \
  env.rewards.torso_yaw_near_stance_l2.weight=-9.0 \
  env.rewards.lower_body_joint_vel_l2.weight="${LOWER_JOINT_VEL_WEIGHT}" \
  env.rewards.lower_body_joint_acc_l2.weight="${LOWER_JOINT_ACC_WEIGHT}" \
  env.rewards.lower_body_action_rate_l2.weight="${LOWER_ACTION_RATE_WEIGHT}" \
  env.rewards.lower_body_position_target_error_l2.weight="${TARGET_ERROR_WEIGHT}" \
  env.rewards.ankle_transition_torques_l2.weight="${TRANSITION_ANKLE_TORQUE_WEIGHT}" \
  env.rewards.ankle_separation_speed_l2.weight="${ANKLE_SEPARATION_SPEED_WEIGHT}" \
  env.rewards.foot_contact_force_excess_l2.weight="${FOOT_FORCE_EXCESS_WEIGHT}" \
  env.rewards.foot_contact_force_excess_l2.params.threshold_n=240.0 \
  "$@"
