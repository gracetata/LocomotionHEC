#!/usr/bin/env bash
# Two-profile smooth stance-recovery continuation from the validated 2000-iteration model.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BASE_RUN="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-14_18-32-52_armhack_stand_low_torque_robust_explicit_3pose_2000_from_stage2_20260814"
BASE_MODEL="${BASE_RUN}/model_1999.pt"
SMOOTH_PROFILE=${SMOOTH_PROFILE:-kinematic}
LOWER_POSITION_TARGET_ERROR_WEIGHT=${LOWER_POSITION_TARGET_ERROR_WEIGHT:-0.0}
ANKLE_TRANSITION_TORQUE_WEIGHT=${ANKLE_TRANSITION_TORQUE_WEIGHT:-0.0}

case "${SMOOTH_PROFILE}" in
    kinematic)
        RUN_NAME_DEFAULT=armhack_stand_smooth_kinematic_gpu0_from1999_20260815
        LOWER_JOINT_VEL_WEIGHT=-6.0e-3
        LOWER_JOINT_ACC_WEIGHT=-1.5e-6
        LOWER_ACTION_RATE_WEIGHT=-2.0e-2
        ANKLE_SEPARATION_SPEED_WEIGHT=-8.0e-1
        FOOT_FORCE_EXCESS_WEIGHT=-8.0e-1
        FOOT_FORCE_THRESHOLD_N=280.0
        ;;
    impact)
        RUN_NAME_DEFAULT=armhack_stand_smooth_impact_gpu1_from1999_20260815
        LOWER_JOINT_VEL_WEIGHT=-3.0e-3
        LOWER_JOINT_ACC_WEIGHT=-4.0e-6
        LOWER_ACTION_RATE_WEIGHT=-3.5e-2
        ANKLE_SEPARATION_SPEED_WEIGHT=-1.5
        FOOT_FORCE_EXCESS_WEIGHT=-2.0
        FOOT_FORCE_THRESHOLD_N=240.0
        ;;
    *)
        echo "Error: SMOOTH_PROFILE must be kinematic or impact; got ${SMOOTH_PROFILE}." >&2
        exit 1
        ;;
esac

BACK_POSE='[0.91,0.91,0.52,-0.52,0.11,-0.11,0.01,0.01,-0.12,0.12,-1.03,-1.03,0.01,-0.01]'
DOWN_POSE='[0.2504,0.2504,0.265,-0.265,-0.0919,0.0919,0.8356,0.8356,0.0031,-0.0031,0.0104,0.0104,-0.0102,0.0102]'
FRONT_POSE='[0.27,0.27,0.79,-0.79,-0.22,0.22,-0.49,-0.49,0.85,-0.85,0.4,0.4,0.05,-0.05]'

echo "Smooth profile : ${SMOOTH_PROFILE}"
echo "Smooth rewards : qvel=${LOWER_JOINT_VEL_WEIGHT} qacc=${LOWER_JOINT_ACC_WEIGHT} action_rate=${LOWER_ACTION_RATE_WEIGHT}"
echo "                 target_error=${LOWER_POSITION_TARGET_ERROR_WEIGHT} ankle_speed=${ANKLE_SEPARATION_SPEED_WEIGHT}"
echo "                 transition_ankle_torque=${ANKLE_TRANSITION_TORQUE_WEIGHT}"
echo "                 foot_force=${FOOT_FORCE_EXCESS_WEIGHT} above ${FOOT_FORCE_THRESHOLD_N}N"

BASE_CHECKPOINT="${BASE_MODEL}" \
BASELINE_KL_CHECKPOINT="${BASE_MODEL}" \
EXPECTED_BASE_SHA256="9ab48719840c98f1332693a56f58ed069463c0670737e339b90411985484a729" \
EXPECTED_BASE_SIZE=14825781 \
NUM_ENVS=${NUM_ENVS:-4096} \
MAX_ITERATIONS=${MAX_ITERATIONS:-1200} \
RUN_NAME=${RUN_NAME:-${RUN_NAME_DEFAULT}} \
SEED=${SEED:-20260817} \
DEVICE=${DEVICE:-cuda:0} \
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
LEARNING_RATE=8.0e-6 \
DESIRED_KL=0.008 \
ENTROPY_COEF=0.0002 \
BASELINE_KL_SCALE=0.0003 \
CURRICULUM_STEP_OFFSET=12000 \
bash "${SCRIPT_DIR}/train_g1_armhack_stand_foot_recovery.sh" \
  agent.save_interval=200 \
  "env.upper_body_perturbation.random_extra_pose_names=[back,down,front]" \
  "env.upper_body_perturbation.random_extra_pose_set=[${BACK_POSE},${DOWN_POSE},${FRONT_POSE}]" \
  "env.upper_body_perturbation.random_extra_pose_weights=[1.0,1.0,1.0]" \
  env.upper_body_perturbation.random_extra_pose_probability=0.30 \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_xy_position_l2=[[0,-7.0],[12000,-7.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_yaw_l2=[[0,-4.0],[12000,-4.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.root_xy_position_l2=[[0,-3.5],[12000,-3.5]]" \
  env.rewards.torso_xy_position_near_stance_l2.weight=-16.0 \
  env.rewards.torso_yaw_near_stance_l2.weight=-8.0 \
  env.rewards.lower_body_joint_vel_l2.weight="${LOWER_JOINT_VEL_WEIGHT}" \
  env.rewards.lower_body_joint_acc_l2.weight="${LOWER_JOINT_ACC_WEIGHT}" \
  env.rewards.lower_body_action_rate_l2.weight="${LOWER_ACTION_RATE_WEIGHT}" \
  env.rewards.lower_body_position_target_error_l2.weight="${LOWER_POSITION_TARGET_ERROR_WEIGHT}" \
  env.rewards.ankle_transition_torques_l2.weight="${ANKLE_TRANSITION_TORQUE_WEIGHT}" \
  env.rewards.ankle_separation_speed_l2.weight="${ANKLE_SEPARATION_SPEED_WEIGHT}" \
  env.rewards.foot_contact_force_excess_l2.weight="${FOOT_FORCE_EXCESS_WEIGHT}" \
  env.rewards.foot_contact_force_excess_l2.params.threshold_n="${FOOT_FORCE_THRESHOLD_N}" \
  "$@"
