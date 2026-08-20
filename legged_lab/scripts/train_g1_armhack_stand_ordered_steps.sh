#!/usr/bin/env bash
# Fine-tune smooth stance recovery into an ordered left-step then right-step task.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
# The project carries the PPO extensions used by this task.  Prepend it so an
# unrelated editable rsl_rl installation cannot silently shadow them.
export PYTHONPATH="${PROJECT_DIR}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
SOURCE_MODEL=${SOURCE_MODEL:-"${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-15_02-12-31_armhack_stand_ordered_steps_strict_swing_gpu1_20260815/model_200.pt"}
BASELINE_MODEL=${BASELINE_MODEL:-"${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-15_02-12-31_armhack_stand_ordered_steps_strict_swing_gpu1_20260815/model_200.pt"}
EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:-67b884d8bb884b7cfa1152e4c4ebb2882dfe03987c7b802cef8fba33e27fcbde}
EXPECTED_SOURCE_SIZE=${EXPECTED_SOURCE_SIZE:-14825679}
ORDERED_STEP_PROFILE=${ORDERED_STEP_PROFILE:-balanced}

case "${ORDERED_STEP_PROFILE}" in
    balanced)
        RUN_NAME_DEFAULT=armhack_stand_ordered_steps_left_skill_balanced_gpu0_20260815
        STEP_PROGRESS_WEIGHT=10.0
        STEP_TARGET_WEIGHT=10.0
        STEP_CLEARANCE_WEIGHT=8.0
        STEP_COMPLETION_WEIGHT=50.0
        STEP_LIFT_WEIGHT=30.0
        ACTIVE_CONTACT_WEIGHT=-4.0
        SINGLE_SUPPORT_WEIGHT=5.0
        STEP_LANDING_WEIGHT=15.0
        SUPPORT_FOOT_WEIGHT=-100.0
        STANCE_L1_WEIGHT=-0.2
        STANCE_EXP_WEIGHT=0.2
        STANCE_SUCCESS_WEIGHT=0.0
        PHASE_ONE_RESET_PROBABILITY=0.40
        CLEARANCE_L2_WEIGHT=-500.0
        UPWARD_VELOCITY_WEIGHT=2.0
        ORDER_VIOLATION_WEIGHT=-10.0
        FINAL_TARGET_WEIGHT=-18.0
        TARGET_ERROR_WEIGHT=-0.4
        LOWER_JOINT_VEL_WEIGHT=-1.0e-3
        LOWER_JOINT_ACC_WEIGHT=-5.0e-7
        LOWER_ACTION_RATE_WEIGHT=-5.0e-3
        FOOT_FORCE_WEIGHT=-0.30
        ANKLE_SEPARATION_SPEED_WEIGHT=-0.20
        TORSO_XY_WEIGHT=-3.0
        TORSO_YAW_WEIGHT=-2.0
        ROOT_XY_WEIGHT=-1.5
        TORSO_NEAR_WEIGHT=-6.0
        TORSO_YAW_NEAR_WEIGHT=-3.0
        PUSH_MAX_PROFILE=0.15
        PUSH_YAW_PROFILE=0.20
        FORCE_MAX_PROFILE=8.0
        TORQUE_MAX_PROFILE=1.5
        LEARNING_RATE_PROFILE=3.0e-5
        DESIRED_KL_PROFILE=0.015
        ENTROPY_PROFILE=0.002
        BASELINE_KL_PROFILE=5.0e-5
        MAX_PELVIS_DISPLACEMENT_M=0.25
        ;;
    strict)
        RUN_NAME_DEFAULT=armhack_stand_ordered_steps_left_skill_strong_gpu1_20260815
        STEP_PROGRESS_WEIGHT=14.0
        STEP_TARGET_WEIGHT=14.0
        STEP_CLEARANCE_WEIGHT=12.0
        STEP_COMPLETION_WEIGHT=80.0
        STEP_LIFT_WEIGHT=50.0
        ACTIVE_CONTACT_WEIGHT=-8.0
        SINGLE_SUPPORT_WEIGHT=8.0
        STEP_LANDING_WEIGHT=25.0
        SUPPORT_FOOT_WEIGHT=-150.0
        STANCE_L1_WEIGHT=-0.2
        STANCE_EXP_WEIGHT=0.2
        STANCE_SUCCESS_WEIGHT=0.0
        PHASE_ONE_RESET_PROBABILITY=0.50
        CLEARANCE_L2_WEIGHT=-1000.0
        UPWARD_VELOCITY_WEIGHT=3.0
        ORDER_VIOLATION_WEIGHT=-16.0
        FINAL_TARGET_WEIGHT=-22.0
        TARGET_ERROR_WEIGHT=-0.2
        LOWER_JOINT_VEL_WEIGHT=-5.0e-4
        LOWER_JOINT_ACC_WEIGHT=-2.0e-7
        LOWER_ACTION_RATE_WEIGHT=-2.0e-3
        FOOT_FORCE_WEIGHT=-0.15
        ANKLE_SEPARATION_SPEED_WEIGHT=-0.10
        TORSO_XY_WEIGHT=-2.0
        TORSO_YAW_WEIGHT=-1.5
        ROOT_XY_WEIGHT=-1.0
        TORSO_NEAR_WEIGHT=-4.0
        TORSO_YAW_NEAR_WEIGHT=-2.0
        PUSH_MAX_PROFILE=0.10
        PUSH_YAW_PROFILE=0.15
        FORCE_MAX_PROFILE=5.0
        TORQUE_MAX_PROFILE=1.0
        LEARNING_RATE_PROFILE=6.0e-5
        DESIRED_KL_PROFILE=0.02
        ENTROPY_PROFILE=0.004
        BASELINE_KL_PROFILE=0.0
        MAX_PELVIS_DISPLACEMENT_M=0.25
        ;;
    stabilize_skill)
        RUN_NAME_DEFAULT=armhack_stand_ordered_steps_stabilize_from_skill150_gpu0_20260815
        STEP_PROGRESS_WEIGHT=10.0
        STEP_TARGET_WEIGHT=15.0
        STEP_CLEARANCE_WEIGHT=6.0
        STEP_COMPLETION_WEIGHT=60.0
        STEP_LIFT_WEIGHT=24.0
        ACTIVE_CONTACT_WEIGHT=-2.5
        SINGLE_SUPPORT_WEIGHT=4.0
        STEP_LANDING_WEIGHT=30.0
        SUPPORT_FOOT_WEIGHT=-180.0
        STANCE_L1_WEIGHT=-0.1
        STANCE_EXP_WEIGHT=0.1
        STANCE_SUCCESS_WEIGHT=0.0
        PHASE_ONE_RESET_PROBABILITY=0.45
        CLEARANCE_L2_WEIGHT=-350.0
        UPWARD_VELOCITY_WEIGHT=2.0
        ORDER_VIOLATION_WEIGHT=-14.0
        FINAL_TARGET_WEIGHT=-30.0
        TARGET_ERROR_WEIGHT=-1.2
        LOWER_JOINT_VEL_WEIGHT=-4.0e-3
        LOWER_JOINT_ACC_WEIGHT=-4.0e-6
        LOWER_ACTION_RATE_WEIGHT=-4.0e-2
        FOOT_FORCE_WEIGHT=-2.0
        ANKLE_SEPARATION_SPEED_WEIGHT=-1.5
        TORSO_XY_WEIGHT=-60.0
        TORSO_YAW_WEIGHT=-10.0
        ROOT_XY_WEIGHT=-30.0
        TORSO_NEAR_WEIGHT=-100.0
        TORSO_YAW_NEAR_WEIGHT=-20.0
        PUSH_MAX_PROFILE=0.25
        PUSH_YAW_PROFILE=0.30
        FORCE_MAX_PROFILE=15.0
        TORQUE_MAX_PROFILE=2.5
        LEARNING_RATE_PROFILE=8.0e-6
        DESIRED_KL_PROFILE=0.008
        ENTROPY_PROFILE=0.0005
        BASELINE_KL_PROFILE=1.0e-4
        MAX_PELVIS_DISPLACEMENT_M=0.20
        ;;
    stabilize_safe)
        RUN_NAME_DEFAULT=armhack_stand_ordered_steps_stabilize_from_skill100_gpu1_20260815
        STEP_PROGRESS_WEIGHT=12.0
        STEP_TARGET_WEIGHT=18.0
        STEP_CLEARANCE_WEIGHT=8.0
        STEP_COMPLETION_WEIGHT=70.0
        STEP_LIFT_WEIGHT=30.0
        ACTIVE_CONTACT_WEIGHT=-3.0
        SINGLE_SUPPORT_WEIGHT=5.0
        STEP_LANDING_WEIGHT=40.0
        SUPPORT_FOOT_WEIGHT=-250.0
        STANCE_L1_WEIGHT=-0.1
        STANCE_EXP_WEIGHT=0.1
        STANCE_SUCCESS_WEIGHT=0.0
        PHASE_ONE_RESET_PROBABILITY=0.50
        CLEARANCE_L2_WEIGHT=-450.0
        UPWARD_VELOCITY_WEIGHT=3.0
        ORDER_VIOLATION_WEIGHT=-16.0
        FINAL_TARGET_WEIGHT=-35.0
        TARGET_ERROR_WEIGHT=-1.0
        LOWER_JOINT_VEL_WEIGHT=-3.0e-3
        LOWER_JOINT_ACC_WEIGHT=-3.0e-6
        LOWER_ACTION_RATE_WEIGHT=-3.0e-2
        FOOT_FORCE_WEIGHT=-1.5
        ANKLE_SEPARATION_SPEED_WEIGHT=-1.0
        TORSO_XY_WEIGHT=-100.0
        TORSO_YAW_WEIGHT=-15.0
        ROOT_XY_WEIGHT=-50.0
        TORSO_NEAR_WEIGHT=-150.0
        TORSO_YAW_NEAR_WEIGHT=-30.0
        PUSH_MAX_PROFILE=0.22
        PUSH_YAW_PROFILE=0.28
        FORCE_MAX_PROFILE=12.0
        TORQUE_MAX_PROFILE=2.0
        LEARNING_RATE_PROFILE=1.2e-5
        DESIRED_KL_PROFILE=0.010
        ENTROPY_PROFILE=0.0008
        BASELINE_KL_PROFILE=3.0e-4
        MAX_PELVIS_DISPLACEMENT_M=0.15
        ;;
    stabilize_lift_safe)
        # Preserve a visible airborne step while retaining most of the
        # velocity/acceleration/contact-force regularization of stabilize_safe.
        # Phase-reset sampling covers the left step, right step, and final
        # double-support hold in one policy.
        RUN_NAME_DEFAULT=armhack_stand_ordered_steps_stabilize_lift_safe_gpu0_20260815
        STEP_PROGRESS_WEIGHT=16.0
        STEP_TARGET_WEIGHT=22.0
        STEP_CLEARANCE_WEIGHT=18.0
        STEP_COMPLETION_WEIGHT=100.0
        STEP_LIFT_WEIGHT=80.0
        ACTIVE_CONTACT_WEIGHT=-6.0
        SINGLE_SUPPORT_WEIGHT=8.0
        STEP_LANDING_WEIGHT=50.0
        SUPPORT_FOOT_WEIGHT=-250.0
        STANCE_L1_WEIGHT=-0.1
        STANCE_EXP_WEIGHT=0.1
        STANCE_SUCCESS_WEIGHT=0.0
        PHASE_ONE_RESET_PROBABILITY=0.35
        CLEARANCE_L2_WEIGHT=-700.0
        UPWARD_VELOCITY_WEIGHT=4.0
        ORDER_VIOLATION_WEIGHT=-20.0
        FINAL_TARGET_WEIGHT=-100.0
        TARGET_ERROR_WEIGHT=-0.8
        LOWER_JOINT_VEL_WEIGHT=-2.0e-3
        LOWER_JOINT_ACC_WEIGHT=-2.0e-6
        LOWER_ACTION_RATE_WEIGHT=-2.0e-2
        FOOT_FORCE_WEIGHT=-1.0
        ANKLE_SEPARATION_SPEED_WEIGHT=-0.7
        TORSO_XY_WEIGHT=-100.0
        TORSO_YAW_WEIGHT=-15.0
        ROOT_XY_WEIGHT=-50.0
        TORSO_NEAR_WEIGHT=-150.0
        TORSO_YAW_NEAR_WEIGHT=-30.0
        PUSH_MAX_PROFILE=0.22
        PUSH_YAW_PROFILE=0.28
        FORCE_MAX_PROFILE=12.0
        TORQUE_MAX_PROFILE=2.0
        LEARNING_RATE_PROFILE=4.0e-6
        DESIRED_KL_PROFILE=0.010
        ENTROPY_PROFILE=0.0005
        BASELINE_KL_PROFILE=5.0e-4
        MAX_PELVIS_DISPLACEMENT_M=0.15
        ;;
    *)
        echo "Error: unsupported ORDERED_STEP_PROFILE=${ORDERED_STEP_PROFILE}." >&2
        exit 1
        ;;
esac

PHASE_ONE_RESET_PROBABILITY=${PHASE_ONE_RESET_PROBABILITY_OVERRIDE:-${PHASE_ONE_RESET_PROBABILITY}}
if [[ "${ORDERED_STEP_PROFILE}" == "stabilize_lift_safe" ]]; then
    PHASE_TWO_RESET_PROBABILITY_DEFAULT=0.25
    ACTIVE_FOOT_SPEED_WEIGHT_DEFAULT=-25.0
    FINAL_DISTANCE_WEIGHT_DEFAULT=30.0
else
    PHASE_TWO_RESET_PROBABILITY_DEFAULT=0.0
    ACTIVE_FOOT_SPEED_WEIGHT_DEFAULT=-10.0
    FINAL_DISTANCE_WEIGHT_DEFAULT=10.0
fi
PHASE_TWO_RESET_PROBABILITY=${PHASE_TWO_RESET_PROBABILITY_OVERRIDE:-${PHASE_TWO_RESET_PROBABILITY_DEFAULT}}
ACTIVE_FOOT_SPEED_WEIGHT=${ACTIVE_FOOT_SPEED_WEIGHT_OVERRIDE:-${ACTIVE_FOOT_SPEED_WEIGHT_DEFAULT}}
FINAL_DISTANCE_WEIGHT=${FINAL_DISTANCE_WEIGHT_OVERRIDE:-${FINAL_DISTANCE_WEIGHT_DEFAULT}}
BASELINE_KL_PROFILE=${BASELINE_KL_SCALE_OVERRIDE:-${BASELINE_KL_PROFILE}}
LEARNING_RATE_PROFILE=${LEARNING_RATE_OVERRIDE:-${LEARNING_RATE_PROFILE}}
DESIRED_KL_PROFILE=${DESIRED_KL_OVERRIDE:-${DESIRED_KL_PROFILE}}
ENTROPY_PROFILE=${ENTROPY_COEF_OVERRIDE:-${ENTROPY_PROFILE}}
STEP_CLEARANCE_WEIGHT=${STEP_CLEARANCE_WEIGHT_OVERRIDE:-${STEP_CLEARANCE_WEIGHT}}
STEP_LIFT_WEIGHT=${STEP_LIFT_WEIGHT_OVERRIDE:-${STEP_LIFT_WEIGHT}}
ACTIVE_CONTACT_WEIGHT=${ACTIVE_CONTACT_WEIGHT_OVERRIDE:-${ACTIVE_CONTACT_WEIGHT}}
CLEARANCE_L2_WEIGHT=${CLEARANCE_L2_WEIGHT_OVERRIDE:-${CLEARANCE_L2_WEIGHT}}
FINAL_TARGET_WEIGHT=${FINAL_TARGET_WEIGHT_OVERRIDE:-${FINAL_TARGET_WEIGHT}}
UPWARD_VELOCITY_WEIGHT=${UPWARD_VELOCITY_WEIGHT_OVERRIDE:-${UPWARD_VELOCITY_WEIGHT}}
FEET_SLIDE_WEIGHT=${FEET_SLIDE_WEIGHT_OVERRIDE:--1.0}
ASYMMETRIC_SUPPORT_PROBABILITY=${ASYMMETRIC_SUPPORT_PROBABILITY:-0.0}
MIRROR_PHASE_ONE_KL=${MIRROR_PHASE_ONE_KL:-False}
POLICY_NOISE_STD_OVERRIDE=${POLICY_NOISE_STD_OVERRIDE:-0.20}
if [[ "${ORDERED_STEP_PROFILE}" == "stabilize_lift_safe" ]]; then
    MIN_CLEARANCE_M_DEFAULT=0.030
    TARGET_CLEARANCE_M_DEFAULT=0.050
    MAX_CLEARANCE_M_DEFAULT=0.100
    LANDING_TOLERANCE_M_DEFAULT=0.045
else
    MIN_CLEARANCE_M_DEFAULT=0.035
    TARGET_CLEARANCE_M_DEFAULT=0.055
    MAX_CLEARANCE_M_DEFAULT=0.110
    LANDING_TOLERANCE_M_DEFAULT=0.055
fi
MIN_CLEARANCE_M=${MIN_CLEARANCE_M:-${MIN_CLEARANCE_M_DEFAULT}}
TARGET_CLEARANCE_M=${TARGET_CLEARANCE_M:-${TARGET_CLEARANCE_M_DEFAULT}}
MAX_CLEARANCE_M=${MAX_CLEARANCE_M:-${MAX_CLEARANCE_M_DEFAULT}}
LANDING_TOLERANCE_M=${LANDING_TOLERANCE_M:-${LANDING_TOLERANCE_M_DEFAULT}}

BACK_POSE='[0.91,0.91,0.52,-0.52,0.11,-0.11,0.01,0.01,-0.12,0.12,-1.03,-1.03,0.01,-0.01]'
DOWN_POSE='[0.2504,0.2504,0.265,-0.265,-0.0919,0.0919,0.8356,0.8356,0.0031,-0.0031,0.0104,0.0104,-0.0102,0.0102]'
FRONT_POSE='[0.27,0.27,0.79,-0.79,-0.22,0.22,-0.49,-0.49,0.85,-0.85,0.4,0.4,0.05,-0.05]'

echo "Ordered-step profile: ${ORDERED_STEP_PROFILE}"
echo "Sequence rewards    : progress=${STEP_PROGRESS_WEIGHT} target=${STEP_TARGET_WEIGHT} clearance=${STEP_CLEARANCE_WEIGHT}"
echo "                      lift=${STEP_LIFT_WEIGHT} completion=${STEP_COMPLETION_WEIGHT} order=${ORDER_VIOLATION_WEIGHT} final=${FINAL_TARGET_WEIGHT}"
echo "                      active_contact=${ACTIVE_CONTACT_WEIGHT} single_support=${SINGLE_SUPPORT_WEIGHT}"
echo "                      landing=${STEP_LANDING_WEIGHT}"
echo "                      support_lock=${SUPPORT_FOOT_WEIGHT}"
echo "                      phase1_reset_probability=${PHASE_ONE_RESET_PROBABILITY}"
echo "                      phase2_reset_probability=${PHASE_TWO_RESET_PROBABILITY}"
echo "                      asymmetric_support_probability=${ASYMMETRIC_SUPPORT_PROBABILITY}"
echo "                      dense_clearance_l2=${CLEARANCE_L2_WEIGHT}"
echo "                      upward_velocity=${UPWARD_VELOCITY_WEIGHT}"
echo "                      active_foot_speed=${ACTIVE_FOOT_SPEED_WEIGHT}"
echo "                      mirror_phase_one_kl=${MIRROR_PHASE_ONE_KL}"
echo "                      policy_noise_std=${POLICY_NOISE_STD_OVERRIDE}"
echo "                      clearance min/target/max=${MIN_CLEARANCE_M}/${TARGET_CLEARANCE_M}/${MAX_CLEARANCE_M} m"
echo "                      landing_tolerance=${LANDING_TOLERANCE_M} m"
echo "                      final_30cm_distance=${FINAL_DISTANCE_WEIGHT}"

BASE_CHECKPOINT="${SOURCE_MODEL}" \
BASELINE_KL_CHECKPOINT="${BASELINE_MODEL}" \
EXPECTED_BASE_SHA256="${EXPECTED_SOURCE_SHA256}" \
EXPECTED_BASE_SIZE="${EXPECTED_SOURCE_SIZE}" \
NUM_ENVS=${NUM_ENVS:-4096} \
MAX_ITERATIONS=${MAX_ITERATIONS:-600} \
RUN_NAME=${RUN_NAME:-${RUN_NAME_DEFAULT}} \
SEED=${SEED:-20260819} \
DEVICE=${DEVICE:-cuda:0} \
AGENT_DEVICE=${AGENT_DEVICE:-${DEVICE}} \
STANCE_MIN_M=0.08 \
STANCE_MAX_M=0.38 \
CLOSE_STANCE_MIN_M=0.08 \
CLOSE_STANCE_MAX_M=0.14 \
CLOSE_STANCE_PROB=0.50 \
POSITION_SCALE_MIN=0.92 \
POSITION_SCALE_MAX=1.08 \
INITIAL_JOINT_VEL_MIN=-0.10 \
INITIAL_JOINT_VEL_MAX=0.10 \
TARGET_STANCE_M=0.30 \
STANCE_SUCCESS_TOLERANCE_M=0.018 \
PUSH_MAX_MPS="${PUSH_MAX_PROFILE}" \
PUSH_YAW_MAX_RADPS="${PUSH_YAW_PROFILE}" \
PUSH_INTERVAL_MIN_S=3.0 \
PUSH_INTERVAL_MAX_S=5.0 \
EXTERNAL_FORCE_MAX_N="${FORCE_MAX_PROFILE}" \
EXTERNAL_TORQUE_MAX_NM="${TORQUE_MAX_PROFILE}" \
ANKLE_TORQUE_WEIGHT=-1.8e-3 \
LEARNING_RATE="${LEARNING_RATE_PROFILE}" \
DESIRED_KL="${DESIRED_KL_PROFILE}" \
ENTROPY_COEF="${ENTROPY_PROFILE}" \
BASELINE_KL_SCALE="${BASELINE_KL_PROFILE}" \
CURRICULUM_STEP_OFFSET=0 \
bash "${SCRIPT_DIR}/train_g1_armhack_stand_foot_recovery.sh" \
  agent.save_interval="${SAVE_INTERVAL:-50}" \
  env.events.reset_robot_joints.params.asymmetric_support_probability="${ASYMMETRIC_SUPPORT_PROBABILITY}" \
  "env.upper_body_perturbation.random_extra_pose_names=[back,down,front]" \
  "env.upper_body_perturbation.random_extra_pose_set=[${BACK_POSE},${DOWN_POSE},${FRONT_POSE}]" \
  "env.upper_body_perturbation.random_extra_pose_weights=[1.0,1.0,1.0]" \
  env.upper_body_perturbation.random_extra_pose_probability=0.30 \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_distance_l1=[[0,${STANCE_L1_WEIGHT}],[12000,${STANCE_L1_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_distance_exp=[[0,${STANCE_EXP_WEIGHT}],[12000,${STANCE_EXP_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.ankle_distance_success=[[0,${STANCE_SUCCESS_WEIGHT}],[12000,${STANCE_SUCCESS_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_xy_position_l2=[[0,${TORSO_XY_WEIGHT}],[12000,${TORSO_XY_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.torso_yaw_l2=[[0,${TORSO_YAW_WEIGHT}],[12000,${TORSO_YAW_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.root_xy_position_l2=[[0,${ROOT_XY_WEIGHT}],[12000,${ROOT_XY_WEIGHT}]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.double_support=[[0,0.0],[12000,0.0]]" \
  "env.curriculum.stance_recovery.params.reward_weight_schedules.feet_slide=[[0,${FEET_SLIDE_WEIGHT}],[12000,${FEET_SLIDE_WEIGHT}]]" \
  env.rewards.torso_xy_position_near_stance_l2.weight="${TORSO_NEAR_WEIGHT}" \
  env.rewards.torso_yaw_near_stance_l2.weight="${TORSO_YAW_NEAR_WEIGHT}" \
  env.rewards.lower_body_joint_vel_l2.weight="${LOWER_JOINT_VEL_WEIGHT}" \
  env.rewards.lower_body_joint_acc_l2.weight="${LOWER_JOINT_ACC_WEIGHT}" \
  env.rewards.lower_body_action_rate_l2.weight="${LOWER_ACTION_RATE_WEIGHT}" \
  env.rewards.lower_body_position_target_error_l2.weight="${TARGET_ERROR_WEIGHT}" \
  env.rewards.ankle_transition_torques_l2.weight=-2.0e-4 \
  env.rewards.ankle_separation_speed_l2.weight="${ANKLE_SEPARATION_SPEED_WEIGHT}" \
  env.rewards.foot_contact_force_excess_l2.weight="${FOOT_FORCE_WEIGHT}" \
  env.rewards.foot_contact_force_excess_l2.params.threshold_n=240.0 \
  env.rewards.sequential_foot_step_progress.weight="${STEP_PROGRESS_WEIGHT}" \
  env.rewards.sequential_foot_step_progress.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_progress.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_target_exp.weight="${STEP_TARGET_WEIGHT}" \
  env.rewards.sequential_foot_step_target_exp.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_target_exp.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_clearance_exp.weight="${STEP_CLEARANCE_WEIGHT}" \
  env.rewards.sequential_foot_step_clearance_exp.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_clearance_exp.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_clearance_exp.params.target_clearance_m="${TARGET_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_contact.weight="${ACTIVE_CONTACT_WEIGHT}" \
  env.rewards.sequential_active_foot_contact.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_contact.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_active_foot_clearance_l2.weight="${CLEARANCE_L2_WEIGHT}" \
  env.rewards.sequential_active_foot_clearance_l2.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_clearance_l2.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_active_foot_clearance_l2.params.target_clearance_m="${TARGET_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_clearance_l2.params.max_clearance_m="${MAX_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_upward_velocity.weight="${UPWARD_VELOCITY_WEIGHT}" \
  env.rewards.sequential_active_foot_upward_velocity.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_upward_velocity.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_active_foot_velocity_l2.weight="${ACTIVE_FOOT_SPEED_WEIGHT}" \
  env.rewards.sequential_active_foot_velocity_l2.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_velocity_l2.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_active_foot_single_support.weight="${SINGLE_SUPPORT_WEIGHT}" \
  env.rewards.sequential_active_foot_single_support.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_active_foot_single_support.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_landing_exp.weight="${STEP_LANDING_WEIGHT}" \
  env.rewards.sequential_foot_step_landing_exp.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_landing_exp.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_completion.weight="${STEP_COMPLETION_WEIGHT}" \
  env.rewards.sequential_foot_step_completion.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_completion.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_lift.weight="${STEP_LIFT_WEIGHT}" \
  env.rewards.sequential_foot_step_lift.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_lift.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_step_order_violation.weight="${ORDER_VIOLATION_WEIGHT}" \
  env.rewards.sequential_foot_step_order_violation.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_step_order_violation.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_foot_final_target_l2.weight="${FINAL_TARGET_WEIGHT}" \
  env.rewards.sequential_foot_final_target_l2.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_foot_final_target_l2.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_final_ankle_distance_exp.weight="${FINAL_DISTANCE_WEIGHT}" \
  env.rewards.sequential_final_ankle_distance_exp.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_final_ankle_distance_exp.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.rewards.sequential_support_foot_drift_l2.weight="${SUPPORT_FOOT_WEIGHT}" \
  env.rewards.sequential_support_foot_drift_l2.params.min_clearance_m="${MIN_CLEARANCE_M}" \
  env.rewards.sequential_support_foot_drift_l2.params.landing_tolerance_m="${LANDING_TOLERANCE_M}" \
  env.events.reset_robot_joints.params.phase_one_probability="${PHASE_ONE_RESET_PROBABILITY}" \
  env.events.reset_robot_joints.params.phase_two_probability="${PHASE_TWO_RESET_PROBABILITY}" \
  env.terminations.sequential_pelvis_xy_out_of_bounds.params.max_displacement_m="${MAX_PELVIS_DISPLACEMENT_M}" \
  agent.algorithm.baseline_kl_cfg.exempt_obs_index=94 \
  agent.algorithm.baseline_kl_cfg.exempt_obs_threshold=0.5 \
  agent.algorithm.baseline_kl_cfg.mirror_phase_one="${MIRROR_PHASE_ONE_KL}" \
  agent.algorithm.baseline_kl_cfg.lift_obs_index=95 \
  agent.policy_only_noise_std_override="${POLICY_NOISE_STD_OVERRIDE}" \
  "$@"
