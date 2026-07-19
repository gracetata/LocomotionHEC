#!/usr/bin/env bash
# Visualize nominal, initial-recovery, or full-disturbance behavior in Isaac Sim.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

CHECKPOINT=${CHECKPOINT:-"${PROJECT_ROOT}/checkpoint/stand/model_2999.pt"}
MODE=${MODE:-recovery}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-False}
MAX_STEPS=${MAX_STEPS:-}

[[ -f "${CHECKPOINT}" ]] || { echo "Error: CHECKPOINT not found: ${CHECKPOINT}" >&2; exit 1; }

COMMON_OVERRIDES=(
    env.commands.base_velocity.ranges.lin_vel_x='[0.0,0.0]'
    env.commands.base_velocity.ranges.lin_vel_y='[0.0,0.0]'
    env.commands.base_velocity.ranges.ang_vel_z='[0.0,0.0]'
)

case "${MODE}" in
    nominal)
        RANDOMIZATION_STRENGTH=0
        MODE_OVERRIDES=(
            env.events.reset_leg_joints_with_noise.params.position_range='[0.0,0.0]'
            env.events.reset_leg_joints_with_noise.params.velocity_range='[0.0,0.0]'
            env.events.reset_waist_joints_with_noise.params.position_range='[0.0,0.0]'
            env.events.reset_waist_joints_with_noise.params.velocity_range='[0.0,0.0]'
            env.events.reset_arm_joints_with_noise.params.position_range='[0.0,0.0]'
            env.events.reset_arm_joints_with_noise.params.velocity_range='[0.0,0.0]'
            env.events.reset_base.params.pose_range.x='[0.0,0.0]'
            env.events.reset_base.params.pose_range.y='[0.0,0.0]'
            env.events.reset_base.params.pose_range.z='[0.0,0.0]'
            env.events.reset_base.params.pose_range.roll='[0.0,0.0]'
            env.events.reset_base.params.pose_range.pitch='[0.0,0.0]'
            env.events.reset_base.params.pose_range.yaw='[0.0,0.0]'
            env.events.reset_base.params.velocity_range.x='[0.0,0.0]'
            env.events.reset_base.params.velocity_range.y='[0.0,0.0]'
            env.events.reset_base.params.velocity_range.z='[0.0,0.0]'
            env.events.reset_base.params.velocity_range.roll='[0.0,0.0]'
            env.events.reset_base.params.velocity_range.pitch='[0.0,0.0]'
            env.events.reset_base.params.velocity_range.yaw='[0.0,0.0]'
            env.events.random_torso_external_wrench=null
            env.events.random_pelvis_external_wrench=null
            env.events.random_arm_external_wrench=null
            env.events.random_leg_external_wrench=null
        )
        ;;
    recovery)
        RANDOMIZATION_STRENGTH=0
        MODE_OVERRIDES=(
            env.events.random_torso_external_wrench=null
            env.events.random_pelvis_external_wrench=null
            env.events.random_arm_external_wrench=null
            env.events.random_leg_external_wrench=null
        )
        ;;
    robust)
        RANDOMIZATION_STRENGTH=1
        MODE_OVERRIDES=()
        ;;
    *)
        echo "Error: MODE must be nominal, recovery, or robust; got ${MODE}." >&2
        exit 1
        ;;
esac

echo "Visual mode      : ${MODE}"
echo "Checkpoint       : $(realpath "${CHECKPOINT}")"
echo "Action contract  : network controls all 29 joints; no ArmHack adapter"

TASK=LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-Play-v0 \
CHECKPOINT="${CHECKPOINT}" \
NUM_ENVS="${NUM_ENVS}" \
HEADLESS="${HEADLESS}" \
MAX_STEPS="${MAX_STEPS}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
bash "${LEGGED_LAB_DIR}/scripts/vis_isaacsim_g1_amp.sh" \
    "${COMMON_OVERRIDES[@]}" "${MODE_OVERRIDES[@]}" "$@"
