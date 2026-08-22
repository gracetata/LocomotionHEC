#!/usr/bin/env bash
# Unified launcher for the first-principles ArmHack Stand/Walk curriculum.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODE=${MODE:?set MODE=stand|walk_base|walk_lateral|walk_yaw}
STAGE=${STAGE:-smoke}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:?set exact source checkpoint}
SOURCE_SHA256=${SOURCE_SHA256:?set exact source SHA-256}
PRODUCER_STATE_PATH=${PRODUCER_STATE_PATH:-}
RUN_NAME=${RUN_NAME:-armhack_${MODE}_${STAGE}_first_principles_20260822}
SEED=${SEED:-20260822}

case "${STAGE}" in
    smoke) NUM_ENVS=${NUM_ENVS:-128}; MAX_ITERATIONS=${MAX_ITERATIONS:-1}; KL_SCALE=${KL_SCALE:-0.50} ;;
    nominal) NUM_ENVS=${NUM_ENVS:-4096}; MAX_ITERATIONS=${MAX_ITERATIONS:-300}; KL_SCALE=${KL_SCALE:-0.30} ;;
    randomized) NUM_ENVS=${NUM_ENVS:-4096}; MAX_ITERATIONS=${MAX_ITERATIONS:-400}; KL_SCALE=${KL_SCALE:-0.50} ;;
    handoff) NUM_ENVS=${NUM_ENVS:-4096}; MAX_ITERATIONS=${MAX_ITERATIONS:-300}; KL_SCALE=${KL_SCALE:-0.75} ;;
    *) echo "Error: STAGE must be smoke, nominal, randomized, or handoff" >&2; exit 1 ;;
esac

[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Error: missing ${SOURCE_CHECKPOINT}" >&2; exit 1; }
SOURCE_CHECKPOINT=$(realpath "${SOURCE_CHECKPOINT}")
[[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] \
    || { echo "Error: source SHA mismatch" >&2; exit 1; }
if [[ -n "${PRODUCER_STATE_PATH}" ]]; then
    [[ -f "${PRODUCER_STATE_PATH}" ]] || { echo "Error: missing producer states" >&2; exit 1; }
    PRODUCER_STATE_PATH=$(realpath "${PRODUCER_STATE_PATH}")
fi

extra=(
    agent.save_interval=20
    agent.algorithm.learning_rate=2.0e-6
    agent.algorithm.num_learning_epochs=2
    agent.algorithm.entropy_coef=0.0002
)
if [[ "${STAGE}" == "smoke" || "${STAGE}" == "nominal" ]]; then
    RANDOMIZATION_STRENGTH_STAGE=0
    extra+=(
        env.events.physics_material=null
        env.events.add_base_mass=null
        env.events.randomize_rigid_body_com=null
        env.events.scale_link_mass=null
        env.events.scale_actuator_gains=null
        env.events.scale_joint_parameters=null
        env.events.randomize_left_wrist_wrench=null
        env.events.randomize_right_wrist_wrench=null
    )
else
    RANDOMIZATION_STRENGTH_STAGE=1
fi
if [[ -n "${PRODUCER_STATE_PATH}" && "${STAGE}" == "handoff" ]]; then
    extra+=(
        "env.upper_body_perturbation.producer_state_path='${PRODUCER_STATE_PATH}'"
        env.upper_body_perturbation.producer_state_probability=0.35
    )
else
    extra+=(env.upper_body_perturbation.producer_state_probability=0.0)
fi

case "${MODE}" in
    stand)
        if [[ "${STAGE}" == "smoke" || "${STAGE}" == "nominal" ]]; then
            extra+=(env.curriculum.stance_recovery=null env.events.random_torso_external_wrench=null env.events.push_robot=null)
        fi
        TASK=LeggedLab-Isaac-AMP-G1-StandFirstPrinciples-v0 \
        SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" SOURCE_SHA256="${SOURCE_SHA256}" \
        TEACHER_CHECKPOINT="${SOURCE_CHECKPOINT}" TEACHER_KL_SCALE="${KL_SCALE}" \
        PHASE_ONE_PROBABILITY=0.20 PHASE_TWO_PROBABILITY=0.25 \
        RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH_STAGE}" \
        NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" SEED="${SEED}" \
        RUN_NAME="${RUN_NAME}" \
        bash "${LEGGED_LAB_DIR}/scripts/train_g1_armhack_stand_adaptive_switch.sh" \
        "${extra[@]}" "$@"
        ;;
    walk_base|walk_lateral|walk_yaw)
        branch=${MODE#walk_}
        case "${branch}" in
            base) task=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesBase-v0 ;;
            lateral) task=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesLateral-v0 ;;
            yaw) task=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesYaw-v0 ;;
        esac
        BRANCH="${branch}" TASK_OVERRIDE="${task}" \
        SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" SOURCE_SHA256="${SOURCE_SHA256}" \
        NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" SEED="${SEED}" \
        RUN_NAME="${RUN_NAME}" KL_SCALE="${KL_SCALE}" \
        RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH_STAGE}" \
        bash "${LEGGED_LAB_DIR}/scripts/train_g1_armhack_walk_ankle_spacing.sh" \
        "${extra[@]}" "$@"
        ;;
    *) echo "Error: MODE must be stand|walk_base|walk_lateral|walk_yaw" >&2; exit 1 ;;
esac
