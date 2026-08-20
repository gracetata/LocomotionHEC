#!/usr/bin/env bash
# Quantitative MuJoCo validation for randomized stance recovery and pushes.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CHECKPOINT=${CHECKPOINT:-"${LEGGED_LAB_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-18_15-53-17_armhack_stand_robust_wrench_joint_dr_from_model2999_full_hec5090_20260718/model_2999.pt"}
DISTANCES=${DISTANCES:-"0.08 0.10 0.14 0.24 0.30 0.32"}
PUSH_SEEDS=${PUSH_SEEDS:-"20260814 20260815"}
MUJOCO_PUSH_ENABLE=${MUJOCO_PUSH_ENABLE:-True}
SIMULATION_DURATION=${SIMULATION_DURATION:-20.0}
MODE=${MODE:-randomized_trajectory}
TRAJECTORY_INDEX=${TRAJECTORY_INDEX:-5}
PAYLOAD_KG=${PAYLOAD_KG:-1.0}
JOINT_RANDOM_ENABLE=${JOINT_RANDOM_ENABLE:-False}
NON_ARM_JOINT_TARGET_NOISE_ENABLE=${NON_ARM_JOINT_TARGET_NOISE_ENABLE:-False}

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: foot-recovery checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

failures=0
for distance in ${DISTANCES}; do
    for push_seed in ${PUSH_SEEDS}; do
        echo "[Foot recovery MuJoCo] distance=${distance}m push_seed=${push_seed}"
        CHECKPOINT="${CHECKPOINT}" \
        MODE="${MODE}" \
        TRAJECTORY_INDEX="${TRAJECTORY_INDEX}" \
        PAYLOAD_KG="${PAYLOAD_KG}" \
        SIMULATION_DURATION="${SIMULATION_DURATION}" \
        JOINT_RANDOM_ENABLE="${JOINT_RANDOM_ENABLE}" \
        NON_ARM_JOINT_TARGET_NOISE_ENABLE="${NON_ARM_JOINT_TARGET_NOISE_ENABLE}" \
        FOOT_RECOVERY_ENABLE=True \
        INITIAL_ANKLE_DISTANCE_M="${distance}" \
        TARGET_ANKLE_DISTANCE_M=0.30 \
        ANKLE_DISTANCE_TOLERANCE_M=0.03 \
        MUJOCO_PUSH_ENABLE="${MUJOCO_PUSH_ENABLE}" \
        MUJOCO_PUSH_SEED="${push_seed}" \
        MODEL_ID="$(basename "${CHECKPOINT}" .pt)_d${distance}_push${MUJOCO_PUSH_ENABLE}_seed${push_seed}" \
        USE_GLFW=False \
        REAL_TIME=False \
        bash "${LEGGED_LAB_DIR}/scripts/val_mujoco_g1_armhack_stand.sh" || failures=$((failures + 1))
    done
done

if (( failures > 0 )); then
    echo "Error: ${failures} MuJoCo rollout(s) failed to execute." >&2
    exit 1
fi
