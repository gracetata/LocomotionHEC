#!/usr/bin/env bash
# Saved ArmHack Stand MuJoCo stability visualization command.

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${LEGGED_LAB_DIR}"

POLICY_PATH="${POLICY_PATH:-${LEGGED_LAB_DIR}/deployment/armhack_stand/stand.onnx}" \
MODE=randomized_trajectory \
ITEM="${ITEM:-5}" \
PAYLOAD_KG="${PAYLOAD_KG:-1.0}" \
JOINT_RANDOM_ENABLE="${JOINT_RANDOM_ENABLE:-True}" \
JOINT_RANDOM_SEED="${JOINT_RANDOM_SEED:-20260718}" \
JOINT_POS_NOISE_RAD="${JOINT_POS_NOISE_RAD:-0.03}" \
JOINT_VEL_NOISE_RAD_PER_S="${JOINT_VEL_NOISE_RAD_PER_S:-0.10}" \
NON_ARM_JOINT_TARGET_NOISE_ENABLE="${NON_ARM_JOINT_TARGET_NOISE_ENABLE:-True}" \
NON_ARM_JOINT_TARGET_NOISE_SEED="${NON_ARM_JOINT_TARGET_NOISE_SEED:-20260719}" \
NON_ARM_JOINT_TARGET_NOISE_RAD="${NON_ARM_JOINT_TARGET_NOISE_RAD:-0.02}" \
USE_GLFW="${USE_GLFW:-True}" \
REAL_TIME="${REAL_TIME:-True}" \
bash scripts/val_mujoco_g1_armhack_stand.sh
