#!/usr/bin/env bash
# 多随机种子测试：只随机化29个关节初始姿态，检查是否恢复到默认全身姿态。

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${SEEDS:-20260722,20260723,20260724,20260725,20260726}
DURATION=${DURATION:-15.0}
USE_GLFW=${USE_GLFW:-False}
REQUIRE_PASS=${REQUIRE_PASS:-False}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/random_pose_recovery_$(date +%Y%m%d_%H%M%S)"}

if [[ "${USE_GLFW,,}" == "true" || "${USE_GLFW}" == "1" ]]; then
    first_seed=${SEEDS%%,*}
    PROFILE=pose_recovery SEED="${first_seed}" USE_GLFW=True REAL_TIME=True \
    SIMULATION_DURATION="${DURATION}" RESULTS_ROOT="${RESULTS_ROOT}" \
        bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"
else
    SUITE=True SUITE_PROFILES=pose_recovery SUITE_SEEDS="${SEEDS}" \
    SUITE_DURATION="${DURATION}" USE_GLFW=False REAL_TIME=False \
    SUITE_RESULTS_ROOT="${RESULTS_ROOT}" REQUIRE_PASS="${REQUIRE_PASS}" \
        bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"
fi
