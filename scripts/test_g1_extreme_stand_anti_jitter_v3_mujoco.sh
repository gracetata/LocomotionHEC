#!/usr/bin/env bash
# 最新 Anti-Jitter V3 model_2999：7 场景 × 3 seeds 的 40 秒 MuJoCo 长期测试。
# 前 10 秒作为恢复段；报告重点统计稳态 jerk、20–25 Hz 抖动和双脚距离。

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT=${CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_resume1400_to2999_full_20260724/model_2999.pt"}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-e2c694d2d7710315f41f1c6c75849ffb95b53d0fb29e612aa211e1525a7cb1e4}
EXPORT_DIR=${EXPORT_DIR:-"$(dirname "${CHECKPOINT}")/exported_extreme_stand_recovery"}
POLICY_PATH=${POLICY_PATH:-"${EXPORT_DIR}/policy.onnx"}
RESULTS_ROOT=${RESULTS_ROOT:-"${EXPORT_DIR}/mujoco_tests/anti_jitter_v3_model2999_long_20260726"}

PROFILES=${PROFILES:-nominal,pose_recovery,feet_distance_recovery,recovery,robust,stress,large_push}
SEEDS=${SEEDS:-20260726,20260727,20260728}
DURATION=${DURATION:-40.0}
STEADY_START_S=${STEADY_START_S:-10.0}
REQUIRE_PASS=${REQUIRE_PASS:-False}
FORCE_EXPORT=${FORCE_EXPORT:-False}

echo "============================================================"
echo "  Extreme Stand Anti-Jitter V3 long MuJoCo test"
echo "============================================================"
echo "Checkpoint : ${CHECKPOINT}"
echo "Policy     : ${POLICY_PATH}"
echo "Profiles   : ${PROFILES}"
echo "Seeds      : ${SEEDS}"
echo "Duration   : ${DURATION}s; steady starts at ${STEADY_START_S}s"
echo "Report     : ${RESULTS_ROOT}/REPORT.md"
echo "============================================================"

EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256}" \
VERIFY_CHECKPOINT_SHA256=True \
CHECKPOINT="${CHECKPOINT}" \
EXPORT_DIR="${EXPORT_DIR}" \
POLICY_PATH="${POLICY_PATH}" \
MODEL_LABEL=anti_jitter_v3_model2999 \
SUITE=True \
USE_GLFW=False \
REAL_TIME=False \
SUITE_PROFILES="${PROFILES}" \
SUITE_SEEDS="${SEEDS}" \
SUITE_DURATION="${DURATION}" \
STEADY_START_S="${STEADY_START_S}" \
FEET_GAUSSIAN_VARIANCE_M2=1.0e-4 \
JOINT_JERK_REWARD_WEIGHT=-1.0e-8 \
SUITE_RESULTS_ROOT="${RESULTS_ROOT}" \
REQUIRE_PASS="${REQUIRE_PASS}" \
FORCE_EXPORT="${FORCE_EXPORT}" \
    bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"

echo "Long-term report: ${RESULTS_ROOT}/REPORT.md"
