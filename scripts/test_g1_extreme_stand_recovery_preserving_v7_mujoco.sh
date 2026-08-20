#!/usr/bin/env bash
# Run the fixed 68-rollout V4 gate against a selected V7 checkpoint.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
V7_CHECKPOINT=${V7_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_14-45-01_g1_extreme_stand_recovery_recovery_preserving_v7_from_v4_model2999/model_599.pt"}
V7_LABEL=${V7_LABEL:-"v7_$(basename "${V7_CHECKPOINT}" .pt)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_${V7_LABEL}_full_gate_20260807"}

[[ -f "${V7_CHECKPOINT}" ]] || {
    echo "Error: V7 checkpoint 不存在: ${V7_CHECKPOINT}" >&2
    echo "请先根据 checkpoint sweep 报告设置 V7_CHECKPOINT。" >&2
    exit 1
}

V7_EXPECTED_SHA256=${V7_EXPECTED_SHA256:-$(sha256sum "${V7_CHECKPOINT}" | awk '{print $1}')}

BASELINE_LABEL=v4 \
CANDIDATE_LABEL="${V7_LABEL}" \
V5_CHECKPOINT="${V7_CHECKPOINT}" \
V5_EXPECTED_SHA256="${V7_EXPECTED_SHA256}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
CPU_AFFINITY="${CPU_AFFINITY:-0-7,10-31}" \
SKIP_EXISTING="${SKIP_EXISTING:-True}" \
REQUIRE_PASS="${REQUIRE_PASS:-False}" \
bash "${ROOT_DIR}/scripts/test_g1_extreme_stand_smooth_settle_v5_mujoco.sh"
