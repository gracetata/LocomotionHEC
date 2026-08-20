#!/usr/bin/env bash
# Run the fixed 68-rollout V4 gate against a selected V8 checkpoint.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
V8_CHECKPOINT=${V8_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_15-34-19_g1_extreme_stand_recovery_support_lock_v8_from_v4_model2999_full_20260807/model_399.pt"}
V8_LABEL=${V8_LABEL:-"v8_$(basename "${V8_CHECKPOINT}" .pt)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_${V8_LABEL}_full_gate_20260807"}

[[ -f "${V8_CHECKPOINT}" ]] || {
    echo "Error: V8 checkpoint 不存在: ${V8_CHECKPOINT}" >&2
    echo "请先根据 V8 checkpoint sweep 报告设置 V8_CHECKPOINT。" >&2
    exit 1
}

V8_EXPECTED_SHA256=${V8_EXPECTED_SHA256:-$(sha256sum "${V8_CHECKPOINT}" | awk '{print $1}')}

BASELINE_LABEL=v4 \
CANDIDATE_LABEL="${V8_LABEL}" \
V5_CHECKPOINT="${V8_CHECKPOINT}" \
V5_EXPECTED_SHA256="${V8_EXPECTED_SHA256}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
CPU_AFFINITY="${CPU_AFFINITY:-0-7,10-31}" \
SKIP_EXISTING="${SKIP_EXISTING:-True}" \
REQUIRE_PASS="${REQUIRE_PASS:-False}" \
bash "${ROOT_DIR}/scripts/test_g1_extreme_stand_smooth_settle_v5_mujoco.sh"
