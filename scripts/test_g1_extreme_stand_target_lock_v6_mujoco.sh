#!/usr/bin/env bash
# Run the fixed V4 baseline gate against the completed Target-Lock V6 model.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
V6_CHECKPOINT=${V6_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_13-24-03_g1_extreme_stand_recovery_target_lock_v6_from_v5_model1499_full_20260807/model_999.pt"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_v6_target_lock_20260807"}

if [[ ! -f "${V6_CHECKPOINT}" ]]; then
    echo "Error: V6 final checkpoint 不存在: ${V6_CHECKPOINT}" >&2
    exit 1
fi

V6_EXPECTED_SHA256=${V6_EXPECTED_SHA256:-$(sha256sum "${V6_CHECKPOINT}" | awk '{print $1}')}

BASELINE_LABEL=v4 \
CANDIDATE_LABEL=v6 \
V5_CHECKPOINT="${V6_CHECKPOINT}" \
V5_EXPECTED_SHA256="${V6_EXPECTED_SHA256}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
CPU_AFFINITY="${CPU_AFFINITY:-0-7,10-31}" \
SKIP_EXISTING="${SKIP_EXISTING:-True}" \
REQUIRE_PASS="${REQUIRE_PASS:-False}" \
bash "${ROOT_DIR}/scripts/test_g1_extreme_stand_smooth_settle_v5_mujoco.sh"
