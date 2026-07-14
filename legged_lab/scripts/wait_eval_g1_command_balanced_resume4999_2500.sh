#!/usr/bin/env bash
# Wait for the current command-balanced finetune checkpoint, then run diagnostics.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

RUN_DIR=${RUN_DIR:-logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500}
CHECKPOINT=${CHECKPOINT:-${RUN_DIR}/model_7499.pt}
POLL_SECONDS=${POLL_SECONDS:-300}
EVAL_LOG=${EVAL_LOG:-logs/rsl_rl/g1_amp/eval_s3_g1_command_balanced_resume4999_2500_20260616.log}

mkdir -p "$(dirname "${EVAL_LOG}")"

while [[ ! -f "${CHECKPOINT}" ]]; do
    printf '[%(%F %T)T] waiting for %s\n' -1 "${CHECKPOINT}"
    sleep "${POLL_SECONDS}"
done

printf '[%(%F %T)T] found %s; starting directional eval\n' -1 "${CHECKPOINT}"
CHECKPOINT="${CHECKPOINT}" bash scripts/eval_g1_command_balanced_directional.sh >"${EVAL_LOG}" 2>&1
rc=$?
printf '[%(%F %T)T] eval exit_code=%s log=%s\n' -1 "${rc}" "${EVAL_LOG}"
exit "${rc}"
