#!/usr/bin/env bash
# Deterministic headless Walk evaluation matrix with per-condition logs.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CHECKPOINT=${CHECKPOINT:-}
SUITE=${SUITE:-smoke}
MAX_STEPS_OVERRIDE=${MAX_STEPS:-}
SEED=${SEED:-42}
REPORT_ROOT=${REPORT_ROOT:-"${PROJECT_DIR}/ArmHack Checkpoints/WalkPerturbFinetune/Test Reports"}

die() {
    echo "Error: $*" >&2
    exit 1
}

[[ -n "${CHECKPOINT}" ]] || die "CHECKPOINT must point to a Walk model_*.pt"
if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${PROJECT_DIR}/${CHECKPOINT}"
fi
[[ -f "${CHECKPOINT}" ]] || die "Walk checkpoint not found: ${CHECKPOINT}"
CHECKPOINT=$(realpath "${CHECKPOINT}")

case "${SUITE}" in
    smoke)
        DEFAULT_MAX_STEPS=20
        POSES=("${POSE_NAME:-pos2_down}")
        PAYLOAD_CASES=("${LEFT_PAYLOAD_KG:-${PAYLOAD_KG:-0.0}}:${RIGHT_PAYLOAD_KG:-${PAYLOAD_KG:-0.0}}")
        COMMANDS=("${COMMAND_CASE:-nav2}")
        ;;
    core)
        DEFAULT_MAX_STEPS=1000
        POSES=(pos1_back pos2_down pos3_front)
        PAYLOAD_CASES=(0.0:0.0 1.0:1.0)
        COMMANDS=(nav2 mode:forward_slow mode:lateral_left mode:turn_left)
        ;;
    full)
        DEFAULT_MAX_STEPS=1000
        POSES=(pos1_back pos2_down pos3_front)
        PAYLOAD_CASES=(0.0:0.0 0.5:0.5 1.0:1.0 0.0:1.0 1.0:0.0)
        COMMANDS=(
            nav2
            mode:stand mode:forward_slow mode:forward_normal mode:backward
            mode:lateral_left mode:lateral_right mode:turn_left mode:turn_right
        )
        ;;
    *) die "SUITE must be smoke, core, or full" ;;
esac
MAX_STEPS=${MAX_STEPS_OVERRIDE:-${DEFAULT_MAX_STEPS}}
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be a positive integer"

CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
CHECKPOINT_STEM=$(basename "${CHECKPOINT}" .pt)
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${REPORT_ROOT}/${CHECKPOINT_STEM}_${CHECKPOINT_SHA256:0:12}/${STAMP}_${SUITE}"
mkdir -p "${REPORT_DIR}"
SUMMARY="${REPORT_DIR}/summary.md"
printf '# ArmHack Walk 测试报告\n\n' >"${SUMMARY}"
printf -- '- checkpoint: `%s`\n' "${CHECKPOINT}" >>"${SUMMARY}"
printf -- '- SHA-256: `%s`\n' "${CHECKPOINT_SHA256}" >>"${SUMMARY}"
printf -- '- suite: `%s`\n' "${SUITE}" >>"${SUMMARY}"
printf -- '- seed: `%s`\n' "${SEED}" >>"${SUMMARY}"
printf -- '- 每项步数: `%s`\n\n' "${MAX_STEPS}" >>"${SUMMARY}"
printf '| 姿态 | 左腕 kg | 右腕 kg | 命令 | 结果 | 日志 |\n' >>"${SUMMARY}"
printf '|---|---:|---:|---|---|---|\n' >>"${SUMMARY}"

failures=0
for pose in "${POSES[@]}"; do
    for payload_case in "${PAYLOAD_CASES[@]}"; do
        IFS=: read -r left_payload right_payload <<<"${payload_case}"
        for command_case in "${COMMANDS[@]}"; do
            command_source=nav2
            mode_name=forward_slow
            if [[ "${command_case}" == mode:* ]]; then
                command_source=mode
                mode_name=${command_case#mode:}
            elif [[ "${command_case}" == "hybrid" ]]; then
                command_source=hybrid
            fi
            left_payload_tag=${left_payload//./p}
            right_payload_tag=${right_payload//./p}
            condition="${pose}__payload_L${left_payload_tag}_R${right_payload_tag}kg__${command_case//:/_}"
            log_path="${REPORT_DIR}/${condition}.log"
            echo "[TEST] ${condition}"

            set +e
            CHECKPOINT="${CHECKPOINT}" \
            POSE_NAME="${pose}" \
            LEFT_PAYLOAD_KG="${left_payload}" \
            RIGHT_PAYLOAD_KG="${right_payload}" \
            COMMAND_SOURCE="${command_source}" \
            MODE_NAME="${mode_name}" \
            NUM_ENVS=1 \
            MAX_STEPS="${MAX_STEPS}" \
            SEED="${SEED}" \
            HEADLESS=True \
            REAL_TIME=False \
            FOLLOW_CAMERA=False \
            VIDEO=False \
            SKIP_EXPORT=True \
            bash "${SCRIPT_DIR}/vis_g1_armhack_walk.sh" 2>&1 | tee "${log_path}"
            run_status=${PIPESTATUS[0]}
            set -e

            result=PASS
            if [[ ${run_status} -ne 0 ]] || \
                rg -q 'Traceback \(most recent call last\)|Error executing job|Segmentation fault|Fatal Python error' "${log_path}" || \
                ! rg -q "Reached max_steps=${MAX_STEPS}" "${log_path}" || \
                ! rg -q '\[METRIC\] IsaacSim play task tracking:' "${log_path}"; then
                result=FAIL
                failures=$((failures + 1))
            fi
            printf '| `%s` | `%s` | `%s` | `%s` | **%s** | [%s](%s) |\n' \
                "${pose}" "${left_payload}" "${right_payload}" "${command_case}" "${result}" \
                "${condition}.log" "${condition}.log" >>"${SUMMARY}"
        done
    done
done

echo "Report: ${SUMMARY}"
if [[ ${failures} -ne 0 ]]; then
    die "${failures} Walk evaluation condition(s) failed"
fi
echo "All Walk evaluation conditions passed."
