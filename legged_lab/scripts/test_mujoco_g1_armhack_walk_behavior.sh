#!/usr/bin/env bash
# Scheduled MuJoCo acceptance matrix for ArmHack Walk behavior refinement.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

CHECKPOINT=${CHECKPOINT:-${PROJECT_ROOT}/checkpoint/walk/model_10990.pt}
SUITE=${SUITE:-smoke}
ENFORCE_THRESHOLDS=${ENFORCE_THRESHOLDS:-False}
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/user/anaconda3/envs/gmr/bin/python}
REPORT_ROOT=${REPORT_ROOT:-"${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkBehaviorFinetune/MuJoCo Test Reports"}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}
die() {
    echo "Error: $*" >&2
    exit 1
}

[[ -f "${CHECKPOINT}" ]] || die "Walk checkpoint not found: ${CHECKPOINT}"
[[ -x "${UNITREE_PYTHON}" ]] || die "UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}"
CHECKPOINT=$(realpath "${CHECKPOINT}")
CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
CHECKPOINT_STEM=$(basename "${CHECKPOINT}" .pt)

case "${SUITE}" in
    smoke)
        POSES=("${POSE_NAME:-pos2_down}")
        SCENARIOS=(smoke_walk_to_zero)
        SETTLE_TIME=0.10
        ;;
    core)
        POSES=("${POSE_NAME:-pos2_down}")
        SCENARIOS=(
            zero_hold walk_to_zero
            micro_forward micro_lateral micro_diagonal
            turn_in_place_left turn_in_place_right
            lateral_left lateral_right
            diagonal_front_left diagonal_front_right
            forward_cadence
        )
        SETTLE_TIME=0.75
        ;;
    full)
        POSES=(pos1_back pos2_down pos3_front)
        SCENARIOS=(
            zero_hold walk_to_zero
            micro_forward micro_lateral micro_diagonal
            turn_in_place_left turn_in_place_right
            lateral_left lateral_right
            diagonal_front_left diagonal_front_right
            forward_cadence
        )
        SETTLE_TIME=0.75
        ;;
    *) die "SUITE must be smoke, core, or full" ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${REPORT_ROOT}/${CHECKPOINT_STEM}_${CHECKPOINT_SHA256:0:12}/${STAMP}_${SUITE}"
mkdir -p "${REPORT_DIR}"
SUMMARY="${REPORT_DIR}/summary.md"
{
    printf '# ArmHack Walk MuJoCo 行为测试\n\n'
    printf -- '- checkpoint: `%s`\n' "${CHECKPOINT}"
    printf -- '- SHA-256: `%s`\n' "${CHECKPOINT_SHA256}"
    printf -- '- suite: `%s`\n' "${SUITE}"
    printf -- '- 强制行为阈值: `%s`\n\n' "${ENFORCE_THRESHOLDS}"
    printf '| 双臂姿态 | 场景 | 运行 | 行为阈值 | 指标 | 日志 |\n'
    printf '|---|---|---|---|---|---|\n'
} >"${SUMMARY}"

runtime_failures=0
behavior_failures=0
for pose in "${POSES[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        condition="${pose}__${scenario}"
        metrics="${REPORT_DIR}/${condition}.metrics.json"
        assessment="${REPORT_DIR}/${condition}.assessment.json"
        log="${REPORT_DIR}/${condition}.log"
        echo "[MUJOCO TEST] ${condition}"

        set +e
        CHECKPOINT="${CHECKPOINT}" \
        EXPECTED_CHECKPOINT_SHA256="${CHECKPOINT_SHA256}" \
        POSE_NAME="${pose}" \
        SCENARIO_NAME="${scenario}" \
        BEHAVIOR_SETTLE_TIME_S="${SETTLE_TIME}" \
        USE_GLFW=False \
        REAL_TIME=False \
        METRICS_PATH="${metrics}" \
        bash "${SCRIPT_DIR}/val_mujoco_g1_armhack_walk_behavior.sh" 2>&1 | tee "${log}"
        run_status=${PIPESTATUS[0]}
        set -e

        runtime_result=PASS
        behavior_result=NOT_RUN
        if [[ ${run_status} -ne 0 ]] || [[ ! -f "${metrics}" ]] || \
            rg -q 'Traceback \(most recent call last\)|Segmentation fault|Fatal Python error' "${log}"; then
            runtime_result=FAIL
            runtime_failures=$((runtime_failures + 1))
        else
            analyzer_args=("${metrics}" --scenario "${scenario}" --output "${assessment}")
            if is_true "${ENFORCE_THRESHOLDS}"; then
                analyzer_args+=(--enforce)
            fi
            set +e
            "${UNITREE_PYTHON}" "${SCRIPT_DIR}/analyze_armhack_walk_behavior_mujoco.py" \
                "${analyzer_args[@]}" | tee -a "${log}"
            analyze_status=${PIPESTATUS[0]}
            set -e
            behavior_result=$("${UNITREE_PYTHON}" - "${assessment}" <<'PY'
import json, sys
print("PASS" if json.load(open(sys.argv[1], encoding="utf-8"))["passed"] else "FAIL")
PY
)
            if [[ "${behavior_result}" == "FAIL" ]]; then
                behavior_failures=$((behavior_failures + 1))
            fi
            if is_true "${ENFORCE_THRESHOLDS}" && [[ ${analyze_status} -ne 0 ]]; then
                runtime_result=PASS
            fi
        fi
        printf '| `%s` | `%s` | **%s** | **%s** | [%s](%s) | [%s](%s) |\n' \
            "${pose}" "${scenario}" "${runtime_result}" "${behavior_result}" \
            "$(basename "${metrics}")" "$(basename "${metrics}")" \
            "$(basename "${log}")" "$(basename "${log}")" >>"${SUMMARY}"
    done
done

echo "Report: ${SUMMARY}"
[[ ${runtime_failures} -eq 0 ]] || die "${runtime_failures} MuJoCo scenario(s) failed to run"
if is_true "${ENFORCE_THRESHOLDS}" && [[ ${behavior_failures} -ne 0 ]]; then
    die "${behavior_failures} scenario(s) failed behavior acceptance"
fi
echo "MuJoCo runtime suite passed; behavior failures=${behavior_failures}, enforce=${ENFORCE_THRESHOLDS}."
