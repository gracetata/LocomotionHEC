#!/usr/bin/env bash
# Fixed-baseline V4/V5 Extreme Stand comparison.
#
# Core: 6 profiles x 3 fixed seeds x 40 s for each model.
# Push: 120/180/240/360 N x four fixed horizontal directions x 40 s.
# Every run is headless, real-time disabled, zero-command, and records metrics.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
V4_CHECKPOINT=${V4_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}
V5_CHECKPOINT=${V5_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_12-25-12_g1_extreme_stand_smooth_settle_v5_resume900_to1499_20260807/model_1499.pt"}
V4_EXPECTED_SHA256=${V4_EXPECTED_SHA256:-e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff}
V5_EXPECTED_SHA256=${V5_EXPECTED_SHA256:-}
BASELINE_LABEL=${BASELINE_LABEL:-v4}
CANDIDATE_LABEL=${CANDIDATE_LABEL:-v5}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_v5_fixed_baseline_$(date +%Y%m%d_%H%M%S)"}
CORE_PROFILES=${CORE_PROFILES:-nominal,pose_recovery,feet_distance_recovery,recovery,robust,stress}
CORE_SEEDS=${CORE_SEEDS:-20260806,20260807,20260808}
PUSH_FORCES_N=${PUSH_FORCES_N:-120,180,240,360}
PUSH_DIRECTIONS=${PUSH_DIRECTIONS:-0,1,2,3}
DURATION=${DURATION:-40.0}
STEADY_START_S=${STEADY_START_S:-10.0}
SKIP_EXISTING=${SKIP_EXISTING:-True}
REQUIRE_PASS=${REQUIRE_PASS:-False}
# This workstation has previously produced non-deterministic Python import faults
# on logical CPUs 8-9. Keep every child process on the verified-safe CPU set.
# Override explicitly only on a host where all logical CPUs are known-good.
CPU_AFFINITY=${CPU_AFFINITY:-0-7,10-31}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

run_cpu_safe() {
    taskset -c "${CPU_AFFINITY}" "$@"
}

for path_var in V4_CHECKPOINT V5_CHECKPOINT RESULTS_ROOT; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${ROOT_DIR}/${value}"
    fi
done

[[ -x "${PYTHON}" ]] || { echo "Error: Python 不可执行: ${PYTHON}" >&2; exit 1; }
taskset -c "${CPU_AFFINITY}" true >/dev/null || {
    echo "Error: CPU_AFFINITY 无效: ${CPU_AFFINITY}" >&2
    exit 1
}
[[ -f "${V4_CHECKPOINT}" ]] || { echo "Error: V4 checkpoint 不存在: ${V4_CHECKPOINT}" >&2; exit 1; }
[[ -f "${V5_CHECKPOINT}" ]] || { echo "Error: V5 checkpoint 不存在: ${V5_CHECKPOINT}" >&2; exit 1; }

actual_v4_sha=$(sha256sum "${V4_CHECKPOINT}" | awk '{print $1}')
actual_v5_sha=$(sha256sum "${V5_CHECKPOINT}" | awk '{print $1}')
[[ "${actual_v4_sha}" == "${V4_EXPECTED_SHA256}" ]] || {
    echo "Error: V4 SHA256 不匹配: expected=${V4_EXPECTED_SHA256}, actual=${actual_v4_sha}" >&2
    exit 1
}
if [[ -n "${V5_EXPECTED_SHA256}" && "${actual_v5_sha}" != "${V5_EXPECTED_SHA256}" ]]; then
    echo "Error: V5 SHA256 不匹配: expected=${V5_EXPECTED_SHA256}, actual=${actual_v5_sha}" >&2
    exit 1
fi
V5_EXPECTED_SHA256=${V5_EXPECTED_SHA256:-${actual_v5_sha}}

mkdir -p "${RESULTS_ROOT}"
cat >"${RESULTS_ROOT}/MODEL_IDENTITIES.txt" <<EOF
V4_CHECKPOINT=${V4_CHECKPOINT}
V4_SHA256=${actual_v4_sha}
V5_CHECKPOINT=${V5_CHECKPOINT}
V5_SHA256=${actual_v5_sha}
BASELINE_LABEL=${BASELINE_LABEL}
CANDIDATE_LABEL=${CANDIDATE_LABEL}
CORE_PROFILES=${CORE_PROFILES}
CORE_SEEDS=${CORE_SEEDS}
PUSH_FORCES_N=${PUSH_FORCES_N}
PUSH_DIRECTIONS=${PUSH_DIRECTIONS}
DURATION=${DURATION}
STEADY_START_S=${STEADY_START_S}
CPU_AFFINITY=${CPU_AFFINITY}
EOF

echo "============================================================"
echo "  Extreme Stand ${BASELINE_LABEL}/${CANDIDATE_LABEL} fixed-baseline MuJoCo comparison"
echo "============================================================"
echo "V4       : ${V4_CHECKPOINT} (${actual_v4_sha})"
echo "V5       : ${V5_CHECKPOINT} (${actual_v5_sha})"
echo "Core     : ${CORE_PROFILES}; seeds=${CORE_SEEDS}"
echo "Push     : ${PUSH_FORCES_N} N; directions=${PUSH_DIRECTIONS}"
echo "Duration : ${DURATION}s; steady starts at ${STEADY_START_S}s"
echo "CPU set  : ${CPU_AFFINITY}"
echo "Results  : ${RESULTS_ROOT}"
echo "============================================================"

run_model() {
    local label=$1 checkpoint=$2 checkpoint_sha=$3
    local model_root="${RESULTS_ROOT}/${label}"
    local export_dir="${model_root}/exported"
    local policy="${export_dir}/policy.onnx"
    local core_root="${model_root}/core"

    if ! is_true "${SKIP_EXISTING}" || [[ ! -s "${core_root}/summary.json" ]]; then
        EXPECTED_CHECKPOINT_SHA256="${checkpoint_sha}" VERIFY_CHECKPOINT_SHA256=True \
        CHECKPOINT="${checkpoint}" EXPORT_DIR="${export_dir}" POLICY_PATH="${policy}" \
        MODEL_LABEL="${label}" FORCE_EXPORT=False \
        SUITE=True USE_GLFW=False REAL_TIME=False \
        SUITE_PROFILES="${CORE_PROFILES}" SUITE_SEEDS="${CORE_SEEDS}" \
        SUITE_DURATION="${DURATION}" STEADY_START_S="${STEADY_START_S}" \
        SUITE_RESULTS_ROOT="${core_root}" REQUIRE_PASS=False \
            run_cpu_safe bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"
    else
        echo "[skip] ${label} core: ${core_root}/summary.json"
    fi

    IFS=',' read -r -a forces <<<"${PUSH_FORCES_N}"
    IFS=',' read -r -a directions <<<"${PUSH_DIRECTIONS}"
    for force_n in "${forces[@]}"; do
        local push_root="${model_root}/push_${force_n}n"
        local direction_index seed metrics
        for direction_index in "${directions[@]}"; do
            seed=$((20260860 + direction_index))
            metrics="${push_root}/large_push/seed_${seed}/metrics.json"
            if is_true "${SKIP_EXISTING}" && [[ -s "${metrics}" ]]; then
                echo "[skip] ${label} ${force_n} N direction=${direction_index}: ${metrics}"
                continue
            fi
            EXPECTED_CHECKPOINT_SHA256="${checkpoint_sha}" VERIFY_CHECKPOINT_SHA256=True \
            CHECKPOINT="${checkpoint}" EXPORT_DIR="${export_dir}" POLICY_PATH="${policy}" \
            MODEL_LABEL="${label}" FORCE_EXPORT=False \
            PROFILE=large_push SEED="${seed}" USE_GLFW=False REAL_TIME=False \
            SIMULATION_DURATION="${DURATION}" STEADY_START_S="${STEADY_START_S}" \
            LARGE_PUSH_FORCE_N="${force_n}" LARGE_PUSH_DURATION_S=0.20 \
            LARGE_PUSH_TIME_S=5.0 LARGE_PUSH_DIRECTION_INDEX="${direction_index}" \
            RESULTS_ROOT="${push_root}" \
                run_cpu_safe bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"
        done
        run_cpu_safe "${PYTHON}" "${ROOT_DIR}/scripts/summarize_g1_extreme_stand_recovery_mujoco.py" \
            --results-root "${push_root}" \
            --output-json "${push_root}/summary.json" \
            --output-markdown "${push_root}/REPORT.md" \
            --model-label "${label}_${force_n}n_four_directions" \
            --checkpoint "${checkpoint}" --policy "${policy}"
    done
}

run_model "${BASELINE_LABEL}" "${V4_CHECKPOINT}" "${actual_v4_sha}"
run_model "${CANDIDATE_LABEL}" "${V5_CHECKPOINT}" "${actual_v5_sha}"

compare_args=(
    "${ROOT_DIR}/scripts/compare_g1_extreme_stand_mujoco.py"
    --baseline-root "${RESULTS_ROOT}/${BASELINE_LABEL}"
    --candidate-root "${RESULTS_ROOT}/${CANDIDATE_LABEL}"
    --output-json "${RESULTS_ROOT}/comparison.json"
    --output-markdown "${RESULTS_ROOT}/COMPARISON.md"
)
if is_true "${REQUIRE_PASS}"; then
    compare_args+=(--require-pass)
fi
run_cpu_safe "${PYTHON}" "${compare_args[@]}"

echo "Comparison report: ${RESULTS_ROOT}/COMPARISON.md"
