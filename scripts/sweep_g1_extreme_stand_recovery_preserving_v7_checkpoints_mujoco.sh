#!/usr/bin/env bash
# Screen every V7 checkpoint against the exact V4 baseline before the full gate.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}
RUN_DIR=${RUN_DIR:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_14-45-01_g1_extreme_stand_recovery_recovery_preserving_v7_from_v4_model2999"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_v7_recovery_sweep_20260807"}
CHECKPOINT_IDS=${CHECKPOINT_IDS:-0,50,100,150,200,250,300,350,400,450,500,550,599}
PROFILES=${PROFILES:-nominal,pose_recovery,feet_distance_recovery}
SEEDS=${SEEDS:-20260806}
DURATION=${DURATION:-40.0}
STEADY_START_S=${STEADY_START_S:-10.0}
CPU_AFFINITY=${CPU_AFFINITY:-0-7,10-31}
SKIP_EXISTING=${SKIP_EXISTING:-True}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

[[ -x "${PYTHON}" ]] || { echo "Error: Python 不可执行: ${PYTHON}" >&2; exit 1; }
[[ -f "${BASELINE_CHECKPOINT}" ]] || { echo "Error: V4 基线不存在: ${BASELINE_CHECKPOINT}" >&2; exit 1; }
[[ -d "${RUN_DIR}" ]] || { echo "Error: V7 checkpoint 目录不存在: ${RUN_DIR}" >&2; exit 1; }
taskset -c "${CPU_AFFINITY}" true >/dev/null || {
    echo "Error: CPU_AFFINITY 无效: ${CPU_AFFINITY}" >&2
    exit 1
}

mkdir -p "${RESULTS_ROOT}"
cat >"${RESULTS_ROOT}/SWEEP_IDENTITY.txt" <<EOF
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT}
BASELINE_SHA256=$(sha256sum "${BASELINE_CHECKPOINT}" | awk '{print $1}')
RUN_DIR=${RUN_DIR}
CHECKPOINT_IDS=${CHECKPOINT_IDS}
PROFILES=${PROFILES}
SEEDS=${SEEDS}
DURATION=${DURATION}
STEADY_START_S=${STEADY_START_S}
CPU_AFFINITY=${CPU_AFFINITY}
EOF

run_model() {
    local label=$1 checkpoint=$2
    local model_root="${RESULTS_ROOT}/${label}"
    local export_dir="${model_root}/exported"
    local policy="${export_dir}/policy.onnx"
    local suite_root="${model_root}/core"
    local checkpoint_sha
    checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')

    if is_true "${SKIP_EXISTING}" && [[ -s "${suite_root}/summary.json" ]]; then
        echo "[skip] ${label}: ${suite_root}/summary.json"
        return
    fi

    echo "============================================================"
    echo "Screen ${label}: ${checkpoint_sha}"
    echo "============================================================"
    EXPECTED_CHECKPOINT_SHA256="${checkpoint_sha}" VERIFY_CHECKPOINT_SHA256=True \
    CHECKPOINT="${checkpoint}" EXPORT_DIR="${export_dir}" POLICY_PATH="${policy}" \
    MODEL_LABEL="${label}" FORCE_EXPORT=False \
    SUITE=True USE_GLFW=False REAL_TIME=False \
    SUITE_PROFILES="${PROFILES}" SUITE_SEEDS="${SEEDS}" \
    SUITE_DURATION="${DURATION}" STEADY_START_S="${STEADY_START_S}" \
    SUITE_RESULTS_ROOT="${suite_root}" REQUIRE_PASS=False \
        taskset -c "${CPU_AFFINITY}" \
        bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"
}

run_model v4_baseline "${BASELINE_CHECKPOINT}"

IFS=',' read -r -a checkpoint_ids <<<"${CHECKPOINT_IDS}"
for checkpoint_id in "${checkpoint_ids[@]}"; do
    checkpoint="${RUN_DIR}/model_${checkpoint_id}.pt"
    [[ -f "${checkpoint}" ]] || {
        echo "Error: V7 checkpoint 不存在: ${checkpoint}" >&2
        exit 1
    }
    run_model "v7_model${checkpoint_id}" "${checkpoint}"
done

"${PYTHON}" - "${RESULTS_ROOT}" "${SEEDS%%,*}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = sys.argv[2]

def load_profile(model_root: Path, profile: str):
    path = model_root / "core" / profile / f"seed_{seed}" / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else {}

def recovery(data):
    return data.get("extreme_stand_recovery", {}).get("default_pose_recovery", {})

def motion(data):
    return data.get("extreme_stand_recovery", {}).get("motion_quality", {})

def feet_recovery(data):
    return data.get("extreme_stand_recovery", {}).get("foot_spacing_recovery", {})

def jerk(data):
    return motion(data).get("joint_jerk_rad_s3", {}).get("rms")

model_roots = [root / "v4_baseline"] + sorted(
    root.glob("v7_model*"), key=lambda p: int(p.name.removeprefix("v7_model"))
)
rows = []
for model_root in model_roots:
    nominal = load_profile(model_root, "nominal")
    pose = load_profile(model_root, "pose_recovery")
    feet = load_profile(model_root, "feet_distance_recovery")
    if not nominal or not pose or not feet:
        continue
    nominal_pose = recovery(nominal)
    pose_result = recovery(pose)
    feet_result = feet_recovery(feet)
    rows.append({
        "model": model_root.name,
        "nominal_healthy": nominal.get("health", {}).get("healthy"),
        "nominal_mae_rad": nominal_pose.get("final_joint_mae_rad"),
        "nominal_max_rad": nominal_pose.get("final_joint_max_abs_error_rad"),
        "nominal_jerk_rms": jerk(nominal),
        "pose_healthy": pose.get("health", {}).get("healthy"),
        "pose_recovered": pose_result.get("pose_recovered"),
        "pose_mae_rad": pose_result.get("final_joint_mae_rad"),
        "pose_max_rad": pose_result.get("final_joint_max_abs_error_rad"),
        "pose_jerk_rms": jerk(pose),
        "feet_healthy": feet.get("health", {}).get("healthy"),
        "feet_recovered": feet_result.get("distance_recovered"),
        "feet_error_m": feet_result.get("final_error_mean_abs_m"),
        "feet_jerk_rms": jerk(feet),
    })

baseline = next((row for row in rows if row["model"] == "v4_baseline"), None)
for row in rows:
    if row["model"] == "v4_baseline" or baseline is None:
        row["screen_pass"] = row["model"] == "v4_baseline"
        continue
    required = (
        row["nominal_healthy"] is True
        and row["pose_healthy"] is True
        and row["feet_healthy"] is True
        and row["pose_recovered"] is True
        and row["feet_recovered"] is True
    )
    nominal_ok = (
        row["nominal_mae_rad"] is not None
        and baseline["nominal_mae_rad"] is not None
        and row["nominal_mae_rad"] <= baseline["nominal_mae_rad"] * 1.10
    )
    pose_jerk_ok = (
        row["pose_jerk_rms"] is not None
        and baseline["pose_jerk_rms"] is not None
        and row["pose_jerk_rms"] <= max(0.05, baseline["pose_jerk_rms"] * 1.50)
    )
    nominal_jerk_ok = (
        row["nominal_jerk_rms"] is not None
        and baseline["nominal_jerk_rms"] is not None
        and row["nominal_jerk_rms"] <= max(0.05, baseline["nominal_jerk_rms"] * 1.50)
    )
    row["screen_pass"] = bool(required and nominal_ok and pose_jerk_ok and nominal_jerk_ok)

(root / "sweep_summary.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

headers = [
    "model", "nominal H", "nominal MAE", "nominal jerk",
    "pose H", "pose recovered", "pose MAE", "pose jerk",
    "feet H", "feet recovered", "feet error", "feet jerk", "screen pass",
]
lines = [
    "# V4 / Recovery-Preserving V7 中间 checkpoint 筛选",
    "",
    "筛选只是完整门禁的前置条件：三场景均健康、姿态与脚距严格恢复、nominal MAE 不超过 V4 的110%，且 nominal/pose-recovery 静止段 jerk 不超过0.05 rad/s³或V4的150%（取较宽者）。主动扰动场景的15% jerk改善由后续完整门禁判断。",
    "",
    "| " + " | ".join(headers) + " |",
    "|" + "---|" * len(headers),
]
for row in rows:
    values = [
        row["model"], row["nominal_healthy"], row["nominal_mae_rad"],
        row["nominal_jerk_rms"], row["pose_healthy"], row["pose_recovered"],
        row["pose_mae_rad"], row["pose_jerk_rms"], row["feet_healthy"],
        row["feet_recovered"], row["feet_error_m"], row["feet_jerk_rms"],
        row["screen_pass"],
    ]
    lines.append("| " + " | ".join(str(value) for value in values) + " |")
(root / "SWEEP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Sweep report: ${RESULTS_ROOT}/SWEEP_REPORT.md"
