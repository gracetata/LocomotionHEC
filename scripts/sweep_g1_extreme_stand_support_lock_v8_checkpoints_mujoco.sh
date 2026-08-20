#!/usr/bin/env bash
# Screen a checkpoint family against V4, including physical PD-target drift.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"}
RUN_DIR=${RUN_DIR:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_15-34-19_g1_extreme_stand_recovery_support_lock_v8_from_v4_model2999_full_20260807"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v4_v8_support_lock_sweep_20260807"}
CHECKPOINT_IDS=${CHECKPOINT_IDS:-0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,399}
CANDIDATE_PREFIX=${CANDIDATE_PREFIX:-v8}
CANDIDATE_DISPLAY_NAME=${CANDIDATE_DISPLAY_NAME:-Support-Lock V8}
PROFILES=${PROFILES:-nominal,pose_recovery,feet_distance_recovery}
SEEDS=${SEEDS:-20260806}
DURATION=${DURATION:-40.0}
STEADY_START_S=${STEADY_START_S:-10.0}
CPU_AFFINITY=${CPU_AFFINITY:-0-7,10-31}
SKIP_EXISTING=${SKIP_EXISTING:-True}
CANDIDATE_TARGET_LIMITER_ENABLE=${CANDIDATE_TARGET_LIMITER_ENABLE:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

[[ -x "${PYTHON}" ]] || { echo "Error: Python 不可执行: ${PYTHON}" >&2; exit 1; }
[[ -f "${BASELINE_CHECKPOINT}" ]] || { echo "Error: V4 基线不存在: ${BASELINE_CHECKPOINT}" >&2; exit 1; }
[[ -d "${RUN_DIR}" ]] || { echo "Error: 候选 checkpoint 目录不存在: ${RUN_DIR}" >&2; exit 1; }
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
CANDIDATE_PREFIX=${CANDIDATE_PREFIX}
CANDIDATE_DISPLAY_NAME=${CANDIDATE_DISPLAY_NAME}
EOF

run_model() {
    local label=$1 checkpoint=$2 target_limiter_enable=${3:-False}
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
    TARGET_LIMITER_ENABLE="${target_limiter_enable}" \
    SUITE_RESULTS_ROOT="${suite_root}" REQUIRE_PASS=False \
        taskset -c "${CPU_AFFINITY}" \
        bash "${ROOT_DIR}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh"
}

run_model v4_baseline "${BASELINE_CHECKPOINT}" False

IFS=',' read -r -a checkpoint_ids <<<"${CHECKPOINT_IDS}"
for checkpoint_id in "${checkpoint_ids[@]}"; do
    checkpoint="${RUN_DIR}/model_${checkpoint_id}.pt"
    [[ -f "${checkpoint}" ]] || {
        echo "Error: 候选 checkpoint 不存在: ${checkpoint}" >&2
        exit 1
    }
    run_model \
        "${CANDIDATE_PREFIX}_model${checkpoint_id}" \
        "${checkpoint}" \
        "${CANDIDATE_TARGET_LIMITER_ENABLE}"
done

"${PYTHON}" - "${RESULTS_ROOT}" "${SEEDS%%,*}" "${CANDIDATE_PREFIX}" "${CANDIDATE_DISPLAY_NAME}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = sys.argv[2]
candidate_prefix = sys.argv[3]
candidate_display_name = sys.argv[4]

def load_profile(model_root: Path, profile: str):
    path = model_root / "core" / profile / f"seed_{seed}" / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else {}

def extreme(data):
    return data.get("extreme_stand_recovery", {})

def recovery(data):
    return extreme(data).get("default_pose_recovery", {})

def motion(data):
    return extreme(data).get("motion_quality", {})

def feet_recovery(data):
    return extreme(data).get("foot_spacing_recovery", {})

def jerk(data):
    return motion(data).get("joint_jerk_rad_s3", {}).get("rms")

def target_default(data, field):
    return (
        motion(data)
        .get("target_joint_position_rad", {})
        .get("default_error", {})
        .get(field)
    )

model_roots = [root / "v4_baseline"] + sorted(
    root.glob(f"{candidate_prefix}_model*"),
    key=lambda p: int(p.name.removeprefix(f"{candidate_prefix}_model")),
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
        "nominal_target_default_mean_abs_rad": target_default(nominal, "mean_abs"),
        "nominal_target_default_max_abs_rad": target_default(nominal, "max_abs"),
        "pose_healthy": pose.get("health", {}).get("healthy"),
        "pose_recovered": pose_result.get("pose_recovered"),
        "pose_mae_rad": pose_result.get("final_joint_mae_rad"),
        "pose_max_rad": pose_result.get("final_joint_max_abs_error_rad"),
        "pose_jerk_rms": jerk(pose),
        "pose_target_default_mean_abs_rad": target_default(pose, "mean_abs"),
        "pose_target_default_max_abs_rad": target_default(pose, "max_abs"),
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
    target_ok = all(
        row[key] is not None and baseline[key] is not None and row[key] <= limit
        for key, limit in (
            (
                "nominal_target_default_mean_abs_rad",
                baseline["nominal_target_default_mean_abs_rad"] * 1.10,
            ),
            (
                "pose_target_default_mean_abs_rad",
                baseline["pose_target_default_mean_abs_rad"] * 1.10,
            ),
            (
                "pose_target_default_max_abs_rad",
                max(0.25, baseline["pose_target_default_max_abs_rad"] * 1.25),
            ),
        )
    )
    row["screen_pass"] = bool(
        required and nominal_ok and pose_jerk_ok and nominal_jerk_ok and target_ok
    )

(root / "sweep_summary.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

headers = [
    "model", "nominal H", "nominal MAE", "nominal jerk", "nominal target mean",
    "pose H", "pose recovered", "pose MAE", "pose max", "pose jerk",
    "pose target mean", "pose target max", "feet H", "feet recovered",
    "feet error", "screen pass",
]
lines = [
    f"# V4 / {candidate_display_name} 中间 checkpoint 筛选",
    "",
    "筛选要求三场景健康、姿态与脚距严格恢复、nominal MAE不超过V4的110%、nominal/pose静态jerk不超过0.05 rad/s³或V4的150%，且physical PD target默认误差均值不超过V4的110%、pose峰值不超过0.25 rad或V4的125%。主动扰动jerk的15%改善由完整门禁判断。",
    "",
    "| " + " | ".join(headers) + " |",
    "|" + "---|" * len(headers),
]
for row in rows:
    values = [
        row["model"], row["nominal_healthy"], row["nominal_mae_rad"],
        row["nominal_jerk_rms"], row["nominal_target_default_mean_abs_rad"],
        row["pose_healthy"], row["pose_recovered"], row["pose_mae_rad"],
        row["pose_max_rad"], row["pose_jerk_rms"],
        row["pose_target_default_mean_abs_rad"], row["pose_target_default_max_abs_rad"],
        row["feet_healthy"], row["feet_recovered"], row["feet_error_m"],
        row["screen_pass"],
    ]
    lines.append("| " + " | ".join(str(value) for value in values) + " |")
(root / "SWEEP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Sweep report: ${RESULTS_ROOT}/SWEEP_REPORT.md"
