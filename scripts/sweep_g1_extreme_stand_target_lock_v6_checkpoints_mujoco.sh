#!/usr/bin/env bash
# Screen Target-Lock V6 intermediate checkpoints with the three recovery-critical
# MuJoCo profiles before committing to another full 68-rollout gate.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
RUN_DIR=${RUN_DIR:-"${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-08-07_13-24-03_g1_extreme_stand_recovery_target_lock_v6_from_v5_model1499_full_20260807"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT_DIR}/legged_lab/logs/monitoring/extreme_stand_v6_intermediate_recovery_sweep_20260807"}
CHECKPOINT_IDS=${CHECKPOINT_IDS:-0,200,400,600,700,800,900,999}
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
[[ -d "${RUN_DIR}" ]] || { echo "Error: checkpoint 目录不存在: ${RUN_DIR}" >&2; exit 1; }
taskset -c "${CPU_AFFINITY}" true >/dev/null || {
    echo "Error: CPU_AFFINITY 无效: ${CPU_AFFINITY}" >&2
    exit 1
}

mkdir -p "${RESULTS_ROOT}"
cat >"${RESULTS_ROOT}/SWEEP_IDENTITY.txt" <<EOF
RUN_DIR=${RUN_DIR}
CHECKPOINT_IDS=${CHECKPOINT_IDS}
PROFILES=${PROFILES}
SEEDS=${SEEDS}
DURATION=${DURATION}
STEADY_START_S=${STEADY_START_S}
CPU_AFFINITY=${CPU_AFFINITY}
EOF

IFS=',' read -r -a checkpoint_ids <<<"${CHECKPOINT_IDS}"
for checkpoint_id in "${checkpoint_ids[@]}"; do
    checkpoint="${RUN_DIR}/model_${checkpoint_id}.pt"
    label="v6_model${checkpoint_id}"
    model_root="${RESULTS_ROOT}/${label}"
    export_dir="${model_root}/exported"
    policy="${export_dir}/policy.onnx"
    suite_root="${model_root}/core"

    [[ -f "${checkpoint}" ]] || { echo "Error: checkpoint 不存在: ${checkpoint}" >&2; exit 1; }
    checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')

    if is_true "${SKIP_EXISTING}" && [[ -s "${suite_root}/summary.json" ]]; then
        echo "[skip] ${label}: ${suite_root}/summary.json"
        continue
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
done

"${PYTHON}" - "${RESULTS_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for model_root in sorted(root.glob("v6_model*"), key=lambda p: int(p.name.removeprefix("v6_model"))):
    metrics = {}
    for profile in ("nominal", "pose_recovery", "feet_distance_recovery"):
        path = model_root / "core" / profile / "seed_20260806" / "metrics.json"
        if path.is_file():
            metrics[profile] = json.loads(path.read_text())
    if not metrics:
        continue

    nominal = metrics.get("nominal", {})
    pose = metrics.get("pose_recovery", {})
    feet = metrics.get("feet_distance_recovery", {})

    def recovery(data):
        return data.get("extreme_stand_recovery", {}).get("default_pose_recovery", {})

    def motion(data):
        return data.get("extreme_stand_recovery", {}).get("motion_quality", {})

    def feet_recovery(data):
        return data.get("extreme_stand_recovery", {}).get("foot_spacing_recovery", {})

    nominal_pose = recovery(nominal)
    nominal_motion = motion(nominal)
    pose_result = recovery(pose)
    feet_result = feet_recovery(feet)
    row = {
        "model": model_root.name,
        "nominal_healthy": nominal.get("health", {}).get("healthy"),
        "nominal_final_joint_mae_rad": nominal_pose.get("final_joint_mae_rad"),
        "nominal_jerk_rms_rad_s3": nominal_motion.get("joint_jerk_rad_s3", {}).get("rms"),
        "pose_healthy": pose.get("health", {}).get("healthy"),
        "pose_recovered": pose_result.get("pose_recovered"),
        "pose_final_joint_mae_rad": pose_result.get("final_joint_mae_rad"),
        "feet_healthy": feet.get("health", {}).get("healthy"),
        "feet_recovered": feet_result.get("distance_recovered"),
        "feet_final_error_m": feet_result.get("final_error_mean_abs_m"),
    }
    rows.append(row)

(root / "sweep_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

headers = [
    "model", "nominal healthy", "nominal MAE rad", "nominal jerk RMS",
    "pose healthy", "pose recovered", "pose MAE rad",
    "feet healthy", "feet recovered", "feet error m",
]
lines = [
    "# V6 中间 checkpoint 恢复能力筛选",
    "",
    "| " + " | ".join(headers) + " |",
    "|" + "---|" * len(headers),
]
for row in rows:
    values = [
        row["model"], row["nominal_healthy"], row["nominal_final_joint_mae_rad"],
        row["nominal_jerk_rms_rad_s3"], row["pose_healthy"], row["pose_recovered"],
        row["pose_final_joint_mae_rad"], row["feet_healthy"], row["feet_recovered"],
        row["feet_final_error_m"],
    ]
    lines.append("| " + " | ".join(str(value) for value in values) + " |")
(root / "SWEEP_REPORT.md").write_text("\n".join(lines) + "\n")
PY

echo "Sweep report: ${RESULTS_ROOT}/SWEEP_REPORT.md"
