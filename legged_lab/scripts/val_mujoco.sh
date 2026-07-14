#!/usr/bin/env bash
# Validate a G1 AMP checkpoint in the adapted Unitree MuJoCo headless runner.
#
# The script exports an RSL-RL checkpoint to TorchScript, runs a fixed-command
# and robustness MuJoCo suite, then writes per-case metrics plus aggregate
# humanoidness and velocity-tracking summaries.
#
# Usage:
#   bash scripts/val_mujoco.sh
#   CHECKPOINT=logs/rsl_rl/g1_amp/<run>/model_8997.pt bash scripts/val_mujoco.sh
#   POLICY_PATH=logs/rsl_rl/g1_amp/<run>/exported/policy.pt RUN_SUITE=core bash scripts/val_mujoco.sh
#
# Environment variables:
#   CHECKPOINT      : RSL-RL model_*.pt to export when POLICY_PATH is unset.
#   POLICY_PATH     : Existing TorchScript policy.pt. If set, export is skipped unless FORCE_EXPORT=True.
#   ROBOT_ASSET     : MuJoCo robot preset, default s3_g1_29dof.
#   OUTPUT_DIR      : Validation output dir. Default: <checkpoint_dir>/mujoco_val_<timestamp>.
#   RUN_SUITE       : smoke, core, full, or robust. Default full.
#   DURATION        : Fixed-command case duration in seconds. Default 12.
#   RANDOM_DURATION : Random/robust case duration in seconds. Default 18.
#   FORCE_EXPORT    : True forces checkpoint export even if POLICY_PATH exists. Default False.
#   ISAAC_PYTHON    : IsaacLab Python used for offline export.
#   UNITREE_PYTHON  : Unitree/MuJoCo Python used by sim2sim_g1_amp_mujoco.sh.
#   SIM2SIM_SCRIPT  : MuJoCo wrapper script path.
#   EXPORT_ONNX     : Kept for compatibility; the current exporter writes policy.onnx beside policy.pt.
#   QUIET           : True stores MuJoCo stdout in per-case logs. Default False.
#
# Outputs:
#   metrics/<case>.json       : raw MuJoCo report from deploy_mujoco_g1_amp.py.
#   logs/<case>.log           : launch and runner output.
#   validation_summary.json   : aggregate machine-readable scores.
#   validation_summary.md     : compact human-readable report.
#
# Common commands:
: <<'BLOCK'
# 1. Validate the current V2 checkpoint with the full suite.
CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500/model_8997.pt \
bash scripts/val_mujoco.sh

# 2. Fast smoke test after export.
RUN_SUITE=smoke DURATION=5 bash scripts/val_mujoco.sh

# 3. Core fixed-command tracking only, no robustness sweep.
RUN_SUITE=core DURATION=10 QUIET=True bash scripts/val_mujoco.sh

# 4. Robustness-only command/random dynamics sweep.
RUN_SUITE=robust RANDOM_DURATION=20 QUIET=True bash scripts/val_mujoco.sh
BLOCK

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

DEFAULT_CHECKPOINT="logs/rsl_rl/g1_amp/2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500/model_8997.pt"

CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
POLICY_PATH=${POLICY_PATH:-}
ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
RUN_SUITE=${RUN_SUITE:-full}
DURATION=${DURATION:-12}
RANDOM_DURATION=${RANDOM_DURATION:-18}
FORCE_EXPORT=${FORCE_EXPORT:-False}
EXPORT_ONNX=${EXPORT_ONNX:-True}
QUIET=${QUIET:-False}
ISAAC_PYTHON=${ISAAC_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
SIM2SIM_SCRIPT=${SIM2SIM_SCRIPT:-${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh}

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${LEGGED_LAB_DIR}/${CHECKPOINT}"
fi
if [[ -n "${POLICY_PATH}" && "${POLICY_PATH}" != /* ]]; then
    POLICY_PATH="${LEGGED_LAB_DIR}/${POLICY_PATH}"
fi

if [[ ! -f "${CHECKPOINT}" && -z "${POLICY_PATH}" ]]; then
    echo "Error: CHECKPOINT does not exist and POLICY_PATH is unset: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Error: ISAAC_PYTHON is not executable: ${ISAAC_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${SIM2SIM_SCRIPT}" ]]; then
    echo "Error: SIM2SIM_SCRIPT does not exist: ${SIM2SIM_SCRIPT}" >&2
    exit 1
fi

timestamp=$(date +%Y%m%d_%H%M%S)
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    if [[ -n "${POLICY_PATH}" ]]; then
        base_dir=$(dirname "$(dirname "${POLICY_PATH}")")
    else
        base_dir=$(dirname "${CHECKPOINT}")
    fi
    OUTPUT_DIR="${base_dir}/mujoco_val_${timestamp}"
elif [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${LEGGED_LAB_DIR}/${OUTPUT_DIR}"
fi

METRICS_DIR="${OUTPUT_DIR}/metrics"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${METRICS_DIR}" "${LOG_DIR}"

bool_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ -z "${POLICY_PATH}" ]]; then
    export_dir="$(dirname "${CHECKPOINT}")/exported"
    POLICY_PATH="${export_dir}/policy.pt"
fi

if bool_true "${FORCE_EXPORT}" || [[ ! -f "${POLICY_PATH}" ]]; then
    mkdir -p "$(dirname "${POLICY_PATH}")"
    onnx_path="$(dirname "${POLICY_PATH}")/policy.onnx"
    metadata_path="$(dirname "${POLICY_PATH}")/policy.deploy.json"
    echo "[val_mujoco] Exporting checkpoint:"
    echo "  checkpoint=${CHECKPOINT}"
    echo "  torchscript=${POLICY_PATH}"
    if bool_true "${EXPORT_ONNX}"; then
        "${ISAAC_PYTHON}" "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
            --robot g1 \
            --checkpoint "${CHECKPOINT}" \
            --output "${onnx_path}" \
            --jit-output "${POLICY_PATH}" \
            --metadata "${metadata_path}"
    else
        "${ISAAC_PYTHON}" "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
            --robot g1 \
            --checkpoint "${CHECKPOINT}" \
            --output "${onnx_path}" \
            --jit-output "${POLICY_PATH}" \
            --metadata "${metadata_path}"
    fi
fi

if [[ ! -f "${POLICY_PATH}" ]]; then
    echo "Error: POLICY_PATH does not exist after export: ${POLICY_PATH}" >&2
    exit 1
fi

case "${RUN_SUITE}" in
    smoke)
        CASES=(
            "normal_walk|fixed|0.82|0.00|0.00|${DURATION}|nominal|False|independent"
            "backward|fixed|-0.40|0.00|0.00|${DURATION}|nominal|False|independent"
        )
        ;;
    core)
        CASES=(
            "normal_walk|fixed|0.82|0.00|0.00|${DURATION}|nominal|False|independent"
            "slow_walk|fixed|0.42|0.00|0.00|${DURATION}|nominal|False|independent"
            "backward|fixed|-0.40|0.00|0.00|${DURATION}|nominal|False|independent"
            "lateral_left|fixed|0.00|0.30|0.00|${DURATION}|nominal|False|independent"
            "lateral_right|fixed|0.00|-0.30|0.00|${DURATION}|nominal|False|independent"
            "turn_left|fixed|0.35|0.00|0.50|${DURATION}|nominal|False|independent"
            "turn_right|fixed|0.35|0.00|-0.50|${DURATION}|nominal|False|independent"
            "diagonal_left|fixed|0.45|0.20|0.20|${DURATION}|nominal|False|independent"
            "diagonal_right|fixed|0.45|-0.20|-0.20|${DURATION}|nominal|False|independent"
        )
        ;;
    robust)
        CASES=(
            "random_curvature|random|0.30|0.00|0.00|${RANDOM_DURATION}|nominal|True|curvature"
            "random_omni|random|0.20|0.00|0.00|${RANDOM_DURATION}|nominal|True|independent"
            "dyn_low_passive|fixed|0.60|0.00|0.00|${DURATION}|low_passive|False|independent"
            "dyn_high_passive|fixed|0.60|0.00|0.00|${DURATION}|high_passive|False|independent"
            "dyn_high_armature|fixed|0.60|0.00|0.00|${DURATION}|high_armature|False|independent"
        )
        ;;
    full)
        CASES=(
            "normal_walk|fixed|0.82|0.00|0.00|${DURATION}|nominal|False|independent"
            "slow_walk|fixed|0.42|0.00|0.00|${DURATION}|nominal|False|independent"
            "backward|fixed|-0.40|0.00|0.00|${DURATION}|nominal|False|independent"
            "lateral_left|fixed|0.00|0.30|0.00|${DURATION}|nominal|False|independent"
            "lateral_right|fixed|0.00|-0.30|0.00|${DURATION}|nominal|False|independent"
            "turn_left|fixed|0.35|0.00|0.50|${DURATION}|nominal|False|independent"
            "turn_right|fixed|0.35|0.00|-0.50|${DURATION}|nominal|False|independent"
            "diagonal_left|fixed|0.45|0.20|0.20|${DURATION}|nominal|False|independent"
            "diagonal_right|fixed|0.45|-0.20|-0.20|${DURATION}|nominal|False|independent"
            "random_curvature|random|0.30|0.00|0.00|${RANDOM_DURATION}|nominal|True|curvature"
            "random_omni|random|0.20|0.00|0.00|${RANDOM_DURATION}|nominal|True|independent"
            "dyn_low_passive|fixed|0.60|0.00|0.00|${DURATION}|low_passive|False|independent"
            "dyn_high_passive|fixed|0.60|0.00|0.00|${DURATION}|high_passive|False|independent"
            "dyn_high_armature|fixed|0.60|0.00|0.00|${DURATION}|high_armature|False|independent"
        )
        ;;
    *)
        echo "Error: RUN_SUITE must be smoke, core, full, or robust. Got: ${RUN_SUITE}" >&2
        exit 1
        ;;
esac

echo "====================================="
echo "  G1 AMP MuJoCo Validation Suite"
echo "====================================="
echo "Checkpoint : ${CHECKPOINT}"
echo "Policy     : ${POLICY_PATH}"
echo "Robot      : ${ROBOT_ASSET}"
echo "Suite      : ${RUN_SUITE}"
echo "Output     : ${OUTPUT_DIR}"
echo "Cases      : ${#CASES[@]}"
echo "====================================="

run_case() {
    local row="$1"
    local name kind vx vy wz duration dynamics random_commands command_mode
    IFS='|' read -r name kind vx vy wz duration dynamics random_commands command_mode <<< "${row}"

    local metrics_path="${METRICS_DIR}/${name}.json"
    local log_path="${LOG_DIR}/${name}.log"
    local command_ramp=False
    if [[ "${kind}" == "random" || "${command_mode}" == "curvature" ]]; then
        command_ramp=True
    fi

    local joint_damping=0.05
    local joint_armature=0.01
    local joint_frictionloss=0.20
    local wrist_frictionloss=0.10
    case "${dynamics}" in
        nominal) ;;
        low_passive)
            joint_damping=0.03
            joint_armature=0.006
            joint_frictionloss=0.08
            wrist_frictionloss=0.04
            ;;
        high_passive)
            joint_damping=0.08
            joint_armature=0.014
            joint_frictionloss=0.35
            wrist_frictionloss=0.18
            ;;
        high_armature)
            joint_damping=0.06
            joint_armature=0.025
            joint_frictionloss=0.22
            wrist_frictionloss=0.12
            ;;
        *)
            echo "Error: unknown dynamics profile: ${dynamics}" >&2
            exit 1
            ;;
    esac

    echo "[val_mujoco] ${name}: cmd=[${vx}, ${vy}, ${wz}] duration=${duration}s dynamics=${dynamics} random=${random_commands} mode=${command_mode}"
    local cmd=(
        env
        "UNITREE_PYTHON=${UNITREE_PYTHON}"
        "POLICY_PATH=${POLICY_PATH}"
        "ROBOT_ASSET=${ROBOT_ASSET}"
        "USE_GLFW=False"
        "REAL_TIME=False"
        "SIMULATION_DURATION=${duration}"
        "CMD_INIT=[${vx}, ${vy}, ${wz}]"
        "RANDOM_COMMANDS=${random_commands}"
        "COMMAND_MODE=${command_mode}"
        "COMMAND_RAMP=${command_ramp}"
        "COMMAND_SEED=17"
        "COMMAND_INTERVAL=2.5"
        "CMD_LIN_X_RANGE=[-0.45,0.95]"
        "CMD_LIN_Y_RANGE=[-0.35,0.35]"
        "CMD_YAW_RANGE=[-0.75,0.75]"
        "CMD_CURVATURE_RANGE=[-0.85,0.85]"
        "CMD_LOW_SPEED_LIN_X_RANGE=[-0.35,0.45]"
        "CMD_LOW_SPEED_LIN_Y_RANGE=[-0.35,0.35]"
        "CMD_LOW_SPEED_YAW_RANGE=[-0.70,0.70]"
        "JOINT_DAMPING=${joint_damping}"
        "JOINT_ARMATURE=${joint_armature}"
        "JOINT_FRICTIONLOSS=${joint_frictionloss}"
        "WRIST_FRICTIONLOSS=${wrist_frictionloss}"
        "TORSO_TRACE_ENABLE=False"
        "TASK_TRACE_ENABLE=False"
        "METRICS_PATH=${metrics_path}"
        bash "${SIM2SIM_SCRIPT}"
    )

    if bool_true "${QUIET}"; then
        "${cmd[@]}" > "${log_path}" 2>&1
    else
        "${cmd[@]}" 2>&1 | tee "${log_path}"
    fi
}

for row in "${CASES[@]}"; do
    run_case "${row}"
done

"${UNITREE_PYTHON}" - "${OUTPUT_DIR}" "${POLICY_PATH}" "${CHECKPOINT}" "${RUN_SUITE}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
policy_path = sys.argv[2]
checkpoint = sys.argv[3]
run_suite = sys.argv[4]
metrics_dir = out_dir / "metrics"

def exp_score(error: float, scale: float) -> float:
    return 100.0 * math.exp(-max(float(error), 0.0) / max(float(scale), 1.0e-6))

def get(data: dict, path: str, default=0.0):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

rows = []
for path in sorted(metrics_dir.glob("*.json")):
    data = json.loads(path.read_text())
    tracking = data["task_tracking"]
    health = data["health"]
    important = data["important_metrics"]
    early = data.get("early_motion", {})
    score = data["score"]
    cmd_x = float(tracking.get("mean_command_lin_vel_x", tracking["command_lin_vel_x"]))
    cmd_y = float(tracking.get("mean_command_lin_vel_y", tracking["command_lin_vel_y"]))
    cmd_wz = float(tracking.get("mean_command_yaw_rate", tracking["command_yaw_rate"]))
    vx = float(tracking["mean_lin_vel_x"])
    vy = float(tracking["mean_lin_vel_y"])
    wz = float(tracking["mean_yaw_rate"])
    x_mae = abs(vx - cmd_x)
    y_mae = abs(vy - cmd_y)
    yaw_mae_axis = abs(wz - cmd_wz)
    early_drop = float(early.get("torso_height_drop_m", 0.0) or 0.0)
    early_pitch = float(early.get("max_abs_pitch_rad", 0.0) or 0.0)
    early_score = 0.55 * exp_score(early_drop, 0.08) + 0.45 * exp_score(early_pitch, 0.25)
    humanoid_score = (
        0.30 * float(score["health_score"])
        + 0.35 * float(score["torso_score"])
        + 0.20 * float(score["contact_score"])
        + 0.15 * early_score
    )
    rows.append(
        {
            "case": path.stem,
            "sim_time": float(data["sim_time"]),
            "healthy": bool(health["healthy"]),
            "fallen": bool(health["fallen"]),
            "fall_time": health["fall_time"],
            "cmd": [cmd_x, cmd_y, cmd_wz],
            "vel": [vx, vy, wz],
            "x_abs_error": x_mae,
            "y_abs_error": y_mae,
            "yaw_abs_error": yaw_mae_axis,
            "lin_vel_xy_mae": float(tracking["lin_vel_xy_mae"]),
            "yaw_rate_mae": float(tracking["yaw_rate_mae"]),
            "total_score": float(score["total_score"]),
            "tracking_score": float(score["tracking_score"]),
            "humanoid_score": float(humanoid_score),
            "torso_score": float(score["torso_score"]),
            "health_score": float(score["health_score"]),
            "contact_score": float(score["contact_score"]),
            "torso_roll_error_rad": float(important["torso_roll_error_rad"]),
            "torso_pitch_error_rad": float(important["torso_pitch_error_rad"]),
            "torso_height_error_m": float(important["torso_height_error_m"]),
            "torso_vertical_vel_error_m_per_s": float(important["torso_vertical_vel_error_m_per_s"]),
            "early_torso_height_drop_m": early_drop,
            "early_max_abs_pitch_rad": early_pitch,
            "foot_contact_duty": float(health["foot_contact_duty"]),
            "max_abs_pitch": float(health["max_abs_pitch"]),
            "max_abs_roll": float(health["max_abs_roll"]),
        }
    )

def mean(values):
    return float(sum(values) / len(values)) if values else float("nan")

fixed_rows = [row for row in rows if not row["case"].startswith("random_") and not row["case"].startswith("dyn_")]
robust_rows = [row for row in rows if row["case"].startswith("random_") or row["case"].startswith("dyn_")]
summary = {
    "checkpoint": checkpoint,
    "policy_path": policy_path,
    "run_suite": run_suite,
    "num_cases": len(rows),
    "num_healthy": sum(1 for row in rows if row["healthy"]),
    "fall_cases": [row["case"] for row in rows if row["fallen"]],
    "aggregate": {
        "total_score_mean": mean([row["total_score"] for row in rows]),
        "tracking_score_mean": mean([row["tracking_score"] for row in rows]),
        "humanoid_score_mean": mean([row["humanoid_score"] for row in rows]),
        "fixed_tracking_score_mean": mean([row["tracking_score"] for row in fixed_rows]),
        "robust_tracking_score_mean": mean([row["tracking_score"] for row in robust_rows]),
        "fixed_x_abs_error_mean": mean([row["x_abs_error"] for row in fixed_rows]),
        "fixed_y_abs_error_mean": mean([row["y_abs_error"] for row in fixed_rows]),
        "fixed_yaw_abs_error_mean": mean([row["yaw_abs_error"] for row in fixed_rows]),
        "torso_pitch_error_mean": mean([row["torso_pitch_error_rad"] for row in rows]),
        "early_torso_height_drop_mean": mean([row["early_torso_height_drop_m"] for row in rows]),
    },
    "rows": rows,
}

summary_json = out_dir / "validation_summary.json"
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value: float, digits: int = 3) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"

lines = [
    "# G1 AMP MuJoCo Validation Summary",
    "",
    f"- checkpoint: `{checkpoint}`",
    f"- policy: `{policy_path}`",
    f"- suite: `{run_suite}`",
    f"- healthy cases: `{summary['num_healthy']}/{summary['num_cases']}`",
    f"- fall cases: `{', '.join(summary['fall_cases']) if summary['fall_cases'] else 'none'}`",
    "",
    "## Aggregate",
    "",
    "| metric | value |",
    "| --- | --- |",
]
for key, value in summary["aggregate"].items():
    lines.append(f"| {key} | {fmt(value)} |")

lines += [
    "",
    "## Cases",
    "",
    "| case | cmd | vel | x/y/yaw err | lin/yaw mae | scores total/tracking/humanoid | torso pitch | early drop | healthy |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for row in rows:
    cmd = "(" + ",".join(fmt(v, 2) for v in row["cmd"]) + ")"
    vel = "(" + ",".join(fmt(v, 2) for v in row["vel"]) + ")"
    axis_err = f"{fmt(row['x_abs_error'])}/{fmt(row['y_abs_error'])}/{fmt(row['yaw_abs_error'])}"
    mae = f"{fmt(row['lin_vel_xy_mae'])}/{fmt(row['yaw_rate_mae'])}"
    scores = f"{fmt(row['total_score'], 1)}/{fmt(row['tracking_score'], 1)}/{fmt(row['humanoid_score'], 1)}"
    lines.append(
        f"| {row['case']} | {cmd} | {vel} | {axis_err} | {mae} | {scores} | "
        f"{fmt(row['torso_pitch_error_rad'], 4)} | {fmt(row['early_torso_height_drop_m'], 4)} | {row['healthy']} |"
    )

summary_md = out_dir / "validation_summary.md"
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"[val_mujoco] Wrote {summary_json}")
print(f"[val_mujoco] Wrote {summary_md}")
print(
    "[val_mujoco] aggregate "
    f"total={fmt(summary['aggregate']['total_score_mean'], 1)} "
    f"tracking={fmt(summary['aggregate']['tracking_score_mean'], 1)} "
    f"humanoid={fmt(summary['aggregate']['humanoid_score_mean'], 1)} "
    f"healthy={summary['num_healthy']}/{summary['num_cases']}"
)
PY

echo "====================================="
echo "Validation summary:"
echo "  ${OUTPUT_DIR}/validation_summary.md"
echo "  ${OUTPUT_DIR}/validation_summary.json"
echo "====================================="
