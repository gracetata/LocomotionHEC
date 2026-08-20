#!/usr/bin/env bash
# Multi-command, all-arm-pose MuJoCo acceptance for the 30-cm ankle objective.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
CANDIDATE_POLICY=${CANDIDATE_POLICY:?set CANDIDATE_POLICY to the fine-tuned policy.onnx}
BASELINE_POLICY=${BASELINE_POLICY:-"${PROJECT_ROOT}/checkpoint/walk/armhack_two_goal_20260811/policy.onnx"}
UNITREE_PYTHON=${UNITREE_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${LEGGED_LAB_DIR}/evaluations/armhack_walk_ankle30_20260814"}
DURATION=${DURATION:-6.0}
POSE_DATA="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"
CONTRACT_DATA="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"

CANDIDATE_POLICY=$(realpath "${CANDIDATE_POLICY}")
BASELINE_POLICY=$(realpath "${BASELINE_POLICY}")
[[ -f "${CANDIDATE_POLICY}" && -f "${BASELINE_POLICY}" ]] || { echo "policy missing" >&2; exit 1; }
PYTHONNOUSERSITE=1 "${UNITREE_PYTHON}" -c 'import mujoco,numpy,onnxruntime,yaml'
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(realpath "${OUTPUT_ROOT}")

run_case() {
    local policy=$1 variant=$2 pose=$3 scenario=$4 command=$5
    local output="${OUTPUT_ROOT}/${variant}_${pose}_${scenario}"
    mkdir -p "${output}"
    local attempt
    for attempt in 1 2 3; do
        if G1_AMP_ARMHACK_WALK_ENABLE=True \
            G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_DATA}" \
            G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_DATA}" \
            G1_AMP_ARMHACK_WALK_POSE_NAME="${pose}" \
            G1_AMP_ARMHACK_WALK_FIXED_COMMAND="${command}" \
            G1_AMP_ARMHACK_WALK_START_ACTIVE=True \
            PYTHONNOUSERSITE=1 UNITREE_PYTHON="${UNITREE_PYTHON}" POLICY_PATH="${policy}" \
            ROBOT_ASSET=s3_g1_29dof USE_GLFW=False REAL_TIME=False \
            SIMULATION_DURATION="${DURATION}" CMD_INIT='[0,0,0]' RANDOM_COMMANDS=False \
            COMMAND_MODE=independent COMMAND_RAMP=True COMMAND_MAX_LINEAR_ACCEL=100 \
            COMMAND_MAX_YAW_ACCEL=100 METRICS_PATH="${output}/metrics.json" \
            TORSO_TRACE_ENABLE=False TASK_TRACE_ENABLE=False \
                bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh" \
                >"${output}/rollout.log" 2>&1; then
            return 0
        fi
        printf 'rollout retry %d/3: %s/%s/%s\n' \
            "${attempt}" "${variant}" "${pose}" "${scenario}" >&2
    done
    echo "rollout failed after 3 attempts: ${variant}/${pose}/${scenario}" >&2
    return 1
}

poses=(pos1_back pos2_down pos3_front)
scenarios=(stand forward backward lateral_left lateral_right diagonal yaw_left yaw_right)
commands=(
    '[0.0,0.0,0.0]'
    '[0.5,0.0,0.0]'
    '[-0.15,0.0,0.0]'
    '[0.0,0.25,0.0]'
    '[0.0,-0.25,0.0]'
    '[0.25,0.15,0.0]'
    '[0.0,0.0,0.35]'
    '[0.0,0.0,-0.35]'
)
for pose in "${poses[@]}"; do
    for index in "${!scenarios[@]}"; do
        run_case "${CANDIDATE_POLICY}" candidate "${pose}" "${scenarios[index]}" "${commands[index]}"
        run_case "${BASELINE_POLICY}" baseline "${pose}" "${scenarios[index]}" "${commands[index]}"
    done
done

"${UNITREE_PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
poses = ("pos1_back", "pos2_down", "pos3_front")
scenarios = ("stand", "forward", "backward", "lateral_left", "lateral_right", "diagonal", "yaw_left", "yaw_right")
summary, candidate_rmse, baseline_rmse, candidate_within5 = {}, [], [], []

def load(variant, pose, scenario):
    return json.loads((root / f"{variant}_{pose}_{scenario}" / "metrics.json").read_text())

for pose in poses:
    for scenario in scenarios:
        candidate, baseline = load("candidate", pose, scenario), load("baseline", pose, scenario)
        if not candidate["health"]["healthy"] or candidate["health"]["fallen"]:
            raise RuntimeError(f"{pose}/{scenario}: candidate unhealthy")
        if candidate["health"]["sole_clearance_violation_fraction"] != 0.0:
            raise RuntimeError(f"{pose}/{scenario}: sole clearance violation")
        ankle, base_ankle = candidate["ankle_spacing"], baseline["ankle_spacing"]
        if not 0.20 <= ankle["mean_m"] <= 0.36:
            raise RuntimeError(f"{pose}/{scenario}: mean ankle distance {ankle['mean_m']:.3f}m")
        if ankle["rmse_to_target_m"] > base_ankle["rmse_to_target_m"]:
            raise RuntimeError(f"{pose}/{scenario}: ankle target RMSE regressed")
        candidate_rmse.append(ankle["rmse_to_target_m"])
        baseline_rmse.append(base_ankle["rmse_to_target_m"])
        candidate_within5.append(ankle["within_0p05_fraction"])
        tracking = candidate["task_tracking"]
        if scenario == "stand" and (math.hypot(tracking["mean_lin_vel_x"], tracking["mean_lin_vel_y"]) > 0.04 or abs(tracking["mean_yaw_rate"]) > 0.05):
            raise RuntimeError(f"{pose}: stand retention failed")
        if scenario == "forward" and tracking["mean_lin_vel_x"] < 0.35:
            raise RuntimeError(f"{pose}: forward response failed")
        if scenario == "backward" and tracking["mean_lin_vel_x"] > -0.08:
            raise RuntimeError(f"{pose}: backward response failed")
        if scenario == "lateral_left" and tracking["mean_lin_vel_y"] < 0.15:
            raise RuntimeError(f"{pose}: left response failed")
        if scenario == "lateral_right" and tracking["mean_lin_vel_y"] > -0.15:
            raise RuntimeError(f"{pose}: right response failed")
        if scenario == "diagonal" and (tracking["mean_lin_vel_x"] < 0.15 or tracking["mean_lin_vel_y"] < 0.08):
            raise RuntimeError(f"{pose}: diagonal response failed")
        if scenario.startswith("yaw_"):
            sign = 1.0 if scenario.endswith("left") else -1.0
            if sign * tracking["mean_yaw_rate"] < 0.20:
                raise RuntimeError(f"{pose}/{scenario}: yaw response failed")
            if math.hypot(tracking["mean_lin_vel_x"], tracking["mean_lin_vel_y"]) > 0.055:
                raise RuntimeError(f"{pose}/{scenario}: excessive planar drift")
        summary[f"{pose}/{scenario}"] = {
            "ankle_mean_m": ankle["mean_m"],
            "ankle_rmse_m": ankle["rmse_to_target_m"],
            "within_5cm": ankle["within_0p05_fraction"],
            "baseline_rmse_m": base_ankle["rmse_to_target_m"],
        }

candidate_mean_rmse = sum(candidate_rmse) / len(candidate_rmse)
baseline_mean_rmse = sum(baseline_rmse) / len(baseline_rmse)
mean_within5 = sum(candidate_within5) / len(candidate_within5)
if candidate_mean_rmse > 0.90 * baseline_mean_rmse:
    raise RuntimeError(f"aggregate ankle RMSE did not improve >=10%: {candidate_mean_rmse:.4f} vs {baseline_mean_rmse:.4f}")
if mean_within5 < 0.35:
    raise RuntimeError(f"aggregate within-5cm fraction too low: {mean_within5:.3f}")
result = {
    "candidate_mean_rmse_m": candidate_mean_rmse,
    "baseline_mean_rmse_m": baseline_mean_rmse,
    "relative_improvement": 1.0 - candidate_mean_rmse / baseline_mean_rmse,
    "candidate_mean_within_5cm_fraction": mean_within5,
    "cases": summary,
}
(root / "acceptance_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
print("[ArmHack Walk ankle30 MuJoCo] PASS")
PY

echo "Acceptance passed: ${OUTPUT_ROOT}"
