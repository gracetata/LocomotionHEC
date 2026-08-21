#!/usr/bin/env bash
# Continuous-state MuJoCo acceptance for ArmHack Stand/Walk switching.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
WALK_POLICY=${WALK_POLICY:-"${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkPrecisionSwitch/final_mujoco_pass_20260821/policy.onnx"}
STAND_POLICY=${STAND_POLICY:-"${LEGGED_LAB_DIR}/ArmHack Checkpoints/StandPerturb/final_sim2sim_spacing35_20260821/policy.onnx"}
POSE_PATH=${POSE_PATH:-"${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"}
CONTRACT_PATH=${CONTRACT_PATH:-"${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"}
SCHEDULE_PATH=${SCHEDULE_PATH:-"${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/continuous_switch_scenarios.json"}
UNITREE_PYTHON=${UNITREE_PYTHON:-${HOME}/anaconda3/envs/gmr/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${LEGGED_LAB_DIR}/evaluations/continuous_switch_20260821"}
VISUALIZE=${VISUALIZE:-False}
SCENARIO=${SCENARIO:-full_cycle}
VISUAL_PUSH_FORCE_N=${VISUAL_PUSH_FORCE_N:-0}

for required in "${WALK_POLICY}" "${STAND_POLICY}" "${POSE_PATH}" "${CONTRACT_PATH}" "${SCHEDULE_PATH}"; do
    [[ -f "${required}" ]] || { echo "Error: missing ${required}" >&2; exit 1; }
done
"${UNITREE_PYTHON}" -c 'import mujoco,numpy,onnxruntime,yaml,torch'
mkdir -p "${OUTPUT_ROOT}"

scenario_duration() {
    "${UNITREE_PYTHON}" - "${SCHEDULE_PATH}" "$1" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(sum(float(x["duration_s"]) for x in p["scenarios"][sys.argv[2]]["segments"]) + 0.05)
PY
}

run_case() {
    local scenario=$1 variant=$2 push_force=$3 visual=$4
    local duration output
    duration=$(scenario_duration "${scenario}")
    output="${OUTPUT_ROOT}/${scenario}_${variant}"
    mkdir -p "${output}"
    G1_AMP_SECONDARY_POLICY_PATH="${STAND_POLICY}" \
    G1_AMP_ADAPTIVE_STAND_PHASE_OBS=True \
    G1_AMP_ARMHACK_WALK_ENABLE=True \
    G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_PATH}" \
    G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_PATH}" \
    G1_AMP_ARMHACK_WALK_POSE_NAME=pos2_down \
    G1_AMP_ARMHACK_WALK_SCHEDULE_PATH="${SCHEDULE_PATH}" \
    G1_AMP_ARMHACK_WALK_SCENARIO_NAME="${scenario}" \
    G1_AMP_ARMHACK_WALK_START_ACTIVE=True \
    G1_AMP_ARMHACK_WALK_POSE_TRANSITION_S=2.5 \
    G1_AMP_CONTINUOUS_PUSH_FORCE_N="${push_force}" \
    G1_AMP_CONTINUOUS_PUSH_START_S=3.0 \
    G1_AMP_CONTINUOUS_PUSH_PERIOD_S=4.0 \
    G1_AMP_CONTINUOUS_PUSH_DURATION_S=0.20 \
    PYTHONNOUSERSITE=1 UNITREE_PYTHON="${UNITREE_PYTHON}" POLICY_PATH="${WALK_POLICY}" \
    ROBOT_ASSET=s3_g1_29dof USE_GLFW="${visual}" REAL_TIME="${visual}" \
    SIMULATION_DURATION="${duration}" CMD_INIT='[0,0,0]' RANDOM_COMMANDS=False \
    COMMAND_MODE=independent COMMAND_RAMP=True COMMAND_MAX_LINEAR_ACCEL=1.2 \
    COMMAND_MAX_YAW_ACCEL=1.5 METRICS_PATH="${output}/metrics.json" \
    TORSO_TRACE_ENABLE=True TORSO_TRACE_PATH="${output}/torso.csv" \
    TASK_TRACE_ENABLE=True TASK_TRACE_PATH="${output}/task.csv" \
        bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh" \
        >"${output}/rollout.log" 2>&1
}

if [[ "${VISUALIZE}" == "True" || "${VISUALIZE}" == "true" || "${VISUALIZE}" == "1" ]]; then
    run_case "${SCENARIO}" visual "${VISUAL_PUSH_FORCE_N}" True
    exit 0
fi

scenarios=(arms_down_to_front_stand raise_arms_while_walking walk_stop_then_move_arms full_cycle)
for scenario in "${scenarios[@]}"; do
    run_case "${scenario}" nominal 0 False
    run_case "${scenario}" push40 40 False
    run_case "${scenario}" push80 80 False
    run_case "${scenario}" push120 120 False
done

"${UNITREE_PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); results={}
required_failures=[]
for path in sorted(root.glob("*/metrics.json")):
    case=path.parent.name; data=json.loads(path.read_text()); health=data["health"]
    required=(case.endswith("_nominal") or case.endswith("_push40"))
    passed=bool(health["healthy"] and not health["fallen"] and health["sole_clearance_violation_fraction"] == 0.0)
    if required and not passed:
        required_failures.append(case)
    log=(path.parent/"rollout.log").read_text(errors="ignore")
    if "full_cycle" in case or "walk_stop_then_move_arms" in case:
        if "[POLICY SWITCH]" not in log:
            raise RuntimeError(f"{case}: policy switch was not executed")
    results[case]={
        "required":required,
        "passed":passed,
        "min_root_height":health["min_root_height"],
        "max_abs_roll":health["max_abs_roll"],
        "max_abs_pitch":health["max_abs_pitch"],
        "ankle_mean_m":data["ankle_spacing"]["mean_m"],
        "ankle_rmse_m":data["ankle_spacing"]["rmse_to_target_m"],
        "torso_xy_error_mps":data["important_metrics"]["torso_lin_vel_xy_cmd_error_m_per_s"],
    }
(root/"acceptance_summary.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
if required_failures:
    raise RuntimeError(f"required nominal/40N cases failed: {required_failures}")
print("[ArmHack continuous switch MuJoCo] PASS")
PY
