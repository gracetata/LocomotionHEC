#!/usr/bin/env bash
# Strict, shape-aware MuJoCo acceptance for the gated ArmHack Walk policy.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
BASE="${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"
BASE_SIZE=14826139
BASE_SHA256="1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
CHECKPOINT=${CHECKPOINT:-${1:-}}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
UNITREE_PYTHON=${UNITREE_PYTHON:-${HOME}/anaconda3/envs/gmr/bin/python}
DURATION=${DURATION:-6.0}
ALL_POSES=${ALL_POSES:-True}
POSE_DATA="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"
CONTRACT_DATA="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"

die() { echo "Error: $*" >&2; exit 1; }
is_true() { [[ "${1,,}" == true || "${1}" == 1 || "${1,,}" == yes ]]; }
verify_base() {
    [[ -f "${BASE}" ]] && [[ "$(stat -c '%s' "${BASE}")" == "${BASE_SIZE}" ]] &&
        [[ "$(sha256sum "${BASE}" | awk '{print $1}')" == "${BASE_SHA256}" ]]
}
verify_base || die "protected model_10990 failed size/SHA-256 contract"
[[ -n "${CHECKPOINT}" ]] || die "set CHECKPOINT to a merged ArmHack two-goal checkpoint"
CHECKPOINT=$(realpath "${CHECKPOINT}")
[[ -f "${CHECKPOINT}" ]] || die "checkpoint does not exist: ${CHECKPOINT}"
[[ -x "${ISAACLAB_PYTHON}" && -x "${UNITREE_PYTHON}" ]] || die "Python environment missing"
EXPORT_ROOT=${EXPORT_ROOT:-"$(dirname "${CHECKPOINT}")/strict_mujoco_acceptance"}
mkdir -p "${EXPORT_ROOT}/candidate" "${EXPORT_ROOT}/baseline"
EXPORT_ROOT=$(realpath "${EXPORT_ROOT}")

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ISAACLAB_PYTHON}" - "${CHECKPOINT}" "${BASE}" <<'PY'
import sys, torch
candidate = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
baseline = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
state, base = candidate["model_state_dict"], baseline["model_state_dict"]
for key, shape in {"actor.0.weight": (512, 96), "actor.6.weight": (29, 128)}.items():
    if tuple(state[key].shape) != shape:
        raise RuntimeError(f"{key} shape mismatch")
for prefix in ("lateral_expert_actor", "pure_yaw_expert_actor"):
    if tuple(state[f"{prefix}.0.weight"].shape) != (512, 96):
        raise RuntimeError(f"missing/invalid {prefix}")
    if tuple(state[f"{prefix}.6.weight"].shape) != (29, 128):
        raise RuntimeError(f"invalid {prefix} output")
for key, value in base.items():
    if key.startswith("actor.") and not torch.equal(state[key], value):
        raise RuntimeError(f"protected base actor changed: {key}")
for value in candidate.values():
    if torch.is_tensor(value) and not torch.isfinite(value).all():
        raise RuntimeError("candidate contains non-finite tensor")
print("[contract] finite 96->29; base actor frozen; lateral/yaw full experts present")
PY

export_policy() {
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ISAACLAB_PYTHON}" \
        "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" --robot g1 \
        --checkpoint "$1" --output "$2/policy.onnx" --jit-output "$2/policy.pt" \
        --metadata "$2/policy.deploy.json" --default-command 0 0 0
}
export_policy "${CHECKPOINT}" "${EXPORT_ROOT}/candidate"
export_policy "${BASE}" "${EXPORT_ROOT}/baseline"

if is_true "${ALL_POSES}"; then
    POSES=(pos1_back pos2_down pos3_front)
else
    POSES=(pos2_down)
fi

run_case() {
    local policy=$1 output=$2 pose=$3 command=$4
    mkdir -p "${output}"
    G1_AMP_ARMHACK_WALK_ENABLE=True \
    G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_DATA}" \
    G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_DATA}" \
    G1_AMP_ARMHACK_WALK_POSE_NAME="${pose}" \
    G1_AMP_ARMHACK_WALK_FIXED_COMMAND="${command}" \
    G1_AMP_ARMHACK_WALK_START_ACTIVE=True \
    UNITREE_PYTHON="${UNITREE_PYTHON}" POLICY_PATH="${policy}" ROBOT_ASSET=s3_g1_29dof \
    USE_GLFW=False REAL_TIME=False SIMULATION_DURATION="${DURATION}" \
    CMD_INIT='[0.0,0.0,0.0]' RANDOM_COMMANDS=False COMMAND_MODE=independent \
    COMMAND_RAMP=True COMMAND_MAX_LINEAR_ACCEL=100 COMMAND_MAX_YAW_ACCEL=100 \
    METRICS_PATH="${output}/metrics.json" TORSO_TRACE_PATH="${output}/torso_trace.csv" \
    TASK_TRACE_PATH="${output}/task_trace.csv" \
    bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh" >"${output}/rollout.log" 2>&1
}

for pose in "${POSES[@]}"; do
    run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/${pose}_lateral_left" "${pose}" '[0.0,0.25,0.0]'
    run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/${pose}_lateral_right" "${pose}" '[0.0,-0.25,0.0]'
    run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/${pose}_yaw_left" "${pose}" '[0.0,0.0,0.35]'
    run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/${pose}_yaw_right" "${pose}" '[0.0,0.0,-0.35]'
    run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/${pose}_stand" "${pose}" '[0.0,0.0,0.0]'
    run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/${pose}_forward" "${pose}" '[0.5,0.0,0.0]'
    run_case "${EXPORT_ROOT}/baseline/policy.pt" "${EXPORT_ROOT}/${pose}_baseline_forward" "${pose}" '[0.5,0.0,0.0]'
done

"${UNITREE_PYTHON}" - "${EXPORT_ROOT}" "${POSES[@]}" <<'PY'
import json, math, sys
from pathlib import Path
root, poses = Path(sys.argv[1]), sys.argv[2:]
summary = {}
def load(name): return json.loads((root / name / "metrics.json").read_text())
def healthy(name, m):
    h=m["health"]
    if not h["healthy"] or h["fallen"]: raise RuntimeError(f"{name}: unhealthy/fallen")
def safe(name, m, clearance):
    healthy(name,m); h=m["health"]
    if h["sole_clearance_violation_fraction"] != 0.0: raise RuntimeError(f"{name}: sole violation")
    if h["min_signed_sole_clearance_m"] < clearance: raise RuntimeError(f"{name}: clearance {h['min_signed_sole_clearance_m']:.4f} < {clearance}")
for pose in poses:
    for suffix,sign in (("lateral_left",1),("lateral_right",-1)):
        name=f"{pose}_{suffix}"; m=load(name); safe(name,m,.030); t,h=m["task_tracking"],m["health"]
        if sign*t["mean_lin_vel_y"] < .18: raise RuntimeError(f"{name}: lateral response failed")
        if abs(t["mean_lin_vel_x"]) > .04 or abs(t["mean_yaw_rate"]) > .08: raise RuntimeError(f"{name}: lateral leakage")
        if h["foot_touchdown_count"] < 4: raise RuntimeError(f"{name}: insufficient touchdowns")
        summary[name]={"signed_vy":sign*t["mean_lin_vel_y"],"clearance":h["min_signed_sole_clearance_m"]}
    for suffix,sign in (("yaw_left",1),("yaw_right",-1)):
        name=f"{pose}_{suffix}"; m=load(name); safe(name,m,.025); t,h=m["task_tracking"],m["health"]
        drift=math.hypot(t["mean_lin_vel_x"],t["mean_lin_vel_y"])
        if sign*t["mean_yaw_rate"] < .25: raise RuntimeError(f"{name}: yaw response failed")
        if drift > .035: raise RuntimeError(f"{name}: planar drift {drift:.4f}")
        if h["foot_touchdown_count"] < 4: raise RuntimeError(f"{name}: insufficient touchdowns")
        summary[name]={"signed_yaw":sign*t["mean_yaw_rate"],"drift":drift,"clearance":h["min_signed_sole_clearance_m"]}
    stand=load(f"{pose}_stand"); healthy(f"{pose}_stand",stand); st=stand["task_tracking"]
    if math.hypot(st["mean_lin_vel_x"],st["mean_lin_vel_y"]) > .03 or abs(st["mean_yaw_rate"]) > .05: raise RuntimeError(f"{pose}: stand retention")
    forward,base=load(f"{pose}_forward"),load(f"{pose}_baseline_forward"); healthy("forward",forward); healthy("baseline",base)
    if forward["task_tracking"]["mean_lin_vel_x"] < .85*base["task_tracking"]["mean_lin_vel_x"]: raise RuntimeError(f"{pose}: forward retention")
(root/"acceptance_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2)); print("[ArmHack two-goal MuJoCo] PASS")
PY
verify_base || die "protected model_10990 changed during acceptance"
echo "Acceptance passed: ${EXPORT_ROOT}"
