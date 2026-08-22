#!/usr/bin/env bash
# Strict first-principles acceptance for unified ArmHack Stand/Walk policies.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
WALK_POLICY=${WALK_POLICY:?set unified/gated Walk policy.onnx}
STAND_POLICY=${STAND_POLICY:?set unified Stand policy.onnx}
UNITREE_PYTHON=${UNITREE_PYTHON:-${HOME}/anaconda3/envs/gmr/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${LEGGED_LAB_DIR}/evaluations/first_principles_acceptance"}
POSE_PATH="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"
CONTRACT_PATH="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"
SCHEDULE_PATH="${LEGGED_LAB_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/continuous_switch_scenarios.json"

for path in "${WALK_POLICY}" "${STAND_POLICY}" "${POSE_PATH}" "${CONTRACT_PATH}" "${SCHEDULE_PATH}"; do
    [[ -f "${path}" ]] || { echo "Error: missing ${path}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}"
duration=$("${UNITREE_PYTHON}" - "${SCHEDULE_PATH}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))["scenarios"]["repeated_handoff"]["segments"]
print(sum(float(s["duration_s"]) for s in x)+0.05)
PY
)

for force in 0 40; do
    out="${OUTPUT_ROOT}/repeated_handoff_push${force}"
    mkdir -p "${out}"
    G1_AMP_SECONDARY_POLICY_PATH="${STAND_POLICY}" \
    G1_AMP_ADAPTIVE_STAND_PHASE_OBS=True \
    G1_AMP_STAND_HOLD_ACTION_FILTER_TAU=0.0 \
    G1_AMP_STAND_TO_WALK_BLEND_S=1.0 \
    G1_AMP_STAND_POST_COMPLETE_MIN_AIR_S=0.10 \
    G1_AMP_ARMHACK_WALK_ENABLE=True \
    G1_AMP_ARMHACK_WALK_POSE_PATH="${POSE_PATH}" \
    G1_AMP_ARMHACK_WALK_CONTRACT_PATH="${CONTRACT_PATH}" \
    G1_AMP_ARMHACK_WALK_POSE_NAME=pos2_down \
    G1_AMP_ARMHACK_WALK_SCHEDULE_PATH="${SCHEDULE_PATH}" \
    G1_AMP_ARMHACK_WALK_SCENARIO_NAME=repeated_handoff \
    G1_AMP_ARMHACK_WALK_START_ACTIVE=True \
    G1_AMP_ARMHACK_WALK_POSE_TRANSITION_S=2.5 \
    G1_AMP_CONTINUOUS_PUSH_FORCE_N="${force}" \
    G1_AMP_CONTINUOUS_PUSH_START_S=3.0 \
    G1_AMP_CONTINUOUS_PUSH_PERIOD_S=4.0 \
    G1_AMP_CONTINUOUS_PUSH_DURATION_S=0.20 \
    PYTHONNOUSERSITE=1 UNITREE_PYTHON="${UNITREE_PYTHON}" POLICY_PATH="${WALK_POLICY}" \
    ROBOT_ASSET=s3_g1_29dof USE_GLFW=False REAL_TIME=False \
    SIMULATION_DURATION="${duration}" CMD_INIT='[0,0,0]' RANDOM_COMMANDS=False \
    COMMAND_MODE=independent COMMAND_RAMP=True COMMAND_MAX_LINEAR_ACCEL=1.2 \
    COMMAND_MAX_YAW_ACCEL=1.5 METRICS_PATH="${out}/metrics.json" \
    TORSO_TRACE_ENABLE=True TORSO_TRACE_PATH="${out}/torso.csv" \
    TASK_TRACE_ENABLE=True TASK_TRACE_PATH="${out}/task.csv" \
        bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh" >"${out}/rollout.log" 2>&1
done

"${UNITREE_PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json,math,sys
from pathlib import Path

root=Path(sys.argv[1]); summary={}
def yaw(q):
    return math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))

for force in (0,40):
    case=root/f"repeated_handoff_push{force}"
    x=json.loads((case/"metrics.json").read_text()); h=x["health"]; p=x["adaptive_stand_phase"]
    if not h["healthy"] or h["fallen"]:
        raise RuntimeError(f"push{force}: fall/unhealthy")
    if p["completion_count"] != 2 or p["phase"] != 2:
        raise RuntimeError(f"push{force}: final Stand did not complete exactly two steps")
    if p["post_complete_air_events"] != 0:
        raise RuntimeError(f"push{force}: repeated post-completion stepping")
    if p["post_complete_foot_xy_displacement_max_m"] > 0.010:
        raise RuntimeError(f"push{force}: planted-foot displacement too large")
    if p["post_complete_lower_joint_peak_to_peak_max_rad"] > 0.080:
        raise RuntimeError(f"push{force}: planted-leg peak-to-peak oscillation too large")
    if p["post_complete_lower_joint_velocity_rms_rad_per_s"] > 0.100:
        raise RuntimeError(f"push{force}: planted-leg velocity RMS too large")
    if p["post_complete_lower_action_delta_rms"] > 0.015:
        raise RuntimeError(f"push{force}: planted-leg action-rate too large")
    ankle=x["ankle_spacing"]["final_1s_mean_m"]
    if abs(ankle-0.30) > 0.03:
        raise RuntimeError(f"push{force}: final ankle spacing {ankle:.3f}m")
    switches=x.get("policy_switch_states",[])
    if len(switches) < 5:
        raise RuntimeError(f"push{force}: expected >=5 policy switches")
    last_to_stand=[s for s in switches if s["from"]=="primary" and s["to"]=="secondary"][-1]
    start=last_to_stand["root_position"]; end=p["completion_root_position"]
    xy=math.hypot(end[0]-start[0],end[1]-start[1])
    dyaw=abs((yaw(p["completion_root_quaternion_wxyz"])-yaw(last_to_stand["root_quaternion_wxyz"])+math.pi)%(2*math.pi)-math.pi)
    if xy > 0.05 or dyaw > 0.10:
        raise RuntimeError(f"push{force}: Stand SE2 moved xy={xy:.3f} yaw={dyaw:.3f}")
    for segment in x["command_segments"]:
        name=str(segment.get("name",""))
        if not name.startswith("walk_") or "zero" in name or segment.get("steady_samples",0) <= 0:
            continue
        ankle_mean=float(segment["steady_mean_ankle_distance_m"])
        if not 0.27 <= ankle_mean <= 0.33:
            raise RuntimeError(f"push{force}/{name}: Walk ankle spacing {ankle_mean:.3f}m")
        if "yaw" in name:
            drift=math.hypot(segment["steady_mean_lin_vel_x"],segment["steady_mean_lin_vel_y"])
            if drift > 0.05 or abs(segment["steady_mean_yaw_rate"]) < 0.20:
                raise RuntimeError(f"push{force}/{name}: pure-yaw tracking/drift failed")
    summary[f"push{force}"]={
        "ankle_final_1s_m":ankle,
        "stand_se2_xy_m":xy,
        "stand_se2_yaw_rad":dyaw,
        "joint_peak_to_peak_rad":p["post_complete_lower_joint_peak_to_peak_max_rad"],
        "joint_velocity_rms_rad_s":p["post_complete_lower_joint_velocity_rms_rad_per_s"],
        "action_delta_rms":p["post_complete_lower_action_delta_rms"],
    }
(root/"acceptance_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2)); print("[ArmHack first-principles MuJoCo] PASS")
PY
