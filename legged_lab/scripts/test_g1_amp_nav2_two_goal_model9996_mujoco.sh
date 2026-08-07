#!/usr/bin/env bash
# Strict MuJoCo acceptance for the two model_9996 specialization goals.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

MODEL9996="${PROJECT_ROOT}/checkpoint/nav2_behavior_model9996_source/model_9996.pt"
MODEL9996_SIZE=16202421
MODEL9996_SHA256="bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"
CHECKPOINT=${CHECKPOINT:-${1:-}}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
DURATION=${DURATION:-6.0}

die() {
    echo "Error: $*" >&2
    exit 1
}

verify_model9996() {
    [[ -f "${MODEL9996}" ]] || return 1
    [[ "$(stat -c '%s' "${MODEL9996}")" == "${MODEL9996_SIZE}" ]] || return 1
    [[ "$(sha256sum "${MODEL9996}" | awk '{print $1}')" == "${MODEL9996_SHA256}" ]]
}

verify_on_exit() {
    local status=$?
    trap - EXIT
    if ! verify_model9996; then
        echo "Error: protected model_9996 changed during acceptance." >&2
        status=1
    fi
    exit "${status}"
}
trap verify_on_exit EXIT

verify_model9996 || die "protected model_9996 failed its size/SHA-256 contract"
[[ -n "${CHECKPOINT}" ]] || die "set CHECKPOINT to a model_9996 two-goal checkpoint"
[[ -x "${ISAACLAB_PYTHON}" ]] || die "ISAACLAB_PYTHON is not executable"
CHECKPOINT=$(realpath "${CHECKPOINT}")
[[ -f "${CHECKPOINT}" ]] || die "checkpoint does not exist: ${CHECKPOINT}"
[[ "${CHECKPOINT}" != "${MODEL9996}" ]] || die "candidate must contain learned residual adapters"

EXPORT_ROOT=${EXPORT_ROOT:-"$(dirname "${CHECKPOINT}")/strict_model9996_acceptance"}
mkdir -p "${EXPORT_ROOT}/candidate" "${EXPORT_ROOT}/baseline"
EXPORT_ROOT=$(realpath "${EXPORT_ROOT}")

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${ISAACLAB_PYTHON}" - "${CHECKPOINT}" "${MODEL9996}" <<'PY'
import sys
import torch

candidate_path, baseline_path = sys.argv[1:]
candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
state = candidate["model_state_dict"]
baseline_state = baseline["model_state_dict"]

def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True

if not finite_tree(candidate):
    raise RuntimeError("candidate checkpoint contains a non-finite tensor")
if tuple(state["actor.0.weight"].shape) != (512, 96):
    raise RuntimeError("actor input contract is not 96")
if tuple(state["actor.6.weight"].shape) != (29, 128):
    raise RuntimeError("actor output contract is not 29")
if tuple(state["critic.0.weight"].shape) != (512, 297):
    raise RuntimeError("critic input contract is not 297")
if float(state["fixed_command_bridge_fraction"]) != 0.0:
    raise RuntimeError("deployed carrier bridge must be exactly zero")
for key, value in baseline_state.items():
    if key.startswith("actor.") and not torch.equal(state[key], value):
        raise RuntimeError(f"frozen model_9996 base actor changed: {key}")
if "pure_yaw_command_residual.0.weight" not in state:
    raise RuntimeError("missing pure-yaw specialization adapter")
if float(state["pure_yaw_command_residual.2.weight"].norm()) <= 0.0:
    raise RuntimeError("pure-yaw specialization adapter stayed identically zero")

# A lateral specialization may be either the compact residual used during early
# experiments or the final full-capacity expert. The latter is selected only
# for strict [0, vy, 0] commands by the exporter; all other commands continue
# through the bit-identical protected model_9996 actor.
if "lateral_expert_actor.0.weight" in state:
    if tuple(state["lateral_expert_actor.0.weight"].shape) != (512, 96):
        raise RuntimeError("lateral expert input contract is not 96")
    if tuple(state["lateral_expert_actor.6.weight"].shape) != (29, 128):
        raise RuntimeError("lateral expert output contract is not 29")
    if abs(float(state["lateral_expert_forward_command"]) + 0.14) > 1.0e-7:
        raise RuntimeError("unexpected lateral expert forward calibration")
    if abs(float(state["lateral_expert_same_yaw_abs"]) - 0.10) > 1.0e-7:
        raise RuntimeError("unexpected lateral expert yaw calibration")
    lateral_contract = "full lateral expert"
else:
    if "lateral_command_residual.0.weight" not in state:
        raise RuntimeError("missing lateral specialization adapter")
    if float(state["lateral_command_residual.2.weight"].norm()) <= 0.0:
        raise RuntimeError("lateral specialization adapter stayed identically zero")
    lateral_contract = "lateral residual"
disc = candidate["amp_discriminator_state_dict"]
disc_weight = next(
    value for key, value in disc.items()
    if key.endswith("0.weight") and torch.is_tensor(value) and value.ndim == 2 and value.shape[1] == 280
)
if tuple(disc_weight.shape)[1] != 280:
    raise RuntimeError("discriminator input contract is not 280")
for key in ("optimizer_state_dict", "amp_discriminator_optimizer_state_dict"):
    if key not in candidate:
        raise RuntimeError(f"missing optimizer contract: {key}")
print(
    "[two-goal-contract] finite, 96/297/29, discriminator=280, "
    f"base actor frozen, bridge=0, {lateral_contract}"
)
PY

export_policy() {
    local checkpoint=$1
    local output_dir=$2
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ISAACLAB_PYTHON}" \
        "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
        --robot g1 \
        --checkpoint "${checkpoint}" \
        --output "${output_dir}/policy.onnx" \
        --jit-output "${output_dir}/policy.pt" \
        --metadata "${output_dir}/policy.deploy.json" \
        --default-command 0.0 0.0 0.0
}

run_case() {
    local policy=$1
    local output_dir=$2
    local command=$3
    mkdir -p "${output_dir}"
    POLICY_PATH="${policy}" \
    ROBOT_ASSET=s3_g1_29dof \
    USE_GLFW=False \
    REAL_TIME=False \
    SIMULATION_DURATION="${DURATION}" \
    CMD_INIT="${command}" \
    RANDOM_COMMANDS=False \
    COMMAND_MODE=independent \
    COMMAND_RAMP=False \
    METRICS_PATH="${output_dir}/metrics.json" \
    TORSO_TRACE_PATH="${output_dir}/torso_trace.csv" \
    TASK_TRACE_PATH="${output_dir}/task_trace.csv" \
    bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh" \
        >"${output_dir}/rollout.log" 2>&1
}

export_policy "${CHECKPOINT}" "${EXPORT_ROOT}/candidate"
export_policy "${MODEL9996}" "${EXPORT_ROOT}/baseline"

run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/lateral_left" '[0.0,0.25,0.0]'
run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/lateral_right" '[0.0,-0.25,0.0]'
run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/yaw_left" '[0.0,0.0,0.35]'
run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/yaw_right" '[0.0,0.0,-0.35]'
run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/stand" '[0.0,0.0,0.0]'
run_case "${EXPORT_ROOT}/candidate/policy.pt" "${EXPORT_ROOT}/forward" '[0.5,0.0,0.0]'
run_case "${EXPORT_ROOT}/baseline/policy.pt" "${EXPORT_ROOT}/baseline_forward" '[0.5,0.0,0.0]'

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${ISAACLAB_PYTHON}" - "${EXPORT_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load(name):
    return json.loads((root / name / "metrics.json").read_text())

def require_basic_health(name, metrics):
    health = metrics["health"]
    if not health["healthy"] or health["fallen"]:
        raise RuntimeError(f"{name}: unhealthy rollout: {health}")

def require_two_goal_safety(name, metrics):
    require_basic_health(name, metrics)
    health = metrics["health"]
    if health["sole_clearance_violation_fraction"] != 0.0:
        raise RuntimeError(
            f"{name}: hard sole-clearance violation fraction is "
            f"{health['sole_clearance_violation_fraction']:.6f}"
        )
    if health["min_signed_sole_clearance_m"] < 0.025:
        raise RuntimeError(
            f"{name}: min oriented sole clearance "
            f"{health['min_signed_sole_clearance_m']:.6f} m < 0.025 m"
        )

summary = {}
for name, sign in (("lateral_left", 1.0), ("lateral_right", -1.0)):
    metrics = load(name)
    require_two_goal_safety(name, metrics)
    tracking, health = metrics["task_tracking"], metrics["health"]
    signed_vy = sign * tracking["mean_lin_vel_y"]
    if signed_vy < 0.18:
        raise RuntimeError(f"{name}: signed lateral speed {signed_vy:.4f} < 0.18 m/s")
    if abs(tracking["mean_lin_vel_x"]) > 0.04:
        raise RuntimeError(f"{name}: forward leak exceeds 0.04 m/s")
    if abs(tracking["mean_yaw_rate"]) > 0.08:
        raise RuntimeError(f"{name}: yaw leak exceeds 0.08 rad/s")
    if health["foot_touchdown_count"] < 4:
        raise RuntimeError(f"{name}: no sustained alternating stepping")
    summary[name] = {
        "signed_vy": signed_vy,
        "vx_leak": tracking["mean_lin_vel_x"],
        "yaw_leak": tracking["mean_yaw_rate"],
        "min_clearance": health["min_signed_sole_clearance_m"],
    }

for name, sign in (("yaw_left", 1.0), ("yaw_right", -1.0)):
    metrics = load(name)
    require_two_goal_safety(name, metrics)
    tracking, health = metrics["task_tracking"], metrics["health"]
    signed_yaw = sign * tracking["mean_yaw_rate"]
    planar_drift = math.hypot(tracking["mean_lin_vel_x"], tracking["mean_lin_vel_y"])
    if signed_yaw < 0.25:
        raise RuntimeError(f"{name}: signed yaw rate {signed_yaw:.4f} < 0.25 rad/s")
    if planar_drift > 0.035:
        raise RuntimeError(f"{name}: planar drift {planar_drift:.4f} > 0.035 m/s")
    if health["foot_touchdown_count"] < 4:
        raise RuntimeError(f"{name}: turn has no sustained alternating touchdowns")
    summary[name] = {
        "signed_yaw": signed_yaw,
        "planar_drift": planar_drift,
        "min_clearance": health["min_signed_sole_clearance_m"],
    }

stand = load("stand")
require_basic_health("stand", stand)
stand_tracking = stand["task_tracking"]
stand_drift = math.hypot(stand_tracking["mean_lin_vel_x"], stand_tracking["mean_lin_vel_y"])
if stand_drift > 0.03 or abs(stand_tracking["mean_yaw_rate"]) > 0.05:
    raise RuntimeError("stand retention is not stationary")

forward = load("forward")
baseline_forward = load("baseline_forward")
require_basic_health("forward", forward)
require_basic_health("baseline_forward", baseline_forward)
candidate_vx = forward["task_tracking"]["mean_lin_vel_x"]
baseline_vx = baseline_forward["task_tracking"]["mean_lin_vel_x"]
if candidate_vx < 0.85 * baseline_vx:
    raise RuntimeError(
        f"forward retention degraded by more than 15%: candidate={candidate_vx:.4f}, "
        f"baseline={baseline_vx:.4f}"
    )
summary["stand"] = {"planar_drift": stand_drift, "yaw_rate": stand_tracking["mean_yaw_rate"]}
summary["forward_retention"] = {"candidate_vx": candidate_vx, "baseline_vx": baseline_vx}
(root / "acceptance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("[two-goal-acceptance] PASS")
PY

verify_model9996 || die "protected model_9996 failed its post-test contract"
echo "Strict model_9996 two-goal MuJoCo acceptance passed: ${EXPORT_ROOT}"
