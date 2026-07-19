#!/usr/bin/env bash
# Continue the completed ArmHack Walk model on strict stop/micro/turn/foot-placement behavior.
# This launcher is Walk-only and never imports, starts or writes a Stand task.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-WalkBehaviorFinetune-v0"
EXPERIMENT_NAME="g1_walk_behavior"
OUTPUT_NAMESPACE="WalkBehaviorFinetune"
MODE=${MODE:-base}
PHASE=${PHASE:-stop_micro_turn}
ALLOW_PHASE_BASE=${ALLOW_PHASE_BASE:-False}
POSE_NAME=${POSE_NAME:-random}

CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_isaaclab}
CONDA_BASE=${CONDA_BASE:-${HOME}/anaconda3}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_ROOT}/checkpoint/walk/model_10990.pt"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-14826139}
MODE_DATA=${MODE_DATA:-"${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/armhack_walk_behavior_50hz/task_sampling_config.json"}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS_OVERRIDE=${MAX_ITERATIONS:-}
RUN_NAME=${RUN_NAME:-armhack_walk_behavior_${PHASE}_from_model10990}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}
RSI_ENABLE=${RSI_ENABLE:-False}
RSI_RATIO=${RSI_RATIO:-0.5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
DESIRED_KL=${DESIRED_KL:-0.008}
ENTROPY_COEF=${ENTROPY_COEF:-0.0015}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.003}
AMP_GRAD_PENALTY_SCALE=${AMP_GRAD_PENALTY_SCALE:-20.0}
RESUME_RUN=${RESUME_RUN:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

die() {
    echo "Error: $*" >&2
    exit 1
}

is_true() {
    [[ "$1" == "True" || "$1" == "true" || "$1" == "1" ]]
}

case "${PHASE}" in
    stop_micro_turn)
        PHASE_ITERATIONS=1500
        MODE_PROBABILITY=1.0
        STYLE_REWARD_SCALE=0.0
        TASK_STYLE_LERP=1.0
        RANDOMIZATION_STRENGTH=0
        DISABLE_PUSH=True
        ;;
    lateral_geometry)
        PHASE_ITERATIONS=1500
        MODE_PROBABILITY=0.95
        STYLE_REWARD_SCALE=3.0
        TASK_STYLE_LERP=0.75
        RANDOMIZATION_STRENGTH=1
        DISABLE_PUSH=True
        ;;
    robust)
        PHASE_ITERATIONS=2000
        MODE_PROBABILITY=0.85
        STYLE_REWARD_SCALE=5.0
        TASK_STYLE_LERP=0.65
        RANDOMIZATION_STRENGTH=1
        DISABLE_PUSH=False
        ;;
    *) die "PHASE must be stop_micro_turn, lateral_geometry, or robust" ;;
esac
MAX_ITERATIONS=${MAX_ITERATIONS_OVERRIDE:-${PHASE_ITERATIONS}}

[[ -x "${ISAACLAB_PYTHON}" ]] || die "IsaacLab Python is not executable: ${ISAACLAB_PYTHON}"
[[ -f "${BASE_CHECKPOINT}" ]] || die "completed Walk checkpoint not found: ${BASE_CHECKPOINT}"
[[ -f "${MODE_DATA}" ]] || die "behavior mode config not found: ${MODE_DATA}"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be a positive integer"
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be a positive integer"
case "${POSE_NAME}" in
    pos1_back|pos2_down|pos3_front|random) ;;
    *) die "POSE_NAME must be pos1_back, pos2_down, pos3_front, or random" ;;
esac
case "${MODE}" in
    base|resume) ;;
    *) die "MODE must be base or resume" ;;
esac
if [[ "${MODE}" == "base" && "${PHASE}" != "stop_micro_turn" ]] && ! is_true "${ALLOW_PHASE_BASE}"; then
    die "MODE=base is restricted to PHASE=stop_micro_turn; use resume for later phases"
fi

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|env.scene.robot*|env.upper_body_perturbation.pose_name=*|env.commands.base_velocity.*|env.rewards.strict_zero_*|env.rewards.nonzero_*|env.rewards.command_response_*|env.rewards.rapid_footstep_*|env.rewards.oriented_footprint_*|agent.experiment_name=*|agent.checkpoint_output_dir=*|agent.load_policy_only=*|agent.reset_iteration_on_policy_only_load=*|agent.reset_amp_on_load=*|agent.algorithm.*)
            die "Protected Walk behavior setting cannot be overridden: ${arg}"
            ;;
    esac
done

if pgrep -af 'train.py --task LeggedLab-Isaac-AMP-G1-Stand' >/dev/null; then
    die "Stand training is active; Walk behavior training refuses to start"
fi
if pgrep -af 'train.py --task LeggedLab-Isaac-AMP-G1-Walk' >/dev/null; then
    die "another ArmHack Walk training process is already active"
fi

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
[[ "${ACTUAL_BASE_SHA256}" == "${EXPECTED_BASE_SHA256}" ]] || \
    die "model_10990 SHA-256 mismatch: expected ${EXPECTED_BASE_SHA256}, got ${ACTUAL_BASE_SHA256}"
[[ "${ACTUAL_BASE_SIZE}" == "${EXPECTED_BASE_SIZE}" ]] || \
    die "model_10990 size mismatch: expected ${EXPECTED_BASE_SIZE}, got ${ACTUAL_BASE_SIZE}"

"${ISAACLAB_PYTHON}" - "${BASE_CHECKPOINT}" "${MODE_DATA}" <<'PY'
import json
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
mode_config = json.load(open(sys.argv[2], encoding="utf-8"))
state = checkpoint.get("model_state_dict", {})
expected_shapes = {
    "actor.0.weight": (512, 96),
    "actor.6.weight": (29, 128),
    "critic.0.weight": (512, 297),
    "critic.6.weight": (1, 128),
}
errors = []
if int(checkpoint.get("iter", -1)) != 10990:
    errors.append(f"iteration={checkpoint.get('iter')!r}, expected 10990")
for key, shape in expected_shapes.items():
    actual = tuple(state[key].shape) if key in state else None
    if actual != shape:
        errors.append(f"{key} shape={actual}, expected {shape}")
required = {
    "stand", "micro_forward", "micro_backward", "micro_lateral_left",
    "micro_lateral_right", "micro_diagonal_front_left",
    "micro_diagonal_front_right", "turn_in_place_left",
    "turn_in_place_right", "lateral_left", "lateral_right",
    "diagonal_front_left", "diagonal_front_right", "forward_normal",
}
if set(mode_config.get("modes", {})) != required:
    errors.append("behavior mode config does not contain the exact 14 modes")
weights = mode_config.get("mode_weights", {})
if set(weights) != required or abs(sum(float(value) for value in weights.values()) - 1.0) > 1.0e-9:
    errors.append("behavior mode weights must cover all modes and sum to 1")
if errors:
    raise SystemExit("Invalid Walk behavior base/config: " + "; ".join(errors))
print("Verified model_10990 full state: iteration=10990, actor=96->29, critic=297->1")
print("Verified behavior distribution: exact zero, micro, pure yaw, lateral and diagonal")
PY

LOG_ROOT="${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
EXPORT_ROOT="${LEGGED_LAB_DIR}/ArmHack Checkpoints/${OUTPUT_NAMESPACE}"
LOAD_POLICY_ONLY=False
RESET_ITERATION=False

if [[ "${MODE}" == "base" ]]; then
    BASE_RUN_NAME="_completed_walk_model10990"
    BASE_RUN_DIR="${LOG_ROOT}/${BASE_RUN_NAME}"
    mkdir -p "${BASE_RUN_DIR}"
    ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_10990.pt"
    LOAD_RUN="^${BASE_RUN_NAME}$"
    CHECKPOINT="^model_10990.pt$"
else
    [[ -n "${RESUME_RUN}" ]] || die "MODE=resume requires RESUME_RUN"
    [[ "${RESUME_RUN}" != */* ]] || die "RESUME_RUN must be one run-directory name"
    [[ "${RESUME_CHECKPOINT}" =~ ^model_[0-9]+\.pt$ ]] || \
        die "MODE=resume requires RESUME_CHECKPOINT=model_<iteration>.pt"
    LOG_CHECKPOINT="${LOG_ROOT}/${RESUME_RUN}/${RESUME_CHECKPOINT}"
    EXPORTED_CHECKPOINT="${EXPORT_ROOT}/${RESUME_RUN}/${RESUME_CHECKPOINT}"
    [[ -f "${LOG_CHECKPOINT}" ]] || die "Walk behavior log checkpoint not found: ${LOG_CHECKPOINT}"
    [[ -f "${EXPORTED_CHECKPOINT}" ]] || die "Walk behavior exported checkpoint not found: ${EXPORTED_CHECKPOINT}"
    [[ "$(sha256sum "${LOG_CHECKPOINT}" | awk '{print $1}')" == "$(sha256sum "${EXPORTED_CHECKPOINT}" | awk '{print $1}')" ]] || \
        die "log and exported resume checkpoints do not match"
    LOAD_RUN="^${RESUME_RUN}$"
    CHECKPOINT="^${RESUME_CHECKPOINT}$"
fi

echo "=================================================="
echo " ArmHack Walk behavior refinement"
echo "=================================================="
echo "Mode / phase      : ${MODE} / ${PHASE}"
echo "Task / pose       : ${TASK} / ${POSE_NAME}"
echo "Base checkpoint   : ${BASE_CHECKPOINT}"
echo "Resume selector   : ${LOAD_RUN} / ${CHECKPOINT}"
echo "Training          : +${MAX_ITERATIONS} iterations, ${NUM_ENVS} envs"
echo "Command modes     : probability=${MODE_PROBABILITY}; hard zero enabled"
echo "AMP               : style=${STYLE_REWARD_SCALE}, task_lerp=${TASK_STYLE_LERP}"
echo "Physical DR / push: ${RANDOMIZATION_STRENGTH} / $([[ "${DISABLE_PUSH}" == "True" ]] && echo off || echo on)"
echo "Output            : ${EXPORT_ROOT}"
echo "=================================================="

PHASE_HYDRA_ARGS=(
    "env.upper_body_perturbation.pose_name=${POSE_NAME}"
    "env.commands.base_velocity.mode_probability=${MODE_PROBABILITY}"
    "agent.load_policy_only=${LOAD_POLICY_ONLY}"
    "agent.reset_iteration_on_policy_only_load=${RESET_ITERATION}"
    "agent.reset_amp_on_load=False"
    "agent.algorithm.learning_rate=${LEARNING_RATE}"
    "agent.algorithm.desired_kl=${DESIRED_KL}"
)
if [[ "${DISABLE_PUSH}" == "True" ]]; then
    PHASE_HYDRA_ARGS+=("env.events.push_robot=null")
fi

TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
RESUME=True \
LOAD_RUN="${LOAD_RUN}" \
CHECKPOINT="${CHECKPOINT}" \
HEADLESS="${HEADLESS}" \
QUIET_TERMINAL="${QUIET_TERMINAL}" \
ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE="${RSI_ENABLE}" \
RSI_RATIO="${RSI_RATIO}" \
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
STYLE_REWARD_SCALE="${STYLE_REWARD_SCALE}" \
TASK_STYLE_LERP="${TASK_STYLE_LERP}" \
ENTROPY_COEF="${ENTROPY_COEF}" \
AMP_GRAD_PENALTY_SCALE="${AMP_GRAD_PENALTY_SCALE}" \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${BASE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    "${PHASE_HYDRA_ARGS[@]}" \
    "$@"
