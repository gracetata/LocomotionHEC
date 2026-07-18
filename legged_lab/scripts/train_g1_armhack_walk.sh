#!/usr/bin/env bash
# Train the isolated P0/P1 ArmHack Walk task on S3 G1.
#
# MODE=init   : policy-only initialization from verified locomotion model_9996.
# MODE=resume : full optimizer/discriminator/iteration restore from a Walk run.
#
# Legacy task retained only for reproducibility: TASK="LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0"
# Legacy checkpoint namespace: EXPERIMENT_NAME="g1_walk_perturb"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-v0"
ROBOT_ASSET_NAME="s3_g1_29dof"
EXPERIMENT_NAME="g1_walk_robust"
LEGACY_EXPERIMENT_NAME="g1_walk_perturb"
MODE=${MODE:-init}
PHASE=${PHASE:-amp_warmup}
ALLOW_PHASE_INIT=${ALLOW_PHASE_INIT:-False}
POSE_NAME=${POSE_NAME:-pos2_down}

CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_isaaclab}
CONDA_BASE=${CONDA_BASE:-${HOME}/anaconda3}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python}

BASE_ONNX=${BASE_ONNX:-"${PROJECT_DIR}/../checkpoint/model_9996/locomotion.onnx"}
BASE_METADATA=${BASE_METADATA:-"${PROJECT_DIR}/../checkpoint/model_9996/locomotion.deploy.json"}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${PROJECT_DIR}/ArmHack Checkpoints/WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt"}
EXPECTED_ONNX_SHA256=${EXPECTED_ONNX_SHA256:-"05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6"}
EXPECTED_BASE_SHA256=${EXPECTED_BASE_SHA256:-"bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"}
EXPECTED_BASE_SIZE=${EXPECTED_BASE_SIZE:-16202421}
POSE_DATA=${POSE_DATA:-"${PROJECT_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"}
NAV2_DATA=${NAV2_DATA:-"${PROJECT_DIR}/Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv"}
EXPECTED_NAV2_SHA256=${EXPECTED_NAV2_SHA256:-"76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a"}
MODE_DATA=${MODE_DATA:-"${PROJECT_DIR}/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/command_balanced_directional_50hz/task_sampling_config.json"}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS_OVERRIDE=${MAX_ITERATIONS:-}
RUN_NAME=${RUN_NAME:-armhack_walk_robust_${PHASE}}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-True}
QUIET_TERMINAL=${QUIET_TERMINAL:-False}
RSI_ENABLE=${RSI_ENABLE:-False}
RSI_RATIO=${RSI_RATIO:-0.5}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
DESIRED_KL=${DESIRED_KL:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.002}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.003}
AMP_GRAD_PENALTY_SCALE=${AMP_GRAD_PENALTY_SCALE:-20.0}

RESUME_RUN=${RESUME_RUN:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
DISABLE_ACTUATOR_DR=False
DISABLE_LINK_MASS_DR=False

die() {
    echo "Error: $*" >&2
    exit 1
}

is_true() {
    [[ "$1" == "True" || "$1" == "true" || "$1" == "1" ]]
}

case "${PHASE}" in
    amp_warmup)
        PHASE_ITERATIONS=1000
        STYLE_REWARD_SCALE=1.0
        TASK_STYLE_LERP=0.85
        PAYLOAD_MAX_KG=0.0
        MODE_PROBABILITY=0.0
        NAV2_FAMILY="complex_turn"
        NAV2_COMMAND_SCALE="[0.85,0.75,0.75]"
        MODE_COMMAND_SCALE="[0.75,0.75,0.75]"
        RANDOMIZATION_STRENGTH=0
        DISABLE_PUSH=True
        ;;
    amp_target)
        PHASE_ITERATIONS=1000
        STYLE_REWARD_SCALE=3.0
        TASK_STYLE_LERP=0.70
        PAYLOAD_MAX_KG=0.0
        MODE_PROBABILITY=0.0
        NAV2_FAMILY="complex_turn"
        NAV2_COMMAND_SCALE="[0.85,0.75,0.75]"
        MODE_COMMAND_SCALE="[0.75,0.75,0.75]"
        RANDOMIZATION_STRENGTH=0
        DISABLE_PUSH=True
        ;;
    payload_half)
        PHASE_ITERATIONS=500
        STYLE_REWARD_SCALE=3.0
        TASK_STYLE_LERP=0.70
        PAYLOAD_MAX_KG=0.5
        MODE_PROBABILITY=0.0
        NAV2_FAMILY="complex_turn"
        NAV2_COMMAND_SCALE="[0.85,0.75,0.75]"
        MODE_COMMAND_SCALE="[0.75,0.75,0.75]"
        RANDOMIZATION_STRENGTH=0
        DISABLE_PUSH=True
        ;;
    payload_full)
        PHASE_ITERATIONS=1000
        STYLE_REWARD_SCALE=3.0
        TASK_STYLE_LERP=0.70
        PAYLOAD_MAX_KG=1.0
        MODE_PROBABILITY=0.0
        NAV2_FAMILY="complex_turn"
        NAV2_COMMAND_SCALE="[0.85,0.75,0.75]"
        MODE_COMMAND_SCALE="[0.75,0.75,0.75]"
        RANDOMIZATION_STRENGTH=0
        DISABLE_PUSH=True
        ;;
    command)
        PHASE_ITERATIONS=2000
        STYLE_REWARD_SCALE=5.0
        TASK_STYLE_LERP=0.60
        PAYLOAD_MAX_KG=1.0
        MODE_PROBABILITY=0.20
        NAV2_FAMILY="*"
        NAV2_COMMAND_SCALE="[1.0,1.0,1.0]"
        MODE_COMMAND_SCALE="[0.75,0.75,0.75]"
        RANDOMIZATION_STRENGTH=0
        DISABLE_PUSH=True
        ;;
    domain_base)
        PHASE_ITERATIONS=500
        STYLE_REWARD_SCALE=5.0
        TASK_STYLE_LERP=0.60
        PAYLOAD_MAX_KG=1.0
        MODE_PROBABILITY=0.30
        NAV2_FAMILY="*"
        NAV2_COMMAND_SCALE="[1.0,1.0,1.0]"
        MODE_COMMAND_SCALE="[1.0,1.0,1.0]"
        RANDOMIZATION_STRENGTH=1
        DISABLE_PUSH=True
        DISABLE_ACTUATOR_DR=True
        DISABLE_LINK_MASS_DR=True
        ;;
    domain_actuator)
        PHASE_ITERATIONS=500
        STYLE_REWARD_SCALE=5.0
        TASK_STYLE_LERP=0.60
        PAYLOAD_MAX_KG=1.0
        MODE_PROBABILITY=0.30
        NAV2_FAMILY="*"
        NAV2_COMMAND_SCALE="[1.0,1.0,1.0]"
        MODE_COMMAND_SCALE="[1.0,1.0,1.0]"
        RANDOMIZATION_STRENGTH=1
        DISABLE_PUSH=True
        DISABLE_LINK_MASS_DR=True
        ;;
    domain_link)
        PHASE_ITERATIONS=250
        STYLE_REWARD_SCALE=5.0
        TASK_STYLE_LERP=0.60
        PAYLOAD_MAX_KG=1.0
        MODE_PROBABILITY=0.30
        NAV2_FAMILY="*"
        NAV2_COMMAND_SCALE="[1.0,1.0,1.0]"
        MODE_COMMAND_SCALE="[1.0,1.0,1.0]"
        RANDOMIZATION_STRENGTH=1
        DISABLE_PUSH=True
        ;;
    robust)
        PHASE_ITERATIONS=250
        STYLE_REWARD_SCALE=5.0
        TASK_STYLE_LERP=0.60
        PAYLOAD_MAX_KG=1.0
        MODE_PROBABILITY=0.30
        NAV2_FAMILY="*"
        NAV2_COMMAND_SCALE="[1.0,1.0,1.0]"
        MODE_COMMAND_SCALE="[1.0,1.0,1.0]"
        RANDOMIZATION_STRENGTH=1
        DISABLE_PUSH=False
        ;;
    *)
        die "PHASE must be amp_warmup, amp_target, payload_half, payload_full, command, domain_base, domain_actuator, domain_link, or robust"
        ;;
esac
MAX_ITERATIONS=${MAX_ITERATIONS_OVERRIDE:-${PHASE_ITERATIONS}}
RESET_AMP_ON_LOAD=False
if [[ "${MODE}" == "resume" && "${PHASE}" == "amp_warmup" ]]; then
    RESET_AMP_ON_LOAD=True
fi

[[ -x "${ISAACLAB_PYTHON}" ]] || die "IsaacLab Python is not executable: ${ISAACLAB_PYTHON}"
[[ -f "${BASE_ONNX}" ]] || die "Walk locomotion ONNX not found: ${BASE_ONNX}"
[[ -f "${BASE_METADATA}" ]] || die "Walk locomotion metadata not found: ${BASE_METADATA}"
[[ -f "${BASE_CHECKPOINT}" ]] || die "Walk source checkpoint model_9996 not found: ${BASE_CHECKPOINT}"
[[ -f "${POSE_DATA}" ]] || die "Walk arm-pose JSON not found: ${POSE_DATA}"
[[ -f "${NAV2_DATA}" ]] || die "Walk Nav2 command CSV not found: ${NAV2_DATA}"
[[ -f "${MODE_DATA}" ]] || die "Walk mode-sampling JSON not found: ${MODE_DATA}"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITERATIONS must be a positive integer"
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS must be a positive integer"

case "${POSE_NAME}" in
    pos1_back|pos2_down|pos3_front|random) ;;
    *) die "POSE_NAME must be pos1_back, pos2_down, pos3_front, or random" ;;
esac
case "${MODE}" in
    init|resume) ;;
    *) die "MODE must be init or resume" ;;
esac
if [[ "${MODE}" == "init" && "${PHASE}" != "amp_warmup" ]] && ! is_true "${ALLOW_PHASE_INIT}"; then
    die "MODE=init is restricted to PHASE=amp_warmup; set ALLOW_PHASE_INIT=True only for smoke diagnostics"
fi

for arg in "$@"; do
    case "${arg}" in
        --task|--task=*|--resume|--resume=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*|env.scene.robot*|env.upper_body_perturbation.pose_name=*|env.commands.base_velocity.*|env.events.randomize_*_end_effector_payload*|env.events.physics_material*|env.events.add_base_mass*|env.events.randomize_rigid_body_com*|env.events.scale_link_mass*|env.events.scale_actuator_gains*|env.events.scale_joint_parameters*|env.events.push_robot*|agent.experiment_name=*|agent.load_policy_only=*|agent.reset_iteration_on_policy_only_load=*|agent.reset_amp_on_load=*|agent.checkpoint_output_dir=*|agent.algorithm.learning_rate=*|agent.algorithm.desired_kl=*|agent.algorithm.entropy_coef=*|agent.algorithm.amp_cfg.*|agent.algorithm.baseline_kl_cfg.*)
            die "Protected Walk training setting cannot be overridden: ${arg}"
            ;;
    esac
done

if pgrep -af 'train.py --task LeggedLab-Isaac-AMP-G1-Stand' >/dev/null; then
    die "Stand training is active; Walk refuses to start so it cannot disturb the Stand run"
fi
if pgrep -af 'train.py --task LeggedLab-Isaac-AMP-G1-Walk' >/dev/null; then
    die "another ArmHack Walk training process is already active"
fi

BASE_CHECKPOINT=$(realpath "${BASE_CHECKPOINT}")
BASE_ONNX=$(realpath "${BASE_ONNX}")
BASE_METADATA=$(realpath "${BASE_METADATA}")
ACTUAL_ONNX_SHA256=$(sha256sum "${BASE_ONNX}" | awk '{print $1}')
ACTUAL_BASE_SHA256=$(sha256sum "${BASE_CHECKPOINT}" | awk '{print $1}')
ACTUAL_BASE_SIZE=$(stat -c '%s' "${BASE_CHECKPOINT}")
[[ "${ACTUAL_ONNX_SHA256}" == "${EXPECTED_ONNX_SHA256}" ]] || \
    die "locomotion.onnx SHA-256 mismatch: expected ${EXPECTED_ONNX_SHA256}, got ${ACTUAL_ONNX_SHA256}"
[[ "${ACTUAL_BASE_SHA256}" == "${EXPECTED_BASE_SHA256}" ]] || \
    die "model_9996 SHA-256 mismatch: expected ${EXPECTED_BASE_SHA256}, got ${ACTUAL_BASE_SHA256}"
[[ "${ACTUAL_BASE_SIZE}" == "${EXPECTED_BASE_SIZE}" ]] || \
    die "model_9996 size mismatch: expected ${EXPECTED_BASE_SIZE}, got ${ACTUAL_BASE_SIZE}"
ACTUAL_NAV2_SHA256=$(sha256sum "${NAV2_DATA}" | awk '{print $1}')
[[ "${ACTUAL_NAV2_SHA256}" == "${EXPECTED_NAV2_SHA256}" ]] || \
    die "Nav2 command CSV SHA-256 mismatch: expected ${EXPECTED_NAV2_SHA256}, got ${ACTUAL_NAV2_SHA256}"

"${ISAACLAB_PYTHON}" - "${BASE_CHECKPOINT}" "${BASE_ONNX}" "${BASE_METADATA}" "${MODE_DATA}" <<'PY'
import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
onnx_model = onnx.load(sys.argv[2])
onnx.checker.check_model(onnx_model)
with open(sys.argv[3], encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
with open(sys.argv[4], encoding="utf-8") as mode_file:
    mode_config = json.load(mode_file)
state = checkpoint.get("model_state_dict", {})
expected = {
    "actor.0.weight": (512, 96),
    "actor.6.weight": (29, 128),
    "critic.0.weight": (512, 297),
    "critic.6.weight": (1, 128),
}
errors = []
if int(checkpoint.get("iter", -1)) != 9996:
    errors.append(f"iteration={checkpoint.get('iter')!r}, expected 9996")
for key, shape in expected.items():
    actual = tuple(state[key].shape) if key in state else None
    if actual != shape:
        errors.append(f"{key} shape={actual}, expected {shape}")
if metadata.get("robot") != "g1" or metadata.get("obs_dim") != 96 or metadata.get("action_dim") != 29:
    errors.append("locomotion.deploy.json must describe the G1 96->29 actor")
if metadata.get("action_joint_names") != metadata.get("full_joint_names"):
    errors.append("locomotion.deploy.json action/full joint order mismatch")
required_modes = {
    "stand", "forward_slow", "forward_normal", "backward", "lateral_left",
    "lateral_right", "turn_left", "turn_right",
}
if set(mode_config.get("modes", {})) != required_modes:
    errors.append("task_sampling_config.json must contain the exact eight directional modes")

onnx_tensors = {item.name: numpy_helper.to_array(item) for item in onnx_model.graph.initializer}
actor_keys = [
    "actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias",
    "actor.4.weight", "actor.4.bias", "actor.6.weight", "actor.6.bias",
]
for key in actor_keys:
    if key not in onnx_tensors or key not in state:
        errors.append(f"missing actor tensor: {key}")
    elif not np.array_equal(onnx_tensors[key], state[key].cpu().numpy()):
        errors.append(f"ONNX/checkpoint actor mismatch: {key}")
if errors:
    raise SystemExit("Invalid Walk baseline/data: " + "; ".join(errors))
print("Verified locomotion.onnx == model_9996 actor: iteration=9996, actor=96->29, critic=297->1")
print("Verified hybrid command config: eight directional modes")
PY

LOG_ROOT="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
LEGACY_LOG_ROOT="${PROJECT_DIR}/logs/rsl_rl/${LEGACY_EXPERIMENT_NAME}"
LOAD_POLICY_ONLY=True
RESET_ITERATION=True

if [[ "${MODE}" == "init" ]]; then
    BASE_RUN_NAME="_armhack_walk_baseline_locomotion_model9996"
    BASE_RUN_DIR="${LOG_ROOT}/${BASE_RUN_NAME}"
    mkdir -p "${BASE_RUN_DIR}"
    ln -sfn "${BASE_CHECKPOINT}" "${BASE_RUN_DIR}/model_9996.pt"
    LOAD_RUN="^${BASE_RUN_NAME}$"
    CHECKPOINT="^model_9996.pt$"
else
    [[ -n "${RESUME_RUN}" ]] || die "MODE=resume requires RESUME_RUN"
    [[ "${RESUME_RUN}" != */* ]] || die "RESUME_RUN must be one Walk run-directory name"
    [[ "${RESUME_CHECKPOINT}" =~ ^model_[0-9]+\.pt$ ]] || \
        die "MODE=resume requires RESUME_CHECKPOINT=model_<iteration>.pt"
    LOG_CHECKPOINT="${LOG_ROOT}/${RESUME_RUN}/${RESUME_CHECKPOINT}"
    if [[ ! -f "${LOG_CHECKPOINT}" && -f "${LEGACY_LOG_ROOT}/${RESUME_RUN}/${RESUME_CHECKPOINT}" ]]; then
        mkdir -p "${LOG_ROOT}"
        ln -sfn "${LEGACY_LOG_ROOT}/${RESUME_RUN}" "${LOG_ROOT}/${RESUME_RUN}"
        LOG_CHECKPOINT="${LOG_ROOT}/${RESUME_RUN}/${RESUME_CHECKPOINT}"
    fi
    EXPORTED_CHECKPOINT="${PROJECT_DIR}/ArmHack Checkpoints/WalkPerturbFinetune/${RESUME_RUN}/${RESUME_CHECKPOINT}"
    [[ -f "${LOG_CHECKPOINT}" ]] || die "Walk log checkpoint not found: ${LOG_CHECKPOINT}"
    [[ -f "${EXPORTED_CHECKPOINT}" ]] || die "Walk exported checkpoint not found: ${EXPORTED_CHECKPOINT}"
    [[ "$(sha256sum "${LOG_CHECKPOINT}" | awk '{print $1}')" == "$(sha256sum "${EXPORTED_CHECKPOINT}" | awk '{print $1}')" ]] || \
        die "Walk log and exported resume checkpoints do not match"
    LOAD_RUN="^${RESUME_RUN}$"
    CHECKPOINT="^${RESUME_CHECKPOINT}$"
    LOAD_POLICY_ONLY=False
    RESET_ITERATION=False
fi

echo "=================================================="
echo " ArmHack Walk P0/P1 robust fine-tune"
echo "=================================================="
echo "Mode / phase      : ${MODE} / ${PHASE}"
echo "Pose              : ${POSE_NAME}"
echo "Task / asset      : ${TASK} / ${ROBOT_ASSET_NAME}"
echo "Base checkpoint   : ${BASE_CHECKPOINT}"
echo "Resume selector   : ${LOAD_RUN} / ${CHECKPOINT}"
echo "Training          : +${MAX_ITERATIONS} iterations, ${NUM_ENVS} envs, seed ${SEED}"
echo "AMP               : style=${STYLE_REWARD_SCALE}, task_lerp=${TASK_STYLE_LERP}, grad_penalty=${AMP_GRAD_PENALTY_SCALE}"
echo "Payload           : independent left/right U(0, ${PAYLOAD_MAX_KG} kg)"
echo "Commands          : Nav2 family=${NAV2_FAMILY}, mode_probability=${MODE_PROBABILITY}"
echo "Command scales    : Nav2=${NAV2_COMMAND_SCALE}, mode=${MODE_COMMAND_SCALE}"
echo "Physical DR / push: ${RANDOMIZATION_STRENGTH} / $([[ "${DISABLE_PUSH}" == "True" ]] && echo off || echo on)"
echo "Actuator/link DR  : $([[ "${DISABLE_ACTUATOR_DR}" == "True" ]] && echo off || echo on) / $([[ "${DISABLE_LINK_MASS_DR}" == "True" ]] && echo off || echo on)"
echo "Optimizer         : lr=${LEARNING_RATE}, desired_kl=${DESIRED_KL}, entropy=${ENTROPY_COEF}"
echo "Baseline KL       : ${BASELINE_KL_SCALE}"
echo "Reset AMP on load : ${RESET_AMP_ON_LOAD}"
echo "Run name          : ${RUN_NAME}"
echo "=================================================="

PHASE_HYDRA_ARGS=(
    "env.upper_body_perturbation.pose_name=${POSE_NAME}"
    "env.commands.base_velocity.scenario_family_filter=${NAV2_FAMILY}"
    "env.commands.base_velocity.command_scale=${NAV2_COMMAND_SCALE}"
    "env.commands.base_velocity.mode_probability=${MODE_PROBABILITY}"
    "env.commands.base_velocity.mode_command_scale=${MODE_COMMAND_SCALE}"
    "env.events.randomize_left_end_effector_payload.params.mass_distribution_params=[0.0,${PAYLOAD_MAX_KG}]"
    "env.events.randomize_right_end_effector_payload.params.mass_distribution_params=[0.0,${PAYLOAD_MAX_KG}]"
    "agent.algorithm.learning_rate=${LEARNING_RATE}"
    "agent.algorithm.desired_kl=${DESIRED_KL}"
    "agent.load_policy_only=${LOAD_POLICY_ONLY}"
    "agent.reset_iteration_on_policy_only_load=${RESET_ITERATION}"
    "agent.reset_amp_on_load=${RESET_AMP_ON_LOAD}"
)
if [[ "${DISABLE_PUSH}" == "True" && "${RANDOMIZATION_STRENGTH}" != "0" ]]; then
    PHASE_HYDRA_ARGS+=("env.events.push_robot=null")
fi
if [[ "${DISABLE_ACTUATOR_DR}" == "True" && "${RANDOMIZATION_STRENGTH}" != "0" ]]; then
    PHASE_HYDRA_ARGS+=("env.events.scale_actuator_gains=null")
    PHASE_HYDRA_ARGS+=("env.events.scale_joint_parameters=null")
fi
if [[ "${DISABLE_LINK_MASS_DR}" == "True" && "${RANDOMIZATION_STRENGTH}" != "0" ]]; then
    PHASE_HYDRA_ARGS+=("env.events.scale_link_mass=null")
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
ROBOT_ASSET="${ROBOT_ASSET_NAME}" \
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
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
    "${PHASE_HYDRA_ARGS[@]}" \
    "$@"
