#!/usr/bin/env bash
# Test or visualize one ArmHack Walk robust checkpoint in Isaac Sim.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TASK="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-Play-v0"
CHECKPOINT=${CHECKPOINT:-}
POSE_NAME=${POSE_NAME:-pos2_down}
PAYLOAD_KG=${PAYLOAD_KG:-0.0}
LEFT_PAYLOAD_KG=${LEFT_PAYLOAD_KG:-${PAYLOAD_KG}}
RIGHT_PAYLOAD_KG=${RIGHT_PAYLOAD_KG:-${PAYLOAD_KG}}
COMMAND_SOURCE=${COMMAND_SOURCE:-nav2}
MODE_NAME=${MODE_NAME:-forward_slow}
MODE_PROBABILITY=${MODE_PROBABILITY:-0.30}
NAV2_FAMILY=${NAV2_FAMILY:-"*"}
NUM_ENVS=${NUM_ENVS:-1}
MAX_STEPS=${MAX_STEPS:-1000}
SEED=${SEED:-42}
HEADLESS=${HEADLESS:-False}
REAL_TIME=${REAL_TIME:-True}
FOLLOW_CAMERA=${FOLLOW_CAMERA:-True}
CAMERA_VIEW=${CAMERA_VIEW:-front}
VIDEO=${VIDEO:-False}
VIDEO_LENGTH=${VIDEO_LENGTH:-${MAX_STEPS}}
SKIP_EXPORT=${SKIP_EXPORT:-True}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_isaaclab}
CONDA_BASE=${CONDA_BASE:-${HOME}/anaconda3}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python}

die() {
    echo "Error: $*" >&2
    exit 1
}

[[ -n "${CHECKPOINT}" ]] || die "CHECKPOINT must point to a Walk model_*.pt"
[[ -x "${ISAACLAB_PYTHON}" ]] || die "IsaacLab Python is not executable: ${ISAACLAB_PYTHON}"
if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${PROJECT_DIR}/${CHECKPOINT}"
fi
[[ -f "${CHECKPOINT}" ]] || die "Walk checkpoint not found: ${CHECKPOINT}"
CHECKPOINT=$(realpath "${CHECKPOINT}")
[[ "$(basename "${CHECKPOINT}")" =~ ^model_[0-9]+\.pt$ ]] || \
    die "CHECKPOINT basename must be model_<iteration>.pt"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be a positive integer"

case "${POSE_NAME}" in
    pos1_back|pos2_down|pos3_front|random) ;;
    *) die "POSE_NAME must be pos1_back, pos2_down, pos3_front, or random" ;;
esac
case "${COMMAND_SOURCE}" in
    nav2)
        EFFECTIVE_MODE_PROBABILITY=0.0
        FORCED_MODE=""
        ;;
    hybrid)
        EFFECTIVE_MODE_PROBABILITY=${MODE_PROBABILITY}
        FORCED_MODE=""
        ;;
    mode)
        EFFECTIVE_MODE_PROBABILITY=1.0
        FORCED_MODE=${MODE_NAME}
        case "${FORCED_MODE}" in
            stand|forward_slow|forward_normal|backward|lateral_left|lateral_right|turn_left|turn_right) ;;
            *) die "MODE_NAME is not one of the eight command-balanced modes" ;;
        esac
        ;;
    *) die "COMMAND_SOURCE must be nav2, hybrid, or mode" ;;
esac

"${ISAACLAB_PYTHON}" - "${LEFT_PAYLOAD_KG}" "${RIGHT_PAYLOAD_KG}" "${EFFECTIVE_MODE_PROBABILITY}" <<'PY'
import sys

left_payload = float(sys.argv[1])
right_payload = float(sys.argv[2])
probability = float(sys.argv[3])
if not 0.0 <= left_payload <= 1.0 or not 0.0 <= right_payload <= 1.0:
    raise SystemExit("LEFT_PAYLOAD_KG and RIGHT_PAYLOAD_KG must be in [0, 1]")
if not 0.0 <= probability <= 1.0:
    raise SystemExit("MODE_PROBABILITY must be in [0, 1]")
PY

HYDRA_ARGS=(
    --seed "${SEED}"
    "env.upper_body_perturbation.pose_name=${POSE_NAME}"
    "env.events.randomize_left_end_effector_payload.params.mass_distribution_params=[${LEFT_PAYLOAD_KG},${LEFT_PAYLOAD_KG}]"
    "env.events.randomize_right_end_effector_payload.params.mass_distribution_params=[${RIGHT_PAYLOAD_KG},${RIGHT_PAYLOAD_KG}]"
    "env.commands.base_velocity.scenario_family_filter=${NAV2_FAMILY}"
    "env.commands.base_velocity.command_scale=[1.0,1.0,1.0]"
    "env.commands.base_velocity.mode_probability=${EFFECTIVE_MODE_PROBABILITY}"
    "env.commands.base_velocity.mode_command_scale=[1.0,1.0,1.0]"
)
if [[ -n "${FORCED_MODE}" ]]; then
    HYDRA_ARGS+=("env.commands.base_velocity.forced_mode=${FORCED_MODE}")
fi
if [[ "${VIDEO}" == "True" || "${VIDEO}" == "true" || "${VIDEO}" == "1" ]]; then
    HYDRA_ARGS+=(--video --video_length "${VIDEO_LENGTH}")
fi

echo "=================================================="
echo " ArmHack Walk test / visualization"
echo "=================================================="
echo "Checkpoint       : ${CHECKPOINT}"
echo "Pose / payload   : ${POSE_NAME} / left=${LEFT_PAYLOAD_KG}, right=${RIGHT_PAYLOAD_KG} kg"
echo "Command source   : ${COMMAND_SOURCE}, mode=${FORCED_MODE:-sampled}, p=${EFFECTIVE_MODE_PROBABILITY}"
echo "Nav2 family      : ${NAV2_FAMILY}"
echo "Steps / envs     : ${MAX_STEPS} / ${NUM_ENVS}"
echo "Headless / video : ${HEADLESS} / ${VIDEO}"
echo "=================================================="

TASK="${TASK}" \
CHECKPOINT="${CHECKPOINT}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_STEPS="${MAX_STEPS}" \
SEED="${SEED}" \
HEADLESS="${HEADLESS}" \
REAL_TIME="${REAL_TIME}" \
FOLLOW_CAMERA="${FOLLOW_CAMERA}" \
CAMERA_VIEW="${CAMERA_VIEW}" \
SKIP_EXPORT="${SKIP_EXPORT}" \
ROBOT_ASSET=s3_g1_29dof \
ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
bash "${PROJECT_DIR}/scripts/vis_isaacsim_g1_amp.sh" \
    "${HYDRA_ARGS[@]}" \
    "$@"
