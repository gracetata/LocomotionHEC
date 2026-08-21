#!/usr/bin/env bash
# Refine the latest 30-cm ArmHack Walk actor for useful low-speed tracking,
# zero-to-motion transitions and higher swing-foot clearance.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"${PROJECT_DIR}/logs/training_sources/walk_ankle30_92c51_split/source_base.pt"}
SOURCE_SHA256=${SOURCE_SHA256:-62ee29b8c4fbbf8a4b96424d3cdffd698f89eeacab860dd6f3081edd6e1413d4}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-600}
SEED=${SEED:-20260821}
RUN_NAME=${RUN_NAME:-armhack_walk_precision_switch_30cm_20260820}
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-ArmHackWalkPrecisionSwitch-v0}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${PROJECT_DIR}/logs/monitoring/${RUN_NAME}.log"}
KL_SCALE=${KL_SCALE:-0.20}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-${SOURCE_CHECKPOINT}}
ZERO_COMMAND_TEACHER_ONLY=${ZERO_COMMAND_TEACHER_ONLY:-False}
KL_HARD_LIMIT=${KL_HARD_LIMIT:-0.25}

die() { echo "Error: $*" >&2; exit 1; }
[[ -x "${ISAACLAB_PYTHON}" ]] || die "Isaac Python missing: ${ISAACLAB_PYTHON}"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "Walk source missing: ${SOURCE_CHECKPOINT}"
[[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] || die "Walk source SHA mismatch"
[[ -f "${TEACHER_CHECKPOINT}" ]] || die "Walk transition teacher missing: ${TEACHER_CHECKPOINT}"

EXPERIMENT=g1_armhack_walk_precision_switch
STAGING_RUN=_source_walk_precision_${SOURCE_SHA256:0:12}
STAGING_DIR="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT}/${STAGING_RUN}"
mkdir -p "${STAGING_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "$(realpath "${SOURCE_CHECKPOINT}")" "${STAGING_DIR}/model_source.pt"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
TASK="${TASK}" \
NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" RESUME=True LOAD_RUN="^${STAGING_RUN}$" CHECKPOINT='^model_source.pt$' \
HEADLESS=True QUIET_TERMINAL=True TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False RANDOMIZATION_STRENGTH=1 STYLE_REWARD_SCALE=5.0 TASK_STYLE_LERP=1.0 \
AMP_GRAD_PENALTY_SCALE=20.0 BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT="${TEACHER_CHECKPOINT}" BASELINE_KL_SCALE="${KL_SCALE}" \
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
  agent.experiment_name="${EXPERIMENT}" \
  agent.checkpoint_output_dir="ArmHack Checkpoints/WalkPrecisionSwitch" \
  agent.load_policy_only=True agent.reset_iteration_on_policy_only_load=True \
  agent.freeze_base_actor=False agent.freeze_actor_hidden_layers=0 \
  agent.restore_configured_learning_rate_on_load=True \
  agent.algorithm.learning_rate=1.0e-5 agent.algorithm.schedule=fixed \
  agent.algorithm.entropy_coef=0.0003 agent.algorithm.desired_kl=0.01 \
  agent.algorithm.baseline_kl_cfg.enabled=True \
  agent.algorithm.baseline_kl_cfg.checkpoint_path="$(realpath "${TEACHER_CHECKPOINT}")" \
  agent.algorithm.baseline_kl_cfg.scale="${KL_SCALE}" \
  agent.algorithm.baseline_kl_cfg.hard_limit="${KL_HARD_LIMIT}" \
  agent.algorithm.baseline_kl_cfg.zero_command_only="${ZERO_COMMAND_TEACHER_ONLY}" \
  agent.algorithm.baseline_kl_cfg.zero_command_threshold=0.02 \
  agent.algorithm.baseline_kl_cfg.zero_command_obs_start_index=6 \
  agent.algorithm.amp_cfg.freeze_discriminator=True \
  "$@"
