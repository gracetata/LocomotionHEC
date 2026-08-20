#!/usr/bin/env bash
# Continue the requested first-generation ArmHack Stand checkpoint with
# contact-force-selected two-step recovery and reset-relative SE(2) anchoring.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/2026-08-14_18-32-52_armhack_stand_low_torque_robust_explicit_3pose_2000_from_stage2_20260814/model_1999.pt"}
SOURCE_SHA256=${SOURCE_SHA256:-9ab48719840c98f1332693a56f58ed069463c0670737e339b90411985484a729}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-2000}
SEED=${SEED:-20260820}
RUN_NAME=${RUN_NAME:-armhack_stand_adaptive_contact_first_se2_30cm_20260820}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-${HOME}/anaconda3/envs/env_isaaclab/bin/python}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-"${PROJECT_DIR}/logs/monitoring/${RUN_NAME}.log"}
POSE_BANK=${POSE_BANK:-"${PROJECT_DIR}/Reference Data/ArmHack/StandPerturb/RandomizedTraining/random_arm_pose_bank_seed20260715.json"}

die() { echo "Error: $*" >&2; exit 1; }
[[ -x "${ISAACLAB_PYTHON}" ]] || die "Isaac Python missing: ${ISAACLAB_PYTHON}"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "Stand source missing: ${SOURCE_CHECKPOINT}"
[[ "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] || die "Stand source SHA mismatch"
[[ -f "${POSE_BANK}" ]] || die "pose bank missing: ${POSE_BANK}"

STAGING_RUN=_source_stand_adaptive_${SOURCE_SHA256:0:12}
STAGING_DIR="${PROJECT_DIR}/logs/rsl_rl/g1_stand_perturb/${STAGING_RUN}"
mkdir -p "${STAGING_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "$(realpath "${SOURCE_CHECKPOINT}")" "${STAGING_DIR}/model_source.pt"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
TASK=LeggedLab-Isaac-AMP-G1-StandAdaptiveSwitch-v0 \
NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" SEED="${SEED}" \
DEVICE=cuda:0 AGENT_DEVICE=cuda:0 ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" \
RUN_NAME="${RUN_NAME}" RESUME=True LOAD_RUN="^${STAGING_RUN}$" CHECKPOINT='^model_source.pt$' \
HEADLESS=True QUIET_TERMINAL=True TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" ROBOT_ASSET=s3_g1_29dof \
RSI_ENABLE=False RANDOMIZATION_STRENGTH=1 STYLE_REWARD_SCALE=0.0 TASK_STYLE_LERP=1.0 \
BASELINE_KL_ENABLE=True BASELINE_KL_CHECKPOINT="${SOURCE_CHECKPOINT}" BASELINE_KL_SCALE=0.0003 \
bash "${PROJECT_DIR}/scripts/train_g1_amp.sh" \
  "env.upper_body_perturbation.random_pose_bank_path='$(realpath "${POSE_BANK}")'" \
  env.events.reset_robot_joints.params.phase_one_probability=0.0 \
  env.events.reset_robot_joints.params.phase_two_probability=0.0 \
  agent.load_policy_only=True agent.reset_iteration_on_policy_only_load=True \
  agent.policy_only_noise_std_override=0.15 \
  agent.algorithm.learning_rate=8.0e-6 agent.algorithm.schedule=fixed \
  agent.algorithm.entropy_coef=0.0005 agent.algorithm.desired_kl=0.01 \
  agent.algorithm.baseline_kl_cfg.enabled=True \
  agent.algorithm.baseline_kl_cfg.checkpoint_path="$(realpath "${SOURCE_CHECKPOINT}")" \
  agent.algorithm.baseline_kl_cfg.scale=0.0003 \
  agent.algorithm.baseline_kl_cfg.exempt_obs_index=94 \
  agent.algorithm.baseline_kl_cfg.exempt_obs_threshold=0.5 \
  agent.algorithm.baseline_kl_cfg.mirror_phase_one=True \
  agent.algorithm.baseline_kl_cfg.lift_obs_index=95 \
  "$@"
