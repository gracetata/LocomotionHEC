#!/usr/bin/env bash
# Train exactly one ArmHack actor (Stand or Walk) from the original source checkpoint.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKTREE_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
SOURCE_REPO=${SOURCE_REPO:-/home/tata/Workspace/Locomotion/G1-Locomotion}
MODEL=${MODEL:?set MODEL=stand or MODEL=walk}
STAND_PROFILE=${STAND_PROFILE:-acquisition}
WALK_PROFILE=${WALK_PROFILE:-acquisition}
SMOKE=${SMOKE:-False}
NUM_ENVS=${NUM_ENVS:-6144}
MAX_ITERATIONS=${MAX_ITERATIONS:-2000}
SEED=${SEED:-20260823}
DEVICE=${DEVICE:-cuda:0}
ALLOW_CONCURRENT_GPU_RUNS=${ALLOW_CONCURRENT_GPU_RUNS:-False}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-/home/tata/anaconda3/envs/env_isaaclab/bin/python}
HANDOFF_STATE_LIBRARY=${HANDOFF_STATE_LIBRARY:-}
HANDOFF_RESET_PROBABILITY=${HANDOFF_RESET_PROBABILITY:-0.0}
RUN_NAME=${RUN_NAME:-armhack_${MODEL}_first_principles_single_2000_20260823}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-${LEGGED_LAB_DIR}/logs/monitoring/${RUN_NAME}.log}

die() { echo "Error: $*" >&2; exit 1; }
case "$(hostname)" in
    tata-futurelab|hecggdz-System-Product-Name|wenduo-System-Product-Name) ;;
    *) die "training host is not an approved Future/HEC/Dual 5090 worker" ;;
esac
[[ -x "${ISAACLAB_PYTHON}" ]] || die "IsaacLab Python missing: ${ISAACLAB_PYTHON}"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 5090' || die "RTX 5090 not detected"
[[ "${MODEL}" == "stand" || "${MODEL}" == "walk" ]] || die "MODEL must be stand or walk"
[[ "${MAX_ITERATIONS}" =~ ^[0-9]+$ ]] || die "MAX_ITERATIONS must be an integer"
if [[ "${SMOKE}" != "True" && "${MAX_ITERATIONS}" -lt 2000 ]]; then
    die "formal continuation runs must be at least 2000 iterations"
fi
if [[ "${ALLOW_CONCURRENT_GPU_RUNS}" == "True" || "${ALLOW_CONCURRENT_GPU_RUNS}" == "true" || \
      "${ALLOW_CONCURRENT_GPU_RUNS}" == "1" ]]; then
    [[ "${DEVICE}" =~ ^cuda:[0-9]+$ ]] || \
        die "concurrent worker runs require an explicit cuda:<index> device"
    if pgrep -af "scripts/rsl_rl/train.py.*ArmHack.*FirstPrinciplesSingle.*--device ${DEVICE}([[:space:]]|$)" >/dev/null; then
        die "another first-principles ArmHack training process is active on ${DEVICE}"
    fi
elif pgrep -af 'scripts/rsl_rl/train.py.*ArmHack.*FirstPrinciplesSingle' >/dev/null; then
    die "another first-principles ArmHack training process is active"
fi

if [[ "${MODEL}" == "stand" ]]; then
    case "${STAND_PROFILE}" in
        acquisition) TASK=LeggedLab-Isaac-AMP-G1-ArmHackStandFirstPrinciplesSingle-v0 ;;
        strict) TASK=LeggedLab-Isaac-AMP-G1-ArmHackStandFirstPrinciplesStrictSingle-v0 ;;
        one_step) TASK=LeggedLab-Isaac-AMP-G1-ArmHackStandFirstPrinciplesOneStepSingle-v0 ;;
        *) die "STAND_PROFILE must be acquisition, strict or one_step" ;;
    esac
    EXPERIMENT=g1_armhack_stand_first_principles_single
    SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${SOURCE_REPO}/legged_lab/logs/rsl_rl/g1_stand_perturb/2026-08-14_18-32-52_armhack_stand_low_torque_robust_explicit_3pose_2000_from_stage2_20260814/model_1999.pt}
    EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:-9ab48719840c98f1332693a56f58ed069463c0670737e339b90411985484a729}
    STYLE_REWARD_SCALE=0.0
    TASK_STYLE_LERP=1.0
    BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-2.0e-4}
else
    case "${WALK_PROFILE}" in
        acquisition) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesSingle-v0 ;;
        strict) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesStrictSingle-v0 ;;
        robust) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesRobustSingle-v0 ;;
        response) TASK=LeggedLab-Isaac-AMP-G1-ArmHackWalkFirstPrinciplesResponseSingle-v0 ;;
        *) die "WALK_PROFILE must be acquisition, strict, robust or response" ;;
    esac
    EXPERIMENT=g1_armhack_walk_first_principles_single
    SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${SOURCE_REPO}/legged_lab/ArmHack Checkpoints/WalkAnkleSpacingFinetune/base/2026-08-14_16-56-58_ankle30_base_full_20260814/model_199.pt}
    EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:-9d4583a535ea67086f429b20793a4f75dd00afbacab7c2aee5bc868be5a6e355}
    STYLE_REWARD_SCALE=1.0
    TASK_STYLE_LERP=0.90
    BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-5.0e-4}
fi

[[ -f "${SOURCE_CHECKPOINT}" ]] || die "source checkpoint missing: ${SOURCE_CHECKPOINT}"
SOURCE_CHECKPOINT=$(realpath "${SOURCE_CHECKPOINT}")
ACTUAL_SOURCE_SHA256=$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')
[[ "${ACTUAL_SOURCE_SHA256}" == "${EXPECTED_SOURCE_SHA256}" ]] || \
    die "source checkpoint SHA mismatch: ${ACTUAL_SOURCE_SHA256}"
if [[ -n "${HANDOFF_STATE_LIBRARY}" ]]; then
    [[ -f "${HANDOFF_STATE_LIBRARY}" ]] || die "handoff state library missing: ${HANDOFF_STATE_LIBRARY}"
    HANDOFF_STATE_LIBRARY=$(realpath "${HANDOFF_STATE_LIBRARY}")
fi

STAGING_RUN=_source_${MODEL}_${ACTUAL_SOURCE_SHA256:0:12}
STAGING_DIR=${LEGGED_LAB_DIR}/logs/rsl_rl/${EXPERIMENT}/${STAGING_RUN}
mkdir -p "${STAGING_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${SOURCE_CHECKPOINT}" "${STAGING_DIR}/model_source.pt"

echo "model=${MODEL} task=${TASK} actor=ActorCritic(single)"
echo "source=${SOURCE_CHECKPOINT} sha256=${ACTUAL_SOURCE_SHA256}"
echo "training=${NUM_ENVS} envs x ${MAX_ITERATIONS} iterations device=${DEVICE}"
echo "handoff=${HANDOFF_STATE_LIBRARY:-disabled} probability=${HANDOFF_RESET_PROBABILITY}"
echo "log=${TRAIN_LOG_FILE}"

export PYTHONPATH="${WORKTREE_ROOT}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
TASK="${TASK}" NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" \
SEED="${SEED}" DEVICE="${DEVICE}" AGENT_DEVICE="${DEVICE}" \
ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" RUN_NAME="${RUN_NAME}" \
RESUME=True LOAD_RUN="^${STAGING_RUN}$" CHECKPOINT='^model_source.pt$' \
HEADLESS=True QUIET_TERMINAL=True TRAIN_LOG_FILE="${TRAIN_LOG_FILE}" \
ROBOT_ASSET=s3_g1_29dof RSI_ENABLE=False RANDOMIZATION_STRENGTH=1 \
STYLE_REWARD_SCALE="${STYLE_REWARD_SCALE}" TASK_STYLE_LERP="${TASK_STYLE_LERP}" \
BASELINE_KL_ENABLE=True BASELINE_KL_CHECKPOINT="${SOURCE_CHECKPOINT}" \
BASELINE_KL_SCALE="${BASELINE_KL_SCALE}" \
bash "${LEGGED_LAB_DIR}/scripts/train_g1_amp.sh" \
    agent.policy.class_name=ActorCritic \
    agent.load_policy_only=True \
    agent.reset_iteration_on_policy_only_load=True \
    agent.restore_configured_learning_rate_on_load=True \
    agent.save_interval=100 \
    agent.algorithm.baseline_kl_cfg.enabled=True \
    agent.algorithm.baseline_kl_cfg.checkpoint_path="${SOURCE_CHECKPOINT}" \
    agent.algorithm.baseline_kl_cfg.scale="${BASELINE_KL_SCALE}" \
    env.events.handoff_state_reset.params.state_library_path="${HANDOFF_STATE_LIBRARY}" \
    env.events.handoff_state_reset.params.probability="${HANDOFF_RESET_PROBABILITY}" \
    "$@"

grep -q 'Learning iteration' "${TRAIN_LOG_FILE}" || die "no PPO iteration was recorded"
if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|nan detected|TypeError:' "${TRAIN_LOG_FILE}"; then
    die "fatal error detected in training log"
fi
if [[ "${SMOKE}" != "True" ]]; then
    FINAL_ITERATION=$((MAX_ITERATIONS - 1))
    grep -q "Learning iteration ${FINAL_ITERATION}/${MAX_ITERATIONS}" "${TRAIN_LOG_FILE}" \
        || die "formal run stopped before iteration ${FINAL_ITERATION}/${MAX_ITERATIONS}"
    grep -q "model_${FINAL_ITERATION}.pt" "${TRAIN_LOG_FILE}" \
        || die "formal run did not save model_${FINAL_ITERATION}.pt"
fi
echo "Training completed: ${RUN_NAME}"
