#!/usr/bin/env bash
# Train the Unitree G1 29DoF AMP ToTarget precise positioning task.
#
# Usage:
#   bash scripts/train_g1_toTarget.sh
#   NUM_ENVS=128 MAX_ITERATIONS=2 bash scripts/train_g1_toTarget.sh
#   RESUME=True LOAD_RUN=<run> CHECKPOINT=model_4999.pt bash scripts/train_g1_toTarget.sh
#   TO_TARGET_PRESET=command_balanced bash scripts/train_g1_toTarget.sh
#   TASK=LeggedLab-Isaac-AMP-G1-ToTarget-v0 RUN_NAME=s3_g1_29dof_toTarget_cmu_core_slow_amp_5000 bash scripts/train_g1_toTarget.sh
#
# The script delegates to train_g1_amp.sh after setting ToTarget defaults, so
# all existing variables from train_g1_amp.sh remain available.
# Environment variables:
#   TO_TARGET_PRESET : v2/command_balanced/custom (default: v2 unless TASK implies otherwise).
#   TO_TARGET_V2_EXTRA_ARGS: Extra Hydra defaults appended for v2. Set to empty to disable.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]

# 1. ToTarget v2 smoke run：使用当前推荐的 v2 任务和参数。
NUM_ENVS=128 MAX_ITERATIONS=2 RUN_NAME=smoke_s3_g1_toTarget_v2 bash scripts/train_g1_toTarget.sh

# 2. ToTarget v2 正式 5000 iter：从稳定行走 checkpoint warm-start。
RESUME=True LOAD_RUN=2026-06-15_00-54-43_probe_s3_g1_toTarget_v2_warmstart_stop_amp CHECKPOINT=model_5098.pt MAX_ITERATIONS=5000 RUN_NAME=s3_g1_toTarget_v2_warmstart_stop_amp_5000 bash scripts/train_g1_toTarget.sh

# 3. ToTarget v2 继续加强 1m 边界：从最终 checkpoint 续训，适合远目标误差还想压低时使用。
RESUME=True LOAD_RUN=2026-06-15_01-02-37_s3_g1_toTarget_v2_warmstart_stop_amp_5000 CHECKPOINT=model_10097.pt MAX_ITERATIONS=1000 RUN_NAME=s3_g1_toTarget_v2_continue_full_radius_1000 bash scripts/train_g1_toTarget.sh

# 4. 旧 command-balanced ToTarget 任务入口：保留用于对照旧实验。
TO_TARGET_PRESET=command_balanced MAX_ITERATIONS=5000 RUN_NAME=s3_g1_29dof_toTarget_command_balanced_amp_5000 bash scripts/train_g1_toTarget.sh

# 5. 旧 ToTarget task id 对照入口。
TASK=LeggedLab-Isaac-AMP-G1-ToTarget-v0 RUN_NAME=s3_g1_29dof_toTarget_cmu_core_slow_amp_5000 bash scripts/train_g1_toTarget.sh
BLOCK
# ============================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TO_TARGET_PRESET=${TO_TARGET_PRESET:-}
if [[ -z "${TO_TARGET_PRESET}" ]]; then
    if [[ -z "${TASK:-}" || "${TASK:-}" == *"ToTarget-V2"* ]]; then
        TO_TARGET_PRESET=v2
    elif [[ "${TASK}" == *"ToTarget-CommandBalanced"* || "${TASK}" == "LeggedLab-Isaac-AMP-G1-ToTarget-v0" ]]; then
        TO_TARGET_PRESET=command_balanced
    else
        TO_TARGET_PRESET=custom
    fi
fi

append_extra_hydra_args() {
    local args_to_append="$1"
    [[ -n "${args_to_append}" ]] || return 0
    if [[ -z "${EXTRA_HYDRA_ARGS:-}" ]]; then
        export EXTRA_HYDRA_ARGS="${args_to_append}"
    else
        export EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS} ${args_to_append}"
    fi
}

case "${TO_TARGET_PRESET}" in
    v2)
        export TASK=${TASK:-LeggedLab-Isaac-AMP-G1-ToTarget-V2-v0}
        export RUN_NAME=${RUN_NAME:-s3_g1_29dof_toTarget_v2_command_balanced_amp_5000}
        export STYLE_REWARD_SCALE=${STYLE_REWARD_SCALE:-10.0}
        export TASK_STYLE_LERP=${TASK_STYLE_LERP:-0.50}
        export RSI_ENABLE=${RSI_ENABLE:-True}
        export RSI_RATIO=${RSI_RATIO:-0.50}
        export RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-0}
        if [[ -z "${TO_TARGET_V2_EXTRA_ARGS+x}" ]]; then
            TO_TARGET_V2_EXTRA_ARGS="agent.policy.init_noise_std=0.70 agent.algorithm.amp_cfg.normalizer_mode=policy_demo"
        fi
        append_extra_hydra_args "${TO_TARGET_V2_EXTRA_ARGS}"
        ;;
    command_balanced|v1|legacy)
        export TASK=${TASK:-LeggedLab-Isaac-AMP-G1-ToTarget-CommandBalanced-v0}
        export RUN_NAME=${RUN_NAME:-s3_g1_29dof_toTarget_command_balanced_amp_5000}
        export STYLE_REWARD_SCALE=${STYLE_REWARD_SCALE:-8.0}
        export TASK_STYLE_LERP=${TASK_STYLE_LERP:-0.55}
        export RSI_ENABLE=${RSI_ENABLE:-False}
        export RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-1}
        ;;
    custom)
        export TASK=${TASK:?TASK must be set when TO_TARGET_PRESET=custom}
        export RUN_NAME=${RUN_NAME:-s3_g1_29dof_toTarget_custom}
        export STYLE_REWARD_SCALE=${STYLE_REWARD_SCALE:-8.0}
        export TASK_STYLE_LERP=${TASK_STYLE_LERP:-0.55}
        export RSI_ENABLE=${RSI_ENABLE:-False}
        export RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-1}
        ;;
    *)
        echo "Error: unknown TO_TARGET_PRESET: ${TO_TARGET_PRESET}" >&2
        echo "Valid values: v2, command_balanced, custom" >&2
        exit 1
        ;;
esac

export NUM_ENVS=${NUM_ENVS:-8192}
export MAX_ITERATIONS=${MAX_ITERATIONS:-5000}
export ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
export AMP_GRAD_PENALTY_SCALE=${AMP_GRAD_PENALTY_SCALE:-10.0}
export QUIET_TERMINAL=${QUIET_TERMINAL:-True}

exec "${SCRIPT_DIR}/train_g1_amp.sh" "$@"
