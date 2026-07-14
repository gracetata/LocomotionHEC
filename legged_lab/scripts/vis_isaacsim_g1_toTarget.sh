#!/usr/bin/env bash
# Visualize the Unitree G1 AMP ToTarget policy in IsaacLab/Isaac Sim.
#
# The ToTarget play task shows non-physical pose arrows at the sampled target
# pose and the current robot base pose. The policy robot resets at the origin
# and moves toward a new target sampled uniformly inside a 1 m disk each episode.
#
# Usage:
#   bash scripts/vis_isaacsim_g1_toTarget.sh
#   CHECKPOINT=logs/rsl_rl/g1_amp/<run>/model_4999.pt bash scripts/vis_isaacsim_g1_toTarget.sh
#   NUM_ENVS=4 TARGET_RADIUS_RANGE='[0.0,1.0]' bash scripts/vis_isaacsim_g1_toTarget.sh
#   TO_TARGET_PRESET=command_balanced CHECKPOINT=logs/rsl_rl/g1_amp/<run>/model_4999.pt bash scripts/vis_isaacsim_g1_toTarget.sh
#
# Environment variables:
#   TO_TARGET_PRESET     : v2/command_balanced/custom (default: v2 unless TASK implies otherwise).
#   TASK                 : Play task id (default: LeggedLab-Isaac-AMP-G1-ToTarget-V2-Play-v0).
#   CHECKPOINT           : Checkpoint path. If unset, the latest ToTarget checkpoint under logs is used.
#   NUM_ENVS             : Number of replay envs (default: 1).
#   TARGET_RADIUS_RANGE  : Hydra list for target radius in meters (default: [0.0,1.0]).
#   TARGET_HEADING_RANGE : Hydra list for target heading delta in radians (default: [-pi,pi]).
#   TARGET_RESAMPLE_TIME : Seconds before command resampling (default: 8.0).
#   TARGET_DEBUG_VIS     : True/False show the target/current pose arrows (default: True).
#   TARGET_ROOT_HEIGHT_OFFSET: Visual-only height offset for the pose arrows (default: 0.0).
#   EXTRA_HYDRA_ARGS     : Extra Hydra overrides appended after these defaults.
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]

# 1. 当前推荐 ToTarget v2 可视化：1m 内随机目标。
CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-15_01-02-37_s3_g1_toTarget_v2_warmstart_stop_amp_5000/model_10097.pt bash scripts/vis_isaacsim_g1_toTarget.sh

# 2. ToTarget v2 近目标 headless 验证。
CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-15_01-02-37_s3_g1_toTarget_v2_warmstart_stop_amp_5000/model_10097.pt HEADLESS=True MAX_STEPS=400 REAL_TIME=False NUM_ENVS=4 SKIP_EXPORT=True TARGET_RADIUS_RANGE='[0.0,0.3]' bash scripts/vis_isaacsim_g1_toTarget.sh

# 3. ToTarget v2 远目标 headless 验证。
CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-15_01-02-37_s3_g1_toTarget_v2_warmstart_stop_amp_5000/model_10097.pt HEADLESS=True MAX_STEPS=400 REAL_TIME=False NUM_ENVS=4 SKIP_EXPORT=True TARGET_RADIUS_RANGE='[0.7,1.0]' bash scripts/vis_isaacsim_g1_toTarget.sh

# 4. 旧 ToTarget command-balanced run 对照。
TO_TARGET_PRESET=command_balanced CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-14_15-09-58_s3_g1_29dof_toTarget_cmu_core_slow_amp_5000/model_4999.pt bash scripts/vis_isaacsim_g1_toTarget.sh
TO_TARGET_PRESET=command_balanced TARGET_RADIUS_RANGE='[0.0,0.3]' CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-14_15-09-58_s3_g1_29dof_toTarget_cmu_core_slow_amp_5000/model_4999.pt bash scripts/vis_isaacsim_g1_toTarget.sh
BLOCK

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

discover_checkpoint() {
    local log_dir="${ROOT_DIR}/logs/rsl_rl/g1_amp"
    [[ -d "${log_dir}" ]] || return 0
    find "${log_dir}" -maxdepth 2 -type f -name 'model_*.pt' \
        | grep -Ei 'to.?target|target' \
        | sort -V \
        | tail -n 1
}

if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT=$(discover_checkpoint || true)
fi
if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "Error: could not find a ToTarget checkpoint. Set CHECKPOINT=logs/rsl_rl/g1_amp/<run>/model_*.pt." >&2
    exit 1
fi

TO_TARGET_PRESET=${TO_TARGET_PRESET:-}
if [[ -z "${TO_TARGET_PRESET}" ]]; then
    if [[ -z "${TASK:-}" || "${TASK:-}" == *"ToTarget-V2"* ]]; then
        TO_TARGET_PRESET=v2
    elif [[ "${TASK}" == *"ToTarget-Play"* ]]; then
        TO_TARGET_PRESET=command_balanced
    else
        TO_TARGET_PRESET=custom
    fi
fi

case "${TO_TARGET_PRESET}" in
    v2)
        export TASK=${TASK:-LeggedLab-Isaac-AMP-G1-ToTarget-V2-Play-v0}
        ;;
    command_balanced|v1|legacy)
        export TASK=${TASK:-LeggedLab-Isaac-AMP-G1-ToTarget-Play-v0}
        ;;
    custom)
        export TASK=${TASK:?TASK must be set when TO_TARGET_PRESET=custom}
        ;;
    *)
        echo "Error: unknown TO_TARGET_PRESET: ${TO_TARGET_PRESET}" >&2
        echo "Valid values: v2, command_balanced, custom" >&2
        exit 1
        ;;
esac

export CHECKPOINT
export NUM_ENVS=${NUM_ENVS:-1}
export ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
export HEADLESS=${HEADLESS:-False}
export REAL_TIME=${REAL_TIME:-True}
export RSI_ENABLE=${RSI_ENABLE:-False}
export RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-0}
export SKIP_EXPORT=${SKIP_EXPORT:-False}

TARGET_RADIUS_RANGE=${TARGET_RADIUS_RANGE:-'[0.0,1.0]'}
TARGET_HEADING_RANGE=${TARGET_HEADING_RANGE:-'[-3.141592653589793,3.141592653589793]'}
TARGET_RESAMPLE_TIME=${TARGET_RESAMPLE_TIME:-8.0}
TARGET_DEBUG_VIS=${TARGET_DEBUG_VIS:-True}
TARGET_ROOT_HEIGHT_OFFSET=${TARGET_ROOT_HEIGHT_OFFSET:-0.0}

TO_TARGET_HYDRA_ARGS=(
    env.commands.base_velocity.debug_vis="${TARGET_DEBUG_VIS}"
    env.commands.base_velocity.target_root_height_offset="${TARGET_ROOT_HEIGHT_OFFSET}"
    env.commands.base_velocity.ranges.radius="${TARGET_RADIUS_RANGE}"
    env.commands.base_velocity.ranges.heading="${TARGET_HEADING_RANGE}"
    env.commands.base_velocity.resampling_time_range="[${TARGET_RESAMPLE_TIME},${TARGET_RESAMPLE_TIME}]"
    'env.events.reset_base.params.pose_range={x:[0.0,0.0],y:[0.0,0.0],yaw:[0.0,0.0]}'
    'env.events.reset_base.params.velocity_range={x:[0.0,0.0],y:[0.0,0.0],z:[0.0,0.0],roll:[0.0,0.0],pitch:[0.0,0.0],yaw:[0.0,0.0]}'
    'env.events.reset_robot_joints.params.position_range=[1.0,1.0]'
    'env.events.reset_robot_joints.params.velocity_range=[0.0,0.0]'
)

USER_EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-}
export EXTRA_HYDRA_ARGS="${TO_TARGET_HYDRA_ARGS[*]} ${USER_EXTRA_HYDRA_ARGS}"

echo "=========================================="
echo "  Visualizing G1 AMP ToTarget in IsaacSim"
echo "=========================================="
echo "Preset            : ${TO_TARGET_PRESET}"
echo "Task              : ${TASK}"
echo "Checkpoint        : ${CHECKPOINT}"
echo "Num Envs          : ${NUM_ENVS}"
echo "Target Radius     : ${TARGET_RADIUS_RANGE}"
echo "Target Heading    : ${TARGET_HEADING_RANGE}"
echo "Target Resampling : ${TARGET_RESAMPLE_TIME}"
echo "Pose Arrows       : ${TARGET_DEBUG_VIS} (green target, blue current)"
echo "Start Pose        : origin, yaw=0, zero velocity"
echo "=========================================="

exec "${SCRIPT_DIR}/vis_isaacsim_g1_amp.sh" "$@"
