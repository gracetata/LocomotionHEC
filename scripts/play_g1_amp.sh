#!/usr/bin/env bash
# Replay a Unitree G1 AMP checkpoint in IsaacLab and print task/torso metrics.
# Usage:
#   CHECKPOINT=legged_lab/logs/rsl_rl/g1_amp/<run>/model_2999.pt bash scripts/play_g1_amp.sh
#
# Environment variables:
#   ISAACLAB_PYTHON : Python executable from legged_lab/.envrc conda env.
#   TASK            : Play task id (default: LeggedLab-Isaac-AMP-G1-Play-v0).
#   CHECKPOINT      : Checkpoint path (default: successful 3000-iteration G1 AMP baseline).
#   NUM_ENVS        : Number of replay envs (default: 4).
#   DEVICE          : Simulation device (default: cuda:0).
#   HEADLESS        : True/False for Isaac Sim UI (default: True).
#   MAX_STEPS       : Stop after this many policy steps; empty means run until window closes.
#   REAL_TIME       : True/False real-time pacing (default: False).
#   SKIP_EXPORT     : Deprecated compatibility flag; export uses scripts/export_g1_amp_policy.sh.
#   RANDOM_COMMANDS : True overrides play command ranges to joystick-like random tasks (default: False).
#   CURVATURE_COMMANDS: True switches default TASK to curvature Play and applies curvature command overrides.
#   COMMAND_RESAMPLING_TIME: Seconds between command samples in IsaacSim play (default: 2.0).
#   CMD_LIN_X_RANGE : Hydra list for total lin x envelope, e.g. '[-0.2,1.5]'.
#   CMD_LIN_Y_RANGE : Hydra list for lin y range, e.g. '[-0.25,0.25]'.
#   CMD_YAW_RANGE   : Hydra list for yaw range, e.g. '[-0.6,0.6]'.
#   CMD_CURVATURE_RANGE / CMD_MAX_CURVATURE: Curvature command range and cap for curvature Play.
#   EXTRA_HYDRA_ARGS: Extra Hydra overrides for task/command/reward parameters.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 回放已跑通策略并打印 task tracking + torso Important Metrics
bash scripts/play_g1_amp.sh

# 2. 打开 Isaac Sim GUI 回放
HEADLESS=False NUM_ENVS=4 REAL_TIME=True bash scripts/play_g1_amp.sh

# 3. 只跑 200 step 做 smoke test
MAX_STEPS=200 SKIP_EXPORT=True bash scripts/play_g1_amp.sh

# 4. 随机摇杆任务可视化/评估
HEADLESS=False REAL_TIME=True RANDOM_COMMANDS=True SKIP_EXPORT=True bash scripts/play_g1_amp.sh

# 5. 曲率约束 command 任务可视化/评估
HEADLESS=False REAL_TIME=True CURVATURE_COMMANDS=True SKIP_EXPORT=True bash scripts/play_g1_amp.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}/legged_lab"

ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-Play-v0}
CHECKPOINT=${CHECKPOINT:-${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp/2026-05-20_23-50-16_baseline_3000_accad50hz_rsi_20260520/model_2999.pt}
NUM_ENVS=${NUM_ENVS:-4}
DEVICE=${DEVICE:-cuda:0}
HEADLESS=${HEADLESS:-True}
MAX_STEPS=${MAX_STEPS:-}
REAL_TIME=${REAL_TIME:-False}
SKIP_EXPORT=${SKIP_EXPORT:-False}
RANDOM_COMMANDS=${RANDOM_COMMANDS:-False}
CURVATURE_COMMANDS=${CURVATURE_COMMANDS:-False}
COMMAND_RESAMPLING_TIME=${COMMAND_RESAMPLING_TIME:-2.0}
CMD_LIN_X_RANGE=${CMD_LIN_X_RANGE:-'[-0.2,1.5]'}
CMD_LIN_Y_RANGE=${CMD_LIN_Y_RANGE:-'[-0.25,0.25]'}
CMD_YAW_RANGE=${CMD_YAW_RANGE:-'[-0.6,0.6]'}
CMD_CURVATURE_RANGE=${CMD_CURVATURE_RANGE:-'[-0.7,0.7]'}
CMD_MAX_CURVATURE=${CMD_MAX_CURVATURE:-0.7}
CMD_LOW_SPEED_LIN_X_RANGE=${CMD_LOW_SPEED_LIN_X_RANGE:-'[-0.20,0.35]'}
CMD_LOW_SPEED_LIN_Y_RANGE=${CMD_LOW_SPEED_LIN_Y_RANGE:-'[-0.25,0.25]'}
CMD_LOW_SPEED_YAW_RANGE=${CMD_LOW_SPEED_YAW_RANGE:-'[-0.50,0.50]'}
CMD_YAW_NOISE_RANGE=${CMD_YAW_NOISE_RANGE:-'[-0.05,0.05]'}
CMD_REL_LOW_SPEED_ENVS=${CMD_REL_LOW_SPEED_ENVS:-0.25}
CMD_HIGH_SPEED_LATERAL_VEL=${CMD_HIGH_SPEED_LATERAL_VEL:-0.06}
CMD_SMOOTHING_TAU=${CMD_SMOOTHING_TAU:-0.30}
CMD_MAX_LINEAR_ACCEL=${CMD_MAX_LINEAR_ACCEL:-0.80}
CMD_MAX_YAW_ACCEL=${CMD_MAX_YAW_ACCEL:-1.00}
EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-}

if [[ "${CURVATURE_COMMANDS}" == "True" || "${CURVATURE_COMMANDS}" == "true" || "${CURVATURE_COMMANDS}" == "1" ]]; then
    if [[ "${TASK}" == "LeggedLab-Isaac-AMP-G1-Play-v0" ]]; then
        TASK=LeggedLab-Isaac-AMP-G1-CurvatureFinetune-Play-v0
    fi
fi

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${ROOT_DIR}/${CHECKPOINT}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/env_isaaclab/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${LEGGED_LAB_DIR}/source/legged_lab${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_LOG_DEFAULT_LEVEL=${OMNI_LOG_DEFAULT_LEVEL:-error}
export OMNI_KIT_QUIET=${OMNI_KIT_QUIET:-1}

args=(
    scripts/rsl_rl/play.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    agent.device="${DEVICE}"
)

args+=("$@")

if [[ "${HEADLESS}" == "True" || "${HEADLESS}" == "true" || "${HEADLESS}" == "1" ]]; then
    export HEADLESS=1
    args+=(--headless)
else
    export HEADLESS=0
fi
if [[ -n "${MAX_STEPS}" ]]; then
    args+=(--max_steps "${MAX_STEPS}")
fi
if [[ "${REAL_TIME}" == "True" || "${REAL_TIME}" == "true" || "${REAL_TIME}" == "1" ]]; then
    args+=(--real-time)
fi
if [[ "${SKIP_EXPORT}" == "True" || "${SKIP_EXPORT}" == "true" || "${SKIP_EXPORT}" == "1" ]]; then
    args+=(--skip_export)
fi
if [[ "${CURVATURE_COMMANDS}" == "True" || "${CURVATURE_COMMANDS}" == "true" || "${CURVATURE_COMMANDS}" == "1" ]]; then
    args+=(env.commands.base_velocity.heading_command=False)
    args+=(env.commands.base_velocity.rel_heading_envs=0.0)
    args+=(env.commands.base_velocity.ranges.lin_vel_x="${CMD_LIN_X_RANGE}")
    args+=(env.commands.base_velocity.ranges.lin_vel_y="${CMD_LIN_Y_RANGE}")
    args+=(env.commands.base_velocity.ranges.ang_vel_z="${CMD_YAW_RANGE}")
    args+=(env.commands.base_velocity.ranges.curvature="${CMD_CURVATURE_RANGE}")
    args+=(env.commands.base_velocity.ranges.low_speed_lin_vel_x="${CMD_LOW_SPEED_LIN_X_RANGE}")
    args+=(env.commands.base_velocity.ranges.low_speed_lin_vel_y="${CMD_LOW_SPEED_LIN_Y_RANGE}")
    args+=(env.commands.base_velocity.ranges.low_speed_ang_vel_z="${CMD_LOW_SPEED_YAW_RANGE}")
    args+=(env.commands.base_velocity.ranges.yaw_noise="${CMD_YAW_NOISE_RANGE}")
    args+=(env.commands.base_velocity.rel_low_speed_envs="${CMD_REL_LOW_SPEED_ENVS}")
    args+=(env.commands.base_velocity.high_speed_lateral_velocity="${CMD_HIGH_SPEED_LATERAL_VEL}")
    args+=(env.commands.base_velocity.max_curvature="${CMD_MAX_CURVATURE}")
    args+=(env.commands.base_velocity.smoothing_time_constant="${CMD_SMOOTHING_TAU}")
    args+=(env.commands.base_velocity.max_linear_accel="${CMD_MAX_LINEAR_ACCEL}")
    args+=(env.commands.base_velocity.max_yaw_accel="${CMD_MAX_YAW_ACCEL}")
    args+=(env.commands.base_velocity.resampling_time_range="[${COMMAND_RESAMPLING_TIME},${COMMAND_RESAMPLING_TIME}]")
elif [[ "${RANDOM_COMMANDS}" == "True" || "${RANDOM_COMMANDS}" == "true" || "${RANDOM_COMMANDS}" == "1" ]]; then
    args+=(env.commands.base_velocity.heading_command=False)
    args+=(env.commands.base_velocity.rel_heading_envs=0.0)
    args+=(env.commands.base_velocity.ranges.lin_vel_x="${CMD_LIN_X_RANGE}")
    args+=(env.commands.base_velocity.ranges.lin_vel_y="${CMD_LIN_Y_RANGE}")
    args+=(env.commands.base_velocity.ranges.ang_vel_z="${CMD_YAW_RANGE}")
    args+=(env.commands.base_velocity.resampling_time_range="[${COMMAND_RESAMPLING_TIME},${COMMAND_RESAMPLING_TIME}]")
fi

echo "====================================="
echo "  Playing G1 AMP Policy"
echo "====================================="
echo "Task             : ${TASK}"
echo "Checkpoint       : ${CHECKPOINT}"
echo "Num Envs         : ${NUM_ENVS}"
echo "Device           : ${DEVICE}"
echo "Headless         : ${HEADLESS}"
echo "Skip Export      : ${SKIP_EXPORT}"
echo "Random Commands  : ${RANDOM_COMMANDS}"
echo "Curvature Cmds   : ${CURVATURE_COMMANDS} kappa=${CMD_CURVATURE_RANGE} max=${CMD_MAX_CURVATURE}"
echo "Command Ranges   : x=${CMD_LIN_X_RANGE} y=${CMD_LIN_Y_RANGE} yaw=${CMD_YAW_RANGE}"
echo "Extra Hydra Args : ${EXTRA_HYDRA_ARGS}"
echo "====================================="

cd "${LEGGED_LAB_DIR}"
"${ISAACLAB_PYTHON}" "${args[@]}" ${EXTRA_HYDRA_ARGS}
