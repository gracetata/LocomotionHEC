#!/usr/bin/env bash
# Visualize a Unitree G1 AMP checkpoint in IsaacLab/Isaac Sim and print task/torso metrics.
# Usage:
#   bash scripts/vis_isaacsim_g1_amp.sh
#   CHECKPOINT=logs/rsl_rl/g1_amp/<run>/model_3999.pt bash scripts/vis_isaacsim_g1_amp.sh
#
# Environment variables:
#   ISAACLAB_PYTHON : Python executable from .envrc conda env.
#   CONDA_ENV_NAME  : Conda env name used when ISAACLAB_PYTHON is unset (default: env_isaaclab).
#   CONDA_BASE      : Conda installation root (default: /home/hecggdz/miniconda3).
#   TASK            : Play task id (default: LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0).
#   CHECKPOINT      : Checkpoint path (default: 2026-06-07 local XML run, model_3999.pt).
#   NUM_ENVS        : Number of replay envs (default: 4).
#   DEVICE          : Simulation device (default: cuda:0).
#   HEADLESS        : True/False for Isaac Sim UI (default: False).
#   MAX_STEPS       : Stop after this many policy steps; empty means run until window closes.
#   REAL_TIME       : True/False real-time pacing (default: True).
#   SKIP_EXPORT     : Deprecated compatibility flag; export uses scripts/export_g1_amp_policy.sh.
#   ROBOT_ASSET     : Robot asset preset: s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof, g1_29dof_mjcf (default: s3_g1_29dof).
#   ROBOT_XML       : Local MJCF XML override used by *_mjcf presets.
#   ROBOT_USD_PATH  : Informational path for prebuilt/cached USD presets.
#   ROBOT_USD_DIR   : USD conversion output dir for *_mjcf presets.
#   FORCE_USD_CONVERSION: True/False force MJCF-to-USD conversion after XML edits (default: False).
#   RSI_ENABLE      : True/False enable reset_from_ref RSI while visualizing (default: False).
#   RSI_RATIO       : Fraction of reset envs initialized from reference state when RSI_ENABLE=True (default: 0.5).
#   POS_RSI         : True keeps reference XY position; False aligns to env origin (default: False).
#   RANDOMIZATION_STRENGTH: 0 disables startup/interval randomization for cleaner XML diagnostics (default: 0).
#   RANDOM_COMMANDS : True overrides play command ranges to joystick-like random tasks (default: False).
#   CURVATURE_COMMANDS: Legacy name for segmented-yaw command overrides; switches TASK only if TASK is base G1 Play.
#   SEGMENTED_YAW_VIEW: all/random/walk/slow_walk/normal_walk/lateral/backward.
#                       random keeps speed_xy < 0.4, slow_walk keeps 0.4-0.7,
#                       normal_walk keeps 0.7-1.0, walk keeps 0.4-1.0,
#                       lateral fixes vy=0.3, backward fixes vx=-0.4 (default: all).
#   COMMAND_RESAMPLING_TIME: Seconds between command samples in IsaacSim play (default: 10.0).
#   CMD_LIN_X_RANGE : Hydra list for total lin x envelope, e.g. '[-0.4,1.0]'.
#   CMD_LIN_Y_RANGE : Hydra list for lin y range, e.g. '[-0.3,0.3]'.
#   CMD_YAW_RANGE   : Hydra list for high-speed yaw range, e.g. '[-0.3,0.3]'.
#   EXTRA_HYDRA_ARGS: Extra Hydra overrides for task/command/reward parameters.
#   FOLLOW_CAMERA  : True enables robot-relative play/video camera tracking (default: False).
#   CAMERA_VIEW    : front/chase/side. front keeps camera in front of robot looking back (default: front).
#   CAMERA_DISTANCE: Horizontal camera distance from robot root in meters (default: 3.0).
#   CAMERA_HEIGHT  : Camera eye height above robot root in meters (default: 1.25).
#   CAMERA_TARGET_HEIGHT: Look-at height above robot root in meters (default: 0.85).
#   CAMERA_LATERAL : Lateral camera offset in robot-right meters (default: 0.0).
#   CAMERA_SMOOTHING: Camera low-pass smoothing in [0,0.95] (default: 0.25).
#   CAMERA_ENV_INDEX: Environment index to track when NUM_ENVS > 1 (default: 0).
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 打开 Isaac Sim GUI，用本地 S3/G1 缓存 USD 做初始化检查
bash scripts/vis_isaacsim_g1_amp.sh

# 2. 只跑 200 step 做 headless smoke test
HEADLESS=True MAX_STEPS=200 bash scripts/vis_isaacsim_g1_amp.sh

# 3. 带 RSI 和随机化回放，贴近训练分布
RSI_ENABLE=True RANDOMIZATION_STRENGTH=1 bash scripts/vis_isaacsim_g1_amp.sh

# 4. 随机摇杆任务可视化/评估
RANDOM_COMMANDS=True SKIP_EXPORT=True bash scripts/vis_isaacsim_g1_amp.sh

# 5. 分段 yaw command 任务可视化/评估
CURVATURE_COMMANDS=True SKIP_EXPORT=True bash scripts/vis_isaacsim_g1_amp.sh

# 6. 以 S3 XML 为输入显式重建一次 S3 USD；不改 XML
ROBOT_ASSET=s3_g1_29dof_mjcf FORCE_USD_CONVERSION=True bash scripts/vis_isaacsim_g1_amp.sh

ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-11_17-43-37_orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000/model_3999.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=normal_walk bash scripts/vis_isaacsim_g1_amp.sh
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Strict-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-12_19-45-54_orig_g1_29dof_cmu_walk_washed_strict_upright_resume3999_2000/model_5998.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=normal_walk bash scripts/vis_isaacsim_g1_amp.sh

ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-11_17-43-37_orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000/model_3999.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=slow_walk bash scripts/vis_isaacsim_g1_amp.sh

ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-11_17-43-37_orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000/model_3999.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=normal_walk bash scripts/vis_isaacsim_g1_amp.sh

# 7. 可视化 2026-06-08 S3 segmented-yaw 500-iter finetune checkpoint
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=all bash scripts/vis_isaacsim_g1_amp.sh

# 7a. 只看 random 段：speed_xy < 0.4，低速下 yaw 从 [-0.5, 0.5] 均匀采样
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=random bash scripts/vis_isaacsim_g1_amp.sh

# 7b. 只看 walk 段：0.4 <= speed_xy <= 1.0，高速下 yaw 按 N(0, 0.1) 采样并裁剪到 [-0.3, 0.3]
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=walk bash scripts/vis_isaacsim_g1_amp.sh

# 7c. 只看 slow walk 段：0.4 <= speed_xy < 0.7
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=slow_walk bash scripts/vis_isaacsim_g1_amp.sh

# 7d. 只看 normal walk 段：0.7 <= speed_xy <= 1.0
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=normal_walk bash scripts/vis_isaacsim_g1_amp.sh

# 7e. 只看横移：vx=0.0, vy=0.3, yaw=0.0
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=lateral bash scripts/vis_isaacsim_g1_amp.sh

# 7f. 只看后退：vx=-0.4, vy=0.0, yaw=0.0
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710/model_3498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=backward bash scripts/vis_isaacsim_g1_amp.sh

# 8. Command-balanced directional strict arm-prior policy with fixed mode-balanced command configs.
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500/model_7498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=normal_walk bash scripts/vis_isaacsim_g1_amp.sh
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500/model_7498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=lateral bash scripts/vis_isaacsim_g1_amp.sh
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500/model_7498.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=backward bash scripts/vis_isaacsim_g1_amp.sh

# 8a. V2 reward task after the next finetune; replace run/checkpoint with the latest V2 model.
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V2-Play-v0 CHECKPOINT=logs/rsl_rl/g1_amp/<v2_run>/model_*.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=backward bash scripts/vis_isaacsim_g1_amp.sh

tensorboard --logdir logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710

ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkCore-Adaptive-v0 CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-10_16-14-39_orig_g1_29dof_cmu_walk_core_adaptive_4000/model_3999.pt CURVATURE_COMMANDS=True SEGMENTED_YAW_VIEW=normal_walk bash scripts/vis_isaacsim_g1_amp.sh

tensorboard --logdir logs/rsl_rl/g1_amp/2026-06-08_19-57-17_debug_s3_segmented_yaw_resume2999_aligned_500_20260608_195710

BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_leglab}
CONDA_BASE=${CONDA_BASE:-/home/hecggdz/miniconda3}
if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
        ISAACLAB_PYTHON="${CONDA_PREFIX}/bin/python"
    else
        ISAACLAB_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
    fi
fi

TASK=${TASK:-LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0}
CHECKPOINT=${CHECKPOINT:-${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp/2026-06-07_22-16-58/model_3999.pt}
NUM_ENVS=${NUM_ENVS:-4}
DEVICE=${DEVICE:-cuda:0}
HEADLESS=${HEADLESS:-False}
MAX_STEPS=${MAX_STEPS:-}
REAL_TIME=${REAL_TIME:-True}
SKIP_EXPORT=${SKIP_EXPORT:-False}
ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
ROBOT_XML=${ROBOT_XML:-}
ROBOT_USD_PATH=${ROBOT_USD_PATH:-}
ROBOT_USD_DIR=${ROBOT_USD_DIR:-}
ROBOT_USD_FILE_NAME=${ROBOT_USD_FILE_NAME:-}
FORCE_USD_CONVERSION=${FORCE_USD_CONVERSION:-False}
RSI_ENABLE=${RSI_ENABLE:-False}
RSI_RATIO=${RSI_RATIO:-0.5}
POS_RSI=${POS_RSI:-False}
RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-0}
RANDOM_COMMANDS=${RANDOM_COMMANDS:-False}
CURVATURE_COMMANDS=${CURVATURE_COMMANDS:-False}
SEGMENTED_YAW_VIEW=${SEGMENTED_YAW_VIEW:-all}
COMMAND_RESAMPLING_TIME=${COMMAND_RESAMPLING_TIME:-10.0}
CMD_LIN_X_RANGE=${CMD_LIN_X_RANGE:-'[-0.4,1.0]'}
CMD_LIN_Y_RANGE=${CMD_LIN_Y_RANGE:-'[-0.3,0.3]'}
CMD_YAW_RANGE=${CMD_YAW_RANGE:-'[-0.3,0.3]'}
CMD_LOW_SPEED_YAW_RANGE=${CMD_LOW_SPEED_YAW_RANGE:-'[-0.50,0.50]'}
CMD_LOW_SPEED_THRESHOLD=${CMD_LOW_SPEED_THRESHOLD:-0.40}
CMD_HIGH_SPEED_YAW_MEAN=${CMD_HIGH_SPEED_YAW_MEAN:-0.0}
CMD_HIGH_SPEED_YAW_STD=${CMD_HIGH_SPEED_YAW_STD:-0.10}
CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-}
SEGMENTED_YAW_SPEED_BAND="cfg"
EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-}
FOLLOW_CAMERA=${FOLLOW_CAMERA:-False}
CAMERA_VIEW=${CAMERA_VIEW:-front}
CAMERA_DISTANCE=${CAMERA_DISTANCE:-3.0}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-1.25}
CAMERA_TARGET_HEIGHT=${CAMERA_TARGET_HEIGHT:-0.85}
CAMERA_LATERAL=${CAMERA_LATERAL:-0.0}
CAMERA_SMOOTHING=${CAMERA_SMOOTHING:-0.25}
CAMERA_ENV_INDEX=${CAMERA_ENV_INDEX:-0}

SEGMENTED_YAW_VIEW=$(printf "%s" "${SEGMENTED_YAW_VIEW}" | tr '[:upper:]' '[:lower:]')
case "${SEGMENTED_YAW_VIEW}" in
    all|full|default)
        SEGMENTED_YAW_VIEW="all"
        ;;
    random|low|low_speed)
        SEGMENTED_YAW_VIEW="random"
        SEGMENTED_YAW_SPEED_BAND="<0.4"
        CMD_LIN_X_RANGE=${SEGMENTED_RANDOM_LIN_X_RANGE:-'[-0.28,0.28]'}
        CMD_LIN_Y_RANGE=${SEGMENTED_RANDOM_LIN_Y_RANGE:-'[-0.28,0.28]'}
        CMD_LOW_SPEED_YAW_RANGE=${SEGMENTED_RANDOM_YAW_RANGE:-'[-0.50,0.50]'}
        CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-0.0}
        ;;
    walk|high|high_speed)
        SEGMENTED_YAW_VIEW="walk"
        SEGMENTED_YAW_SPEED_BAND="0.4-1.0"
        CMD_LIN_X_RANGE=${SEGMENTED_WALK_LIN_X_RANGE:-'[0.41,0.95]'}
        CMD_LIN_Y_RANGE=${SEGMENTED_WALK_LIN_Y_RANGE:-'[-0.30,0.30]'}
        CMD_YAW_RANGE=${SEGMENTED_WALK_YAW_RANGE:-'[-0.30,0.30]'}
        CMD_HIGH_SPEED_YAW_MEAN=${SEGMENTED_WALK_YAW_MEAN:-0.0}
        CMD_HIGH_SPEED_YAW_STD=${SEGMENTED_WALK_YAW_STD:-0.10}
        CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-0.0}
        ;;
    slow_walk|slow|slowwalk)
        SEGMENTED_YAW_VIEW="slow_walk"
        SEGMENTED_YAW_SPEED_BAND="0.4-0.7"
        CMD_LIN_X_RANGE=${SEGMENTED_SLOW_WALK_LIN_X_RANGE:-'[0.41,0.63]'}
        CMD_LIN_Y_RANGE=${SEGMENTED_SLOW_WALK_LIN_Y_RANGE:-'[-0.30,0.30]'}
        CMD_YAW_RANGE=${SEGMENTED_SLOW_WALK_YAW_RANGE:-'[-0.30,0.30]'}
        CMD_HIGH_SPEED_YAW_MEAN=${SEGMENTED_SLOW_WALK_YAW_MEAN:-0.0}
        CMD_HIGH_SPEED_YAW_STD=${SEGMENTED_SLOW_WALK_YAW_STD:-0.10}
        CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-0.0}
        ;;
    normal_walk|normal|normalwalk)
        SEGMENTED_YAW_VIEW="normal_walk"
        SEGMENTED_YAW_SPEED_BAND="0.7-1.0"
        CMD_LIN_X_RANGE=${SEGMENTED_NORMAL_WALK_LIN_X_RANGE:-'[0.70,0.95]'}
        CMD_LIN_Y_RANGE=${SEGMENTED_NORMAL_WALK_LIN_Y_RANGE:-'[-0.30,0.30]'}
        CMD_YAW_RANGE=${SEGMENTED_NORMAL_WALK_YAW_RANGE:-'[-0.30,0.30]'}
        CMD_HIGH_SPEED_YAW_MEAN=${SEGMENTED_NORMAL_WALK_YAW_MEAN:-0.0}
        CMD_HIGH_SPEED_YAW_STD=${SEGMENTED_NORMAL_WALK_YAW_STD:-0.10}
        CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-0.0}
        ;;
    lateral|side|strafe)
        SEGMENTED_YAW_VIEW="lateral"
        SEGMENTED_YAW_SPEED_BAND="vy=0.3"
        CMD_LIN_X_RANGE=${SEGMENTED_LATERAL_LIN_X_RANGE:-'[0.0,0.0]'}
        CMD_LIN_Y_RANGE=${SEGMENTED_LATERAL_LIN_Y_RANGE:-'[0.30,0.30]'}
        CMD_LOW_SPEED_YAW_RANGE=${SEGMENTED_LATERAL_LOW_SPEED_YAW_RANGE:-'[0.0,0.0]'}
        CMD_YAW_RANGE=${SEGMENTED_LATERAL_YAW_RANGE:-'[0.0,0.0]'}
        CMD_HIGH_SPEED_YAW_MEAN=${SEGMENTED_LATERAL_YAW_MEAN:-0.0}
        CMD_HIGH_SPEED_YAW_STD=${SEGMENTED_LATERAL_YAW_STD:-0.0}
        CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-0.0}
        ;;
    backward|back|reverse)
        SEGMENTED_YAW_VIEW="backward"
        SEGMENTED_YAW_SPEED_BAND="vx=-0.4"
        CMD_LIN_X_RANGE=${SEGMENTED_BACKWARD_LIN_X_RANGE:-'[-0.40,-0.40]'}
        CMD_LIN_Y_RANGE=${SEGMENTED_BACKWARD_LIN_Y_RANGE:-'[0.0,0.0]'}
        CMD_LOW_SPEED_YAW_RANGE=${SEGMENTED_BACKWARD_LOW_SPEED_YAW_RANGE:-'[0.0,0.0]'}
        CMD_YAW_RANGE=${SEGMENTED_BACKWARD_YAW_RANGE:-'[0.0,0.0]'}
        CMD_HIGH_SPEED_YAW_MEAN=${SEGMENTED_BACKWARD_YAW_MEAN:-0.0}
        CMD_HIGH_SPEED_YAW_STD=${SEGMENTED_BACKWARD_YAW_STD:-0.0}
        CMD_REL_STANDING_ENVS=${CMD_REL_STANDING_ENVS:-0.0}
        ;;
    *)
        echo "Error: unknown SEGMENTED_YAW_VIEW: ${SEGMENTED_YAW_VIEW}" >&2
        echo "Valid values: all, random, walk, slow_walk, normal_walk, lateral, backward" >&2
        exit 1
        ;;
esac

if [[ "${CURVATURE_COMMANDS}" == "True" || "${CURVATURE_COMMANDS}" == "true" || "${CURVATURE_COMMANDS}" == "1" ]]; then
    if [[ "${TASK}" == "LeggedLab-Isaac-AMP-G1-Play-v0" ]]; then
        TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-Play-v0
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

case "${ROBOT_ASSET}" in
    s3_g1_29dof|s3)
        ROBOT_ASSET="s3_g1_29dof"
        ROBOT_USD_PATH=${ROBOT_USD_PATH:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/usd/s3_g1_29dof.usd}
        ;;
    s3_g1_29dof_mjcf|s3_mjcf)
        ROBOT_ASSET="s3_g1_29dof_mjcf"
        ROBOT_XML=${ROBOT_XML:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/g1_29dof.xml}
        ROBOT_USD_DIR=${ROBOT_USD_DIR:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/usd}
        ROBOT_USD_FILE_NAME=${ROBOT_USD_FILE_NAME:-s3_g1_29dof.usd}
        ;;
    g1_29dof|g1|original_g1)
        ROBOT_ASSET="g1_29dof"
        ROBOT_USD_PATH=${ROBOT_USD_PATH:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd}
        ;;
    g1_29dof_mjcf|g1_mjcf)
        ROBOT_ASSET="g1_29dof_mjcf"
        ROBOT_XML=${ROBOT_XML:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/g1_29dof.xml}
        ROBOT_USD_DIR=${ROBOT_USD_DIR:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_mjcf}
        ROBOT_USD_FILE_NAME=${ROBOT_USD_FILE_NAME:-g1_29dof_mjcf.usd}
        ;;
    *)
        echo "Error: unknown ROBOT_ASSET preset: ${ROBOT_ASSET}" >&2
        echo "Valid values: s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof, g1_29dof_mjcf" >&2
        exit 1
        ;;
esac

if [[ -n "${ROBOT_USD_PATH}" && "${ROBOT_USD_PATH}" != /* ]]; then
    ROBOT_USD_PATH="${ROOT_DIR}/${ROBOT_USD_PATH}"
fi
if [[ -n "${ROBOT_USD_PATH}" && ! -f "${ROBOT_USD_PATH}" ]]; then
    echo "Error: ROBOT_USD_PATH does not exist: ${ROBOT_USD_PATH}" >&2
    exit 1
fi
if [[ -n "${ROBOT_XML}" && "${ROBOT_XML}" != /* ]]; then
    ROBOT_XML="${ROOT_DIR}/${ROBOT_XML}"
fi
if [[ -n "${ROBOT_XML}" && ! -f "${ROBOT_XML}" ]]; then
    echo "Error: ROBOT_XML does not exist: ${ROBOT_XML}" >&2
    exit 1
fi
if [[ -n "${ROBOT_USD_DIR}" && "${ROBOT_USD_DIR}" != /* ]]; then
    ROBOT_USD_DIR="${ROOT_DIR}/${ROBOT_USD_DIR}"
fi
if [[ -n "${ROBOT_XML}" && ( -z "${ROBOT_USD_DIR}" || -z "${ROBOT_USD_FILE_NAME}" ) ]]; then
    echo "Error: ROBOT_USD_DIR and ROBOT_USD_FILE_NAME must be set when ROBOT_XML is set." >&2
    exit 1
fi

ISAACLAB_CONDA_PREFIX=$(cd "$(dirname "${ISAACLAB_PYTHON}")/.." && pwd)
export LD_LIBRARY_PATH="${ISAACLAB_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${LEGGED_LAB_DIR}/source/legged_lab${PYTHONPATH:+:${PYTHONPATH}}"
export LEGGED_LAB_G1_AMP_ROBOT_ASSET="${ROBOT_ASSET}"
export OMNI_LOG_DEFAULT_LEVEL=${OMNI_LOG_DEFAULT_LEVEL:-error}
export OMNI_KIT_QUIET=${OMNI_KIT_QUIET:-1}

COMMAND_BALANCED_SAMPLING_CONFIG=""
if [[ "${TASK}" == *"CommandBalancedDirectional"* && ( "${CURVATURE_COMMANDS}" == "True" || "${CURVATURE_COMMANDS}" == "true" || "${CURVATURE_COMMANDS}" == "1" ) ]]; then
    parse_range() {
        local raw="$1"
        raw="${raw#[}"
        raw="${raw%]}"
        raw="${raw// /}"
        local low high
        IFS=',' read -r low high <<<"${raw}"
        printf "%s %s" "${low}" "${high}"
    }
    read -r cmd_x_low cmd_x_high < <(parse_range "${CMD_LIN_X_RANGE}")
    read -r cmd_y_low cmd_y_high < <(parse_range "${CMD_LIN_Y_RANGE}")
    read -r cmd_w_low cmd_w_high < <(parse_range "${CMD_YAW_RANGE}")
    mode_name="vis_${SEGMENTED_YAW_VIEW}"
    COMMAND_BALANCED_SAMPLING_CONFIG="${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp/vis_${SEGMENTED_YAW_VIEW}_fixed_sampling_config.json"
    mkdir -p "$(dirname "${COMMAND_BALANCED_SAMPLING_CONFIG}")"
    printf '{\n' >"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '  "command_frame": "robot_base_local",\n' >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '  "fps": 50,\n' >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '  "mode_weights": {"%s": 1.0},\n' "${mode_name}" >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '  "mode_name_by_id": {"0": "%s"},\n' "${mode_name}" >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '  "modes": {"%s": {"lin_vel_x": [%s, %s], "lin_vel_y": [%s, %s], "ang_vel_z": [%s, %s]}},\n' \
        "${mode_name}" "${cmd_x_low}" "${cmd_x_high}" "${cmd_y_low}" "${cmd_y_high}" "${cmd_w_low}" "${cmd_w_high}" \
        >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '  "source_dataset": "vis_fixed_command"\n' >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
    printf '}\n' >>"${COMMAND_BALANCED_SAMPLING_CONFIG}"
fi

args=(
    scripts/rsl_rl/play.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    agent.device="${DEVICE}"
)

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
if [[ "${FOLLOW_CAMERA}" == "True" || "${FOLLOW_CAMERA}" == "true" || "${FOLLOW_CAMERA}" == "1" ]]; then
    args+=(--follow_camera)
    args+=(--camera_view "${CAMERA_VIEW}")
    args+=(--camera_distance "${CAMERA_DISTANCE}")
    args+=(--camera_height "${CAMERA_HEIGHT}")
    args+=(--camera_target_height "${CAMERA_TARGET_HEIGHT}")
    args+=(--camera_lateral "${CAMERA_LATERAL}")
    args+=(--camera_smoothing "${CAMERA_SMOOTHING}")
    args+=(--camera_env_index "${CAMERA_ENV_INDEX}")
fi
if [[ -n "${ROBOT_XML}" ]]; then
    args+=(env.scene.robot.spawn.asset_path="${ROBOT_XML}")
    args+=(env.scene.robot.spawn.usd_dir="${ROBOT_USD_DIR}")
    args+=(env.scene.robot.spawn.usd_file_name="${ROBOT_USD_FILE_NAME}")
    args+=(env.scene.robot.spawn.force_usd_conversion="${FORCE_USD_CONVERSION}")
fi
if [[ "${RSI_ENABLE}" == "False" || "${RSI_ENABLE}" == "false" || "${RSI_ENABLE}" == "0" ]]; then
    args+=(env.events.reset_from_ref=null)
else
    args+=(env.events.reset_from_ref.params.rsi_ratio="${RSI_RATIO}")
    args+=(env.events.reset_from_ref.params.pos_rsi="${POS_RSI}")
fi
if [[ "${RANDOMIZATION_STRENGTH}" == "0" || "${RANDOMIZATION_STRENGTH}" == "0.0" ]]; then
    args+=(env.events.physics_material=null)
    args+=(env.events.add_base_mass=null)
    args+=(env.events.randomize_rigid_body_com=null)
    args+=(env.events.scale_link_mass=null)
    args+=(env.events.scale_actuator_gains=null)
    args+=(env.events.scale_joint_parameters=null)
    args+=(env.events.push_robot=null)
fi
if [[ "${CURVATURE_COMMANDS}" == "True" || "${CURVATURE_COMMANDS}" == "true" || "${CURVATURE_COMMANDS}" == "1" ]]; then
    args+=(env.commands.base_velocity.heading_command=False)
    args+=(env.commands.base_velocity.rel_heading_envs=1.0)
    args+=(env.commands.base_velocity.heading_control_stiffness=0.5)
    args+=(env.commands.base_velocity.ranges.lin_vel_x="${CMD_LIN_X_RANGE}")
    args+=(env.commands.base_velocity.ranges.lin_vel_y="${CMD_LIN_Y_RANGE}")
    args+=(env.commands.base_velocity.ranges.ang_vel_z="${CMD_YAW_RANGE}")
    args+=(env.commands.base_velocity.resampling_time_range="[${COMMAND_RESAMPLING_TIME},${COMMAND_RESAMPLING_TIME}]")
    if [[ -n "${COMMAND_BALANCED_SAMPLING_CONFIG}" ]]; then
        args+=(env.commands.base_velocity.sampling_config_path="${COMMAND_BALANCED_SAMPLING_CONFIG}")
    else
        args+=(env.commands.base_velocity.low_speed_threshold="${CMD_LOW_SPEED_THRESHOLD}")
        args+=(env.commands.base_velocity.high_speed_ang_vel_z_mean="${CMD_HIGH_SPEED_YAW_MEAN}")
        args+=(env.commands.base_velocity.high_speed_ang_vel_z_std="${CMD_HIGH_SPEED_YAW_STD}")
        args+=(env.commands.base_velocity.ranges.low_speed_ang_vel_z="${CMD_LOW_SPEED_YAW_RANGE}")
    fi
    if [[ -n "${CMD_REL_STANDING_ENVS}" ]]; then
        args+=(env.commands.base_velocity.rel_standing_envs="${CMD_REL_STANDING_ENVS}")
    fi
elif [[ "${RANDOM_COMMANDS}" == "True" || "${RANDOM_COMMANDS}" == "true" || "${RANDOM_COMMANDS}" == "1" ]]; then
    args+=(env.commands.base_velocity.heading_command=False)
    args+=(env.commands.base_velocity.rel_heading_envs=0.0)
    args+=(env.commands.base_velocity.ranges.lin_vel_x="${CMD_LIN_X_RANGE}")
    args+=(env.commands.base_velocity.ranges.lin_vel_y="${CMD_LIN_Y_RANGE}")
    args+=(env.commands.base_velocity.ranges.ang_vel_z="${CMD_YAW_RANGE}")
    args+=(env.commands.base_velocity.resampling_time_range="[${COMMAND_RESAMPLING_TIME},${COMMAND_RESAMPLING_TIME}]")
fi

args+=("$@")

echo "====================================="
echo "  Visualizing G1 AMP Policy in IsaacSim"
echo "====================================="
echo "Task             : ${TASK}"
echo "Checkpoint       : ${CHECKPOINT}"
echo "Robot Asset      : ${ROBOT_ASSET}"
echo "Robot USD Path   : ${ROBOT_USD_PATH}"
echo "Robot XML        : ${ROBOT_XML}"
echo "Robot USD Dir    : ${ROBOT_USD_DIR}"
echo "Force USD Conv   : ${FORCE_USD_CONVERSION}"
echo "Num Envs         : ${NUM_ENVS}"
echo "Device           : ${DEVICE}"
echo "Headless         : ${HEADLESS}"
echo "Skip Export      : ${SKIP_EXPORT}"
echo "RSI Enable/Ratio : ${RSI_ENABLE}/${RSI_RATIO}"
echo "Randomization    : ${RANDOMIZATION_STRENGTH}"
echo "Random Commands  : ${RANDOM_COMMANDS}"
echo "Segmented Yaw    : ${CURVATURE_COMMANDS} view=${SEGMENTED_YAW_VIEW} speed=${SEGMENTED_YAW_SPEED_BAND} threshold=${CMD_LOW_SPEED_THRESHOLD} standing=${CMD_REL_STANDING_ENVS:-cfg} low_yaw=${CMD_LOW_SPEED_YAW_RANGE} high_yaw=${CMD_YAW_RANGE} high_std=${CMD_HIGH_SPEED_YAW_STD}"
echo "Command Ranges   : x=${CMD_LIN_X_RANGE} y=${CMD_LIN_Y_RANGE} yaw=${CMD_YAW_RANGE}"
echo "Mode Config      : ${COMMAND_BALANCED_SAMPLING_CONFIG:-cfg}"
echo "Follow Camera   : ${FOLLOW_CAMERA} view=${CAMERA_VIEW} distance=${CAMERA_DISTANCE} height=${CAMERA_HEIGHT} target_height=${CAMERA_TARGET_HEIGHT} lateral=${CAMERA_LATERAL} smoothing=${CAMERA_SMOOTHING} env=${CAMERA_ENV_INDEX}"
echo "Extra Hydra Args : ${EXTRA_HYDRA_ARGS}"
echo "====================================="

cd "${LEGGED_LAB_DIR}"
"${ISAACLAB_PYTHON}" "${args[@]}" ${EXTRA_HYDRA_ARGS}
