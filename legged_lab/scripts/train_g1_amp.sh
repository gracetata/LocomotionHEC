#!/usr/bin/env bash
# Train Unitree G1 29DoF AMP segmented-yaw fine-tune task in IsaacLab/RSL-RL.
# Usage:
#   bash scripts/train_g1_amp.sh
#   NUM_ENVS=4096 MAX_ITERATIONS=3000 bash scripts/train_g1_amp.sh
#
# Environment variables:
#   ISAACLAB_PYTHON      : Python executable from .envrc conda env.
#   CONDA_ENV_NAME       : Conda env name used when ISAACLAB_PYTHON is unset (default: env_isaaclab).
#   CONDA_BASE           : Conda installation root (default: ${HOME}/anaconda3).
#   TASK                 : Gym task id (default: LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-v0).
#   NUM_ENVS             : Number of IsaacLab parallel envs (default: 8192).
#   MAX_ITERATIONS       : PPOAMP training iterations (default: 4000).
#   DEVICE               : IsaacLab simulation device (default: cuda:0).
#   AGENT_DEVICE         : RSL-RL device override (default: same as DEVICE).
#   SEED                 : Random seed (default: 42).
#   RUN_NAME             : Optional run-name suffix.
#   RESUME               : True/False resume from an existing checkpoint (default: False).
#   LOAD_RUN             : Existing run directory selector used with RESUME=True.
#   CHECKPOINT           : Checkpoint file selector used with RESUME=True.
#   HEADLESS             : True/False for Isaac Sim UI (default: True).
#   RSI_ENABLE           : True/False enable reset_from_ref RSI (default: True).
#   RSI_RATIO            : Fraction of reset envs initialized from reference state (default: 0.5).
#   POS_RSI              : True keeps reference XY position; False aligns to env origin (default: False).
#   ROBOT_ASSET          : Robot asset preset: config, g1_29dof, s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof_mjcf (default: s3_g1_29dof).
#   ROBOT_XML            : Optional MJCF XML override used by *_mjcf presets.
#   ROBOT_USD_PATH       : Informational path for prebuilt USD presets.
#   ROBOT_USD_DIR        : Optional USD conversion output dir for ROBOT_XML.
#   ROBOT_USD_FILE_NAME  : Optional USD file name for ROBOT_XML.
#   ROBOT_FORCE_USD_CONVERSION: True/False force MJCF-to-USD conversion after XML edits (default: False).
#   RANDOMIZATION_STRENGTH: 0 disables startup/interval randomization; nonzero keeps config defaults (default: 1).
#   TRACK_LIN_WEIGHT     : Optional override for XY command tracking weight.
#   TRACK_ANG_WEIGHT     : Optional override for yaw command tracking weight.
#   TRACK_LIN_STD        : Optional override for XY command tracking std.
#   TRACK_ANG_STD        : Optional override for yaw command tracking std.
#   STYLE_REWARD_SCALE   : AMP discriminator style reward scale (default: 5.0, strict tasks: 8.0).
#   TASK_STYLE_LERP      : Task/style interpolation in PPOAMP (default: 0.4, strict tasks: 0.50).
#   ENTROPY_COEF         : Optional PPO entropy coefficient override (ToTarget v2 default: 0.002).
#   AMP_GRAD_PENALTY_SCALE: AMP discriminator gradient penalty scale (default: 10.0).
#   BASELINE_KL_ENABLE   : True/False enable frozen baseline policy KL anchor (default: False).
#   BASELINE_KL_CHECKPOINT: Checkpoint path used as frozen baseline policy.
#   BASELINE_KL_SCALE    : KL loss multiplier when BASELINE_KL_ENABLE=True (default: 0.0).
#   BASELINE_KL_MIN_STD  : Minimum std for analytic Gaussian KL (default: 1e-4).
#   QUIET_TERMINAL       : True redirects verbose training stdout/stderr to TRAIN_LOG_FILE (default: True).
#   TRAIN_LOG_FILE       : File used when QUIET_TERMINAL=True.
#   EXTRA_HYDRA_ARGS     : Additional Hydra overrides, e.g. 'env.rewards.action_rate_l2.weight=-0.01'.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 默认 S3 G1 29DoF segmented-yaw finetune：AMP + RSI + GMR ACCAD g1used 50Hz motion，4000 iterations
bash scripts/train_g1_amp.sh

# 2. 快速 smoke run
NUM_ENVS=128 MAX_ITERATIONS=2 bash scripts/train_g1_amp.sh

# 3. 关闭 RSI 观察 baseline 差异
RSI_ENABLE=False RUN_NAME=no_rsi bash scripts/train_g1_amp.sh

# 4. 关闭 domain randomization 只观察 AMP imitation
RANDOMIZATION_STRENGTH=0 RUN_NAME=no_rand bash scripts/train_g1_amp.sh

# 5. 默认 S3 segmented-yaw 长训练时降低终端输出，避免每个 iteration 唤醒交互端
QUIET_TERMINAL=True RUN_NAME=s3_segmented_yaw_4000_accad50hz_rsi bash scripts/train_g1_amp.sh

# 6. 复现 2026-05-20 baseline 的任务入口：官方 G1 USD + 普通 G1 AMP 任务 + 3000 iterations
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-v0 MAX_ITERATIONS=3000 RUN_NAME=repro_baseline_3000_accad50hz_rsi QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 7. 官方 G1 USD + CMU walk core adaptive AMP 任务 + 4000 iterations
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkCore-Adaptive-v0 MAX_ITERATIONS=4000 RUN_NAME=orig_g1_29dof_cmu_walk_core_adaptive_4000 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 8. 官方 G1 USD + CMU walk full adaptive AMP 任务 + 4000 iterations
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkFull-Adaptive-v0 MAX_ITERATIONS=4000 RUN_NAME=orig_g1_29dof_cmu_walk_full_adaptive_4000 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 9. 官方 G1 USD + CMU walk washed adaptive AMP smoke run
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0 NUM_ENVS=128 MAX_ITERATIONS=2 RUN_NAME=smoke_g1_cmu_walk_washed_adaptive STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.35 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 10. 官方 G1 USD + CMU walk washed adaptive AMP 中等风格/yaw 调参 + 4000 iterations
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0 MAX_ITERATIONS=4000 RUN_NAME=orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.35 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 10a. 从 washed adaptive model_3999 续跑 strict upright gait reward finetune
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Strict-v0 RESUME=True LOAD_RUN=2026-06-11_17-43-37_orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000 CHECKPOINT=model_3999.pt MAX_ITERATIONS=1500 RUN_NAME=orig_g1_29dof_cmu_walk_washed_strict_upright_resume3999_1500 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.50 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 11. 使用官方 G1 USD 从 5 月 baseline checkpoint 续跑 segmented-yaw finetune，观察任务切换本身的影响
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-v0 RESUME=True LOAD_RUN=2026-05-20_23-50-16_baseline_3000_accad50hz_rsi_20260520 CHECKPOINT=model_2999.pt RSI_ENABLE=True RANDOMIZATION_STRENGTH=1 RUN_NAME=orig_g1_29dof_segmented_yaw_resume2999 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 12. 使用 S3 G1 USD 从同一个 baseline checkpoint 做质量/惯量对齐后的 segmented-yaw finetune
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-v0 RESUME=True LOAD_RUN=2026-05-20_23-50-16_baseline_3000_accad50hz_rsi_20260520 CHECKPOINT=model_2999.pt RSI_ENABLE=True RANDOMIZATION_STRENGTH=1 RUN_NAME=s3_g1_29dof_segmented_yaw_resume2999 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 13. 使用原版 G1 29DoF 资产目录从随机策略跑同一 AMP segmented-yaw finetune 任务
ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-v0 RSI_ENABLE=True RANDOMIZATION_STRENGTH=1 RUN_NAME=orig_g1_29dof_segmented_yaw QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 14. 以 S3 XML 为输入显式重建一次 S3 USD；不改 XML，之后回到 ROBOT_ASSET=s3_g1_29dof 使用缓存 USD
ROBOT_ASSET=s3_g1_29dof_mjcf ROBOT_FORCE_USD_CONVERSION=True NUM_ENVS=128 MAX_ITERATIONS=1 RUN_NAME=rebuild_s3_g1_usd QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 15. S3 G1 command-balanced directional strict arm-prior AMP smoke run
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-v0 NUM_ENVS=128 MAX_ITERATIONS=2 RUN_NAME=smoke_s3_g1_command_balanced_directional_strict_armprior STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.50 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 16. S3 G1 command-balanced directional strict arm-prior AMP 5000 iterations
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-v0 MAX_ITERATIONS=5000 RUN_NAME=s3_g1_29dof_command_balanced_directional_strict_armprior_5000 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.50 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 17. S3 G1 ToTarget with command-balanced AMP demos
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-ToTarget-CommandBalanced-v0 MAX_ITERATIONS=5000 RUN_NAME=s3_g1_29dof_toTarget_command_balanced_amp_5000 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.55 RSI_ENABLE=False QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 18. 从 command-balanced model_7498 续跑 V2，重点修后退双脚跳和侧向速度不足
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V2-v0 RESUME=True LOAD_RUN=2026-06-16_14-39-11_s3_g1_29dof_command_balanced_directional_strict_armprior_resume4999_2500 CHECKPOINT=model_7498.pt MAX_ITERATIONS=1500 RUN_NAME=s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.50 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh

# 19. 从 V2 model_8997 续跑 V3，重点修 mask/MuJoCo 暴露出的 backward/lateral 过度单支撑和侧移速度泄漏
ROBOT_ASSET=s3_g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V3-v0 RESUME=True LOAD_RUN=2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500 CHECKPOINT=model_8997.pt MAX_ITERATIONS=1000 RUN_NAME=s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.50 QUIET_TERMINAL=True bash scripts/train_g1_amp.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_isaaclab}
CONDA_BASE=${CONDA_BASE:-${HOME}/anaconda3}
if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
        ISAACLAB_PYTHON="${CONDA_PREFIX}/bin/python"
    else
        ISAACLAB_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
    fi
fi
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-SegmentedYawFinetune-v0}
NUM_ENVS=${NUM_ENVS:-8192}
MAX_ITERATIONS=${MAX_ITERATIONS:-4000}
DEVICE=${DEVICE:-cuda:0}
AGENT_DEVICE=${AGENT_DEVICE:-${DEVICE}}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-}
RESUME=${RESUME:-False}
LOAD_RUN=${LOAD_RUN:-}
CHECKPOINT=${CHECKPOINT:-}
HEADLESS=${HEADLESS:-True}
if [[ -z "${RSI_ENABLE:-}" ]]; then
    if [[ "${TASK}" == *"StandPerturb"* ]]; then
        RSI_ENABLE=False
    else
        RSI_ENABLE=True
    fi
fi
RSI_RATIO=${RSI_RATIO:-0.5}
POS_RSI=${POS_RSI:-False}
ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
ROBOT_XML=${ROBOT_XML:-}
ROBOT_USD_PATH=${ROBOT_USD_PATH:-}
ROBOT_USD_DIR=${ROBOT_USD_DIR:-}
ROBOT_USD_FILE_NAME=${ROBOT_USD_FILE_NAME:-}
ROBOT_FORCE_USD_CONVERSION=${ROBOT_FORCE_USD_CONVERSION:-}
RANDOMIZATION_STRENGTH=${RANDOMIZATION_STRENGTH:-1}
TRACK_LIN_WEIGHT=${TRACK_LIN_WEIGHT:-}
TRACK_ANG_WEIGHT=${TRACK_ANG_WEIGHT:-}
TRACK_LIN_STD=${TRACK_LIN_STD:-}
TRACK_ANG_STD=${TRACK_ANG_STD:-}
if [[ -z "${STYLE_REWARD_SCALE:-}" ]]; then
    if [[ "${TASK}" == *"StandPerturb"* ]]; then
        STYLE_REWARD_SCALE=0.0
    elif [[ "${TASK}" == *"ToTarget-V2"* ]]; then
        STYLE_REWARD_SCALE=10.0
    elif [[ "${TASK}" == *"Strict"* || "${TASK}" == *"ToTarget-CommandBalanced"* ]]; then
        STYLE_REWARD_SCALE=8.0
    else
        STYLE_REWARD_SCALE=5.0
    fi
fi
if [[ -z "${TASK_STYLE_LERP:-}" ]]; then
    if [[ "${TASK}" == *"StandPerturb"* ]]; then
        TASK_STYLE_LERP=1.0
    elif [[ "${TASK}" == *"ToTarget-V2"* ]]; then
        TASK_STYLE_LERP=0.50
    elif [[ "${TASK}" == *"ToTarget-CommandBalanced"* ]]; then
        TASK_STYLE_LERP=0.55
    elif [[ "${TASK}" == *"Strict"* ]]; then
        TASK_STYLE_LERP=0.50
    else
        TASK_STYLE_LERP=0.4
    fi
fi
if [[ -z "${ENTROPY_COEF:-}" && "${TASK}" == *"ToTarget-V2"* ]]; then
    ENTROPY_COEF=0.002
fi
AMP_GRAD_PENALTY_SCALE=${AMP_GRAD_PENALTY_SCALE:-10.0}
BASELINE_KL_ENABLE=${BASELINE_KL_ENABLE:-False}
BASELINE_KL_CHECKPOINT=${BASELINE_KL_CHECKPOINT:-}
BASELINE_KL_SCALE=${BASELINE_KL_SCALE:-0.0}
BASELINE_KL_MIN_STD=${BASELINE_KL_MIN_STD:-1e-4}
QUIET_TERMINAL=${QUIET_TERMINAL:-True}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-${LEGGED_LAB_DIR}/logs/rsl_rl/g1_amp/train_${RUN_NAME:-g1_amp}_$(date +%Y%m%d_%H%M%S).log}
EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-}

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Error: ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi

case "${ROBOT_ASSET}" in
    config)
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
        ROBOT_FORCE_USD_CONVERSION=${ROBOT_FORCE_USD_CONVERSION:-False}
        ;;
    s3_g1_29dof|s3)
        ROBOT_ASSET="s3_g1_29dof"
        ROBOT_USD_PATH=${ROBOT_USD_PATH:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/usd/s3_g1_29dof.usd}
        ;;
    s3_g1_29dof_mjcf|s3_mjcf)
        ROBOT_ASSET="s3_g1_29dof_mjcf"
        ROBOT_XML=${ROBOT_XML:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/g1_29dof.xml}
        ROBOT_USD_DIR=${ROBOT_USD_DIR:-${LEGGED_LAB_DIR}/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/usd}
        ROBOT_USD_FILE_NAME=${ROBOT_USD_FILE_NAME:-s3_g1_29dof.usd}
        ROBOT_FORCE_USD_CONVERSION=${ROBOT_FORCE_USD_CONVERSION:-False}
        ;;
    *)
        echo "Error: unknown ROBOT_ASSET preset: ${ROBOT_ASSET}" >&2
        echo "Valid values: config, g1_29dof, s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof_mjcf" >&2
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

args=(
    scripts/rsl_rl/train.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${MAX_ITERATIONS}"
    --seed "${SEED}"
    --device "${DEVICE}"
    agent.device="${AGENT_DEVICE}"
    agent.algorithm.amp_cfg.amp_discriminator.style_reward_scale="${STYLE_REWARD_SCALE}"
    agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp="${TASK_STYLE_LERP}"
    agent.algorithm.amp_cfg.grad_penalty_scale="${AMP_GRAD_PENALTY_SCALE}"
)
if [[ -n "${ENTROPY_COEF:-}" ]]; then
    args+=(agent.algorithm.entropy_coef="${ENTROPY_COEF}")
fi

if [[ "${RESUME}" == "True" || "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
    args+=(--resume)
fi
if [[ -n "${LOAD_RUN}" ]]; then
    args+=(--load_run "${LOAD_RUN}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
    args+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -n "${TRACK_LIN_WEIGHT}" ]]; then
    args+=(env.rewards.track_lin_vel_xy_exp.weight="${TRACK_LIN_WEIGHT}")
fi
if [[ -n "${TRACK_LIN_STD}" ]]; then
    args+=(env.rewards.track_lin_vel_xy_exp.params.std="${TRACK_LIN_STD}")
fi
if [[ -n "${TRACK_ANG_WEIGHT}" ]]; then
    args+=(env.rewards.track_ang_vel_z_exp.weight="${TRACK_ANG_WEIGHT}")
fi
if [[ -n "${TRACK_ANG_STD}" ]]; then
    args+=(env.rewards.track_ang_vel_z_exp.params.std="${TRACK_ANG_STD}")
fi

if [[ -n "${RUN_NAME}" ]]; then
    args+=(--run_name "${RUN_NAME}")
fi

if [[ "${HEADLESS}" == "True" || "${HEADLESS}" == "true" || "${HEADLESS}" == "1" ]]; then
    export HEADLESS=1
    args+=(--headless)
else
    export HEADLESS=0
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

if [[ -n "${ROBOT_XML}" ]]; then
    args+=(env.scene.robot.spawn.asset_path="${ROBOT_XML}")
    args+=(env.scene.robot.spawn.usd_dir="${ROBOT_USD_DIR}")
    args+=(env.scene.robot.spawn.usd_file_name="${ROBOT_USD_FILE_NAME}")
    args+=(env.scene.robot.spawn.force_usd_conversion="${ROBOT_FORCE_USD_CONVERSION}")
fi

if [[ "${BASELINE_KL_ENABLE}" == "True" || "${BASELINE_KL_ENABLE}" == "true" || "${BASELINE_KL_ENABLE}" == "1" ]]; then
    if [[ -z "${BASELINE_KL_CHECKPOINT}" ]]; then
        echo "Error: BASELINE_KL_CHECKPOINT must be set when BASELINE_KL_ENABLE=True" >&2
        exit 1
    fi
    if [[ "${BASELINE_KL_CHECKPOINT}" != /* ]]; then
        BASELINE_KL_CHECKPOINT="${ROOT_DIR}/${BASELINE_KL_CHECKPOINT}"
    fi
    args+=(agent.algorithm.baseline_kl_cfg.enabled=True)
    args+=(agent.algorithm.baseline_kl_cfg.checkpoint_path="${BASELINE_KL_CHECKPOINT}")
    args+=(agent.algorithm.baseline_kl_cfg.scale="${BASELINE_KL_SCALE}")
    args+=(agent.algorithm.baseline_kl_cfg.min_std="${BASELINE_KL_MIN_STD}")
fi

args+=("$@")

echo "====================================="
echo "  Training G1 AMP"
echo "====================================="
echo "Task              : ${TASK}"
echo "Num Envs          : ${NUM_ENVS}"
echo "Max Iterations    : ${MAX_ITERATIONS}"
echo "Device            : ${DEVICE}"
echo "Agent Device      : ${AGENT_DEVICE}"
echo "Resume           : ${RESUME} load_run=${LOAD_RUN} checkpoint=${CHECKPOINT}"
echo "RSI Enable/Ratio  : ${RSI_ENABLE}/${RSI_RATIO}"
echo "Randomization     : ${RANDOMIZATION_STRENGTH}"
echo "Robot Asset       : ${ROBOT_ASSET}"
echo "Robot USD Path    : ${ROBOT_USD_PATH}"
echo "Robot XML         : ${ROBOT_XML}"
echo "Robot USD Dir     : ${ROBOT_USD_DIR}"
echo "Force USD Conv    : ${ROBOT_FORCE_USD_CONVERSION}"
echo "Run Name          : ${RUN_NAME}"
echo "Tracking          : lin_w=${TRACK_LIN_WEIGHT} yaw_w=${TRACK_ANG_WEIGHT} lin_std=${TRACK_LIN_STD} yaw_std=${TRACK_ANG_STD}"
echo "AMP               : style=${STYLE_REWARD_SCALE} lerp=${TASK_STYLE_LERP} grad_penalty=${AMP_GRAD_PENALTY_SCALE} entropy=${ENTROPY_COEF:-cfg}"
echo "Baseline KL       : enable=${BASELINE_KL_ENABLE} scale=${BASELINE_KL_SCALE} checkpoint=${BASELINE_KL_CHECKPOINT}"
echo "Quiet Terminal    : ${QUIET_TERMINAL}"
if [[ "${QUIET_TERMINAL}" == "True" || "${QUIET_TERMINAL}" == "true" || "${QUIET_TERMINAL}" == "1" ]]; then
    echo "Train Log File    : ${TRAIN_LOG_FILE}"
fi
echo "Extra Hydra Args  : ${EXTRA_HYDRA_ARGS}"
echo "====================================="

cd "${LEGGED_LAB_DIR}"
if [[ "${QUIET_TERMINAL}" == "True" || "${QUIET_TERMINAL}" == "true" || "${QUIET_TERMINAL}" == "1" ]]; then
    mkdir -p "$(dirname "${TRAIN_LOG_FILE}")"
    echo "Training stdout/stderr redirected to: ${TRAIN_LOG_FILE}"
    "${ISAACLAB_PYTHON}" "${args[@]}" ${EXTRA_HYDRA_ARGS} >"${TRAIN_LOG_FILE}" 2>&1
else
    "${ISAACLAB_PYTHON}" "${args[@]}" ${EXTRA_HYDRA_ARGS}
fi
