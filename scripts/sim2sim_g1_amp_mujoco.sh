#!/usr/bin/env bash
# Run IsaacLab-exported Unitree G1 29DoF AMP policy in MuJoCo for sim2sim validation.
# Usage:
#   bash scripts/sim2sim_g1_amp_mujoco.sh
#   POLICY_PATH=legged_lab/logs/rsl_rl/g1_amp/<run>/exported/policy.pt USE_GLFW=True bash scripts/sim2sim_g1_amp_mujoco.sh
#
# Environment variables:
#   UNITREE_PYTHON      : Python executable from unitree_sim2sim2real/.envrc conda env (default: /home/hecggdz/miniconda3/envs/env_leglab/bin/python).
#   POLICY_PATH         : TorchScript policy exported by scripts/export_g1_amp_policy.sh.
#   ROBOT_ASSET         : MuJoCo robot XML preset: s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof, g1_29dof_mjcf (default: s3_g1_29dof).
#   XML_PATH            : Explicit MuJoCo XML override; unset selects XML from ROBOT_ASSET.
#   USE_GLFW            : True opens mujoco.viewer; False runs headless stepping (default: True).
#   SIMULATION_DURATION : Wall-clock/sim duration seconds used by config override (default: 20.0).
#   REAL_TIME           : True/False. False runs headless metrics faster than wall-clock (default: True).
#   ADD_FLOOR           : True injects a generated floor when XML has no plane geom (default: True).
#   ENSURE_LIGHTING     : True injects headlight/key light into the generated temporary scene if the XML has none (default: True).
#   REPAIR_MISSING_MESHES: True remaps known Unitree rev mesh names in a temporary XML; source assets are not modified (default: True).
#   DROP_MISSING_MESH_GEOMS: True drops only unresolved missing mesh geoms from the temporary XML (default: True).
#   APPLY_JOINT_PASSIVE_PARAMS: True adds deployment damping/armature/frictionloss to temporary XML (default: True).
#   METRICS_PATH        : JSON output path for health, task tracking, and Important Metrics.
#   TORSO_BODY_NAME     : MuJoCo torso body used for IMU metrics (default: torso_link).
#   TORSO_TRACE_ENABLE  : True draws/saves a fixed torso-point trajectory (default: True).
#   TORSO_TRACE_PATH    : CSV path for torso fixed-point trajectory samples.
#   TORSO_TRACE_LOCAL_POINT: YAML-like local point on torso in meters, e.g. '[0.0,0.0,0.18]'.
#   TORSO_TRACE_STRIDE  : Save one tace sample every N sim steps (default: 10).
#   TORSO_TRACE_MAX_POINTS: Max recent GLFW marker spheres (default: 300).
#   FOLLOW_CAMERA_ENABLE: True keeps the GLFW free camera following the robot torso.
#   FOLLOW_CAMERA_DISTANCE/AZIMUTH_DEG/ELEVATION_DEG: MuJoCo free-camera orbit parameters.
#   FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET: Robot local offset used as the follow-camera lookat point.
#   TASK_TRACE_ENABLE   : True draws/saves command-integrated world-frame task trajectory.
#   TASK_TRACE_PATH     : CSV path for task trajectory samples.
#   TASK_TRACE_HEIGHT   : Z height in meters for task trajectory markers (default: 0.05).
#   TASK_TRACE_STRIDE   : Save one task trace sample every N sim steps (default: 10).
#   TASK_TRACE_MAX_POINTS: Max recent GLFW task trajectory marker spheres (default: 300).
#   CMD_INIT            : Three command values as YAML-like list string, e.g. '[0.5, 0.0, 0.0]'.
#   RANDOM_COMMANDS     : True samples joystick-like commands during rollout (default: False).
#   COMMAND_MODE        : independent, curvature, nav2, or joystick (default: independent).
#   COMMAND_RAMP        : True ramps the command visible to the policy toward each target command.
#   COMMAND_INTERVAL    : Seconds between random command samples (default: 2.0).
#   COMMAND_SEED        : Random command seed (default: 1).
#   CMD_LIN_X_RANGE     : YAML-like [min,max] total lin x envelope for random commands (default: [-0.2,1.5]).
#   CMD_LIN_Y_RANGE     : YAML-like [min,max] lin y range for random commands (default: [-0.25,0.25]).
#   CMD_YAW_RANGE       : YAML-like [min,max] yaw rate range for random commands (default: [-0.6,0.6]).
#   CMD_CURVATURE_RANGE / CMD_MAX_CURVATURE: Curvature range/cap for COMMAND_MODE=curvature.
#   CMD_LOW_SPEED_*_RANGE: Low-speed omnidirectional command ranges for COMMAND_MODE=curvature.
#   NAV2_DATA_PATH      : Nav2 cmd_vel CSV for COMMAND_MODE=nav2.
#   NAV2_AUGMENTATION_FILTER: Comma-separated augmentation labels, default none,mirror_lr.
#   NAV2_*_FILTER       : Optional scenario_family/combo/controller/planner filters for COMMAND_MODE=nav2.
#   NAV2_WINDOW_DURATION_S: Continuous Nav2 window duration; <=0 uses SIMULATION_DURATION.
#   NAV2_COMMAND_SCALE  : Stage curriculum scale [vx,vy,wz], default [0.70,0.55,0.55].
#   JOYSTICK_DEVICE     : Linux joystick device for COMMAND_MODE=joystick (default: /dev/input/js0).
#   JOYSTICK_*_RANGE    : [negative,positive] command limits for joystick axes.
#   JOYSTICK_AXIS_*     : Axis ids for lin_x, lin_y, yaw (defaults: 1, 0, 3).
#   JOYSTICK_SIGN_*     : Axis sign multipliers; default lin_x=-1 makes left-stick-up forward.
#   JOYSTICK_AXIS_MAX   : Raw absolute axis max, default 32768.
#   JOYSTICK_DEADZONE   : Normalized axis deadzone, default 0.05.
#   EARLY_MOTION_ENABLE / EARLY_MOTION_WINDOW_S: First-step diagnostic output controls.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 使用默认已导出 checkpoint 在 MuJoCo GLFW 中可视化
bash scripts/sim2sim_g1_amp_mujoco.sh

# 2. Headless 跑 5 秒 smoke test
USE_GLFW=False SIMULATION_DURATION=5 bash scripts/sim2sim_g1_amp_mujoco.sh

# 3. Headless 快速评估健康状态、任务跟踪和 Important Metrics
USE_GLFW=False REAL_TIME=False SIMULATION_DURATION=10 bash scripts/sim2sim_g1_amp_mujoco.sh

# 4. 随机摇杆任务评估，并输出分段 tracking score
USE_GLFW=False REAL_TIME=False RANDOM_COMMANDS=True SIMULATION_DURATION=12 bash scripts/sim2sim_g1_amp_mujoco.sh

# 5. 曲率约束随机任务评估，使用训练同分布 command ramp
POLICY_PATH=outputs/curvature_finetune/probe_200_exported/policy.pt USE_GLFW=True RANDOM_COMMANDS=True COMMAND_MODE=curvature COMMAND_RAMP=True SIMULATION_DURATION=60 bash scripts/sim2sim_g1_amp_mujoco.sh

# 6. GLFW 中显示 torso 固定点轨迹，并同时保存 CSV
USE_GLFW=True TORSO_TRACE_PATH=outputs/g1_amp_torso_trace.csv bash scripts/sim2sim_g1_amp_mujoco.sh

# 7. 使用 /dev/input/js0 手柄实时控制 baseline，并显示任务轨迹(橙色)和 torso 轨迹(蓝色)
TORSO_TRACE_ENABLE=False TASK_TRACE_ENABLE=False \
POLICY_PATH=legged_lab/logs/rsl_rl/g1_amp/2026-06-12_23-16-11_s3_g1_29dof_cmu_walk_washed_strict_armprior_scratch_5000/exported/policy.pt \
ROBOT_ASSET=s3_g1_29dof \
COMMAND_MODE=joystick USE_GLFW=True REAL_TIME=True SIMULATION_DURATION=120 COMMAND_RAMP=False \
JOYSTICK_DEVICE=/dev/input/js0 \
JOYSTICK_LIN_X_RANGE='[-0.6,1.3]' JOYSTICK_LIN_Y_RANGE='[-0.6,0.6]' JOYSTICK_YAW_RANGE='[-0.6,0.6]' \
bash scripts/sim2sim_g1_amp_mujoco.sh


COMMAND_MODE=joystick USE_GLFW=True REAL_TIME=True SIMULATION_DURATION=120 COMMAND_RAMP=True \
JOYSTICK_LIN_X_RANGE='[-0.2,1.5]' JOYSTICK_LIN_Y_RANGE='[-0.25,0.25]' JOYSTICK_YAW_RANGE='[-0.6,0.6]' \
bash scripts/sim2sim_g1_amp_mujoco.sh

# 8. 使用当前 Nav2 loopback 数据集的连续 cmd_vel 窗口进行 MuJoCo 可视化
POLICY_PATH=legged_lab/logs/rsl_rl/g1_amp/2026-06-01_00-11-53_finetune_nav2_stage1_kl/exported/policy.pt \
COMMAND_MODE=nav2 USE_GLFW=True REAL_TIME=True SIMULATION_DURATION=60 \
bash scripts/sim2sim_g1_amp_mujoco.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"

UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
POLICY_PATH=${POLICY_PATH:-${ROOT_DIR}/outputs/baseline/exported/policy.pt}
ROBOT_ASSET=${ROBOT_ASSET:-s3_g1_29dof}
XML_PATH=${XML_PATH:-}
USE_GLFW=${USE_GLFW:-True}
SIMULATION_DURATION=${SIMULATION_DURATION:-20.0}
REAL_TIME=${REAL_TIME:-True}
ADD_FLOOR=${ADD_FLOOR:-True}
ENSURE_LIGHTING=${ENSURE_LIGHTING:-True}
REPAIR_MISSING_MESHES=${REPAIR_MISSING_MESHES:-True}
DROP_MISSING_MESH_GEOMS=${DROP_MISSING_MESH_GEOMS:-True}
APPLY_JOINT_PASSIVE_PARAMS=${APPLY_JOINT_PASSIVE_PARAMS:-True}
JOINT_DAMPING=${JOINT_DAMPING:-0.05}
JOINT_ARMATURE=${JOINT_ARMATURE:-0.01}
JOINT_FRICTIONLOSS=${JOINT_FRICTIONLOSS:-0.2}
WRIST_FRICTIONLOSS=${WRIST_FRICTIONLOSS:-0.1}
METRICS_PATH=${METRICS_PATH:-$(dirname "${POLICY_PATH}")/mujoco_metrics.json}
TORSO_BODY_NAME=${TORSO_BODY_NAME:-torso_link}
TORSO_TRACE_ENABLE=${TORSO_TRACE_ENABLE:-True}
TORSO_TRACE_PATH=${TORSO_TRACE_PATH:-$(dirname "${POLICY_PATH}")/mujoco_torso_trace.csv}
TORSO_TRACE_LOCAL_POINT=${TORSO_TRACE_LOCAL_POINT:-'[0.0, 0.0, 0.18]'}
TORSO_TRACE_STRIDE=${TORSO_TRACE_STRIDE:-10}
TORSO_TRACE_MAX_POINTS=${TORSO_TRACE_MAX_POINTS:-300}
FOLLOW_CAMERA_ENABLE=${FOLLOW_CAMERA_ENABLE:-True}
FOLLOW_CAMERA_DISTANCE=${FOLLOW_CAMERA_DISTANCE:-3.2}
FOLLOW_CAMERA_AZIMUTH_DEG=${FOLLOW_CAMERA_AZIMUTH_DEG:-145.0}
FOLLOW_CAMERA_ELEVATION_DEG=${FOLLOW_CAMERA_ELEVATION_DEG:--20.0}
FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET=${FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET:-'[-0.35, -0.20, 0.20]'}
TASK_TRACE_ENABLE=${TASK_TRACE_ENABLE:-True}
TASK_TRACE_PATH=${TASK_TRACE_PATH:-$(dirname "${POLICY_PATH}")/mujoco_task_trace.csv}
TASK_TRACE_HEIGHT=${TASK_TRACE_HEIGHT:-0.05}
TASK_TRACE_STRIDE=${TASK_TRACE_STRIDE:-10}
TASK_TRACE_MAX_POINTS=${TASK_TRACE_MAX_POINTS:-300}
CMD_INIT=${CMD_INIT:-'[0.5, 0.0, 0.0]'}
RANDOM_COMMANDS=${RANDOM_COMMANDS:-False}
COMMAND_MODE=${COMMAND_MODE:-independent}
if [[ -z "${COMMAND_RAMP+x}" ]]; then
    if [[ "${COMMAND_MODE}" == "curvature" || "${COMMAND_MODE}" == "joystick" || "${COMMAND_MODE}" == "nav2" ]]; then
        COMMAND_RAMP=True
    else
        COMMAND_RAMP=False
    fi
fi
COMMAND_INTERVAL=${COMMAND_INTERVAL:-2.0}
COMMAND_SEED=${COMMAND_SEED:-1}
CMD_LIN_X_RANGE=${CMD_LIN_X_RANGE:-'[-0.2,1.5]'}
CMD_LIN_Y_RANGE=${CMD_LIN_Y_RANGE:-'[-0.25,0.25]'}
CMD_YAW_RANGE=${CMD_YAW_RANGE:-'[-0.6,0.6]'}
CMD_CURVATURE_RANGE=${CMD_CURVATURE_RANGE:-'[-0.7,0.7]'}
CMD_MAX_CURVATURE=${CMD_MAX_CURVATURE:-0.7}
CMD_LOW_SPEED_LIN_X_RANGE=${CMD_LOW_SPEED_LIN_X_RANGE:-'[-0.20,0.35]'}
CMD_LOW_SPEED_LIN_Y_RANGE=${CMD_LOW_SPEED_LIN_Y_RANGE:-'[-0.25,0.25]'}
CMD_LOW_SPEED_YAW_RANGE=${CMD_LOW_SPEED_YAW_RANGE:-'[-0.50,0.50]'}
CMD_YAW_NOISE_RANGE=${CMD_YAW_NOISE_RANGE:-'[-0.05,0.05]'}
CMD_REL_LOW_SPEED=${CMD_REL_LOW_SPEED:-0.25}
CMD_LATERAL_DECAY_START_SPEED=${CMD_LATERAL_DECAY_START_SPEED:-0.25}
CMD_LATERAL_DECAY_END_SPEED=${CMD_LATERAL_DECAY_END_SPEED:-0.80}
CMD_HIGH_SPEED_LATERAL_VEL=${CMD_HIGH_SPEED_LATERAL_VEL:-0.06}
COMMAND_SMOOTHING_TAU=${COMMAND_SMOOTHING_TAU:-0.30}
if [[ -z "${COMMAND_MAX_LINEAR_ACCEL+x}" ]]; then
    if [[ "${COMMAND_MODE}" == "nav2" ]]; then
        COMMAND_MAX_LINEAR_ACCEL=0.60
    else
        COMMAND_MAX_LINEAR_ACCEL=0.80
    fi
fi
if [[ -z "${COMMAND_MAX_YAW_ACCEL+x}" ]]; then
    if [[ "${COMMAND_MODE}" == "nav2" ]]; then
        COMMAND_MAX_YAW_ACCEL=0.80
    else
        COMMAND_MAX_YAW_ACCEL=1.00
    fi
fi
NAV2_DATA_PATH=${NAV2_DATA_PATH:-${ROOT_DIR}/nav2_loopback_actual/actual_augmented/all_cmd_vel_augmented.csv}
NAV2_AUGMENTATION_FILTER=${NAV2_AUGMENTATION_FILTER:-none,mirror_lr}
NAV2_SCENARIO_FAMILY_FILTER=${NAV2_SCENARIO_FAMILY_FILTER:-}
NAV2_COMBO_FILTER=${NAV2_COMBO_FILTER:-}
NAV2_CONTROLLER_FILTER=${NAV2_CONTROLLER_FILTER:-}
NAV2_PLANNER_FILTER=${NAV2_PLANNER_FILTER:-}
NAV2_DATASET_SAMPLE_DT=${NAV2_DATASET_SAMPLE_DT:-0.05}
NAV2_WINDOW_DURATION_S=${NAV2_WINDOW_DURATION_S:-0.0}
NAV2_COMMAND_SCALE=${NAV2_COMMAND_SCALE:-'[0.70,0.55,0.55]'}
NAV2_COMMAND_CLIP_MIN=${NAV2_COMMAND_CLIP_MIN:-'[-0.6,-0.3,-0.6]'}
NAV2_COMMAND_CLIP_MAX=${NAV2_COMMAND_CLIP_MAX:-'[0.6,0.3,0.6]'}
EARLY_MOTION_ENABLE=${EARLY_MOTION_ENABLE:-True}
EARLY_MOTION_WINDOW_S=${EARLY_MOTION_WINDOW_S:-1.0}
JOYSTICK_DEVICE=${JOYSTICK_DEVICE:-/dev/input/js0}
JOYSTICK_AXIS_LIN_X=${JOYSTICK_AXIS_LIN_X:-1}
JOYSTICK_AXIS_LIN_Y=${JOYSTICK_AXIS_LIN_Y:-0}
JOYSTICK_AXIS_YAW=${JOYSTICK_AXIS_YAW:-3}
JOYSTICK_SIGN_LIN_X=${JOYSTICK_SIGN_LIN_X:--1.0}
JOYSTICK_SIGN_LIN_Y=${JOYSTICK_SIGN_LIN_Y:--1.0}
JOYSTICK_SIGN_YAW=${JOYSTICK_SIGN_YAW:--1.0}
JOYSTICK_AXIS_MAX=${JOYSTICK_AXIS_MAX:-32768.0}
JOYSTICK_DEADZONE=${JOYSTICK_DEADZONE:-0.05}
JOYSTICK_LIN_X_RANGE=${JOYSTICK_LIN_X_RANGE:-${CMD_LIN_X_RANGE}}
JOYSTICK_LIN_Y_RANGE=${JOYSTICK_LIN_Y_RANGE:-${CMD_LIN_Y_RANGE}}
JOYSTICK_YAW_RANGE=${JOYSTICK_YAW_RANGE:-${CMD_YAW_RANGE}}

case "${ROBOT_ASSET}" in
    s3_g1_29dof|s3|s3_g1_29dof_mjcf|s3_mjcf)
        ROBOT_ASSET="s3_g1_29dof"
        DEFAULT_XML_PATH="${ROOT_DIR}/legged_lab/source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/g1_29dof.xml"
        ;;
    g1_29dof|g1|original_g1)
        ROBOT_ASSET="g1_29dof"
        DEFAULT_XML_PATH="${ROOT_DIR}/unitree_sim2sim2real/resources/robots/g1_description/g1_29dof.xml"
        ;;
    g1_29dof_mjcf|g1_mjcf)
        ROBOT_ASSET="g1_29dof_mjcf"
        DEFAULT_XML_PATH="${ROOT_DIR}/legged_lab/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/g1_29dof.xml"
        ;;
    *)
        echo "Error: unknown ROBOT_ASSET preset: ${ROBOT_ASSET}" >&2
        echo "Valid values: s3_g1_29dof, s3_g1_29dof_mjcf, g1_29dof, g1_29dof_mjcf" >&2
        exit 1
        ;;
esac

if [[ -z "${XML_PATH}" ]]; then
    XML_PATH="${DEFAULT_XML_PATH}"
fi

if [[ "${POLICY_PATH}" != /* ]]; then
    POLICY_PATH="${ROOT_DIR}/${POLICY_PATH}"
fi
if [[ "${XML_PATH}" != /* ]]; then
    XML_PATH="${ROOT_DIR}/${XML_PATH}"
fi
if [[ "${NAV2_DATA_PATH}" != /* ]]; then
    NAV2_DATA_PATH="${ROOT_DIR}/${NAV2_DATA_PATH}"
fi
if [[ "${METRICS_PATH}" != /* ]]; then
    METRICS_PATH="${ROOT_DIR}/${METRICS_PATH}"
fi
if [[ -n "${TORSO_TRACE_PATH}" && "${TORSO_TRACE_PATH}" != /* ]]; then
    TORSO_TRACE_PATH="${ROOT_DIR}/${TORSO_TRACE_PATH}"
fi
if [[ -n "${TASK_TRACE_PATH}" && "${TASK_TRACE_PATH}" != /* ]]; then
    TASK_TRACE_PATH="${ROOT_DIR}/${TASK_TRACE_PATH}"
fi

if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${POLICY_PATH}" ]]; then
    echo "Error: POLICY_PATH does not exist: ${POLICY_PATH}" >&2
    echo "Run: CHECKPOINT=<checkpoint> bash scripts/export_g1_amp_policy.sh" >&2
    exit 1
fi
if [[ ! -f "${XML_PATH}" ]]; then
    echo "Error: XML_PATH does not exist: ${XML_PATH}" >&2
    exit 1
fi
if [[ "${COMMAND_MODE}" == "nav2" && ! -f "${NAV2_DATA_PATH}" ]]; then
    echo "Error: NAV2_DATA_PATH does not exist: ${NAV2_DATA_PATH}" >&2
    exit 1
fi
if [[ "${COMMAND_MODE}" == "joystick" && ! -r "${JOYSTICK_DEVICE}" ]]; then
    echo "Error: JOYSTICK_DEVICE is not readable: ${JOYSTICK_DEVICE}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export G1_AMP_POLICY_PATH="${POLICY_PATH}"
export G1_AMP_ROBOT_ASSET="${ROBOT_ASSET}"
export G1_AMP_XML_PATH="${XML_PATH}"
export G1_AMP_USE_GLFW="${USE_GLFW}"
export G1_AMP_SIMULATION_DURATION="${SIMULATION_DURATION}"
export G1_AMP_REAL_TIME="${REAL_TIME}"
export G1_AMP_ADD_FLOOR="${ADD_FLOOR}"
export G1_AMP_ENSURE_LIGHTING="${ENSURE_LIGHTING}"
export G1_AMP_REPAIR_MISSING_MESHES="${REPAIR_MISSING_MESHES}"
export G1_AMP_DROP_MISSING_MESH_GEOMS="${DROP_MISSING_MESH_GEOMS}"
export G1_AMP_APPLY_JOINT_PASSIVE_PARAMS="${APPLY_JOINT_PASSIVE_PARAMS}"
export G1_AMP_JOINT_DAMPING="${JOINT_DAMPING}"
export G1_AMP_JOINT_ARMATURE="${JOINT_ARMATURE}"
export G1_AMP_JOINT_FRICTIONLOSS="${JOINT_FRICTIONLOSS}"
export G1_AMP_WRIST_FRICTIONLOSS="${WRIST_FRICTIONLOSS}"
export G1_AMP_METRICS_PATH="${METRICS_PATH}"
export G1_AMP_TORSO_BODY_NAME="${TORSO_BODY_NAME}"
export G1_AMP_TORSO_TRACE_ENABLE="${TORSO_TRACE_ENABLE}"
export G1_AMP_TORSO_TRACE_PATH="${TORSO_TRACE_PATH}"
export G1_AMP_TORSO_TRACE_LOCAL_POINT="${TORSO_TRACE_LOCAL_POINT}"
export G1_AMP_TORSO_TRACE_STRIDE="${TORSO_TRACE_STRIDE}"
export G1_AMP_TORSO_TRACE_MAX_POINTS="${TORSO_TRACE_MAX_POINTS}"
export G1_AMP_FOLLOW_CAMERA_ENABLE="${FOLLOW_CAMERA_ENABLE}"
export G1_AMP_FOLLOW_CAMERA_DISTANCE="${FOLLOW_CAMERA_DISTANCE}"
export G1_AMP_FOLLOW_CAMERA_AZIMUTH_DEG="${FOLLOW_CAMERA_AZIMUTH_DEG}"
export G1_AMP_FOLLOW_CAMERA_ELEVATION_DEG="${FOLLOW_CAMERA_ELEVATION_DEG}"
export G1_AMP_FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET="${FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET}"
export G1_AMP_TASK_TRACE_ENABLE="${TASK_TRACE_ENABLE}"
export G1_AMP_TASK_TRACE_PATH="${TASK_TRACE_PATH}"
export G1_AMP_TASK_TRACE_HEIGHT="${TASK_TRACE_HEIGHT}"
export G1_AMP_TASK_TRACE_STRIDE="${TASK_TRACE_STRIDE}"
export G1_AMP_TASK_TRACE_MAX_POINTS="${TASK_TRACE_MAX_POINTS}"
export G1_AMP_CMD_INIT="${CMD_INIT}"
export G1_AMP_RANDOM_COMMANDS="${RANDOM_COMMANDS}"
export G1_AMP_COMMAND_MODE="${COMMAND_MODE}"
export G1_AMP_COMMAND_RAMP="${COMMAND_RAMP}"
export G1_AMP_COMMAND_INTERVAL="${COMMAND_INTERVAL}"
export G1_AMP_COMMAND_SEED="${COMMAND_SEED}"
export G1_AMP_CMD_LIN_X_RANGE="${CMD_LIN_X_RANGE}"
export G1_AMP_CMD_LIN_Y_RANGE="${CMD_LIN_Y_RANGE}"
export G1_AMP_CMD_YAW_RANGE="${CMD_YAW_RANGE}"
export G1_AMP_CMD_CURVATURE_RANGE="${CMD_CURVATURE_RANGE}"
export G1_AMP_CMD_MAX_CURVATURE="${CMD_MAX_CURVATURE}"
export G1_AMP_CMD_LOW_SPEED_LIN_X_RANGE="${CMD_LOW_SPEED_LIN_X_RANGE}"
export G1_AMP_CMD_LOW_SPEED_LIN_Y_RANGE="${CMD_LOW_SPEED_LIN_Y_RANGE}"
export G1_AMP_CMD_LOW_SPEED_YAW_RANGE="${CMD_LOW_SPEED_YAW_RANGE}"
export G1_AMP_CMD_YAW_NOISE_RANGE="${CMD_YAW_NOISE_RANGE}"
export G1_AMP_CMD_REL_LOW_SPEED="${CMD_REL_LOW_SPEED}"
export G1_AMP_CMD_LATERAL_DECAY_START_SPEED="${CMD_LATERAL_DECAY_START_SPEED}"
export G1_AMP_CMD_LATERAL_DECAY_END_SPEED="${CMD_LATERAL_DECAY_END_SPEED}"
export G1_AMP_CMD_HIGH_SPEED_LATERAL_VEL="${CMD_HIGH_SPEED_LATERAL_VEL}"
export G1_AMP_COMMAND_SMOOTHING_TAU="${COMMAND_SMOOTHING_TAU}"
export G1_AMP_COMMAND_MAX_LINEAR_ACCEL="${COMMAND_MAX_LINEAR_ACCEL}"
export G1_AMP_COMMAND_MAX_YAW_ACCEL="${COMMAND_MAX_YAW_ACCEL}"
export G1_AMP_NAV2_DATA_PATH="${NAV2_DATA_PATH}"
export G1_AMP_NAV2_AUGMENTATION_FILTER="${NAV2_AUGMENTATION_FILTER}"
export G1_AMP_NAV2_SCENARIO_FAMILY_FILTER="${NAV2_SCENARIO_FAMILY_FILTER}"
export G1_AMP_NAV2_COMBO_FILTER="${NAV2_COMBO_FILTER}"
export G1_AMP_NAV2_CONTROLLER_FILTER="${NAV2_CONTROLLER_FILTER}"
export G1_AMP_NAV2_PLANNER_FILTER="${NAV2_PLANNER_FILTER}"
export G1_AMP_NAV2_DATASET_SAMPLE_DT="${NAV2_DATASET_SAMPLE_DT}"
export G1_AMP_NAV2_WINDOW_DURATION_S="${NAV2_WINDOW_DURATION_S}"
export G1_AMP_NAV2_COMMAND_SCALE="${NAV2_COMMAND_SCALE}"
export G1_AMP_NAV2_COMMAND_CLIP_MIN="${NAV2_COMMAND_CLIP_MIN}"
export G1_AMP_NAV2_COMMAND_CLIP_MAX="${NAV2_COMMAND_CLIP_MAX}"
export G1_AMP_EARLY_MOTION_ENABLE="${EARLY_MOTION_ENABLE}"
export G1_AMP_EARLY_MOTION_WINDOW_S="${EARLY_MOTION_WINDOW_S}"
export G1_AMP_JOYSTICK_DEVICE="${JOYSTICK_DEVICE}"
export G1_AMP_JOYSTICK_AXIS_LIN_X="${JOYSTICK_AXIS_LIN_X}"
export G1_AMP_JOYSTICK_AXIS_LIN_Y="${JOYSTICK_AXIS_LIN_Y}"
export G1_AMP_JOYSTICK_AXIS_YAW="${JOYSTICK_AXIS_YAW}"
export G1_AMP_JOYSTICK_SIGN_LIN_X="${JOYSTICK_SIGN_LIN_X}"
export G1_AMP_JOYSTICK_SIGN_LIN_Y="${JOYSTICK_SIGN_LIN_Y}"
export G1_AMP_JOYSTICK_SIGN_YAW="${JOYSTICK_SIGN_YAW}"
export G1_AMP_JOYSTICK_AXIS_MAX="${JOYSTICK_AXIS_MAX}"
export G1_AMP_JOYSTICK_DEADZONE="${JOYSTICK_DEADZONE}"
export G1_AMP_JOYSTICK_LIN_X_RANGE="${JOYSTICK_LIN_X_RANGE}"
export G1_AMP_JOYSTICK_LIN_Y_RANGE="${JOYSTICK_LIN_Y_RANGE}"
export G1_AMP_JOYSTICK_YAW_RANGE="${JOYSTICK_YAW_RANGE}"

echo "====================================="
echo "  MuJoCo G1 AMP Sim2Sim"
echo "====================================="
echo "Policy Path : ${POLICY_PATH}"
echo "Robot Asset : ${ROBOT_ASSET}"
echo "XML Path    : ${XML_PATH}"
echo "Use GLFW    : ${USE_GLFW}"
echo "Duration    : ${SIMULATION_DURATION}"
echo "Real Time   : ${REAL_TIME}"
echo "Add Floor   : ${ADD_FLOOR}"
echo "Lighting    : ${ENSURE_LIGHTING}"
echo "Mesh Repair : ${REPAIR_MISSING_MESHES}"
echo "Drop Missing: ${DROP_MISSING_MESH_GEOMS}"
echo "Passive Dyn : ${APPLY_JOINT_PASSIVE_PARAMS} damping=${JOINT_DAMPING} armature=${JOINT_ARMATURE} friction=${JOINT_FRICTIONLOSS} wrist=${WRIST_FRICTIONLOSS}"
echo "Metrics     : ${METRICS_PATH}"
echo "Torso Body  : ${TORSO_BODY_NAME}"
echo "Torso Trace : ${TORSO_TRACE_ENABLE} point=${TORSO_TRACE_LOCAL_POINT} stride=${TORSO_TRACE_STRIDE} csv=${TORSO_TRACE_PATH}"
echo "Follow Cam  : ${FOLLOW_CAMERA_ENABLE} dist=${FOLLOW_CAMERA_DISTANCE} az=${FOLLOW_CAMERA_AZIMUTH_DEG} el=${FOLLOW_CAMERA_ELEVATION_DEG} lookat=${FOLLOW_CAMERA_LOOKAT_LOCAL_OFFSET}"
echo "Task Trace  : ${TASK_TRACE_ENABLE} height=${TASK_TRACE_HEIGHT} stride=${TASK_TRACE_STRIDE} csv=${TASK_TRACE_PATH}"
echo "Command     : ${CMD_INIT}"
echo "Random Cmds : ${RANDOM_COMMANDS}"
echo "Cmd Mode    : ${COMMAND_MODE} ramp=${COMMAND_RAMP} tau=${COMMAND_SMOOTHING_TAU} lin_acc=${COMMAND_MAX_LINEAR_ACCEL} yaw_acc=${COMMAND_MAX_YAW_ACCEL}"
echo "Cmd Interval: ${COMMAND_INTERVAL}"
echo "Cmd Ranges  : x=${CMD_LIN_X_RANGE} y=${CMD_LIN_Y_RANGE} yaw=${CMD_YAW_RANGE}"
echo "Curvature   : kappa=${CMD_CURVATURE_RANGE} max=${CMD_MAX_CURVATURE} low_frac=${CMD_REL_LOW_SPEED} high_y=${CMD_HIGH_SPEED_LATERAL_VEL}"
echo "Nav2 Replay : data=${NAV2_DATA_PATH} aug=${NAV2_AUGMENTATION_FILTER} scale=${NAV2_COMMAND_SCALE} window=${NAV2_WINDOW_DURATION_S}s"
echo "Joystick    : device=${JOYSTICK_DEVICE} axes=(${JOYSTICK_AXIS_LIN_X},${JOYSTICK_AXIS_LIN_Y},${JOYSTICK_AXIS_YAW}) ranges=x${JOYSTICK_LIN_X_RANGE} y${JOYSTICK_LIN_Y_RANGE} yaw${JOYSTICK_YAW_RANGE} deadzone=${JOYSTICK_DEADZONE}"
echo "Early Motion: ${EARLY_MOTION_ENABLE} window=${EARLY_MOTION_WINDOW_S}s"
echo "====================================="

cd "${UNITREE_DIR}"
"${UNITREE_PYTHON}" deploy/deploy_mujoco/deploy_mujoco_g1_amp.py deploy/deploy_mujoco/configs/g1_amp.yaml
