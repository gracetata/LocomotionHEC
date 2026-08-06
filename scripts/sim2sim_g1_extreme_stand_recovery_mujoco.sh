#!/usr/bin/env bash
# MuJoCo sim2sim launcher and multi-profile suite for the latest Smooth-Torque V4 policy.
#
# GUI interactive (SPACE cycles large torso pushes and reset poses; R immediately resets
# to the default standing pose; K samples foot spacing; F toggles periodic random wrench):
#   PROFILE=interactive USE_GLFW=True SIMULATION_DURATION=300 \
#     bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
# Full suite (7 profiles x 3 seeds):
#   SUITE=True USE_GLFW=False \
#     bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_CHECKPOINT="${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt"
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
EXPORT_DIR=${EXPORT_DIR:-"$(dirname "${CHECKPOINT}")/exported_extreme_stand_recovery"}
POLICY_PATH=${POLICY_PATH:-"${EXPORT_DIR}/policy.onnx"}
MODEL_LABEL=${MODEL_LABEL:-smooth_torque_v4_model2999}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
if [[ -z "${UNITREE_PYTHON:-}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/gmr/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            UNITREE_PYTHON="${candidate}"
            break
        fi
    done
fi
FORCE_EXPORT=${FORCE_EXPORT:-False}

PROFILE=${PROFILE:-interactive}
SEED=${SEED:-20260719}
USE_GLFW=${USE_GLFW:-True}
SIMULATION_DURATION=${SIMULATION_DURATION:-30.0}
REAL_TIME=${REAL_TIME:-${USE_GLFW}}
RENDER_FPS=${RENDER_FPS:-60}
REALTIME_STATUS_INTERVAL_S=${REALTIME_STATUS_INTERVAL_S:-5}
FOLLOW_CAMERA=${FOLLOW_CAMERA:-False}
SUITE=${SUITE:-False}
SUITE_PROFILES=${SUITE_PROFILES:-nominal,pose_recovery,feet_distance_recovery,recovery,robust,stress,large_push}
SUITE_SEEDS=${SUITE_SEEDS:-20260719,20260720,20260721}
SUITE_DURATION=${SUITE_DURATION:-12.0}
REQUIRE_PASS=${REQUIRE_PASS:-False}
RESULTS_ROOT=${RESULTS_ROOT:-"${EXPORT_DIR}/mujoco_tests/manual"}
STEADY_START_S=${STEADY_START_S:-10.0}
FEET_GAUSSIAN_VARIANCE_M2=${FEET_GAUSSIAN_VARIANCE_M2:-1.0e-4}
JOINT_JERK_REWARD_WEIGHT=${JOINT_JERK_REWARD_WEIGHT:--5.0e-8}
FOOT_SPACING_MIN_DELTA_M=${FOOT_SPACING_MIN_DELTA_M:-0.05}
FOOT_SPACING_MAX_DELTA_M=${FOOT_SPACING_MAX_DELTA_M:-0.12}
FOOT_SPACING_MAX_ROLL_OFFSET_RAD=${FOOT_SPACING_MAX_ROLL_OFFSET_RAD:-0.35}
FOOT_SPACING_SEARCH_SAMPLES=${FOOT_SPACING_SEARCH_SAMPLES:-141}
FOOT_SPACING_RECOVERY_TOLERANCE_M=${FOOT_SPACING_RECOVERY_TOLERANCE_M:-0.02}
LARGE_PUSH_FORCE_N=${LARGE_PUSH_FORCE_N:-120.0}
LARGE_PUSH_DURATION_S=${LARGE_PUSH_DURATION_S:-0.20}
LARGE_PUSH_TIME_S=${LARGE_PUSH_TIME_S:-5.0}
LARGE_PUSH_DIRECTION_INDEX=${LARGE_PUSH_DIRECTION_INDEX:--1}
LARGE_PUSH_BODY_NAME=${LARGE_PUSH_BODY_NAME:-torso_link}
POST_PUSH_SETTLE_S=${POST_PUSH_SETTLE_S:-2.0}
TARGET_LIMITER_ENABLE=${TARGET_LIMITER_ENABLE:-False}
TARGET_LEG_VELOCITY_LIMIT_RAD_S=${TARGET_LEG_VELOCITY_LIMIT_RAD_S:-25.0}
TARGET_WAIST_VELOCITY_LIMIT_RAD_S=${TARGET_WAIST_VELOCITY_LIMIT_RAD_S:-10.0}
TARGET_ARM_VELOCITY_LIMIT_RAD_S=${TARGET_ARM_VELOCITY_LIMIT_RAD_S:-15.0}
TARGET_LEG_ACCELERATION_LIMIT_RAD_S2=${TARGET_LEG_ACCELERATION_LIMIT_RAD_S2:-600.0}
TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2=${TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2:-250.0}
TARGET_ARM_ACCELERATION_LIMIT_RAD_S2=${TARGET_ARM_ACCELERATION_LIMIT_RAD_S2:-400.0}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

for path_var in CHECKPOINT EXPORT_DIR POLICY_PATH RESULTS_ROOT; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${ROOT_DIR}/${value}"
    fi
done

[[ -n "${UNITREE_PYTHON:-}" && -x "${UNITREE_PYTHON}" ]] || { echo "Error: MuJoCo Python 未设置或不可执行: ${UNITREE_PYTHON:-<unset>}" >&2; exit 1; }

need_export=False
if [[ ! -f "${POLICY_PATH}" ]] || is_true "${FORCE_EXPORT}"; then
    need_export=True
elif [[ -f "${CHECKPOINT}" && "${CHECKPOINT}" -nt "${POLICY_PATH}" ]]; then
    need_export=True
fi
if is_true "${need_export}"; then
    [[ -f "${CHECKPOINT}" ]] || { echo "Error: policy 缺失且没有可导出的 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
    [[ -x "${ISAACLAB_PYTHON}" ]] || { echo "Error: 导出所需 IsaacLab Python 不可执行: ${ISAACLAB_PYTHON}" >&2; exit 1; }
    CHECKPOINT="${CHECKPOINT}" EXPORT_DIR="${EXPORT_DIR}" ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" \
        bash "${ROOT_DIR}/scripts/export_g1_extreme_stand_recovery.sh"
fi
[[ -f "${POLICY_PATH}" ]] || { echo "Error: MuJoCo policy 不存在: ${POLICY_PATH}" >&2; exit 1; }

if is_true "${SUITE}"; then
    if is_true "${USE_GLFW}"; then
        echo "Error: SUITE=True 必须使用 USE_GLFW=False；批量测试不能打开多个窗口。" >&2
        exit 1
    fi
    SUITE_RESULTS_ROOT=${SUITE_RESULTS_ROOT:-"${EXPORT_DIR}/mujoco_tests/suite_$(date +%Y%m%d_%H%M%S)"}
    if [[ "${SUITE_RESULTS_ROOT}" != /* ]]; then
        SUITE_RESULTS_ROOT="${ROOT_DIR}/${SUITE_RESULTS_ROOT}"
    fi
    mkdir -p "${SUITE_RESULTS_ROOT}"
    IFS=',' read -r -a profiles <<<"${SUITE_PROFILES}"
    IFS=',' read -r -a seeds <<<"${SUITE_SEEDS}"
    for profile in "${profiles[@]}"; do
        for seed in "${seeds[@]}"; do
            echo
            echo "===== MuJoCo suite: profile=${profile}, seed=${seed} ====="
            SUITE=False PROFILE="${profile}" SEED="${seed}" \
            USE_GLFW=False REAL_TIME=False SIMULATION_DURATION="${SUITE_DURATION}" \
            CHECKPOINT="${CHECKPOINT}" EXPORT_DIR="${EXPORT_DIR}" POLICY_PATH="${POLICY_PATH}" \
            MODEL_LABEL="${MODEL_LABEL}" STEADY_START_S="${STEADY_START_S}" \
            FEET_GAUSSIAN_VARIANCE_M2="${FEET_GAUSSIAN_VARIANCE_M2}" \
            JOINT_JERK_REWARD_WEIGHT="${JOINT_JERK_REWARD_WEIGHT}" \
            FOOT_SPACING_MIN_DELTA_M="${FOOT_SPACING_MIN_DELTA_M}" \
            FOOT_SPACING_MAX_DELTA_M="${FOOT_SPACING_MAX_DELTA_M}" \
            FOOT_SPACING_MAX_ROLL_OFFSET_RAD="${FOOT_SPACING_MAX_ROLL_OFFSET_RAD}" \
            FOOT_SPACING_SEARCH_SAMPLES="${FOOT_SPACING_SEARCH_SAMPLES}" \
            FOOT_SPACING_RECOVERY_TOLERANCE_M="${FOOT_SPACING_RECOVERY_TOLERANCE_M}" \
            LARGE_PUSH_FORCE_N="${LARGE_PUSH_FORCE_N}" \
            LARGE_PUSH_DURATION_S="${LARGE_PUSH_DURATION_S}" \
            LARGE_PUSH_TIME_S="${LARGE_PUSH_TIME_S}" \
            LARGE_PUSH_DIRECTION_INDEX="${LARGE_PUSH_DIRECTION_INDEX}" \
            LARGE_PUSH_BODY_NAME="${LARGE_PUSH_BODY_NAME}" \
            POST_PUSH_SETTLE_S="${POST_PUSH_SETTLE_S}" \
            TARGET_LIMITER_ENABLE="${TARGET_LIMITER_ENABLE}" \
            TARGET_LEG_VELOCITY_LIMIT_RAD_S="${TARGET_LEG_VELOCITY_LIMIT_RAD_S}" \
            TARGET_WAIST_VELOCITY_LIMIT_RAD_S="${TARGET_WAIST_VELOCITY_LIMIT_RAD_S}" \
            TARGET_ARM_VELOCITY_LIMIT_RAD_S="${TARGET_ARM_VELOCITY_LIMIT_RAD_S}" \
            TARGET_LEG_ACCELERATION_LIMIT_RAD_S2="${TARGET_LEG_ACCELERATION_LIMIT_RAD_S2}" \
            TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2="${TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2}" \
            TARGET_ARM_ACCELERATION_LIMIT_RAD_S2="${TARGET_ARM_ACCELERATION_LIMIT_RAD_S2}" \
            ISAACLAB_PYTHON="${ISAACLAB_PYTHON}" UNITREE_PYTHON="${UNITREE_PYTHON}" \
            FORCE_EXPORT=False RESULTS_ROOT="${SUITE_RESULTS_ROOT}" \
                bash "${BASH_SOURCE[0]}"
        done
    done
    summary_args=(
        "${ROOT_DIR}/scripts/summarize_g1_extreme_stand_recovery_mujoco.py"
        --results-root "${SUITE_RESULTS_ROOT}"
        --output-json "${SUITE_RESULTS_ROOT}/summary.json"
        --output-markdown "${SUITE_RESULTS_ROOT}/REPORT.md"
        --model-label "${MODEL_LABEL}"
        --checkpoint "${CHECKPOINT}"
        --policy "${POLICY_PATH}"
    )
    if is_true "${REQUIRE_PASS}"; then
        summary_args+=(--require-pass)
    fi
    "${ISAACLAB_PYTHON}" "${summary_args[@]}"
    echo "Suite report: ${SUITE_RESULTS_ROOT}/REPORT.md"
    exit 0
fi

default_interactive=False
default_pose_start_random=False
default_wrench_start_enabled=False
default_foot_spacing_start_random=False
default_large_push_enable=False

case "${PROFILE}" in
    interactive)
        default_leg_noise=0.25; default_waist_noise=0.35; default_arm_noise=0.60
        default_joint_vel_noise=0.0; default_root_rp_noise=0.0; default_root_yaw_noise=0.0
        default_root_lin_vel_noise=0.0; default_root_ang_vel_noise=0.0
        default_force_max=35.0; default_torque_max=5.0; default_wrench_interval=2.5; default_wrench_duration=0.25
        default_interactive=True
        ;;
    nominal)
        default_leg_noise=0.0; default_waist_noise=0.0; default_arm_noise=0.0
        default_joint_vel_noise=0.0; default_root_rp_noise=0.0; default_root_yaw_noise=0.0
        default_root_lin_vel_noise=0.0; default_root_ang_vel_noise=0.0
        default_force_max=0.0; default_torque_max=0.0; default_wrench_interval=1000.0; default_wrench_duration=0.25
        ;;
    pose_recovery)
        default_leg_noise=0.25; default_waist_noise=0.35; default_arm_noise=0.60
        default_joint_vel_noise=0.0; default_root_rp_noise=0.0; default_root_yaw_noise=0.0
        default_root_lin_vel_noise=0.0; default_root_ang_vel_noise=0.0
        default_force_max=0.0; default_torque_max=0.0; default_wrench_interval=1000.0; default_wrench_duration=0.25
        ;;
    feet_distance_recovery)
        default_leg_noise=0.0; default_waist_noise=0.0; default_arm_noise=0.0
        default_joint_vel_noise=0.0; default_root_rp_noise=0.0; default_root_yaw_noise=0.0
        default_root_lin_vel_noise=0.0; default_root_ang_vel_noise=0.0
        default_force_max=0.0; default_torque_max=0.0; default_wrench_interval=1000.0; default_wrench_duration=0.25
        default_foot_spacing_start_random=True
        ;;
    recovery)
        default_leg_noise=0.25; default_waist_noise=0.35; default_arm_noise=0.60
        default_joint_vel_noise=1.0; default_root_rp_noise=0.25; default_root_yaw_noise=0.30
        default_root_lin_vel_noise=0.50; default_root_ang_vel_noise=0.80
        default_force_max=0.0; default_torque_max=0.0; default_wrench_interval=1000.0; default_wrench_duration=0.25
        ;;
    robust)
        default_leg_noise=0.25; default_waist_noise=0.35; default_arm_noise=0.60
        default_joint_vel_noise=1.0; default_root_rp_noise=0.25; default_root_yaw_noise=0.30
        default_root_lin_vel_noise=0.50; default_root_ang_vel_noise=0.80
        default_force_max=35.0; default_torque_max=5.0; default_wrench_interval=2.5; default_wrench_duration=0.25
        ;;
    stress)
        default_leg_noise=0.35; default_waist_noise=0.50; default_arm_noise=0.80
        default_joint_vel_noise=1.50; default_root_rp_noise=0.35; default_root_yaw_noise=0.45
        default_root_lin_vel_noise=0.70; default_root_ang_vel_noise=1.10
        default_force_max=50.0; default_torque_max=8.0; default_wrench_interval=2.0; default_wrench_duration=0.30
        ;;
    large_push)
        default_leg_noise=0.0; default_waist_noise=0.0; default_arm_noise=0.0
        default_joint_vel_noise=0.0; default_root_rp_noise=0.0; default_root_yaw_noise=0.0
        default_root_lin_vel_noise=0.0; default_root_ang_vel_noise=0.0
        default_force_max=0.0; default_torque_max=0.0; default_wrench_interval=1000.0; default_wrench_duration=0.25
        default_large_push_enable=True
        ;;
    *)
        echo "Error: PROFILE 必须是 interactive、nominal、pose_recovery、feet_distance_recovery、recovery、robust、stress 或 large_push；当前为 ${PROFILE}" >&2
        exit 1
        ;;
esac

LEG_NOISE_RAD=${LEG_NOISE_RAD:-${default_leg_noise}}
WAIST_NOISE_RAD=${WAIST_NOISE_RAD:-${default_waist_noise}}
ARM_NOISE_RAD=${ARM_NOISE_RAD:-${default_arm_noise}}
JOINT_VEL_NOISE_RAD_S=${JOINT_VEL_NOISE_RAD_S:-${default_joint_vel_noise}}
ROOT_RP_NOISE_RAD=${ROOT_RP_NOISE_RAD:-${default_root_rp_noise}}
ROOT_YAW_NOISE_RAD=${ROOT_YAW_NOISE_RAD:-${default_root_yaw_noise}}
ROOT_LIN_VEL_NOISE_M_S=${ROOT_LIN_VEL_NOISE_M_S:-${default_root_lin_vel_noise}}
ROOT_ANG_VEL_NOISE_RAD_S=${ROOT_ANG_VEL_NOISE_RAD_S:-${default_root_ang_vel_noise}}
FORCE_MAX_N=${FORCE_MAX_N:-${default_force_max}}
TORQUE_MAX_NM=${TORQUE_MAX_NM:-${default_torque_max}}
WRENCH_INTERVAL_S=${WRENCH_INTERVAL_S:-${default_wrench_interval}}
WRENCH_DURATION_S=${WRENCH_DURATION_S:-${default_wrench_duration}}
JOINT_LIMIT_MARGIN_RAD=${JOINT_LIMIT_MARGIN_RAD:-0.02}
JOINT_MAE_THRESHOLD_RAD=${JOINT_MAE_THRESHOLD_RAD:-0.12}
JOINT_MAX_THRESHOLD_RAD=${JOINT_MAX_THRESHOLD_RAD:-0.20}
RECOVERY_HOLD_TIME_S=${RECOVERY_HOLD_TIME_S:-1.0}
RECOVERY_FINAL_WINDOW_S=${RECOVERY_FINAL_WINDOW_S:-1.0}
STEADY_START_S=${STEADY_START_S:-10.0}
FEET_GAUSSIAN_VARIANCE_M2=${FEET_GAUSSIAN_VARIANCE_M2:-1.0e-4}
JOINT_JERK_REWARD_WEIGHT=${JOINT_JERK_REWARD_WEIGHT:--1.0e-8}
INTERACTIVE_ENABLE=${INTERACTIVE_ENABLE:-${default_interactive}}
INTERACTIVE_POSE_START_RANDOM=${INTERACTIVE_POSE_START_RANDOM:-${default_pose_start_random}}
INTERACTIVE_WRENCH_START_ENABLED=${INTERACTIVE_WRENCH_START_ENABLED:-${default_wrench_start_enabled}}
FOOT_SPACING_START_RANDOM=${FOOT_SPACING_START_RANDOM:-${default_foot_spacing_start_random}}
LARGE_PUSH_ENABLE=${LARGE_PUSH_ENABLE:-${default_large_push_enable}}
INTERACTIVE_DATA_LOG=${INTERACTIVE_DATA_LOG:-${default_interactive}}

RUN_DIR="${RESULTS_ROOT}/${PROFILE}/seed_${SEED}"
METRICS_PATH=${METRICS_PATH:-"${RUN_DIR}/metrics.json"}
TORSO_TRACE_PATH=${TORSO_TRACE_PATH:-"${RUN_DIR}/torso_trace.csv"}
TASK_TRACE_PATH=${TASK_TRACE_PATH:-"${RUN_DIR}/task_trace.csv"}
MOTION_TRACE_PATH=${MOTION_TRACE_PATH:-"${RUN_DIR}/motion_quality_trace.csv"}
PUSH_DIAGNOSTIC_PLOT_PATH=${PUSH_DIAGNOSTIC_PLOT_PATH:-"${RUN_DIR}/large_push_diagnostics.png"}
INTERACTIVE_LOG_SESSION=${INTERACTIVE_LOG_SESSION:-$(date +%Y%m%d_%H%M%S)}
INTERACTIVE_LOG_ROOT=${INTERACTIVE_LOG_ROOT:-"${RUN_DIR}/interactive_logs/${INTERACTIVE_LOG_SESSION}"}
INTERACTIVE_LOG_PATH=${INTERACTIVE_LOG_PATH:-"${INTERACTIVE_LOG_ROOT}/interactive_diagnostics_all.csv"}
INTERACTIVE_TRIALS_DIR=${INTERACTIVE_TRIALS_DIR:-"${INTERACTIVE_LOG_ROOT}/space_trials"}
INTERACTIVE_EVENTS_PATH=${INTERACTIVE_EVENTS_PATH:-"${INTERACTIVE_LOG_ROOT}/interactive_events.csv"}
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "  Extreme Stand Recovery MuJoCo Sim2Sim"
echo "============================================================"
if [[ -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint      : ${CHECKPOINT}"
else
    echo "Checkpoint      : not packaged (using verified use/ ONNX)"
fi
echo "Policy          : ${POLICY_PATH}"
echo "Profile / seed  : ${PROFILE} / ${SEED}"
echo "Duration        : ${SIMULATION_DURATION}s, GLFW=${USE_GLFW}, real-time=${REAL_TIME}"
echo "Rendering       : target=${RENDER_FPS} FPS, RTF status every ${REALTIME_STATUS_INTERVAL_S}s"
echo "Camera          : follow_robot=${FOLLOW_CAMERA} (False = free mouse camera)"
echo "Joint noise     : leg=${LEG_NOISE_RAD}, waist=${WAIST_NOISE_RAD}, arm=${ARM_NOISE_RAD} rad"
echo "Root noise      : rp=${ROOT_RP_NOISE_RAD}, yaw=${ROOT_YAW_NOISE_RAD} rad"
echo "Velocity noise  : joint=${JOINT_VEL_NOISE_RAD_S}, root_lin=${ROOT_LIN_VEL_NOISE_M_S}, root_ang=${ROOT_ANG_VEL_NOISE_RAD_S}"
echo "Wrench          : +/-${FORCE_MAX_N} N, +/-${TORQUE_MAX_NM} Nm, ${WRENCH_DURATION_S}s every ${WRENCH_INTERVAL_S}s"
echo "Large torso push: enabled=${LARGE_PUSH_ENABLE}, body=${LARGE_PUSH_BODY_NAME}, force=${LARGE_PUSH_FORCE_N}N, duration=${LARGE_PUSH_DURATION_S}s, time=${LARGE_PUSH_TIME_S}s, direction_index=${LARGE_PUSH_DIRECTION_INDEX}"
echo "Joint margin    : ${JOINT_LIMIT_MARGIN_RAD} rad (initial-state clipping only)"
echo "Pose recovery   : joint MAE <= ${JOINT_MAE_THRESHOLD_RAD} rad AND max <= ${JOINT_MAX_THRESHOLD_RAD} rad for ${RECOVERY_HOLD_TIME_S}s"
echo "Motion quality  : steady metrics start at ${STEADY_START_S}s; jerk weight=${JOINT_JERK_REWARD_WEIGHT}"
echo "Feet objective  : default distance, Gaussian variance=${FEET_GAUSSIAN_VARIANCE_M2} m^2"
echo "Feet reset test : start_random=${FOOT_SPACING_START_RANDOM}, delta=${FOOT_SPACING_MIN_DELTA_M}..${FOOT_SPACING_MAX_DELTA_M} m, tolerance=${FOOT_SPACING_RECOVERY_TOLERANCE_M} m"
echo "Policy command  : [0, 0, 0] (forced)"
echo "Action contract : all 29 actor outputs are used; action_override=false"
echo "Target limiter  : enabled=${TARGET_LIMITER_ENABLE}"
echo "  velocity      : leg=${TARGET_LEG_VELOCITY_LIMIT_RAD_S}, waist=${TARGET_WAIST_VELOCITY_LIMIT_RAD_S}, arm=${TARGET_ARM_VELOCITY_LIMIT_RAD_S} rad/s"
echo "  acceleration  : leg=${TARGET_LEG_ACCELERATION_LIMIT_RAD_S2}, waist=${TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2}, arm=${TARGET_ARM_ACCELERATION_LIMIT_RAD_S2} rad/s^2"
if is_true "${INTERACTIVE_ENABLE}"; then
    echo "GUI controls    : SPACE cycles +X/-X/+Y/-Y torso pushes, RANDOM POSE, RANDOM FOOT SPACING, DEFAULT; R=IMMEDIATE DEFAULT STANDING RESET; K=NEW RANDOM FOOT SPACING; F=OFF/ON random wrench; C=FREE/FOLLOW camera"
    echo "Interactive init: random_pose=${INTERACTIVE_POSE_START_RANDOM}, foot_spacing=${FOOT_SPACING_START_RANDOM}, wrench=${INTERACTIVE_WRENCH_START_ENABLED}"
fi
echo "Metrics         : ${METRICS_PATH}"
echo "Push plot       : ${PUSH_DIAGNOSTIC_PLOT_PATH}"
echo "Interactive log : enabled=${INTERACTIVE_DATA_LOG}, all=${INTERACTIVE_LOG_PATH}"
echo "Log session     : ${INTERACTIVE_LOG_SESSION}"
echo "Space trials    : ${INTERACTIVE_TRIALS_DIR}"
echo "Operator events : ${INTERACTIVE_EVENTS_PATH}"
echo "============================================================"

# 某些本机 MuJoCo/GLFW 组合会在 viewer 已完成、报告已落盘后的解释器退出阶段返回 139。
# 禁止生成大 core 文件；下面只在“GUI + rc=139 + metrics 已存在”三个条件同时满足时
# 将其识别为 viewer shutdown 问题，策略/仿真中途的任何失败仍保持非零退出。
ulimit -c 0 || true
set +e
G1_AMP_ARMHACK_STAND_ENABLE=False \
G1_AMP_ARMHACK_WALK_ENABLE=False \
G1_AMP_EXTREME_STAND_RECOVERY_ENABLE=True \
G1_AMP_EXTREME_STAND_RECOVERY_SEED="${SEED}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_ENABLE="${INTERACTIVE_ENABLE}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_POSE_START_RANDOM="${INTERACTIVE_POSE_START_RANDOM}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_WRENCH_START_ENABLED="${INTERACTIVE_WRENCH_START_ENABLED}" \
G1_AMP_EXTREME_STAND_FOOT_SPACING_START_RANDOM="${FOOT_SPACING_START_RANDOM}" \
G1_AMP_EXTREME_STAND_LARGE_PUSH_ENABLE="${LARGE_PUSH_ENABLE}" \
G1_AMP_EXTREME_STAND_LARGE_PUSH_BODY_NAME="${LARGE_PUSH_BODY_NAME}" \
G1_AMP_EXTREME_STAND_LARGE_PUSH_FORCE_N="${LARGE_PUSH_FORCE_N}" \
G1_AMP_EXTREME_STAND_LARGE_PUSH_DURATION_S="${LARGE_PUSH_DURATION_S}" \
G1_AMP_EXTREME_STAND_LARGE_PUSH_TIME_S="${LARGE_PUSH_TIME_S}" \
G1_AMP_EXTREME_STAND_LARGE_PUSH_DIRECTION_INDEX="${LARGE_PUSH_DIRECTION_INDEX}" \
G1_AMP_EXTREME_STAND_POST_PUSH_SETTLE_S="${POST_PUSH_SETTLE_S}" \
G1_AMP_EXTREME_STAND_TARGET_LIMITER_ENABLE="${TARGET_LIMITER_ENABLE}" \
G1_AMP_EXTREME_STAND_TARGET_LEG_VELOCITY_LIMIT_RAD_S="${TARGET_LEG_VELOCITY_LIMIT_RAD_S}" \
G1_AMP_EXTREME_STAND_TARGET_WAIST_VELOCITY_LIMIT_RAD_S="${TARGET_WAIST_VELOCITY_LIMIT_RAD_S}" \
G1_AMP_EXTREME_STAND_TARGET_ARM_VELOCITY_LIMIT_RAD_S="${TARGET_ARM_VELOCITY_LIMIT_RAD_S}" \
G1_AMP_EXTREME_STAND_TARGET_LEG_ACCELERATION_LIMIT_RAD_S2="${TARGET_LEG_ACCELERATION_LIMIT_RAD_S2}" \
G1_AMP_EXTREME_STAND_TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2="${TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2}" \
G1_AMP_EXTREME_STAND_TARGET_ARM_ACCELERATION_LIMIT_RAD_S2="${TARGET_ARM_ACCELERATION_LIMIT_RAD_S2}" \
G1_AMP_EXTREME_STAND_LEG_NOISE_RAD="${LEG_NOISE_RAD}" \
G1_AMP_EXTREME_STAND_WAIST_NOISE_RAD="${WAIST_NOISE_RAD}" \
G1_AMP_EXTREME_STAND_ARM_NOISE_RAD="${ARM_NOISE_RAD}" \
G1_AMP_EXTREME_STAND_JOINT_VEL_NOISE_RAD_S="${JOINT_VEL_NOISE_RAD_S}" \
G1_AMP_EXTREME_STAND_ROOT_RP_NOISE_RAD="${ROOT_RP_NOISE_RAD}" \
G1_AMP_EXTREME_STAND_ROOT_YAW_NOISE_RAD="${ROOT_YAW_NOISE_RAD}" \
G1_AMP_EXTREME_STAND_ROOT_LIN_VEL_NOISE_M_S="${ROOT_LIN_VEL_NOISE_M_S}" \
G1_AMP_EXTREME_STAND_ROOT_ANG_VEL_NOISE_RAD_S="${ROOT_ANG_VEL_NOISE_RAD_S}" \
G1_AMP_EXTREME_STAND_FORCE_MAX_N="${FORCE_MAX_N}" \
G1_AMP_EXTREME_STAND_TORQUE_MAX_NM="${TORQUE_MAX_NM}" \
G1_AMP_EXTREME_STAND_WRENCH_INTERVAL_S="${WRENCH_INTERVAL_S}" \
G1_AMP_EXTREME_STAND_WRENCH_DURATION_S="${WRENCH_DURATION_S}" \
G1_AMP_EXTREME_STAND_JOINT_LIMIT_MARGIN_RAD="${JOINT_LIMIT_MARGIN_RAD}" \
G1_AMP_EXTREME_STAND_JOINT_MAE_THRESHOLD_RAD="${JOINT_MAE_THRESHOLD_RAD}" \
G1_AMP_EXTREME_STAND_JOINT_MAX_THRESHOLD_RAD="${JOINT_MAX_THRESHOLD_RAD}" \
G1_AMP_EXTREME_STAND_HOLD_TIME_S="${RECOVERY_HOLD_TIME_S}" \
G1_AMP_EXTREME_STAND_FINAL_WINDOW_S="${RECOVERY_FINAL_WINDOW_S}" \
G1_AMP_EXTREME_STAND_STEADY_START_S="${STEADY_START_S}" \
G1_AMP_EXTREME_STAND_FEET_GAUSSIAN_VARIANCE_M2="${FEET_GAUSSIAN_VARIANCE_M2}" \
G1_AMP_EXTREME_STAND_JOINT_JERK_REWARD_WEIGHT="${JOINT_JERK_REWARD_WEIGHT}" \
G1_AMP_EXTREME_STAND_FOOT_SPACING_MIN_DELTA_M="${FOOT_SPACING_MIN_DELTA_M}" \
G1_AMP_EXTREME_STAND_FOOT_SPACING_MAX_DELTA_M="${FOOT_SPACING_MAX_DELTA_M}" \
G1_AMP_EXTREME_STAND_FOOT_SPACING_MAX_ROLL_OFFSET_RAD="${FOOT_SPACING_MAX_ROLL_OFFSET_RAD}" \
G1_AMP_EXTREME_STAND_FOOT_SPACING_SEARCH_SAMPLES="${FOOT_SPACING_SEARCH_SAMPLES}" \
G1_AMP_EXTREME_STAND_FOOT_SPACING_RECOVERY_TOLERANCE_M="${FOOT_SPACING_RECOVERY_TOLERANCE_M}" \
G1_AMP_EXTREME_STAND_MOTION_TRACE_PATH="${MOTION_TRACE_PATH}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_LOG_ENABLE="${INTERACTIVE_DATA_LOG}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_LOG_PATH="${INTERACTIVE_LOG_PATH}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_TRIALS_DIR="${INTERACTIVE_TRIALS_DIR}" \
G1_AMP_EXTREME_STAND_INTERACTIVE_EVENTS_PATH="${INTERACTIVE_EVENTS_PATH}" \
G1_AMP_RENDER_FPS="${RENDER_FPS}" \
G1_AMP_REALTIME_STATUS_INTERVAL_S="${REALTIME_STATUS_INTERVAL_S}" \
G1_AMP_FOLLOW_CAMERA_ENABLE="${FOLLOW_CAMERA}" \
FOLLOW_CAMERA_ENABLE="${FOLLOW_CAMERA}" \
UNITREE_PYTHON="${UNITREE_PYTHON}" \
POLICY_PATH="${POLICY_PATH}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" \
REAL_TIME="${REAL_TIME}" \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
CMD_INIT='[0.0,0.0,0.0]' \
RANDOM_COMMANDS=False \
COMMAND_MODE=independent \
COMMAND_RAMP=False \
TASK_TRACE_ENABLE=False \
TORSO_TRACE_ENABLE=True \
METRICS_PATH="${METRICS_PATH}" \
TORSO_TRACE_PATH="${TORSO_TRACE_PATH}" \
TASK_TRACE_PATH="${TASK_TRACE_PATH}" \
    bash "${ROOT_DIR}/scripts/sim2sim_g1_amp_mujoco.sh" "$@"
sim_rc=$?
set -e

if [[ ${sim_rc} -ne 0 ]]; then
    if is_true "${USE_GLFW}" && [[ ${sim_rc} -eq 139 && -s "${METRICS_PATH}" ]]; then
        echo "[WARN] MuJoCo/GLFW 在 viewer 关闭后返回 139；完整 metrics 已落盘，按可视化成功处理。" >&2
    else
        echo "Error: MuJoCo 仿真退出码=${sim_rc}，且不满足安全的 viewer-shutdown 兼容条件。" >&2
        exit "${sim_rc}"
    fi
fi

"${UNITREE_PYTHON}" - "${METRICS_PATH}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
stand = report.get("extreme_stand_recovery", {})
tracking = report.get("task_tracking", {})
if stand.get("action_override") is not False:
    raise SystemExit("MuJoCo report contract failed: action_override must be false")
command = [
    float(tracking.get("mean_command_lin_vel_x", math.nan)),
    float(tracking.get("mean_command_lin_vel_y", math.nan)),
    float(tracking.get("mean_command_yaw_rate", math.nan)),
]
if not all(math.isfinite(value) and abs(value) <= 1.0e-6 for value in command):
    raise SystemExit(f"MuJoCo report contract failed: command is not zero: {command}")
health = report.get("health", {})
recovery = stand.get("default_pose_recovery", {})
motion = stand.get("motion_quality", {})
jerk = motion.get("joint_jerk_rad_s3", {})
feet = motion.get("feet_planar_distance_m", {})
feet_recovery = stand.get("foot_spacing_recovery", {})
push_diagnostics = stand.get("large_push", {}).get("post_push_diagnostics", {})
print(
    "MuJoCo result: "
    f"healthy={health.get('healthy')} fallen={health.get('fallen')} "
    f"fall_time={health.get('fall_time')} score={report.get('score', {}).get('total_score')} "
    f"wrench_events={stand.get('wrench', {}).get('event_count')} "
    f"pose_recovered={recovery.get('pose_recovered')} "
    f"joint_mae={recovery.get('initial_joint_mae_rad')}->{recovery.get('final_joint_mae_rad')} "
    f"steady_jerk_rms={jerk.get('rms')} "
    f"feet_error_rms={feet.get('error_rms')} "
    f"feet_within_1cm={feet.get('within_1cm_fraction')} "
    f"feet_initial={feet_recovery.get('actual_initial_distance_m')} "
    f"feet_default={feet_recovery.get('default_distance_m')} "
    f"feet_recovered={feet_recovery.get('distance_recovered')} "
    f"feet_recovery_time={feet_recovery.get('recovery_time_s')} "
    f"large_push_diagnosis={push_diagnostics.get('diagnosis')} "
    f"large_push_flags={push_diagnostics.get('flags')} "
    f"large_push_ratios={push_diagnostics.get('post_over_pre')}"
)
PY

if [[ "${PROFILE}" == "large_push" || "${PROFILE}" == "interactive" ]]; then
    PLOT_PYTHON=${PLOT_PYTHON:-${ISAACLAB_PYTHON}}
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/g1-extreme-stand-matplotlib}" \
    "${PLOT_PYTHON}" "${ROOT_DIR}/scripts/plot_g1_extreme_stand_push_diagnostics.py" \
        --metrics "${METRICS_PATH}" \
        --motion-trace "${MOTION_TRACE_PATH}" \
        --output "${PUSH_DIAGNOSTIC_PLOT_PATH}"
    echo "Large-push diagnostic plot: ${PUSH_DIAGNOSTIC_PLOT_PATH}"
fi
