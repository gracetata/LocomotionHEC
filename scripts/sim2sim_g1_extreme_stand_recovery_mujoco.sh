#!/usr/bin/env bash
# MuJoCo sim2sim launcher and multi-profile suite for the final Pose V2 Extreme Stand policy.
#
# GUI interactive (SPACE toggles initial pose, F toggles random wrench):
#   PROFILE=interactive USE_GLFW=True SIMULATION_DURATION=300 \
#     bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
# Full suite (5 profiles x 3 seeds):
#   SUITE=True USE_GLFW=False \
#     bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_CHECKPOINT="${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt"
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
EXPORT_DIR=${EXPORT_DIR:-"$(dirname "${CHECKPOINT}")/exported_extreme_stand_recovery"}
DEFAULT_USE_ONNX="${ROOT_DIR}/use/extreme_stand_recovery_pose_v2_model2999.onnx"
POLICY_PATH=${POLICY_PATH:-"${DEFAULT_USE_ONNX}"}
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
SUITE=${SUITE:-False}
SUITE_PROFILES=${SUITE_PROFILES:-nominal,pose_recovery,recovery,robust,stress}
SUITE_SEEDS=${SUITE_SEEDS:-20260719,20260720,20260721}
SUITE_DURATION=${SUITE_DURATION:-12.0}
REQUIRE_PASS=${REQUIRE_PASS:-False}
RESULTS_ROOT=${RESULTS_ROOT:-"${EXPORT_DIR}/mujoco_tests/manual"}

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
    if [[ "${POLICY_PATH}" == "${DEFAULT_USE_ONNX}" ]]; then
        cp --preserve=mode,timestamps "${EXPORT_DIR}/policy.onnx" "${DEFAULT_USE_ONNX}"
    fi
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
    *)
        echo "Error: PROFILE 必须是 interactive、nominal、pose_recovery、recovery、robust 或 stress；当前为 ${PROFILE}" >&2
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
INTERACTIVE_ENABLE=${INTERACTIVE_ENABLE:-${default_interactive}}
INTERACTIVE_POSE_START_RANDOM=${INTERACTIVE_POSE_START_RANDOM:-${default_pose_start_random}}
INTERACTIVE_WRENCH_START_ENABLED=${INTERACTIVE_WRENCH_START_ENABLED:-${default_wrench_start_enabled}}

RUN_DIR="${RESULTS_ROOT}/${PROFILE}/seed_${SEED}"
METRICS_PATH=${METRICS_PATH:-"${RUN_DIR}/metrics.json"}
TORSO_TRACE_PATH=${TORSO_TRACE_PATH:-"${RUN_DIR}/torso_trace.csv"}
TASK_TRACE_PATH=${TASK_TRACE_PATH:-"${RUN_DIR}/task_trace.csv"}
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
echo "Joint noise     : leg=${LEG_NOISE_RAD}, waist=${WAIST_NOISE_RAD}, arm=${ARM_NOISE_RAD} rad"
echo "Root noise      : rp=${ROOT_RP_NOISE_RAD}, yaw=${ROOT_YAW_NOISE_RAD} rad"
echo "Velocity noise  : joint=${JOINT_VEL_NOISE_RAD_S}, root_lin=${ROOT_LIN_VEL_NOISE_M_S}, root_ang=${ROOT_ANG_VEL_NOISE_RAD_S}"
echo "Wrench          : +/-${FORCE_MAX_N} N, +/-${TORQUE_MAX_NM} Nm, ${WRENCH_DURATION_S}s every ${WRENCH_INTERVAL_S}s"
echo "Joint margin    : ${JOINT_LIMIT_MARGIN_RAD} rad (initial-state clipping only)"
echo "Pose recovery   : joint MAE <= ${JOINT_MAE_THRESHOLD_RAD} rad AND max <= ${JOINT_MAX_THRESHOLD_RAD} rad for ${RECOVERY_HOLD_TIME_S}s"
echo "Policy command  : [0, 0, 0] (forced)"
echo "Action contract : all 29 actor outputs are used; action_override=false"
if is_true "${INTERACTIVE_ENABLE}"; then
    echo "GUI controls    : SPACE=DEFAULT/RANDOM initial pose; F=OFF/ON random wrench"
    echo "Interactive init: random_pose=${INTERACTIVE_POSE_START_RANDOM}, wrench=${INTERACTIVE_WRENCH_START_ENABLED}"
fi
echo "Metrics         : ${METRICS_PATH}"
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
print(
    "MuJoCo result: "
    f"healthy={health.get('healthy')} fallen={health.get('fallen')} "
    f"fall_time={health.get('fall_time')} score={report.get('score', {}).get('total_score')} "
    f"wrench_events={stand.get('wrench', {}).get('event_count')} "
    f"pose_recovered={recovery.get('pose_recovered')} "
    f"joint_mae={recovery.get('initial_joint_mae_rad')}->{recovery.get('final_joint_mae_rad')}"
)
PY
