#!/usr/bin/env bash
# Run interactive Unitree G1 high-level FSM handoff, manual WASD, optional rt/arm_sdk arm swing,
# or guarded ZeroTorque full-body rt/lowcmd takeover.
# Usage:
#   NET=enp11s0 PROBE_ONLY=True bash scripts/g1_loco_fsm_wasd.sh
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 VERIFY_ZERO_COMMAND=True PROBE_ONLY=True bash scripts/g1_loco_fsm_wasd.sh
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 bash scripts/g1_loco_fsm_wasd.sh
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 只读检查：确认 IP/DDS/lowstate/MotionSwitcher/robot_state/Loco RPC。
NET=enp11s0 PROBE_ONLY=True bash scripts/g1_loco_fsm_wasd.sh

# 2. 安全零速度命令验证：只发送 SetVelocity(0,0,0)，不进入站立流程。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 PROBE_ONLY=True VERIFY_ZERO_COMMAND=True START_SERVICES=ai_sport \
    bash scripts/g1_loco_fsm_wasd.sh

# 3. 完整交互流程：每一步按空格确认，WASD 锁存并持续重发，q 退出 WASD 后继续右臂持续摆动。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 SPEED_MPS=0.5 COMMAND_DURATION_S=1.0 \
    ARM_JOINT_SET=right_arm5 ARM_AMPLITUDE_SCALE=1.0 ARM_SWING_S=0 \
    bash scripts/g1_loco_fsm_wasd.sh

# 4. 非交互真机诊断：自动前进 3s，用腿部 q/dq 读数验收，然后右臂限时大幅摆动。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 MANUAL_STEP_CONFIRM=False AUTO_FORWARD_TEST=True \
    AUTO_FORWARD_REQUIRE_MOTION=True ARM_SWING_S=4.0 ARM_AMPLITUDE_SCALE=1.0 \
    bash scripts/g1_loco_fsm_wasd.sh

# 4b. 只测摆臂：跳过前进，进入 BalanceStand 后做大幅右臂摆动并用 lowstate 验收实际关节运动。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 MANUAL_STEP_CONFIRM=False ARM_ONLY_TEST=True \
    ARM_SWING_S=6.0 ARM_REQUIRE_SWING_MOTION=True ARM_MIN_SWING_DELTA_RAD=0.20 \
    bash scripts/g1_loco_fsm_wasd.sh

# 4c. 单关节摆臂：只让右肩 pitch 前后摆动，其它腰/左臂/右臂关节全部保持进入前姿态。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 MANUAL_STEP_CONFIRM=False ARM_ONLY_TEST=True \
    ARM_JOINT_SET=right_shoulder_pitch ARM_SWING_S=16.0 ARM_REQUIRE_SWING_MOTION=True \
    ARM_SWING_FREQUENCY_HZ=0.60 \
    ARM_HOLD_KP=120 ARM_HOLD_KD=5 ARM_WAIST_HOLD_KP=160 ARM_WAIST_HOLD_KD=6 \
    ARM_MIN_SWING_DELTA_RAD=0.20 bash scripts/g1_loco_fsm_wasd.sh
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 MANUAL_STEP_CONFIRM=False ARM_ONLY_TEST=True \
    ARM_JOINT_SET=right_shoulder_pitch ARM_SWING_S=16.0 ARM_REQUIRE_SWING_MOTION=True \
    ARM_SWING_FREQUENCY_HZ=0.05 \
    ARM_HOLD_KP=120 ARM_HOLD_KD=5 ARM_WAIST_HOLD_KP=160 ARM_WAIST_HOLD_KD=6 \
    ARM_MIN_SWING_DELTA_RAD=0.20 bash scripts/g1_loco_fsm_wasd.sh
# 5. 29DoF 右手腕也参与的持续测试；首次使用仍建议吊装并保持幅度 1.0 或更低。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 ARM_JOINT_SET=right_arm7 ARM_AMPLITUDE_SCALE=1.0 ARM_SWING_S=0 \
    bash scripts/g1_loco_fsm_wasd.sh

# 6. 主运控 -> ZeroTorque 瞬态全身低层接管 -> 双膝微弯；全程按空格推进。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 RUN_ZERO_TORQUE_TAKEOVER=True \
    RUN_ARM_SWING_AFTER_WASD=False TAKEOVER_KNEE_BEND_RAD=0.06 \
    bash scripts/g1_loco_fsm_wasd.sh

# 7. 直接进入 ZeroTorque/FSM 0：腿部零力矩，通过 rt/lowcmd 电机命令摆右肩 pitch，其它 arm_sdk 范围关节按当前姿态 hold。
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 DIRECT_ZERO_ARM_MOTOR_TEST=True RUN_ARM_SWING_AFTER_WASD=False \
    ARM_JOINT_SET=right_shoulder_pitch ARM_SWING_S=16.0 ARM_REQUIRE_SWING_MOTION=True \
    ARM_SWING_FREQUENCY_HZ=0.05 \
    ARM_HOLD_KP=120 ARM_HOLD_KD=5 ARM_WAIST_HOLD_KP=160 ARM_WAIST_HOLD_KD=6 \
    ARM_MIN_SWING_DELTA_RAD=0.20 bash scripts/g1_loco_fsm_wasd.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}

# === 环境变量说明 ===
# NET                         : 必填。Unitree DDS 网卡名，例如 enp11s0。
# ROBOT_IP                    : 机器人机载电脑 IP，默认 192.168.123.161。
# LOCO_SERVICE                : G1 Loco RPC 服务名，默认 sport。
# START_SERVICES              : 逗号分隔的 robot_state 服务名；非 PROBE_ONLY 默认 ai_sport。
# PROBE_ONLY                  : True 只探测连接，不进入 FSM/WASD。
# VERIFY_ZERO_COMMAND         : True 发送 SetVelocity(0,0,0) 验证命令下发。
# CONFIRM_REAL_ROBOT          : 任何命令模式必须等于 I_UNDERSTAND。
# MANUAL_STEP_CONFIRM         : True 时每个状态切换/关键命令前等待空格，默认 True。
# RPC_TIMEOUT_S               : RPC 调用超时，默认 5.0 s。
# SPORT_STATE_TOPIC           : 高层运控状态 DDS topic，默认 rt/lf/sportmodestate。
# SPORT_STATE_TIMEOUT_S       : 等待 SportModeState 的时间，默认 1.5 s。
# ALLOW_MISSING_SPORT_STATE   : True 时即使没有 SportModeState 也继续，默认 False。
# SPEED_MPS                   : WASD 线速度幅值，默认 0.5 m/s，便于观察真机移动。
# COMMAND_DURATION_S          : 每次速度指令持续时间，默认 1.0 s；脚本会按 WASD_REPEAT_S 持续重发。
# WASD_REPEAT_S               : WASD 锁存速度的重发周期，默认 0.25 s。
# KEYBOARD_SECONDS            : WASD 控制时长，0 表示直到按 q。
# AUTO_FORWARD_TEST           : True 时跳过人工 WASD，自动前进并用腿部读数验收。
# AUTO_FORWARD_SPEED_MPS      : 自动前进测试速度，默认 0.5 m/s。
# AUTO_FORWARD_DURATION_S     : 自动前进测试时长，默认 3.0 s。
# AUTO_FORWARD_REQUIRE_MOTION : True 时未观测到腿部动作则报错，默认 False。
# AUTO_FORWARD_CONTINUOUS_GAIT: True 时测试前开启 ContinuousGait，默认 False。
# HANDOFF_FLOW                : squat2stand_auto 或 standup_start，默认 standup_start。
# LOCOMOTION_FSM_IDS          : 认为已进入主运控的 FSM 列表，默认 500,801,802。
# START_FSM_IDS               : StandUp 后按顺序尝试的主运控 FSM，默认 500,801,802。
# SET_LOCOMOTION_AFTER_STANDUP: True 时，如果未观察到主运控候选 FSM，则重新请求 START_FSM_IDS。默认 False。
# ALLOW_MISSING_FSM_500       : True 时即使未确认主运控候选 FSM 也进入 WASD，默认 False。
# RUN_ZERO_TORQUE_TAKEOVER    : True 时主运控后执行 rt/lowcmd 全身预热、FSM 0 接管和双膝微弯流程。
# DIRECT_ZERO_ARM_MOTOR_TEST  : True 时跳过主运控交接，直接进入 FSM 0，并通过 rt/lowcmd 控制所选手臂关节。
# TAKEOVER_LOWCMD_TOPIC       : 全身低层接管 DDS topic，默认 rt/lowcmd。
# TAKEOVER_CONTROL_DT_S       : 全身接管发布周期，默认 0.002 s (500Hz)。
# TAKEOVER_WARMUP_MIN_CYCLES  : ZeroTorque 前至少预热发布次数，默认 50。
# TAKEOVER_ALIGNMENT_S        : 主运控零速度驻车后锁存 q_t0 前等待时间，默认 1.0 s。
# TAKEOVER_TAU_FF_MODE        : latched 使用切换前 tau_est 前馈；zero 不使用前馈，默认 latched。
# TAKEOVER_TAU_FF_SCALE       : 锁存前馈力矩缩放，默认 1.0。
# TAKEOVER_MAX_ABS_TAU_FF     : 单关节前馈力矩绝对值上限，默认 120。
# TAKEOVER_LEG_KP/KD          : 腿部接管 PD，默认 25/1.0，主要做小误差微调。
# TAKEOVER_WAIST_KP/KD        : 腰部接管 PD，默认 20/1.0。
# TAKEOVER_HOLD_KP/KD         : 上肢接管 PD，默认 15/0.8。
# TAKEOVER_KNEE_BEND_RAD      : 双膝微弯目标增量，默认 0.06 rad；正值为弯曲方向。
# TAKEOVER_KNEE_BEND_S        : 双膝微弯插值时间，默认 1.0 s。
# TAKEOVER_RECOVER_FSM_ID     : 停止后恢复的内置 FSM，默认 1 (Damp)；可改 4 进入 StandUp。
# FINAL_DAMP_ON_EXIT          : True 时脚本最外层 finally 会尽力请求 SetFsmId(1)，默认 True。
# RUN_ARM_SWING_AFTER_WASD    : True 时 WASD 完成后 StopMove -> BalanceStand -> rt/arm_sdk 右臂摆动，默认 True。
# ARM_ONLY_TEST               : True 时跳过 FSM/WASD/前进，只从当前内置站立模式运行 arm_sdk 摆臂诊断。
# BALANCE_STAND_AFTER_WASD    : True 时摆臂前发送 BalanceStand()，默认 True。
# ARM_TOPIC                   : 上肢 DDS topic，默认 rt/arm_sdk。
# ARM_JOINT_SET               : right_arm5/right_arm7/arm5/arm7/right_shoulder_pitch 等，默认 right_arm5。
# ARM_HOLD_NON_TARGET         : True 时非目标 arm_sdk 关节保持启用瞬间姿态，默认 True。
# ARM_ALLOWED_FSM_IDS         : 允许执行 arm_sdk 的 FSM，默认 4,500,501,801,802。
# ARM_REQUIRE_STATIC_FSM_MODE : True 时要求 fsm_mode=0，默认 True。
# ARM_AMPLITUDE_SCALE         : 摆臂幅度缩放，默认 1.0；右臂基础幅度已按限位放大。
# ARM_LIMIT_MARGIN_RAD        : 目标离 XML 关节限位的最小余量，默认 0.15 rad。
# ARM_MAX_AMPLITUDE_RAD       : 单关节最大正弦幅度，默认 0.9 rad。
# ARM_HOLD_ERROR_THRESHOLD_RAD: 非目标 hold 关节允许偏离启用姿态的阈值，默认 0.15 rad。
# ARM_WAIST_HOLD_KP/KD        : 腰部 hold 专用刚度/阻尼，默认 160/6，高于其它 hold 关节。
# ARM_HOLD_GRACE_S            : arm_sdk 接管初期 hold 误差宽限时间，默认 1.5 s。
# ARM_ABORT_ON_ANY_HOLD_ERROR : True 时任何非目标 hold 关节超阈值都中止；默认 False，仅腰部严格中止。
# ARM_REQUIRE_SWING_MOTION    : True 时要求 lowstate 观测到目标手臂实际运动幅度达到阈值。
# ARM_MIN_SWING_DELTA_RAD     : ARM_REQUIRE_SWING_MOTION 的实际关节运动阈值，默认 0.20 rad。
# ARM_SWING_S                 : 中间正弦摆臂持续时间，默认 0 表示持续摆动，按空格/q 停止。
# SERVICE_START_WAIT_S        : START_SERVICES 启动后等待时间，默认 1.0 s。
# DRY_RUN                     : True 打印解析后的命令，不连接真机。
# =====================

NET=${NET:-}
ROBOT_IP=${ROBOT_IP:-192.168.123.161}
LOCO_SERVICE=${LOCO_SERVICE:-sport}
START_SERVICES=${START_SERVICES:-}
PROBE_ONLY=${PROBE_ONLY:-False}
VERIFY_ZERO_COMMAND=${VERIFY_ZERO_COMMAND:-False}
CONFIRM_REAL_ROBOT=${CONFIRM_REAL_ROBOT:-}
MANUAL_STEP_CONFIRM=${MANUAL_STEP_CONFIRM:-True}
PING_ROBOT=${PING_ROBOT:-True}
DRY_RUN=${DRY_RUN:-False}

LOWSTATE_TIMEOUT_S=${LOWSTATE_TIMEOUT_S:-3.0}
RPC_TIMEOUT_S=${RPC_TIMEOUT_S:-5.0}
SPORT_STATE_TOPIC=${SPORT_STATE_TOPIC:-rt/lf/sportmodestate}
SPORT_STATE_TIMEOUT_S=${SPORT_STATE_TIMEOUT_S:-1.5}
ALLOW_MISSING_SPORT_STATE=${ALLOW_MISSING_SPORT_STATE:-False}
SERVICE_START_WAIT_S=${SERVICE_START_WAIT_S:-1.0}
ZERO_COMMAND_DURATION_S=${ZERO_COMMAND_DURATION_S:-0.2}
FINAL_DAMP_ON_EXIT=${FINAL_DAMP_ON_EXIT:-True}
SAFE_DAMP_RPC_RETRIES=${SAFE_DAMP_RPC_RETRIES:-3}
SAFE_DAMP_RETRY_INTERVAL_S=${SAFE_DAMP_RETRY_INTERVAL_S:-0.2}
DAMP_HOLD_S=${DAMP_HOLD_S:-0.8}
HANDOFF_FLOW=${HANDOFF_FLOW:-standup_start}
LOCOMOTION_FSM_IDS=${LOCOMOTION_FSM_IDS:-500,801,802}
START_FSM_IDS=${START_FSM_IDS:-500,801,802}
STANDUP_WAIT_S=${STANDUP_WAIT_S:-6.0}
FSM_POLL_TIMEOUT_S=${FSM_POLL_TIMEOUT_S:-5.0}
SET_LOCOMOTION_AFTER_STANDUP=${SET_LOCOMOTION_AFTER_STANDUP:-False}
ALLOW_MISSING_FSM_500=${ALLOW_MISSING_FSM_500:-False}
RUN_ZERO_TORQUE_TAKEOVER=${RUN_ZERO_TORQUE_TAKEOVER:-False}
DIRECT_ZERO_ARM_MOTOR_TEST=${DIRECT_ZERO_ARM_MOTOR_TEST:-False}
TAKEOVER_LOWCMD_TOPIC=${TAKEOVER_LOWCMD_TOPIC:-rt/lowcmd}
TAKEOVER_CONTROL_DT_S=${TAKEOVER_CONTROL_DT_S:-0.002}
TAKEOVER_WARMUP_MIN_CYCLES=${TAKEOVER_WARMUP_MIN_CYCLES:-50}
TAKEOVER_WARMUP_TIMEOUT_S=${TAKEOVER_WARMUP_TIMEOUT_S:-1.0}
TAKEOVER_ALIGNMENT_S=${TAKEOVER_ALIGNMENT_S:-1.0}
TAKEOVER_POST_ZERO_SETTLE_S=${TAKEOVER_POST_ZERO_SETTLE_S:-0.2}
TAKEOVER_ZERO_FSM_TIMEOUT_S=${TAKEOVER_ZERO_FSM_TIMEOUT_S:-2.0}
TAKEOVER_MONITOR_PERIOD_S=${TAKEOVER_MONITOR_PERIOD_S:-1.0}
TAKEOVER_TAU_FF_MODE=${TAKEOVER_TAU_FF_MODE:-latched}
TAKEOVER_TAU_FF_SCALE=${TAKEOVER_TAU_FF_SCALE:-1.0}
TAKEOVER_MAX_ABS_TAU_FF=${TAKEOVER_MAX_ABS_TAU_FF:-120.0}
TAKEOVER_LEG_KP=${TAKEOVER_LEG_KP:-25.0}
TAKEOVER_LEG_KD=${TAKEOVER_LEG_KD:-1.0}
TAKEOVER_WAIST_KP=${TAKEOVER_WAIST_KP:-20.0}
TAKEOVER_WAIST_KD=${TAKEOVER_WAIST_KD:-1.0}
TAKEOVER_HOLD_KP=${TAKEOVER_HOLD_KP:-15.0}
TAKEOVER_HOLD_KD=${TAKEOVER_HOLD_KD:-0.8}
TAKEOVER_KNEE_BEND_RAD=${TAKEOVER_KNEE_BEND_RAD:-0.06}
TAKEOVER_KNEE_BEND_S=${TAKEOVER_KNEE_BEND_S:-1.0}
TAKEOVER_AUTO_STOP_HOLD_S=${TAKEOVER_AUTO_STOP_HOLD_S:-1.0}
TAKEOVER_REQUIRE_KNEE_MOTION=${TAKEOVER_REQUIRE_KNEE_MOTION:-True}
TAKEOVER_MIN_KNEE_DELTA_RAD=${TAKEOVER_MIN_KNEE_DELTA_RAD:-0.02}
TAKEOVER_RECOVER_FSM_ID=${TAKEOVER_RECOVER_FSM_ID:-1}
TAKEOVER_RECOVER_CONFIRM_FSM_IDS=${TAKEOVER_RECOVER_CONFIRM_FSM_IDS:-1}
TAKEOVER_RECOVER_TIMEOUT_S=${TAKEOVER_RECOVER_TIMEOUT_S:-5.0}
TAKEOVER_RECOVER_HOLD_S=${TAKEOVER_RECOVER_HOLD_S:-1.0}
TAKEOVER_EMERGENCY_DAMPING_S=${TAKEOVER_EMERGENCY_DAMPING_S:-1.0}
TAKEOVER_EMERGENCY_DAMPING_KD=${TAKEOVER_EMERGENCY_DAMPING_KD:-8.0}
TAKEOVER_MAX_ABS_ROLL_RAD=${TAKEOVER_MAX_ABS_ROLL_RAD:-0.35}
TAKEOVER_MAX_ABS_PITCH_RAD=${TAKEOVER_MAX_ABS_PITCH_RAD:-0.35}
SPEED_MPS=${SPEED_MPS:-0.5}
COMMAND_DURATION_S=${COMMAND_DURATION_S:-1.0}
KEYBOARD_POLL_S=${KEYBOARD_POLL_S:-0.05}
KEYBOARD_SECONDS=${KEYBOARD_SECONDS:-0.0}
WASD_REPEAT_S=${WASD_REPEAT_S:-0.25}
AUTO_FORWARD_TEST=${AUTO_FORWARD_TEST:-False}
AUTO_FORWARD_SPEED_MPS=${AUTO_FORWARD_SPEED_MPS:-0.5}
AUTO_FORWARD_DURATION_S=${AUTO_FORWARD_DURATION_S:-3.0}
AUTO_COMMAND_PERIOD_S=${AUTO_COMMAND_PERIOD_S:-0.2}
AUTO_FORWARD_MIN_LEG_DQ=${AUTO_FORWARD_MIN_LEG_DQ:-0.10}
AUTO_FORWARD_MIN_LEG_DELTA_Q=${AUTO_FORWARD_MIN_LEG_DELTA_Q:-0.02}
AUTO_FORWARD_REQUIRE_MOTION=${AUTO_FORWARD_REQUIRE_MOTION:-False}
AUTO_FORWARD_CONTINUOUS_GAIT=${AUTO_FORWARD_CONTINUOUS_GAIT:-False}
RUN_ARM_SWING_AFTER_WASD=${RUN_ARM_SWING_AFTER_WASD:-True}
ARM_ONLY_TEST=${ARM_ONLY_TEST:-False}
BALANCE_STAND_AFTER_WASD=${BALANCE_STAND_AFTER_WASD:-True}
POST_WASD_STABILIZE_S=${POST_WASD_STABILIZE_S:-1.0}
ARM_TOPIC=${ARM_TOPIC:-rt/arm_sdk}
ARM_JOINT_SET=${ARM_JOINT_SET:-right_arm5}
ARM_HOLD_NON_TARGET=${ARM_HOLD_NON_TARGET:-True}
ARM_ALLOWED_FSM_IDS=${ARM_ALLOWED_FSM_IDS:-4,500,501,801,802}
ARM_REQUIRE_STATIC_FSM_MODE=${ARM_REQUIRE_STATIC_FSM_MODE:-True}
ARM_FSM_TIMEOUT_S=${ARM_FSM_TIMEOUT_S:-3.0}
ARM_CONTROL_DT_S=${ARM_CONTROL_DT_S:-0.02}
ARM_RAMP_S=${ARM_RAMP_S:-1.0}
ARM_SWING_S=${ARM_SWING_S:-0.0}
ARM_RETURN_S=${ARM_RETURN_S:-1.0}
ARM_RELEASE_S=${ARM_RELEASE_S:-0.8}
ARM_SWING_FREQUENCY_HZ=${ARM_SWING_FREQUENCY_HZ:-0.35}
ARM_AMPLITUDE_SCALE=${ARM_AMPLITUDE_SCALE:-1.0}
ARM_KP=${ARM_KP:-40.0}
ARM_KD=${ARM_KD:-1.0}
ARM_HOLD_KP=${ARM_HOLD_KP:-120.0}
ARM_HOLD_KD=${ARM_HOLD_KD:-5.0}
ARM_WAIST_HOLD_KP=${ARM_WAIST_HOLD_KP:-160.0}
ARM_WAIST_HOLD_KD=${ARM_WAIST_HOLD_KD:-6.0}
ARM_HOLD_ERROR_THRESHOLD_RAD=${ARM_HOLD_ERROR_THRESHOLD_RAD:-0.15}
ARM_HOLD_GRACE_S=${ARM_HOLD_GRACE_S:-1.5}
ARM_ABORT_ON_ANY_HOLD_ERROR=${ARM_ABORT_ON_ANY_HOLD_ERROR:-False}
ARM_REQUIRE_SWING_MOTION=${ARM_REQUIRE_SWING_MOTION:-False}
ARM_MIN_SWING_DELTA_RAD=${ARM_MIN_SWING_DELTA_RAD:-0.20}
ARM_LIMIT_MARGIN_RAD=${ARM_LIMIT_MARGIN_RAD:-0.15}
ARM_MAX_AMPLITUDE_RAD=${ARM_MAX_AMPLITUDE_RAD:-0.9}
ARM_MAX_ABS_ROLL_RAD=${ARM_MAX_ABS_ROLL_RAD:-0.35}
ARM_MAX_ABS_PITCH_RAD=${ARM_MAX_ABS_PITCH_RAD:-0.35}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

if ! is_true "${PROBE_ONLY}" && [[ -z "${START_SERVICES}" ]]; then
    START_SERVICES=ai_sport
fi

if [[ -z "${NET}" ]]; then
    echo "Error: set NET to the Unitree network interface, e.g. NET=enp11s0." >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi

WILL_SEND_COMMAND=False
if is_true "${VERIFY_ZERO_COMMAND}" || ! is_true "${PROBE_ONLY}" || [[ -n "${START_SERVICES}" ]]; then
    WILL_SEND_COMMAND=True
fi
if ! is_true "${DRY_RUN}" && is_true "${WILL_SEND_COMMAND}" && [[ "${CONFIRM_REAL_ROBOT}" != "I_UNDERSTAND" ]]; then
    echo "Error: refusing to send robot commands without CONFIRM_REAL_ROBOT=I_UNDERSTAND." >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_COMMAND=(
    "${UNITREE_PYTHON}"
    "${UNITREE_DIR}/deploy/deploy_real/g1_loco_fsm_wasd.py"
    "${NET}"
    --loco-service "${LOCO_SERVICE}"
    --confirm-real-robot "${CONFIRM_REAL_ROBOT}"
    --rpc-timeout-s "${RPC_TIMEOUT_S}"
    --service-start-wait-s "${SERVICE_START_WAIT_S}"
    --lowstate-timeout-s "${LOWSTATE_TIMEOUT_S}"
    --sport-state-topic "${SPORT_STATE_TOPIC}"
    --sport-state-timeout-s "${SPORT_STATE_TIMEOUT_S}"
    --zero-command-duration-s "${ZERO_COMMAND_DURATION_S}"
    --safe-damp-rpc-retries "${SAFE_DAMP_RPC_RETRIES}"
    --safe-damp-retry-interval-s "${SAFE_DAMP_RETRY_INTERVAL_S}"
    --damp-hold-s "${DAMP_HOLD_S}"
    --handoff-flow "${HANDOFF_FLOW}"
    --locomotion-fsm-ids "${LOCOMOTION_FSM_IDS}"
    --start-fsm-ids "${START_FSM_IDS}"
    --standup-wait-s "${STANDUP_WAIT_S}"
    --fsm-poll-timeout-s "${FSM_POLL_TIMEOUT_S}"
    --takeover-lowcmd-topic "${TAKEOVER_LOWCMD_TOPIC}"
    --takeover-control-dt-s "${TAKEOVER_CONTROL_DT_S}"
    --takeover-warmup-min-cycles "${TAKEOVER_WARMUP_MIN_CYCLES}"
    --takeover-warmup-timeout-s "${TAKEOVER_WARMUP_TIMEOUT_S}"
    --takeover-alignment-s "${TAKEOVER_ALIGNMENT_S}"
    --takeover-post-zero-settle-s "${TAKEOVER_POST_ZERO_SETTLE_S}"
    --takeover-zero-fsm-timeout-s "${TAKEOVER_ZERO_FSM_TIMEOUT_S}"
    --takeover-monitor-period-s "${TAKEOVER_MONITOR_PERIOD_S}"
    --takeover-tau-ff-mode "${TAKEOVER_TAU_FF_MODE}"
    --takeover-tau-ff-scale "${TAKEOVER_TAU_FF_SCALE}"
    --takeover-max-abs-tau-ff "${TAKEOVER_MAX_ABS_TAU_FF}"
    --takeover-leg-kp "${TAKEOVER_LEG_KP}"
    --takeover-leg-kd "${TAKEOVER_LEG_KD}"
    --takeover-waist-kp "${TAKEOVER_WAIST_KP}"
    --takeover-waist-kd "${TAKEOVER_WAIST_KD}"
    --takeover-hold-kp "${TAKEOVER_HOLD_KP}"
    --takeover-hold-kd "${TAKEOVER_HOLD_KD}"
    --takeover-knee-bend-rad "${TAKEOVER_KNEE_BEND_RAD}"
    --takeover-knee-bend-s "${TAKEOVER_KNEE_BEND_S}"
    --takeover-auto-stop-hold-s "${TAKEOVER_AUTO_STOP_HOLD_S}"
    --takeover-min-knee-delta-rad "${TAKEOVER_MIN_KNEE_DELTA_RAD}"
    --takeover-recover-fsm-id "${TAKEOVER_RECOVER_FSM_ID}"
    --takeover-recover-confirm-fsm-ids "${TAKEOVER_RECOVER_CONFIRM_FSM_IDS}"
    --takeover-recover-timeout-s "${TAKEOVER_RECOVER_TIMEOUT_S}"
    --takeover-recover-hold-s "${TAKEOVER_RECOVER_HOLD_S}"
    --takeover-emergency-damping-s "${TAKEOVER_EMERGENCY_DAMPING_S}"
    --takeover-emergency-damping-kd "${TAKEOVER_EMERGENCY_DAMPING_KD}"
    --takeover-max-abs-roll-rad "${TAKEOVER_MAX_ABS_ROLL_RAD}"
    --takeover-max-abs-pitch-rad "${TAKEOVER_MAX_ABS_PITCH_RAD}"
    --speed-mps "${SPEED_MPS}"
    --command-duration-s "${COMMAND_DURATION_S}"
    --keyboard-poll-s "${KEYBOARD_POLL_S}"
    --keyboard-seconds "${KEYBOARD_SECONDS}"
    --wasd-repeat-s "${WASD_REPEAT_S}"
    --auto-forward-speed-mps "${AUTO_FORWARD_SPEED_MPS}"
    --auto-forward-duration-s "${AUTO_FORWARD_DURATION_S}"
    --auto-command-period-s "${AUTO_COMMAND_PERIOD_S}"
    --auto-forward-min-leg-dq "${AUTO_FORWARD_MIN_LEG_DQ}"
    --auto-forward-min-leg-delta-q "${AUTO_FORWARD_MIN_LEG_DELTA_Q}"
    --post-wasd-stabilize-s "${POST_WASD_STABILIZE_S}"
    --arm-topic "${ARM_TOPIC}"
    --arm-joint-set "${ARM_JOINT_SET}"
    --arm-allowed-fsm-ids "${ARM_ALLOWED_FSM_IDS}"
    --arm-fsm-timeout-s "${ARM_FSM_TIMEOUT_S}"
    --arm-control-dt-s "${ARM_CONTROL_DT_S}"
    --arm-ramp-s "${ARM_RAMP_S}"
    --arm-swing-s "${ARM_SWING_S}"
    --arm-return-s "${ARM_RETURN_S}"
    --arm-release-s "${ARM_RELEASE_S}"
    --arm-swing-frequency-hz "${ARM_SWING_FREQUENCY_HZ}"
    --arm-amplitude-scale "${ARM_AMPLITUDE_SCALE}"
    --arm-kp "${ARM_KP}"
    --arm-kd "${ARM_KD}"
    --arm-hold-kp "${ARM_HOLD_KP}"
    --arm-hold-kd "${ARM_HOLD_KD}"
    --arm-waist-hold-kp "${ARM_WAIST_HOLD_KP}"
    --arm-waist-hold-kd "${ARM_WAIST_HOLD_KD}"
    --arm-hold-error-threshold-rad "${ARM_HOLD_ERROR_THRESHOLD_RAD}"
    --arm-hold-grace-s "${ARM_HOLD_GRACE_S}"
    --arm-min-swing-delta-rad "${ARM_MIN_SWING_DELTA_RAD}"
    --arm-limit-margin-rad "${ARM_LIMIT_MARGIN_RAD}"
    --arm-max-amplitude-rad "${ARM_MAX_AMPLITUDE_RAD}"
    --arm-max-abs-roll-rad "${ARM_MAX_ABS_ROLL_RAD}"
    --arm-max-abs-pitch-rad "${ARM_MAX_ABS_PITCH_RAD}"
)

if is_true "${PROBE_ONLY}"; then
    RUN_COMMAND+=(--probe-only)
fi
if is_true "${VERIFY_ZERO_COMMAND}"; then
    RUN_COMMAND+=(--verify-zero-command)
fi
if is_true "${MANUAL_STEP_CONFIRM}"; then
    RUN_COMMAND+=(--manual-step-confirm)
fi
if is_true "${FINAL_DAMP_ON_EXIT}"; then
    RUN_COMMAND+=(--safe-damp-on-exit)
fi
if is_true "${AUTO_FORWARD_TEST}"; then
    RUN_COMMAND+=(--auto-forward-test)
fi
if is_true "${AUTO_FORWARD_REQUIRE_MOTION}"; then
    RUN_COMMAND+=(--auto-forward-require-motion)
fi
if is_true "${AUTO_FORWARD_CONTINUOUS_GAIT}"; then
    RUN_COMMAND+=(--auto-forward-continuous-gait)
fi
if is_true "${ARM_ONLY_TEST}"; then
    RUN_COMMAND+=(--arm-only-test)
fi
if is_true "${ALLOW_MISSING_SPORT_STATE}"; then
    RUN_COMMAND+=(--allow-missing-sport-state)
fi
if is_true "${SET_LOCOMOTION_AFTER_STANDUP}"; then
    RUN_COMMAND+=(--set-locomotion-after-standup)
fi
if is_true "${ALLOW_MISSING_FSM_500}"; then
    RUN_COMMAND+=(--allow-missing-fsm-500)
fi
if is_true "${RUN_ZERO_TORQUE_TAKEOVER}"; then
    RUN_COMMAND+=(--run-zero-torque-takeover)
fi
if is_true "${DIRECT_ZERO_ARM_MOTOR_TEST}"; then
    RUN_COMMAND+=(--direct-zero-arm-motor-test)
fi
if is_true "${TAKEOVER_REQUIRE_KNEE_MOTION}"; then
    RUN_COMMAND+=(--takeover-require-knee-motion)
fi
if is_true "${RUN_ARM_SWING_AFTER_WASD}"; then
    RUN_COMMAND+=(--run-arm-swing-after-wasd)
fi
if is_true "${BALANCE_STAND_AFTER_WASD}"; then
    RUN_COMMAND+=(--balance-stand-after-wasd)
fi
if is_true "${ARM_HOLD_NON_TARGET}"; then
    RUN_COMMAND+=(--arm-hold-non-target)
fi
if is_true "${ARM_ABORT_ON_ANY_HOLD_ERROR}"; then
    RUN_COMMAND+=(--arm-abort-on-any-hold-error)
fi
if is_true "${ARM_REQUIRE_SWING_MOTION}"; then
    RUN_COMMAND+=(--arm-require-swing-motion)
fi
if is_true "${ARM_REQUIRE_STATIC_FSM_MODE}"; then
    RUN_COMMAND+=(--arm-require-static-fsm-mode)
fi
if [[ -n "${START_SERVICES}" ]]; then
    IFS=',' read -ra SERVICE_ARRAY <<< "${START_SERVICES}"
    for SERVICE_NAME in "${SERVICE_ARRAY[@]}"; do
        [[ -n "${SERVICE_NAME}" ]] && RUN_COMMAND+=(--start-service "${SERVICE_NAME}")
    done
fi

echo "====================================="
echo "  Unitree G1 Loco FSM WASD"
echo "====================================="
echo "Python             : ${UNITREE_PYTHON}"
echo "Net                : ${NET}"
echo "Robot IP           : ${ROBOT_IP}"
echo "Loco Service       : ${LOCO_SERVICE}"
echo "Start Services     : ${START_SERVICES:-<none>}"
echo "SportState Topic   : ${SPORT_STATE_TOPIC}"
echo "Allow Missing Sport: ${ALLOW_MISSING_SPORT_STATE}"
echo "Probe Only         : ${PROBE_ONLY}"
echo "Verify Zero Command: ${VERIFY_ZERO_COMMAND}"
echo "Will Send Command  : ${WILL_SEND_COMMAND}"
echo "Manual Step Confirm: ${MANUAL_STEP_CONFIRM}"
echo "Final Damp On Exit : ${FINAL_DAMP_ON_EXIT}"
echo "Handoff Flow       : ${HANDOFF_FLOW}"
echo "Locomotion FSM IDs : ${LOCOMOTION_FSM_IDS}"
echo "Start FSM IDs      : ${START_FSM_IDS}"
echo "Speed              : ${SPEED_MPS} m/s"
echo "Command Duration   : ${COMMAND_DURATION_S} s"
echo "WASD Repeat        : ${WASD_REPEAT_S} s"
echo "WASD Control       : manual keyboard"
echo "Auto Forward Test  : ${AUTO_FORWARD_TEST} speed=${AUTO_FORWARD_SPEED_MPS} duration=${AUTO_FORWARD_DURATION_S}"
echo "Retry Start FSMs   : ${SET_LOCOMOTION_AFTER_STANDUP}"
echo "ZeroTorque Takeover: ${RUN_ZERO_TORQUE_TAKEOVER}"
echo "Direct Zero Arm    : ${DIRECT_ZERO_ARM_MOTOR_TEST}"
echo "Takeover Topic/DT  : ${TAKEOVER_LOWCMD_TOPIC} / ${TAKEOVER_CONTROL_DT_S}s"
echo "Takeover Warmup    : cycles=${TAKEOVER_WARMUP_MIN_CYCLES} timeout=${TAKEOVER_WARMUP_TIMEOUT_S}s"
echo "Takeover Tau FF    : mode=${TAKEOVER_TAU_FF_MODE} scale=${TAKEOVER_TAU_FF_SCALE} max=${TAKEOVER_MAX_ABS_TAU_FF}"
echo "Takeover PD leg    : ${TAKEOVER_LEG_KP}/${TAKEOVER_LEG_KD} waist=${TAKEOVER_WAIST_KP}/${TAKEOVER_WAIST_KD} hold=${TAKEOVER_HOLD_KP}/${TAKEOVER_HOLD_KD}"
echo "Takeover Knee Bend : delta=${TAKEOVER_KNEE_BEND_RAD}rad time=${TAKEOVER_KNEE_BEND_S}s require=${TAKEOVER_REQUIRE_KNEE_MOTION}"
echo "Takeover Recovery  : fsm=${TAKEOVER_RECOVER_FSM_ID} confirm=${TAKEOVER_RECOVER_CONFIRM_FSM_IDS}"
echo "Takeover E-Damping : kd=${TAKEOVER_EMERGENCY_DAMPING_KD} duration=${TAKEOVER_EMERGENCY_DAMPING_S}s"
echo "Run Arm Swing      : ${RUN_ARM_SWING_AFTER_WASD}"
echo "Arm Only Test      : ${ARM_ONLY_TEST}"
echo "Arm Topic          : ${ARM_TOPIC}"
echo "Arm Joint Set      : ${ARM_JOINT_SET}"
echo "Arm Hold NonTarget : ${ARM_HOLD_NON_TARGET}"
echo "Arm Amplitude Scale: ${ARM_AMPLITUDE_SCALE}"
echo "Arm Swing Duration : ${ARM_SWING_S} s (<=0 means until key)"
echo "Arm Hold kp/kd     : ${ARM_HOLD_KP}/${ARM_HOLD_KD} threshold=${ARM_HOLD_ERROR_THRESHOLD_RAD} rad grace=${ARM_HOLD_GRACE_S}s"
echo "Arm Waist kp/kd    : ${ARM_WAIST_HOLD_KP}/${ARM_WAIST_HOLD_KD}"
echo "Arm Hold Abort Any : ${ARM_ABORT_ON_ANY_HOLD_ERROR}"
echo "Arm Require Motion : ${ARM_REQUIRE_SWING_MOTION} min_delta=${ARM_MIN_SWING_DELTA_RAD} rad"
echo "Arm Limit Margin   : ${ARM_LIMIT_MARGIN_RAD} rad max_amp=${ARM_MAX_AMPLITUDE_RAD} rad"
echo "Dry Run            : ${DRY_RUN}"
echo "====================================="

if is_true "${DRY_RUN}"; then
    printf 'Dry-run command:'
    printf ' %q' "${RUN_COMMAND[@]}"
    printf '\n'
    exit 0
fi

if is_true "${PING_ROBOT}"; then
    ping -c 1 -W 1 "${ROBOT_IP}" >/dev/null
    echo "Ping ${ROBOT_IP}: ok"
fi

"${RUN_COMMAND[@]}"
