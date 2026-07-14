#!/usr/bin/env bash
# Continuously monitor Unitree G1 mode state and optionally switch modes by keyboard.
# Usage:
#   NET=enp11s0 ALLOW_COMMANDS=False bash scripts/g1_mode_console.sh
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 bash scripts/g1_mode_console.sh
#   CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 ACTION=native bash scripts/g1_mode_console.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}

NET=${NET:-}
ROBOT_IP=${ROBOT_IP:-192.168.123.161}
ALLOW_COMMANDS=${ALLOW_COMMANDS:-True}
CONFIRM_REAL_ROBOT=${CONFIRM_REAL_ROBOT:-}
FORCE_UNSAFE=${FORCE_UNSAFE:-False}
LOCO_SERVICE=${LOCO_SERVICE:-sport}
START_FSM_IDS=${START_FSM_IDS:-801,802,500}
RPC_TIMEOUT_S=${RPC_TIMEOUT_S:-0.3}
POLL_S=${POLL_S:-0.1}
REFRESH_S=${REFRESH_S:-0.5}
SWITCH_TIMEOUT_S=${SWITCH_TIMEOUT_S:-20.0}
STAND_TIMEOUT_S=${STAND_TIMEOUT_S:-12.0}
MIN_STANDUP_S=${MIN_STANDUP_S:-6.0}
LOCOMOTION_CANDIDATE_TIMEOUT_S=${LOCOMOTION_CANDIDATE_TIMEOUT_S:-4.0}
SELECT_RETRY_S=${SELECT_RETRY_S:-1.0}
DURATION_S=${DURATION_S:-0.0}
ACTION=${ACTION:-none}
NO_TUI=${NO_TUI:-False}
PING_ROBOT=${PING_ROBOT:-True}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

if [[ -z "${NET}" ]]; then
    echo "Error: set NET to the Unitree network interface, e.g. NET=enp11s0." >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if is_true "${ALLOW_COMMANDS}" && [[ "${CONFIRM_REAL_ROBOT}" != "I_UNDERSTAND" ]] && ! is_true "${DRY_RUN}"; then
    echo "Refusing command-capable console without CONFIRM_REAL_ROBOT=I_UNDERSTAND." >&2
    echo "For read-only monitoring use: ALLOW_COMMANDS=False NET=${NET} bash scripts/g1_mode_console.sh" >&2
    exit 2
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_COMMAND=(
    "${UNITREE_PYTHON}"
    "${UNITREE_DIR}/deploy/deploy_real/g1_mode_console.py"
    "${NET}"
    --loco-service "${LOCO_SERVICE}"
    --start-fsm-ids "${START_FSM_IDS}"
    --rpc-timeout-s "${RPC_TIMEOUT_S}"
    --poll-s "${POLL_S}"
    --refresh-s "${REFRESH_S}"
    --switch-timeout-s "${SWITCH_TIMEOUT_S}"
    --stand-timeout-s "${STAND_TIMEOUT_S}"
    --min-standup-s "${MIN_STANDUP_S}"
    --locomotion-candidate-timeout-s "${LOCOMOTION_CANDIDATE_TIMEOUT_S}"
    --select-retry-s "${SELECT_RETRY_S}"
    --duration-s "${DURATION_S}"
    --action "${ACTION}"
)
if is_true "${ALLOW_COMMANDS}"; then
    RUN_COMMAND+=(--allow-commands --confirm-real-robot "${CONFIRM_REAL_ROBOT}")
fi
if is_true "${FORCE_UNSAFE}"; then
    RUN_COMMAND+=(--force-unsafe)
fi
if is_true "${NO_TUI}"; then
    RUN_COMMAND+=(--no-tui)
fi

echo "====================================="
echo "  G1 Mode Console"
echo "====================================="
echo "Python        : ${UNITREE_PYTHON}"
echo "Net           : ${NET}"
echo "Robot IP      : ${ROBOT_IP}"
echo "Allow Commands: ${ALLOW_COMMANDS}"
echo "Force Unsafe  : ${FORCE_UNSAFE}"
echo "Action        : ${ACTION}"
echo "Dry Run       : ${DRY_RUN}"
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
