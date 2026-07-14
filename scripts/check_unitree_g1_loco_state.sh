#!/usr/bin/env bash
# Read Unitree G1 high-level loco state and optionally request a stand action.
# Usage:
#   NET=enp11s0 bash scripts/check_unitree_g1_loco_state.sh
#   CONFIRM_STAND=I_UNDERSTAND STAND_ACTION=high NET=enp11s0 bash scripts/check_unitree_g1_loco_state.sh
#
# Environment variables:
#   UNITREE_PYTHON : Python executable from unitree_sim2sim2real/.envrc conda env.
#   NET            : Network interface passed to Unitree DDS, e.g. eth0.
#   STAND_ACTION   : none, high, low, squat2stand, damp, zero, or stop. Default: none.
#   CONFIRM_STAND  : Must equal I_UNDERSTAND for non-read-only STAND_ACTION values.
#   DRY_RUN        : True prints the resolved command without connecting to DDS.
# ============================================================
: <<'BLOCK'
# [历史记录与快捷执行指令存放区]
# 1. 只读检查 Unitree high-level 运控状态：
NET=enp11s0 bash scripts/check_unitree_g1_loco_state.sh

# 2. 只打印检查命令，不连接真机：
DRY_RUN=True NET=enp11s0 bash scripts/check_unitree_g1_loco_state.sh

# 3. 显式请求高站立姿态：
CONFIRM_STAND=I_UNDERSTAND STAND_ACTION=high NET=enp11s0 bash scripts/check_unitree_g1_loco_state.sh

# 4. 显式请求阻尼模式：
CONFIRM_STAND=I_UNDERSTAND STAND_ACTION=damp NET=enp11s0 bash scripts/check_unitree_g1_loco_state.sh
BLOCK
# ============================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNITREE_DIR="${ROOT_DIR}/unitree_sim2sim2real"
SDK_DIR="${ROOT_DIR}/unitree_sdk2_python"
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/hecggdz/miniconda3/envs/env_leglab/bin/python}
NET=${NET:-}
STAND_ACTION=${STAND_ACTION:-none}
CONFIRM_STAND=${CONFIRM_STAND:-}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" ]]
}

if [[ -z "${NET}" ]]; then
    echo "Error: set NET to the Unitree network interface, e.g. NET=enp11s0." >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/home/hecggdz/miniconda3/envs/unitree-rl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${UNITREE_DIR}:${SDK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_COMMAND=(
    "${UNITREE_PYTHON}"
    "${UNITREE_DIR}/deploy/deploy_real/check_g1_loco_state.py"
    "${NET}"
    --stand-action "${STAND_ACTION}"
    --confirm-stand "${CONFIRM_STAND}"
)

echo "====================================="
echo "  Unitree G1 Loco State Check"
echo "====================================="
echo "Python      : ${UNITREE_PYTHON}"
echo "Net         : ${NET}"
echo "Stand Action: ${STAND_ACTION}"
echo "Dry Run     : ${DRY_RUN}"
echo "====================================="

if is_true "${DRY_RUN}"; then
    printf 'Dry-run command:'
    printf ' %q' "${RUN_COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${RUN_COMMAND[@]}"