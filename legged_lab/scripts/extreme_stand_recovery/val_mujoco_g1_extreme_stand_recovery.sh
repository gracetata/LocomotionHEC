#!/usr/bin/env bash
# 兼容入口：统一委托给项目根目录的 Pose V2 model_2999 MuJoCo 测试/可视化套件。

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

exec bash "${PROJECT_ROOT}/scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh" "$@"
