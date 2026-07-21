#!/usr/bin/env bash
# 真机部署最终版 G1 Extreme Stand Recovery ONNX。
#
# 本脚本只负责：
#   1. 锁定并校验 Pose V2 model_2999 导出的 ONNX 与部署元数据；
#   2. 校验 96 -> 29 接口、50 Hz、action_scale、关节顺序、默认姿态和 PD 增益；
#   3. 强制零速度指令；
#   4. 完成确认门、依赖、网络和 LowState 检查后，立即从机器人当前姿态开始 ONNX 推理；
#   5. 将真正的低层通信与控制交给现有 deploy_real_g1_amp_onnx.sh。
#
# 因而关节映射、默认角、Kp/Kd、action_scale 和 LowCmd 写入流程与 AMP 真机脚本完全一致。
# AMP 共享执行器链当前没有额外的软件 q_target 关节限位裁剪；本脚本也不擅自添加一套不同限位。
# 这里的“恢复站立”由学习策略完成，不是传统起身动作：脚本不会先做默认姿态插值，
# 也不承诺从倒地、关节越限或训练分布外的任意姿态恢复。首次真机运行必须使用安全架和急停。

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_EXPORT_DIR="${ROOT_DIR}/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery"
DEFAULT_USE_ONNX="${ROOT_DIR}/use/extreme_stand_recovery_pose_v2_model2999.onnx"
ONNX_PATH=${ONNX_PATH:-"${DEFAULT_USE_ONNX}"}
DEFAULT_USE_METADATA="${ROOT_DIR}/use/extreme_stand_recovery_pose_v2_model2999.deploy.json"
METADATA_PATH=${METADATA_PATH:-"${DEFAULT_USE_METADATA}"}
DEPLOY_CONFIG_PATH="${ROOT_DIR}/unitree_sim2sim2real/deploy/deploy_real/configs/g1_amp.yaml"
BASE_LAUNCHER="${ROOT_DIR}/scripts/deploy_real_g1_amp_onnx.sh"

EXPECTED_ONNX_SHA256=${EXPECTED_ONNX_SHA256:-0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1}
EXPECTED_METADATA_SHA256=${EXPECTED_METADATA_SHA256:-2bf0f21c511463b19bd8a1ef1f77122cc43cee41560bfb398e3b06ba00164fd7}
EXPECTED_CONFIG_SHA256=${EXPECTED_CONFIG_SHA256:-99556cd8050588de1f169607b61f3173b5c6633fe093d3f385e21a04d0c9e2fb}

if [[ -z "${UNITREE_PYTHON:-}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/env_leglab/bin/python" \
        "${HOME}/miniconda3/envs/armhack-real/bin/python" \
        "${HOME}/anaconda3/envs/armhack-real/bin/python" \
        "${HOME}/anaconda3/envs/unitree-rl/bin/python" \
        "${HOME}/anaconda3/envs/gmr/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            UNITREE_PYTHON="${candidate}"
            break
        fi
    done
fi
NET=${NET:-}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

for path_var in ONNX_PATH METADATA_PATH; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${ROOT_DIR}/${value}"
    fi
done

[[ -f "${ONNX_PATH}" ]] || { echo "Error: ONNX 不存在: ${ONNX_PATH}" >&2; exit 1; }
[[ -f "${METADATA_PATH}" ]] || { echo "Error: 部署元数据不存在: ${METADATA_PATH}" >&2; exit 1; }
[[ -f "${DEPLOY_CONFIG_PATH}" ]] || { echo "Error: AMP 真机配置不存在: ${DEPLOY_CONFIG_PATH}" >&2; exit 1; }
[[ -f "${BASE_LAUNCHER}" ]] || { echo "Error: AMP 真机启动器不存在: ${BASE_LAUNCHER}" >&2; exit 1; }
[[ -n "${UNITREE_PYTHON:-}" && -x "${UNITREE_PYTHON}" ]] || { echo "Error: UNITREE_PYTHON 未设置或不可执行: ${UNITREE_PYTHON:-<unset>}" >&2; exit 1; }
[[ -n "${NET}" ]] || { echo "Error: 必须设置 Unitree 网卡，例如 NET=enp11s0。" >&2; exit 1; }

actual_onnx_sha=$(sha256sum "${ONNX_PATH}" | awk '{print $1}')
actual_metadata_sha=$(sha256sum "${METADATA_PATH}" | awk '{print $1}')
actual_config_sha=$(sha256sum "${DEPLOY_CONFIG_PATH}" | awk '{print $1}')
if [[ "${actual_onnx_sha}" != "${EXPECTED_ONNX_SHA256}" ]]; then
    echo "Error: ONNX SHA256 不匹配。" >&2
    echo "  expected=${EXPECTED_ONNX_SHA256}" >&2
    echo "  actual  =${actual_onnx_sha}" >&2
    exit 1
fi
if [[ "${actual_metadata_sha}" != "${EXPECTED_METADATA_SHA256}" ]]; then
    echo "Error: 部署元数据 SHA256 不匹配。" >&2
    echo "  expected=${EXPECTED_METADATA_SHA256}" >&2
    echo "  actual  =${actual_metadata_sha}" >&2
    exit 1
fi
if [[ "${actual_config_sha}" != "${EXPECTED_CONFIG_SHA256}" ]]; then
    echo "Error: g1_amp.yaml 已变化，不能证明与既有 AMP 限位/PD/映射链一致。" >&2
    echo "  expected=${EXPECTED_CONFIG_SHA256}" >&2
    echo "  actual  =${actual_config_sha}" >&2
    exit 1
fi

"${UNITREE_PYTHON}" - "${ONNX_PATH}" "${METADATA_PATH}" "${DEPLOY_CONFIG_PATH}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

onnx_path, metadata_path, config_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

errors: list[str] = []
if int(metadata.get("obs_dim", -1)) != 96:
    errors.append(f"metadata obs_dim={metadata.get('obs_dim')}, expected 96")
if int(metadata.get("action_dim", -1)) != 29:
    errors.append(f"metadata action_dim={metadata.get('action_dim')}, expected 29")
if int(metadata.get("control_frequency_hz", -1)) != 50:
    errors.append(f"metadata control_frequency_hz={metadata.get('control_frequency_hz')}, expected 50")
if not math.isclose(float(config.get("control_dt", -1.0)), 0.02, rel_tol=0.0, abs_tol=1.0e-9):
    errors.append(f"config control_dt={config.get('control_dt')}, expected 0.02")
if not math.isclose(float(metadata.get("action_scale", math.nan)), float(config.get("action_scale", math.nan)), rel_tol=0.0, abs_tol=1.0e-9):
    errors.append("metadata/config action_scale mismatch")

joint_names = list(metadata.get("action_joint_names", []))
if joint_names != list(config.get("policy_joint_names", [])):
    errors.append("policy_joint_names does not match exported action_joint_names")
if len(config.get("motor_joint_names", [])) != 29 or len(config.get("motor_indices", [])) != 29:
    errors.append("motor_joint_names/motor_indices must both have length 29")

expected_default = np.asarray([metadata["default_joint_pos"][name] for name in joint_names], dtype=np.float64)
config_default = np.asarray(config.get("default_angles", []), dtype=np.float64)
expected_kp = np.asarray([metadata["pd_gains"][name]["kp"] for name in joint_names], dtype=np.float64)
expected_kd = np.asarray([metadata["pd_gains"][name]["kd"] for name in joint_names], dtype=np.float64)
config_kp = np.asarray(config.get("kps", []), dtype=np.float64)
config_kd = np.asarray(config.get("kds", []), dtype=np.float64)
for label, lhs, rhs in (
    ("default_angles", expected_default, config_default),
    ("kps", expected_kp, config_kp),
    ("kds", expected_kd, config_kd),
):
    if lhs.shape != (29,) or rhs.shape != (29,) or not np.allclose(lhs, rhs, rtol=0.0, atol=1.0e-9):
        errors.append(f"metadata/config {label} mismatch")

default_command = metadata.get("default_command", {})
command = [default_command.get("lin_vel_x"), default_command.get("lin_vel_y"), default_command.get("ang_vel_z")]
if any(value is None or abs(float(value)) > 1.0e-12 for value in command):
    errors.append(f"exported default command is not zero: {command}")

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]
if list(input_info.shape) != [1, 96]:
    errors.append(f"ONNX input shape={input_info.shape}, expected [1, 96]")
action = np.asarray(session.run([output_info.name], {input_info.name: np.zeros((1, 96), dtype=np.float32)})[0])
if action.shape != (1, 29):
    errors.append(f"ONNX output shape={action.shape}, expected (1, 29)")
if not np.isfinite(action).all():
    errors.append("ONNX output contains NaN/Inf")

if errors:
    raise SystemExit("Deployment contract check failed:\n- " + "\n- ".join(errors))

print("部署契约校验通过:")
print("  ONNX: 96 observations -> 29 full-body actions")
print("  control: 50 Hz, action_scale=0.25, command=[0, 0, 0]")
print("  joint order/default angles/Kp/Kd: identical to g1_amp.yaml")
PY

if ! is_true "${DRY_RUN}"; then
    PYTHONPATH="${ROOT_DIR}/unitree_sim2sim2real:${ROOT_DIR}/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}" \
        "${UNITREE_PYTHON}" - <<'PY'
try:
    import cyclonedds  # noqa: F401
    import onnxruntime  # noqa: F401
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "真机 Python 依赖检查失败：需要同一环境提供 cyclonedds、"
        f"unitree_sdk2py 和 onnxruntime；原始错误: {exc!r}"
    ) from exc
print("真机 Python 依赖校验通过: cyclonedds + unitree_sdk2py + onnxruntime")
PY
fi

echo "============================================================"
echo "  G1 Extreme Stand Recovery 真机部署"
echo "============================================================"
echo "ONNX           : ${ONNX_PATH}"
echo "ONNX SHA256    : ${actual_onnx_sha}"
echo "Metadata       : ${METADATA_PATH}"
echo "Metadata SHA256: ${actual_metadata_sha}"
echo "AMP config     : ${DEPLOY_CONFIG_PATH}"
echo "Config SHA256  : ${actual_config_sha}"
echo "Network        : ${NET}"
echo "Policy command : [0, 0, 0]（强制固定）"
echo "Control chain  : 与 deploy_real_g1_amp_onnx.sh 完全相同"
echo "Target formula : q_target = default_angles + 0.25 * actor_action"
echo "Software clamp : 与 AMP 基线一致，当前共享 runner 没有额外 q_target 关节限位裁剪"
echo "Startup mode   : 检查通过并收到 LowState 后，立即从当前姿态自动运行 ONNX 并发送策略 PD 目标"
echo "Pre-positioning: 无默认姿态插值、无单独起身动作；恢复能力仅限训练/验收覆盖的可恢复分布"
echo "Dry run        : ${DRY_RUN}"
echo "============================================================"

# 不允许外部命令源覆盖静态策略的零速度目标；其余安全门和真实通信由基线脚本负责。
ONNX_PATH="${ONNX_PATH}" \
UNITREE_PYTHON="${UNITREE_PYTHON}" \
NET="${NET}" \
CONFIG=g1_amp.yaml \
COMMAND_MODE=fixed \
CMD_INIT='[0.0,0.0,0.0]' \
COMMAND_RAMP=False \
RUN_DURATION="${RUN_DURATION:-0.0}" \
RELEASE_MOTION_MODE=False \
HANDOFF_MODE=stand \
TERMINAL_SPACE_HANDOFF=False \
DEFAULT_MOVE_S=0.0 \
DEFAULT_HOLD_S=0.0 \
SMOKE_TEST_POLICY="${SMOKE_TEST_POLICY:-True}" \
PING_ROBOT="${PING_ROBOT:-True}" \
CONFIRM_REAL_ROBOT="${CONFIRM_REAL_ROBOT:-}" \
DRY_RUN="${DRY_RUN}" \
    bash "${BASE_LAUNCHER}" "${ONNX_PATH}"
