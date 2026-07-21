#!/usr/bin/env bash
# Export and verify the final Pose V2 96-observation -> 29-action Extreme Stand actor.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGGED_LAB_DIR="${ROOT_DIR}/legged_lab"

DEFAULT_CHECKPOINT="${LEGGED_LAB_DIR}/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt"
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264}
VERIFY_CHECKPOINT_SHA256=${VERIFY_CHECKPOINT_SHA256:-True}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-"${HOME}/anaconda3/envs/env_isaaclab/bin/python"}
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
EXPORT_DIR=${EXPORT_DIR:-"$(dirname "${CHECKPOINT}")/exported_extreme_stand_recovery"}
JIT_OUT=${JIT_OUT:-"${EXPORT_DIR}/policy.pt"}
ONNX_OUT=${ONNX_OUT:-"${EXPORT_DIR}/policy.onnx"}
METADATA_OUT=${METADATA_OUT:-"${EXPORT_DIR}/policy.deploy.json"}
DRY_RUN=${DRY_RUN:-False}

is_true() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

for path_var in CHECKPOINT EXPORT_DIR JIT_OUT ONNX_OUT METADATA_OUT; do
    value=${!path_var}
    if [[ "${value}" != /* ]]; then
        printf -v "${path_var}" '%s' "${ROOT_DIR}/${value}"
    fi
done

[[ -x "${ISAACLAB_PYTHON}" ]] || { echo "Error: IsaacLab Python 不可执行: ${ISAACLAB_PYTHON}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Error: checkpoint 不存在: ${CHECKPOINT}" >&2; exit 1; }

actual_checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
if is_true "${VERIFY_CHECKPOINT_SHA256}" && [[ "${actual_checkpoint_sha256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "Error: checkpoint SHA-256 不匹配。" >&2
    echo "  expected: ${EXPECTED_CHECKPOINT_SHA256}" >&2
    echo "  actual  : ${actual_checkpoint_sha256}" >&2
    exit 1
fi

EXPORT_COMMAND=(
    "${ISAACLAB_PYTHON}"
    "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py"
    --robot g1
    --checkpoint "${CHECKPOINT}"
    --output "${ONNX_OUT}"
    --jit-output "${JIT_OUT}"
    --metadata "${METADATA_OUT}"
    --default-command 0 0 0
)

echo "============================================================"
echo "  Extreme Stand Recovery actor export"
echo "============================================================"
echo "Checkpoint      : ${CHECKPOINT}"
echo "Checkpoint SHA  : ${actual_checkpoint_sha256}"
echo "TorchScript     : ${JIT_OUT}"
echo "ONNX            : ${ONNX_OUT}"
echo "Metadata        : ${METADATA_OUT}"
echo "Policy contract : obs[1,96] -> actions[1,29], command=[0,0,0]"
echo "Dry run         : ${DRY_RUN}"
echo "============================================================"

if is_true "${DRY_RUN}"; then
    printf 'Dry-run export command:'
    printf ' %q' "${EXPORT_COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${EXPORT_DIR}"
"${EXPORT_COMMAND[@]}"

"${ISAACLAB_PYTHON}" - "${JIT_OUT}" "${ONNX_OUT}" "${METADATA_OUT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

jit_path, onnx_path, metadata_path = map(Path, sys.argv[1:4])
onnx.checker.check_model(onnx.load(onnx_path))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("obs_dim") != 96 or metadata.get("action_dim") != 29:
    raise SystemExit(f"Invalid metadata actor shape: {metadata.get('obs_dim')} -> {metadata.get('action_dim')}")
command = metadata.get("default_command", {})
if [float(command.get(key, 1.0)) for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")] != [0.0, 0.0, 0.0]:
    raise SystemExit(f"Extreme Stand metadata must use zero command, got: {command}")

jit = torch.jit.load(str(jit_path), map_location="cpu").eval()
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
rng = np.random.default_rng(20260719)
max_abs_diff = 0.0
for obs in (np.zeros((1, 96), dtype=np.float32), rng.normal(size=(1, 96)).astype(np.float32)):
    with torch.inference_mode():
        jit_action = jit(torch.from_numpy(obs)).detach().cpu().numpy()
    onnx_action = np.asarray(session.run([output_name], {input_name: obs})[0])
    if tuple(onnx_action.shape) != (1, 29) or not np.isfinite(onnx_action).all():
        raise SystemExit(f"Invalid ONNX output: shape={onnx_action.shape}, finite={np.isfinite(onnx_action).all()}")
    max_abs_diff = max(max_abs_diff, float(np.max(np.abs(jit_action - onnx_action))))
if max_abs_diff > 1.0e-5:
    raise SystemExit(f"TorchScript/ONNX parity failed: max_abs_diff={max_abs_diff}")
print(f"Export validation passed: obs[1,96] -> actions[1,29], max_abs_diff={max_abs_diff:.9g}")
PY

sha256sum "${JIT_OUT}" "${ONNX_OUT}" "${METADATA_OUT}"
