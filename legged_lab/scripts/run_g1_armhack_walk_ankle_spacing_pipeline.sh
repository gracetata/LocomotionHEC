#!/usr/bin/env bash
# Reproducible Future5090 pipeline for all three actors and final gated merge.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGGED_LAB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SOURCE_DIR="${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkAnkleSpacingFinetune/bootstrap_bias0p30_20260814"
FINAL_DIR="${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkAnkleSpacingFinetune/final_w500_20260814"
STATUS_FILE="${LEGGED_LAB_DIR}/logs/training_supervisor/armhack_walk_ankle30_w500_20260814.tsv"
NUM_ENVS=${NUM_ENVS:-4096}
BASE_ITERATIONS=${BASE_ITERATIONS:-200}
LATERAL_ITERATIONS=${LATERAL_ITERATIONS:-200}
YAW_ITERATIONS=${YAW_ITERATIONS:-200}

[[ "$(hostname)" == "tata-futurelab" ]] || { echo "Future5090 only" >&2; exit 1; }
mkdir -p "$(dirname "${STATUS_FILE}")" "${FINAL_DIR}"
printf 'time\tbranch\tstate\tdetail\n' > "${STATUS_FILE}"

record() {
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" | tee -a "${STATUS_FILE}"
}

latest_checkpoint() {
    local branch=$1 run_name=$2
    local run_dir
    run_dir=$(find "${LEGGED_LAB_DIR}/ArmHack Checkpoints/WalkAnkleSpacingFinetune/${branch}" \
        -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-)
    [[ -n "${run_dir}" ]] || return 1
    find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
        | sort -V | tail -1 | sed "s#^#${run_dir}/#"
}

train_branch() {
    local branch=$1 source_sha=$2 iterations=$3
    local source="${SOURCE_DIR}/source_${branch}.pt"
    local run_name="ankle30_${branch}_full_20260814"
    record "${branch}" started "${iterations} iterations, ${NUM_ENVS} envs"
    BRANCH="${branch}" SOURCE_CHECKPOINT="${source}" SOURCE_SHA256="${source_sha}" \
    NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${iterations}" RUN_NAME="${run_name}" \
        bash "${SCRIPT_DIR}/train_g1_armhack_walk_ankle_spacing.sh"
    local checkpoint
    checkpoint=$(latest_checkpoint "${branch}" "${run_name}")
    [[ -f "${checkpoint}" ]] || { record "${branch}" failed "checkpoint missing"; return 1; }
    local sha
    sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
    record "${branch}" completed "${checkpoint}|${sha}"
    printf '%s' "${checkpoint}" > "${FINAL_DIR}/${branch}_checkpoint_path.txt"
}

train_branch base a60e041d98c86b2b5e36af8bed65c069353587f08ae04a7c0587ae054b71641b "${BASE_ITERATIONS}"
train_branch lateral 9d2bf313dcc9869099f25afcfc6e286892bd39bcefaddd521757992e85686b89 "${LATERAL_ITERATIONS}"
train_branch yaw fefe4c9d44222d8a1b8282375e4d0304dd917f6bc275507ab32d880375c3837a "${YAW_ITERATIONS}"

BASE=$(<"${FINAL_DIR}/base_checkpoint_path.txt")
LATERAL=$(<"${FINAL_DIR}/lateral_checkpoint_path.txt")
YAW=$(<"${FINAL_DIR}/yaw_checkpoint_path.txt")
OUTPUT="${FINAL_DIR}/model_armhack_walk_ankle30.pt"
ONNX_OUTPUT="${FINAL_DIR}/policy_armhack_walk_ankle30.onnx"
JIT_OUTPUT="${FINAL_DIR}/policy_armhack_walk_ankle30.pt"
METADATA_OUTPUT="${FINAL_DIR}/policy_armhack_walk_ankle30.deploy.json"
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }
"${HOME}/anaconda3/envs/env_isaaclab/bin/python" \
    "${SCRIPT_DIR}/merge_g1_armhack_walk_ankle_spacing.py" \
    --base "${BASE}" --lateral "${LATERAL}" --yaw "${YAW}" \
    --gate-source "${SOURCE_DIR}/gated_source.pt" --output "${OUTPUT}"
record merged completed "${OUTPUT}|$(sha256sum "${OUTPUT}" | awk '{print $1}')"

for artifact in "${ONNX_OUTPUT}" "${JIT_OUTPUT}" "${METADATA_OUTPUT}"; do
    [[ ! -e "${artifact}" ]] || { echo "Refusing to overwrite ${artifact}" >&2; exit 1; }
done
PYTHONNOUSERSITE=1 "${HOME}/anaconda3/envs/env_isaaclab/bin/python" \
    "${SCRIPT_DIR}/rsl_rl/export_amp_actor_to_onnx.py" \
    --robot g1 \
    --checkpoint "${OUTPUT}" \
    --output "${ONNX_OUTPUT}" \
    --jit-output "${JIT_OUTPUT}" \
    --metadata "${METADATA_OUTPUT}" \
    --default-command 0 0 0
record exported completed \
    "onnx=${ONNX_OUTPUT}|$(sha256sum "${ONNX_OUTPUT}" | awk '{print $1}')|jit=${JIT_OUTPUT}|$(sha256sum "${JIT_OUTPUT}" | awk '{print $1}')"
