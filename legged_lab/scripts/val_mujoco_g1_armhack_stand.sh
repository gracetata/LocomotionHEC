#!/usr/bin/env bash
# Export and evaluate the ArmHack Stand policy with the project's G1 MuJoCo sim2sim runner.
#
# Examples:
#   bash scripts/val_mujoco_g1_armhack_stand.sh
#   PAYLOAD_KG=1.0 bash scripts/val_mujoco_g1_armhack_stand.sh
#   USE_GLFW=True REAL_TIME=True MODE=all bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=randomized_trajectory ITEM=1 USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=down_to_horizontal USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
TEST_DATA_DIR="${LEGGED_LAB_DIR}/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly"
MANIFEST="${TEST_DATA_DIR}/manifest.json"
DEFAULT_CHECKPOINT="${LEGGED_LAB_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"

MODE=${MODE:-all}
ITEM=${ITEM:-}
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
PAYLOAD_KG=${PAYLOAD_KG:-0.0}
USE_GLFW=${USE_GLFW:-False}
REAL_TIME=${REAL_TIME:-False}
FORCE_EXPORT=${FORCE_EXPORT:-False}
ISAAC_PYTHON=${ISAAC_PYTHON:-/home/user/anaconda3/envs/env_isaaclab/bin/python}
UNITREE_PYTHON=${UNITREE_PYTHON:-/home/user/anaconda3/envs/gmr/bin/python}
METADATA_PYTHON=${METADATA_PYTHON:-python3}
MUJOCO_CPU_THREADS=${MUJOCO_CPU_THREADS:-1}

bool_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: Stand checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Error: ArmHack Stand schema v5 manifest does not exist: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Error: ISAAC_PYTHON is not executable: ${ISAAC_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if ! "${UNITREE_PYTHON}" -c 'import mujoco, torch, yaml, numpy, matplotlib' >/dev/null 2>&1; then
    echo "Error: UNITREE_PYTHON must provide mujoco, torch, yaml, numpy and matplotlib." >&2
    echo "Validated local environment: /home/user/anaconda3/envs/gmr (install PyYAML once if missing)." >&2
    exit 1
fi
if ! awk -v value="${PAYLOAD_KG}" 'BEGIN { exit !(value >= 0.0 && value <= 3.0) }'; then
    echo "Error: PAYLOAD_KG must be within [0, 3] kg per wrist." >&2
    exit 1
fi

case "${MODE}" in
    all)
        CSV_NAME="sequences/all_arm_only_evaluation_sequence_seed20260714_50hz.csv"
        DESCRIPTION="schema v5 complete deterministic pose/trajectory suite"
        ;;
    representative_poses)
        CSV_NAME="sequences/representative_poses_arm_only_sequence_50hz.csv"
        DESCRIPTION="six representative poses with smooth transitions"
        ;;
    synthesized_poses)
        CSV_NAME="sequences/synthesized_poses_arm_only_sequence_50hz.csv"
        DESCRIPTION="three measured-blend poses with smooth transitions"
        ;;
    randomized_poses)
        CSV_NAME="sequences/randomized_poses_arm_only_sequence_50hz.csv"
        DESCRIPTION="eight deterministic randomized-bank coverage poses"
        ;;
    representative_trajectories)
        CSV_NAME="sequences/representative_trajectories_arm_only_sequence_50hz.csv"
        DESCRIPTION="four measured trajectories at 1.0x"
        ;;
    synthesized_trajectories)
        CSV_NAME="sequences/synthesized_trajectories_arm_only_sequence_seed20260714_50hz.csv"
        DESCRIPTION="three measured-trajectory blends at 1.0x"
        ;;
    randomized_trajectories)
        CSV_NAME="sequences/randomized_trajectories_arm_only_sequence_seed20260715_50hz.csv"
        DESCRIPTION="six minimum-jerk randomized-pose trajectories at 1.0x"
        ;;
    down_to_horizontal)
        CSV_NAME="special/arms_down_to_forward_horizontal_20s_50hz.csv"
        DESCRIPTION="5 s arms-down hold, 6 s minimum-jerk lift, 9 s forward-horizontal hold"
        ;;
    representative_pose)
        [[ "${ITEM}" =~ ^[1-6]$ ]] || { echo "Error: MODE=representative_pose requires ITEM=1..6" >&2; exit 1; }
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="poses/representative/representative_arm_pose_${ITEM_PADDED}_hold20s_50hz.csv"
        DESCRIPTION="representative pose ${ITEM_PADDED}"
        ;;
    synthesized_pose)
        [[ "${ITEM}" =~ ^[1-3]$ ]] || { echo "Error: MODE=synthesized_pose requires ITEM=1..3" >&2; exit 1; }
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="poses/synthesized/synthesized_arm_pose_${ITEM_PADDED}_seed20260714_hold20s_50hz.csv"
        DESCRIPTION="measured-blend pose ${ITEM_PADDED}"
        ;;
    randomized_pose)
        [[ "${ITEM}" =~ ^[1-8]$ ]] || { echo "Error: MODE=randomized_pose requires ITEM=1..8" >&2; exit 1; }
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="poses/randomized/randomized_arm_pose_${ITEM_PADDED}_seed20260715_hold20s_50hz.csv"
        DESCRIPTION="randomized-bank pose ${ITEM_PADDED}"
        ;;
    representative_trajectory)
        [[ "${ITEM}" =~ ^[1-4]$ ]] || { echo "Error: MODE=representative_trajectory requires ITEM=1..4" >&2; exit 1; }
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        case "${ITEM}" in
            1) SOURCE_WINDOW="036_041" ;;
            2) SOURCE_WINDOW="102_107" ;;
            3) SOURCE_WINDOW="234_239" ;;
            4) SOURCE_WINDOW="385_390" ;;
        esac
        CSV_NAME="trajectories/representative/representative_arm_trajectory_${ITEM_PADDED}_source_${SOURCE_WINDOW}s_1x_50hz.csv"
        DESCRIPTION="representative trajectory ${ITEM_PADDED} at 1.0x"
        ;;
    synthesized_trajectory)
        [[ "${ITEM}" =~ ^[1-3]$ ]] || { echo "Error: MODE=synthesized_trajectory requires ITEM=1..3" >&2; exit 1; }
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="trajectories/synthesized/synthesized_arm_trajectory_${ITEM_PADDED}_seed20260714_measured_blend_1x_50hz.csv"
        DESCRIPTION="measured-blend trajectory ${ITEM_PADDED} at 1.0x"
        ;;
    randomized_trajectory)
        [[ "${ITEM}" =~ ^[1-6]$ ]] || { echo "Error: MODE=randomized_trajectory requires ITEM=1..6" >&2; exit 1; }
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="trajectories/randomized/randomized_arm_trajectory_${ITEM_PADDED}_seed20260715_minjerk_50hz.csv"
        DESCRIPTION="minimum-jerk randomized trajectory ${ITEM_PADDED} at 1.0x"
        ;;
    *)
        echo "Error: unsupported MODE=${MODE}" >&2
        exit 1
        ;;
esac

CSV_PATH="${TEST_DATA_DIR}/${CSV_NAME}"
if [[ ! -f "${CSV_PATH}" ]]; then
    echo "Error: ArmHack Stand test CSV does not exist: ${CSV_PATH}" >&2
    exit 1
fi

read -r CSV_DURATION_S DEFAULT_SIMULATION_DURATION <<<"$("${METADATA_PYTHON}" - "${CSV_PATH}" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
if not rows or "time_s" not in rows[0]:
    raise SystemExit("ArmHack Stand CSV has no samples or time_s column")
final_time = float(rows[-1]["time_s"])
# Include the final 50 Hz target once. The MuJoCo physics step is 0.002 s.
print(f"{final_time:.8f} {final_time + 0.002:.8f}")
PY
)"
SIMULATION_DURATION=${SIMULATION_DURATION:-${DEFAULT_SIMULATION_DURATION}}

CHECKPOINT_DIR=$(dirname "${CHECKPOINT}")
CHECKPOINT_STEM=$(basename "${CHECKPOINT}")
CHECKPOINT_STEM=${CHECKPOINT_STEM%.*}
CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
CHECKPOINT_SHORT_SHA=${CHECKPOINT_SHA256:0:12}
MODEL_ID=${MODEL_ID:-${CHECKPOINT_STEM}_${CHECKPOINT_SHORT_SHA}}
TEST_ID="${MODE}${ITEM:+_item${ITEM}}"
PAYLOAD_TAG=$(awk -v value="${PAYLOAD_KG}" 'BEGIN { printf "%.6g", value }' | tr '.' 'p')

EXPORT_DIR="${CHECKPOINT_DIR}/MuJoCo Export/StandArmOnly"
POLICY_PATH=${POLICY_PATH:-${EXPORT_DIR}/policy.pt}
ONNX_PATH="${EXPORT_DIR}/policy.onnx"
DEPLOY_METADATA_PATH="${EXPORT_DIR}/policy.deploy.json"
REPORT_DIR="${CHECKPOINT_DIR}/Test Reports/StandArmOnlyMuJoCo"
REPORT_STEM="${MODEL_ID}__mujoco__${TEST_ID}__payload_${PAYLOAD_TAG}kg"
REPORT_PATH="${REPORT_DIR}/${REPORT_STEM}.md"
METRICS_PATH="${REPORT_DIR}/${REPORT_STEM}.json"

if bool_true "${FORCE_EXPORT}" || [[ ! -f "${POLICY_PATH}" || ! -f "${ONNX_PATH}" || ! -f "${DEPLOY_METADATA_PATH}" ]]; then
    mkdir -p "${EXPORT_DIR}"
    echo "[ArmHack Stand MuJoCo] Exporting actor from checkpoint..."
    "${ISAAC_PYTHON}" "${LEGGED_LAB_DIR}/scripts/rsl_rl/export_amp_actor_to_onnx.py" \
        --robot g1 \
        --checkpoint "${CHECKPOINT}" \
        --output "${ONNX_PATH}" \
        --jit-output "${POLICY_PATH}" \
        --metadata "${DEPLOY_METADATA_PATH}"
fi

mkdir -p "${REPORT_DIR}"

echo "============================================================"
echo " ArmHack Stand MuJoCo sim2sim"
echo "============================================================"
echo "Mode        : ${MODE}${ITEM:+ (item ${ITEM})}"
echo "Contents    : ${DESCRIPTION}"
echo "CSV         : ${CSV_PATH}"
echo "CSV duration: ${CSV_DURATION_S} s"
echo "Sim duration: ${SIMULATION_DURATION} s"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Model SHA   : ${CHECKPOINT_SHA256}"
echo "Policy      : ${POLICY_PATH}"
echo "Payload     : ${PAYLOAD_KG} kg per wrist-yaw link"
echo "GLFW/RT     : ${USE_GLFW}/${REAL_TIME}"
echo "Report      : ${REPORT_PATH}"
echo "============================================================"

export G1_AMP_ARMHACK_STAND_ENABLE=True
export G1_AMP_ARMHACK_STAND_CSV_PATH="${CSV_PATH}"
export G1_AMP_ARMHACK_STAND_MANIFEST_PATH="${MANIFEST}"
export G1_AMP_ARMHACK_STAND_CHECKPOINT_PATH="${CHECKPOINT}"
export G1_AMP_ARMHACK_STAND_REPORT_PATH="${REPORT_PATH}"
export G1_AMP_ARMHACK_STAND_TEST_ID="${TEST_ID}"
export G1_AMP_ARMHACK_STAND_PAYLOAD_KG="${PAYLOAD_KG}"

UNITREE_PYTHON="${UNITREE_PYTHON}" \
OMP_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
MKL_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
OPENBLAS_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
NUMEXPR_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
POLICY_PATH="${POLICY_PATH}" \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" \
REAL_TIME="${REAL_TIME}" \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
METRICS_PATH="${METRICS_PATH}" \
CMD_INIT='[0.0, 0.0, 0.0]' \
RANDOM_COMMANDS=False \
COMMAND_MODE=independent \
COMMAND_RAMP=False \
TORSO_TRACE_ENABLE=False \
TASK_TRACE_ENABLE=False \
FOLLOW_CAMERA_ENABLE=${FOLLOW_CAMERA_ENABLE:-True} \
bash "${PROJECT_ROOT}/scripts/sim2sim_g1_amp_mujoco.sh"

echo "[ArmHack Stand MuJoCo] Report: ${REPORT_PATH}"
echo "[ArmHack Stand MuJoCo] JSON  : ${METRICS_PATH}"
