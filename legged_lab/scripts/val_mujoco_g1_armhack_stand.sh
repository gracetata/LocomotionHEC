#!/usr/bin/env bash
# Export and evaluate the ArmHack Stand policy with the project's G1 MuJoCo sim2sim runner.
#
# Examples:
#   bash scripts/val_mujoco_g1_armhack_stand.sh
#   PAYLOAD_KG=1.0 bash scripts/val_mujoco_g1_armhack_stand.sh
#   USE_GLFW=True REAL_TIME=True MODE=all bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=randomized_trajectory ITEM=5 USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=randomized_trajectory TRAJECTORY_INDEX=5 USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=down_to_horizontal USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
TEST_DATA_DIR="${LEGGED_LAB_DIR}/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly"
MANIFEST="${TEST_DATA_DIR}/manifest.json"
DEFAULT_CHECKPOINT="${LEGGED_LAB_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"
DEFAULT_STAND_ONNX="${LEGGED_LAB_DIR}/deployment/armhack_stand/stand.onnx"

MODE=${MODE:-all}
ITEM=${ITEM:-}
TRAJECTORY_INDEX=${TRAJECTORY_INDEX:-}
CUSTOM_CSV_PATH=${CUSTOM_CSV_PATH:-}
CUSTOM_TEST_ID=${CUSTOM_TEST_ID:-custom_pose}
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
POLICY_PATH=${POLICY_PATH:-}
PAYLOAD_KG=${PAYLOAD_KG:-0.0}
USE_GLFW=${USE_GLFW:-False}
REAL_TIME=${REAL_TIME:-False}
FORCE_EXPORT=${FORCE_EXPORT:-False}
ISAAC_PYTHON=${ISAAC_PYTHON:-${HOME}/miniconda3/envs/env_leglab/bin/python}
UNITREE_PYTHON=${UNITREE_PYTHON:-${HOME}/miniconda3/envs/env_leglab/bin/python}
METADATA_PYTHON=${METADATA_PYTHON:-python3}
MUJOCO_CPU_THREADS=${MUJOCO_CPU_THREADS:-1}
JOINT_RANDOM_ENABLE=${JOINT_RANDOM_ENABLE:-True}
JOINT_RANDOM_SEED=${JOINT_RANDOM_SEED:-20260718}
JOINT_POS_NOISE_RAD=${JOINT_POS_NOISE_RAD:-0.03}
JOINT_VEL_NOISE_RAD_PER_S=${JOINT_VEL_NOISE_RAD_PER_S:-0.10}
NON_ARM_JOINT_TARGET_NOISE_ENABLE=${NON_ARM_JOINT_TARGET_NOISE_ENABLE:-True}
NON_ARM_JOINT_TARGET_NOISE_SEED=${NON_ARM_JOINT_TARGET_NOISE_SEED:-20260719}
NON_ARM_JOINT_TARGET_NOISE_RAD=${NON_ARM_JOINT_TARGET_NOISE_RAD:-0.02}
FOOT_RECOVERY_ENABLE=${FOOT_RECOVERY_ENABLE:-False}
ORDERED_STEP_OBSERVATION_ENABLE=${ORDERED_STEP_OBSERVATION_ENABLE:-False}
ORDERED_STEP_MIRROR_POLICY_ENABLE=${ORDERED_STEP_MIRROR_POLICY_ENABLE:-False}
ORDERED_STEP_HOLD_LAST_ACTION_ENABLE=${ORDERED_STEP_HOLD_LAST_ACTION_ENABLE:-False}
ORDERED_STEP_HOLD_POLICY_PATH=${ORDERED_STEP_HOLD_POLICY_PATH:-}
ORDERED_STEP_HOLD_BLEND_DURATION_S=${ORDERED_STEP_HOLD_BLEND_DURATION_S:-1.0}
ORDERED_STEP_TRANSITION_TOLERANCE_M=${ORDERED_STEP_TRANSITION_TOLERANCE_M:-0.055}
ORDERED_STEP_MIN_CLEARANCE_M=${ORDERED_STEP_MIN_CLEARANCE_M:-0.035}
ORDERED_STEP_MIN_DURATION_S=${ORDERED_STEP_MIN_DURATION_S:-0.0}
ORDERED_STEP_ACTION_SMOOTHING_ALPHA=${ORDERED_STEP_ACTION_SMOOTHING_ALPHA:-1.0}
INITIAL_ANKLE_DISTANCE_M=${INITIAL_ANKLE_DISTANCE_M:-0.10}
INTERACTIVE_STANCE_RESET=${INTERACTIVE_STANCE_RESET:-False}
INTERACTIVE_STANCE_DISTANCE_RANGE_M=${INTERACTIVE_STANCE_DISTANCE_RANGE_M:-'[0.08,0.32]'}
INTERACTIVE_STANCE_SEED=${INTERACTIVE_STANCE_SEED:-20260814}
POSITION_RECOVERY_COMMAND_ENABLE=${POSITION_RECOVERY_COMMAND_ENABLE:-True}
POSITION_RECOVERY_COMMAND_XY_CLIP_M=${POSITION_RECOVERY_COMMAND_XY_CLIP_M:-0.50}
POSITION_RECOVERY_COMMAND_YAW_CLIP_RAD=${POSITION_RECOVERY_COMMAND_YAW_CLIP_RAD:-0.60}
POSITION_RECOVERY_COMMAND_XY_GAIN=${POSITION_RECOVERY_COMMAND_XY_GAIN:-2.0}
POSITION_RECOVERY_COMMAND_YAW_GAIN=${POSITION_RECOVERY_COMMAND_YAW_GAIN:-1.5}
TARGET_ANKLE_DISTANCE_M=${TARGET_ANKLE_DISTANCE_M:-0.30}
ANKLE_DISTANCE_TOLERANCE_M=${ANKLE_DISTANCE_TOLERANCE_M:-0.03}
ANKLE_CONVERGENCE_HOLD_S=${ANKLE_CONVERGENCE_HOLD_S:-0.50}
RECOVERY_SETTLE_TIME_S=${RECOVERY_SETTLE_TIME_S:-5.0}
MUJOCO_PUSH_ENABLE=${MUJOCO_PUSH_ENABLE:-False}
MUJOCO_PUSH_SEED=${MUJOCO_PUSH_SEED:-20260814}
MUJOCO_PUSH_FIRST_TIME_S=${MUJOCO_PUSH_FIRST_TIME_S:-6.0}
MUJOCO_PUSH_INTERVAL_RANGE_S=${MUJOCO_PUSH_INTERVAL_RANGE_S:-'[3.0,6.0]'}
MUJOCO_PUSH_DURATION_S=${MUJOCO_PUSH_DURATION_S:-0.12}
MUJOCO_PUSH_FORCE_RANGE_N=${MUJOCO_PUSH_FORCE_RANGE_N:-'[80.0,120.0]'}
MUJOCO_PUSH_YAW_TORQUE_RANGE_NM=${MUJOCO_PUSH_YAW_TORQUE_RANGE_NM:-'[-8.0,8.0]'}
GENERATED_SEQUENCE_DIR=${GENERATED_SEQUENCE_DIR:-${LEGGED_LAB_DIR}/deployment/armhack_stand/generated_mujoco_sequences}
CLIP_RANDOMIZED_TRAJECTORY_COUNT=""
TEST_ID_OVERRIDE=""
ARMHACK_TEST_ID_OVERRIDE=""

bool_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${LEGGED_LAB_DIR}/${CHECKPOINT}"
fi
if [[ -n "${POLICY_PATH}" && "${POLICY_PATH}" != /* ]]; then
    if [[ -f "${LEGGED_LAB_DIR}/${POLICY_PATH}" ]]; then
        POLICY_PATH="${LEGGED_LAB_DIR}/${POLICY_PATH}"
    else
        POLICY_PATH="${PROJECT_ROOT}/${POLICY_PATH}"
    fi
fi

USE_DIRECT_POLICY=False
if [[ -n "${POLICY_PATH}" ]] && ! bool_true "${FORCE_EXPORT}"; then
    USE_DIRECT_POLICY=True
elif [[ ! -f "${CHECKPOINT}" && -f "${DEFAULT_STAND_ONNX}" ]] && ! bool_true "${FORCE_EXPORT}"; then
    POLICY_PATH="${DEFAULT_STAND_ONNX}"
    USE_DIRECT_POLICY=True
fi

if ! bool_true "${USE_DIRECT_POLICY}" && [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: Stand checkpoint does not exist and no direct POLICY_PATH was selected: ${CHECKPOINT}" >&2
    echo "Hint : POLICY_PATH=${DEFAULT_STAND_ONNX} can run the deployed ONNX directly." >&2
    exit 1
fi
if bool_true "${USE_DIRECT_POLICY}" && [[ ! -f "${POLICY_PATH}" ]]; then
    echo "Error: Stand policy does not exist: ${POLICY_PATH}" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Error: ArmHack Stand schema v5 manifest does not exist: ${MANIFEST}" >&2
    exit 1
fi
if ! bool_true "${USE_DIRECT_POLICY}" && [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Error: ISAAC_PYTHON is not executable: ${ISAAC_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not executable: ${UNITREE_PYTHON}" >&2
    exit 1
fi
if ! "${UNITREE_PYTHON}" - "${USE_DIRECT_POLICY}" "${POLICY_PATH}" <<'PY' >/dev/null 2>&1
import sys

use_direct_policy = sys.argv[1].lower() in {"1", "true", "yes", "on"}
policy_path = sys.argv[2]
required = ["mujoco", "torch", "yaml", "numpy", "matplotlib"]
if use_direct_policy and policy_path.endswith(".onnx"):
    required.append("onnxruntime")
for module in required:
    __import__(module)
PY
then
    echo "Error: UNITREE_PYTHON must provide mujoco, torch, yaml, numpy, matplotlib and onnxruntime when needed." >&2
    echo "UNITREE_PYTHON=${UNITREE_PYTHON}" >&2
    exit 1
fi
if ! awk -v value="${PAYLOAD_KG}" 'BEGIN { exit !(value >= 0.0 && value <= 3.0) }'; then
    echo "Error: PAYLOAD_KG must be within [0, 3] kg per wrist." >&2
    exit 1
fi

case "${MODE}" in
    custom_pose_csv)
        [[ -n "${CUSTOM_CSV_PATH}" ]] || { echo "Error: MODE=custom_pose_csv requires CUSTOM_CSV_PATH" >&2; exit 1; }
        if [[ "${CUSTOM_CSV_PATH}" != /* ]]; then
            CUSTOM_CSV_PATH="${PROJECT_ROOT}/${CUSTOM_CSV_PATH}"
        fi
        CSV_PATH="${CUSTOM_CSV_PATH}"
        DESCRIPTION="custom explicit arm-pose hold"
        TEST_ID_OVERRIDE="${CUSTOM_TEST_ID}"
        ARMHACK_TEST_ID_OVERRIDE="${CUSTOM_TEST_ID}"
        ;;
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
        if [[ -n "${TRAJECTORY_INDEX}" ]]; then
            [[ "${TRAJECTORY_INDEX}" =~ ^[1-6]$ ]] || { echo "Error: TRAJECTORY_INDEX requires 1..6" >&2; exit 1; }
            printf -v ITEM_PADDED "%02d" "${TRAJECTORY_INDEX}"
            ITEM="${TRAJECTORY_INDEX}"
            CSV_NAME="trajectories/randomized/randomized_arm_trajectory_${ITEM_PADDED}_seed20260715_minjerk_50hz.csv"
            DESCRIPTION="minimum-jerk randomized trajectory ${ITEM_PADDED} at 1.0x"
            TEST_ID_OVERRIDE="randomized_trajectory_item${TRAJECTORY_INDEX}"
            ARMHACK_TEST_ID_OVERRIDE="${TEST_ID_OVERRIDE}"
        else
            [[ "${ITEM}" =~ ^[1-6]$ ]] || { echo "Error: MODE=randomized_trajectory requires ITEM=1..6 as a trajectory count; use TRAJECTORY_INDEX=1..6 for one exact item" >&2; exit 1; }
            CSV_NAME="sequences/randomized_trajectories_arm_only_sequence_seed20260715_50hz.csv"
            DESCRIPTION="first ${ITEM} minimum-jerk randomized trajectories with smooth bridges"
            CLIP_RANDOMIZED_TRAJECTORY_COUNT="${ITEM}"
            TEST_ID_OVERRIDE="randomized_trajectories_first${ITEM}"
            ARMHACK_TEST_ID_OVERRIDE="randomized_trajectories"
        fi
        ;;
    *)
        echo "Error: unsupported MODE=${MODE}" >&2
        exit 1
        ;;
esac

CSV_PATH=${CSV_PATH:-"${TEST_DATA_DIR}/${CSV_NAME}"}
if [[ ! -f "${CSV_PATH}" ]]; then
    echo "Error: ArmHack Stand test CSV does not exist: ${CSV_PATH}" >&2
    exit 1
fi
if [[ -n "${CLIP_RANDOMIZED_TRAJECTORY_COUNT}" ]]; then
    mkdir -p "${GENERATED_SEQUENCE_DIR}"
    CSV_PATH="$("${METADATA_PYTHON}" - "${CSV_PATH}" "${MANIFEST}" "${GENERATED_SEQUENCE_DIR}" "${CLIP_RANDOMIZED_TRAJECTORY_COUNT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
count = int(sys.argv[4])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
metadata = manifest["files"]["randomized_trajectories"]
timeline = metadata.get("detailed_timeline") or metadata.get("timeline") or []
target_label = f"randomized_trajectory_{count:02d}"
end_s = None
for stage in timeline:
    if stage.get("kind") == "section" and stage.get("label") == target_label:
        end_s = float(stage["end_s"])
        break
if end_s is None:
    raise SystemExit(f"Cannot find {target_label} in randomized_trajectories timeline")

output_path = output_dir / f"randomized_trajectories_first_{count:02d}_seed20260715_50hz.csv"
with source_path.open("r", encoding="utf-8", newline="") as src:
    reader = csv.DictReader(src)
    if not reader.fieldnames:
        raise SystemExit(f"CSV has no header: {source_path}")
    rows = [row for row in reader if float(row["time_s"]) <= end_s + 1.0e-9]

with output_path.open("w", encoding="utf-8", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(output_path)
PY
)"
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

if bool_true "${USE_DIRECT_POLICY}"; then
    MODEL_SOURCE_PATH="${POLICY_PATH}"
    MODEL_SOURCE_KIND="Policy"
else
    MODEL_SOURCE_PATH="${CHECKPOINT}"
    MODEL_SOURCE_KIND="Checkpoint"
fi

CHECKPOINT_DIR=$(dirname "${MODEL_SOURCE_PATH}")
CHECKPOINT_STEM=$(basename "${MODEL_SOURCE_PATH}")
CHECKPOINT_STEM=${CHECKPOINT_STEM%.*}
CHECKPOINT_SHA256=$(sha256sum "${MODEL_SOURCE_PATH}" | awk '{print $1}')
CHECKPOINT_SHORT_SHA=${CHECKPOINT_SHA256:0:12}
MODEL_ID=${MODEL_ID:-${CHECKPOINT_STEM}_${CHECKPOINT_SHORT_SHA}}
TEST_ID=${TEST_ID_OVERRIDE:-${MODE}${ITEM:+_item${ITEM}}}
ARMHACK_TEST_ID=${ARMHACK_TEST_ID_OVERRIDE:-${TEST_ID}}
PAYLOAD_TAG=$(awk -v value="${PAYLOAD_KG}" 'BEGIN { printf "%.6g", value }' | tr '.' 'p')

EXPORT_DIR="${CHECKPOINT_DIR}/MuJoCo Export/StandArmOnly"
POLICY_PATH=${POLICY_PATH:-${EXPORT_DIR}/policy.pt}
ONNX_PATH="${EXPORT_DIR}/policy.onnx"
DEPLOY_METADATA_PATH="${EXPORT_DIR}/policy.deploy.json"
REPORT_DIR="${CHECKPOINT_DIR}/Test Reports/StandArmOnlyMuJoCo"
REPORT_STEM="${MODEL_ID}__mujoco__${TEST_ID}__payload_${PAYLOAD_TAG}kg"
REPORT_PATH="${REPORT_DIR}/${REPORT_STEM}.md"
METRICS_PATH="${REPORT_DIR}/${REPORT_STEM}.json"

if ! bool_true "${USE_DIRECT_POLICY}" && { bool_true "${FORCE_EXPORT}" || [[ ! -f "${POLICY_PATH}" || ! -f "${ONNX_PATH}" || ! -f "${DEPLOY_METADATA_PATH}" ]]; }; then
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
echo "Mode        : ${TEST_ID}"
echo "Contents    : ${DESCRIPTION}"
echo "CSV         : ${CSV_PATH}"
echo "CSV duration: ${CSV_DURATION_S} s"
echo "Sim duration: ${SIMULATION_DURATION} s"
echo "${MODEL_SOURCE_KIND}  : ${MODEL_SOURCE_PATH}"
echo "Model SHA   : ${CHECKPOINT_SHA256}"
echo "Policy      : ${POLICY_PATH}"
echo "Payload     : ${PAYLOAD_KG} kg per wrist-yaw link"
echo "Joint random: ${JOINT_RANDOM_ENABLE} seed=${JOINT_RANDOM_SEED} qpos=+/-${JOINT_POS_NOISE_RAD} rad qvel=+/-${JOINT_VEL_NOISE_RAD_PER_S} rad/s"
echo "Non-arm noise: ${NON_ARM_JOINT_TARGET_NOISE_ENABLE} seed=${NON_ARM_JOINT_TARGET_NOISE_SEED} target=+/-${NON_ARM_JOINT_TARGET_NOISE_RAD} rad"
echo "Foot recovery: ${FOOT_RECOVERY_ENABLE} initial=${INITIAL_ANKLE_DISTANCE_M}m target=${TARGET_ANKLE_DISTANCE_M}m tolerance=${ANKLE_DISTANCE_TOLERANCE_M}m"
echo "Ordered step : min_clearance=${ORDERED_STEP_MIN_CLEARANCE_M}m min_duration=${ORDERED_STEP_MIN_DURATION_S}s transition_tol=${ORDERED_STEP_TRANSITION_TOLERANCE_M}m"
echo "Space reset : ${INTERACTIVE_STANCE_RESET} range=${INTERACTIVE_STANCE_DISTANCE_RANGE_M}m seed=${INTERACTIVE_STANCE_SEED}"
echo "Pose return : ${POSITION_RECOVERY_COMMAND_ENABLE} gain_xy=${POSITION_RECOVERY_COMMAND_XY_GAIN} gain_yaw=${POSITION_RECOVERY_COMMAND_YAW_GAIN}"
echo "Push test   : ${MUJOCO_PUSH_ENABLE} first=${MUJOCO_PUSH_FIRST_TIME_S}s force=${MUJOCO_PUSH_FORCE_RANGE_N}N"
echo "GLFW/RT     : ${USE_GLFW}/${REAL_TIME}"
echo "Report      : ${REPORT_PATH}"
echo "============================================================"

export G1_AMP_ARMHACK_STAND_ENABLE=True
export G1_AMP_ARMHACK_STAND_CSV_PATH="${CSV_PATH}"
export G1_AMP_ARMHACK_STAND_MANIFEST_PATH="${MANIFEST}"
export G1_AMP_ARMHACK_STAND_CHECKPOINT_PATH="${MODEL_SOURCE_PATH}"
export G1_AMP_ARMHACK_STAND_REPORT_PATH="${REPORT_PATH}"
export G1_AMP_ARMHACK_STAND_TEST_ID="${ARMHACK_TEST_ID}"
export G1_AMP_ARMHACK_STAND_PAYLOAD_KG="${PAYLOAD_KG}"

UNITREE_PYTHON="${UNITREE_PYTHON}" \
OMP_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
MKL_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
OPENBLAS_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
NUMEXPR_NUM_THREADS="${MUJOCO_CPU_THREADS}" \
POLICY_PATH="${POLICY_PATH}" \
POLICY_RUNTIME=auto \
ROBOT_ASSET=s3_g1_29dof \
USE_GLFW="${USE_GLFW}" \
REAL_TIME="${REAL_TIME}" \
SIMULATION_DURATION="${SIMULATION_DURATION}" \
METRICS_PATH="${METRICS_PATH}" \
JOINT_RANDOM_ENABLE="${JOINT_RANDOM_ENABLE}" \
JOINT_RANDOM_SEED="${JOINT_RANDOM_SEED}" \
JOINT_POS_NOISE_RAD="${JOINT_POS_NOISE_RAD}" \
JOINT_VEL_NOISE_RAD_PER_S="${JOINT_VEL_NOISE_RAD_PER_S}" \
NON_ARM_JOINT_TARGET_NOISE_ENABLE="${NON_ARM_JOINT_TARGET_NOISE_ENABLE}" \
NON_ARM_JOINT_TARGET_NOISE_SEED="${NON_ARM_JOINT_TARGET_NOISE_SEED}" \
NON_ARM_JOINT_TARGET_NOISE_RAD="${NON_ARM_JOINT_TARGET_NOISE_RAD}" \
FOOT_RECOVERY_ENABLE="${FOOT_RECOVERY_ENABLE}" \
ORDERED_STEP_OBSERVATION_ENABLE="${ORDERED_STEP_OBSERVATION_ENABLE}" \
ORDERED_STEP_MIRROR_POLICY_ENABLE="${ORDERED_STEP_MIRROR_POLICY_ENABLE}" \
ORDERED_STEP_HOLD_LAST_ACTION_ENABLE="${ORDERED_STEP_HOLD_LAST_ACTION_ENABLE}" \
ORDERED_STEP_HOLD_POLICY_PATH="${ORDERED_STEP_HOLD_POLICY_PATH}" \
ORDERED_STEP_HOLD_BLEND_DURATION_S="${ORDERED_STEP_HOLD_BLEND_DURATION_S}" \
ORDERED_STEP_TRANSITION_TOLERANCE_M="${ORDERED_STEP_TRANSITION_TOLERANCE_M}" \
ORDERED_STEP_MIN_CLEARANCE_M="${ORDERED_STEP_MIN_CLEARANCE_M}" \
ORDERED_STEP_MIN_DURATION_S="${ORDERED_STEP_MIN_DURATION_S}" \
ORDERED_STEP_ACTION_SMOOTHING_ALPHA="${ORDERED_STEP_ACTION_SMOOTHING_ALPHA}" \
INITIAL_ANKLE_DISTANCE_M="${INITIAL_ANKLE_DISTANCE_M}" \
INTERACTIVE_STANCE_RESET="${INTERACTIVE_STANCE_RESET}" \
INTERACTIVE_STANCE_DISTANCE_RANGE_M="${INTERACTIVE_STANCE_DISTANCE_RANGE_M}" \
INTERACTIVE_STANCE_SEED="${INTERACTIVE_STANCE_SEED}" \
POSITION_RECOVERY_COMMAND_ENABLE="${POSITION_RECOVERY_COMMAND_ENABLE}" \
POSITION_RECOVERY_COMMAND_XY_CLIP_M="${POSITION_RECOVERY_COMMAND_XY_CLIP_M}" \
POSITION_RECOVERY_COMMAND_YAW_CLIP_RAD="${POSITION_RECOVERY_COMMAND_YAW_CLIP_RAD}" \
POSITION_RECOVERY_COMMAND_XY_GAIN="${POSITION_RECOVERY_COMMAND_XY_GAIN}" \
POSITION_RECOVERY_COMMAND_YAW_GAIN="${POSITION_RECOVERY_COMMAND_YAW_GAIN}" \
TARGET_ANKLE_DISTANCE_M="${TARGET_ANKLE_DISTANCE_M}" \
ANKLE_DISTANCE_TOLERANCE_M="${ANKLE_DISTANCE_TOLERANCE_M}" \
ANKLE_CONVERGENCE_HOLD_S="${ANKLE_CONVERGENCE_HOLD_S}" \
RECOVERY_SETTLE_TIME_S="${RECOVERY_SETTLE_TIME_S}" \
MUJOCO_PUSH_ENABLE="${MUJOCO_PUSH_ENABLE}" \
MUJOCO_PUSH_SEED="${MUJOCO_PUSH_SEED}" \
MUJOCO_PUSH_FIRST_TIME_S="${MUJOCO_PUSH_FIRST_TIME_S}" \
MUJOCO_PUSH_INTERVAL_RANGE_S="${MUJOCO_PUSH_INTERVAL_RANGE_S}" \
MUJOCO_PUSH_DURATION_S="${MUJOCO_PUSH_DURATION_S}" \
MUJOCO_PUSH_FORCE_RANGE_N="${MUJOCO_PUSH_FORCE_RANGE_N}" \
MUJOCO_PUSH_YAW_TORQUE_RANGE_NM="${MUJOCO_PUSH_YAW_TORQUE_RANGE_NM}" \
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
