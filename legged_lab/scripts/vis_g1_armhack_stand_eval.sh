#!/usr/bin/env bash
# Deterministic visualization for the ArmHack Stand checkpoint.
#
# The CSV suite is selected offline from the training-reachable ranges of the
# complete arm dataset. Every CSV contains only time_s plus 14 arm joints. This
# launcher never randomizes the CSV start phase at runtime, so every mode is
# reproducible and can be reviewed item by item.
#
# Examples:
#   bash scripts/vis_g1_armhack_stand_eval.sh
#   MODE=representative_poses bash scripts/vis_g1_armhack_stand_eval.sh
#   MODE=representative_trajectory ITEM=3 bash scripts/vis_g1_armhack_stand_eval.sh
#   MODE=down_to_horizontal bash scripts/vis_g1_armhack_stand_eval.sh
#   MODE=synthesized_trajectory ITEM=2 HEADLESS=True MAX_STEPS=1000 \
#     bash scripts/vis_g1_armhack_stand_eval.sh

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DATA_DIR="${LEGGED_LAB_DIR}/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly"
MANIFEST="${TEST_DATA_DIR}/manifest.json"
DEFAULT_CHECKPOINT="${LEGGED_LAB_DIR}/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"

MODE=${MODE:-all}
ITEM=${ITEM:-}
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
METADATA_PYTHON=${METADATA_PYTHON:-python3}
PAYLOAD_KG=${PAYLOAD_KG:-0.0}

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Error: deterministic visualization manifest does not exist: ${MANIFEST}" >&2
    echo "Build it with: ${METADATA_PYTHON} scripts/tools/build_armhack_stand_visualization_suite.py" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: Stand checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
fi
if ! command -v "${METADATA_PYTHON}" >/dev/null 2>&1; then
    echo "Error: metadata Python is unavailable: ${METADATA_PYTHON}" >&2
    exit 1
fi
if ! awk -v value="${PAYLOAD_KG}" 'BEGIN { exit !(value >= 0.0 && value <= 3.0) }'; then
    echo "Error: PAYLOAD_KG must be within [0, 3]." >&2
    exit 1
fi

for argument in "$@"; do
    case "${argument}" in
        env.episode_length_s=*|env.upper_body_perturbation.*)
            echo "Error: ${argument} would change the fixed deterministic ArmHack evaluation protocol." >&2
            exit 1
            ;;
    esac
done

case "${MODE}" in
    all)
        CSV_NAME="sequences/all_arm_only_evaluation_sequence_seed20260714_50hz.csv"
        DESCRIPTION="6 representative + 3 measured-blend + 8 randomized poses; 4 measured + 3 measured-blend + 6 randomized interpolation trajectories"
        ;;
    representative_poses)
        CSV_NAME="sequences/representative_poses_arm_only_sequence_50hz.csv"
        DESCRIPTION="six training-reachable representative source poses with smooth transitions; arms-down tail pose removed"
        ;;
    synthesized_poses)
        CSV_NAME="sequences/synthesized_poses_arm_only_sequence_50hz.csv"
        DESCRIPTION="three fixed-seed synthesized poses with smooth transitions"
        ;;
    randomized_poses)
        CSV_NAME="sequences/randomized_poses_arm_only_sequence_50hz.csv"
        DESCRIPTION="eight deterministic coverage poses from the 512-pose randomized training bank"
        ;;
    representative_trajectories)
        CSV_NAME="sequences/representative_trajectories_arm_only_sequence_50hz.csv"
        DESCRIPTION="four representative source trajectories at the original 1.0x speed"
        ;;
    synthesized_trajectories)
        CSV_NAME="sequences/synthesized_trajectories_arm_only_sequence_seed20260714_50hz.csv"
        DESCRIPTION="three fixed-seed convex blends of measured 1.0x arm trajectories"
        ;;
    randomized_trajectories)
        CSV_NAME="sequences/randomized_trajectories_arm_only_sequence_seed20260715_50hz.csv"
        DESCRIPTION="six velocity-limited minimum-jerk trajectories interpolated between randomized poses"
        ;;
    down_to_horizontal)
        CSV_NAME="special/arms_down_to_forward_horizontal_20s_50hz.csv"
        DESCRIPTION="5 s arms-down hold, 6 s minimum-jerk lift, 9 s forward-horizontal hold"
        ;;
    representative_pose)
        if [[ ! "${ITEM}" =~ ^[1-6]$ ]]; then
            echo "Error: MODE=representative_pose requires ITEM=1..6" >&2
            exit 1
        fi
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="poses/representative/representative_arm_pose_${ITEM_PADDED}_hold20s_50hz.csv"
        DESCRIPTION="representative source pose ${ITEM_PADDED}"
        ;;
    synthesized_pose)
        if [[ ! "${ITEM}" =~ ^[1-3]$ ]]; then
            echo "Error: MODE=synthesized_pose requires ITEM=1..3" >&2
            exit 1
        fi
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="poses/synthesized/synthesized_arm_pose_${ITEM_PADDED}_seed20260714_hold20s_50hz.csv"
        DESCRIPTION="fixed-seed synthesized pose ${ITEM_PADDED}"
        ;;
    randomized_pose)
        if [[ ! "${ITEM}" =~ ^[1-8]$ ]]; then
            echo "Error: MODE=randomized_pose requires ITEM=1..8" >&2
            exit 1
        fi
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="poses/randomized/randomized_arm_pose_${ITEM_PADDED}_seed20260715_hold20s_50hz.csv"
        DESCRIPTION="deterministic randomized-bank coverage pose ${ITEM_PADDED}"
        ;;
    representative_trajectory)
        if [[ ! "${ITEM}" =~ ^[1-4]$ ]]; then
            echo "Error: MODE=representative_trajectory requires ITEM=1..4" >&2
            exit 1
        fi
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        case "${ITEM}" in
            1) SOURCE_WINDOW="036_041" ;;
            2) SOURCE_WINDOW="102_107" ;;
            3) SOURCE_WINDOW="234_239" ;;
            4) SOURCE_WINDOW="385_390" ;;
        esac
        CSV_NAME="trajectories/representative/representative_arm_trajectory_${ITEM_PADDED}_source_${SOURCE_WINDOW}s_1x_50hz.csv"
        DESCRIPTION="representative source trajectory ${ITEM_PADDED} at the original 1.0x speed"
        ;;
    synthesized_trajectory)
        if [[ ! "${ITEM}" =~ ^[1-3]$ ]]; then
            echo "Error: MODE=synthesized_trajectory requires ITEM=1..3" >&2
            exit 1
        fi
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="trajectories/synthesized/synthesized_arm_trajectory_${ITEM_PADDED}_seed20260714_measured_blend_1x_50hz.csv"
        DESCRIPTION="fixed-seed synthesized measured-trajectory blend ${ITEM_PADDED} at 1.0x speed"
        ;;
    randomized_trajectory)
        if [[ ! "${ITEM}" =~ ^[1-6]$ ]]; then
            echo "Error: MODE=randomized_trajectory requires ITEM=1..6" >&2
            exit 1
        fi
        printf -v ITEM_PADDED "%02d" "${ITEM}"
        CSV_NAME="trajectories/randomized/randomized_arm_trajectory_${ITEM_PADDED}_seed20260715_minjerk_50hz.csv"
        DESCRIPTION="velocity-limited minimum-jerk randomized-pose trajectory ${ITEM_PADDED}"
        ;;
    *)
        echo "Error: unknown MODE=${MODE}" >&2
        echo "Valid modes: all, representative_poses, synthesized_poses, randomized_poses," >&2
        echo "             representative_trajectories, synthesized_trajectories, randomized_trajectories," >&2
        echo "             down_to_horizontal," >&2
        echo "             representative_pose, synthesized_pose, randomized_pose," >&2
        echo "             representative_trajectory, synthesized_trajectory, randomized_trajectory" >&2
        exit 1
        ;;
esac

CSV_PATH="${TEST_DATA_DIR}/${CSV_NAME}"
if [[ ! -f "${CSV_PATH}" ]]; then
    echo "Error: visualization CSV does not exist: ${CSV_PATH}" >&2
    exit 1
fi

read -r CSV_DURATION_S EPISODE_LENGTH_S <<<"$("${METADATA_PYTHON}" - "${CSV_PATH}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames or "time_s" not in reader.fieldnames:
        raise SystemExit(f"CSV has no time_s column: {path}")
    final_time = 0.0
    rows = 0
    for row in reader:
        final_time = float(row["time_s"])
        rows += 1
if rows == 0:
    raise SystemExit(f"CSV has no data rows: {path}")
print(f"{final_time:.8f} {max(final_time + 1.0, 5.0):.8f}")
PY
)"

CHECKPOINT_DIR=$(dirname "${CHECKPOINT}")
CHECKPOINT_STEM=$(basename "${CHECKPOINT}")
CHECKPOINT_STEM=${CHECKPOINT_STEM%.*}
CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
CHECKPOINT_SHORT_SHA=${CHECKPOINT_SHA256:0:12}
MODEL_ID=${MODEL_ID:-${CHECKPOINT_STEM}_${CHECKPOINT_SHORT_SHA}}
if [[ ! "${MODEL_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
TEST_ID="${MODE}${ITEM:+_item${ITEM}}"
PAYLOAD_TAG=$(awk -v value="${PAYLOAD_KG}" 'BEGIN { printf "%.6g", value }' | tr '.' 'p')
REPORT_CONDITION_ID="${TEST_ID}__payload_${PAYLOAD_TAG}kg"
REPORT_DIR="${CHECKPOINT_DIR}/Test Reports/StandArmOnly"
REPORT_PATH="${REPORT_DIR}/${MODEL_ID}__${REPORT_CONDITION_ID}.md"

echo "============================================================"
echo " ArmHack Stand deterministic visualization"
echo "============================================================"
echo "Mode        : ${MODE}${ITEM:+ (item ${ITEM})}"
echo "Contents    : ${DESCRIPTION}"
echo "CSV         : ${CSV_PATH}"
echo "CSV duration: ${CSV_DURATION_S} s"
echo "Episode     : ${EPISODE_LENGTH_S} s (includes a 1 s final hold)"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Model ID    : ${MODEL_ID} (SHA-256 ${CHECKPOINT_SHA256})"
echo "Report      : ${REPORT_PATH} (written when playback exits)"
echo "Runtime RNG : disabled for arm pose/trajectory selection"
echo "Fixed payload: ${PAYLOAD_KG} kg added to each wrist-yaw link"
echo "============================================================"

cd "${LEGGED_LAB_DIR}"

TASK=LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-Play-v0 \
CHECKPOINT="${CHECKPOINT}" \
NUM_ENVS=${NUM_ENVS:-1} \
HEADLESS=${HEADLESS:-False} \
REAL_TIME=${REAL_TIME:-True} \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
EXTRA_HYDRA_ARGS="" \
FOLLOW_CAMERA=${FOLLOW_CAMERA:-True} \
CAMERA_VIEW=${CAMERA_VIEW:-front} \
CAMERA_DISTANCE=${CAMERA_DISTANCE:-3.0} \
CAMERA_HEIGHT=${CAMERA_HEIGHT:-0.7} \
CAMERA_TARGET_HEIGHT=${CAMERA_TARGET_HEIGHT:-0.7} \
MAX_STEPS=${MAX_STEPS:-} \
bash scripts/vis_isaacsim_g1_amp.sh \
    --armhack_stand_report_path "${REPORT_PATH}" \
    --armhack_stand_test_id "${TEST_ID}" \
    --armhack_stand_test_data "${CSV_PATH}" \
    --armhack_stand_manifest "${MANIFEST}" \
    --armhack_stand_payload_kg "${PAYLOAD_KG}" \
    env.episode_length_s="${EPISODE_LENGTH_S}" \
    env.upper_body_perturbation.source=csv \
    "env.upper_body_perturbation.csv_path='${CSV_PATH}'" \
    env.upper_body_perturbation.csv_randomize_start_on_reset=False \
    env.upper_body_perturbation.csv_initialize_joint_state_on_reset=True \
    env.upper_body_perturbation.csv_curriculum_enabled=False \
    env.upper_body_perturbation.csv_curriculum_motion_scale=1.0 \
    env.upper_body_perturbation.csv_loop=False \
    env.upper_body_perturbation.csv_end_margin_s=0.0 \
    "env.events.randomize_end_effector_payload.params.mass_distribution_params=[${PAYLOAD_KG},${PAYLOAD_KG}]" \
    "$@"
