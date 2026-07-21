#!/usr/bin/env bash
# Export and evaluate the ArmHack Stand policy with the project's G1 MuJoCo sim2sim runner.
#
# Examples:
#   bash scripts/val_mujoco_g1_armhack_stand.sh
#   PAYLOAD_KG=1.0 bash scripts/val_mujoco_g1_armhack_stand.sh
#   USE_GLFW=True REAL_TIME=True MODE=all bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=randomized_trajectory ITEM=1 USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=generated_random_trajectory RANDOM_SEED=20260720 RANDOM_WAYPOINTS=8 \
#     USE_GLFW=True REAL_TIME=True bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=down_to_horizontal USE_GLFW=True bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=default_forward_return_down USE_GLFW=True REAL_TIME=True \
#     bash scripts/val_mujoco_g1_armhack_stand.sh
#   MODE=interactive bash scripts/val_mujoco_g1_armhack_stand.sh

set -euo pipefail

LEGGED_LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "${LEGGED_LAB_DIR}/.." && pwd)
TEST_DATA_DIR="${LEGGED_LAB_DIR}/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly"
MANIFEST="${TEST_DATA_DIR}/manifest.json"
RANDOM_POSE_BANK="${LEGGED_LAB_DIR}/Reference Data/ArmHack/StandPerturb/RandomizedTraining/random_arm_pose_bank_seed20260715.json"
RANDOM_TRAJECTORY_GENERATOR="${LEGGED_LAB_DIR}/scripts/tools/generate_armhack_stand_random_trajectory.py"
ARM_PRESET_PATH=${ARM_PRESET_PATH:-${LEGGED_LAB_DIR}/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json}
DEFAULT_CHECKPOINT="${PROJECT_ROOT}/checkpoint/stand/model_2999.pt"
PACKAGED_POLICY="${PROJECT_ROOT}/use/armhack_stand_model_2999.torchscript.pt"
PACKAGED_ONNX="${PROJECT_ROOT}/use/armhack_stand_model_2999.onnx"
PACKAGED_METADATA="${PROJECT_ROOT}/use/armhack_stand_model_2999.deploy.json"

MODE=${MODE:-all}
ITEM=${ITEM:-}
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f}
POLICY_PATH=${POLICY_PATH:-${PACKAGED_POLICY}}
ONNX_PATH=${ONNX_PATH:-${PACKAGED_ONNX}}
DEPLOY_METADATA_PATH=${DEPLOY_METADATA_PATH:-${PACKAGED_METADATA}}
PAYLOAD_KG=${PAYLOAD_KG:-0.0}
if [[ "${MODE}" == "interactive" ]]; then
    USE_GLFW=${USE_GLFW:-True}
    REAL_TIME=${REAL_TIME:-True}
else
    USE_GLFW=${USE_GLFW:-False}
    REAL_TIME=${REAL_TIME:-False}
fi
FORCE_EXPORT=${FORCE_EXPORT:-False}
ISAAC_PYTHON=${ISAAC_PYTHON:-${HOME}/miniconda3/envs/env_leglab/bin/python}
if [[ -z "${UNITREE_PYTHON:-}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/env_leglab/bin/python" \
        "${HOME}/anaconda3/envs/gmr/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            UNITREE_PYTHON="${candidate}"
            break
        fi
    done
fi
METADATA_PYTHON=${METADATA_PYTHON:-python3}
MUJOCO_CPU_THREADS=${MUJOCO_CPU_THREADS:-1}
RANDOM_SEED=${RANDOM_SEED:-20260720}
RANDOM_WAYPOINTS=${RANDOM_WAYPOINTS:-8}
RANDOM_HOLD_S=${RANDOM_HOLD_S:-0.5}
RANDOM_TRANSITION_S=${RANDOM_TRANSITION_S:-2.0}
INTERACTIVE_AUTO_ENTER_S=${INTERACTIVE_AUTO_ENTER_S:--1.0}
INTERACTIVE_AUTO_SPACE_INTERVAL_S=${INTERACTIVE_AUTO_SPACE_INTERVAL_S:--1.0}
INTERACTIVE_AUTO_SPACE_MAX_SWITCHES=${INTERACTIVE_AUTO_SPACE_MAX_SWITCHES:-0}
INTERACTIVE_TRANSITION_S=${INTERACTIVE_TRANSITION_S:-7.5}

bool_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Error: ArmHack Stand schema v5 manifest does not exist: ${MANIFEST}" >&2
    exit 1
fi
if [[ "${MODE}" == "interactive" && ! -f "${ARM_PRESET_PATH}" ]]; then
    echo "Error: interactive Stand arm preset does not exist: ${ARM_PRESET_PATH}" >&2
    exit 1
fi
if [[ -z "${UNITREE_PYTHON:-}" || ! -x "${UNITREE_PYTHON}" ]]; then
    echo "Error: UNITREE_PYTHON is not set or executable: ${UNITREE_PYTHON:-<unset>}" >&2
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
    interactive)
        CSV_NAME="special/arms_down_flat_forward_return_flat_25p5s_50hz.csv"
        DESCRIPTION="ENTER: policy-on natural-down -> flat-default -> forward -> flat-default; SPACE after init"
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
    default_forward_return_down)
        CSV_NAME="special/arms_default_forward_return_down_39p5s_50hz.csv"
        DESCRIPTION="default hold, simultaneous forward reach, simultaneous return, then natural arms-down"
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
    generated_random_trajectory)
        [[ "${RANDOM_SEED}" =~ ^[0-9]+$ ]] || {
            echo "Error: RANDOM_SEED must be a non-negative integer." >&2
            exit 1
        }
        [[ "${RANDOM_WAYPOINTS}" =~ ^[0-9]+$ ]] || {
            echo "Error: RANDOM_WAYPOINTS must be an integer within [2, 512]." >&2
            exit 1
        }
        [[ "${RANDOM_HOLD_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
            echo "Error: RANDOM_HOLD_S must be a non-negative decimal number." >&2
            exit 1
        }
        [[ "${RANDOM_TRANSITION_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
            echo "Error: RANDOM_TRANSITION_S must be a positive decimal number." >&2
            exit 1
        }
        if [[ ! -f "${RANDOM_POSE_BANK}" || ! -f "${RANDOM_TRAJECTORY_GENERATOR}" ]]; then
            echo "Error: random pose bank or trajectory generator is missing." >&2
            echo "Pose bank: ${RANDOM_POSE_BANK}" >&2
            echo "Generator: ${RANDOM_TRAJECTORY_GENERATOR}" >&2
            exit 1
        fi
        HOLD_TAG=${RANDOM_HOLD_S//./p}
        TRANSITION_TAG=${RANDOM_TRANSITION_S//./p}
        GENERATED_TAG="seed${RANDOM_SEED}_wp${RANDOM_WAYPOINTS}_hold${HOLD_TAG}s_transition${TRANSITION_TAG}s"
        GENERATED_DIR="${TEST_DATA_DIR}/trajectories/generated"
        CSV_PATH="${GENERATED_DIR}/generated_random_arm_trajectory_${GENERATED_TAG}_50hz.csv"
        GENERATED_METADATA_PATH="${GENERATED_DIR}/generated_random_arm_trajectory_${GENERATED_TAG}_50hz.json"
        "${UNITREE_PYTHON}" "${RANDOM_TRAJECTORY_GENERATOR}" \
            --pose-bank "${RANDOM_POSE_BANK}" \
            --manifest "${MANIFEST}" \
            --output "${CSV_PATH}" \
            --metadata "${GENERATED_METADATA_PATH}" \
            --seed "${RANDOM_SEED}" \
            --waypoint-count "${RANDOM_WAYPOINTS}" \
            --hold-s "${RANDOM_HOLD_S}" \
            --transition-s "${RANDOM_TRANSITION_S}" \
            --fps 50
        DESCRIPTION="reproducible arm-only random pose-bank trajectory (${RANDOM_WAYPOINTS} waypoints, seed ${RANDOM_SEED})"
        ;;
    *)
        echo "Error: unsupported MODE=${MODE}" >&2
        exit 1
        ;;
esac

if [[ -z "${CSV_PATH:-}" ]]; then
    CSV_PATH="${TEST_DATA_DIR}/${CSV_NAME}"
fi
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
if [[ "${MODE}" == "interactive" ]]; then
    SIMULATION_DURATION=${SIMULATION_DURATION:-300.0}
else
    SIMULATION_DURATION=${SIMULATION_DURATION:-${DEFAULT_SIMULATION_DURATION}}
fi

CHECKPOINT_DIR=$(dirname "${CHECKPOINT}")
CHECKPOINT_STEM=$(basename "${CHECKPOINT}")
CHECKPOINT_STEM=${CHECKPOINT_STEM%.*}
if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
    if [[ "${CHECKPOINT_SHA256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
        echo "Error: Stand checkpoint SHA-256 mismatch." >&2
        exit 1
    fi
else
    CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256}"
fi
CHECKPOINT_SHORT_SHA=${CHECKPOINT_SHA256:0:12}
MODEL_ID=${MODEL_ID:-${CHECKPOINT_STEM}_${CHECKPOINT_SHORT_SHA}}
if [[ "${MODE}" == "generated_random_trajectory" ]]; then
    TEST_ID="${MODE}_${GENERATED_TAG}"
else
    TEST_ID="${MODE}${ITEM:+_item${ITEM}}"
fi
PAYLOAD_TAG=$(awk -v value="${PAYLOAD_KG}" 'BEGIN { printf "%.6g", value }' | tr '.' 'p')

EXPORT_DIR="${CHECKPOINT_DIR}/MuJoCo Export/StandArmOnly"
REPORT_DIR="${CHECKPOINT_DIR}/Test Reports/StandArmOnlyMuJoCo"
REPORT_STEM="${MODEL_ID}__mujoco__${TEST_ID}__payload_${PAYLOAD_TAG}kg"
REPORT_PATH="${REPORT_DIR}/${REPORT_STEM}.md"
METRICS_PATH="${REPORT_DIR}/${REPORT_STEM}.json"

if bool_true "${FORCE_EXPORT}" || [[ ! -f "${POLICY_PATH}" || ! -f "${ONNX_PATH}" || ! -f "${DEPLOY_METADATA_PATH}" ]]; then
    [[ -f "${CHECKPOINT}" ]] || { echo "Error: packaged policy 缺失，且没有可导出的 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
    [[ -x "${ISAAC_PYTHON}" ]] || { echo "Error: 导出所需 ISAAC_PYTHON 不可执行: ${ISAAC_PYTHON}" >&2; exit 1; }
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
if [[ -n "${GENERATED_METADATA_PATH:-}" ]]; then
    echo "Metadata    : ${GENERATED_METADATA_PATH}"
fi
echo "CSV duration: ${CSV_DURATION_S} s"
echo "Sim duration: ${SIMULATION_DURATION} s"
if [[ -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint  : ${CHECKPOINT}"
else
    echo "Checkpoint  : not packaged (verified SHA metadata only)"
fi
echo "Model SHA   : ${CHECKPOINT_SHA256}"
echo "Policy      : ${POLICY_PATH}"
echo "Payload     : ${PAYLOAD_KG} kg per wrist-yaw link"
echo "GLFW/RT     : ${USE_GLFW}/${REAL_TIME}"
if [[ "${MODE}" == "interactive" ]]; then
    echo "Keys        : ENTER=policy/debug, SPACE=next arm pose after initialization, Q=stop"
    echo "Auto keys   : enter=${INTERACTIVE_AUTO_ENTER_S}s, space_interval=${INTERACTIVE_AUTO_SPACE_INTERVAL_S}s, max=${INTERACTIVE_AUTO_SPACE_MAX_SWITCHES}"
fi
echo "Report      : ${REPORT_PATH}"
echo "============================================================"

export G1_AMP_ARMHACK_STAND_ENABLE=True
export G1_AMP_ARMHACK_STAND_CSV_PATH="${CSV_PATH}"
export G1_AMP_ARMHACK_STAND_MANIFEST_PATH="${MANIFEST}"
export G1_AMP_ARMHACK_STAND_CHECKPOINT_PATH="${CHECKPOINT}"
export G1_AMP_ARMHACK_STAND_CHECKPOINT_SHA256="${CHECKPOINT_SHA256}"
export G1_AMP_ARMHACK_STAND_REPORT_PATH="${REPORT_PATH}"
export G1_AMP_ARMHACK_STAND_TEST_ID="${TEST_ID}"
export G1_AMP_ARMHACK_STAND_PAYLOAD_KG="${PAYLOAD_KG}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_ENABLE=$([[ "${MODE}" == "interactive" ]] && echo True || echo False)
export G1_AMP_ARMHACK_STAND_PRESET_PATH="${ARM_PRESET_PATH}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_TRANSITION_S="${INTERACTIVE_TRANSITION_S}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_ENTER_S="${INTERACTIVE_AUTO_ENTER_S}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_SPACE_INTERVAL_S="${INTERACTIVE_AUTO_SPACE_INTERVAL_S}"
export G1_AMP_ARMHACK_STAND_INTERACTIVE_AUTO_SPACE_MAX_SWITCHES="${INTERACTIVE_AUTO_SPACE_MAX_SWITCHES}"

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
