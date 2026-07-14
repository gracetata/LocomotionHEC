import ast
import pickle
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWER_SCRIPT = REPO_ROOT / "scripts/tools/vis_g1_amp_reference_motion.py"
VIEWER_WRAPPER = REPO_ROOT / "scripts/vis_g1_amp_reference_motion.sh"
G1_AMP_ENV_CFG_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
)
G1_CMU_WALK_CORE_DATA_DIR = (
    REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz_task_core"
)


def _install_numpy_pickle_shim():
    if hasattr(np, "_core"):
        return
    np._core = np.core
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray


def _load_first_cmu_motion():
    _install_numpy_pickle_shim()
    motion_path = sorted(G1_CMU_WALK_CORE_DATA_DIR.glob("*.pkl"))[0]
    with motion_path.open("rb") as f:
        return pickle.load(f)


def _g1_locomotion_joint_names():
    module = ast.parse(G1_AMP_ENV_CFG_PATH.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "G1_LOCOMOTION_JOINT_NAMES":
                    return ast.literal_eval(node.value)
    raise AssertionError("G1_LOCOMOTION_JOINT_NAMES not found")


def test_reference_motion_viewer_exposes_required_modes_and_defaults():
    viewer_text = VIEWER_SCRIPT.read_text()
    wrapper_text = VIEWER_WRAPPER.read_text()

    assert "training_current" in viewer_text
    assert "name_aligned" in viewer_text
    assert "name_aligned_xyzw" in viewer_text
    assert "--motion_selection" in viewer_text
    assert "cycle_all" in viewer_text
    assert "CycleAllState" in viewer_text
    assert "path_turn" in viewer_text
    assert "misaligned" in viewer_text
    assert "heading-yaw" in viewer_text
    assert "cmu_walk_50hz_task_core" in viewer_text
    assert "default=0.15" in viewer_text
    assert "disable_gravity = True" in viewer_text
    assert "collision_enabled=False" in viewer_text
    assert "activate_contact_sensors = False" in viewer_text
    assert "ROBOT_ASSET=g1_29dof" in wrapper_text
    assert "MOTION_SELECTION=cycle_all" in wrapper_text
    assert "MOTION_SELECTION=path_turn" in wrapper_text
    assert "--motion_selection" in wrapper_text
    assert "VIEW_MODE=name_aligned_xyzw" in wrapper_text
    assert "VIEW_MODE=training_current" in wrapper_text
    assert "MAX_STEPS=20 VIEW_MODE=name_aligned_xyzw NUM_ENVS=2" in wrapper_text


def test_cmu_walk_core_joint_order_requires_name_alignment():
    motion = _load_first_cmu_motion()
    target_names = _g1_locomotion_joint_names()
    source_names = motion["dof_names"]

    assert len(source_names) == len(target_names) == 29
    assert source_names != target_names
    mismatches = [(idx, target, source) for idx, (target, source) in enumerate(zip(target_names, source_names)) if target != source]
    assert len(mismatches) == 27

    source_by_name = {name: idx for idx, name in enumerate(source_names)}
    permutation = [source_by_name[name] for name in target_names]
    assert permutation[:15] == [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10]
    assert sorted(permutation) == list(range(29))


def test_cmu_walk_core_root_quaternion_looks_xyzw_and_viewer_converts_it():
    motion = _load_first_cmu_motion()
    quat = motion["root_rot"][0]
    viewer_text = VIEWER_SCRIPT.read_text()

    assert abs(float(quat[3])) > 0.99
    assert max(abs(float(value)) for value in quat[:3]) < 0.05
    assert "def _convert_xyzw_to_wxyz" in viewer_text
    assert "quat_xyzw[..., [3, 0, 1, 2]]" in viewer_text
    assert 'view_mode == "name_aligned_xyzw"' in viewer_text
