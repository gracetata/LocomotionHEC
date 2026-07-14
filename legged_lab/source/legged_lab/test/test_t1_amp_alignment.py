import importlib.util
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
T1_SYMMETRY_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/symmetry/t1.py"
)
T1_AMP_ENV_CFG_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/t1/t1_amp_env_cfg.py"
)
T1_TASK_REGISTRY_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/t1/__init__.py"
)
T1_AGENT_CFG_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/t1/agents/rsl_rl_ppo_cfg.py"
)
T1_ROBOPARTY_PATH = REPO_ROOT / "source/legged_lab/legged_lab/assets/roboparty.py"
T1_EVENTS_PATH = REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/events.py"
T1_REWARDS_PATH = REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"
T1_URDF_PATH = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/T1_29dof/T1_29dof.urdf"
T1_ARM_STYLE_PRIOR_SCRIPT_PATH = REPO_ROOT / "scripts/tools/train_t1_arm_style_prior.py"
T1_CONVERT_AMP_PATH = REPO_ROOT / "scripts/tools/retarget/convert_t1_pkl_to_amp_format.py"
T1_RETARGET_TOOLS_DIR = REPO_ROOT / "scripts/tools/retarget"
T1_GMR_MOTION_DIR = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/T1_GMR/accad_g1used_50hz"
T1_LAB_MOTION_DIR = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/T1_Lab/accad_g1used_50hz"


def _load_t1_symmetry_module():
    module_spec = importlib.util.spec_from_file_location("t1_symmetry_under_test", T1_SYMMETRY_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _load_t1_convert_module():
    if str(T1_RETARGET_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(T1_RETARGET_TOOLS_DIR))
    module_spec = importlib.util.spec_from_file_location("t1_convert_under_test", T1_CONVERT_AMP_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _install_numpy_core_pickle_shim():
    if hasattr(np, "_core"):
        return

    import importlib
    import numpy.core as numpy_core

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(numpy_core.__dict__)
    sys.modules.setdefault("numpy._core", shim)
    for submodule_name in ("multiarray", "umath", "numeric", "numerictypes", "fromnumeric"):
        full_name = f"numpy._core.{submodule_name}"
        if full_name not in sys.modules:
            sys.modules[full_name] = importlib.import_module(f"numpy.core.{submodule_name}")


def test_t1_symmetry_flips_waist_yaw_for_no_head_layout():
    t1_symmetry = _load_t1_symmetry_module()
    joint_data = torch.zeros(1, 27)
    joint_data[0, 14] = 0.37

    mirrored = t1_symmetry._switch_t1_27dof_joints_left_right(joint_data)

    assert torch.isclose(mirrored[0, 14], torch.tensor(-0.37))


def test_t1_symmetry_flips_waist_yaw_for_full_layout():
    t1_symmetry = _load_t1_symmetry_module()
    joint_data = torch.zeros(1, 29)
    joint_data[0, 16] = -0.42

    mirrored = t1_symmetry._switch_t1_29dof_joints_left_right(joint_data)

    assert torch.isclose(mirrored[0, 16], torch.tensor(0.42))


def test_t1_lab_motion_preserves_gmr_joint_zero_convention():
    _install_numpy_core_pickle_shim()
    t1_convert = _load_t1_convert_module()
    motion_name = "B10_-__Walk_turn_left_45_stageii.pkl"
    raw_motion = joblib.load(T1_GMR_MOTION_DIR / motion_name)
    lab_motion = joblib.load(T1_LAB_MOTION_DIR / motion_name)

    raw_joint_pos = np.asarray(raw_motion["dof_pos"], dtype=np.float32)
    expected_joint_pos = np.zeros((raw_joint_pos.shape[0], 29), dtype=np.float32)
    for lab_index, raw_index in enumerate(t1_convert.URDF_FROM_DATASET):
        if raw_index >= 0:
            expected_joint_pos[:, lab_index] = raw_joint_pos[:, raw_index]

    np.testing.assert_allclose(lab_motion["dof_pos"], expected_joint_pos, atol=1.0e-7, rtol=0.0)


def test_t1_convert_pkl_to_amp_format_preserves_gmr_joint_angles(tmp_path):
    t1_convert = _load_t1_convert_module()
    raw_joint_pos = np.arange(54, dtype=np.float32).reshape(2, 27) * 0.01

    raw_motion = {
        "fps": 50,
        "root_pos": np.zeros((2, 3), dtype=np.float32),
        "root_rot": np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
        "dof_pos": raw_joint_pos,
        "local_body_pos": np.zeros((2, 0, 3), dtype=np.float32),
        "link_body_list": [],
    }
    src_path = tmp_path / "raw.pkl"
    dst_path = tmp_path / "amp.pkl"
    joblib.dump(raw_motion, src_path)

    t1_convert.convert_pkl(str(src_path), str(dst_path))
    amp_motion = joblib.load(dst_path)

    expected_joint_pos = np.zeros((2, 29), dtype=np.float32)
    for lab_index, raw_index in enumerate(t1_convert.URDF_FROM_DATASET):
        if raw_index >= 0:
            expected_joint_pos[:, lab_index] = raw_joint_pos[:, raw_index]
    np.testing.assert_allclose(amp_motion["dof_pos"], expected_joint_pos, atol=1.0e-7, rtol=0.0)


def test_t1_lab_has_no_gmr_joint_zero_offsets():
    t1_convert = _load_t1_convert_module()

    assert t1_convert.T1_LAB_BAKED_ZERO_OFFSETS == {}


def test_t1_action_config_preserves_no_head_joint_order():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()

    assert "self.actions.joint_pos.preserve_order = True" in config_text


def test_t1_shoulder_pitch_guard_rewards_are_configured():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()

    assert "shoulder_pitch_pos_limits" in config_text
    assert "shoulder_pitch_action_l2" in config_text
    assert "params={\"action_indices\": [0, 7]}" in config_text
    assert "joint_deviation_arms = None" in config_text


def test_t1_forward_back_task_uses_filtered_demo_space():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()
    registry_text = T1_TASK_REGISTRY_PATH.read_text()

    assert "class T1AmpForwardBackEnvCfg" in config_text
    assert "t1_29dof_forward_back_filtered_50hz" in config_text
    assert "self.commands.base_velocity.heading_command = False" in config_text
    assert "self.commands.base_velocity.ranges.lin_vel_x = _metadata_range" in config_text
    assert "LeggedLab-Isaac-AMP-T1-FwdBack-v0" in registry_text


def test_t1_feet_air_time_reward_is_disabled():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()

    assert "feet_air_time = None" in config_text
    assert "feet_air_time_positive_biped" not in config_text


def test_t1_demo_static_normalizer_agent_is_configured():
    agent_text = T1_AGENT_CFG_PATH.read_text()
    registry_text = T1_TASK_REGISTRY_PATH.read_text()

    assert "class T1RslRlOnPolicyRunnerAmpDemoNormCfg" in agent_text
    assert "disc_learning_rate=1.0e-5" in agent_text
    assert "normalizer_mode = \"demo_static\"" in agent_text
    assert "demo_normalizer_init_batches = 16" in agent_text
    assert "LeggedLab-Isaac-AMP-T1-FwdBack-DemoNorm-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-T1-CmuWalkCore-DemoNorm-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-T1-CmuWalkCore-CommandOnly-DemoNorm-v0" in registry_text


def test_t1_cmu_walk_core_command_only_reward_variant_is_registered():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()
    registry_text = T1_TASK_REGISTRY_PATH.read_text()

    assert "def _keep_only_command_tracking_rewards" in config_text
    assert "class T1AmpCmuWalkCoreCommandOnlyEnvCfg" in config_text
    assert "rewards.flat_orientation_l2 = None" in config_text
    assert "rewards.termination_penalty = None" in config_text
    assert "T1AmpCmuWalkCoreCommandOnlyEnvCfg" in registry_text


def test_t1_arm_style_prior_reward_is_configured():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()
    registry_text = T1_TASK_REGISTRY_PATH.read_text()
    rewards_text = T1_REWARDS_PATH.read_text()
    prior_script_text = T1_ARM_STYLE_PRIOR_SCRIPT_PATH.read_text()

    assert "DEFAULT_ARM_STYLE_PRIOR_PATH" in config_text
    assert "class T1AmpCmuWalkCoreCommandOnlyArmPriorEnvCfg" in config_text
    assert "func=mdp.arm_style_prior_exp" in config_text
    assert "LeggedLab-Isaac-AMP-T1-CmuWalkCore-CommandOnlyArmPrior-DemoNorm-v0" in registry_text
    assert "def arm_style_prior_exp" in rewards_text
    assert "class ArmStylePrior" in prior_script_text


def test_t1_natural_arm_default_init_is_configured():
    robot_text = T1_ROBOPARTY_PATH.read_text()

    assert '"Left_Shoulder_Roll": -1.25' in robot_text
    assert '"Right_Shoulder_Roll": 1.25' in robot_text
    assert '"Left_Hand_Roll": -0.26' in robot_text
    assert '"Right_Hand_Roll": 0.26' in robot_text
    assert '"Right_Elbow_Yaw": 0.09' in robot_text


def test_t1_amp_rsi_ratio_reset_is_configured():
    config_text = T1_AMP_ENV_CFG_PATH.read_text()
    events_text = T1_EVENTS_PATH.read_text()

    assert "DEFAULT_RSI_RATIO = 0.5" in config_text
    assert "func=mdp.ref_state_init_subset" in config_text
    assert '"rsi_ratio": DEFAULT_RSI_RATIO' in config_text
    assert "def ref_state_init_subset" in events_text
    assert "motion_dataset: str | None = None" in events_text


def test_t1_xml_zero_arm_limits_allow_demo_pose():
    urdf_text = T1_URDF_PATH.read_text()

    assert 'lower="-1.74"' in urdf_text
    assert 'upper="1.57"' in urdf_text
    assert 'lower="-1.57"' in urdf_text
    assert 'upper="1.74"' in urdf_text
    assert urdf_text.count('lower="-2.27"') >= 5
    assert urdf_text.count('upper="2.27"') >= 6