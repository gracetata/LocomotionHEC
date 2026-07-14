from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
G1_AMP_ENV_CFG_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
)
G1_TASK_REGISTRY_PATH = (
    REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/__init__.py"
)
AMP_REWARDS_PATH = REPO_ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"
MOTION_DATA_MANAGER_PATH = REPO_ROOT / "source/legged_lab/legged_lab/managers/motion_data_manager.py"
TRAIN_G1_AMP_SCRIPT_PATH = REPO_ROOT / "scripts/train_g1_amp.sh"
G1_CMU_WALK_CORE_DATA_DIR = (
    REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz_task_core"
)
G1_CMU_WALK_FULL_DATA_DIR = (
    REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_50hz"
)
G1_CMU_WALK_WASHED_DATA_DIR = (
    REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/cmu_walk_washed_50hz"
)


def test_g1_cmu_walk_core_adaptive_task_is_registered_and_launchable():
    config_text = G1_AMP_ENV_CFG_PATH.read_text()
    registry_text = G1_TASK_REGISTRY_PATH.read_text()
    script_text = TRAIN_G1_AMP_SCRIPT_PATH.read_text()

    assert "G1_CMU_WALK_CORE_MOTION_DATA_DIR" in config_text
    assert "G1_CMU_WALK_FULL_MOTION_DATA_DIR" in config_text
    assert "G1_CMU_WALK_WASHED_MOTION_DATA_DIR" in config_text
    assert '"cmu_walk_50hz_task_core"' in config_text
    assert '"cmu_walk_50hz"' in config_text
    assert '"cmu_walk_washed_50hz"' in config_text
    assert 'self.motion_data.motion_dataset.root_rot_order = "xyzw"' in config_text
    assert "self.motion_data.motion_dataset.target_dof_names = G1_LOCOMOTION_JOINT_NAMES" in config_text
    assert "self.motion_data.motion_dataset.strict_dof_names = True" in config_text
    assert "class G1AmpCmuWalkCoreAdaptiveEnvCfg" in config_text
    assert "class G1AmpCmuWalkCoreAdaptiveEnvCfg_PLAY" in config_text
    assert "class G1AmpCmuWalkFullAdaptiveEnvCfg" in config_text
    assert "class G1AmpCmuWalkFullAdaptiveEnvCfg_PLAY" in config_text
    assert "class G1AmpCmuWalkWashedAdaptiveEnvCfg" in config_text
    assert "class G1AmpCmuWalkWashedAdaptiveEnvCfg_PLAY" in config_text
    assert "low_speed_threshold=0.40" in config_text
    assert "high_speed_ang_vel_z_std=0.10" in config_text
    assert "lin_vel_x=(-0.40, 1.00)" in config_text
    assert "lin_vel_y=(-0.30, 0.30)" in config_text
    assert "ang_vel_z=(-0.30, 0.30)" in config_text
    assert "low_speed_ang_vel_z=(-0.50, 0.50)" in config_text
    assert "LeggedLab-Isaac-AMP-G1-CmuWalkCore-Adaptive-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-G1-CmuWalkCore-Adaptive-Play-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-G1-CmuWalkFull-Adaptive-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-G1-CmuWalkFull-Adaptive-Play-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0" in registry_text
    assert "LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Play-v0" in registry_text
    assert "ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkCore-Adaptive-v0" in script_text
    assert "ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkFull-Adaptive-v0" in script_text
    assert "ROBOT_ASSET=g1_29dof TASK=LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0" in script_text
    assert "RUN_NAME=smoke_g1_cmu_walk_washed_adaptive STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.35" in script_text
    assert "RUN_NAME=orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000 STYLE_REWARD_SCALE=8.0 TASK_STYLE_LERP=0.35" in script_text
    assert "MAX_ITERATIONS=3000" in script_text
    assert 'env.rewards.track_lin_vel_xy_exp.params.std="${TRACK_LIN_STD}"' in script_text
    assert 'env.rewards.track_ang_vel_z_exp.params.std="${TRACK_ANG_STD}"' in script_text


def test_g1_cmu_walk_core_dataset_is_available_for_static_config():
    motion_files = sorted(G1_CMU_WALK_CORE_DATA_DIR.glob("*.pkl"))
    full_motion_files = sorted(G1_CMU_WALK_FULL_DATA_DIR.glob("*.pkl"))
    washed_motion_files = sorted(G1_CMU_WALK_WASHED_DATA_DIR.glob("*.pkl"))
    motion_manager_text = MOTION_DATA_MANAGER_PATH.read_text()

    assert G1_CMU_WALK_CORE_DATA_DIR.is_dir()
    assert G1_CMU_WALK_FULL_DATA_DIR.is_dir()
    assert G1_CMU_WALK_WASHED_DATA_DIR.is_dir()
    assert len(motion_files) == 29
    assert len(full_motion_files) == 109
    assert len(washed_motion_files) == 84
    assert "def _root_quat_from_raw" in motion_manager_text
    assert "def _dof_pos_from_raw" in motion_manager_text
    assert "root_rot[:, [3, 0, 1, 2]]" in motion_manager_text
    assert "Reordered dof_pos by dof_names" in motion_manager_text
    assert 'motion_raw_data.get("loop_mode", LoopMode.CLAMP.value)' in motion_manager_text
    assert '"local_body_pos" in motion_raw_data' in motion_manager_text
    assert "math_utils.quat_apply(root_quat_w, key_body_pos_b)" in motion_manager_text


def test_g1_cmu_walk_core_adaptive_reward_pruning_is_configured():
    config_text = G1_AMP_ENV_CFG_PATH.read_text()
    rewards_text = AMP_REWARDS_PATH.read_text()

    assert "def track_lin_vel_xy_adaptive_exp" in rewards_text
    assert "def track_ang_vel_z_adaptive_exp" in rewards_text
    assert "def _adaptive_tracking_sigma" in rewards_text
    assert "Adaptive Tracking/{key}_sigma" in rewards_text
    assert "def _configure_cmu_walk_core_adaptive_rewards" in config_text
    assert "def _configure_cmu_walk_washed_adaptive_rewards" in config_text
    assert "func=mdp.track_lin_vel_xy_adaptive_exp" in config_text
    assert "func=mdp.track_ang_vel_z_adaptive_exp" in config_text
    assert '"ema_decay": 0.995' in config_text
    assert '"min_sigma": 0.04' in config_text
    assert '"min_sigma": 0.01' in config_text
    assert "rewards.track_ang_vel_z_exp.weight = 1.25" in config_text
    assert 'rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.7)' in config_text
    assert 'rewards.track_ang_vel_z_exp.params["ema_decay"] = 0.99' in config_text
    assert 'rewards.track_ang_vel_z_exp.params["min_sigma"] = 0.04' in config_text
    assert "rewards.torso_roll_pitch_l2.weight = -0.05" in config_text
    assert "rewards.feet_air_time = None" in config_text
    assert "rewards.feet_slide = None" in config_text
    assert "rewards.joint_deviation_hip = None" in config_text
    assert "rewards.joint_deviation_arms = None" in config_text
    assert "rewards.joint_deviation_waist = None" in config_text
    assert "rewards.termination_penalty = None" in config_text
