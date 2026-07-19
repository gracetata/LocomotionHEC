"""Static contract tests that do not require launching Isaac Sim."""

from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent / "legged_lab"
PROJECT_ROOT = TEST_DIR.parents[2]
TASK_DIR = (
    PACKAGE_ROOT
    / "tasks"
    / "locomotion"
    / "amp"
    / "config"
    / "g1_extreme_stand_recovery"
)
ENV_CFG = TASK_DIR / "g1_extreme_stand_recovery_env_cfg.py"
TASK_INIT = TASK_DIR / "__init__.py"
RUNNER_CFG = TASK_DIR / "agents" / "rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "train_g1_extreme_stand_recovery.sh"
VIS_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "vis_g1_extreme_stand_recovery.sh"
MUJOCO_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "val_mujoco_g1_extreme_stand_recovery.sh"
MUJOCO_ADAPTER = PROJECT_ROOT.parent / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "extreme_stand_recovery.py"
MUJOCO_RUNNER = PROJECT_ROOT.parent / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "deploy_mujoco_g1_amp.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_extreme_stand_recovery_files_exist():
    for path in (
        ENV_CFG,
        TASK_INIT,
        RUNNER_CFG,
        TRAIN_SCRIPT,
        VIS_SCRIPT,
        MUJOCO_SCRIPT,
        MUJOCO_ADAPTER,
        MUJOCO_RUNNER,
    ):
        assert path.is_file(), f"Missing Extreme Stand file: {path}"


def test_task_uses_standard_amp_env_and_full_body_policy_contract():
    task_text = _read(TASK_INIT)
    cfg_text = _read(ENV_CFG)
    assert 'entry_point="legged_lab.envs:ManagerBasedAmpEnv"' in task_text
    assert "G1PerturbAmpEnv" not in task_text
    assert "G1WalkPerturbAmpEnv" not in task_text
    assert "upper_body_perturbation" not in cfg_text
    assert "G1_LOCOMOTION_JOINT_NAMES" in cfg_text
    assert "self.actions.joint_pos.joint_names = G1_LOCOMOTION_JOINT_NAMES" in cfg_text
    assert "self.observations.policy.joint_pos.params" in cfg_text
    assert "self.observations.policy.joint_vel.params" in cfg_text


def test_training_distribution_has_additive_joint_noise_and_multi_body_forces():
    cfg_text = _read(ENV_CFG)
    assert cfg_text.count("func=mdp.reset_joints_by_offset") == 3
    assert "reset_leg_joints_with_noise" in cfg_text
    assert "reset_waist_joints_with_noise" in cfg_text
    assert "reset_arm_joints_with_noise" in cfg_text
    assert "random_torso_external_wrench" in cfg_text
    assert "random_pelvis_external_wrench" in cfg_text
    assert "random_arm_external_wrench" in cfg_text
    assert "random_leg_external_wrench" in cfg_text
    assert "func=mdp.push_by_setting_velocity" in cfg_text
    assert "default_joint_pose_exp" in cfg_text
    assert "weight=-1000.0" in cfg_text


def test_launchers_never_enable_armhack_action_adapters():
    train_text = _read(TRAIN_SCRIPT)
    vis_text = _read(VIS_SCRIPT)
    mujoco_text = _read(MUJOCO_SCRIPT)
    adapter_text = _read(MUJOCO_ADAPTER)
    assert "96 observations -> 29 full-body joint actions" in train_text
    assert "ExtremeStandRecovery-v0" in train_text
    assert "ExtremeStandRecovery-Play-v0" in vis_text
    assert "G1_AMP_ARMHACK_STAND_ENABLE=False" in mujoco_text
    assert "G1_AMP_ARMHACK_WALK_ENABLE=False" in mujoco_text
    assert '"action_override": False' in adapter_text
    assert "compose_action" not in adapter_text


def test_generic_mujoco_runner_keeps_recovery_separate_from_armhack():
    runner_text = _read(MUJOCO_RUNNER)
    assert "ExtremeStandRecoveryPerturbation" in runner_text
    assert "cannot be combined with an ArmHack action adapter" in runner_text
    assert "extreme_stand_recovery.update_external_wrench(data, sim_time)" in runner_text
    assert 'report["extreme_stand_recovery"]' in runner_text
