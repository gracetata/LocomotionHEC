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
POSE_V2_TRAIN_SCRIPT = (
    PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "train_g1_extreme_stand_recovery_pose_v2.sh"
)
VIS_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "vis_g1_extreme_stand_recovery.sh"
MUJOCO_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "val_mujoco_g1_extreme_stand_recovery.sh"
ROOT_MUJOCO_SCRIPT = PROJECT_ROOT.parent / "scripts" / "sim2sim_g1_extreme_stand_recovery_mujoco.sh"
EXPORT_SCRIPT = PROJECT_ROOT.parent / "scripts" / "export_g1_extreme_stand_recovery.sh"
REAL_DEPLOY_SCRIPT = PROJECT_ROOT.parent / "scripts" / "deploy_real_g1_extreme_stand_recovery_onnx.sh"
SUMMARY_SCRIPT = PROJECT_ROOT.parent / "scripts" / "summarize_g1_extreme_stand_recovery_mujoco.py"
POSE_RECOVERY_SCRIPT = PROJECT_ROOT.parent / "scripts" / "test_g1_extreme_stand_random_pose_recovery_mujoco.sh"
MUJOCO_ADAPTER = PROJECT_ROOT.parent / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "extreme_stand_recovery.py"
MUJOCO_RUNNER = PROJECT_ROOT.parent / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "deploy_mujoco_g1_amp.py"
AMP_EVENTS = PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "mdp" / "events.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_extreme_stand_recovery_files_exist():
    for path in (
        ENV_CFG,
        TASK_INIT,
        RUNNER_CFG,
        TRAIN_SCRIPT,
        POSE_V2_TRAIN_SCRIPT,
        VIS_SCRIPT,
        MUJOCO_SCRIPT,
        ROOT_MUJOCO_SCRIPT,
        EXPORT_SCRIPT,
        REAL_DEPLOY_SCRIPT,
        SUMMARY_SCRIPT,
        POSE_RECOVERY_SCRIPT,
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


def test_pose_v2_rewards_use_default_generalized_cartesian_and_foot_references():
    cfg_text = _read(ENV_CFG)
    rewards_text = _read(TASK_DIR / "rewards.py")
    train_text = _read(POSE_V2_TRAIN_SCRIPT)
    events_text = _read(AMP_EVENTS)
    assert "cache_default_cartesian_pose" in cfg_text
    assert "cache_default_feet_pose" in cfg_text
    assert "default_leg_joint_pose_exp" in cfg_text
    assert "default_key_body_pose_exp" in cfg_text
    assert "default_feet_distance_l2" in cfg_text
    assert "weight=5.0" in cfg_text
    assert "weight=3.0" in cfg_text
    assert "weight=2.5" in cfg_text
    assert "weight=-8.0" in cfg_text
    assert "torch.square(current_distance - reference_distance)" in rewards_text
    assert "env_ids: torch.Tensor | None," in events_text
    assert "model_4999.pt" in train_text
    assert "16af8b298fe4789194b6f798ee5591a3cc61edab307724a82906cc5e9a038fe7" in train_text


def test_launchers_never_enable_armhack_action_adapters():
    train_text = _read(TRAIN_SCRIPT)
    vis_text = _read(VIS_SCRIPT)
    mujoco_text = _read(ROOT_MUJOCO_SCRIPT)
    adapter_text = _read(MUJOCO_ADAPTER)
    assert "96 observations -> 29 full-body joint actions" in train_text
    assert "ExtremeStandRecovery-v0" in train_text
    assert "ExtremeStandRecovery-Play-v0" in vis_text
    assert "G1_AMP_ARMHACK_STAND_ENABLE=False" in mujoco_text
    assert "G1_AMP_ARMHACK_WALK_ENABLE=False" in mujoco_text
    assert '"action_override": False' in adapter_text
    assert "compose_action" not in adapter_text


def test_final_deployment_launchers_pin_zero_command_and_amp_control_chain():
    export_text = _read(EXPORT_SCRIPT)
    mujoco_text = _read(ROOT_MUJOCO_SCRIPT)
    real_text = _read(REAL_DEPLOY_SCRIPT)
    compatibility_text = _read(MUJOCO_SCRIPT)
    pose_recovery_text = _read(POSE_RECOVERY_SCRIPT)
    adapter_text = _read(MUJOCO_ADAPTER)
    assert "2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt" in export_text
    assert "ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264" in export_text
    assert "--default-command 0 0 0" in export_text
    assert "2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt" in mujoco_text
    assert "CMD_INIT='[0.0,0.0,0.0]'" in mujoco_text
    assert "use/extreme_stand_recovery_pose_v2_model2999.onnx" in mujoco_text
    assert "action_override=false" in mujoco_text
    assert "deploy_real_g1_amp_onnx.sh" in real_text
    assert "2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery" in real_text
    assert "0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1" in real_text
    assert "2bf0f21c511463b19bd8a1ef1f77122cc43cee41560bfb398e3b06ba00164fd7" in real_text
    assert "COMMAND_MODE=fixed" in real_text
    assert "use/extreme_stand_recovery_pose_v2_model2999.onnx" in real_text
    assert "CMD_INIT='[0.0,0.0,0.0]'" in real_text
    assert "EXPECTED_ONNX_SHA256" in real_text
    assert "EXPECTED_CONFIG_SHA256" in real_text
    assert "sim2sim_g1_extreme_stand_recovery_mujoco.sh" in compatibility_text
    assert "SUITE_PROFILES=pose_recovery" in pose_recovery_text
    assert "initial_joint_mae_rad" in adapter_text
    assert "final_joint_max_abs_error_rad" in adapter_text
    assert "recovery_joint_max_threshold" in adapter_text
    assert "pose_recovered" in adapter_text
    assert '"body_event_counts"' in adapter_text
    assert '"events": list(self.wrench_events)' in adapter_text
    assert "[Extreme Stand wrench]" in adapter_text
    assert "SPACE" in adapter_text
    assert 'int(keycode) in (70, 102)' in adapter_text
    assert "process_interaction_requests" in adapter_text
    assert "PROFILE=${PROFILE:-interactive}" in mujoco_text
    assert "G1_AMP_EXTREME_STAND_INTERACTIVE_ENABLE" in mujoco_text
    assert "Startup mode" in real_text
    assert "DEFAULT_MOVE_S=0.0" in real_text
    assert "DEFAULT_HOLD_S=0.0" in real_text
    assert "TERMINAL_SPACE_HANDOFF=False" in real_text


def test_generic_mujoco_runner_keeps_recovery_separate_from_armhack():
    runner_text = _read(MUJOCO_RUNNER)
    assert "ExtremeStandRecoveryPerturbation" in runner_text
    assert "extreme_stand_recovery.key_callback" in runner_text
    assert "extreme_stand_recovery.process_interaction_requests" in runner_text
    assert "cannot be combined with an ArmHack action adapter" in runner_text
    assert "extreme_stand_recovery.update_external_wrench(data, sim_time)" in runner_text
    assert "draw_extreme_stand_external_wrench" in runner_text
    assert "data.xfrc_applied[body_id, :3]" in runner_text
    assert "onnxruntime" in runner_text
    assert "InferenceSession" in runner_text
    assert 'report["extreme_stand_recovery"]' in runner_text
