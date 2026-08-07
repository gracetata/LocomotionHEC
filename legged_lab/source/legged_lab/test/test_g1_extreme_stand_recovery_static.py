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
V5_ENV_CFG = TASK_DIR / "g1_extreme_stand_recovery_v5_env_cfg.py"
V6_ENV_CFG = TASK_DIR / "g1_extreme_stand_recovery_v6_env_cfg.py"
V5_DISTURBANCES = TASK_DIR / "disturbances.py"
TASK_INIT = TASK_DIR / "__init__.py"
RUNNER_CFG = TASK_DIR / "agents" / "rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "train_g1_extreme_stand_recovery.sh"
POSE_V2_TRAIN_SCRIPT = (
    PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "train_g1_extreme_stand_recovery_pose_v2.sh"
)
ANTI_JITTER_V3_TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "extreme_stand_recovery"
    / "train_g1_extreme_stand_recovery_anti_jitter_v3.sh"
)
SMOOTH_TORQUE_V4_TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "extreme_stand_recovery"
    / "train_g1_extreme_stand_recovery_smooth_torque_v4.sh"
)
SMOOTH_SETTLE_V5_TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "extreme_stand_recovery"
    / "train_g1_extreme_stand_recovery_smooth_settle_v5.sh"
)
TARGET_LOCK_V6_TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "extreme_stand_recovery"
    / "train_g1_extreme_stand_recovery_target_lock_v6.sh"
)
VIS_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "vis_g1_extreme_stand_recovery.sh"
MUJOCO_SCRIPT = PROJECT_ROOT / "scripts" / "extreme_stand_recovery" / "val_mujoco_g1_extreme_stand_recovery.sh"
ROOT_MUJOCO_SCRIPT = PROJECT_ROOT.parent / "scripts" / "sim2sim_g1_extreme_stand_recovery_mujoco.sh"
EXPORT_SCRIPT = PROJECT_ROOT.parent / "scripts" / "export_g1_extreme_stand_recovery.sh"
REAL_DEPLOY_SCRIPT = PROJECT_ROOT.parent / "scripts" / "deploy_real_g1_extreme_stand_recovery_onnx.sh"
SUMMARY_SCRIPT = PROJECT_ROOT.parent / "scripts" / "summarize_g1_extreme_stand_recovery_mujoco.py"
PUSH_DIAGNOSTIC_PLOT_SCRIPT = (
    PROJECT_ROOT.parent / "scripts" / "plot_g1_extreme_stand_push_diagnostics.py"
)
POSE_RECOVERY_SCRIPT = PROJECT_ROOT.parent / "scripts" / "test_g1_extreme_stand_random_pose_recovery_mujoco.sh"
ANTI_JITTER_MUJOCO_SCRIPT = (
    PROJECT_ROOT.parent / "scripts" / "test_g1_extreme_stand_anti_jitter_v3_mujoco.sh"
)
TARGET_LOCK_V6_MUJOCO_SCRIPT = (
    PROJECT_ROOT.parent / "scripts" / "test_g1_extreme_stand_target_lock_v6_mujoco.sh"
)
MUJOCO_ADAPTER = PROJECT_ROOT.parent / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "extreme_stand_recovery.py"
MUJOCO_RUNNER = PROJECT_ROOT.parent / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "deploy_mujoco_g1_amp.py"
AMP_EVENTS = PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "mdp" / "events.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_extreme_stand_recovery_files_exist():
    for path in (
        ENV_CFG,
        V5_ENV_CFG,
        V6_ENV_CFG,
        V5_DISTURBANCES,
        TASK_INIT,
        RUNNER_CFG,
        TRAIN_SCRIPT,
        POSE_V2_TRAIN_SCRIPT,
        ANTI_JITTER_V3_TRAIN_SCRIPT,
        SMOOTH_TORQUE_V4_TRAIN_SCRIPT,
        SMOOTH_SETTLE_V5_TRAIN_SCRIPT,
        TARGET_LOCK_V6_TRAIN_SCRIPT,
        VIS_SCRIPT,
        MUJOCO_SCRIPT,
        ROOT_MUJOCO_SCRIPT,
        EXPORT_SCRIPT,
        REAL_DEPLOY_SCRIPT,
        SUMMARY_SCRIPT,
        PUSH_DIAGNOSTIC_PLOT_SCRIPT,
        POSE_RECOVERY_SCRIPT,
        ANTI_JITTER_MUJOCO_SCRIPT,
        TARGET_LOCK_V6_MUJOCO_SCRIPT,
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


def test_pose_v3_rewards_use_jerk_and_narrow_default_pose_gaussians():
    cfg_text = _read(ENV_CFG)
    rewards_text = _read(TASK_DIR / "rewards.py")
    train_text = _read(ANTI_JITTER_V3_TRAIN_SCRIPT)
    pose_v2_train_text = _read(POSE_V2_TRAIN_SCRIPT)
    events_text = _read(AMP_EVENTS)
    assert "cache_default_cartesian_pose" in cfg_text
    assert "cache_default_feet_pose" in cfg_text
    assert "default_leg_joint_pose_exp" in cfg_text
    assert "default_key_body_pose_gaussian" in cfg_text
    assert "default_feet_distance_l2" in cfg_text
    assert "default_feet_distance_gaussian" in cfg_text
    assert "joint_jerk_l2" in cfg_text
    assert "weight=5.0" in cfg_text
    assert "weight=3.0" in cfg_text
    assert "weight=8.0" in cfg_text
    assert "weight=-8.0" in cfg_text
    assert "weight=-1.0e-8" in cfg_text
    assert "torch.square(current_distance - reference_distance)" in rewards_text
    assert "class joint_jerk_l2(ManagerTermBase)" in rewards_text
    assert "(current - self._previous_joint_acc) / env.step_dt" in rewards_text
    assert "torch.where(self._history_valid" in rewards_text
    assert "self._history_valid[env_ids] = False" in rewards_text
    assert "torch.exp(-0.5 * mean_square_distance / variance)" in rewards_text
    assert "torch.exp(-0.5 * square_error / variance)" in rewards_text
    assert "env_ids: torch.Tensor | None," in events_text
    assert "pose_v2_from_model4999_full_20260720/model_2999.pt" in train_text
    assert "ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264" in train_text
    assert "DEFAULT_CARTESIAN_POSE_VARIANCE" in train_text
    assert "DEFAULT_FEET_GAUSSIAN_VARIANCE" in train_text
    assert "env.rewards.default_key_body_pose_gaussian.weight=0.0" in pose_v2_train_text
    assert "env.rewards.default_feet_distance_gaussian.weight=0.0" in pose_v2_train_text
    assert "env.rewards.joint_jerk_l2.weight=0.0" in pose_v2_train_text


def test_pose_v4_adds_action_curvature_and_applied_torque_rate_penalties():
    cfg_text = _read(ENV_CFG)
    rewards_text = _read(TASK_DIR / "rewards.py")
    train_text = _read(SMOOTH_TORQUE_V4_TRAIN_SCRIPT)
    pose_v2_train_text = _read(POSE_V2_TRAIN_SCRIPT)

    assert "action_second_difference_l2" in cfg_text
    assert "joint_torque_rate_l2" in cfg_text
    assert "weight=0.0" in cfg_text
    assert "class action_second_difference_l2(ManagerTermBase)" in rewards_text
    assert "action_delta - self._previous_action_delta" in rewards_text
    assert "class joint_torque_rate_l2(ManagerTermBase)" in rewards_text
    assert "(current - self._previous_joint_torque) / env.step_dt" in rewards_text
    assert rewards_text.count("torch.where(self._history_valid") >= 3

    assert (
        "2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_"
        "resume1400_to2999_full_20260724/model_2999.pt"
    ) in train_text
    assert "e2c694d2d7710315f41f1c6c75849ffb95b53d0fb29e612aa211e1525a7cb1e4" in train_text
    assert "JOINT_TORQUE_PENALTY" in train_text
    assert "JOINT_TORQUE_RATE_PENALTY" in train_text
    assert "JOINT_ACCELERATION_PENALTY" in train_text
    assert "JOINT_JERK_PENALTY" in train_text
    assert "ACTION_RATE_PENALTY" in train_text
    assert "ACTION_SECOND_DIFFERENCE_PENALTY" in train_text
    assert "env.rewards.dof_torques_l2.weight=-${JOINT_TORQUE_PENALTY}" in train_text
    assert "env.rewards.joint_torque_rate_l2.weight=-${JOINT_TORQUE_RATE_PENALTY}" in train_text
    assert "env.rewards.action_second_difference_l2.weight=-${ACTION_SECOND_DIFFERENCE_PENALTY}" in train_text
    assert "TORSO_FORCE_MAX_N=${TORSO_FORCE_MAX_N:-45.0}" in train_text
    assert "LEG_NOISE_RAD=${LEG_NOISE_RAD:-0.30}" in train_text
    assert "env.rewards.action_second_difference_l2.weight=0.0" in pose_v2_train_text
    assert "env.rewards.joint_torque_rate_l2.weight=0.0" in pose_v2_train_text


def test_v5_uses_target_settling_topk_normalization_and_single_push_clock():
    cfg_text = _read(V5_ENV_CFG)
    rewards_text = _read(TASK_DIR / "rewards.py")
    disturbance_text = _read(V5_DISTURBANCES)
    train_text = _read(SMOOTH_SETTLE_V5_TRAIN_SCRIPT)
    task_text = _read(TASK_INIT)

    for reward_name in (
        "action_l2",
        "target_q_default_error_l2",
        "target_q_velocity_l2",
        "target_q_acceleration_l2",
        "normalized_joint_jerk_topk_l2",
        "normalized_joint_torque_topk_l2",
        "normalized_joint_torque_rate_topk_l2",
        "soft_peak_joint_torque_topk_l2",
        "joint_velocity_l2",
        "joint_acceleration_l2",
        "mechanical_power_l2",
        "near_default_settle_penalty",
        "post_disturbance_pose_recovery",
        "post_disturbance_stillness",
    ):
        assert reward_name in cfg_text
        assert reward_name in rewards_text or reward_name in {
            "action_l2",
            "joint_velocity_l2",
            "joint_acceleration_l2",
        }

    assert "torch.topk" in rewards_text
    assert "joint_effort_limits" in rewards_text
    assert "torch.relu(relative - soft_ratio)" in rewards_text
    assert "_near_default_gate" in rewards_text
    assert "self.events.random_torso_external_wrench = None" in cfg_text
    assert "self.events.random_pelvis_external_wrench = None" in cfg_text
    assert "self.events.random_arm_external_wrench = None" in cfg_text
    assert "self.events.random_leg_external_wrench = None" in cfg_text
    assert "self.events.push_robot = None" in cfg_text
    assert "force_magnitudes_n\": (10.0, 20.0, 36.0, 45.0)" in cfg_text
    assert "stage_step_thresholds\": (7200, 14400, 24000)" in cfg_text
    assert "active_duration_range_s\": (0.10, 0.30)" in cfg_text
    assert "quiet_duration_range_s\": (6.0, 10.0)" in cfg_text
    assert "direction_probabilities\": (0.15, 0.30, 0.20, 0.35)" in cfg_text
    assert "class single_body_force_curriculum(ManagerTermBase)" in disturbance_text
    assert "self._clear_wrench(env_ids)" in disturbance_text
    assert "SmoothSettleV5" in task_text
    assert "DISTURBANCE_MODE=single" in train_text
    assert "MAX_ITERATIONS=${MAX_ITERATIONS:-1500}" in train_text
    assert "SAVE_INTERVAL=${SAVE_INTERVAL:-100}" in train_text
    assert '"agent.save_interval=${SAVE_INTERVAL}"' in train_text
    assert "e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff" in train_text


def test_v6_fixes_dead_settle_signal_and_covers_large_push_impulse():
    cfg_text = _read(V6_ENV_CFG)
    rewards_text = _read(TASK_DIR / "rewards.py")
    disturbance_text = _read(V5_DISTURBANCES)
    train_text = _read(TARGET_LOCK_V6_TRAIN_SCRIPT)
    task_text = _read(TASK_INIT)

    assert "TargetLockV6" in task_text
    assert "near_default_target_lock_penalty" in cfg_text
    assert "post_disturbance_pose_recovery_rational" in cfg_text
    assert "post_disturbance_stillness_rational" in cfg_text
    assert "torch.reciprocal(1.0 + velocity_mse / velocity_scale)" in rewards_text
    assert "torch.reciprocal(1.0 + mean_square_error / pose_scale)" in rewards_text
    assert "force_magnitudes_n\": (45.0, 90.0, 150.0, 240.0)" in cfg_text
    assert "stage_step_thresholds\": (4800, 9600, 16800)" in cfg_text
    assert "quiet_duration_range_s\": (8.0, 12.0)" in cfg_text
    assert "body_force_scales" in cfg_text
    assert "applied_magnitude" in disturbance_text
    assert "disturbance_applied_force_n" in disturbance_text
    assert "MAX_ITERATIONS=${MAX_ITERATIONS:-1000}" in train_text
    assert "LEARNING_RATE=${LEARNING_RATE:-5.0e-6}" in train_text
    assert "13538475518be2a323dfedff230949b3c6b8057c8f4f9af000adbbfd90c7ee7c" in train_text


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
    runner_text = _read(MUJOCO_RUNNER)
    assert "2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt" in export_text
    assert "e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff" in export_text
    assert "--default-command 0 0 0" in export_text
    assert "2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt" in mujoco_text
    assert "CMD_INIT='[0.0,0.0,0.0]'" in mujoco_text
    assert 'POLICY_PATH=${POLICY_PATH:-"${EXPORT_DIR}/policy.onnx"}' in mujoco_text
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
    assert 'int(keycode) in (82, 114)' in adapter_text
    assert "pending_default_pose_reset" in adapter_text
    assert "manual_reset_default_standing" in adapter_text
    assert 'int(keycode) in (75, 107)' in adapter_text
    assert 'int(keycode) in (70, 102)' in adapter_text
    assert 'int(keycode) in (67, 99)' in adapter_text
    assert 'self.config["follow_camera_enable"] = follow_enabled' in adapter_text
    assert "foot_spacing_recovery" in adapter_text
    assert "feet_distance_recovery" in mujoco_text
    assert "large_push" in mujoco_text
    assert "G1_AMP_EXTREME_STAND_LARGE_PUSH_ENABLE" in mujoco_text
    assert "LARGE_PUSH_FORCE_N" in mujoco_text
    assert "SPACE cycles +X/-X/+Y/-Y torso pushes" in mujoco_text
    assert "R=IMMEDIATE DEFAULT STANDING RESET" in mujoco_text
    assert "C=FREE/FOLLOW camera" in mujoco_text
    assert "INTERACTIVE_DATA_LOG=${INTERACTIVE_DATA_LOG:-${default_interactive}}" in mujoco_text
    assert "interactive_diagnostics_all.csv" in mujoco_text
    assert "INTERACTIVE_LOG_SESSION=${INTERACTIVE_LOG_SESSION:-$(date +%Y%m%d_%H%M%S)}" in mujoco_text
    assert "interactive_logs/${INTERACTIVE_LOG_SESSION}" in mujoco_text
    assert "interactive_events.csv" in mujoco_text
    assert "space_trials" in mujoco_text
    assert "G1_AMP_EXTREME_STAND_INTERACTIVE_LOG_ENABLE" in mujoco_text
    assert "RENDER_FPS=${RENDER_FPS:-60}" in mujoco_text
    assert 'G1_AMP_RENDER_FPS="${RENDER_FPS}"' in mujoco_text
    assert "FOLLOW_CAMERA=${FOLLOW_CAMERA:-False}" in mujoco_text
    assert 'G1_AMP_FOLLOW_CAMERA_ENABLE="${FOLLOW_CAMERA}"' in mujoco_text
    assert 'FOLLOW_CAMERA_ENABLE="${FOLLOW_CAMERA}"' in mujoco_text
    assert "TARGET_LIMITER_ENABLE=${TARGET_LIMITER_ENABLE:-False}" in mujoco_text
    assert "G1_AMP_EXTREME_STAND_TARGET_LIMITER_ENABLE" in mujoco_text
    assert "TARGET_LEG_VELOCITY_LIMIT_RAD_S" in mujoco_text
    assert "TARGET_ARM_ACCELERATION_LIMIT_RAD_S2" in mujoco_text
    assert "render_period" in runner_text
    assert "RTF=" in runner_text
    assert "force_once=True" in runner_text
    assert "post_push_diagnostics" in adapter_text
    assert "pd_torque_saturation_after_push" in adapter_text
    assert "policy_action_driven_high_frequency_oscillation" in adapter_text
    assert "G1_AMP_EXTREME_STAND_FOOT_SPACING_START_RANDOM" in mujoco_text
    assert "process_interaction_requests" in adapter_text
    assert "PROFILE=${PROFILE:-interactive}" in mujoco_text
    assert "G1_AMP_EXTREME_STAND_INTERACTIVE_ENABLE" in mujoco_text
    assert "Startup mode" in real_text
    assert "DEFAULT_MOVE_S=0.0" in real_text
    assert "DEFAULT_HOLD_S=0.0" in real_text
    assert "TERMINAL_SPACE_HANDOFF=False" in real_text


def test_generic_mujoco_runner_keeps_recovery_separate_from_armhack():
    runner_text = _read(MUJOCO_RUNNER)
    adapter_text = _read(MUJOCO_ADAPTER)
    assert "ExtremeStandRecoveryPerturbation" in runner_text
    assert "extreme_stand_recovery.key_callback" in runner_text
    assert "extreme_stand_recovery.process_interaction_requests" in runner_text
    assert "cannot be combined with an ArmHack action adapter" in runner_text
    assert "extreme_stand_recovery.update_external_wrench(data, sim_time)" in runner_text
    assert "if force_once:" in runner_text
    assert "preserving mouse orbit and zoom" in runner_text
    assert "floor_geom_ids," in runner_text
    assert "target_command," in runner_text
    assert "raw_target_policy" in runner_text
    assert "limit_target_position" in runner_text
    assert "_ground_reaction_wrenches_world" in adapter_text
    assert "qacc_mujoco_rad_s2" in adapter_text
    assert "raw_target_qpos_rad" in adapter_text
    assert "target_limiter_velocity_rad_s" in adapter_text
    assert "target_limiter_acceleration_rad_s2" in adapter_text
    assert "pd_torque_command_nm" in adapter_text
    assert "joint_anchor_world_m" in adapter_text
    assert "feet/planar_distance_m" in adapter_text
    assert "feet/distance_3d_m" in adapter_text
    assert "feet/planar_distance_error_m" in adapter_text
    assert "data.xanchor" in adapter_text
    assert "interactive_diagnostics_all" not in adapter_text
    assert "_begin_space_trial" in adapter_text
    assert "space_trial_csv_files" in adapter_text
    assert "policy_actuator_torque_limits" in runner_text
    assert "data.actuator_force[policy_actuator_ids]" in runner_text
    assert "draw_extreme_stand_external_wrench" in runner_text
    assert "data.xfrc_applied[body_id, :3]" in runner_text
    assert "onnxruntime" in runner_text
    assert "InferenceSession" in runner_text
    assert 'report["extreme_stand_recovery"]' in runner_text
