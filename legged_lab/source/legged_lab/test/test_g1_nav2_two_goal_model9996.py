import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LEGGED_LAB_ROOT = REPO_ROOT / "legged_lab"
PACKAGE_ROOT = LEGGED_LAB_ROOT / "source" / "legged_lab" / "legged_lab"
G1_ROOT = PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1"
ENV_CFG_FILE = G1_ROOT / "g1_amp_env_cfg.py"
REGISTRY_FILE = G1_ROOT / "__init__.py"
AGENT_CFG_FILE = G1_ROOT / "agents" / "rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = LEGGED_LAB_ROOT / "scripts" / "train_g1_amp_nav2_two_goal_model9996.sh"
FULL_ACTOR_TRAIN_SCRIPT = (
    LEGGED_LAB_ROOT / "scripts" / "train_g1_amp_nav2_two_goal_model9996_full_actor.sh"
)
FULL_ACTOR_LATERAL_TRAIN_SCRIPT = (
    LEGGED_LAB_ROOT / "scripts" / "train_g1_amp_nav2_model9996_full_actor_lateral.sh"
)
CALIBRATE_SCRIPT = LEGGED_LAB_ROOT / "scripts" / "calibrate_g1_amp_nav2_two_goal_residual.py"
ACCEPTANCE_SCRIPT = (
    LEGGED_LAB_ROOT / "scripts" / "test_g1_amp_nav2_two_goal_model9996_mujoco.sh"
)
MERGE_SCRIPT = LEGGED_LAB_ROOT / "scripts" / "merge_g1_amp_nav2_two_goal_model9996.py"
GATED_MERGE_SCRIPT = (
    LEGGED_LAB_ROOT / "scripts" / "merge_g1_amp_nav2_model9996_gated_experts.py"
)
MODE_ROOT = (
    PACKAGE_ROOT
    / "data"
    / "MotionData"
    / "g1_29dof"
    / "amp"
    / "nav2_behavior_50hz"
)
MODEL9996 = REPO_ROOT / "checkpoint" / "nav2_behavior_model9996_source" / "model_9996.pt"
MODEL9996_SIZE = 16_202_421
MODEL9996_SHA256 = "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_model9996_is_the_exact_and_only_protected_origin():
    assert MODEL9996.stat().st_size == MODEL9996_SIZE
    assert _sha256(MODEL9996) == MODEL9996_SHA256
    script = TRAIN_SCRIPT.read_text()
    assert "model_9996.pt" in script
    assert str(MODEL9996_SIZE) in script
    assert MODEL9996_SHA256 in script
    assert "model_10990" not in script
    assert "model_12995" not in script
    assert '"${STAGE} source must be the protected model_9996"' in script


def test_commands_are_strict_and_balanced_with_a_retention_anchor():
    stage1 = json.loads((MODE_ROOT / "task_sampling_two_goal_config.json").read_text())
    stage2 = json.loads((MODE_ROOT / "task_sampling_two_goal_stage2_config.json").read_text())
    expected_weights = {
        "lateral_left": 0.25,
        "lateral_right": 0.25,
        "turn_in_place_left": 0.25,
        "turn_in_place_right": 0.25,
    }
    for config in (stage1, stage2):
        assert config["mode_weights"] == expected_weights
        for name, mode in config["modes"].items():
            assert mode["lin_vel_x"] == [0.0, 0.0]
            if name.startswith("lateral"):
                assert mode["ang_vel_z"] == [0.0, 0.0]
            else:
                assert mode["lin_vel_y"] == [0.0, 0.0]

    env_text = ENV_CFG_FILE.read_text()
    block = env_text[env_text.index("class G1AmpNav2TwoGoalFinetuneEnvCfg") :]
    assert "mode_probability = 0.80" in block
    assert "recorded Nav2 remainder is a retention anchor" in block

    lateral = json.loads(
        (MODE_ROOT / "task_sampling_two_goal_lateral_only_config.json").read_text()
    )
    yaw = json.loads(
        (MODE_ROOT / "task_sampling_two_goal_yaw_only_config.json").read_text()
    )
    assert set(lateral["modes"]) == {"lateral_left", "lateral_right"}
    assert set(yaw["modes"]) == {"turn_in_place_left", "turn_in_place_right"}
    for mode in lateral["modes"].values():
        assert mode["lin_vel_x"] == [0.0, 0.0]
        assert mode["ang_vel_z"] == [0.0, 0.0]
    for mode in yaw["modes"].values():
        assert mode["lin_vel_x"] == [0.0, 0.0]
        assert mode["lin_vel_y"] == [0.0, 0.0]


def test_tasks_are_manager_amp_and_model9996_specific():
    registry = REGISTRY_FILE.read_text()
    for task, env_cfg, agent_cfg in (
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996Bootstrap-v0",
            "G1AmpNav2TwoGoalModel9996BootstrapEnvCfg",
            "G1Nav2TwoGoalModel9996BootstrapRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996Corrective-v0",
            "G1AmpNav2TwoGoalModel9996CorrectiveEnvCfg",
            "G1Nav2TwoGoalModel9996CorrectiveRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996BarrierCorrective-v0",
            "G1AmpNav2TwoGoalModel9996BarrierCorrectiveEnvCfg",
            "G1Nav2TwoGoalModel9996BarrierCorrectiveRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996LateralSpecialist-v0",
            "G1AmpNav2TwoGoalModel9996LateralSpecialistEnvCfg",
            "G1Nav2TwoGoalModel9996LateralSpecialistRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996YawSpecialist-v0",
            "G1AmpNav2TwoGoalModel9996YawSpecialistEnvCfg",
            "G1Nav2TwoGoalModel9996YawSpecialistRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActor-v0",
            "G1AmpNav2TwoGoalModel9996FullActorEnvCfg",
            "G1Nav2TwoGoalModel9996FullActorRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActorLateral-v0",
            "G1AmpNav2TwoGoalModel9996FullActorLateralEnvCfg",
            "G1Nav2TwoGoalModel9996FullActorLateralRslRlOnPolicyRunnerAmpCfg",
        ),
        (
            "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996FullActorLateralFinal-v0",
            "G1AmpNav2TwoGoalModel9996FullActorLateralFinalEnvCfg",
            "G1Nav2TwoGoalModel9996FullActorLateralFinalRslRlOnPolicyRunnerAmpCfg",
        ),
    ):
        assert f'id="{task}"' in registry
        task_block = registry[registry.index(f'id="{task}"') :]
        task_block = task_block[: task_block.index("\n)\n")]
        assert 'entry_point="legged_lab.envs:ManagerBasedAmpEnv"' in task_block
        assert env_cfg in task_block
        assert agent_cfg in task_block


def test_deployment_has_no_carrier_and_retention_is_structural():
    agent = AGENT_CFG_FILE.read_text()
    start = agent.index("class G1Nav2TwoGoalModel9996BootstrapRslRlOnPolicyRunnerAmpCfg")
    block = agent[start:]
    assert 'experiment_name = "g1_amp_nav2_two_goal_model9996"' in block
    assert 'checkpoint_output_dir = "Nav2TwoGoalModel9996"' in block
    assert "load_actor_amp_only = True" in block
    assert "freeze_base_actor = True" in agent
    assert "freeze_pure_yaw_residual = False" in block
    assert "self.policy.fixed_command_bridge_fraction = 0.0" in block
    assert "self.algorithm.baseline_kl_cfg.specialization_scale = 0.0" in block
    assert "self.algorithm.command_bridge_cfg.enabled = True" in block

    corrective = block[block.index("class G1Nav2TwoGoalModel9996Corrective") :]
    assert "load_policy_only = True" in corrective
    assert "self.algorithm.command_bridge_cfg.enabled = False" in corrective
    assert "self.algorithm.command_bridge_cfg.scale = 0.0" in corrective
    assert "self.algorithm.command_bridge_cfg.residual_learning_rate = 0.0" in corrective


def test_real_motion_rewards_and_large_oriented_sole_barrier_are_active():
    env_text = ENV_CFG_FILE.read_text()
    start = env_text.index("class G1AmpNav2TwoGoalModel9996BootstrapEnvCfg")
    block = env_text[start: env_text.index("class G1AmpNav2TwoGoalCarrierFinetuneEnvCfg")]
    for reward in (
        "lateral_command_progress",
        "pure_yaw_command_progress",
        "dense_root_pose_command_progress",
        "two_goal_response_shortfall",
        "swept_oriented_footprint_soft_margin_l2",
        "swept_oriented_footprint_hard_barrier",
    ):
        assert reward in block
    assert block.count("swept_oriented_footprint_hard_barrier.weight = -12.0") >= 2
    assert block.count('["overlap_scale"] = 4.0') == 2
    assert '"hard_clearance": 0.025' in env_text
    assert '"soft_clearance": 0.040' in env_text

    barrier_start = env_text.index("class G1AmpNav2TwoGoalModel9996BarrierCorrectiveEnvCfg")
    barrier = env_text[barrier_start: env_text.index("class G1AmpNav2TwoGoalCarrierFinetuneEnvCfg")]
    assert "two_goal_response_shortfall.weight = -12.0" in barrier
    assert "lateral_command_leak_l2.weight = -5.0" in barrier
    assert "pure_yaw_planar_drift_l2.weight = -3.0" in barrier
    assert '"soft_clearance": 0.080' in barrier
    assert '"hard_clearance": 0.040' in barrier
    assert "swept_oriented_footprint_soft_margin_l2.weight = -4.0" in barrier
    assert "swept_oriented_footprint_hard_barrier.weight = -50.0" in barrier

    lateral_start = env_text.index("class G1AmpNav2TwoGoalModel9996LateralSpecialistEnvCfg")
    lateral = env_text[lateral_start: env_text.index("class G1AmpNav2TwoGoalModel9996YawSpecialistEnvCfg")]
    assert "two_goal_signed_root_response" in lateral
    assert "weight=20.0" in lateral
    assert "two_goal_response_shortfall.weight = -150.0" in lateral
    assert '"target_fraction": 0.80' in lateral
    assert '"max_penalty": 100.0' in lateral
    assert '"hard_clearance": 0.045' in lateral
    assert '"soft_max_penalty": 100.0' in lateral
    assert "lateral_foot_ordering_l2" in lateral
    assert '"foot_half_width": 0.035' in lateral
    assert "weight=-20.0" in lateral
    assert '"min_clearance": 0.150' in lateral
    assert '"shortfall_scale": 0.050' in lateral
    assert "swept_oriented_footprint_soft_margin_l2.weight = -8.0" in lateral
    assert "swept_oriented_footprint_hard_barrier.weight = -1.0" in lateral
    polish_start = env_text.index("class G1AmpNav2TwoGoalModel9996LateralSafetyPolishEnvCfg")
    polish = env_text[polish_start: env_text.index("class G1AmpNav2TwoGoalModel9996YawSpecialistEnvCfg")]
    assert "lateral_foot_ordering_l2.weight = -8.0" in polish
    assert "swept_oriented_footprint_hard_barrier.weight = -100.0" in polish

    leakage_start = env_text.index(
        "class G1AmpNav2TwoGoalModel9996LateralLeakageCorrectiveEnvCfg"
    )
    leakage = env_text[
        leakage_start: env_text.index("class G1AmpNav2TwoGoalModel9996YawSpecialistEnvCfg")
    ]
    assert "two_goal_signed_root_response.weight = 100.0" in leakage
    assert "two_goal_response_shortfall.weight = -500.0" in leakage
    assert "lateral_command_leak_l2.weight = -5.0" in leakage
    assert "lateral_foot_ordering_l2.weight = -1.0" in leakage
    assert "swept_oriented_footprint_soft_margin_l2.weight = -0.5" in leakage
    assert "swept_oriented_footprint_hard_barrier.weight = -100.0" in leakage

    yaw_start = env_text.index("class G1AmpNav2TwoGoalModel9996YawSpecialistEnvCfg")
    yaw = env_text[yaw_start: env_text.index("class G1AmpNav2TwoGoalCarrierFinetuneEnvCfg")]
    assert "two_goal_signed_root_response" in yaw
    assert "pure_yaw_planar_drift_l2.weight = -0.75" in yaw
    assert '"max_penalty": 100.0' in yaw
    assert "pure_yaw_root_rate_error_l2" in yaw
    assert '"error_scale": 0.10' in yaw


def test_training_script_separates_bootstrap_and_corrective_contracts():
    script = TRAIN_SCRIPT.read_text()
    assert "load_actor_amp_only=\"${LOAD_ACTOR_AMP_ONLY}\"" in script
    assert "load_policy_only=\"${LOAD_POLICY_ONLY}\"" in script
    assert "barrier_corrective" in script
    assert "lateral_direct" in script
    assert "lateral_direct must start with zero residuals from protected model_9996" in script
    assert "yaw_proxy_bootstrap" in script
    assert "yaw_proxy_bootstrap must start with zero residuals from protected model_9996" in script
    assert "pure_yaw_teacher_forward_command=0.0" in script
    assert "pure_yaw_positive_teacher_yaw_min=1.2" in script
    assert "pure_yaw_negative_teacher_yaw_min=1.2" in script
    assert "teacher_delta_fraction=1.0" in script
    assert "lateral_proxy_bootstrap" in script
    assert "lateral_proxy_bootstrap must start from protected model_9996" in script
    assert "lateral_teacher_forward_command=0.30" in script
    assert "lateral_teacher_min_abs_command=0.25" in script
    assert "lateral_teacher_opposite_yaw_abs=0.80" in script
    assert "lateral_cancel" in script
    assert "lateral_teacher_forward_command=-0.30" in script
    assert "teacher_delta_fraction=0.35" in script
    assert "lateral_specialist" in script
    assert "lateral_safety_polish" in script
    assert "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996LateralSafetyPolish-v0" in script
    assert "lateral_leakage_corrective" in script
    assert "LeggedLab-Isaac-AMP-G1-Nav2TwoGoalModel9996LateralLeakageCorrective-v0" in script
    assert "yaw_specialist" in script
    assert 'agent.freeze_lateral_residual="${FREEZE_LATERAL_RESIDUAL}"' in script
    assert 'agent.freeze_pure_yaw_residual="${FREEZE_PURE_YAW_RESIDUAL}"' in script
    assert "training log contains a fatal Python/CUDA/numerical error" in script
    assert "training produced no dedicated checkpoint" in script
    assert "agent.policy.fixed_command_bridge_fraction=0.0" in script
    assert "BASELINE_KL_CHECKPOINT=\"${PROTECTED_MODEL9996}\"" in script
    assert "RSI_ENABLE=False" in script
    assert "RANDOMIZATION_STRENGTH=0" in script
    assert "Deployed carrier  : disabled" in script
    assert 'OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2TwoGoalModel9996"' in script


def test_full_actor_escape_stage_is_exact_model9996_and_safety_constrained():
    script = FULL_ACTOR_TRAIN_SCRIPT.read_text()
    assert MODEL9996_SHA256 in script
    assert "model_10990" not in script
    assert "model_12995" not in script
    assert "agent.load_actor_amp_only=True" in script
    assert "agent.freeze_base_actor=False" in script
    assert "agent.freeze_actor_hidden_layers=1" in script
    assert "agent.actor_warmup_iterations=8" in script
    assert "BASELINE_KL_CHECKPOINT=\"${MODEL9996}\"" in script

    agent = AGENT_CFG_FILE.read_text()
    start = agent.index("class G1Nav2TwoGoalModel9996FullActorRslRlOnPolicyRunnerAmpCfg")
    block = agent[start:]
    assert "freeze_actor_hidden_layers = 1" in block
    assert "freeze_base_actor = False" in block
    assert "actor_warmup_iterations = 8" in block
    assert "self.algorithm.learning_rate = 1.5e-5" in block
    assert "self.algorithm.baseline_kl_cfg.specialization_scale = 0.005" in block
    assert "self.algorithm.amp_cfg.freeze_discriminator = True" in block

    env_text = ENV_CFG_FILE.read_text()
    start = env_text.index("class G1AmpNav2TwoGoalModel9996FullActorEnvCfg")
    full_actor = env_text[start: env_text.index("class G1AmpNav2TwoGoalCarrierFinetuneEnvCfg")]
    assert "lateral_foot_ordering_l2" in full_actor
    assert '"min_clearance": 0.040' in full_actor
    assert "swept_oriented_footprint_hard_barrier.weight = -100.0" in full_actor
    assert "two_goal_signed_root_response" in full_actor

    lateral_script = FULL_ACTOR_LATERAL_TRAIN_SCRIPT.read_text()
    assert MODEL9996_SHA256 in lateral_script
    assert "agent.load_policy_only=True" in lateral_script
    assert "agent.algorithm.baseline_kl_cfg.enabled=False" in lateral_script
    lateral_start = env_text.index("class G1AmpNav2TwoGoalModel9996FullActorLateralEnvCfg")
    lateral = env_text[
        lateral_start: env_text.index("class G1AmpNav2TwoGoalCarrierFinetuneEnvCfg")
    ]
    assert "two_goal_signed_root_response.weight = 100.0" in lateral
    assert "two_goal_response_shortfall.weight = -300.0" in lateral
    assert "swept_oriented_footprint_hard_barrier.weight = -150.0" in lateral
    final_start = env_text.index("class G1AmpNav2TwoGoalModel9996FullActorLateralFinalEnvCfg")
    final = env_text[final_start: env_text.index("class G1AmpNav2TwoGoalCarrierFinetuneEnvCfg")]
    assert "two_goal_signed_root_response.weight = 80.0" in final
    assert "two_goal_response_shortfall.weight = -400.0" in final
    assert "lateral_command_leak_l2.weight = -20.0" in final
    assert "swept_oriented_footprint_hard_barrier.weight = -150.0" in final


def test_residual_calibration_preserves_base_and_scales_only_final_layer():
    script = CALIBRATE_SCRIPT.read_text()
    assert '"lateral": "lateral_command_residual"' in script
    assert '"pure_yaw": "pure_yaw_command_residual"' in script
    assert 'scaled_keys = (f"{prefix}.2.weight", f"{prefix}.2.bias")' in script
    assert 'key.startswith("actor.")' in script
    assert "torch.equal(state[key], value)" in script
    assert "fixed_command_bridge_fraction" in script
    assert "Refusing to overwrite" in script
    assert MODEL9996_SHA256 in script


def test_gated_merge_preserves_model9996_and_uses_strict_lateral_calibration():
    script = GATED_MERGE_SCRIPT.read_text()
    assert MODEL9996_SHA256 in script
    assert '"lateral_expert_actor."' in script
    assert 'key.startswith("actor.")' in script
    assert 'key.startswith("lateral_command_residual.")' in script
    assert 'state["fixed_command_bridge_fraction"] = torch.tensor(0.0)' in script
    assert 'default=-0.14' in script
    assert 'default=0.10' in script
    assert "Refusing to overwrite" in script
    exporter = (LEGGED_LAB_ROOT / "scripts" / "rsl_rl" / "export_amp_actor_to_onnx.py").read_text()
    assert 'torch.tensor(-0.14)' in exporter


def test_mujoco_acceptance_is_strict_and_bidirectional():
    script = ACCEPTANCE_SCRIPT.read_text()
    for scenario in (
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
        "stand",
        "forward",
        "baseline_forward",
    ):
        assert scenario in script
    assert 'float(state["fixed_command_bridge_fraction"]) != 0.0' in script
    assert '"lateral_expert_actor.0.weight" in state' in script
    assert 'state["lateral_expert_forward_command"]' in script
    assert 'state["lateral_expert_same_yaw_abs"]' in script
    assert 'signed lateral speed {signed_vy:.4f} < 0.18 m/s' in script
    assert 'signed yaw rate {signed_yaw:.4f} < 0.25 rad/s' in script
    assert 'planar drift {planar_drift:.4f} > 0.035 m/s' in script
    assert 'health["sole_clearance_violation_fraction"] != 0.0' in script
    assert 'health["min_signed_sole_clearance_m"] < 0.025' in script
    assert "forward retention degraded by more than 15%" in script


def test_branch_composition_is_exact_and_model9996_protected():
    script = MERGE_SCRIPT.read_text()
    assert MODEL9996_SHA256 in script
    assert "refusing to overwrite existing output" in script
    assert 'require_equal_state("actor.", lateral_state, yaw_state)' in script
    assert 'require_equal_state("actor.", lateral_state, base_state)' in script
    assert 'float(bridge.item()) != 0.0' in script
    assert 'key.startswith("lateral_command_residual.")' in script
    assert 'key.startswith("pure_yaw_command_residual.")' in script
    assert "two_goal_model9996_provenance" in script
