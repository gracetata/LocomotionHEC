"""Static regression checks for the isolated P0/P1 ArmHack Walk task."""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_DIR / "source" / "legged_lab" / "legged_lab"
PERTURB_CFG_DIR = SOURCE_DIR / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb"
COMMAND_DIR = SOURCE_DIR / "tasks" / "locomotion" / "amp" / "mdp" / "commands"
ROBUST_CFG = PERTURB_CFG_DIR / "g1_walk_robust_env_cfg.py"
HYBRID_COMMAND = COMMAND_DIR / "hybrid_nav2_mode_velocity_command.py"
REGISTRY = PERTURB_CFG_DIR / "__init__.py"
AGENT_CFG = PERTURB_CFG_DIR / "agents" / "rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = PROJECT_DIR / "scripts" / "train_g1_armhack_walk.sh"
VIS_SCRIPT = PROJECT_DIR / "scripts" / "vis_g1_armhack_walk.sh"
TEST_SCRIPT = PROJECT_DIR / "scripts" / "test_g1_armhack_walk.sh"
AMP_RUNNER = PROJECT_DIR.parent / "rsl_rl" / "rsl_rl" / "runners" / "amp_runner.py"
STAND_CFG = PERTURB_CFG_DIR / "g1_stand_perturb_env_cfg.py"
STAND_RANDOM_CFG = PERTURB_CFG_DIR / "g1_stand_randomized_payload_env_cfg.py"
MODE_CONFIG = (
    SOURCE_DIR
    / "data"
    / "MotionData"
    / "g1_29dof"
    / "amp"
    / "command_balanced_directional_50hz"
    / "task_sampling_config.json"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_new_walk_python_files_parse_and_are_walk_only():
    for path in (ROBUST_CFG, HYBRID_COMMAND):
        ast.parse(_text(path), filename=str(path))

    robust_text = _text(ROBUST_CFG)
    assert "G1WalkPerturbFinetuneEnvCfg" in robust_text
    assert "G1Stand" not in robust_text
    assert "g1_stand" not in robust_text
    assert "G1WalkRobustFinetuneEnvCfg_PLAY" in robust_text
    assert "self.observations" not in robust_text
    assert "target_dof_names = G1_LOCOMOTION_JOINT_NAMES" in robust_text
    assert "strict_dof_names = True" in robust_text

    # Stand imports only the legacy common Walk helper, never the robust task.
    assert "g1_walk_robust" not in _text(STAND_CFG)
    assert "g1_walk_robust" not in _text(STAND_RANDOM_CFG)


def test_lower_body_amp_payload_and_hybrid_command_contracts():
    robust_text = _text(ROBUST_CFG)
    command_text = _text(HYBRID_COMMAND)

    assert "HybridNav2ModeVelocityCommandCfg" in robust_text
    assert 'mode_probability=0.0' in robust_text
    assert 'body_names=["left_wrist_yaw_link"]' in robust_text
    assert 'body_names=["right_wrist_yaw_link"]' in robust_text
    assert robust_text.count('"mass_distribution_params": (0.0, 1.0)') == 2
    assert '"distribution": "uniform"' in robust_text
    assert '"recompute_inertia": True' in robust_text

    assert "class HybridNav2ModeVelocityCommand(Nav2RecordedVelocityCommand)" in command_text
    assert "super()._resample_command(env_ids_tensor)" in command_text
    assert "super()._update_command()" in command_text
    assert "self.is_mode_env" in command_text
    assert "self.current_mode_ids" in command_text
    assert 'self.metrics["source_nav2_ratio"]' in command_text
    assert 'self.metrics["source_mode_ratio"]' in command_text
    assert "mode_probability must be in [0, 1]" in command_text
    assert "privileged mode label" in command_text


def test_mode_dataset_is_the_exact_eight_mode_distribution():
    with MODE_CONFIG.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    expected = {
        "stand",
        "forward_slow",
        "forward_normal",
        "backward",
        "lateral_left",
        "lateral_right",
        "turn_left",
        "turn_right",
    }
    assert set(config["modes"]) == expected
    assert set(config["mode_weights"]) == expected
    assert abs(sum(float(value) for value in config["mode_weights"].values()) - 1.0) < 1.0e-9


def test_task_and_runner_registration_enable_masked_amp():
    registry_text = _text(REGISTRY)
    agent_text = _text(AGENT_CFG)
    assert 'id="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-v0"' in registry_text
    assert 'id="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-Play-v0"' in registry_text
    assert "G1WalkPerturbAmpEnv" in registry_text
    assert "G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg" in agent_text
    assert 'experiment_name = "g1_walk_robust"' in agent_text
    assert "style_reward_scale = 1.0" in agent_text
    assert "task_style_lerp = 0.85" in agent_text


def test_training_curriculum_and_launchers_are_explicit():
    train_text = _text(TRAIN_SCRIPT)
    vis_text = _text(VIS_SCRIPT)
    test_text = _text(TEST_SCRIPT)
    amp_runner_text = _text(AMP_RUNNER)

    for phase in (
        "amp_warmup",
        "amp_target",
        "payload_half",
        "payload_full",
        "command",
        "domain_base",
        "domain_actuator",
        "domain_link",
        "robust",
    ):
        assert f"    {phase})" in train_text
    assert "PHASE_ITERATIONS=2000" in train_text
    assert "PAYLOAD_MAX_KG=1.0" in train_text
    assert "MODE_PROBABILITY=0.30" in train_text
    assert "RANDOMIZATION_STRENGTH=1" in train_text
    assert 'PHASE_HYDRA_ARGS+=("env.events.push_robot=null")' in train_text
    assert 'PHASE_HYDRA_ARGS+=("env.events.scale_actuator_gains=null")' in train_text
    assert 'PHASE_HYDRA_ARGS+=("env.events.scale_joint_parameters=null")' in train_text
    assert 'PHASE_HYDRA_ARGS+=("env.events.scale_link_mass=null")' in train_text
    assert "MODE=init is restricted to PHASE=amp_warmup" in train_text
    assert 'RESET_AMP_ON_LOAD=True' in train_text
    assert 'agent.reset_amp_on_load=${RESET_AMP_ON_LOAD}' in train_text
    assert 'agent.load_policy_only=${LOAD_POLICY_ONLY}' in train_text
    assert "BASELINE_KL_ENABLE=True" in train_text

    assert 'TASK="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-Play-v0"' in vis_text
    assert "COMMAND_SOURCE must be nav2, hybrid, or mode" in vis_text
    assert "forced_mode" in vis_text
    assert "mass_distribution_params=[${LEFT_PAYLOAD_KG},${LEFT_PAYLOAD_KG}]" in vis_text
    assert "mass_distribution_params=[${RIGHT_PAYLOAD_KG},${RIGHT_PAYLOAD_KG}]" in vis_text
    assert "--video_length" in vis_text

    assert "SUITE must be smoke, core, or full" in test_text
    assert "mode:forward_normal" in test_text
    assert "Reached max_steps=${MAX_STEPS}" in test_text
    assert "[METRIC\\] IsaacSim play task tracking:" in test_text
    assert "Test Reports" in test_text
    assert 'reset_amp_on_load = bool(self.cfg.get("reset_amp_on_load", False))' in amp_runner_text
    assert "fresh_amp_state = copy.deepcopy" in amp_runner_text
    assert "Reset AMP discriminator, normalizer, and optimizer" in amp_runner_text
