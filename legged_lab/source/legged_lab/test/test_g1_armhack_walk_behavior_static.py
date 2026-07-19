"""Static regression checks for Walk behavior refinement and scheduled MuJoCo tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path


LEGGED_LAB_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = LEGGED_LAB_DIR.parent
SOURCE_DIR = LEGGED_LAB_DIR / "source" / "legged_lab" / "legged_lab"
PERTURB_DIR = SOURCE_DIR / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb"
BEHAVIOR_CFG = PERTURB_DIR / "g1_walk_behavior_env_cfg.py"
REWARDS = SOURCE_DIR / "tasks" / "locomotion" / "amp" / "mdp" / "rewards.py"
HYBRID_COMMAND = SOURCE_DIR / "tasks" / "locomotion" / "amp" / "mdp" / "commands" / "hybrid_nav2_mode_velocity_command.py"
MODE_CONFIG = SOURCE_DIR / "data" / "MotionData" / "g1_29dof" / "amp" / "armhack_walk_behavior_50hz" / "task_sampling_config.json"
REGISTRY = PERTURB_DIR / "__init__.py"
AGENT_CFG = PERTURB_DIR / "agents" / "rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = LEGGED_LAB_DIR / "scripts" / "train_g1_armhack_walk_behavior.sh"
VAL_SCRIPT = LEGGED_LAB_DIR / "scripts" / "val_mujoco_g1_armhack_walk_behavior.sh"
TEST_SCRIPT = LEGGED_LAB_DIR / "scripts" / "test_mujoco_g1_armhack_walk_behavior.sh"
ANALYZER = LEGGED_LAB_DIR / "scripts" / "analyze_armhack_walk_behavior_mujoco.py"
SCENARIOS = LEGGED_LAB_DIR / "Reference Data" / "ArmHack" / "WalkPerturbFinetune" / "behavior_test_scenarios.json"
MUJOCO_ADAPTER = REPO_DIR / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "armhack_walk.py"
MUJOCO_RUNNER = REPO_DIR / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "deploy_mujoco_g1_amp.py"
STAND_CFG = PERTURB_DIR / "g1_stand_perturb_env_cfg.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_behavior_python_files_parse_and_task_is_walk_only():
    for path in (BEHAVIOR_CFG, REWARDS, HYBRID_COMMAND, ANALYZER, MUJOCO_ADAPTER, MUJOCO_RUNNER):
        ast.parse(_text(path), filename=str(path))
    behavior_text = _text(BEHAVIOR_CFG)
    assert "G1WalkRobustFinetuneEnvCfg" in behavior_text
    assert "G1Stand" not in behavior_text
    assert "g1_stand" not in behavior_text
    assert "g1_walk_behavior" not in _text(STAND_CFG)


def test_distribution_has_exact_zero_micro_pure_turn_and_diagonal_modes():
    config = json.loads(_text(MODE_CONFIG))
    modes = config["modes"]
    assert modes["stand"]["lin_vel_x"] == [0.0, 0.0]
    assert modes["stand"]["lin_vel_y"] == [0.0, 0.0]
    assert modes["stand"]["ang_vel_z"] == [0.0, 0.0]
    assert modes["micro_forward"]["lin_vel_x"][0] > 0.0
    assert modes["micro_forward"]["lin_vel_x"][0] < 0.1
    assert modes["turn_in_place_left"]["lin_vel_x"] == [0.0, 0.0]
    assert modes["turn_in_place_left"]["lin_vel_y"] == [0.0, 0.0]
    assert "diagonal_front_left" in modes and "diagonal_front_right" in modes
    assert set(config["mode_weights"]) == set(modes)
    assert abs(sum(config["mode_weights"].values()) - 1.0) < 1.0e-9


def test_reward_contract_has_no_nonzero_deadband_and_uses_oriented_sole_shape():
    reward_text = _text(REWARDS)
    behavior_text = _text(BEHAVIOR_CFG)
    command_text = _text(HYBRID_COMMAND)
    for name in (
        "strict_zero_command_body_motion_l2",
        "strict_zero_command_feet_motion_l2",
        "strict_zero_command_joint_vel_l2",
        "strict_zero_command_double_support",
        "nonzero_command_single_stance",
        "command_response_shortfall_l1",
        "rapid_footstep_l1",
        "oriented_footprint_proximity_l2",
    ):
        assert f"def {name}(" in reward_text
        assert name in behavior_text
    assert "torch.linalg.vector_norm(command, dim=1)" in reward_text
    assert "axis_separation" in reward_text
    assert "center_offset_x" in reward_text
    assert "cfg.rewards.feet_air_time = None" in behavior_text
    assert "hard_zero_stand = True" in behavior_text
    assert "if self.cfg.hard_zero_stand" in command_text


def test_task_runner_and_full_state_model10990_launcher_are_registered():
    registry = _text(REGISTRY)
    agent = _text(AGENT_CFG)
    train = _text(TRAIN_SCRIPT)
    assert 'LeggedLab-Isaac-AMP-G1-WalkBehaviorFinetune-v0' in registry
    assert 'LeggedLab-Isaac-AMP-G1-WalkBehaviorFinetune-Play-v0' in registry
    assert 'experiment_name = "g1_walk_behavior"' in agent
    assert 'checkpoint_output_dir = "ArmHack Checkpoints/WalkBehaviorFinetune"' in agent
    assert "checkpoint/walk/model_10990.pt" in train
    assert "EXPECTED_BASE_SHA256" in train
    assert 'LOAD_POLICY_ONLY=False' in train
    assert 'RESET_ITERATION=False' in train
    for phase in ("stop_micro_turn", "lateral_geometry", "robust"):
        assert f"    {phase})" in train


def test_mujoco_schedule_and_targeted_metrics_cover_all_reported_failures():
    scenarios = json.loads(_text(SCENARIOS))["scenarios"]
    required = {
        "zero_hold", "walk_to_zero", "micro_forward", "micro_lateral",
        "micro_diagonal", "turn_in_place_left", "turn_in_place_right",
        "lateral_left", "lateral_right", "diagonal_front_left",
        "diagonal_front_right", "forward_cadence",
    }
    assert required.issubset(scenarios)
    adapter = _text(MUJOCO_ADAPTER)
    runner = _text(MUJOCO_RUNNER)
    val = _text(VAL_SCRIPT)
    suite = _text(TEST_SCRIPT)
    assert "load_command_schedule" in adapter
    assert "current_schedule_segment" in adapter
    assert "armhack_walk_hard_zero_command" in runner
    assert "oriented_sole_signed_clearance" in runner
    assert "steady_step_frequency_hz" in runner
    assert "steady_min_signed_sole_clearance_m" in runner
    assert "G1_AMP_ARMHACK_WALK_HARD_ZERO_COMMAND=True" in val
    assert "ONNX/TorchScript actor mismatch" in val
    assert "SUITE must be smoke, core, or full" in suite
    assert "ENFORCE_THRESHOLDS" in suite
