"""Static contracts for adaptive Stand and precise low-speed Walk training."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MDP_REWARDS = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"
MDP_OBSERVATIONS = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/observations.py"
REGISTRY = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py"
STAND_CFG = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_stand_adaptive_switch_env_cfg.py"
WALK_CFG = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_precision_switch_env_cfg.py"
WALK_SAMPLING = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/walk_precision_task_sampling.json"
STAND_TRAIN = ROOT / "legged_lab/scripts/train_g1_armhack_stand_adaptive_switch.sh"
WALK_TRAIN = ROOT / "legged_lab/scripts/train_g1_armhack_walk_precision_switch.sh"
MUJOCO_RUNNER = ROOT / "unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py"
CONTINUOUS_SCRIPT = ROOT / "legged_lab/scripts/test_g1_armhack_continuous_switch_mujoco.sh"
CONTINUOUS_SCENARIOS = ROOT / "legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/continuous_switch_scenarios.json"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_stand_selects_lower_contact_force_and_exact_se2_targets():
    source = text(MDP_REWARDS)
    block = source[source.index("def _sequential_foot_step_state"):source.index("def sequential_foot_step_progress")]
    assert "contact_sensor.data.net_forces_w" in block
    assert "contact_force[:, 0] <= contact_force[:, 1]" in block
    assert 'state["first_foot_index"]' in block
    assert 'state["second_foot_index"]' in block
    assert "reset_pelvis_pos" in block
    assert "lateral_target_offset_m" in block


def test_stand_observation_exposes_physical_active_foot_without_new_dimensions():
    source = text(MDP_OBSERVATIONS)
    block = source[source.index("def sequential_phase_augmented_last_action"):]
    assert 'active_index = state["active_index"]' in block
    assert "2.0 * active_index.float() - 1.0" in block
    assert "return actions" in block


def test_stand_preserves_spacing_torque_pose_and_se2_constraints():
    source = text(STAND_CFG)
    for token in (
        "sequential_foot_final_target_l2.weight = -80.0",
        "sequential_final_ankle_distance_exp.weight = 25.0",
        "torso_xy_position_l2.weight = -12.0",
        "torso_yaw_l2.weight = -8.0",
        "ankle_torques_l2.weight = -1.2e-3",
        "asymmetric_support_probability",
    ):
        assert token in source


def test_walk_precision_has_deadband_bounded_cube_spacing_and_clearance():
    rewards = text(MDP_REWARDS)
    cfg = text(WALK_CFG)
    assert "def precision_torso_velocity_tracking_exp" in rewards
    assert "min_command: float = 0.04" in rewards
    assert "max_command: float = 0.40" in rewards
    assert "command.reset_command_to_zero = True" in cfg
    assert "command.command_clip_min = (-0.40, -0.40, -0.40)" in cfg
    assert '"target_height": 0.065' in cfg
    assert "ankle_distance_30cm_kernel.weight = 80.0" in cfg
    assert "precision_torso_velocity_tracking" in cfg
    assert "G1WalkAnkleSpacingBaseEnvCfg" in cfg
    sampling = text(WALK_SAMPLING)
    assert '"stand": 0.10' in sampling
    assert '"lin_vel_x": [0.08, 0.40]' in sampling
    assert '"ang_vel_z": [-0.40, -0.08]' in sampling


def test_training_sources_are_identity_locked_and_tasks_registered():
    stand = text(STAND_TRAIN)
    walk = text(WALK_TRAIN)
    registry = text(REGISTRY)
    assert "9ab48719840c98f1332693a56f58ed069463c0670737e339b90411985484a729" in stand
    assert "62ee29b8c4fbbf8a4b96424d3cdffd698f89eeacab860dd6f3081edd6e1413d4" in walk
    assert "LeggedLab-Isaac-AMP-G1-StandAdaptiveSwitch-v0" in registry
    assert "LeggedLab-Isaac-AMP-G1-ArmHackWalkPrecisionSwitch-v0" in registry
    assert "ZERO_COMMAND_TEACHER_ONLY" in walk
    algorithm = text(ROOT / "rsl_rl/rsl_rl/algorithms/ppo_amp.py")
    assert "baseline_kl_zero_command_only" in algorithm
    mujoco_runner = text(MUJOCO_RUNNER)
    assert "G1_AMP_ADAPTIVE_STAND_PHASE_OBS" in mujoco_runner
    assert "foot_contact_forces_with_floor" in mujoco_runner
    assert 'obs[94] = 2.0 * float(active) - 1.0' in mujoco_runner


def test_continuous_switch_mujoco_contract_is_explicit_and_bounded():
    runner = text(MUJOCO_RUNNER)
    script = text(CONTINUOUS_SCRIPT)
    scenarios = text(CONTINUOUS_SCENARIOS)
    assert "G1_AMP_SECONDARY_POLICY_PATH" in runner
    assert "[POLICY SWITCH]" in runner
    assert "G1_AMP_CONTINUOUS_PUSH_FORCE_N" in runner
    assert "arms_down_to_front_stand" in scenarios
    assert "raise_arms_while_walking" in scenarios
    assert "walk_stop_then_move_arms" in scenarios
    assert "full_cycle" in scenarios
    assert "push40" in script and "push80" in script and "push120" in script
    assert "VISUAL_PUSH_FORCE_N" in script
    assert 'key == "M"' in runner
    assert "[STAND STEP WARNING]" in runner
    assert "post_complete_air_events" in runner
