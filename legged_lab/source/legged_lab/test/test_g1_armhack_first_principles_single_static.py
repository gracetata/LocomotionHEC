from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_armhack_first_principles_single_env_cfg.py"
RUNNER = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py"
LAUNCHER = ROOT / "scripts/train_g1_armhack_first_principles_single.sh"
EVENTS = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/events.py"


def test_exactly_two_single_actor_tasks_are_registered_by_design():
    text = CFG.read_text()
    assert "G1ArmHackStandFirstPrinciplesSingleEnvCfg" in text
    assert "G1ArmHackWalkFirstPrinciplesSingleEnvCfg" in text
    assert "G1ArmHackStandFirstPrinciplesStrictSingleEnvCfg" in text
    forbidden = ("ActorCriticCommandResidual", "TwoGoal", "Gated", "ExpertEnvCfg")
    assert not any(token in text for token in forbidden)


def test_runner_forces_plain_actor_critic_and_formal_runs_are_2000_plus():
    runner_text = RUNNER.read_text()
    launcher_text = LAUNCHER.read_text()
    assert "G1ArmHackStandFirstPrinciplesSingleRunnerCfg" in runner_text
    assert "G1ArmHackWalkFirstPrinciplesSingleRunnerCfg" in runner_text
    assert "agent.policy.class_name=ActorCritic" in launcher_text
    assert '"${MAX_ITERATIONS}" -lt 2000' in launcher_text
    assert "formal continuation runs must be at least 2000 iterations" in launcher_text
    assert "formal run stopped before iteration" in launcher_text
    assert "formal run did not save model_" in launcher_text


def test_objectives_and_bidirectional_producer_reset_are_explicit():
    cfg_text = CFG.read_text()
    events_text = EVENTS.read_text()
    for token in (
        "post_completion_airborne",
        "post_completion_contact_imbalance_l2",
        "post_completion_ankle_torque_l2",
        "sequential_active_foot_air_time_excess_l2",
        "sequential_active_foot_descent_exp",
        "ankle_distance_30cm",
        "useful_low_speed_tracking_l2",
        "pure_yaw_planar_drift_l2",
        "pure_yaw_torso_pitch_l2",
        "random_end_effector_wrench",
        "handoff_state_reset",
        "self.curriculum.stance_recovery = None",
        "self.rewards.feet_planar_separation_l2.weight = 0.0",
    ):
        assert token in cfg_text
    for field in ("root_state", "joint_pos", "joint_vel", "action"):
        assert field in events_text
    runner_text = RUNNER.read_text()
    assert runner_text.count("baseline_kl_cfg.hard_limit = 0.0") >= 2
