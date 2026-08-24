from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_armhack_first_principles_single_env_cfg.py"
RUNNER = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py"
LAUNCHER = ROOT / "scripts/train_g1_armhack_first_principles_single.sh"
EVENTS = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/events.py"
MUJOCO_RUNNER = ROOT.parent / "unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py"
SWITCH_SCRIPT = ROOT.parent / "scripts/vis_mujoco_g1_armhack_stand_walk_switch.sh"


def test_exactly_two_single_actor_tasks_are_registered_by_design():
    text = CFG.read_text()
    assert "G1ArmHackStandFirstPrinciplesSingleEnvCfg" in text
    assert "G1ArmHackWalkFirstPrinciplesSingleEnvCfg" in text
    assert "G1ArmHackStandFirstPrinciplesStrictSingleEnvCfg" in text
    assert "G1ArmHackStandFirstPrinciplesOneStepSingleEnvCfg" in text
    assert "G1ArmHackWalkFirstPrinciplesStrictSingleEnvCfg" in text
    assert "G1ArmHackWalkFirstPrinciplesRobustSingleEnvCfg" in text
    assert "G1ArmHackWalkFirstPrinciplesResponseSingleEnvCfg" in text
    assert "G1ArmHackWalkDeadzoneYawSingleEnvCfg" in text
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
        "sequential_repeated_lift_event",
        "sequential_incomplete_step_penalty",
        "sequential_phase_time_excess_l2",
        "sequential_exact_step_budget_success",
        "sequential_active_contact_slide_l2",
        "post_completion_torso_xy_l2",
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


def test_mujoco_switch_uses_two_plain_actors_and_explicit_keys():
    runner_text = MUJOCO_RUNNER.read_text()
    script_text = SWITCH_SCRIPT.read_text()
    for token in (
        'policy_switch_state = {"mode": "paused"',
        '"[POLICY SWITCH] ENTER: PAUSED -> STAND"',
        '"[POLICY SWITCH] ENTER: STAND -> WALK; command ramps from zero."',
        '"[POLICY SWITCH] ENTER: WALK -> STAND; zero command and new torso SE(2) reference."',
        "armhack_stand.reset_switch_reference(data, sim_time)",
        "infer_walk_policy(obs)",
    ):
        assert token in runner_text
    for token in (
        "G1_AMP_POLICY_SWITCH_ENABLE=True",
        "ENTER   : after startup, toggle STAND <-> WALK",
        "SPACE/P : cycle shared arm poses",
        "92c51b2a2a4556ea993a7f9675cbe2ff06675c7681ca254df83c2ee27acc569e",
    ):
        assert token in script_text
