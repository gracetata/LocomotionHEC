from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent / "legged_lab"

ENV_FILE = PACKAGE_ROOT / "envs" / "g1_perturb_env.py"
ENV_INIT_FILE = PACKAGE_ROOT / "envs" / "__init__.py"
TASK_INIT_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "__init__.py"
)
STAND_CFG_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "g1_stand_perturb_env_cfg.py"
)
WALK_CFG_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "g1_walk_perturb_env_cfg.py"
)
AGENT_CFG_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "agents" / "rsl_rl_ppo_cfg.py"
)
AMP_RUNNER_FILE = Path(__file__).resolve().parents[4] / "rsl_rl" / "rsl_rl" / "runners" / "amp_runner.py"
CSV_PLAYBACK_FILE = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "visualize_g1_csv_full_body_motion.py"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_g1_perturb_files_exist():
    for path in [
        ENV_FILE,
        ENV_INIT_FILE,
        TASK_INIT_FILE,
        STAND_CFG_FILE,
        WALK_CFG_FILE,
        AGENT_CFG_FILE,
        AMP_RUNNER_FILE,
        CSV_PLAYBACK_FILE,
    ]:
        assert path.is_file(), f"Missing expected perturbation file: {path}"


def test_g1_perturb_env_exports_and_joint_groups_present():
    env_init_text = _read_text(ENV_INIT_FILE)
    env_text = _read_text(ENV_FILE)

    assert "G1PerturbAmpEnv" in env_init_text
    assert "UpperBodyPerturbationCfg" in env_init_text
    assert "G1_UPPER_BODY_JOINT_NAMES" in env_text
    assert "G1_LOWER_BODY_JOINT_NAMES" in env_text
    assert "G1_FULL_BODY_ACTION_JOINT_NAMES" in env_text
    assert "G1_FULL_BODY_SDK_JOINT_NAMES" in env_text
    assert "class G1PerturbAmpEnv(ManagerBasedAmpEnv):" in env_text
    assert "def step(self, action: torch.Tensor):" in env_text
    assert 'source: Literal["sine", "csv", "pose_set"] = "sine"' in env_text
    assert "csv_use_g1_action_order_q_columns" in env_text
    assert 'csv_q_column_joint_order: Literal["lab", "sdk"] = "lab"' in env_text
    assert "csv_randomize_start_on_reset" in env_text
    assert "def _resolve_csv_q_column_source_order" in env_text
    assert "def _resolve_csv_q_target_indices" in env_text
    assert 'cfg.csv_q_column_joint_order == "sdk"' in env_text
    assert "raw_q_values = [float(value) for value in row[-len(q_column_source_order):]]" in env_text
    assert "pose_set: list[list[float]] = []" in env_text


def test_g1_perturb_task_registration_present():
    task_init_text = _read_text(TASK_INIT_FILE)

    assert "LeggedLab-Isaac-AMP-G1-StandPerturb-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-StandPerturb-Play-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0" in task_init_text
    assert 'entry_point="legged_lab.envs:G1PerturbAmpEnv"' in task_init_text


def test_g1_perturb_cfgs_capture_disc_split_and_command_intent():
    stand_text = _read_text(STAND_CFG_FILE)
    walk_text = _read_text(WALK_CFG_FILE)

    assert '_configure_perturbation_common(self)' in stand_text
    assert 'source="csv"' in stand_text
    assert 'csv_use_g1_action_order_q_columns=True' in stand_text
    assert 'csv_q_column_joint_order="sdk"' in stand_text
    assert 'csv_randomize_start_on_reset=True' in stand_text
    assert 'whole_body_joints_20260708_143133.csv' in stand_text
    assert 'self.rewards.track_lin_vel_xy_exp = None' in stand_text
    assert 'self.rewards.track_ang_vel_z_exp = None' in stand_text
    assert 'self.events.reset_from_ref = None' in stand_text
    assert 'self.rewards.torso_roll_pitch_l2.weight = -4.0' in stand_text

    assert 'cfg.observations.disc.joint_pos.params = {"asset_cfg": lower_body_joint_cfg}' in walk_text
    assert 'cfg.observations.disc_demo.ref_joint_pos.params["joint_ids"] = G1_LOWER_BODY_JOINT_IDS' in walk_text
    assert 'cfg.rewards.action_rate_l2 = RewTerm(' in walk_text
    assert 'func=mdp.action_rate_l2_selected' in walk_text
    assert 'cfg.rewards.arm_style_prior = None' in walk_text
    assert 'source="pose_set"' in walk_text
    assert 'G1_WALK_PERTURB_POSE_SET' in walk_text
    assert 'Nav2RecordedVelocityCommandCfg' in walk_text
    assert 'self.curriculum.lin_vel_cmd_levels = CurrTerm(' in walk_text
    assert 'self.curriculum.ang_vel_cmd_levels = CurrTerm(' in walk_text


def test_csv_full_body_playback_defaults_to_sdk_to_lab_mapping():
    playback_text = _read_text(CSV_PLAYBACK_FILE)

    assert 'choices=("sdk", "lab")' in playback_text
    assert 'default="sdk"' in playback_text
    assert "UNITREE_G1_29DOF_CFG.joint_sdk_names" in playback_text
    assert "G1_LOCOMOTION_JOINT_NAMES" in playback_text
    assert "source_to_lab_indices = [source_index_by_name[name] for name in G1_LOCOMOTION_JOINT_NAMES]" in playback_text
    assert "raw_joint_values = [float(value) for value in row[-num_joints:]]" in playback_text
    assert "Reordered CSV q-columns to IsaacLab policy/action order before replay." in playback_text


def test_runner_cfg_and_amp_runner_support_policy_only_resume():
    agent_cfg_text = _read_text(AGENT_CFG_FILE)
    amp_runner_text = _read_text(AMP_RUNNER_FILE)

    assert "load_policy_only = False" in agent_cfg_text
    assert "load_policy_only = True" in agent_cfg_text
    assert 'experiment_name = "g1_amp"' in agent_cfg_text
    assert "style_reward_scale = 0.0" in agent_cfg_text
    assert "task_style_lerp = 1.0" in agent_cfg_text

    assert 'load_policy_only = bool(self.cfg.get("load_policy_only", False))' in amp_runner_text
    assert "Loaded policy-only AMP checkpoint from:" in amp_runner_text
