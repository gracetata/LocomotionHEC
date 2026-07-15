import csv
import hashlib
import json
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent / "legged_lab"
PROJECT_ROOT = TEST_DIR.parents[2]

ENV_FILE = PACKAGE_ROOT / "envs" / "g1_perturb_env.py"
WALK_ENV_FILE = PACKAGE_ROOT / "envs" / "g1_walk_perturb_env.py"
ENV_INIT_FILE = PACKAGE_ROOT / "envs" / "__init__.py"
TASK_INIT_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "__init__.py"
)
STAND_CFG_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "g1_stand_perturb_env_cfg.py"
)
STAND_RANDOMIZED_CFG_FILE = (
    PACKAGE_ROOT
    / "tasks"
    / "locomotion"
    / "amp"
    / "config"
    / "g1_perturb"
    / "g1_stand_randomized_payload_env_cfg.py"
)
WALK_CFG_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "g1_walk_perturb_env_cfg.py"
)
AGENT_CFG_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "agents" / "rsl_rl_ppo_cfg.py"
)
REFERENCE_DATA_MODULE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "config" / "g1_perturb" / "reference_data.py"
)
STAND_REFERENCE_CSV = (
    PROJECT_ROOT / "Reference Data" / "ArmHack" / "StandPerturb" / "g1_arm_trajectory_named_50hz.csv"
)
STAND_RAW_CSV = (
    PROJECT_ROOT
    / "Reference Data"
    / "ArmHack"
    / "StandPerturb"
    / "raw"
    / "g1_full_body_motion_sdk_50hz.csv"
)
WALK_REFERENCE_POSES = (
    PROJECT_ROOT / "Reference Data" / "ArmHack" / "WalkPerturbFinetune" / "g1_arm_pose_set.json"
)
AMP_RUNNER_FILE = Path(__file__).resolve().parents[4] / "rsl_rl" / "rsl_rl" / "runners" / "amp_runner.py"
NAV2_COMMAND_FILE = (
    PACKAGE_ROOT / "tasks" / "locomotion" / "amp" / "mdp" / "commands" / "nav2_recorded_velocity_command.py"
)
CSV_PLAYBACK_FILE = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "visualize_g1_csv_full_body_motion.py"
REFERENCE_CHECK_FILE = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "check_armhack_reference_data.py"
TRAIN_SCRIPT_FILE = Path(__file__).resolve().parents[3] / "scripts" / "train_g1_amp.sh"
STAND_TRAIN_SCRIPT_FILE = (
    Path(__file__).resolve().parents[3] / "scripts" / "train_g1_armhack_stand.sh"
)
STAND_RANDOMIZED_TRAIN_SCRIPT_FILE = (
    Path(__file__).resolve().parents[3] / "scripts" / "train_g1_armhack_stand_randomized_payload.sh"
)
STAND_RANDOM_DATA_BUILDER_FILE = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "tools"
    / "build_armhack_stand_randomized_training_data.py"
)
STAND_RANDOM_POSE_BANK = (
    PROJECT_ROOT
    / "Reference Data"
    / "ArmHack"
    / "StandPerturb"
    / "RandomizedTraining"
    / "random_arm_pose_bank_seed20260715.json"
)
STAND_VIS_EVAL_SCRIPT_FILE = (
    Path(__file__).resolve().parents[3] / "scripts" / "vis_g1_armhack_stand_eval.sh"
)
STAND_VIS_BUILDER_FILE = (
    Path(__file__).resolve().parents[3] / "scripts" / "tools" / "build_armhack_stand_visualization_suite.py"
)
STAND_VIS_MANIFEST = (
    PROJECT_ROOT
    / "Reference Data"
    / "ArmHack"
    / "StandPerturb"
    / "TestData"
    / "ArmOnly"
    / "manifest.json"
)
PLAY_SCRIPT_FILE = Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl" / "play.py"
WALK_TRAIN_SCRIPT_FILE = (
    Path(__file__).resolve().parents[3] / "scripts" / "train_g1_armhack_walk.sh"
)
VIS_SCRIPT_FILE = Path(__file__).resolve().parents[3] / "scripts" / "vis_isaacsim_g1_amp.sh"
CHECKPOINT_README = PROJECT_ROOT / "ArmHack Checkpoints" / "README.md"
STAND_CHECKPOINT_KEEP = PROJECT_ROOT / "ArmHack Checkpoints" / "StandPerturb" / ".gitkeep"
WALK_CHECKPOINT_KEEP = PROJECT_ROOT / "ArmHack Checkpoints" / "WalkPerturbFinetune" / ".gitkeep"
WALK_BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "ArmHack Checkpoints"
    / "WalkPerturbFinetune"
    / "BaselineLocomotionModel9996"
    / "model_9996.pt"
)
WALK_BASE_ONNX = PROJECT_ROOT.parent / "checkpoint" / "model_9996" / "locomotion.onnx"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_g1_perturb_files_exist():
    for path in [
        ENV_FILE,
        WALK_ENV_FILE,
        ENV_INIT_FILE,
        TASK_INIT_FILE,
        STAND_CFG_FILE,
        STAND_RANDOMIZED_CFG_FILE,
        WALK_CFG_FILE,
        AGENT_CFG_FILE,
        REFERENCE_DATA_MODULE,
        STAND_REFERENCE_CSV,
        STAND_RAW_CSV,
        WALK_REFERENCE_POSES,
        AMP_RUNNER_FILE,
        NAV2_COMMAND_FILE,
        CSV_PLAYBACK_FILE,
        REFERENCE_CHECK_FILE,
        TRAIN_SCRIPT_FILE,
        STAND_TRAIN_SCRIPT_FILE,
        STAND_RANDOMIZED_TRAIN_SCRIPT_FILE,
        STAND_RANDOM_DATA_BUILDER_FILE,
        STAND_RANDOM_POSE_BANK,
        STAND_VIS_EVAL_SCRIPT_FILE,
        STAND_VIS_BUILDER_FILE,
        STAND_VIS_MANIFEST,
        PLAY_SCRIPT_FILE,
        WALK_TRAIN_SCRIPT_FILE,
        VIS_SCRIPT_FILE,
        CHECKPOINT_README,
        STAND_CHECKPOINT_KEEP,
        WALK_CHECKPOINT_KEEP,
    ]:
        assert path.is_file(), f"Missing expected perturbation file: {path}"


def test_g1_perturb_env_exports_and_joint_groups_present():
    env_init_text = _read_text(ENV_INIT_FILE)
    env_text = _read_text(ENV_FILE)
    walk_env_text = _read_text(WALK_ENV_FILE)

    assert "G1PerturbAmpEnv" in env_init_text
    assert "UpperBodyPerturbationCfg" in env_init_text
    assert "G1_UPPER_BODY_JOINT_NAMES" in env_text
    assert "G1_LOWER_BODY_JOINT_NAMES" in env_text
    assert "G1_FULL_BODY_ACTION_JOINT_NAMES" in env_text
    assert "G1_FULL_BODY_SDK_JOINT_NAMES" in env_text
    assert "class G1PerturbAmpEnv(ManagerBasedAmpEnv):" in env_text
    assert "def step(self, action: torch.Tensor):" in env_text
    assert 'source: Literal["sine", "csv", "pose_set", "random_pose_trajectory"] = "sine"' in env_text
    assert "_advance_random_pose_trajectories" in env_text
    assert "contains_future_policy_observation" not in env_text
    assert "csv_use_g1_action_order_q_columns" in env_text
    assert 'csv_q_column_joint_order: Literal["lab", "sdk"] = "lab"' in env_text
    assert "csv_randomize_start_on_reset" in env_text
    assert "csv_initialize_joint_state_on_reset" in env_text
    assert "csv_curriculum_enabled" in env_text
    assert "def _csv_curriculum_motion_scale" in env_text
    assert "def _advance_csv_sample_times" in env_text
    assert "def _initialize_csv_arm_joint_state" in env_text
    assert "torch.lerp(lower_targets, upper_targets" in env_text
    assert "robot.write_joint_state_to_sim(" in env_text
    assert 'log_extras["ArmHack/csv_start_time_std_s"]' in env_text
    assert "def _resolve_csv_q_column_source_order" in env_text
    assert "def _resolve_csv_q_target_indices" in env_text
    assert 'cfg.csv_q_column_joint_order == "sdk"' in env_text
    assert "raw_q_values = [float(value) for value in row[-len(q_column_source_order):]]" in env_text
    assert "pose_set: list[list[float]] = []" in env_text
    assert "G1WalkPerturbAmpEnv" in env_init_text
    assert "WalkUpperBodyPerturbationCfg" in env_init_text
    assert "class G1WalkPerturbAmpEnv(G1PerturbAmpEnv):" in walk_env_text
    assert "def _initialize_walk_arm_state" in walk_env_text
    assert "robot.write_joint_state_to_sim(" in walk_env_text
    assert "robot.set_joint_position_target(" in walk_env_text
    assert "self.action_manager._prev_action" in walk_env_text
    assert 'log_extras["ArmHack/walk_pose_init_max_error_rad"]' in walk_env_text


def test_g1_perturb_task_registration_present():
    task_init_text = _read_text(TASK_INIT_FILE)

    assert "LeggedLab-Isaac-AMP-G1-StandPerturb-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-StandPerturb-Play-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-Play-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0" in task_init_text
    assert "LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0" in task_init_text
    assert 'entry_point="legged_lab.envs:G1PerturbAmpEnv"' in task_init_text
    assert 'entry_point="legged_lab.envs:G1WalkPerturbAmpEnv"' in task_init_text


def test_g1_perturb_cfgs_capture_disc_split_and_command_intent():
    stand_text = _read_text(STAND_CFG_FILE)
    walk_text = _read_text(WALK_CFG_FILE)

    assert '_configure_perturbation_common(self)' in stand_text
    assert 'source="csv"' in stand_text
    assert 'csv_use_g1_action_order_q_columns=False' in stand_text
    assert 'csv_q_column_joint_order="sdk"' in stand_text
    assert 'csv_randomize_start_on_reset=True' in stand_text
    assert 'csv_initialize_joint_state_on_reset=True' in stand_text
    assert 'csv_curriculum_static_steps=12_000' in stand_text
    assert 'csv_curriculum_ramp_steps=24_000' in stand_text
    assert 'csv_curriculum_motion_scale=1.0' in stand_text
    assert "STAND_ARM_MOTION_RELATIVE_PATH" in stand_text
    assert 'self.rewards.track_lin_vel_xy_exp = None' in stand_text
    assert 'self.rewards.track_ang_vel_z_exp = None' in stand_text
    assert 'self.events.reset_from_ref = None' in stand_text
    assert 'self.events.push_robot = None' in stand_text
    assert 'self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)' in stand_text
    assert 'func=mdp.double_support' in stand_text
    assert 'func=mdp.root_xy_position_l2' in stand_text
    assert 'self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)' in stand_text
    assert 'if self.rewards.termination_penalty is not None' not in stand_text

    assert 'cfg.observations.disc.joint_pos.params = {"asset_cfg": lower_body_joint_cfg}' in walk_text
    assert 'cfg.observations.disc_demo.ref_joint_pos.params["joint_ids"] = G1_LOWER_BODY_JOINT_IDS' in walk_text
    assert 'cfg.rewards.action_rate_l2 = RewTerm(' in walk_text
    assert 'func=mdp.action_rate_l2_selected' in walk_text
    assert 'cfg.rewards.arm_style_prior = None' in walk_text
    assert 'source="pose_set"' in walk_text
    assert "class G1WalkPerturbFinetuneEnvCfg(G1AmpNav2FinetuneEnvCfg):" in walk_text
    assert "WalkUpperBodyPerturbationCfg" in walk_text
    assert "G1_WALK_PERTURB_POSE_NAMES" in walk_text
    assert 'pose_name="pos2_down"' in walk_text
    assert "initialize_joint_state_on_reset=True" in walk_text
    assert 'G1_WALK_PERTURB_POSE_SET' in walk_text
    assert 'self.commands.base_velocity = mdp.Nav2RecordedVelocityCommandCfg(' in walk_text
    assert "G1_WALK_PERTURB_NAV2_COMMAND_PATH" in walk_text
    assert 'augmentation_filter="none,mirror_lr"' in walk_text
    assert "synthesize_mirror_lr=True" in walk_text
    assert 'scenario_family_filter="complex_turn"' in walk_text
    assert 'window_duration_s=4.0' in walk_text
    assert 'command_scale=(0.85, 0.75, 0.75)' in walk_text
    assert 'smoothing_time_constant=0.30' in walk_text
    assert "CurrTerm" not in walk_text
    assert "lin_vel_cmd_levels" not in walk_text
    assert "ang_vel_cmd_levels" not in walk_text
    assert "self.rewards.track_lin_vel_xy_exp.weight = 1.8" in walk_text
    assert "self.rewards.track_ang_vel_z_exp.weight = 1.5" in walk_text
    assert "self.rewards.torso_roll_pitch_l2.weight = -0.04" in walk_text
    assert "self.rewards.action_rate_l2.weight = -0.006" in walk_text
    assert "self.rewards.termination_penalty.weight = -200.0" in walk_text


def test_armhack_reference_data_is_repository_relative_and_valid():
    reference_data_text = _read_text(REFERENCE_DATA_MODULE)
    stand_text = _read_text(STAND_CFG_FILE)
    walk_text = _read_text(WALK_CFG_FILE)
    nav2_command_text = _read_text(NAV2_COMMAND_FILE)

    assert 'Path("Reference Data") / "ArmHack"' in reference_data_text
    assert '"nav2_cmd_vel_raw_success.csv"' in reference_data_text
    assert "/home/" not in reference_data_text
    assert "/home/" not in stand_text
    assert "/home/" not in walk_text
    assert "def _resolve_nav2_data_path(data_path: str) -> Path:" in nav2_command_text
    assert "_LEGGED_LAB_PROJECT_DIR / path" in nav2_command_text
    assert hashlib.sha256(STAND_RAW_CSV.read_bytes()).hexdigest() == (
        "b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50"
    )
    assert hashlib.sha256(STAND_REFERENCE_CSV.read_bytes()).hexdigest() == (
        "afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601"
    )

    with WALK_REFERENCE_POSES.open("r", encoding="utf-8") as handle:
        pose_payload = json.load(handle)
    assert pose_payload["units"] == "rad"
    assert pose_payload["joint_order_per_arm"] == [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    assert [pose["name"] for pose in pose_payload["poses"]] == ["pos1_back", "pos2_down", "pos3_front"]
    assert all(len(pose[side]) == 7 for pose in pose_payload["poses"] for side in ("left", "right"))
    pos1 = pose_payload["poses"][0]
    interleaved = [value for pair in zip(pos1["left"], pos1["right"]) for value in pair]
    assert interleaved == [
        0.91,
        0.91,
        0.52,
        -0.52,
        0.11,
        -0.11,
        0.01,
        0.01,
        -0.12,
        0.12,
        -1.03,
        -1.03,
        0.01,
        -0.01,
    ]

    with STAND_VIS_MANIFEST.open("r", encoding="utf-8") as handle:
        visualization_manifest = json.load(handle)
    assert visualization_manifest["source"]["sha256"] == (
        "afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601"
    )
    assert visualization_manifest["generation"]["seed"] == 20260714
    assert visualization_manifest["generation"]["runtime_random_sampling"] is False
    assert visualization_manifest["generation"]["trajectory_speed_scale"] == 1.0
    assert visualization_manifest["data_scope"] == "arm_only_14_dof"
    assert visualization_manifest["contains_full_body_state"] is False
    assert visualization_manifest["generation"]["controlled_joint_count"] == 14
    assert visualization_manifest["generation"]["controlled_joint_names"] == [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    assert len(visualization_manifest["representative_poses"]) == 6
    assert len(visualization_manifest["representative_trajectories"]) == 4
    assert len(visualization_manifest["synthesized_poses"]) == 3
    assert len(visualization_manifest["synthesized_trajectories"]) == 3
    assert len(visualization_manifest["randomized_poses"]) == 8
    assert len(visualization_manifest["randomized_trajectories"]) == 6
    assert visualization_manifest["schema_version"] == 5
    assert visualization_manifest["random_pose_bank"]["seed"] == 20260715
    assert visualization_manifest["random_pose_bank"]["bank_size"] == 512
    detailed_timeline = visualization_manifest["files"]["all"]["detailed_timeline"]
    assert len(detailed_timeline) == 59
    assert detailed_timeline[0]["label"] == "representative_pose_01"
    assert detailed_timeline[-1]["label"] == "randomized_trajectory_06"
    assert detailed_timeline[0]["start_s"] == 0.0
    assert detailed_timeline[-1]["end_s"] == visualization_manifest["files"]["all"]["duration_s"]
    assert all(
        trajectory["equivalent_source_speed"] == 1.0
        for trajectory in visualization_manifest["representative_trajectories"]
    )
    assert all(
        trajectory["equivalent_source_speed"] == 1.0
        for trajectory in visualization_manifest["synthesized_trajectories"]
    )


def test_stand_random_pose_bank_is_arm_only_and_reproducible():
    pose_bank = json.loads(STAND_RANDOM_POSE_BANK.read_text(encoding="utf-8"))
    assert pose_bank["schema_version"] == 1
    assert pose_bank["data_scope"] == "arm_only_14_dof"
    assert pose_bank["generation"]["seed"] == 20260715
    assert pose_bank["generation"]["bank_size"] == 512
    assert pose_bank["generation"]["source_anchor_count"] == 64
    assert pose_bank["generation"]["random_convex_pose_count"] == 448
    assert pose_bank["generation"]["independent_per_joint_uniform_sampling"] is False
    assert pose_bank["generation"]["contains_future_policy_observation"] is False
    assert len(pose_bank["joint_names"]) == 14
    assert len(pose_bank["interpolation_velocity_limits_rad_s"]) == 14
    assert len(pose_bank["poses"]) == 512
    assert all(len(pose["positions_rad"]) == 14 for pose in pose_bank["poses"])


def test_stand_reference_csv_contains_only_named_arm_columns():
    expected_arm_joint_names = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    with STAND_REFERENCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == ["time_s", *expected_arm_joint_names]
        row_count = 0
        for row in reader:
            assert len(row) == 15
            assert len([float(value) for value in row]) == 15
            row_count += 1
    assert row_count == 20122


def test_csv_full_body_playback_defaults_to_sdk_to_lab_mapping():
    playback_text = _read_text(CSV_PLAYBACK_FILE)

    assert 'choices=("sdk", "lab")' in playback_text
    assert 'default="sdk"' in playback_text
    assert "UNITREE_G1_29DOF_CFG.joint_sdk_names" in playback_text
    assert "G1_LOCOMOTION_JOINT_NAMES" in playback_text
    assert "source_to_lab_indices = [source_index_by_name[name] for name in G1_LOCOMOTION_JOINT_NAMES]" in playback_text
    assert "raw_joint_values = [float(value) for value in row[-num_joints:]]" in playback_text
    assert "Reordered CSV q-columns to IsaacLab policy/action order before replay." in playback_text
    assert '"Reference Data/ArmHack/StandPerturb/raw/"' in playback_text


def test_runner_cfg_and_amp_runner_support_policy_only_resume():
    agent_cfg_text = _read_text(AGENT_CFG_FILE)
    amp_runner_text = _read_text(AMP_RUNNER_FILE)

    assert "load_policy_only = False" in agent_cfg_text
    assert "load_policy_only = True" in agent_cfg_text
    assert 'experiment_name = "g1_walk_perturb"' in agent_cfg_text
    assert 'checkpoint_output_dir = "ArmHack Checkpoints/StandPerturb"' in agent_cfg_text
    assert 'checkpoint_output_dir = "ArmHack Checkpoints/WalkPerturbFinetune"' in agent_cfg_text
    assert "style_reward_scale = 0.0" in agent_cfg_text
    assert "task_style_lerp = 1.0" in agent_cfg_text

    assert 'load_policy_only = bool(self.cfg.get("load_policy_only", False))' in amp_runner_text
    assert "Loaded policy-only AMP checkpoint from:" in amp_runner_text
    assert 'checkpoint_output_dir = self.cfg.get("checkpoint_output_dir")' in amp_runner_text
    assert "shutil.copy2(primary_path, exported_path)" in amp_runner_text


def test_train_scripts_have_isolated_working_defaults():
    train_script_text = _read_text(TRAIN_SCRIPT_FILE)
    stand_train_script_text = _read_text(STAND_TRAIN_SCRIPT_FILE)
    stand_randomized_train_script_text = _read_text(STAND_RANDOMIZED_TRAIN_SCRIPT_FILE)
    stand_randomized_cfg_text = _read_text(STAND_RANDOMIZED_CFG_FILE)
    stand_random_data_builder_text = _read_text(STAND_RANDOM_DATA_BUILDER_FILE)
    walk_train_script_text = _read_text(WALK_TRAIN_SCRIPT_FILE)
    vis_script_text = _read_text(VIS_SCRIPT_FILE)
    stand_vis_eval_text = _read_text(STAND_VIS_EVAL_SCRIPT_FILE)
    stand_vis_builder_text = _read_text(STAND_VIS_BUILDER_FILE)
    play_script_text = _read_text(PLAY_SCRIPT_FILE)

    assert "CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_isaaclab}" in train_script_text
    assert "CONDA_BASE=${CONDA_BASE:-${HOME}/anaconda3}" in train_script_text
    assert 'if [[ "${TASK}" == *"StandPerturb"* ]]; then' in train_script_text
    assert "STYLE_REWARD_SCALE=0.0" in train_script_text
    assert "TASK_STYLE_LERP=1.0" in train_script_text
    assert "RSI_ENABLE=False" in train_script_text
    assert "BaselineModel9996/model_9996.pt" in stand_train_script_text
    assert "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6" in stand_train_script_text
    assert "STATIC_ITERATIONS=${STATIC_ITERATIONS:-500}" in stand_train_script_text
    assert "RAMP_ITERATIONS=${RAMP_ITERATIONS:-1000}" in stand_train_script_text
    assert "FINAL_MOTION_SCALE=${FINAL_MOTION_SCALE:-1.0}" in stand_train_script_text
    assert "BASELINE_KL_ENABLE=True" in stand_train_script_text
    assert "agent.load_policy_only=True" in stand_train_script_text
    assert 'TASK="LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-v0"' in stand_randomized_train_script_text
    assert "model_2999.pt" in stand_randomized_train_script_text
    assert "2c87cc2cc3706c1024594d14d85a34e7bf468b54f6b66e49b6155ef72a2dbd16" in stand_randomized_train_script_text
    assert "PAYLOAD_MAX_KG=${PAYLOAD_MAX_KG:-1.0}" in stand_randomized_train_script_text
    assert "random_curriculum_static_steps" in stand_randomized_train_script_text
    assert "agent.load_policy_only=True" in stand_randomized_train_script_text
    assert "agent.reset_iteration_on_policy_only_load=True" in stand_randomized_train_script_text
    assert 'body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]' in stand_randomized_cfg_text
    assert '"mass_distribution_params": (0.0, 1.0)' in stand_randomized_cfg_text
    assert 'perturbation.source = "random_pose_trajectory"' in stand_randomized_cfg_text
    assert "2-to-4-parent Dirichlet convex combinations" in stand_random_data_builder_text
    assert "MIN_SAFE_VELOCITY_RAD_S = 0.20" in stand_random_data_builder_text
    assert 'TASK="LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0"' in walk_train_script_text
    assert "checkpoint/model_9996/locomotion.onnx" in walk_train_script_text
    assert "BaselineLocomotionModel9996/model_9996.pt" in walk_train_script_text
    assert "05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6" in walk_train_script_text
    assert "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6" in walk_train_script_text
    assert "Verified locomotion.onnx == model_9996 actor" in walk_train_script_text
    assert 'ROBOT_ASSET_NAME="s3_g1_29dof"' in walk_train_script_text
    assert 'EXPERIMENT_NAME="g1_walk_perturb"' in walk_train_script_text
    assert "MODE=${MODE:-init}" in walk_train_script_text
    assert "POSE_NAME=${POSE_NAME:-pos2_down}" in walk_train_script_text
    assert "RSI_ENABLE=${RSI_ENABLE:-False}" in walk_train_script_text
    assert "76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a" in walk_train_script_text
    assert "agent.load_policy_only=\"${LOAD_POLICY_ONLY}\"" in walk_train_script_text
    assert "BASELINE_KL_ENABLE=True" in walk_train_script_text
    assert "ArmHack Checkpoints/StandPerturb" not in walk_train_script_text
    assert "g1_stand_perturb" not in walk_train_script_text
    assert WALK_BASE_CHECKPOINT.is_file()
    assert hashlib.sha256(WALK_BASE_CHECKPOINT.read_bytes()).hexdigest() == (
        "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"
    )
    assert WALK_BASE_ONNX.is_file()
    assert hashlib.sha256(WALK_BASE_ONNX.read_bytes()).hexdigest() == (
        "05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6"
    )
    assert "CONDA_ENV_NAME=${CONDA_ENV_NAME:-env_isaaclab}" in vis_script_text
    assert "CONDA_BASE=${CONDA_BASE:-${HOME}/anaconda3}" in vis_script_text
    assert "TestData/ArmOnly" in stand_vis_eval_text
    assert "all_arm_only_evaluation_sequence_seed20260714_50hz.csv" in stand_vis_eval_text
    assert "2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715" in stand_vis_eval_text
    assert "Test Reports/StandArmOnly" in stand_vis_eval_text
    assert 'CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}"' in stand_vis_eval_text
    assert "MODEL_ID=${MODEL_ID:-${CHECKPOINT_STEM}_${CHECKPOINT_SHORT_SHA}}" in stand_vis_eval_text
    assert 'REPORT_CONDITION_ID="${TEST_ID}__payload_${PAYLOAD_TAG}kg"' in stand_vis_eval_text
    assert '--armhack_stand_manifest "${MANIFEST}"' in stand_vis_eval_text
    assert "csv_randomize_start_on_reset=False" in stand_vis_eval_text
    assert "csv_curriculum_enabled=False" in stand_vis_eval_text
    assert "csv_curriculum_motion_scale=1.0" in stand_vis_eval_text
    assert 'EXTRA_HYDRA_ARGS=""' in stand_vis_eval_text
    assert "would change the fixed deterministic ArmHack evaluation protocol" in stand_vis_eval_text
    assert "runtime_random_sampling" in stand_vis_builder_text
    assert "np.random.default_rng(args.seed)" in stand_vis_builder_text
    assert '"data_scope": "arm_only_14_dof"' in stand_vis_builder_text
    assert '"contains_full_body_state": False' in stand_vis_builder_text
    assert '"trajectory_speed_scale": args.trajectory_speed_scale' in stand_vis_builder_text
    assert '"schema_version": 5' in stand_vis_builder_text
    assert '"training_sampling_contract"' in stand_vis_builder_text
    assert '"detailed_timeline": all_detailed_timeline' in stand_vis_builder_text
    assert "REMOVED_ARMS_DOWN_SOURCE_TIME_S = 404.897585" in stand_vis_builder_text
    assert "maximum_static_pose_source_time_s" in stand_vis_builder_text
    assert "randomized_poses" in stand_vis_eval_text
    assert "randomized_trajectories" in stand_vis_eval_text
    assert 'TASK=LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-Play-v0' in stand_vis_eval_text
    assert "PAYLOAD_KG=${PAYLOAD_KG:-0.0}" in stand_vis_eval_text
    assert '--armhack_stand_payload_kg "${PAYLOAD_KG}"' in stand_vis_eval_text
    assert '"--armhack_stand_manifest"' in play_script_text
    assert '"--armhack_stand_payload_kg"' in play_script_text
    assert "body_pos_w" in play_script_text
    assert "euler_xyz_from_quat" in play_script_text
    assert "torso_world_6d.png" in play_script_text
    assert "躯干世界坐标系 6D 位移" in play_script_text
    assert "--armhack_stand_report_path" in play_script_text
    assert "mean_abs_step_delta" in play_script_text
    assert "左/右腕末端附加质量" in play_script_text
