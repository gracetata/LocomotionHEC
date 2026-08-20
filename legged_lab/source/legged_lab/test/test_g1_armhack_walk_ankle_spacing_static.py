"""Static contracts for the isolated 30-cm ArmHack Walk fine-tune."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MDP = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"
CFG = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_ankle_spacing_env_cfg.py"
REGISTRY = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py"
AGENT = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py"
TRAIN = ROOT / "legged_lab/scripts/train_g1_armhack_walk_ankle_spacing.sh"
MUJOCO = ROOT / "legged_lab/scripts/test_g1_armhack_walk_ankle_spacing_mujoco.sh"
MUJOCO_RUNNER = ROOT / "unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py"
BOOTSTRAP = ROOT / "legged_lab/scripts/bootstrap_g1_armhack_walk_ankle_spacing.py"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_kernel_is_explicit_3d_symmetric_target_reward():
    source = text(MDP)
    block = source[source.index("def ankle_distance_target_kernel"):source.index("def feet_stumble")]
    assert 'target_distance: float = 0.30' in block
    assert 'asset.data.body_pos_w[:, asset_cfg.body_ids, :]' in block
    assert 'torch.linalg.vector_norm' in block
    assert 'torch.exp(-0.5 * torch.square(normalized_error))' in block


def test_same_large_kernel_is_applied_to_all_three_gated_actors():
    source = text(CFG)
    assert 'ANKLE_DISTANCE_TARGET_M = 0.30' in source
    assert 'ANKLE_DISTANCE_KERNEL_STD_M = 0.06' in source
    assert 'ANKLE_DISTANCE_KERNEL_WEIGHT = 500.0' in source
    assert source.count('_configure_ankle_spacing_kernel(self)') == 3
    assert 'G1WalkAnkleSpacingBaseEnvCfg' in source
    assert 'G1WalkAnkleSpacingLateralEnvCfg' in source
    assert 'G1WalkAnkleSpacingYawEnvCfg' in source
    assert 'self.rewards.pure_yaw_planar_drift_l2.weight = -50.0' in source


def test_registry_and_launcher_are_walk_only_and_future_locked():
    registry = text(REGISTRY)
    train = text(TRAIN)
    assert '("Base", "G1WalkAnkleSpacingBaseEnvCfg")' in registry
    assert '("Lateral", "G1WalkAnkleSpacingLateralEnvCfg")' in registry
    assert '("Yaw", "G1WalkAnkleSpacingYawEnvCfg")' in registry
    assert 'id=f"LeggedLab-Isaac-AMP-G1-ArmHackWalkAnkleSpacing{branch}-v0"' in registry
    assert 'tata-futurelab' in train and 'RTX 5090' in train
    assert 'Stand training is active' in train
    assert 'KL_SCALE=${KL_SCALE:-1.00}' in train
    assert 'freeze_actor_hidden_layers=0' in train


def test_optimizer_is_conservative_and_has_retention_anchor():
    source = text(AGENT)
    block = source[source.index("class G1WalkAnkleSpacingRslRlOnPolicyRunnerAmpCfg"):]
    assert 'self.algorithm.learning_rate = 2.0e-5' in block
    assert 'self.algorithm.clip_param = 0.05' in block
    assert 'self.algorithm.baseline_kl_cfg.enabled = True' in block
    assert 'self.algorithm.baseline_kl_cfg.scale = 1.00' in block
    assert 'self.algorithm.baseline_kl_cfg.hard_limit = 0.25' in block


def test_mujoco_acceptance_measures_target_and_retains_all_commands():
    acceptance = text(MUJOCO)
    runner = text(MUJOCO_RUNNER)
    assert 'ankle_distances_m' in runner
    assert 'rmse_to_target_m' in runner
    assert 'within_0p05_fraction' in runner
    for scenario in ("stand", "forward", "backward", "lateral_left", "lateral_right", "diagonal", "yaw_left", "yaw_right"):
        assert scenario in acceptance
    assert 'candidate_mean_rmse > 0.90 * baseline_mean_rmse' in acceptance
    assert 'mean_within5 < 0.35' in acceptance


def test_bootstrap_is_symmetric_bounded_and_only_changes_hip_roll_bias():
    source = text(BOOTSTRAP)
    assert 'LEFT_HIP_ROLL_ACTION_INDEX = 3' in source
    assert 'RIGHT_HIP_ROLL_ACTION_INDEX = 4' in source
    assert 'ACTION_SCALE_RAD = 0.25' in source
    assert 'bias[LEFT_HIP_ROLL_ACTION_INDEX] += args.action_bias' in source
    assert 'bias[RIGHT_HIP_ROLL_ACTION_INDEX] -= args.action_bias' in source
    assert '0.0 < args.action_bias <= 0.60' in source
