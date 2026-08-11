"""Static contracts for the isolated ArmHack Walk two-goal specialization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ENV = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_two_goal_env_cfg.py"
REGISTRY = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py"
EXPORTER = ROOT / "legged_lab/scripts/rsl_rl/export_amp_actor_to_onnx.py"
TRAIN = ROOT / "legged_lab/scripts/train_g1_armhack_walk_two_goal_expert.sh"
ACCEPTANCE = ROOT / "legged_lab/scripts/test_g1_armhack_walk_two_goal_mujoco.sh"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_two_goal_tasks_are_walk_only_and_registered_separately():
    registry = text(REGISTRY)
    assert registry.count('entry_point="legged_lab.envs:G1WalkPerturbAmpEnv"') >= 4
    for suffix in ("Lateral-v0", "LateralRobust-v0", "Yaw-v0", "YawRobust-v0"):
        assert f"ArmHackWalkTwoGoal{suffix}" in registry
    train = text(TRAIN)
    assert 'hostname)' in train and 'tata-futurelab' in train
    assert 'RTX 5090' in train
    assert 'refusing to start Walk training' in train


def test_lateral_safe_set_uses_swept_oriented_sole_geometry():
    env = text(ENV)
    for contract in (
        '"half_length": 0.090',
        '"half_width": 0.035',
        '"center_offset_x": 0.035',
        '"interpolation_steps": 12',
        '"overlap_scale": 16.0',
        'hard_clearance=0.030, hard_weight=-300.0',
        'func=mdp.lateral_foot_ordering_l2',
    ):
        assert contract in env


def test_pure_yaw_expert_observes_true_zero_linear_command():
    env = text(ENV)
    assert 'mode_command_clip_min = (0.0, 0.0, -0.45)' in env
    assert 'mode_command_clip_max = (0.0, 0.0, 0.45)' in env
    assert 'func=mdp.pure_yaw_root_rate_error_l2' in env
    exporter = text(EXPORTER)
    assert 'self.has_pure_yaw_expert' in exporter
    assert 'pure_yaw_expert_forward_command' in exporter
    assert 'pure_yaw_expert_lateral_command' in exporter
    assert 'actions = torch.where(pure_yaw.unsqueeze(-1), pure_yaw_actions, actions)' in exporter


def test_mujoco_acceptance_is_goal_specific_and_checks_retention():
    acceptance = text(ACCEPTANCE)
    for contract in (
        'min_signed_sole_clearance_m',
        'sole_clearance_violation_fraction',
        'sign*t["mean_lin_vel_y"] < .18',
        'sign*t["mean_yaw_rate"] < .25',
        'drift > .035',
        'foot_touchdown_count',
        'forward retention',
        'stand retention',
    ):
        assert contract in acceptance
