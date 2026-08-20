"""Static contracts for the isolated ArmHack Walk two-goal specialization."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
ENV = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_two_goal_env_cfg.py"
REGISTRY = ROOT / "legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py"
EXPORTER = ROOT / "legged_lab/scripts/rsl_rl/export_amp_actor_to_onnx.py"
TRAIN = ROOT / "legged_lab/scripts/train_g1_armhack_walk_two_goal_expert.sh"
ACCEPTANCE = ROOT / "legged_lab/scripts/test_g1_armhack_walk_two_goal_mujoco.sh"
INTERACTIVE = ROOT / "scripts/vis_g1_armhack_walk_two_goal_keyboard.sh"
MUJOCO_ADAPTER = ROOT / "unitree_sim2sim2real/deploy/deploy_mujoco/armhack_walk.py"
MUJOCO_RUNNER = ROOT / "unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py"
POSES = ROOT / "legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"
CONTRACT = ROOT / "legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"


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


def test_keyboard_visualization_is_realtime_and_has_disjoint_speed_arm_keys():
    launcher = text(INTERACTIVE)
    runner = text(MUJOCO_RUNNER)
    assert "policy.onnx" in launcher
    assert "env_isaaclab/bin/python" in launcher
    assert "USE_GLFW=True REAL_TIME=True" in launcher
    assert "COMMAND_MODE=keyboard" in launcher
    assert "PYTHONNOUSERSITE=1" in launcher
    assert "SPACE/P 循环" in launcher
    assert 'key not in {"P", "Z", "X", "C"}' in runner
    assert "slow-motion playback is forbidden" in runner
    assert 'walk_command_mode not in {' in runner
    assert '"independent",' in runner and '"keyboard",' in runner


def test_arm_pose_keyboard_cycle_uses_minimum_jerk_without_action_jump():
    python = Path.home() / "anaconda3/envs/gmr/bin/python"
    program = r'''
import importlib.util, sys
import numpy as np
adapter_path, pose_path, contract_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("armhack_walk_interactive_test", adapter_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
names, catalog = module.load_walk_pose_catalog(module.Path(pose_path))
assert names == ("pos1_back", "pos2_down", "pos3_front")
assert [module.minimum_jerk(x) for x in (0.0, 0.5, 1.0)] == [0.0, 0.5, 1.0]
policy_names = list(module.ARM_JOINT_NAMES) + [f"dummy_{index}" for index in range(15)]
adapter = module.ArmHackWalkAdapter({
    "armhack_walk_pose_path": pose_path,
    "armhack_walk_contract_path": contract_path,
    "armhack_walk_pose_name": "pos2_down",
    "armhack_walk_fixed_command": [0.0, 0.0, 0.0],
    "armhack_walk_start_active": False,
    "armhack_walk_pose_transition_s": 2.0,
    "command_mode": "keyboard",
}, policy_names, np.zeros(29, dtype=np.float32))
zero_action = np.zeros(29, dtype=np.float32)
adapter.key_callback(32)
start = adapter.compose_action(zero_action, 0.0)[adapter.arm_policy_indices] * 0.25
middle = adapter.compose_action(zero_action, 1.0)[adapter.arm_policy_indices] * 0.25
finish = adapter.compose_action(zero_action, 2.0)[adapter.arm_policy_indices] * 0.25
assert np.allclose(start, catalog["pos2_down"])
assert np.allclose(middle, 0.5 * (catalog["pos2_down"] + catalog["pos3_front"]))
assert np.allclose(finish, catalog["pos3_front"])
assert adapter.summary()["pose_switch_count"] == 1
'''
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(python), "-c", program, str(MUJOCO_ADAPTER), str(POSES), str(CONTRACT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
