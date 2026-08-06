import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
LEGGED_LAB_ROOT = REPO_ROOT / "legged_lab"
PACKAGE_ROOT = LEGGED_LAB_ROOT / "source" / "legged_lab" / "legged_lab"
AMP_ROOT = PACKAGE_ROOT / "tasks" / "locomotion" / "amp"
G1_ROOT = AMP_ROOT / "config" / "g1"
REWARD_MATH_FILE = AMP_ROOT / "mdp" / "reward_math.py"
ENV_CFG_FILE = G1_ROOT / "g1_amp_env_cfg.py"
REGISTRY_FILE = G1_ROOT / "__init__.py"
AGENT_CFG_FILE = G1_ROOT / "agents" / "rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = LEGGED_LAB_ROOT / "scripts" / "train_g1_amp_nav2_two_goal.sh"
MODE_CONFIG = (
    PACKAGE_ROOT
    / "data"
    / "MotionData"
    / "g1_29dof"
    / "amp"
    / "nav2_behavior_50hz"
    / "task_sampling_two_goal_config.json"
)
SOURCE = (
    LEGGED_LAB_ROOT
    / "logs"
    / "rsl_rl"
    / "g1_amp_nav2_behavior"
    / "2026-08-04_14-12-30_nav2_behavior_from_model9996_fullstate_3000_20260804"
    / "model_12995.pt"
)
SOURCE_SIZE = 16_202_843
SOURCE_SHA256 = "6862627cdfe5cc95a1c0916c17bbde50d320c0a551da0ab8312bfbce05f09a70"


def _load_reward_math():
    spec = importlib.util.spec_from_file_location("g1_two_goal_reward_math", REWARD_MATH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REWARD_MATH = _load_reward_math()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rect(center_x: float, center_y: float, yaw: float = 0.0) -> torch.Tensor:
    corners = torch.tensor(
        [[-0.09, -0.035], [0.09, -0.035], [0.09, 0.035], [-0.09, 0.035]],
        dtype=torch.float32,
    )
    rotation = torch.tensor(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=torch.float32,
    )
    return corners @ rotation.T + torch.tensor([center_x, center_y])


def test_protected_source_is_exact():
    assert SOURCE.stat().st_size == SOURCE_SIZE
    assert _sha256(SOURCE) == SOURCE_SHA256


def test_distribution_contains_only_balanced_lateral_and_pure_yaw_modes():
    config = json.loads(MODE_CONFIG.read_text())
    assert config["mode_weights"] == {
        "lateral_left": 0.25,
        "lateral_right": 0.25,
        "turn_in_place_left": 0.25,
        "turn_in_place_right": 0.25,
    }
    for name, mode in config["modes"].items():
        if name.startswith("lateral"):
            assert mode["lin_vel_x"] == [0.0, 0.0]
            assert mode["ang_vel_z"] == [0.0, 0.0]
            assert max(abs(value) for value in mode["lin_vel_y"]) >= 0.15
        else:
            assert mode["lin_vel_x"] == [0.0, 0.0]
            assert mode["lin_vel_y"] == [0.0, 0.0]
            assert max(abs(value) for value in mode["ang_vel_z"]) >= 0.20


def test_two_goal_masks_and_progress_have_no_stationary_credit():
    command = torch.tensor(
        [
            [0.0, 0.25, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 0.30],
            [0.0, 0.0, -0.30],
            [0.30, 0.0, 0.0],
        ]
    )
    lateral, pure_yaw = REWARD_MATH.two_goal_command_masks(command)
    torch.testing.assert_close(lateral, torch.tensor([True, True, False, False, False]))
    torch.testing.assert_close(pure_yaw, torch.tensor([False, False, True, True, False]))

    stationary = REWARD_MATH.signed_command_progress_ratio(
        command[:4, 1] + command[:4, 2], torch.zeros(4), min_command=0.10
    )
    torch.testing.assert_close(stationary, torch.zeros(4))
    matched = REWARD_MATH.signed_command_progress_ratio(
        torch.tensor([0.25, -0.25]), torch.tensor([0.25, -0.25]), min_command=0.10
    )
    wrong = REWARD_MATH.signed_command_progress_ratio(
        torch.tensor([0.25, -0.25]), torch.tensor([-0.10, 0.10]), min_command=0.10
    )
    torch.testing.assert_close(matched, torch.ones(2))
    assert torch.all(wrong < 0.0)


def test_swept_sole_geometry_detects_between_frame_crossing():
    left_previous = left_current = _rect(0.0, 0.0).unsqueeze(0)
    right_previous = _rect(0.0, 0.20).unsqueeze(0)
    right_current = _rect(0.0, -0.20).unsqueeze(0)
    endpoint_clearance = REWARD_MATH.convex_footprint_signed_clearance_xy(
        left_current, right_current
    )
    swept_clearance = REWARD_MATH.swept_convex_footprint_signed_clearance_xy(
        left_previous,
        right_previous,
        left_current,
        right_current,
        interpolation_steps=8,
    )
    assert endpoint_clearance.item() > 0.10
    assert swept_clearance.item() < 0.0


def test_task_is_isolated_and_optimization_is_conservative():
    env_text = ENV_CFG_FILE.read_text()
    registry = REGISTRY_FILE.read_text()
    agent = AGENT_CFG_FILE.read_text()
    script = TRAIN_SCRIPT.read_text()
    block = env_text[env_text.index("class G1AmpNav2TwoGoalFinetuneEnvCfg") :]

    assert 'id="LeggedLab-Isaac-AMP-G1-Nav2TwoGoalFinetune-v0"' in registry
    assert 'entry_point="legged_lab.envs:ManagerBasedAmpEnv"' in registry
    assert "mode_probability = 0.80" in block
    for disabled in (
        "strict_zero_body_motion_l2 = None",
        "nonzero_single_stance = None",
        "relative_command_response_shortfall_l1 = None",
        "oriented_footprint_proximity_l2 = None",
    ):
        assert disabled in block
    for required in (
        "lateral_command_progress",
        "pure_yaw_command_progress",
        "safe_alternating_touchdown_progress",
        "swept_oriented_footprint_proximity_l2",
    ):
        assert required in block
    assert "load_actor_amp_only = True" in agent
    assert "freeze_actor_hidden_layers = 2" in agent
    assert "freeze_discriminator = True" in agent
    assert "learning_rate = 5.0e-6" in agent
    assert "num_learning_epochs = 2" in agent
    assert "hard_limit = 0.20" in agent
    assert "RSI_ENABLE=False" in script
    assert "RANDOMIZATION_STRENGTH=0" in script
    assert "model_10990" not in script


def test_sweep_argument_validation():
    rectangle = _rect(0.0, 0.0).unsqueeze(0)
    with pytest.raises(ValueError, match="at least one"):
        REWARD_MATH.swept_convex_footprint_signed_clearance_xy(
            rectangle, rectangle, rectangle, rectangle, interpolation_steps=0
        )
