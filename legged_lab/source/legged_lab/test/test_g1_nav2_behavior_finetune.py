import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
LEGGED_LAB_ROOT = REPO_ROOT / "legged_lab"
PACKAGE_ROOT = LEGGED_LAB_ROOT / "source" / "legged_lab" / "legged_lab"
AMP_ROOT = PACKAGE_ROOT / "tasks" / "locomotion" / "amp"
G1_CONFIG_ROOT = AMP_ROOT / "config" / "g1"
ENV_CFG_FILE = G1_CONFIG_ROOT / "g1_amp_env_cfg.py"
TASK_REGISTRY_FILE = G1_CONFIG_ROOT / "__init__.py"
AGENT_CFG_FILE = G1_CONFIG_ROOT / "agents" / "rsl_rl_ppo_cfg.py"
REWARD_FILE = AMP_ROOT / "mdp" / "rewards.py"
REWARD_MATH_FILE = AMP_ROOT / "mdp" / "reward_math.py"
MODE_CONFIG_FILE = (
    PACKAGE_ROOT
    / "data"
    / "MotionData"
    / "g1_29dof"
    / "amp"
    / "nav2_behavior_50hz"
    / "task_sampling_config.json"
)
LOW_SPEED_YAW_MODE_CONFIG_FILE = MODE_CONFIG_FILE.with_name(
    "task_sampling_low_speed_yaw_config.json"
)
TRAIN_SCRIPT = LEGGED_LAB_ROOT / "scripts" / "train_g1_amp_nav2_behavior.sh"
MUJOCO_SMOKE_SCRIPT = (
    LEGGED_LAB_ROOT / "scripts" / "test_g1_amp_nav2_behavior_mujoco.sh"
)
AMP_RUNNER_FILE = REPO_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "amp_runner.py"
BASELINE = REPO_ROOT / "checkpoint" / "walk" / "model_10990.pt"
RECORDED_NAV2 = (
    LEGGED_LAB_ROOT
    / "Reference Data"
    / "ArmHack"
    / "WalkPerturbFinetune"
    / "nav2_cmd_vel_raw_success.csv"
)
BASELINE_SIZE = 14_826_139
BASELINE_SHA256 = "1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00"
RECORDED_NAV2_SHA256 = "76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a"


def _load_reward_math():
    spec = importlib.util.spec_from_file_location("g1_nav2_reward_math", REWARD_MATH_FILE)
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


def _rect(
    center_x: float,
    center_y: float,
    yaw: float = 0.0,
    half_length: float = 0.09,
    half_width: float = 0.035,
) -> torch.Tensor:
    corners = torch.tensor(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=torch.float32,
    )
    rotation = torch.tensor(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=torch.float32,
    )
    return corners @ rotation.T + torch.tensor([center_x, center_y])


def test_baseline_and_recorded_nav2_inputs_are_exact():
    assert BASELINE.stat().st_size == BASELINE_SIZE
    assert _sha256(BASELINE) == BASELINE_SHA256
    assert RECORDED_NAV2.is_file()
    assert _sha256(RECORDED_NAV2) == RECORDED_NAV2_SHA256

    checkpoint = torch.load(BASELINE, map_location="cpu", weights_only=False)
    policy = checkpoint["model_state_dict"]
    assert policy["actor.0.weight"].shape[1] == 96
    actor_output_key = max(
        (key for key in policy if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    assert policy[actor_output_key].shape[0] == 29
    assert checkpoint["amp_discriminator_state_dict"]["disc_trunk.0.weight"].shape[1] == 168


def test_task_is_generic_manager_amp_and_has_no_armhack_isolation():
    registry = TASK_REGISTRY_FILE.read_text()
    env_cfg = ENV_CFG_FILE.read_text()
    agent_cfg = AGENT_CFG_FILE.read_text()
    train_script = TRAIN_SCRIPT.read_text()
    task_block = registry[
        registry.index('id="LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0"') :
    ]
    behavior_cfg = env_cfg[
        env_cfg.index("class G1AmpNav2BehaviorFinetuneEnvCfg") :
        env_cfg.index("class G1AmpNav2BehaviorFinetuneEnvCfg_PLAY")
    ]

    assert 'entry_point="legged_lab.envs:ManagerBasedAmpEnv"' in task_block
    assert "G1Nav2BehaviorFinetuneRslRlOnPolicyRunnerAmpCfg" in task_block
    assert "class G1AmpNav2BehaviorFinetuneEnvCfg(G1AmpNav2FinetuneEnvCfg)" in behavior_cfg
    for forbidden in (
        "G1WalkPerturbAmpEnv",
        "upper_body_perturbation",
        "ArmHack",
        "lower_body",
        "disc_joint_mask",
        "action override",
    ):
        assert forbidden not in behavior_cfg
    for forbidden in ("G1WalkPerturbAmpEnv", "ArmHack", "lower_body", "disc_joint_mask"):
        assert forbidden not in train_script
    assert "env.upper_body_perturbation*" in train_script
    assert "G1_LOCOMOTION_JOINT_NAMES" in behavior_cfg
    assert 'experiment_name = "g1_amp_nav2_behavior"' in agent_cfg
    assert 'checkpoint_output_dir = "Nav2BehaviorFinetune"' in agent_cfg
    assert "load_actor_only = True" in agent_cfg


def test_behavior_distribution_separates_zero_micro_and_pure_yaw():
    config = json.loads(MODE_CONFIG_FILE.read_text())
    assert sum(config["mode_weights"].values()) == pytest.approx(1.0)
    assert config["modes"]["stand"] == {
        "lin_vel_x": [0.0, 0.0],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_z": [0.0, 0.0],
    }
    assert config["modes"]["micro_forward"]["lin_vel_x"] == [0.01, 0.15]
    assert config["modes"]["micro_forward"]["lin_vel_y"] == [-0.005, 0.005]
    assert config["modes"]["micro_turn_left"] == {
        "lin_vel_x": [0.0, 0.0],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_z": [0.05, 0.25],
    }
    assert config["modes"]["turn_in_place_right"]["lin_vel_x"] == [0.0, 0.0]
    assert config["modes"]["turn_in_place_right"]["lin_vel_y"] == [0.0, 0.0]

    env_cfg = ENV_CFG_FILE.read_text()
    behavior_cfg = env_cfg[env_cfg.index("class G1AmpNav2BehaviorFinetuneEnvCfg") :]
    assert "mode_probability=0.60" in behavior_cfg
    assert "smoothing_time_constant=0.30" in behavior_cfg
    assert "max_linear_accel=0.60" in behavior_cfg
    assert "max_yaw_accel=0.80" in behavior_cfg
    assert "hard_zero_stand=True" in behavior_cfg


def test_low_speed_yaw_specialization_profile_is_focused_and_balanced():
    config = json.loads(LOW_SPEED_YAW_MODE_CONFIG_FILE.read_text())
    weights = config["mode_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["stand"] == pytest.approx(0.10)
    micro_translation = sum(
        weights[name]
        for name in (
            "micro_forward",
            "micro_backward",
            "micro_lateral_left",
            "micro_lateral_right",
            "micro_diagonal_front_left",
            "micro_diagonal_front_right",
        )
    )
    in_place_yaw = sum(
        weights[name]
        for name in (
            "micro_turn_left",
            "micro_turn_right",
            "turn_in_place_left",
            "turn_in_place_right",
        )
    )
    assert micro_translation == pytest.approx(0.38)
    assert in_place_yaw == pytest.approx(0.52)
    for name in ("micro_turn_left", "micro_turn_right", "turn_in_place_left", "turn_in_place_right"):
        mode = config["modes"][name]
        assert mode["lin_vel_x"] == [0.0, 0.0]
        assert mode["lin_vel_y"] == [0.0, 0.0]


def test_relative_response_has_no_nonzero_deadband():
    command = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.05]],
        dtype=torch.float32,
    )
    penalty = REWARD_MATH.relative_command_response_shortfall_l1(
        command,
        torch.zeros(3, 2),
        torch.zeros(3),
    )
    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.5, 0.5]))

    matched = REWARD_MATH.relative_command_response_shortfall_l1(
        command,
        torch.tensor([[0.0, 0.0], [0.005, 0.0], [0.0, 0.0]]),
        torch.tensor([0.0, 0.0, 0.025]),
    )
    torch.testing.assert_close(matched, torch.zeros(3))


def test_command_conditioned_cadence_formula_reset_and_ema():
    commands = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.25]]
    )
    allowed = REWARD_MATH.allowed_footstep_cadence_hz(commands)
    torch.testing.assert_close(
        allowed, torch.tensor([1.6, 1.6233, 2.998, 1.975]), rtol=1e-5, atol=1e-5
    )

    elapsed = torch.zeros(1)
    ema = torch.zeros(1)
    seen_touchdown = torch.zeros(1, dtype=torch.bool)
    seen_interval = torch.zeros(1, dtype=torch.bool)

    elapsed, ema, seen_touchdown, seen_interval = (
        REWARD_MATH.update_footstep_cadence_state(
            elapsed,
            ema,
            seen_touchdown,
            seen_interval,
            torch.tensor([True]),
            step_dt=0.1,
            ema_alpha=0.5,
        )
    )
    assert seen_touchdown.item() and not seen_interval.item()
    assert ema.item() == 0.0
    for touchdown in (False, True):
        elapsed, ema, seen_touchdown, seen_interval = (
            REWARD_MATH.update_footstep_cadence_state(
                elapsed,
                ema,
                seen_touchdown,
                seen_interval,
                torch.tensor([touchdown]),
                step_dt=0.1,
                ema_alpha=0.5,
            )
        )
    assert seen_interval.item()
    assert ema.item() == pytest.approx(5.0)

    for touchdown in (False, False, True):
        elapsed, ema, seen_touchdown, seen_interval = (
            REWARD_MATH.update_footstep_cadence_state(
                elapsed,
                ema,
                seen_touchdown,
                seen_interval,
                torch.tensor([touchdown]),
                step_dt=0.1,
                ema_alpha=0.5,
            )
        )
    assert ema.item() == pytest.approx((5.0 + 1.0 / 0.3) * 0.5, rel=1e-5)

    # Manager reset semantics are an exact all-zero/false state.
    elapsed.zero_()
    ema.zero_()
    seen_touchdown.zero_()
    seen_interval.zero_()
    assert not torch.any(elapsed) and not torch.any(ema)
    assert not torch.any(seen_touchdown) and not torch.any(seen_interval)


def test_single_stance_reward_is_stronger_for_micro_speed_and_pure_yaw():
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.10, 0.10, 0.05],
            [0.0, 0.0, 0.05],
            [0.40, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scale = REWARD_MATH.nonzero_single_stance_command_scale(commands)
    torch.testing.assert_close(scale, torch.tensor([0.0, 1.5, 1.5, 1.75, 1.0]))

    env_cfg = ENV_CFG_FILE.read_text()
    behavior_cfg = env_cfg[env_cfg.index("class G1AmpNav2BehaviorFinetuneEnvCfg") :]
    assert "weight=1.6" in behavior_cfg
    assert '"base_hz": 1.6' in behavior_cfg
    assert '"linear_gain": 2.33' in behavior_cfg
    assert '"micro_speed_bonus": 0.50' in behavior_cfg
    assert '"pure_yaw_bonus": 0.75' in behavior_cfg


def test_oriented_sole_sat_parallel_rotated_crossing_separated_and_overlap():
    left = torch.stack(
        [
            _rect(0.0, 0.0),
            _rect(0.0, 0.0),
            _rect(0.0, 0.0),
            _rect(0.0, 0.0),
            _rect(0.0, 0.0),
        ]
    )
    right = torch.stack(
        [
            _rect(0.0, 0.20),  # parallel and well separated
            _rect(0.0, 0.095),  # exactly 25 mm sole clearance
            _rect(0.03, 0.02, math.pi / 2),  # rotated/toe crossing
            _rect(0.50, 0.50, math.pi / 4),  # separated and rotated
            _rect(0.0, 0.0),  # complete overlap
        ]
    )
    clearance = REWARD_MATH.convex_footprint_signed_clearance_xy(left, right)
    assert clearance[0].item() == pytest.approx(0.13, abs=1e-6)
    assert clearance[1].item() == pytest.approx(0.025, abs=1e-6)
    assert clearance[2].item() < 0.0
    assert clearance[3].item() > 0.25
    assert clearance[4].item() < 0.0


def test_amp_actor_only_load_keeps_critic_noise_and_iteration_fresh(tmp_path):
    checkpoint_path = tmp_path / "source.pt"
    code = textwrap.dedent(
        f"""
        import sys
        from types import SimpleNamespace
        import torch

        sys.path.insert(0, {str(REPO_ROOT / "rsl_rl")!r})
        from rsl_rl.runners.amp_runner import AMPRunner

        class TinyActorCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Sequential(
                    torch.nn.Linear(3, 4), torch.nn.ELU(), torch.nn.Linear(4, 2)
                )
                self.critic = torch.nn.Sequential(torch.nn.Linear(5, 1))
                self.std = torch.nn.Parameter(torch.ones(2))

        source = TinyActorCritic()
        target = TinyActorCritic()
        with torch.no_grad():
            for parameter in source.actor.parameters():
                parameter.fill_(7.0)
            for parameter in source.critic.parameters():
                parameter.fill_(9.0)
            source.std.fill_(11.0)
        critic_before = {{
            key: value.clone() for key, value in target.critic.state_dict().items()
        }}
        std_before = target.std.detach().clone()
        torch.save(
            {{
                "model_state_dict": source.state_dict(),
                "iter": 10990,
                "amp_discriminator_state_dict": {{"sentinel": torch.tensor(1.0)}},
            }},
            {str(checkpoint_path)!r},
        )
        runner = AMPRunner.__new__(AMPRunner)
        runner.cfg = {{"load_actor_only": True, "load_policy_only": False}}
        ppo_optimizer = object()
        discriminator_optimizer = object()
        amp_state = torch.tensor([3.0])
        normalizer_state = torch.tensor([4.0])
        runner.alg = SimpleNamespace(
            policy=target,
            optimizer=ppo_optimizer,
            disc_optimizer=discriminator_optimizer,
            amp_discriminator=SimpleNamespace(
                state=amp_state,
                disc_obs_normalizer=SimpleNamespace(state=normalizer_state),
            ),
        )
        runner.current_learning_iteration = 123
        runner.load({str(checkpoint_path)!r}, map_location="cpu")
        for parameter in target.actor.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 7.0))
        for key, value in target.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key])
        torch.testing.assert_close(target.std, std_before)
        assert runner.alg.optimizer is ppo_optimizer
        assert runner.alg.disc_optimizer is discriminator_optimizer
        torch.testing.assert_close(runner.alg.amp_discriminator.state, torch.tensor([3.0]))
        torch.testing.assert_close(
            runner.alg.amp_discriminator.disc_obs_normalizer.state, torch.tensor([4.0])
        )
        assert runner.current_learning_iteration == 0
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_wrapper_protects_model_10990_and_dedicated_output():
    text = TRAIN_SCRIPT.read_text()
    mujoco_text = MUJOCO_SMOKE_SCRIPT.read_text()
    runner_text = AMP_RUNNER_FILE.read_text()
    assert BASELINE_SHA256 in text
    assert "EXPECTED_BASE_SIZE=14826139" in text
    assert "trap verify_on_exit EXIT" in text
    assert 'OUTPUT_DIR="${LEGGED_LAB_DIR}/Nav2BehaviorFinetune"' in text
    assert "checkpoint/walk" in text
    assert "agent.load_actor_only=True" in text
    assert "actor-only" in text
    assert 'load_actor_only = bool(self.cfg.get("load_actor_only", False))' in runner_text
    assert 'key.startswith("actor.")' in runner_text
    assert "sim2sim_g1_amp_mujoco.sh" in mujoco_text
    assert "96->29" in mujoco_text
    assert "healthy" in mujoco_text
    assert "armhack" not in mujoco_text.lower()
