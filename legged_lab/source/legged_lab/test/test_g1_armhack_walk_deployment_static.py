"""Static regression checks for the model_3999 Walk deployment bundle."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


LEGGED_LAB_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = LEGGED_LAB_DIR.parent
MUJOCO_ADAPTER = REPO_DIR / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "armhack_walk.py"
MUJOCO_RUNNER = REPO_DIR / "unitree_sim2sim2real" / "deploy" / "deploy_mujoco" / "deploy_mujoco_g1_amp.py"
REAL_RUNNER = REPO_DIR / "unitree_sim2sim2real" / "deploy" / "deploy_real" / "deploy_real_g1_armhack_walk.py"
MUJOCO_LAUNCHER = LEGGED_LAB_DIR / "scripts" / "val_mujoco_g1_armhack_walk.sh"
REAL_LAUNCHER = REPO_DIR / "scripts" / "deploy_real_g1_armhack_walk.sh"
CONTRACT = LEGGED_LAB_DIR / "Reference Data" / "ArmHack" / "WalkPerturbFinetune" / "real_deployment_contract.json"
POSES = LEGGED_LAB_DIR / "Reference Data" / "ArmHack" / "WalkPerturbFinetune" / "g1_arm_pose_set.json"
BUNDLE_DIR = LEGGED_LAB_DIR / "deployment" / "armhack_walk" / "model_3999"
MANIFEST = BUNDLE_DIR / "manifest.json"
USE_MODEL = REPO_DIR / "use" / "armhack_walk_model_10990.onnx"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_real_runner_module():
    spec = importlib.util.spec_from_file_location("armhack_walk_real_runner_test", REAL_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_walk_deployment_python_files_parse_and_stand_is_not_referenced():
    for path in (MUJOCO_ADAPTER, REAL_RUNNER):
        source = _text(path)
        ast.parse(source, filename=str(path))
        assert "armhack_stand" not in source.lower()


def test_bundle_hashes_and_actor_interface_are_locked():
    manifest = json.loads(_text(MANIFEST))
    assert manifest["source_checkpoint_sha256"] == "454c9bc0b5e38b2a9800c6faaa9e8ba6995f7d99bd3844155929a10a4fb8e2ff"
    assert manifest["interface"]["input"] == "obs float32 [1,96]"
    assert manifest["interface"]["output"] == "actions float32 [1,29]"

    for name_key, hash_key in (
        ("onnx", "onnx_sha256"),
        ("torchscript", "torchscript_sha256"),
        ("metadata", "metadata_sha256"),
    ):
        artifact = BUNDLE_DIR / manifest[name_key]
        assert artifact.is_file()
        assert _sha256(artifact) == manifest[hash_key]


def test_nav2_command_contract_and_real_controls_are_explicit():
    contract = json.loads(_text(CONTRACT))
    command = contract["recommended_fixed_command"]
    bounds = contract["raw_command_component_bounds"]
    assert command == [0.35, 0.0, 0.0]
    assert contract["command_switch"]["key"] == "V"
    assert contract["command_switch"]["mode"] == "fixed_only"
    assert contract["arm_pose_switch"]["key"] == "SPACE"
    assert contract["arm_pose_switch"]["order"] == ["pos1_back", "pos2_down", "pos3_front"]
    assert contract["joystick"]["ranges"]["lin_vel_x"] == [-0.2, 0.6]
    for index, value in enumerate(command):
        assert bounds["min"][index] <= value <= bounds["max"][index]

    adapter_text = _text(MUJOCO_ADAPTER)
    real_text = _text(REAL_RUNNER)
    assert "self.command_active = not self.command_active" in adapter_text
    assert "self.command_active = False" in real_text
    assert "self.command_active = not self.command_active" in real_text
    assert "fixed_command if self.command_active else np.zeros" in real_text
    assert 'if key.lower() == "v" and config.command_mode == "fixed"' in real_text
    assert 'if key == " ":' in real_text
    assert "self._cycle_arm_pose(now)" in real_text


def test_real_joystick_pose_cycle_and_direct_targets_are_locked():
    real_text = _text(REAL_RUNNER)
    launcher_text = _text(REAL_LAUNCHER)

    assert 'config.command_mode not in {"fixed", "joystick"}' in real_text
    assert "self._target_command_physical()" in real_text
    assert "load_walk_pose_set" in real_text
    assert "_minimum_jerk(alpha)" in real_text
    assert "np.clip(target, self.lower, self.upper)" in real_text
    assert "previous_target" not in real_text
    assert "max_delta" not in real_text
    assert "JOINT_LIMIT_MARGIN_RAD" not in real_text
    assert "MAX_TARGET_SPEED_RAD_S" not in real_text

    assert 'COMMAND_MODE=${COMMAND_MODE:-fixed}' in launcher_text
    assert 'G1_AMP_COMMAND_MODE="${COMMAND_MODE}"' in launcher_text
    assert "G1_AMP_JOYSTICK_DEVICE" in launcher_text
    assert "G1_AMP_JOYSTICK_LIN_X_RANGE" in launcher_text
    assert "G1_ARMHACK_WALK_ARM_POSE_SWITCH_S" in launcher_text
    assert "JOINT_LIMIT_MARGIN_RAD" not in launcher_text
    assert "MAX_TARGET_SPEED_RAD_S" not in launcher_text


def test_real_pose_set_and_joystick_contract_validation_execute():
    runner = _load_real_runner_module()
    names, values = runner.load_walk_pose_set(POSES)
    assert names == ("pos1_back", "pos2_down", "pos3_front")
    assert values.shape == (3, 14)
    assert runner._minimum_jerk(0.0) == 0.0
    assert runner._minimum_jerk(1.0) == 1.0

    contract = runner.load_contract(CONTRACT)
    runner.validate_joystick_ranges(
        [[-0.2, 0.6], [-0.3, 0.3], [-0.5187280216217041, 0.6]],
        contract,
    )


def test_fixed_arm_action_is_used_for_first_and_following_observations():
    runner_text = _text(MUJOCO_RUNNER)
    real_text = _text(REAL_RUNNER)
    assert "action = armhack_walk.compose_action(action)" in runner_text
    assert "next_action = armhack_walk.compose_action(next_action)" in runner_text
    assert "controller.action_policy = (startup_policy - config.default_angles) / config.action_scale" in real_text
    assert "self.obs[67:96] = self.action_policy" in real_text
    assert "self.action_policy = (target_policy - config.default_angles) / config.action_scale" in real_text


def test_mujoco_keeps_model3999_regression_and_real_uses_packaged_model10990():
    mujoco_text = _text(MUJOCO_LAUNCHER)
    real_text = _text(REAL_LAUNCHER)
    assert "model_3999.pt" in mujoco_text
    assert "walk_model3999.onnx" in mujoco_text
    assert "G1_AMP_ARMHACK_WALK_ENABLE" in mujoco_text
    assert "use/armhack_walk_model_10990.onnx" in real_text
    assert USE_MODEL.is_file()
    assert _sha256(USE_MODEL) == "b052c3b0583834a742ea59e736d55c3c9bafabb75f1d4fae65980166d4a895aa"
    assert "CONFIRM_REAL_ROBOT=I_UNDERSTAND" in real_text
    assert 'if is_true "${DRY_RUN}"' in real_text
    assert "--self-test" in real_text
