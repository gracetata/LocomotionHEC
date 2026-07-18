"""Static regression checks for the model_3999 Walk deployment bundle."""

from __future__ import annotations

import ast
import hashlib
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
BUNDLE_DIR = LEGGED_LAB_DIR / "deployment" / "armhack_walk" / "model_3999"
MANIFEST = BUNDLE_DIR / "manifest.json"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def test_nav2_command_contract_and_space_toggle_are_explicit():
    contract = json.loads(_text(CONTRACT))
    command = contract["recommended_fixed_command"]
    bounds = contract["raw_command_component_bounds"]
    assert command == [0.35, 0.0, 0.0]
    for index, value in enumerate(command):
        assert bounds["min"][index] <= value <= bounds["max"][index]

    adapter_text = _text(MUJOCO_ADAPTER)
    real_text = _text(REAL_RUNNER)
    assert "self.command_active = not self.command_active" in adapter_text
    assert "self.command_active = False" in real_text
    assert "self.command_active = not self.command_active" in real_text
    assert "fixed_command if self.command_active else np.zeros" in real_text


def test_fixed_arm_action_is_used_for_first_and_following_observations():
    runner_text = _text(MUJOCO_RUNNER)
    real_text = _text(REAL_RUNNER)
    assert "action = armhack_walk.compose_action(action)" in runner_text
    assert "next_action = armhack_walk.compose_action(next_action)" in runner_text
    assert "controller.action_policy = (startup_policy - config.default_angles) / config.action_scale" in real_text
    assert "self.obs[67:96] = self.action_policy" in real_text
    assert "self.action_policy = (target_policy - config.default_angles) / config.action_scale" in real_text


def test_launchers_pin_model3999_and_real_mode_requires_explicit_confirmation():
    mujoco_text = _text(MUJOCO_LAUNCHER)
    real_text = _text(REAL_LAUNCHER)
    assert "model_3999.pt" in mujoco_text
    assert "walk_model3999.onnx" in mujoco_text
    assert "G1_AMP_ARMHACK_WALK_ENABLE" in mujoco_text
    assert "walk_model3999.onnx" in real_text
    assert "CONFIRM_REAL_ROBOT=I_UNDERSTAND" in real_text
    assert 'if is_true "${DRY_RUN}"' in real_text
    assert "--self-test" in real_text
