import hashlib
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "checkpoint" / "nav2_behavior_model9996_source" / "model_9996.pt"
TRAIN_SCRIPT = (
    REPO_ROOT / "legged_lab" / "scripts" / "train_g1_amp_nav2_behavior_from_model9996.sh"
)
SOURCE_SIZE = 16_202_421
SOURCE_SHA256 = "bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_model9996_is_exact_finite_full_state_checkpoint():
    assert SOURCE.stat().st_size == SOURCE_SIZE
    assert _sha256(SOURCE) == SOURCE_SHA256
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    assert checkpoint["iter"] == 9996

    model = checkpoint["model_state_dict"]
    discriminator = checkpoint["amp_discriminator_state_dict"]
    actor_weights = [
        value
        for name, value in model.items()
        if name.startswith("actor.") and name.endswith(".weight")
    ]
    critic_weights = [
        value
        for name, value in model.items()
        if name.startswith("critic.") and name.endswith(".weight")
    ]
    discriminator_weights = [
        value for name, value in discriminator.items() if name.endswith("weight")
    ]
    assert actor_weights[0].shape[1] == 96
    assert actor_weights[-1].shape[0] == 29
    assert critic_weights[0].shape[1] == 297
    assert critic_weights[-1].shape[0] == 1
    assert discriminator_weights[0].shape[1] == 280
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["amp_discriminator_optimizer_state_dict"]

    def tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)

    assert all(torch.isfinite(value).all() for value in tensors(checkpoint))


def test_model9996_training_is_full_state_and_has_no_old_baseline_reference():
    text = TRAIN_SCRIPT.read_text()
    assert SOURCE_SHA256 in text
    assert "SOURCE_SIZE=16202421" in text
    assert "trap verify_on_exit EXIT" in text
    assert "agent.load_actor_only=False" in text
    assert "agent.load_policy_only=False" in text
    assert "agent.reset_amp_on_load=False" in text
    assert 'BASELINE_KL_CHECKPOINT="${SOURCE_CHECKPOINT}"' in text
    assert "full actor/critic/PPO/AMP/normalizer/optimizers" in text
    assert "10990" not in text
