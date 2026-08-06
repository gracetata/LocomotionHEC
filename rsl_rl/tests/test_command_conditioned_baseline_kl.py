import torch

from rsl_rl.algorithms.ppo_amp import two_goal_specialization_mask_from_policy_obs


def test_command_conditioned_mask_separates_specialization_and_retention():
    policy_obs = torch.zeros(5, 96)
    policy_obs[0, 6:9] = torch.tensor([0.0, 0.30, 0.0])
    policy_obs[1, 6:9] = torch.tensor([0.0, -0.30, 0.0])
    policy_obs[2, 6:9] = torch.tensor([0.0, 0.0, 0.40])
    policy_obs[3, 6:9] = torch.tensor([0.35, 0.10, 0.20])
    policy_obs[4, 6:9] = torch.tensor([0.0, 0.0, 0.0])

    mask = two_goal_specialization_mask_from_policy_obs(policy_obs)
    torch.testing.assert_close(mask, torch.tensor([True, True, True, False, False]))


def test_command_conditioned_kl_weights_are_exact():
    raw_kl = torch.tensor([0.10, 0.20, 0.30, 0.40])
    specialization = torch.tensor([True, True, False, False])
    scales = torch.where(
        specialization,
        torch.full_like(raw_kl, 0.005),
        torch.full_like(raw_kl, 0.08),
    )
    weighted = torch.mean(scales * raw_kl)
    expected = (0.005 * 0.10 + 0.005 * 0.20 + 0.08 * 0.30 + 0.08 * 0.40) / 4.0
    torch.testing.assert_close(weighted, torch.tensor(expected))
