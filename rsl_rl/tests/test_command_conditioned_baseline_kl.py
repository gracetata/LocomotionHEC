import torch

from rsl_rl.algorithms.ppo_amp import (
    build_two_goal_carrier_teacher_obs,
    command_conditioned_lerp_reward,
    two_goal_specialization_mask_from_policy_obs,
)


def test_carrier_teacher_changes_only_strict_two_goal_command_slice():
    obs = torch.zeros(5, 96)
    obs[0, 6:9] = torch.tensor([0.0, 0.30, 0.0])
    obs[1, 6:9] = torch.tensor([0.0, -0.30, 0.0])
    obs[2, 6:9] = torch.tensor([0.0, 0.0, 0.40])
    obs[3, 6:9] = torch.tensor([0.0, 0.0, -0.40])
    obs[4, 6:9] = torch.tensor([0.40, 0.0, 0.0])
    teacher, lateral, pure_yaw = build_two_goal_carrier_teacher_obs(obs)
    torch.testing.assert_close(lateral, torch.tensor([True, True, False, False, False]))
    torch.testing.assert_close(pure_yaw, torch.tensor([False, False, True, True, False]))
    torch.testing.assert_close(teacher[:2, 6], torch.full((2,), 0.20))
    torch.testing.assert_close(teacher[2:4, 6], torch.full((2,), 0.15))
    torch.testing.assert_close(teacher[:, 7:9], obs[:, 7:9])
    torch.testing.assert_close(teacher[4], obs[4])


def test_command_conditioned_mask_separates_specialization_and_retention():
    policy_obs = torch.zeros(5, 96)
    policy_obs[0, 6:9] = torch.tensor([0.0, 0.30, 0.0])
    policy_obs[1, 6:9] = torch.tensor([0.0, -0.30, 0.0])
    policy_obs[2, 6:9] = torch.tensor([0.0, 0.0, 0.40])
    policy_obs[3, 6:9] = torch.tensor([0.35, 0.10, 0.20])
    policy_obs[4, 6:9] = torch.tensor([0.0, 0.0, 0.0])

    mask = two_goal_specialization_mask_from_policy_obs(policy_obs)
    torch.testing.assert_close(mask, torch.tensor([True, True, True, False, False]))

    carrier_obs = torch.zeros(2, 96)
    carrier_obs[0, 6:9] = torch.tensor([0.20, 0.30, 0.0])
    carrier_obs[1, 6:9] = torch.tensor([0.15, 0.0, 0.40])
    torch.testing.assert_close(
        two_goal_specialization_mask_from_policy_obs(carrier_obs), torch.tensor([True, True])
    )


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


def test_command_conditioned_style_is_disabled_only_for_specialization():
    task = torch.tensor([2.0, 2.0])
    style = torch.tensor([10.0, 10.0])
    specialization = torch.tensor([True, False])
    mixed = command_conditioned_lerp_reward(
        task,
        style,
        specialization,
        retention_task_lerp=0.85,
        specialization_task_lerp=1.0,
    )
    torch.testing.assert_close(mixed, torch.tensor([2.0, 3.2]))
