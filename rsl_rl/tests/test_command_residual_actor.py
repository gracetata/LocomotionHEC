import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticCommandResidual


def _observations(batch: int = 4) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(batch, 96),
            "critic": torch.zeros(batch, 297),
        },
        batch_size=[batch],
    )


def test_zero_initialized_residual_is_exact_base_actor():
    obs = _observations()
    obs["policy"][0, 6:9] = torch.tensor([0.0, 0.25, 0.0])
    obs["policy"][1, 6:9] = torch.tensor([0.0, 0.0, 0.35])
    obs["policy"][2, 6:9] = torch.tensor([0.5, 0.0, 0.0])
    model = ActorCriticCommandResidual(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        29,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        command_residual_hidden_dim=8,
    ).eval()
    base = model.actor(obs["policy"])
    torch.testing.assert_close(model.act_inference(obs), base, rtol=0.0, atol=0.0)


def test_residual_gates_never_change_forward_retention_commands():
    obs = _observations()
    obs["policy"][0, 6:9] = torch.tensor([0.0, 0.25, 0.0])
    obs["policy"][1, 6:9] = torch.tensor([0.0, 0.0, 0.35])
    obs["policy"][2, 6:9] = torch.tensor([0.5, 0.0, 0.0])
    obs["policy"][3, 6:9] = torch.tensor([0.2, 0.25, 0.0])
    model = ActorCriticCommandResidual(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        29,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        command_residual_hidden_dim=8,
    ).eval()
    with torch.no_grad():
        model.lateral_command_residual[-1].bias.fill_(1.0)
        model.pure_yaw_command_residual[-1].bias.fill_(2.0)
    base = model.actor(obs["policy"])
    actual = model.act_inference(obs)
    torch.testing.assert_close(actual[0], base[0] + 1.0)
    torch.testing.assert_close(actual[1], base[1] + 2.0)
    torch.testing.assert_close(actual[2:], base[2:], rtol=0.0, atol=0.0)


def test_fixed_bridge_is_exact_carrier_actor_only_for_strict_commands():
    obs = _observations()
    obs["policy"][0, 6:9] = torch.tensor([0.0, 0.25, 0.0])
    obs["policy"][1, 6:9] = torch.tensor([0.0, 0.0, 0.35])
    obs["policy"][2, 6:9] = torch.tensor([0.5, 0.0, 0.0])
    obs["policy"][3, 6:9] = torch.tensor([0.20, 0.25, 0.0])
    model = ActorCriticCommandResidual(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        29,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        command_residual_hidden_dim=8,
        fixed_command_bridge_fraction=1.0,
    ).eval()
    teacher_obs = obs["policy"].clone()
    teacher_obs[0, 6] = 0.20
    teacher_obs[1, 6] = 0.15
    expected = model.actor(teacher_obs)
    actual = model.act_inference(obs)
    torch.testing.assert_close(actual[:2], expected[:2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2:], model.actor(obs["policy"])[2:], rtol=0.0, atol=0.0)


def test_fixed_bridge_supports_sign_specific_pure_yaw_carriers():
    obs = _observations(batch=2)
    obs["policy"][:, 6:9] = torch.tensor([[0.0, 0.0, 0.35], [0.0, 0.0, -0.35]])
    model = ActorCriticCommandResidual(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        29,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        command_residual_hidden_dim=8,
        fixed_command_bridge_fraction=1.0,
        pure_yaw_teacher_forward_command=0.10,
        pure_yaw_positive_teacher_yaw_scale=1.657142857,
        pure_yaw_negative_teacher_yaw_scale=1.428571429,
    ).eval()
    teacher_obs = obs["policy"].clone()
    teacher_obs[:, 6] = 0.10
    teacher_obs[0, 8] = 0.58
    teacher_obs[1, 8] = -0.50
    torch.testing.assert_close(
        model.act_inference(obs),
        model.actor(teacher_obs),
        rtol=0.0,
        atol=1.0e-7,
    )
