from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from .actor_critic import ActorCritic


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported command-residual activation: {name}")


class ActorCriticCommandResidual(ActorCritic):
    """Baseline actor plus disjoint strict-lateral and pure-yaw residuals.

    Both residual output layers start at exactly zero.  Commands outside the
    two strict specialization families are hard-gated to the unchanged base
    actor, which makes retention structural instead of relying only on KL.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        command_residual_hidden_dim: int = 64,
        command_obs_start_index: int = 6,
        lateral_min_command: float = 0.10,
        pure_yaw_min_command: float = 0.10,
        max_lateral_forward_command: float = 0.02,
        max_lateral_yaw_command: float = 0.05,
        max_pure_yaw_translation_command: float = 0.02,
        fixed_command_bridge_fraction: float = 0.0,
        lateral_teacher_forward_command: float = 0.20,
        lateral_teacher_min_abs_command: float = 0.0,
        pure_yaw_teacher_forward_command: float = 0.15,
        pure_yaw_positive_teacher_yaw_scale: float = 1.0,
        pure_yaw_negative_teacher_yaw_scale: float = 1.0,
        pure_yaw_positive_teacher_yaw_min: float = 0.0,
        pure_yaw_positive_teacher_yaw_max: float = 10.0,
        pure_yaw_negative_teacher_yaw_min: float = 0.0,
        pure_yaw_negative_teacher_yaw_max: float = 10.0,
        activation: str = "elu",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if state_dependent_std:
            raise NotImplementedError("Command residual actor currently requires scalar action noise.")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            activation=activation,
            state_dependent_std=False,
            **kwargs,
        )
        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        hidden_dim = int(command_residual_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("command_residual_hidden_dim must be positive.")
        self.command_obs_start_index = int(command_obs_start_index)
        self.lateral_min_command = float(lateral_min_command)
        self.pure_yaw_min_command = float(pure_yaw_min_command)
        self.max_lateral_forward_command = float(max_lateral_forward_command)
        self.max_lateral_yaw_command = float(max_lateral_yaw_command)
        self.max_pure_yaw_translation_command = float(max_pure_yaw_translation_command)
        if not 0.0 <= float(fixed_command_bridge_fraction) <= 1.0:
            raise ValueError("fixed_command_bridge_fraction must be in [0, 1].")
        if float(lateral_teacher_forward_command) <= self.max_lateral_forward_command:
            raise ValueError("lateral teacher command must exceed the strict forward band.")
        if float(pure_yaw_teacher_forward_command) <= self.max_pure_yaw_translation_command:
            raise ValueError("pure-yaw teacher command must exceed the strict translation band.")
        if float(pure_yaw_positive_teacher_yaw_scale) <= 0.0:
            raise ValueError("positive pure-yaw teacher scale must be positive.")
        if float(pure_yaw_negative_teacher_yaw_scale) <= 0.0:
            raise ValueError("negative pure-yaw teacher scale must be positive.")
        if float(lateral_teacher_min_abs_command) < 0.0:
            raise ValueError("lateral teacher minimum magnitude must be non-negative.")
        if not 0.0 <= float(pure_yaw_positive_teacher_yaw_min) <= float(
            pure_yaw_positive_teacher_yaw_max
        ):
            raise ValueError("invalid positive pure-yaw teacher magnitude envelope.")
        if not 0.0 <= float(pure_yaw_negative_teacher_yaw_min) <= float(
            pure_yaw_negative_teacher_yaw_max
        ):
            raise ValueError("invalid negative pure-yaw teacher magnitude envelope.")
        # Buffers make the deployed analytical bridge self-describing in a
        # checkpoint without adding trainable parameters.
        self.register_buffer(
            "fixed_command_bridge_fraction",
            torch.tensor(float(fixed_command_bridge_fraction)),
        )
        self.register_buffer(
            "lateral_teacher_forward_command",
            torch.tensor(float(lateral_teacher_forward_command)),
        )
        self.register_buffer(
            "lateral_teacher_min_abs_command",
            torch.tensor(float(lateral_teacher_min_abs_command)),
        )
        self.register_buffer(
            "pure_yaw_teacher_forward_command",
            torch.tensor(float(pure_yaw_teacher_forward_command)),
        )
        self.register_buffer(
            "pure_yaw_positive_teacher_yaw_scale",
            torch.tensor(float(pure_yaw_positive_teacher_yaw_scale)),
        )
        self.register_buffer(
            "pure_yaw_negative_teacher_yaw_scale",
            torch.tensor(float(pure_yaw_negative_teacher_yaw_scale)),
        )
        self.register_buffer(
            "pure_yaw_positive_teacher_yaw_min",
            torch.tensor(float(pure_yaw_positive_teacher_yaw_min)),
        )
        self.register_buffer(
            "pure_yaw_positive_teacher_yaw_max",
            torch.tensor(float(pure_yaw_positive_teacher_yaw_max)),
        )
        self.register_buffer(
            "pure_yaw_negative_teacher_yaw_min",
            torch.tensor(float(pure_yaw_negative_teacher_yaw_min)),
        )
        self.register_buffer(
            "pure_yaw_negative_teacher_yaw_max",
            torch.tensor(float(pure_yaw_negative_teacher_yaw_max)),
        )

        def make_residual() -> nn.Sequential:
            module = nn.Sequential(
                nn.Linear(num_actor_obs, hidden_dim),
                _activation(activation),
                nn.Linear(hidden_dim, num_actions),
            )
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
            return module

        self.lateral_command_residual = make_residual()
        self.pure_yaw_command_residual = make_residual()
        print(
            "Command residual actor: strict lateral and pure-yaw adapters "
            f"({num_actor_obs}->{hidden_dim}->{num_actions}), zero initialized."
        )

    def command_masks(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.command_obs_start_index
        command = actor_obs[..., start : start + 3]
        lateral = (
            (torch.abs(command[..., 1]) >= self.lateral_min_command)
            & (torch.abs(command[..., 0]) <= self.max_lateral_forward_command)
            & (torch.abs(command[..., 2]) <= self.max_lateral_yaw_command)
        )
        pure_yaw = (
            (torch.linalg.vector_norm(command[..., :2], dim=-1)
             <= self.max_pure_yaw_translation_command)
            & (torch.abs(command[..., 2]) >= self.pure_yaw_min_command)
        )
        return lateral, pure_yaw

    def actor_mean_from_flat_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        normalized_obs = self.actor_obs_normalizer(actor_obs)
        mean = self.actor(normalized_obs)
        lateral, pure_yaw = self.command_masks(actor_obs)
        teacher_obs = actor_obs.clone()
        command_x = teacher_obs[..., self.command_obs_start_index]
        command_x = torch.where(
            lateral,
            self.lateral_teacher_forward_command.to(command_x.dtype),
            command_x,
        )
        command_x = torch.where(
            pure_yaw,
            self.pure_yaw_teacher_forward_command.to(command_x.dtype),
            command_x,
        )
        teacher_obs[..., self.command_obs_start_index] = command_x
        command_y = teacher_obs[..., self.command_obs_start_index + 1]
        teacher_lateral_magnitude = torch.maximum(
            torch.abs(command_y),
            self.lateral_teacher_min_abs_command.to(command_y.dtype),
        )
        teacher_obs[..., self.command_obs_start_index + 1] = torch.where(
            lateral,
            torch.copysign(teacher_lateral_magnitude, command_y),
            command_y,
        )
        command_yaw = teacher_obs[..., self.command_obs_start_index + 2]
        yaw_scale = torch.where(
            command_yaw >= 0.0,
            self.pure_yaw_positive_teacher_yaw_scale.to(command_yaw.dtype),
            self.pure_yaw_negative_teacher_yaw_scale.to(command_yaw.dtype),
        )
        yaw_min = torch.where(
            command_yaw >= 0.0,
            self.pure_yaw_positive_teacher_yaw_min.to(command_yaw.dtype),
            self.pure_yaw_negative_teacher_yaw_min.to(command_yaw.dtype),
        )
        yaw_max = torch.where(
            command_yaw >= 0.0,
            self.pure_yaw_positive_teacher_yaw_max.to(command_yaw.dtype),
            self.pure_yaw_negative_teacher_yaw_max.to(command_yaw.dtype),
        )
        teacher_yaw = torch.copysign(torch.clamp(torch.abs(command_yaw) * yaw_scale, yaw_min, yaw_max), command_yaw)
        teacher_obs[..., self.command_obs_start_index + 2] = torch.where(
            pure_yaw,
            teacher_yaw,
            command_yaw,
        )
        teacher_mean = self.actor(self.actor_obs_normalizer(teacher_obs))
        bridge_mask = (lateral | pure_yaw).unsqueeze(-1).to(mean.dtype)
        mean = mean + bridge_mask * self.fixed_command_bridge_fraction.to(mean.dtype) * (
            teacher_mean - mean
        )
        mean = mean + lateral.unsqueeze(-1).to(mean.dtype) * self.lateral_command_residual(normalized_obs)
        mean = mean + pure_yaw.unsqueeze(-1).to(mean.dtype) * self.pure_yaw_command_residual(normalized_obs)
        return mean

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        mean = self.actor_mean_from_flat_obs(actor_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_mean_from_flat_obs(self.get_actor_obs(obs))
