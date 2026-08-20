from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent, AMPDiscriminator
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import string_to_callable
from rsl_rl.algorithms import PPO
from rsl_rl.modules.amp import LossType


def _mirror_g1_joint_tensor(values: torch.Tensor, apply_sign: bool = True) -> torch.Tensor:
    """Mirror batched G1 29-DoF joint vectors using the deployment convention."""
    if values.shape[-1] != 29:
        raise ValueError(f"Expected a 29-D G1 joint vector, got {values.shape[-1]} dimensions.")
    mirrored = torch.empty_like(values)
    left = [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    right = [1, 4, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    center = [2, 5, 8]
    mirrored[..., left] = values[..., right]
    mirrored[..., right] = values[..., left]
    mirrored[..., center] = values[..., center]
    if apply_sign:
        mirrored[..., [3, 4, 15, 16, 17, 18, 23, 24]] *= -1.0
        mirrored[..., [6, 7, 19, 20, 27, 28]] *= -1.0
        mirrored[..., [2, 5]] *= -1.0
    return mirrored


def _mirror_g1_policy_observation_tensor(observations: torch.Tensor) -> torch.Tensor:
    """Mirror batched 96-D G1 policy observations left-to-right."""
    if observations.shape[-1] != 96:
        raise ValueError(f"Expected a 96-D G1 observation, got {observations.shape[-1]} dimensions.")
    mirrored = observations.clone()
    mirrored[..., 0:3] *= observations.new_tensor([-1.0, 1.0, -1.0])
    mirrored[..., 3:6] *= observations.new_tensor([1.0, -1.0, 1.0])
    mirrored[..., 6:9] *= observations.new_tensor([1.0, -1.0, -1.0])
    mirrored[..., 9:38] = _mirror_g1_joint_tensor(observations[..., 9:38])
    mirrored[..., 38:67] = _mirror_g1_joint_tensor(observations[..., 38:67])
    mirrored[..., 67:96] = _mirror_g1_joint_tensor(observations[..., 67:96])
    return mirrored


def two_goal_specialization_mask_from_policy_obs(
    policy_obs: torch.Tensor,
    *,
    command_obs_start_index: int = 6,
    lateral_min_command: float = 0.10,
    pure_yaw_min_command: float = 0.10,
    max_forward_command: float = 0.25,
    max_lateral_yaw_command: float = 0.05,
    max_pure_yaw_translation_command: float = 0.25,
) -> torch.Tensor:
    """Classify pure-lateral and pure-yaw samples without changing policy I/O."""
    if policy_obs.ndim != 2:
        raise ValueError("Flattened policy observations must have shape [batch, features].")
    start = int(command_obs_start_index)
    if start < 0 or start + 3 > policy_obs.shape[1]:
        raise ValueError("Velocity-command observation slice is outside the policy input.")
    command = policy_obs[:, start : start + 3]
    lateral = (
        (torch.abs(command[:, 1]) >= float(lateral_min_command))
        & (torch.abs(command[:, 0]) <= float(max_forward_command))
        & (torch.abs(command[:, 2]) <= float(max_lateral_yaw_command))
    )
    pure_yaw = (
        (torch.linalg.vector_norm(command[:, :2], dim=1) <= float(max_pure_yaw_translation_command))
        & (torch.abs(command[:, 2]) >= float(pure_yaw_min_command))
    )
    return lateral | pure_yaw


def build_two_goal_carrier_teacher_obs(
    policy_obs: torch.Tensor,
    *,
    command_obs_start_index: int = 6,
    lateral_min_command: float = 0.10,
    pure_yaw_min_command: float = 0.10,
    max_student_forward_command: float = 0.02,
    max_lateral_yaw_command: float = 0.05,
    max_student_pure_yaw_translation_command: float = 0.02,
    lateral_teacher_forward_command: float = 0.20,
    lateral_teacher_min_abs_command: float = 0.0,
    lateral_teacher_opposite_yaw_abs: float = 0.0,
    pure_yaw_teacher_forward_command: float = 0.15,
    pure_yaw_positive_teacher_yaw_scale: float = 1.0,
    pure_yaw_negative_teacher_yaw_scale: float = 1.0,
    pure_yaw_positive_teacher_yaw_min: float = 0.0,
    pure_yaw_positive_teacher_yaw_max: float = 10.0,
    pure_yaw_negative_teacher_yaw_min: float = 0.0,
    pure_yaw_negative_teacher_yaw_max: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair strict commands with baseline commands known to start the gait.

    Only the command slice is changed.  The returned lateral and pure-yaw
    masks are disjoint and let the caller apply an auxiliary loss exclusively
    to the two new skills, never to Nav2 retention samples.
    """
    if policy_obs.ndim != 2:
        raise ValueError("Flattened policy observations must have shape [batch, features].")
    start = int(command_obs_start_index)
    if start < 0 or start + 3 > policy_obs.shape[1]:
        raise ValueError("Velocity-command observation slice is outside the policy input.")
    if abs(lateral_teacher_forward_command) <= max_student_forward_command:
        raise ValueError("Lateral teacher command magnitude must exceed the student dead-zone band.")
    if pure_yaw_positive_teacher_yaw_scale <= 0.0 or pure_yaw_negative_teacher_yaw_scale <= 0.0:
        raise ValueError("Pure-yaw teacher yaw scales must be positive.")
    if lateral_teacher_opposite_yaw_abs < 0.0:
        raise ValueError("Lateral teacher opposite-yaw magnitude must be non-negative.")

    command = policy_obs[:, start : start + 3]
    lateral = (
        (torch.abs(command[:, 1]) >= float(lateral_min_command))
        & (torch.abs(command[:, 0]) <= float(max_student_forward_command))
        & (torch.abs(command[:, 2]) <= float(max_lateral_yaw_command))
    )
    pure_yaw = (
        (torch.linalg.vector_norm(command[:, :2], dim=1)
         <= float(max_student_pure_yaw_translation_command))
        & (torch.abs(command[:, 2]) >= float(pure_yaw_min_command))
    )
    teacher_obs = policy_obs.detach().clone()
    teacher_obs[lateral, start] = float(lateral_teacher_forward_command)
    lateral_magnitude = torch.maximum(
        torch.abs(teacher_obs[:, start + 1]),
        torch.tensor(float(lateral_teacher_min_abs_command), device=teacher_obs.device),
    )
    teacher_obs[lateral, start + 1] = torch.where(
        teacher_obs[lateral, start + 1] >= 0.0,
        lateral_magnitude[lateral],
        -lateral_magnitude[lateral],
    )
    lateral_teacher_yaw = torch.where(
        teacher_obs[:, start + 1] >= 0.0,
        -float(lateral_teacher_opposite_yaw_abs),
        float(lateral_teacher_opposite_yaw_abs),
    )
    teacher_obs[lateral, start + 2] = lateral_teacher_yaw[lateral]
    teacher_obs[pure_yaw, start] = float(pure_yaw_teacher_forward_command)
    yaw_scale = torch.where(
        teacher_obs[:, start + 2] >= 0.0,
        float(pure_yaw_positive_teacher_yaw_scale),
        float(pure_yaw_negative_teacher_yaw_scale),
    )
    yaw_min = torch.where(
        teacher_obs[:, start + 2] >= 0.0,
        float(pure_yaw_positive_teacher_yaw_min),
        float(pure_yaw_negative_teacher_yaw_min),
    )
    yaw_max = torch.where(
        teacher_obs[:, start + 2] >= 0.0,
        float(pure_yaw_positive_teacher_yaw_max),
        float(pure_yaw_negative_teacher_yaw_max),
    )
    bounded_yaw_magnitude = torch.clamp(
        torch.abs(teacher_obs[:, start + 2]) * yaw_scale,
        yaw_min,
        yaw_max,
    )
    bounded_yaw = torch.where(
        teacher_obs[:, start + 2] >= 0.0,
        bounded_yaw_magnitude,
        -bounded_yaw_magnitude,
    )
    teacher_obs[pure_yaw, start + 2] = bounded_yaw[pure_yaw]
    return teacher_obs, lateral, pure_yaw


def command_conditioned_lerp_reward(
    task_reward: torch.Tensor,
    style_reward: torch.Tensor,
    specialization_mask: torch.Tensor,
    *,
    retention_task_lerp: float,
    specialization_task_lerp: float,
) -> torch.Tensor:
    """Mix task/style rewards with a distinct specialization coefficient."""
    if task_reward.shape != style_reward.shape or task_reward.shape != specialization_mask.shape:
        raise ValueError("Task reward, style reward, and specialization mask shapes must match.")
    if not 0.0 <= retention_task_lerp <= 1.0 or not 0.0 <= specialization_task_lerp <= 1.0:
        raise ValueError("Task/style interpolation coefficients must be in [0, 1].")
    task_lerp = torch.where(
        specialization_mask,
        torch.full_like(task_reward, float(specialization_task_lerp)),
        torch.full_like(task_reward, float(retention_task_lerp)),
    )
    return task_lerp * task_reward + (1.0 - task_lerp) * style_reward


class PPOAMP(PPO):

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer, 
        disc_demo_obs_buffer: CircularBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Frozen baseline policy KL regularizer
        baseline_kl_cfg: dict | None = None,
        # Counterfactual carrier-command action teacher
        command_bridge_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy,
            storage,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            normalize_advantage_per_mini_batch,
            device,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )
        
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for PPOAMP algorithm.")

        self.baseline_kl_cfg = baseline_kl_cfg or {}
        self.command_bridge_cfg = command_bridge_cfg or {}
        self.baseline_kl_scale = float(self.baseline_kl_cfg.get("scale", 0.0))
        self.baseline_kl_min_std = float(self.baseline_kl_cfg.get("min_std", 1.0e-4))
        self.baseline_kl_exempt_obs_index = int(self.baseline_kl_cfg.get("exempt_obs_index", -1))
        self.baseline_kl_exempt_obs_threshold = float(
            self.baseline_kl_cfg.get("exempt_obs_threshold", 0.5)
        )
        self.baseline_kl_mirror_phase_one = bool(self.baseline_kl_cfg.get("mirror_phase_one", False))
        self.baseline_kl_lift_obs_index = int(self.baseline_kl_cfg.get("lift_obs_index", 95))
        self.baseline_kl_mean_only = bool(self.baseline_kl_cfg.get("mean_only", False))
        self.baseline_kl_target = float(self.baseline_kl_cfg.get("target", 0.0))
        self.baseline_kl_min_scale = float(self.baseline_kl_cfg.get("min_scale", 0.0))
        self.baseline_kl_max_scale = float(self.baseline_kl_cfg.get("max_scale", 1.0))
        self.baseline_kl_adaptation_rate = float(self.baseline_kl_cfg.get("adaptation_rate", 1.5))
        self.baseline_kl_hard_limit = float(self.baseline_kl_cfg.get("hard_limit", 0.0))
        self.baseline_kl_command_conditioned = bool(self.baseline_kl_cfg.get("command_conditioned", False))
        self.baseline_kl_command_obs_start_index = int(self.baseline_kl_cfg.get("command_obs_start_index", 6))
        self.baseline_kl_specialization_scale = float(self.baseline_kl_cfg.get("specialization_scale", 0.0))
        self.baseline_kl_lateral_min_command = float(self.baseline_kl_cfg.get("lateral_min_command", 0.10))
        self.baseline_kl_pure_yaw_min_command = float(self.baseline_kl_cfg.get("pure_yaw_min_command", 0.10))
        self.baseline_kl_max_forward_command = float(self.baseline_kl_cfg.get("max_forward_command", 0.02))
        self.baseline_kl_max_lateral_yaw_command = float(
            self.baseline_kl_cfg.get("max_lateral_yaw_command", 0.05)
        )
        self.baseline_kl_max_pure_yaw_translation_command = float(
            self.baseline_kl_cfg.get("max_pure_yaw_translation_command", 0.02)
        )
        if self.baseline_kl_adaptation_rate <= 1.0:
            raise ValueError("baseline KL adaptation_rate must be greater than one.")
        if self.baseline_kl_max_scale < self.baseline_kl_min_scale:
            raise ValueError("baseline KL scale bounds are invalid.")
        self.baseline_policy = None
        if bool(self.baseline_kl_cfg.get("enabled", False)) and self.baseline_kl_scale > 0.0:
            if self.policy.is_recurrent:
                raise NotImplementedError("baseline_kl_cfg currently supports only non-recurrent ActorCritic policies.")
            checkpoint_path = os.path.abspath(os.path.expanduser(str(self.baseline_kl_cfg.get("checkpoint_path", ""))))
            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Baseline KL checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.baseline_policy = copy.deepcopy(self.policy).to(self.device)
            baseline_load = self.baseline_policy.load_state_dict(state_dict, strict=False)
            allowed_missing_prefixes = (
                "lateral_command_residual.",
                "pure_yaw_command_residual.",
                "fixed_command_bridge_fraction",
                "lateral_teacher_forward_command",
                "lateral_teacher_opposite_yaw_abs",
                "pure_yaw_teacher_forward_command",
            )
            unexpected = list(getattr(baseline_load, "unexpected_keys", []))
            missing = [
                key
                for key in getattr(baseline_load, "missing_keys", [])
                if not key.startswith(allowed_missing_prefixes)
            ]
            if unexpected or missing:
                raise RuntimeError(
                    "Frozen baseline policy is incompatible with command-residual policy: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            self.baseline_policy.eval()
            for parameter in self.baseline_policy.parameters():
                parameter.requires_grad_(False)
            # The KL/teacher baseline is always the original actor.  A fixed
            # carrier bridge belongs only to the trainable specialization policy.
            if hasattr(self.baseline_policy, "fixed_command_bridge_fraction"):
                self.baseline_policy.fixed_command_bridge_fraction.zero_()
            print(f"Loaded frozen baseline policy for KL regularization: {checkpoint_path}")

        self.command_bridge_enabled = bool(self.command_bridge_cfg.get("enabled", False))
        self.command_bridge_scale = float(self.command_bridge_cfg.get("scale", 0.0))
        self.command_bridge_teacher_delta_fraction = float(
            self.command_bridge_cfg.get("teacher_delta_fraction", 0.60)
        )
        self.command_bridge_residual_learning_rate = float(
            self.command_bridge_cfg.get("residual_learning_rate", 0.0)
        )
        self.command_bridge_residual_updates_per_batch = int(
            self.command_bridge_cfg.get("residual_updates_per_batch", 1)
        )
        if self.command_bridge_scale < 0.0:
            raise ValueError("command bridge scale must be non-negative.")
        if not 0.0 <= self.command_bridge_teacher_delta_fraction <= 1.0:
            raise ValueError("command bridge teacher_delta_fraction must be in [0, 1].")
        if self.command_bridge_residual_learning_rate < 0.0:
            raise ValueError("command bridge residual_learning_rate must be non-negative.")
        if self.command_bridge_residual_updates_per_batch <= 0:
            raise ValueError("command bridge residual_updates_per_batch must be positive.")
        if self.command_bridge_enabled and self.baseline_policy is None:
            raise ValueError("command bridge requires an enabled frozen baseline policy.")

        self.command_bridge_optimizer = None
        self.command_bridge_residual_parameters: list[torch.nn.Parameter] = []
        if self.command_bridge_enabled and self.command_bridge_residual_learning_rate > 0.0:
            residual_modules = (
                getattr(self.policy, "lateral_command_residual", None),
                getattr(self.policy, "pure_yaw_command_residual", None),
            )
            if any(module is None for module in residual_modules):
                raise ValueError(
                    "A residual-only command bridge requires lateral and pure-yaw residual modules."
                )
            self.command_bridge_residual_parameters = list(
                chain.from_iterable(module.parameters() for module in residual_modules)
            )
            self.command_bridge_optimizer = optim.Adam(
                self.command_bridge_residual_parameters,
                lr=self.command_bridge_residual_learning_rate,
            )
            print(
                "Using residual-only command-bridge optimizer: "
                f"lr={self.command_bridge_residual_learning_rate:.3e}, "
                f"updates_per_batch={self.command_bridge_residual_updates_per_batch}."
            )

        self.disc_normalizer_mode = self.amp_cfg.get("normalizer_mode", "policy")
        if self.disc_normalizer_mode not in {"policy", "policy_demo", "demo_static"}:
            raise ValueError(
                f"Unknown AMP discriminator normalizer mode: {self.disc_normalizer_mode}. "
                "Should be 'policy', 'policy_demo', or 'demo_static'."
            )
        self.command_conditioned_style_reward = bool(
            self.amp_cfg.get("command_conditioned_style_reward", False)
        )
        self.specialization_task_style_lerp = float(
            self.amp_cfg.get("specialization_task_style_lerp", 1.0)
        )
        self.style_command_obs_start_index = int(self.amp_cfg.get("command_obs_start_index", 6))
        if not 0.0 <= self.specialization_task_style_lerp <= 1.0:
            raise ValueError("specialization_task_style_lerp must be in [0, 1].")
        
        if self.amp_cfg["loss_type"] == "GAN":
            self.loss_type = LossType.GAN
        elif self.amp_cfg["loss_type"] == "LSGAN":
            self.loss_type = LossType.LSGAN
        elif self.amp_cfg["loss_type"] == "WGAN":
            self.loss_type = LossType.WGAN
        else:
            raise ValueError(f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'GAN', 'LSGAN', or 'WGAN'")
        
        self.amp_discriminator: AMPDiscriminator = AMPDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.policy.obs_groups,
            loss_type=self.loss_type,
            device=device,
            **self.amp_cfg.get("amp_discriminator", {})
        ).to(self.device)
        self.freeze_amp_discriminator = bool(self.amp_cfg.get("freeze_discriminator", False))
        
        # optimizer for policy and discriminator
        params = [
            {
                "name": "disc_trunk", 
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],  # L2 regularization for the discriminator trunk
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],  # L2 regularization for the discriminator linear layer
            }
        ]
        # use a separate optimizer for the AMP discriminator
        self.disc_optimizer = optim.Adam(
            params,
            lr=self.amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)
        if self.freeze_amp_discriminator:
            for parameter in self.amp_discriminator.parameters():
                parameter.requires_grad_(False)
        
        # Storage for AMP discriminator observations
        self.disc_obs_buffer: CircularBuffer = disc_obs_buffer
        self.disc_demo_obs_buffer: CircularBuffer = disc_demo_obs_buffer

    def _update_amp_normalizer(self, disc_obs_batch: torch.Tensor, disc_demo_obs_batch: torch.Tensor) -> None:
        if self.freeze_amp_discriminator:
            return
        if self.disc_normalizer_mode == "policy":
            self.amp_discriminator.update_normalization(disc_obs_batch)
        elif self.disc_normalizer_mode == "policy_demo":
            self.amp_discriminator.update_normalization(torch.cat((disc_obs_batch, disc_demo_obs_batch), dim=0))

    def _compute_command_bridge_loss(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Distill carrier-command action deltas only on strict two-goal samples."""
        if self.baseline_policy is None:
            return torch.zeros((), device=self.device)
        teacher_obs, lateral_bridge, pure_yaw_bridge = build_two_goal_carrier_teacher_obs(
            actor_obs,
            command_obs_start_index=int(self.command_bridge_cfg.get("command_obs_start_index", 6)),
            lateral_min_command=float(self.command_bridge_cfg.get("lateral_min_command", 0.10)),
            pure_yaw_min_command=float(self.command_bridge_cfg.get("pure_yaw_min_command", 0.10)),
            max_student_forward_command=float(
                self.command_bridge_cfg.get("max_student_forward_command", 0.02)
            ),
            max_lateral_yaw_command=float(
                self.command_bridge_cfg.get("max_lateral_yaw_command", 0.05)
            ),
            max_student_pure_yaw_translation_command=float(
                self.command_bridge_cfg.get("max_student_pure_yaw_translation_command", 0.02)
            ),
            lateral_teacher_forward_command=float(
                self.command_bridge_cfg.get("lateral_teacher_forward_command", 0.20)
            ),
            lateral_teacher_min_abs_command=float(
                self.command_bridge_cfg.get("lateral_teacher_min_abs_command", 0.0)
            ),
            lateral_teacher_opposite_yaw_abs=float(
                self.command_bridge_cfg.get("lateral_teacher_opposite_yaw_abs", 0.0)
            ),
            pure_yaw_teacher_forward_command=float(
                self.command_bridge_cfg.get("pure_yaw_teacher_forward_command", 0.15)
            ),
            pure_yaw_positive_teacher_yaw_scale=float(
                self.command_bridge_cfg.get("pure_yaw_positive_teacher_yaw_scale", 1.0)
            ),
            pure_yaw_negative_teacher_yaw_scale=float(
                self.command_bridge_cfg.get("pure_yaw_negative_teacher_yaw_scale", 1.0)
            ),
            pure_yaw_positive_teacher_yaw_min=float(
                self.command_bridge_cfg.get("pure_yaw_positive_teacher_yaw_min", 0.0)
            ),
            pure_yaw_positive_teacher_yaw_max=float(
                self.command_bridge_cfg.get("pure_yaw_positive_teacher_yaw_max", 10.0)
            ),
            pure_yaw_negative_teacher_yaw_min=float(
                self.command_bridge_cfg.get("pure_yaw_negative_teacher_yaw_min", 0.0)
            ),
            pure_yaw_negative_teacher_yaw_max=float(
                self.command_bridge_cfg.get("pure_yaw_negative_teacher_yaw_max", 10.0)
            ),
        )
        bridge_mask = lateral_bridge | pure_yaw_bridge
        with torch.no_grad():
            baseline_mu = self.baseline_policy.actor(
                self.baseline_policy.actor_obs_normalizer(actor_obs)
            )
            teacher_mu = self.baseline_policy.actor(
                self.baseline_policy.actor_obs_normalizer(teacher_obs)
            )
            target_mu = baseline_mu + self.command_bridge_teacher_delta_fraction * (
                teacher_mu - baseline_mu
            )
        student_mu = self.policy.actor_mean_from_flat_obs(actor_obs)
        per_sample_loss = torch.mean(torch.square(student_mu - target_mu), dim=-1)
        return torch.where(
            bridge_mask,
            per_sample_loss,
            torch.zeros_like(per_sample_loss),
        ).sum() / torch.clamp(bridge_mask.sum(), min=1)
        
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(extras["terminal_obs"], flatten_history_dim=False)
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]
        # Compute the Style Reward
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc_obs, dt=self.amp_cfg["step_dt"])
        # Linearly interpolate between task reward and style reward
        if getattr(self, "command_conditioned_style_reward", False):
            actor_obs = self.policy.get_actor_obs(obs)
            specialization_mask = two_goal_specialization_mask_from_policy_obs(
                actor_obs,
                command_obs_start_index=self.style_command_obs_start_index,
                lateral_min_command=self.baseline_kl_lateral_min_command,
                pure_yaw_min_command=self.baseline_kl_pure_yaw_min_command,
                max_forward_command=self.baseline_kl_max_forward_command,
                max_lateral_yaw_command=self.baseline_kl_max_lateral_yaw_command,
                max_pure_yaw_translation_command=self.baseline_kl_max_pure_yaw_translation_command,
            )
            self.rewards_lerp = command_conditioned_lerp_reward(
                rewards,
                self.style_rewards,
                specialization_mask,
                retention_task_lerp=float(self.amp_discriminator.task_style_lerp),
                specialization_task_lerp=self.specialization_task_style_lerp,
            )
        else:
            self.rewards_lerp = self.amp_discriminator.lerp_reward(
                task_reward=rewards, style_reward=self.style_rewards
            )
        # Store the un-normalized disc obs and disc demo obs into buffers
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        # Call the parent class method with the new rewards
        super().process_env_step(obs, self.rewards_lerp, dones, extras)
        
    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # AMP discriminator loss and other info
        mean_disc_loss = 0
        mean_disc_grad_penalty = 0
        mean_disc_score = 0
        mean_disc_demo_score = 0
        mean_baseline_kl = 0
        mean_baseline_kl_specialization = 0
        mean_baseline_kl_retention = 0
        mean_baseline_kl_weighted = 0
        mean_command_bridge = 0

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        # Iterate over batches
        for samples, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = samples
            
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.baseline_policy is not None:
                baseline_obs_batch = obs_batch[:original_batch_size]
                actor_obs_batch = self.policy.get_actor_obs(baseline_obs_batch)
                phase_one_mask = None
                teacher_obs_batch = baseline_obs_batch
                if self.baseline_kl_mirror_phase_one:
                    if self.baseline_kl_exempt_obs_index < 0:
                        raise ValueError("mirror_phase_one requires a non-negative phase observation index.")
                    if self.baseline_kl_exempt_obs_index >= actor_obs_batch.shape[1]:
                        raise ValueError(
                            f"baseline KL phase index={self.baseline_kl_exempt_obs_index} exceeds "
                            f"observation dimension {actor_obs_batch.shape[1]}."
                        )
                    if self.baseline_kl_lift_obs_index >= actor_obs_batch.shape[1]:
                        raise ValueError(
                            f"baseline KL lift index={self.baseline_kl_lift_obs_index} exceeds "
                            f"observation dimension {actor_obs_batch.shape[1]}."
                        )
                    phase_one_mask = (
                        actor_obs_batch[:, self.baseline_kl_exempt_obs_index]
                        >= self.baseline_kl_exempt_obs_threshold
                    )
                    mirrored_obs_batch = _mirror_g1_policy_observation_tensor(actor_obs_batch)
                    mirrored_obs_batch[:, self.baseline_kl_exempt_obs_index] = 0.0
                    mirrored_obs_batch[:, self.baseline_kl_lift_obs_index] = actor_obs_batch[
                        :, self.baseline_kl_lift_obs_index
                    ]
                    teacher_actor_obs_batch = torch.where(
                        phase_one_mask.unsqueeze(1), mirrored_obs_batch, actor_obs_batch
                    )
                    teacher_obs_batch = baseline_obs_batch.clone()
                    teacher_obs_batch["policy"] = teacher_actor_obs_batch
                with torch.no_grad():
                    self.baseline_policy.act(teacher_obs_batch)
                    baseline_mu_batch = self.baseline_policy.action_mean.detach()
                    if phase_one_mask is not None:
                        baseline_mu_batch = torch.where(
                            phase_one_mask.unsqueeze(1),
                            _mirror_g1_joint_tensor(baseline_mu_batch),
                            baseline_mu_batch,
                        )
                baseline_sigma_batch = self.baseline_policy.action_std.detach().clamp_min(self.baseline_kl_min_std)
                if phase_one_mask is not None:
                    baseline_sigma_batch = torch.where(
                        phase_one_mask.unsqueeze(1),
                        _mirror_g1_joint_tensor(baseline_sigma_batch, apply_sign=False),
                        baseline_sigma_batch,
                    )
                if self.baseline_kl_mean_only:
                    baseline_kl_per_sample = torch.sum(
                        torch.square(mu_batch - baseline_mu_batch)
                        / (2.0 * torch.square(baseline_sigma_batch)),
                        dim=-1,
                    )
                else:
                    current_sigma_batch = sigma_batch.clamp_min(self.baseline_kl_min_std)
                    baseline_kl_per_sample = torch.sum(
                        torch.log(baseline_sigma_batch / current_sigma_batch)
                        + (torch.square(current_sigma_batch) + torch.square(mu_batch - baseline_mu_batch))
                        / (2.0 * torch.square(baseline_sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                if self.baseline_kl_exempt_obs_index >= 0 and not self.baseline_kl_mirror_phase_one:
                    if self.baseline_kl_exempt_obs_index >= actor_obs_batch.shape[1]:
                        raise ValueError(
                            f"baseline KL exempt_obs_index={self.baseline_kl_exempt_obs_index} exceeds "
                            f"observation dimension {actor_obs_batch.shape[1]}."
                        )
                    baseline_mask = (
                        actor_obs_batch[:, self.baseline_kl_exempt_obs_index]
                        < self.baseline_kl_exempt_obs_threshold
                    ).to(baseline_kl_per_sample.dtype)
                    baseline_kl_loss = torch.sum(
                        baseline_kl_per_sample * baseline_mask
                    ) / torch.clamp_min(torch.sum(baseline_mask), 1.0)
                else:
                    baseline_kl_loss = baseline_kl_per_sample.mean()
                if self.baseline_kl_command_conditioned:
                    specialization_mask = two_goal_specialization_mask_from_policy_obs(
                        actor_obs_batch,
                        command_obs_start_index=self.baseline_kl_command_obs_start_index,
                        lateral_min_command=self.baseline_kl_lateral_min_command,
                        pure_yaw_min_command=self.baseline_kl_pure_yaw_min_command,
                        max_forward_command=self.baseline_kl_max_forward_command,
                        max_lateral_yaw_command=self.baseline_kl_max_lateral_yaw_command,
                        max_pure_yaw_translation_command=(
                            self.baseline_kl_max_pure_yaw_translation_command
                        ),
                    )
                    retention_mask = ~specialization_mask
                    baseline_kl_specialization = torch.where(
                        specialization_mask,
                        baseline_kl_per_sample,
                        torch.zeros_like(baseline_kl_per_sample),
                    ).sum() / torch.clamp(specialization_mask.sum(), min=1)
                    baseline_kl_retention = torch.where(
                        retention_mask,
                        baseline_kl_per_sample,
                        torch.zeros_like(baseline_kl_per_sample),
                    ).sum() / torch.clamp(retention_mask.sum(), min=1)
                    kl_scale = torch.where(
                        specialization_mask,
                        torch.full_like(baseline_kl_per_sample, self.baseline_kl_specialization_scale),
                        torch.full_like(baseline_kl_per_sample, self.baseline_kl_scale),
                    )
                    baseline_kl_weighted_loss = torch.mean(kl_scale * baseline_kl_per_sample)
                else:
                    baseline_kl_specialization = torch.zeros((), device=self.device)
                    baseline_kl_retention = baseline_kl_loss
                    baseline_kl_weighted_loss = self.baseline_kl_scale * baseline_kl_loss
                loss = loss + baseline_kl_weighted_loss

                if self.command_bridge_enabled and self.command_bridge_scale > 0.0:
                    command_bridge_loss = self._compute_command_bridge_loss(actor_obs_batch)
                    if self.command_bridge_optimizer is None:
                        loss = loss + self.command_bridge_scale * command_bridge_loss
                else:
                    command_bridge_loss = torch.zeros((), device=self.device)
            else:
                baseline_kl_loss = torch.zeros((), device=self.device)
                baseline_kl_specialization = torch.zeros((), device=self.device)
                baseline_kl_retention = torch.zeros((), device=self.device)
                baseline_kl_weighted_loss = torch.zeros((), device=self.device)
                command_bridge_loss = torch.zeros((), device=self.device)

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # AMP discriminator loss
            with torch.no_grad():
                disc_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch) # [mini_batch_size, disc_obs_steps, disc_obs_dim]
                disc_demo_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)
            
            mini_batch_size = disc_obs_batch_normed.shape[0]
            disc_score = self.amp_discriminator(disc_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            disc_demo_score = self.amp_discriminator(disc_demo_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            
            if self.loss_type == LossType.GAN:
                bce = torch.nn.BCEWithLogitsLoss()
                policy_loss = bce(
                    disc_score, torch.zeros_like(disc_score, device=self.device)
                )
                demo_loss = bce(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.LSGAN:
                policy_loss = torch.nn.MSELoss()(
                    disc_score, -1 * torch.ones_like(disc_score, device=self.device)
                )
                demo_loss = torch.nn.MSELoss()(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.WGAN:
                disc_loss = - torch.mean(disc_demo_score) + torch.mean(disc_score)
            else: 
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'")

            if self.freeze_amp_discriminator:
                disc_grad_penalty = torch.zeros((), device=self.device)
            else:
                disc_grad_penalty = self.amp_discriminator.compute_grad_penalty(
                    demo_data=disc_demo_obs_batch_normed.reshape(mini_batch_size, -1),
                    scale=self.amp_cfg["grad_penalty_scale"]
                )
            disc_total_loss = disc_loss + disc_grad_penalty

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            # Compute the gradients for AMP discriminator
            if not self.freeze_amp_discriminator:
                self.disc_optimizer.zero_grad()
                disc_total_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # A dedicated optimizer lets the carrier teacher establish gait
            # onset without raising the critic/base-actor PPO learning rate.
            if self.command_bridge_optimizer is not None and self.command_bridge_scale > 0.0:
                for _ in range(self.command_bridge_residual_updates_per_batch):
                    self.command_bridge_optimizer.zero_grad()
                    command_bridge_loss = self._compute_command_bridge_loss(actor_obs_batch.detach())
                    command_bridge_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.command_bridge_residual_parameters,
                        self.max_grad_norm,
                    )
                    self.command_bridge_optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # Apply the gradients for AMP discriminator
            if not self.freeze_amp_discriminator:
                self.disc_optimizer.step()
            # Update the AMP normalizer
            self._update_amp_normalizer(disc_obs_batch, disc_demo_obs_batch)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # AMP discriminator loss and other info
            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += disc_grad_penalty.item()
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()
            mean_baseline_kl += baseline_kl_loss.item()
            mean_baseline_kl_specialization += baseline_kl_specialization.item()
            mean_baseline_kl_retention += baseline_kl_retention.item()
            mean_baseline_kl_weighted += baseline_kl_weighted_loss.item()
            mean_command_bridge += command_bridge_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_disc_grad_penalty /= num_updates
        mean_disc_score /= num_updates
        mean_disc_demo_score /= num_updates
        mean_baseline_kl /= num_updates
        mean_baseline_kl_specialization /= num_updates
        mean_baseline_kl_retention /= num_updates
        mean_baseline_kl_weighted /= num_updates
        mean_command_bridge /= num_updates

        if (
            self.baseline_policy is not None
            and self.baseline_kl_hard_limit > 0.0
            and mean_baseline_kl_retention > self.baseline_kl_hard_limit
        ):
            raise RuntimeError(
                "Baseline-policy KL safety limit exceeded: "
                f"{mean_baseline_kl_retention:.6f} > {self.baseline_kl_hard_limit:.6f}. "
                "Refusing to continue a degrading refinement run."
            )

        if self.baseline_policy is not None and self.baseline_kl_target > 0.0:
            if mean_baseline_kl_retention > self.baseline_kl_target * 1.5:
                self.baseline_kl_scale = min(
                    self.baseline_kl_scale * self.baseline_kl_adaptation_rate,
                    self.baseline_kl_max_scale,
                )
            elif mean_baseline_kl_retention < self.baseline_kl_target / 1.5:
                self.baseline_kl_scale = max(
                    self.baseline_kl_scale / self.baseline_kl_adaptation_rate,
                    self.baseline_kl_min_scale,
                )

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.baseline_policy is not None:
            loss_dict["baseline_kl"] = mean_baseline_kl
            loss_dict["baseline_kl_specialization"] = mean_baseline_kl_specialization
            loss_dict["baseline_kl_retention"] = mean_baseline_kl_retention
            loss_dict["baseline_kl_weighted"] = mean_baseline_kl_weighted
            loss_dict["baseline_kl_scale"] = self.baseline_kl_scale
        if self.command_bridge_enabled:
            loss_dict["command_bridge"] = mean_command_bridge
        loss_dict["amp/disc_loss"] = mean_disc_loss
        loss_dict["amp/disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["amp/disc_score"] = mean_disc_score
        loss_dict["amp/disc_demo_score"] = mean_disc_demo_score

        return loss_dict
