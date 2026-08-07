"""RSL-RL configuration classes used by Legged Lab tasks.

Core classes:
    RslRlPpoActorCriticConv2dCfg configures optional convolutional actor-critic
    networks. RslRlPpoAmpAlgorithmCfg configures PPOAMP, AMP style reward, and
    an optional frozen baseline-policy KL anchor for fine-tuning.

Inputs/outputs:
    These config classes are converted to dictionaries by IsaacLab/RSL-RL and
    passed to the runner and algorithm constructors.

Usage:
    agent.algorithm.baseline_kl_cfg.enabled=True
    agent.algorithm.baseline_kl_cfg.checkpoint_path=/path/to/model_2999.pt
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################

@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class RslRlPpoActorCriticCommandResidualCfg(RslRlPpoActorCriticCfg):
    """Feed-forward actor with zero-initialized two-goal residual adapters."""

    class_name: str = "ActorCriticCommandResidual"
    command_residual_hidden_dim: int = 64
    command_obs_start_index: int = 6
    lateral_min_command: float = 0.10
    pure_yaw_min_command: float = 0.10
    max_lateral_forward_command: float = 0.02
    max_lateral_yaw_command: float = 0.05
    max_pure_yaw_translation_command: float = 0.02
    fixed_command_bridge_fraction: float = 0.0
    lateral_teacher_forward_command: float = 0.20
    pure_yaw_teacher_forward_command: float = 0.15
    pure_yaw_positive_teacher_yaw_scale: float = 1.0
    pure_yaw_negative_teacher_yaw_scale: float = 1.0

############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg:
    """Configuration for the AMP algorithm."""

    class_name: str = "PPOAMP"
    """The algorithm class name. Default is PPOAMP."""

    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None

    @configclass
    class BaselineKLCfg:
        """Configuration for a frozen baseline-policy KL regularizer."""

        enabled: bool = False
        """Whether to load a frozen baseline policy and add the KL loss."""

        checkpoint_path: str = ""
        """Checkpoint containing ``model_state_dict`` for the frozen baseline policy."""

        scale: float = 0.0
        """Loss multiplier for KL(current_policy || baseline_policy)."""

        min_std: float = 1.0e-4
        """Minimum standard deviation used in the analytic Gaussian KL."""

        mean_only: bool = False
        """Constrain actor means while allowing a fresh exploration standard deviation."""

        target: float = 0.0
        """Optional target used to adapt the baseline KL multiplier between updates."""

        min_scale: float = 0.0
        """Lower bound for adaptive baseline KL scaling."""

        max_scale: float = 1.0
        """Upper bound for adaptive baseline KL scaling."""

        adaptation_rate: float = 1.5
        """Multiplicative baseline KL scale adjustment when the target is missed."""

        hard_limit: float = 0.0
        """Abort an update when mean baseline-policy KL exceeds this positive limit."""

        command_conditioned: bool = False
        """Apply separate KL scales to specialization and retention commands."""

        command_obs_start_index: int = 6
        """Start of the ``[vx, vy, wz]`` command in the flattened 96-D policy input."""

        specialization_scale: float = 0.0
        """KL multiplier for pure-lateral and exact-zero-linear pure-yaw samples."""

        lateral_min_command: float = 0.10
        pure_yaw_min_command: float = 0.10
        max_forward_command: float = 0.25
        max_lateral_yaw_command: float = 0.05
        max_pure_yaw_translation_command: float = 0.25
        """Thresholds used to identify the two specialization command families."""

    baseline_kl_cfg: BaselineKLCfg = BaselineKLCfg()
    """Optional frozen baseline-policy KL regularizer configuration."""

    @configclass
    class CommandBridgeCfg:
        """Counterfactual teacher used to cross the baseline gait-onset dead zone."""

        enabled: bool = False
        scale: float = 0.0
        command_obs_start_index: int = 6
        lateral_min_command: float = 0.10
        pure_yaw_min_command: float = 0.10
        max_student_forward_command: float = 0.02
        max_lateral_yaw_command: float = 0.05
        max_student_pure_yaw_translation_command: float = 0.02
        lateral_teacher_forward_command: float = 0.20
        pure_yaw_teacher_forward_command: float = 0.15
        pure_yaw_positive_teacher_yaw_scale: float = 1.0
        pure_yaw_negative_teacher_yaw_scale: float = 1.0
        teacher_delta_fraction: float = 0.60
        residual_learning_rate: float = 0.0
        """Optional learning rate for a residual-only teacher optimizer."""

        residual_updates_per_batch: int = 1
        """Teacher-only residual updates after each PPO mini-batch."""

    command_bridge_cfg: CommandBridgeCfg = CommandBridgeCfg()
    """Optional carrier-command action teacher for strict two-goal samples."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


#########################
# Runner configurations #
#########################
