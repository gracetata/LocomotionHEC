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

    baseline_kl_cfg: BaselineKLCfg = BaselineKLCfg()
    """Optional frozen baseline-policy KL regularizer configuration."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


#########################
# Runner configurations #
#########################

    