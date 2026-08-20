"""PPO-AMP runner config for full-body extreme Stand recovery."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RslRlOnPolicyRunnerAmpCfg,
)


@configclass
class G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Policy-only continuation runner; task rewards drive recovery and hold."""

    experiment_name = "g1_extreme_stand_recovery"
    checkpoint_output_dir = "ExtremeStandRecovery Checkpoints"
    load_policy_only = True
    reset_iteration_on_policy_only_load = True
    reset_amp_on_load = False

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.01
        self.algorithm.entropy_coef = 0.003
        self.algorithm.amp_cfg.grad_penalty_scale = 20.0
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0


@configclass
class G1ExtremeStandRecoveryV4JerkLimitedRslRlOnPolicyRunnerAmpCfg(
    G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg
):
    """Minimal-output-layer adaptation after stable damping changes the plant."""

    freeze_actor_hidden_layers = 3
    actor_warmup_iterations = 25
    restore_configured_learning_rate_on_load = True
    save_interval = 5

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 1
        self.algorithm.num_mini_batches = 4
        self.algorithm.clip_param = 0.03
        self.algorithm.entropy_coef = 0.0
        self.algorithm.max_grad_norm = 0.05
        self.algorithm.desired_kl = 0.001


@configclass
class G1ExtremeStandRecoveryConservativeV10RslRlOnPolicyRunnerAmpCfg(
    G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg
):
    """Critic-first, output-layer-only refinement of the stable V4 actor."""

    freeze_actor_hidden_layers = 3
    actor_warmup_iterations = 20
    restore_configured_learning_rate_on_load = True
    save_interval = 1

    def __post_init__(self):
        super().__post_init__()
        # V9 changed all actor layers during its first five-epoch PPO update and
        # immediately created a closed-loop limit cycle.  V10 first adapts the
        # critic, then permits only the final actor projection to move in one
        # conservative epoch per rollout.
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 1
        self.algorithm.num_mini_batches = 4
        self.algorithm.clip_param = 0.05
        self.algorithm.entropy_coef = 0.0
        self.algorithm.max_grad_norm = 0.10
        self.algorithm.desired_kl = 0.002
