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
