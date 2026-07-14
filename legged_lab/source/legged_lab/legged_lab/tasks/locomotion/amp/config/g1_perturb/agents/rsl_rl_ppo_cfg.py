"""RSL-RL runner configs for G1 upper-body perturbation AMP tasks."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


@configclass
class G1PerturbRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Base AMP runner config for perturbation tasks."""

    experiment_name = "g1_perturb"
    load_policy_only = False
    reset_iteration_on_policy_only_load = True


@configclass
class G1StandPerturbRslRlOnPolicyRunnerAmpCfg(G1PerturbRslRlOnPolicyRunnerAmpCfg):
    """Runner config for stand perturbation."""

    experiment_name = "g1_stand_perturb"

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0


@configclass
class G1WalkPerturbFinetuneRslRlOnPolicyRunnerAmpCfg(G1PerturbRslRlOnPolicyRunnerAmpCfg):
    """Runner config for walk perturbation fine-tuning."""

    experiment_name = "g1_amp"
    load_policy_only = True
