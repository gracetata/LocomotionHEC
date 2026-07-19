"""RSL-RL runner configs for G1 upper-body perturbation AMP tasks."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


@configclass
class G1PerturbRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Base AMP runner config for perturbation tasks."""

    experiment_name = "g1_perturb"
    load_policy_only = False
    reset_iteration_on_policy_only_load = True
    reset_amp_on_load = False


@configclass
class G1StandPerturbRslRlOnPolicyRunnerAmpCfg(G1PerturbRslRlOnPolicyRunnerAmpCfg):
    """Runner config for stand perturbation."""

    experiment_name = "g1_stand_perturb"
    checkpoint_output_dir = "ArmHack Checkpoints/StandPerturb"
    load_policy_only = True

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0


@configclass
class G1WalkPerturbFinetuneRslRlOnPolicyRunnerAmpCfg(G1PerturbRslRlOnPolicyRunnerAmpCfg):
    """Runner config for walk perturbation fine-tuning."""

    experiment_name = "g1_walk_perturb"
    checkpoint_output_dir = "ArmHack Checkpoints/WalkPerturbFinetune"
    load_policy_only = False

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.desired_kl = 0.01
        self.algorithm.entropy_coef = 0.002
        self.algorithm.amp_cfg.grad_penalty_scale = 20.0
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0


@configclass
class G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg(
    G1WalkPerturbFinetuneRslRlOnPolicyRunnerAmpCfg
):
    """Walk-only P0/P1 runner with lower-body AMP enabled by default."""

    experiment_name = "g1_walk_robust"
    checkpoint_output_dir = "ArmHack Checkpoints/WalkPerturbFinetune"

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 1.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.85


@configclass
class G1WalkBehaviorFinetuneRslRlOnPolicyRunnerAmpCfg(
    G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg
):
    """Walk-only continuation runner for stop/micro/turn/foot-placement behavior."""

    experiment_name = "g1_walk_behavior"
    checkpoint_output_dir = "ArmHack Checkpoints/WalkBehaviorFinetune"
    load_policy_only = False

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        # The first curriculum phase must let strict stop terms dominate the
        # walking-only discriminator.  The launcher raises style weight later.
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0
