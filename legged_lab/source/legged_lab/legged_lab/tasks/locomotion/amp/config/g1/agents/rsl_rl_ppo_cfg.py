"""RSL-RL PPO-AMP runner configuration for Unitree G1 flat-walking baseline.

Core class:
    G1RslRlOnPolicyRunnerAmpCfg selects AMPRunner, PPOAMP, G1 symmetry
    augmentation, and discriminator hyperparameters for the 29-DoF policy.

Inputs/outputs:
    The runner consumes observation groups named policy, critic, discriminator,
    and discriminator_demonstration from G1AmpEnvCfg. Checkpoints are written
    under logs/rsl_rl/g1_amp and exported by scripts/rsl_rl/play.py.

Usage:
    python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless
    python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 3000
"""

import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg
from legged_lab.rsl_rl import RslRlPpoAmpAlgorithmCfg, RslRlAmpCfg, RslRlPpoActorCriticConv2dCfg
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.mdp.symmetry import g1

@configclass
class G1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "g1_amp"
    obs_groups = {
        "policy": ["policy"], 
        "critic": ["critic"], 
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     actor_obs_normalization=False,
    #     critic_obs_normalization=False,
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=1
    # )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4
            ),
            loss_type="LSGAN"
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True, mirror_loss_coeff=0.1,
        )
    )


@configclass
class G1Nav2BehaviorFinetuneRslRlOnPolicyRunnerAmpCfg(
    G1RslRlOnPolicyRunnerAmpCfg
):
    """Actor-only continuation runner for generic full-body Nav2 behavior."""

    experiment_name = "g1_amp_nav2_behavior"
    checkpoint_output_dir = "Nav2BehaviorFinetune"
    load_actor_only = True
    load_policy_only = False
    reset_iteration_on_policy_only_load = True
    reset_amp_on_load = False

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.amp_cfg.grad_penalty_scale = 20.0
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 5.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.4


@configclass
class G1Nav2TwoGoalFinetuneRslRlOnPolicyRunnerAmpCfg(
    G1RslRlOnPolicyRunnerAmpCfg
):
    """Conservative actor refinement for lateral safety and in-place yaw."""

    experiment_name = "g1_amp_nav2_two_goal"
    checkpoint_output_dir = "Nav2TwoGoalFinetune"
    load_actor_only = False
    load_actor_amp_only = True
    load_policy_only = False
    reset_iteration_on_policy_only_load = True
    reset_amp_on_load = False
    freeze_actor_hidden_layers = 2
    save_interval = 20

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.policy.init_noise_std = 0.25
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.schedule = "fixed"
        self.algorithm.desired_kl = 0.003
        self.algorithm.clip_param = 0.10
        self.algorithm.num_learning_epochs = 2
        self.algorithm.num_mini_batches = 4
        self.algorithm.entropy_coef = 2.0e-4
        self.algorithm.max_grad_norm = 0.5
        self.algorithm.baseline_kl_cfg.mean_only = True
        self.algorithm.baseline_kl_cfg.target = 0.05
        self.algorithm.baseline_kl_cfg.min_scale = 0.01
        self.algorithm.baseline_kl_cfg.max_scale = 0.50
        self.algorithm.baseline_kl_cfg.adaptation_rate = 1.5
        self.algorithm.baseline_kl_cfg.hard_limit = 0.20
        self.algorithm.amp_cfg.freeze_discriminator = True
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 5.0
        # In PPOAMP this is the task fraction: 0.85 task + 0.15 frozen style.
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.85
