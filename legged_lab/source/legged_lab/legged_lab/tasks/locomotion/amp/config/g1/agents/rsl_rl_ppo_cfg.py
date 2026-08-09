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
from legged_lab.rsl_rl import (
    RslRlAmpCfg,
    RslRlPpoActorCriticCommandResidualCfg,
    RslRlPpoActorCriticConv2dCfg,
    RslRlPpoAmpAlgorithmCfg,
)
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
    policy = RslRlPpoActorCriticCommandResidualCfg(
        init_noise_std=0.35,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        command_residual_hidden_dim=64,
        fixed_command_bridge_fraction=1.0,
        lateral_teacher_forward_command=0.20,
        lateral_teacher_min_abs_command=0.25,
        lateral_teacher_opposite_yaw_abs=0.0,
        pure_yaw_teacher_forward_command=0.10,
        pure_yaw_positive_teacher_yaw_scale=1.652,
        pure_yaw_negative_teacher_yaw_scale=1.428571429,
        pure_yaw_positive_teacher_yaw_min=0.55,
        pure_yaw_positive_teacher_yaw_max=0.5782,
        pure_yaw_negative_teacher_yaw_min=0.285714286,
        pure_yaw_negative_teacher_yaw_max=0.50,
    )
    freeze_actor_hidden_layers = 0
    freeze_base_actor = True
    freeze_lateral_residual = False
    freeze_pure_yaw_residual = True
    actor_warmup_iterations = 8
    restore_configured_learning_rate_on_load = True
    save_interval = 10

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.policy.init_noise_std = 0.35
        self.algorithm.learning_rate = 7.5e-6
        self.algorithm.schedule = "fixed"
        self.algorithm.desired_kl = 0.01
        self.algorithm.clip_param = 0.12
        self.algorithm.num_learning_epochs = 2
        self.algorithm.num_mini_batches = 4
        self.algorithm.entropy_coef = 8.0e-4
        self.algorithm.max_grad_norm = 0.5
        self.algorithm.baseline_kl_cfg.mean_only = True
        self.algorithm.baseline_kl_cfg.command_conditioned = True
        self.algorithm.baseline_kl_cfg.command_obs_start_index = 6
        self.algorithm.baseline_kl_cfg.specialization_scale = 0.005
        self.algorithm.baseline_kl_cfg.max_forward_command = 0.25
        self.algorithm.baseline_kl_cfg.max_pure_yaw_translation_command = 0.25
        self.algorithm.baseline_kl_cfg.target = 0.0
        self.algorithm.baseline_kl_cfg.min_scale = 0.08
        self.algorithm.baseline_kl_cfg.max_scale = 0.08
        self.algorithm.baseline_kl_cfg.adaptation_rate = 1.5
        self.algorithm.baseline_kl_cfg.hard_limit = 0.15
        self.algorithm.command_bridge_cfg.enabled = True
        self.algorithm.command_bridge_cfg.scale = 0.20
        self.algorithm.command_bridge_cfg.command_obs_start_index = 6
        self.algorithm.command_bridge_cfg.max_student_forward_command = 0.02
        self.algorithm.command_bridge_cfg.max_student_pure_yaw_translation_command = 0.02
        self.algorithm.command_bridge_cfg.lateral_teacher_forward_command = 0.20
        self.algorithm.command_bridge_cfg.lateral_teacher_min_abs_command = 0.25
        self.algorithm.command_bridge_cfg.lateral_teacher_opposite_yaw_abs = 0.0
        self.algorithm.command_bridge_cfg.pure_yaw_teacher_forward_command = 0.10
        self.algorithm.command_bridge_cfg.pure_yaw_positive_teacher_yaw_scale = 1.652
        self.algorithm.command_bridge_cfg.pure_yaw_negative_teacher_yaw_scale = 1.428571429
        self.algorithm.command_bridge_cfg.pure_yaw_positive_teacher_yaw_min = 0.55
        self.algorithm.command_bridge_cfg.pure_yaw_positive_teacher_yaw_max = 0.5782
        self.algorithm.command_bridge_cfg.pure_yaw_negative_teacher_yaw_min = 0.285714286
        self.algorithm.command_bridge_cfg.pure_yaw_negative_teacher_yaw_max = 0.50
        self.algorithm.command_bridge_cfg.teacher_delta_fraction = 0.60
        self.algorithm.amp_cfg.freeze_discriminator = True
        self.algorithm.amp_cfg.command_conditioned_style_reward = True
        self.algorithm.amp_cfg.specialization_task_style_lerp = 1.0
        self.algorithm.amp_cfg.command_obs_start_index = 6
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 5.0
        # In PPOAMP this is the task fraction: 0.85 task + 0.15 frozen style.
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.85


@configclass
class G1Nav2TwoGoalModel9996BootstrapRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalFinetuneRslRlOnPolicyRunnerAmpCfg
):
    """Bootstrap strict-command residuals from a frozen model_9996 actor."""

    experiment_name = "g1_amp_nav2_two_goal_model9996"
    checkpoint_output_dir = "Nav2TwoGoalModel9996"
    load_actor_amp_only = True
    load_policy_only = False
    freeze_pure_yaw_residual = False
    actor_warmup_iterations = 0
    save_interval = 2

    def __post_init__(self):
        super().__post_init__()
        # The carrier exists only in the auxiliary training target. Inference
        # always remains base actor + learned strict-command residual.
        self.policy.fixed_command_bridge_fraction = 0.0
        self.policy.lateral_teacher_min_abs_command = 0.0
        self.policy.pure_yaw_positive_teacher_yaw_scale = 1.0
        self.policy.pure_yaw_negative_teacher_yaw_scale = 1.0
        self.policy.pure_yaw_positive_teacher_yaw_min = 0.0
        self.policy.pure_yaw_positive_teacher_yaw_max = 10.0
        self.policy.pure_yaw_negative_teacher_yaw_min = 0.0
        self.policy.pure_yaw_negative_teacher_yaw_max = 10.0
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.clip_param = 0.15
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 8.0e-4
        self.algorithm.baseline_kl_cfg.specialization_scale = 0.0
        self.algorithm.command_bridge_cfg.enabled = True
        self.algorithm.command_bridge_cfg.scale = 0.20
        self.algorithm.command_bridge_cfg.teacher_delta_fraction = 0.80
        self.algorithm.command_bridge_cfg.residual_learning_rate = 3.0e-4
        self.algorithm.command_bridge_cfg.residual_updates_per_batch = 1
        self.algorithm.command_bridge_cfg.lateral_teacher_forward_command = 0.20
        self.algorithm.command_bridge_cfg.lateral_teacher_min_abs_command = 0.0
        self.algorithm.command_bridge_cfg.pure_yaw_teacher_forward_command = 0.15
        self.algorithm.command_bridge_cfg.pure_yaw_positive_teacher_yaw_scale = 1.0
        self.algorithm.command_bridge_cfg.pure_yaw_negative_teacher_yaw_scale = 1.0
        self.algorithm.command_bridge_cfg.pure_yaw_positive_teacher_yaw_min = 0.0
        self.algorithm.command_bridge_cfg.pure_yaw_positive_teacher_yaw_max = 10.0
        self.algorithm.command_bridge_cfg.pure_yaw_negative_teacher_yaw_min = 0.0
        self.algorithm.command_bridge_cfg.pure_yaw_negative_teacher_yaw_max = 10.0
        self.algorithm.amp_cfg.specialization_task_style_lerp = 1.0


@configclass
class G1Nav2TwoGoalModel9996CorrectiveRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996BootstrapRslRlOnPolicyRunnerAmpCfg
):
    """Remove carrier-like drift using only physical task rewards."""

    load_actor_amp_only = False
    load_policy_only = True
    reset_iteration_on_policy_only_load = True

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.num_learning_epochs = 4
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.command_bridge_cfg.enabled = False
        self.algorithm.command_bridge_cfg.scale = 0.0
        self.algorithm.command_bridge_cfg.residual_learning_rate = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0


@configclass
class G1Nav2TwoGoalModel9996BarrierCorrectiveRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996CorrectiveRslRlOnPolicyRunnerAmpCfg
):
    """Small policy-only updates for the widened sole barrier stage."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.clip_param = 0.12
        self.algorithm.num_learning_epochs = 4
        self.algorithm.entropy_coef = 4.0e-4


@configclass
class G1Nav2TwoGoalModel9996LateralSpecialistRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996BarrierCorrectiveRslRlOnPolicyRunnerAmpCfg
):
    """Update only the lateral residual; preserve pure yaw bit-for-bit."""

    freeze_lateral_residual = False
    freeze_pure_yaw_residual = True

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.clip_param = 0.10
        self.algorithm.entropy_coef = 3.0e-4


@configclass
class G1Nav2TwoGoalModel9996YawSpecialistRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996BarrierCorrectiveRslRlOnPolicyRunnerAmpCfg
):
    """Update only the pure-yaw residual; preserve lateral motion bit-for-bit."""

    freeze_lateral_residual = True
    freeze_pure_yaw_residual = False

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.clip_param = 0.10
        self.algorithm.entropy_coef = 3.0e-4


@configclass
class G1Nav2TwoGoalModel9996FullActorRslRlOnPolicyRunnerAmpCfg(
    G1RslRlOnPolicyRunnerAmpCfg
):
    """Fine-tune model_9996 with enough capacity to change contact timing."""

    experiment_name = "g1_amp_nav2_two_goal_model9996_full_actor"
    checkpoint_output_dir = "Nav2TwoGoalModel9996FullActor"
    load_actor_only = False
    load_actor_amp_only = True
    load_policy_only = False
    reset_iteration_on_policy_only_load = True
    reset_amp_on_load = False
    freeze_actor_hidden_layers = 1
    freeze_base_actor = False
    actor_warmup_iterations = 8
    restore_configured_learning_rate_on_load = True
    save_interval = 5
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.desired_kl = 0.01
        self.algorithm.clip_param = 0.15
        self.algorithm.num_learning_epochs = 3
        self.algorithm.num_mini_batches = 4
        self.algorithm.entropy_coef = 8.0e-4
        self.algorithm.max_grad_norm = 0.5
        self.algorithm.baseline_kl_cfg.mean_only = True
        self.algorithm.baseline_kl_cfg.command_conditioned = True
        self.algorithm.baseline_kl_cfg.command_obs_start_index = 6
        self.algorithm.baseline_kl_cfg.specialization_scale = 0.005
        self.algorithm.baseline_kl_cfg.max_forward_command = 0.02
        self.algorithm.baseline_kl_cfg.max_lateral_yaw_command = 0.05
        self.algorithm.baseline_kl_cfg.max_pure_yaw_translation_command = 0.02
        self.algorithm.baseline_kl_cfg.target = 0.02
        self.algorithm.baseline_kl_cfg.min_scale = 0.08
        self.algorithm.baseline_kl_cfg.max_scale = 0.08
        self.algorithm.baseline_kl_cfg.adaptation_rate = 1.5
        self.algorithm.baseline_kl_cfg.hard_limit = 0.15
        self.algorithm.command_bridge_cfg.enabled = False
        self.algorithm.command_bridge_cfg.scale = 0.0
        self.algorithm.amp_cfg.freeze_discriminator = True
        self.algorithm.amp_cfg.command_conditioned_style_reward = True
        self.algorithm.amp_cfg.specialization_task_style_lerp = 1.0
        self.algorithm.amp_cfg.command_obs_start_index = 6
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 5.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.85


@configclass
class G1Nav2TwoGoalModel9996FullActorLateralRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996FullActorRslRlOnPolicyRunnerAmpCfg
):
    """Continue the lateral expert; retention is provided by deployment gating."""

    experiment_name = "g1_amp_nav2_two_goal_model9996_full_actor_lateral"
    checkpoint_output_dir = "Nav2TwoGoalModel9996FullActorLateral"
    load_actor_amp_only = False
    load_policy_only = True
    actor_warmup_iterations = 0
    save_interval = 2

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.clip_param = 0.10
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.baseline_kl_cfg.enabled = False
        self.algorithm.baseline_kl_cfg.scale = 0.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 1.0


@configclass
class G1Nav2TwoGoalModel9996FullActorLateralFinalRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996FullActorLateralRslRlOnPolicyRunnerAmpCfg
):
    """Small safe-set updates for the final lateral response/leak trade-off."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.clip_param = 0.08
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 1.0e-4


@configclass
class G1Nav2TwoGoalModel9996FullActorLateralRobustRslRlOnPolicyRunnerAmpCfg(
    G1Nav2TwoGoalModel9996FullActorLateralFinalRslRlOnPolicyRunnerAmpCfg
):
    """Conservative randomized safety polish for the gated lateral expert."""

    experiment_name = "g1_amp_nav2_two_goal_model9996_full_actor_lateral_robust"
    checkpoint_output_dir = "Nav2TwoGoalModel9996FullActorLateralRobust"
    save_interval = 1

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 2.5e-6
        self.algorithm.clip_param = 0.05
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 5.0e-5
