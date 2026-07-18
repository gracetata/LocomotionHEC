"""P0/P1 robust Walk ArmHack task, isolated from all Stand configurations."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1_COMMAND_BALANCED_TASK_SAMPLING_CONFIG_PATH,
)

from .g1_walk_perturb_env_cfg import (
    G1WalkPerturbFinetuneEnvCfg,
    G1_WALK_PERTURB_NAV2_COMMAND_PATH,
)


@configclass
class G1WalkRobustFinetuneEnvCfg(G1WalkPerturbFinetuneEnvCfg):
    """Walk fine-tuning with lower-body AMP, hybrid commands and wrist payload DR.

    The inherited perturbation contract masks the scripted arms from AMP and
    lower-body regularizers.  This class is Walk-only: Stand neither imports it
    nor shares any mutable configuration object with it.
    """

    def __post_init__(self):
        super().__post_init__()

        # Start conservatively on the verified Nav2 complex-turn distribution.
        # Later phases alter only the explicit scale/filter/source fields below.
        self.commands.base_velocity = mdp.HybridNav2ModeVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            data_path=G1_WALK_PERTURB_NAV2_COMMAND_PATH,
            augmentation_filter="none,mirror_lr",
            synthesize_mirror_lr=True,
            scenario_family_filter="complex_turn",
            dataset_sample_dt=0.05,
            window_duration_s=4.0,
            command_scale=(0.85, 0.75, 0.75),
            command_clip_min=(-0.65, -0.45, -1.0),
            command_clip_max=(1.05, 0.45, 1.0),
            smoothing_time_constant=0.30,
            max_linear_accel=0.60,
            max_yaw_accel=0.80,
            reset_command_to_zero=True,
            sample_groups_uniformly=True,
            scenario_family_weights={
                "sample500_random": 1.0,
                "sharp_turn": 2.4,
                "hard_turn": 2.0,
                "complex_turn": 1.6,
                "moving_obstacle": 1.3,
                "random_obstacle": 1.0,
                "baseline": 0.7,
                "large_success": 0.8,
            },
            controller_weights={"mppi": 1.5, "dwb": 1.0},
            augmentation_weights={"none": 1.0, "mirror_lr": 1.0},
            mode_sampling_config_path=G1_COMMAND_BALANCED_TASK_SAMPLING_CONFIG_PATH,
            mode_probability=0.0,
            forced_mode="",
            mode_command_scale=(0.75, 0.75, 0.75),
            mode_command_clip_min=(-0.65, -0.45, -1.0),
            mode_command_clip_max=(1.05, 0.45, 1.0),
            ranges=mdp.HybridNav2ModeVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.65, 1.05),
                lin_vel_y=(-0.45, 0.45),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        )

        # Each terminal link receives an independent U(0, 1 kg) added payload.
        # Recomputing inertia avoids a mass/inertia-inconsistent rigid body.
        self.events.randomize_left_end_effector_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_wrist_yaw_link"],
                    preserve_order=True,
                ),
                "mass_distribution_params": (0.0, 1.0),
                "operation": "add",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_right_end_effector_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["right_wrist_yaw_link"],
                    preserve_order=True,
                ),
                "mass_distribution_params": (0.0, 1.0),
                "operation": "add",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )


@configclass
class G1WalkRobustFinetuneEnvCfg_PLAY(G1WalkRobustFinetuneEnvCfg):
    """Deterministic small-env variant for Walk tests and visualization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        # Evaluation chooses an exact payload via the launcher. Nominal is 0 kg.
        self.events.randomize_left_end_effector_payload.params["mass_distribution_params"] = (
            0.0,
            0.0,
        )
        self.events.randomize_right_end_effector_payload.params["mass_distribution_params"] = (
            0.0,
            0.0,
        )
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.push_robot = None
