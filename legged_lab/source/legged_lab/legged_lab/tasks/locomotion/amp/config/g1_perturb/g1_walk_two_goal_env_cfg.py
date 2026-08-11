"""ArmHack Walk experts for safe lateral stepping and zero-linear pure yaw.

The two skills are trained independently because they require different
contact-sequence corrections.  ``G1WalkPerturbAmpEnv`` keeps one of the three
tracked arm poses fixed for the full episode, so the experts learn the lower-
body compensation with the arm hijack active rather than receiving it only at
deployment time.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1_FOOT_BODY_NAMES,
    G1_NAV2_TWO_GOAL_LATERAL_ONLY_MODE_CONFIG_PATH,
    G1_NAV2_TWO_GOAL_YAW_ONLY_MODE_CONFIG_PATH,
)

from .g1_walk_behavior_env_cfg import G1WalkBehaviorFinetuneEnvCfg


def _configure_shape_aware_safety(cfg, *, hard_clearance: float, hard_weight: float) -> None:
    """Install a swept oriented-sole safe set shared by both experts."""
    feet = SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True)
    common = {
        "command_name": "base_velocity",
        "asset_cfg": feet,
        "center_offset_x": 0.035,
        "half_length": 0.090,
        "half_width": 0.035,
        "soft_clearance": 0.065,
        "hard_clearance": hard_clearance,
        "hard_scale": 1.0,
        "overlap_scale": 16.0,
        "soft_max_penalty": 100.0,
        "interpolation_steps": 12,
    }
    cfg.rewards.oriented_footprint_proximity_l2 = None
    cfg.rewards.swept_oriented_footprint_soft_margin_l2 = RewTerm(
        func=mdp.swept_oriented_footprint_proximity_l2,
        weight=-3.0,
        params={**common, "component": "soft"},
    )
    cfg.rewards.swept_oriented_footprint_hard_barrier = RewTerm(
        func=mdp.swept_oriented_footprint_proximity_l2,
        weight=hard_weight,
        params={**common, "component": "hard"},
    )


def _configure_expert_common(cfg) -> None:
    cfg.upper_body_perturbation.pose_name = "random"
    cfg.commands.base_velocity.mode_probability = 1.0
    cfg.commands.base_velocity.hard_zero_stand = True
    cfg.commands.base_velocity.reset_command_to_zero = False
    cfg.commands.base_velocity.smoothing_time_constant = 0.0
    cfg.commands.base_velocity.max_linear_accel = 100.0
    cfg.commands.base_velocity.max_yaw_accel = 100.0
    cfg.rewards.feet_air_time = None
    cfg.rewards.strict_zero_body_motion_l2 = None
    cfg.rewards.strict_zero_feet_motion_l2 = None
    cfg.rewards.strict_zero_joint_vel_l2 = None
    cfg.rewards.strict_zero_double_support = None
    cfg.rewards.rapid_footstep_l1.weight = -0.25
    cfg.rewards.nonzero_single_stance.weight = 1.5
    cfg.rewards.two_goal_signed_root_response = RewTerm(
        func=mdp.two_goal_signed_root_response,
        weight=80.0,
        params={"command_name": "base_velocity"},
    )
    cfg.rewards.two_goal_response_shortfall = RewTerm(
        func=mdp.two_goal_response_shortfall,
        weight=-300.0,
        params={
            "command_name": "base_velocity",
            "target_fraction": 0.75,
            "max_penalty": 1.0,
        },
    )


@configclass
class G1WalkTwoGoalLateralExpertEnvCfg(G1WalkBehaviorFinetuneEnvCfg):
    """Strict ``[0, vy, 0]`` ArmHack expert with a 25-mm hard sole set."""

    def __post_init__(self):
        super().__post_init__()
        _configure_expert_common(self)
        self.commands.base_velocity.mode_sampling_config_path = (
            G1_NAV2_TWO_GOAL_LATERAL_ONLY_MODE_CONFIG_PATH
        )
        self.commands.base_velocity.mode_command_clip_min = (0.0, -0.35, 0.0)
        self.commands.base_velocity.mode_command_clip_max = (0.0, 0.35, 0.0)
        self.rewards.two_goal_signed_root_response.weight = 100.0
        self.rewards.lateral_command_leak_l2 = RewTerm(
            func=mdp.lateral_command_leak_l2,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "min_lateral_command": 0.10,
                "forward_velocity_scale": 0.040,
                "yaw_rate_scale": 0.060,
                "max_penalty": 100.0,
            },
        )
        self.rewards.lateral_foot_ordering_l2 = RewTerm(
            func=mdp.lateral_foot_ordering_l2,
            weight=-30.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
                ),
                "foot_half_width": 0.035,
                "min_clearance": 0.050,
                "shortfall_scale": 0.040,
                "max_penalty": 100.0,
                "min_lateral_command": 0.10,
            },
        )
        _configure_shape_aware_safety(self, hard_clearance=0.025, hard_weight=-150.0)


@configclass
class G1WalkTwoGoalLateralRobustEnvCfg(G1WalkTwoGoalLateralExpertEnvCfg):
    """Move an already responsive lateral gait into a 30-mm randomized safe set."""

    def __post_init__(self):
        super().__post_init__()
        _configure_shape_aware_safety(self, hard_clearance=0.030, hard_weight=-300.0)
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.55, 1.25),
                "dynamic_friction_range": (0.50, 1.10),
                "restitution_range": (0.0, 0.10),
                "num_buckets": 64,
                "make_consistent": True,
            }
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.randomize_rigid_body_com.params["com_range"] = {
            "x": (-0.015, 0.015),
            "y": (-0.015, 0.015),
            "z": (-0.010, 0.010),
        }
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.90, 1.10)
        self.events.scale_actuator_gains.params.update(
            {
                "stiffness_distribution_params": (0.90, 1.10),
                "damping_distribution_params": (0.90, 1.10),
            }
        )
        self.events.scale_joint_parameters.params.update(
            {
                "friction_distribution_params": (0.80, 1.20),
                "armature_distribution_params": (0.90, 1.10),
            }
        )
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "yaw": (-0.30, 0.30),
        }


@configclass
class G1WalkTwoGoalYawExpertEnvCfg(G1WalkBehaviorFinetuneEnvCfg):
    """Strict zero-linear pure-yaw ArmHack expert with real root-turn rewards."""

    def __post_init__(self):
        super().__post_init__()
        _configure_expert_common(self)
        self.commands.base_velocity.mode_sampling_config_path = (
            G1_NAV2_TWO_GOAL_YAW_ONLY_MODE_CONFIG_PATH
        )
        self.commands.base_velocity.mode_command_clip_min = (0.0, 0.0, -0.45)
        self.commands.base_velocity.mode_command_clip_max = (0.0, 0.0, 0.45)
        self.rewards.two_goal_signed_root_response.weight = 100.0
        self.rewards.two_goal_response_shortfall.weight = -400.0
        self.rewards.pure_yaw_planar_drift_l2 = RewTerm(
            func=mdp.pure_yaw_planar_drift_l2,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.10,
                "velocity_scale": 0.035,
                "max_penalty": 100.0,
            },
        )
        self.rewards.pure_yaw_root_rate_error_l2 = RewTerm(
            func=mdp.pure_yaw_root_rate_error_l2,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.10,
                "error_scale": 0.08,
                "max_penalty": 100.0,
            },
        )
        _configure_shape_aware_safety(self, hard_clearance=0.025, hard_weight=-100.0)


@configclass
class G1WalkTwoGoalYawRobustEnvCfg(G1WalkTwoGoalYawExpertEnvCfg):
    """Pure-yaw expert with the same 30-mm safe set used by lateral polish."""

    def __post_init__(self):
        super().__post_init__()
        _configure_shape_aware_safety(self, hard_clearance=0.030, hard_weight=-200.0)
