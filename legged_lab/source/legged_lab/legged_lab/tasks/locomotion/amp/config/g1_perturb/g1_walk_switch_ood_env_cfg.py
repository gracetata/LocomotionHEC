"""Walk refinement from Stand-like terminal stance states."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_walk_precision_switch_env_cfg import G1WalkPrecisionSwitchEnvCfg


@configclass
class G1WalkSwitchOodEnvCfg(G1WalkPrecisionSwitchEnvCfg):
    """Teach Walk to accept the wide, nearly-static state produced by Stand.

    The reset mixture approximates both left/right Stand completion states by
    randomizing around the 35-cm ankle geometry and small handoff velocities.
    Commands still start at exact zero before following the precision ramp.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_with_random_stance,
            mode="reset",
            params={
                "distance_range": (0.24, 0.38),
                "close_distance_range": (0.24, 0.27),
                "close_stance_probability": 0.15,
                "nominal_distance_range": (0.28, 0.33),
                "nominal_stance_probability": 0.55,
                "asymmetric_support_probability": 0.0,
                "phase_one_probability": 0.0,
                "phase_two_probability": 0.20,
                "support_distance_range": (0.28, 0.38),
                "final_distance": 0.35,
                "kinematic_nominal_distance": 0.237,
                "kinematic_distance_per_rad": 1.22,
                "position_scale_range": (0.99, 1.01),
                "velocity_range": (-0.05, 0.05),
            },
        )
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.03, 0.03),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-0.10, 0.10),
        }
        self.rewards.ankle_distance_30cm_kernel.weight = 120.0
        self.rewards.torso_roll_pitch_l2.weight = -3.0
        self.rewards.pure_yaw_planar_drift_l2 = RewTerm(
            func=mdp.pure_yaw_planar_drift_l2,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "min_yaw_command": 0.10,
                "velocity_scale": 0.08,
                "max_penalty": 100.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.events.randomize_torso_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "force_range": (-40.0, 40.0),
                "torque_range": (-3.0, 3.0),
            },
        )

        for side in ("left", "right"):
            setattr(
                self.events,
                f"randomize_{side}_wrist_wrench",
                EventTerm(
                    func=mdp.apply_external_force_torque,
                    mode="interval",
                    interval_range_s=(2.5, 5.0),
                    is_global_time=False,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=f"{side}_wrist_yaw_link"),
                        "force_range": (-20.0, 20.0),
                        "torque_range": (-1.5, 1.5),
                    },
                ),
            )
