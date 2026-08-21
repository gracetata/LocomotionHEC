"""Pure-yaw refinement with torso posture and wrist-wrench randomization."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_walk_ankle_spacing_env_cfg import G1WalkAnkleSpacingYawEnvCfg


@configclass
class G1WalkYawForceRobustEnvCfg(G1WalkAnkleSpacingYawEnvCfg):
    """Reduce pure-yaw translation/back-lean under randomized end wrenches."""

    def __post_init__(self):
        super().__post_init__()
        # Keep the translation guard stronger than the spacing objective.  The
        # previous -5/500 ratio learned to buy ankle width with planar drift.
        self.rewards.pure_yaw_planar_drift_l2.weight = -50.0
        self.rewards.pure_yaw_planar_drift_l2.params["max_penalty"] = 100.0
        self.rewards.ankle_distance_30cm_kernel.weight = 120.0
        self.rewards.torso_roll_pitch_l2.weight = -3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
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
