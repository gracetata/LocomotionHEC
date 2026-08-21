"""Phase-two hold refinement for adaptive ArmHack Stand."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_stand_adaptive_switch_env_cfg import G1StandAdaptiveSwitchEnvCfg


@configclass
class G1StandAdaptiveHoldEnvCfg(G1StandAdaptiveSwitchEnvCfg):
    """Oversample completed states and suppress the observed 1-Hz stepping loop."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot_joints.params["phase_one_probability"] = 0.15
        self.events.reset_robot_joints.params["phase_two_probability"] = 0.45
        self.events.reset_robot_joints.params["final_distance"] = 0.35
        for reward_name in (
            "sequential_foot_step_progress",
            "sequential_foot_step_target_exp",
            "sequential_foot_step_clearance_exp",
            "sequential_active_foot_contact",
            "sequential_active_foot_clearance_l2",
            "sequential_active_foot_upward_velocity",
            "sequential_active_foot_velocity_l2",
            "sequential_active_foot_single_support",
            "sequential_foot_step_landing_exp",
            "sequential_foot_step_completion",
            "sequential_foot_step_lift",
            "sequential_foot_step_order_violation",
            "sequential_foot_final_target_l2",
            "sequential_final_ankle_distance_exp",
            "sequential_support_foot_drift_l2",
        ):
            getattr(self.rewards, reward_name).params["lateral_target_offset_m"] = 0.175
        self.rewards.sequential_final_ankle_distance_exp.params["target_distance_m"] = 0.35
        for reward_name in (
            "ankle_distance_l1",
            "ankle_distance_exp",
            "ankle_distance_success",
            "ankle_torques_l2",
            "torso_xy_position_near_stance_l2",
            "torso_yaw_near_stance_l2",
        ):
            getattr(self.rewards, reward_name).params["target_distance"] = 0.35
        pelvis_cfg = SceneEntityCfg("robot", body_names="pelvis")
        foot_cfg = SceneEntityCfg(
            "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        sensor_cfg = SceneEntityCfg(
            "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        params = {
            "pelvis_cfg": pelvis_cfg,
            "foot_cfg": foot_cfg,
            "sensor_cfg": sensor_cfg,
            "lateral_target_offset_m": 0.175,
            "min_clearance_m": 0.035,
            "landing_tolerance_m": 0.055,
            "min_step_duration_s": 0.35,
        }
        self.rewards.post_complete_foot_velocity_l2 = RewTerm(
            func=mdp.sequential_post_complete_foot_velocity_l2, weight=-35.0, params=params
        )
        self.rewards.post_complete_contact_loss = RewTerm(
            func=mdp.sequential_post_complete_contact_loss, weight=-30.0, params=params
        )
        self.rewards.post_complete_target_l2 = RewTerm(
            func=mdp.sequential_post_complete_target_l2, weight=-180.0, params=params
        )
        self.rewards.double_support.weight = 0.25
        self.rewards.feet_slide.weight = -0.25
        self.rewards.ankle_torques_l2.weight = -1.5e-3
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
                        "force_range": (-25.0, 25.0),
                        "torque_range": (-2.0, 2.0),
                    },
                ),
            )
