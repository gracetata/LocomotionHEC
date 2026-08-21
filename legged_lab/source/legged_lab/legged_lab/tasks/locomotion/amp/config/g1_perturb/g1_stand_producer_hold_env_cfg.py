"""Phase-two Stand hold trained from real policy-handoff states."""

from __future__ import annotations

from isaaclab.utils import configclass

from .g1_stand_adaptive_hold_env_cfg import G1StandAdaptiveHoldEnvCfg
from .g1_walk_perturb_env_cfg import (
    G1_WALK_PERTURB_POSE_NAMES,
    G1_WALK_PERTURB_POSE_SET,
)


@configclass
class G1StandProducerHoldEnvCfg(G1StandAdaptiveHoldEnvCfg):
    """Learn active double-support stabilization at the measured touchdown state."""

    def __post_init__(self):
        super().__post_init__()
        front_index = G1_WALK_PERTURB_POSE_NAMES.index("pos3_front")
        perturbation = self.upper_body_perturbation
        perturbation.source = "pose_set"
        perturbation.pose_set = [G1_WALK_PERTURB_POSE_SET[front_index]]
        perturbation.pose_probabilities = [1.0]
        perturbation.producer_state_probability = 1.0
        perturbation.producer_joint_position_noise = 0.01
        perturbation.producer_joint_velocity_noise = 0.02

        reset_params = self.events.reset_robot_joints.params
        reset_params["phase_one_probability"] = 0.0
        reset_params["phase_two_probability"] = 1.0
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
        ):
            getattr(self.rewards, reward_name).weight = 0.0
        self.rewards.post_complete_foot_velocity_l2.weight = -45.0
        self.rewards.post_complete_contact_loss.weight = -45.0
        self.rewards.post_complete_target_l2.weight = -220.0
        self.rewards.double_support.weight = 0.75
        self.rewards.torso_xy_position_near_stance_l2.weight = -45.0
        self.rewards.torso_yaw_near_stance_l2.weight = -30.0
