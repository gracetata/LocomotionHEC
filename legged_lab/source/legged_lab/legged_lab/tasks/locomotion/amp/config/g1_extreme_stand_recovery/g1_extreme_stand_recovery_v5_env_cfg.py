"""Smooth-settle V5 task: recover decisively, then stop moving."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp

from . import disturbances
from . import rewards as recovery_rewards
from .g1_extreme_stand_recovery_env_cfg import (
    G1_LOCOMOTION_JOINT_NAMES,
    PERTURBED_BODY_NAMES,
    G1ExtremeStandRecoveryEnvCfg,
)


@configclass
class G1ExtremeStandRecoverySmoothSettleV5EnvCfg(G1ExtremeStandRecoveryEnvCfg):
    """V5 with target settling, Top-K normalization and non-overlapping pushes."""

    def __post_init__(self):
        super().__post_init__()
        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )

        # One state machine replaces five independent clocks.  A 0.1--0.3 s
        # impulse is always followed by 6--10 s in which no new push can occur.
        self.events.random_torso_external_wrench = None
        self.events.random_pelvis_external_wrench = None
        self.events.random_arm_external_wrench = None
        self.events.random_leg_external_wrench = None
        self.events.push_robot = None
        self.events.single_disturbance = EventTerm(
            func=disturbances.single_body_force_curriculum,
            mode="interval",
            interval_range_s=(0.10, 0.10),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=PERTURBED_BODY_NAMES, preserve_order=True
                ),
                "tick_s": 0.10,
                "active_duration_range_s": (0.10, 0.30),
                "quiet_duration_range_s": (6.0, 10.0),
                "initial_quiet_range_s": (1.0, 2.0),
                "force_magnitudes_n": (10.0, 20.0, 36.0, 45.0),
                # PPO uses 24 control steps per iteration: for the 1500-iteration
                # run, stages start at iterations 0, 300, 600 and 1000.
                "stage_step_thresholds": (7200, 14400, 24000),
                # forward, backward, left, right
                "direction_probabilities": (0.15, 0.30, 0.20, 0.35),
            },
        )

        # Replace raw 29-joint means with normalized Top-K penalties so a
        # single knee/hip spike remains visible to PPO.
        self.rewards.dof_torques_l2 = None
        self.rewards.joint_jerk_l2 = None
        self.rewards.joint_torque_rate_l2 = None
        self.rewards.normalized_joint_torque_topk_l2 = RewTerm(
            func=recovery_rewards.normalized_joint_torque_topk_l2,
            weight=-0.10,
            params={"topk": 4, "asset_cfg": full_joint_cfg},
        )
        self.rewards.normalized_joint_torque_rate_topk_l2 = RewTerm(
            func=recovery_rewards.normalized_joint_torque_rate_topk_l2,
            weight=-1.0e-4,
            params={"topk": 4, "asset_cfg": full_joint_cfg},
        )
        self.rewards.soft_peak_joint_torque_topk_l2 = RewTerm(
            func=recovery_rewards.soft_peak_joint_torque_topk_l2,
            weight=-2.0,
            params={"soft_ratio": 0.60, "topk": 4, "asset_cfg": full_joint_cfg},
        )
        self.rewards.normalized_joint_jerk_topk_l2 = RewTerm(
            func=recovery_rewards.normalized_joint_jerk_topk_l2,
            weight=-0.15,
            params={
                "topk": 4,
                "leg_scale": 5000.0,
                "waist_scale": 3000.0,
                "arm_scale": 3000.0,
                "asset_cfg": full_joint_cfg,
            },
        )

        # Constrain both the dimensionless policy output and the physical PD
        # target.  These close the loophole in which q stays near default while
        # the policy slowly moves its target back and forth.
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.02)
        self.rewards.target_q_default_error_l2 = RewTerm(
            func=recovery_rewards.target_q_default_error_l2,
            weight=-5.0,
            params={"action_term_name": "joint_pos", "asset_cfg": full_joint_cfg},
        )
        self.rewards.target_q_velocity_l2 = RewTerm(
            func=recovery_rewards.target_q_velocity_l2,
            weight=-0.02,
            params={"action_term_name": "joint_pos"},
        )
        self.rewards.target_q_acceleration_l2 = RewTerm(
            func=recovery_rewards.target_q_acceleration_l2,
            weight=-1.0e-4,
            params={"action_term_name": "joint_pos"},
        )

        self.rewards.joint_velocity_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.05,
            params={"asset_cfg": full_joint_cfg},
        )
        self.rewards.dof_acc_l2 = None
        self.rewards.joint_acceleration_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-1.0e-6,
            params={"asset_cfg": full_joint_cfg},
        )
        self.rewards.mechanical_power_l2 = RewTerm(
            func=recovery_rewards.mechanical_power_l2,
            weight=-1.0e-4,
            params={"asset_cfg": full_joint_cfg},
        )
        self.rewards.near_default_settle_penalty = RewTerm(
            func=recovery_rewards.near_default_settle_penalty,
            weight=-1.0,
            params={
                "variance": 0.01,
                "joint_velocity_weight": 5.0,
                "action_weight": 5.0,
                "torque_rate_weight": 3.0,
                "jerk_weight": 3.0,
                "topk": 4,
                "leg_jerk_scale": 5000.0,
                "waist_jerk_scale": 3000.0,
                "arm_jerk_scale": 3000.0,
                "asset_cfg": full_joint_cfg,
            },
        )

        # The quiet-window rewards make recovery time explicit and pay the
        # policy for remaining motionless after it has recovered.
        self.rewards.post_disturbance_pose_recovery = RewTerm(
            func=recovery_rewards.post_disturbance_pose_recovery,
            weight=2.0,
            params={
                "event_name": "single_disturbance",
                "variance": 0.01,
                "asset_cfg": full_joint_cfg,
            },
        )
        self.rewards.post_disturbance_stillness = RewTerm(
            func=recovery_rewards.post_disturbance_stillness,
            weight=2.0,
            params={
                "event_name": "single_disturbance",
                "pose_variance": 0.01,
                "velocity_variance": 0.04,
                "asset_cfg": full_joint_cfg,
            },
        )


@configclass
class G1ExtremeStandRecoverySmoothSettleV5EnvCfg_PLAY(
    G1ExtremeStandRecoverySmoothSettleV5EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
