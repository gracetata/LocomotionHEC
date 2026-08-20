"""Recovery-preserving V7: clean anchors plus near-default Top-K target lock."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import disturbances
from . import rewards as recovery_rewards
from .g1_extreme_stand_recovery_env_cfg import (
    G1_LOCOMOTION_JOINT_NAMES,
    PERTURBED_BODY_NAMES,
)
from .g1_extreme_stand_recovery_v5_env_cfg import (
    G1ExtremeStandRecoverySmoothSettleV5EnvCfg,
)


@configclass
class G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg(
    G1ExtremeStandRecoverySmoothSettleV5EnvCfg
):
    """Smooth a V4 policy without forgetting nominal and pose recovery.

    Half of the environments never receive an external wrench.  They still
    start from randomized joints and therefore train recovery followed by a
    long, disturbance-free hold.  The other half use mutually-exclusive short
    impulses.  Global physical-target derivative costs from V5 are removed;
    they inhibited decisive recovery.  Instead, a Top-K target lock turns on
    only as the worst joints approach the default pose.
    """

    def __post_init__(self):
        super().__post_init__()
        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )

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
                "active_duration_range_s": (0.10, 0.25),
                "quiet_duration_range_s": (8.0, 12.0),
                "initial_quiet_range_s": (2.0, 4.0),
                "force_magnitudes_n": (20.0, 45.0, 90.0, 180.0),
                # 24 control steps per PPO iteration: stages 0/150/300/450.
                "stage_step_thresholds": (3600, 7200, 10800),
                "enabled_env_fraction": 0.50,
                # pelvis, torso, shoulders, elbows, hips, knees
                "body_force_scales": (
                    1.0,
                    1.0,
                    0.50,
                    0.50,
                    0.35,
                    0.35,
                    0.65,
                    0.65,
                    0.50,
                    0.50,
                ),
                "direction_probabilities": (0.15, 0.30, 0.20, 0.35),
            },
        )

        # V5's global physical-target velocity/acceleration penalties also
        # applied while the robot was far from default and weakened recovery.
        # Keep only a very small global action magnitude prior; exact target
        # locking is state-dependent below.
        self.rewards.action_l2.weight = -0.005
        self.rewards.target_q_default_error_l2 = None
        self.rewards.target_q_velocity_l2 = None
        self.rewards.target_q_acceleration_l2 = None

        # Conservative smoothness coefficients retain V4's ability to make a
        # decisive recovery step while still exposing the worst four joints.
        self.rewards.normalized_joint_torque_topk_l2.weight = -0.05
        self.rewards.normalized_joint_torque_rate_topk_l2.weight = -5.0e-5
        self.rewards.soft_peak_joint_torque_topk_l2.weight = -1.0
        self.rewards.normalized_joint_jerk_topk_l2.weight = -0.08
        self.rewards.joint_velocity_l2.weight = -0.02
        self.rewards.joint_acceleration_l2.weight = -5.0e-7
        self.rewards.mechanical_power_l2.weight = -5.0e-5
        self.rewards.near_default_settle_penalty.weight = -0.25

        self.rewards.near_default_topk_target_lock_penalty = RewTerm(
            func=recovery_rewards.near_default_topk_target_lock_penalty,
            weight=-1.0,
            params={
                "pose_scale": 0.04,
                "gate_power": 2.0,
                "topk": 4,
                "target_weight": 20.0,
                "action_weight": 4.0,
                "joint_velocity_weight": 4.0,
                "action_term_name": "joint_pos",
                "asset_cfg": full_joint_cfg,
            },
        )

        # Rational quiet-window scores remain useful after large errors, but
        # are deliberately less dominant than in V6.
        self.rewards.post_disturbance_pose_recovery = RewTerm(
            func=recovery_rewards.post_disturbance_pose_recovery_rational,
            weight=2.0,
            params={
                "event_name": "single_disturbance",
                "pose_scale": 0.04,
                "asset_cfg": full_joint_cfg,
            },
        )
        self.rewards.post_disturbance_stillness = RewTerm(
            func=recovery_rewards.post_disturbance_stillness_rational,
            weight=2.0,
            params={
                "event_name": "single_disturbance",
                "pose_scale": 0.04,
                "velocity_scale": 0.25,
                "asset_cfg": full_joint_cfg,
            },
        )


@configclass
class G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg_PLAY(
    G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
