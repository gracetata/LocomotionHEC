"""Target-lock V6 task: recover from larger impulses, then converge to rest."""

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
class G1ExtremeStandRecoveryTargetLockV6EnvCfg(
    G1ExtremeStandRecoverySmoothSettleV5EnvCfg
):
    """V6 fixes V5 target drift and covers the measured large-push impulse."""

    def __post_init__(self):
        super().__post_init__()
        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )

        # 240 N x 0.30 s reaches the 72 N.s impulse used by the 360 N/0.20 s
        # stress test.  Limb scaling prevents that torso-level load from being
        # unrealistically copied to every distal link.
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
                "quiet_duration_range_s": (8.0, 12.0),
                "initial_quiet_range_s": (1.0, 2.0),
                "force_magnitudes_n": (45.0, 90.0, 150.0, 240.0),
                # With 24 policy steps per PPO iteration these stages begin at
                # V6 iterations 0, 200, 400 and 700.
                "stage_step_thresholds": (4800, 9600, 16800),
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

        # Keep broad recovery gradients, then make the final default basin
        # materially more valuable than V5's offset standing solution.
        self.rewards.default_joint_pose_exp.weight = 8.0
        self.rewards.default_leg_joint_pose_exp.weight = 5.0
        self.rewards.default_key_body_pose_exp.weight = 2.0
        self.rewards.default_key_body_pose_gaussian.weight = 10.0
        self.rewards.default_feet_distance_l2.weight = -12.0
        self.rewards.default_feet_distance_gaussian.weight = 5.0

        # The global target cost remains mild enough for large-error recovery;
        # the new gated lock becomes strong only in the recovered basin.
        self.rewards.target_q_default_error_l2.weight = -8.0
        self.rewards.near_default_target_lock_penalty = RewTerm(
            func=recovery_rewards.near_default_target_lock_penalty,
            weight=-1.0,
            params={
                "pose_scale": 0.0225,
                "target_weight": 30.0,
                "action_weight": 5.0,
                "joint_velocity_weight": 5.0,
                "action_term_name": "joint_pos",
                "asset_cfg": full_joint_cfg,
            },
        )

        # Replace V5's effectively zero Gaussian product with bounded rational
        # scores that retain gradient after a large reset or impulse.
        self.rewards.post_disturbance_pose_recovery = RewTerm(
            func=recovery_rewards.post_disturbance_pose_recovery_rational,
            weight=4.0,
            params={
                "event_name": "single_disturbance",
                "pose_scale": 0.0225,
                "asset_cfg": full_joint_cfg,
            },
        )
        self.rewards.post_disturbance_stillness = RewTerm(
            func=recovery_rewards.post_disturbance_stillness_rational,
            weight=4.0,
            params={
                "event_name": "single_disturbance",
                "pose_scale": 0.0225,
                "velocity_scale": 0.25,
                "asset_cfg": full_joint_cfg,
            },
        )


@configclass
class G1ExtremeStandRecoveryTargetLockV6EnvCfg_PLAY(
    G1ExtremeStandRecoveryTargetLockV6EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
