"""Support-gated V8: preserve recovery while eliminating distal target drift."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import disturbances
from . import rewards as recovery_rewards
from .g1_extreme_stand_recovery_env_cfg import (
    G1_ARM_JOINT_NAMES,
    G1_LEG_JOINT_NAMES,
    G1_LOCOMOTION_JOINT_NAMES,
    G1_WAIST_JOINT_NAMES,
    PERTURBED_BODY_NAMES,
)
from .g1_extreme_stand_recovery_v7_env_cfg import (
    G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg,
)


@configclass
class G1ExtremeStandRecoverySupportLockV8EnvCfg(
    G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg
):
    """Use support stability, rather than full-body error, to enable target lock.

    V7 allowed a displaced wrist to suppress the same target-lock reward that
    should pull it home.  V8 gates on legs and waist only, but applies the
    Top-K cost to all 29 physical PD targets.  Seventy-five percent clean
    anchor environments protect nominal and randomized-pose recovery while a
    conservative force curriculum retains robustness.
    """

    def __post_init__(self):
        super().__post_init__()
        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )
        support_joint_cfg = SceneEntityCfg(
            "robot",
            joint_names=G1_LEG_JOINT_NAMES + G1_WAIST_JOINT_NAMES,
            preserve_order=True,
        )
        arm_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_ARM_JOINT_NAMES, preserve_order=True
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
                # 24 control steps per PPO iteration: stages 0/100/200/300.
                "stage_step_thresholds": (2400, 4800, 7200),
                "enabled_env_fraction": 0.25,
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

        # Remove V7's self-disabling full-body gate.  A small always-on Top-K
        # prior prevents distal targets from finding a new offset equilibrium.
        self.rewards.near_default_topk_target_lock_penalty = None
        self.rewards.target_q_default_topk_l2 = RewTerm(
            func=recovery_rewards.target_q_default_topk_l2,
            weight=-0.20,
            params={
                "topk": 4,
                "action_term_name": "joint_pos",
                "asset_cfg": full_joint_cfg,
            },
        )
        self.rewards.support_stable_topk_target_lock_penalty = RewTerm(
            func=recovery_rewards.support_stable_topk_target_lock_penalty,
            weight=-1.0,
            params={
                "support_pose_scale": 0.0225,
                "gate_power": 2.0,
                "topk": 4,
                "target_weight": 40.0,
                "action_weight": 6.0,
                "joint_velocity_weight": 4.0,
                "action_term_name": "joint_pos",
                "asset_cfg": full_joint_cfg,
                "support_asset_cfg": support_joint_cfg,
            },
        )

        # Cartesian key points weakly constrain wrist orientation.  This
        # explicit arm-space reward closes that gap without exposing any future
        # information or overriding the actor's 29 outputs.
        self.rewards.default_arm_joint_pose_exp = RewTerm(
            func=recovery_rewards.default_joint_pose_exp,
            weight=5.0,
            params={"std": 0.20, "asset_cfg": arm_joint_cfg},
        )


@configclass
class G1ExtremeStandRecoverySupportLockV8EnvCfg_PLAY(
    G1ExtremeStandRecoverySupportLockV8EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
