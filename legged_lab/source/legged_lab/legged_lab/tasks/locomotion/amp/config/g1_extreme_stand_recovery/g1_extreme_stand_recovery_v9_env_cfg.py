"""Dynamic-stability V9: joint-independent target locking with V4 retention."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import disturbances
from . import rewards as recovery_rewards
from .g1_extreme_stand_recovery_env_cfg import (
    G1_LOCOMOTION_JOINT_NAMES,
    G1_WAIST_JOINT_NAMES,
    PERTURBED_BODY_NAMES,
)
from .g1_extreme_stand_recovery_v8_env_cfg import (
    G1ExtremeStandRecoverySupportLockV8EnvCfg,
)


@configclass
class G1ExtremeStandRecoveryDynamicLockV9EnvCfg(
    G1ExtremeStandRecoverySupportLockV8EnvCfg
):
    """Prevent every joint from disabling its own default-target correction."""

    def __post_init__(self):
        super().__post_init__()
        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )
        waist_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_WAIST_JOINT_NAMES, preserve_order=True
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
                # 24 control steps per PPO iteration: stages 0/50/100/150.
                "stage_step_thresholds": (1200, 2400, 3600),
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

        self.rewards.support_stable_topk_target_lock_penalty = None
        self.rewards.target_q_default_topk_l2.weight = -0.05
        self.rewards.dynamically_stable_topk_target_lock_penalty = RewTerm(
            func=recovery_rewards.dynamically_stable_topk_target_lock_penalty,
            weight=-1.0,
            params={
                "upright_scale": 0.01,
                "linear_velocity_scale": 0.09,
                "angular_velocity_scale": 0.25,
                "gate_power": 1.0,
                "topk": 4,
                "target_weight": 30.0,
                "action_weight": 6.0,
                "joint_velocity_weight": 6.0,
                "action_term_name": "joint_pos",
                "asset_cfg": full_joint_cfg,
            },
        )
        self.rewards.default_waist_joint_pose_exp = RewTerm(
            func=recovery_rewards.default_joint_pose_exp,
            weight=5.0,
            params={"std": 0.15, "asset_cfg": waist_joint_cfg},
        )


@configclass
class G1ExtremeStandRecoveryDynamicLockV9EnvCfg_PLAY(
    G1ExtremeStandRecoveryDynamicLockV9EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
