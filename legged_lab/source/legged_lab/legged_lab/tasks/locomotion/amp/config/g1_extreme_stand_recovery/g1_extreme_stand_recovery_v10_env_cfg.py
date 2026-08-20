"""V10: conservative actor refinement with stable-state target smoothing."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import rewards as recovery_rewards
from .g1_extreme_stand_recovery_env_cfg import G1_LOCOMOTION_JOINT_NAMES
from .g1_extreme_stand_recovery_v9_env_cfg import (
    G1ExtremeStandRecoveryDynamicLockV9EnvCfg,
)


@configclass
class G1ExtremeStandRecoveryConservativeSmoothV10EnvCfg(
    G1ExtremeStandRecoveryDynamicLockV9EnvCfg
):
    """Penalize closed-loop target curvature in the settled basin.

    V9 preserved final pose recovery but its first actor update introduced
    intermittent wrist-target spikes, followed by a whole-body 1.2 Hz limit
    cycle.  This task keeps the joint-independent V9 lock and adds a direct
    normalized Top-K penalty on target velocity/acceleration and action first/
    second differences.  Root dynamics, never joint pose, gate the term.
    """

    def __post_init__(self):
        super().__post_init__()
        full_joint_cfg = SceneEntityCfg(
            "robot", joint_names=G1_LOCOMOTION_JOINT_NAMES, preserve_order=True
        )
        self.rewards.dynamically_stable_target_smoothness_topk_penalty = RewTerm(
            func=recovery_rewards.dynamically_stable_target_smoothness_topk_penalty,
            weight=-1.0,
            params={
                "upright_scale": 0.01,
                "linear_velocity_scale": 0.09,
                "angular_velocity_scale": 0.25,
                "gate_power": 1.0,
                "target_velocity_scale": 0.50,
                "target_acceleration_scale": 10.0,
                "action_delta_scale": 0.025,
                "action_second_difference_scale": 0.025,
                "target_velocity_weight": 0.25,
                "target_acceleration_weight": 1.0,
                "action_delta_weight": 0.25,
                "action_second_difference_weight": 1.0,
                "topk": 4,
                "action_term_name": "joint_pos",
                "asset_cfg": full_joint_cfg,
            },
        )


@configclass
class G1ExtremeStandRecoveryConservativeSmoothV10EnvCfg_PLAY(
    G1ExtremeStandRecoveryConservativeSmoothV10EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
