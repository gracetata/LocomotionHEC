"""V4-only Extreme Stand task with state-gated stable damping."""

from __future__ import annotations

from isaaclab.utils import configclass

from .actions import StableDampedJointPositionActionCfg
from .g1_extreme_stand_recovery_env_cfg import (
    G1ExtremeStandRecoveryEnvCfg,
    G1_LOCOMOTION_JOINT_NAMES,
)


@configclass
class G1ExtremeStandRecoveryV4JerkLimitedEnvCfg(G1ExtremeStandRecoveryEnvCfg):
    """The original V4 MDP with only stable-region damping added."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = StableDampedJointPositionActionCfg(
            asset_name="robot",
            joint_names=G1_LOCOMOTION_JOINT_NAMES,
            preserve_order=True,
            scale=0.25,
            use_default_offset=True,
        )


@configclass
class G1ExtremeStandRecoveryV4JerkLimitedEnvCfg_PLAY(
    G1ExtremeStandRecoveryV4JerkLimitedEnvCfg
):
    """Small evaluation variant with the identical stable-damping path."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
