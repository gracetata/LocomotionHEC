"""ArmHack Walk fine-tuning with an explicit 30-cm ankle-distance objective."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1_FOOT_BODY_NAMES

from .g1_walk_behavior_env_cfg import G1WalkBehaviorFinetuneEnvCfg
from .g1_walk_two_goal_env_cfg import (
    G1WalkTwoGoalLateralRobustEnvCfg,
    G1WalkTwoGoalYawRobustEnvCfg,
)


ANKLE_DISTANCE_TARGET_M = 0.30
ANKLE_DISTANCE_KERNEL_STD_M = 0.06
ANKLE_DISTANCE_KERNEL_WEIGHT = 500.0


def _configure_ankle_spacing_kernel(cfg) -> None:
    """Install the same command-independent ankle objective on one actor task."""
    cfg.rewards.ankle_distance_30cm_kernel = RewTerm(
        func=mdp.ankle_distance_target_kernel,
        weight=ANKLE_DISTANCE_KERNEL_WEIGHT,
        params={
            "target_distance": ANKLE_DISTANCE_TARGET_M,
            "std": ANKLE_DISTANCE_KERNEL_STD_M,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
            ),
        },
    )


@configclass
class G1WalkAnkleSpacingBaseEnvCfg(G1WalkBehaviorFinetuneEnvCfg):
    """Base gait over zero, straight, diagonal and general Nav2-like commands."""

    def __post_init__(self):
        super().__post_init__()
        _configure_ankle_spacing_kernel(self)


@configclass
class G1WalkAnkleSpacingLateralEnvCfg(G1WalkTwoGoalLateralRobustEnvCfg):
    """Gated lateral expert with the identical 30-cm ankle objective."""

    def __post_init__(self):
        super().__post_init__()
        _configure_ankle_spacing_kernel(self)


@configclass
class G1WalkAnkleSpacingYawEnvCfg(G1WalkTwoGoalYawRobustEnvCfg):
    """Zero-linear pure-yaw expert balancing 30-cm spacing and no drift."""

    def __post_init__(self):
        super().__post_init__()
        _configure_ankle_spacing_kernel(self)
        # The large spacing kernel can otherwise buy a wider stance by leaking
        # backward velocity during a nominal in-place turn.  Keep this local to
        # the yaw expert: base/lateral policies and all Stand tasks are untouched.
        self.rewards.pure_yaw_planar_drift_l2.weight = -50.0
