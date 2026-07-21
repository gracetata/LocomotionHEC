"""Stand continuation from natural-down arms to the flat P0 arm pose."""
from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.envs.g1_perturb_env import G1_UPPER_BODY_JOINT_NAMES

from .g1_stand_robust_env_cfg import G1StandRobustEnvCfg
from .reference_data import load_stand_arm_pose


NATURAL_DOWN_ARM_POSITIONS = load_stand_arm_pose(
    "AD_natural_down", G1_UPPER_BODY_JOINT_NAMES
)
FLAT_DEFAULT_ARM_POSITIONS = load_stand_arm_pose(
    "P0_symmetric_reference", G1_UPPER_BODY_JOINT_NAMES
)


@configclass
class G1StandDownToDefaultEnvCfg(G1StandRobustEnvCfg):
    """Stand robustly while both arms rise together from AD to P0 at varied speeds."""

    def __post_init__(self):
        super().__post_init__()

        perturbation = self.upper_body_perturbation
        perturbation.source = "pose_transition"
        perturbation.pose_transition_start_positions = list(NATURAL_DOWN_ARM_POSITIONS)
        perturbation.pose_transition_goal_positions = list(FLAT_DEFAULT_ARM_POSITIONS)
        perturbation.pose_transition_initialize_joint_state_on_reset = True

        # Each episode starts from the exact deployment AD pose.  It remains
        # there for a per-environment random delay, then both arms share one
        # minimum-jerk phase and therefore start/finish simultaneously.  The
        # sampled duration changes the lift speed without revealing any future
        # target, delay, duration, or phase in the policy observation.
        perturbation.pose_transition_start_delay_range_s = (2.0, 5.0)
        perturbation.pose_transition_duration_range_s = (3.0, 9.0)

        # 24 control steps are collected per PPO iteration.  First train 500
        # iterations of pure natural-down standing, then ramp the reference
        # clock over 500 iterations.  The remaining continuation uses the full
        # randomized 3..9 s transition-speed distribution.
        perturbation.pose_transition_curriculum_enabled = True
        perturbation.pose_transition_curriculum_static_steps = 12_000
        perturbation.pose_transition_curriculum_ramp_steps = 12_000
        perturbation.pose_transition_curriculum_motion_scale = 1.0
