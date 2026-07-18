"""Stand-only ArmHack sim-to-real robustness continuation.

This task deliberately inherits the randomized arm-pose and wrist-payload
distribution, then adds torso wrench disturbances and conservative joint
parameter randomization.  It does not change or participate in the Walk task.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_stand_randomized_payload_env_cfg import G1StandRandomizedPayloadEnvCfg


@configclass
class G1StandRobustEnvCfg(G1StandRandomizedPayloadEnvCfg):
    """Full-speed random-arm Stand task with external wrenches and joint DR."""

    def __post_init__(self):
        super().__post_init__()

        # The input policy already completed the static -> 1.0x arm-motion
        # curriculum.  This continuation starts directly on the final random
        # pose-interpolation distribution instead of relearning the old stages.
        perturbation = self.upper_body_perturbation
        perturbation.random_curriculum_enabled = False
        perturbation.random_curriculum_static_steps = 0
        perturbation.random_curriculum_ramp_steps = 0
        perturbation.random_curriculum_motion_scale = 1.0
        perturbation.random_transition_duration_range_s = (2.0, 6.0)

        # IsaacLab otherwise warns that persistent external wrenches are only
        # applied once per physics step.  The S3 asset already uses four solver
        # velocity iterations, so apply the wrench in every solver iteration.
        self.sim.physx.enable_external_forces_every_iteration = True

        # A fall is a much more serious failure than a small posture error on
        # hardware.  This is a one-step terminal penalty, increased from -200.
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

        # Keep this stage targeted and auditable.  Do not silently turn on the
        # broad parent randomization package (terrain friction, base/link mass,
        # CoM, or velocity teleport pushes).  Wrist payload randomization from
        # G1StandRandomizedPayloadEnvCfg remains active as a separate event.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_link_mass = None
        self.events.push_robot = None

        # Sustained external wrench on the torso.  The reset-time zero-wrench
        # event inherited from the parent clears it at every episode reset;
        # independent interval clocks then resample each environment while the
        # arms continue moving between random poses.
        self.events.random_torso_external_wrench = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "force_range": (-20.0, 20.0),
                "torque_range": (-3.0, 3.0),
            },
        )

        # Per-environment, per-joint actuator uncertainty.  Scaling around the
        # nominal asset values avoids inventing absolute gains with wrong units.
        self.events.scale_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
                "stiffness_distribution_params": (0.90, 1.10),
                "damping_distribution_params": (0.90, 1.10),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # Joint friction and armature are also independently sampled around the
        # asset defaults.  Joint limits are intentionally left unchanged.
        self.events.scale_joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
                "friction_distribution_params": (0.80, 1.20),
                "armature_distribution_params": (0.90, 1.10),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
