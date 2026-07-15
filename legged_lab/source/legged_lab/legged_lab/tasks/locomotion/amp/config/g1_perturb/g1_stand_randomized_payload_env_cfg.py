"""ArmHack Stand continuation task with generated arm motions and hand payloads."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_stand_perturb_env_cfg import G1StandPerturbEnvCfg
from .reference_data import STAND_RANDOM_POSE_BANK_RELATIVE_PATH


@configclass
class G1StandRandomizedPayloadEnvCfg(G1StandPerturbEnvCfg):
    """Stand task using random in-range arm poses and randomized wrist payloads."""

    def __post_init__(self):
        super().__post_init__()

        perturbation = self.upper_body_perturbation
        perturbation.source = "random_pose_trajectory"
        perturbation.random_pose_bank_path = STAND_RANDOM_POSE_BANK_RELATIVE_PATH.as_posix()
        perturbation.random_initialize_joint_state_on_reset = True
        perturbation.random_curriculum_enabled = True
        # 24 control steps are collected per PPO iteration: 500 iterations of
        # static random poses, then 1000 iterations ramping to full speed.
        perturbation.random_curriculum_static_steps = 12_000
        perturbation.random_curriculum_ramp_steps = 24_000
        perturbation.random_curriculum_motion_scale = 1.0
        perturbation.random_transition_duration_range_s = (2.0, 6.0)

        # The controlled arm chain ends at each wrist-yaw link.  Add 0..1 kg to
        # each side independently at startup to model unknown held payloads.
        # Inertia is recomputed consistently with the sampled mass.
        self.events.randomize_end_effector_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                    preserve_order=True,
                ),
                "mass_distribution_params": (0.0, 1.0),
                "operation": "add",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )


@configclass
class G1StandRandomizedPayloadEnvCfg_PLAY(G1StandRandomizedPayloadEnvCfg):
    """Small deterministic play variant; fixed CSV evaluation overrides the generator."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        # Keep the event structurally present so the launcher can choose a
        # deterministic fixed payload.  Zero added mass is the nominal default.
        self.events.randomize_end_effector_payload.params["mass_distribution_params"] = (0.0, 0.0)
