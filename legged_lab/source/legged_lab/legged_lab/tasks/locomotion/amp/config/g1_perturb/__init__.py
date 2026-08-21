import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandPerturb-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_perturb_env_cfg:G1StandPerturbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandPerturb-Play-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_perturb_env_cfg:G1StandPerturbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_stand_randomized_payload_env_cfg:G1StandRandomizedPayloadEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-Play-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_stand_randomized_payload_env_cfg:G1StandRandomizedPayloadEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandFootRecovery-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_stand_foot_recovery_env_cfg:G1StandFootRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandFootRecovery-Play-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_stand_foot_recovery_env_cfg:G1StandFootRecoveryEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandAdaptiveSwitch-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_adaptive_switch_env_cfg:G1StandAdaptiveSwitchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandAdaptiveSwitch-Play-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_adaptive_switch_env_cfg:G1StandAdaptiveSwitchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandAdaptiveHold-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_adaptive_hold_env_cfg:G1StandAdaptiveHoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandRobust-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_robust_env_cfg:G1StandRobustEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-StandDownToDefault-v0",
    entry_point="legged_lab.envs:G1PerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_stand_down_to_default_env_cfg:G1StandDownToDefaultEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPerturbRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_perturb_env_cfg:G1WalkPerturbFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkPerturbFinetuneRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_perturb_env_cfg:G1WalkPerturbFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkPerturbFinetuneRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_robust_env_cfg:G1WalkRobustFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-Play-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_robust_env_cfg:G1WalkRobustFinetuneEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-WalkBehaviorFinetune-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_behavior_env_cfg:G1WalkBehaviorFinetuneEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkBehaviorFinetuneRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-WalkBehaviorFinetune-Play-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_behavior_env_cfg:G1WalkBehaviorFinetuneEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkBehaviorFinetuneRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalLateral-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_two_goal_env_cfg:G1WalkTwoGoalLateralExpertEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkTwoGoalExpertRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalLateralRobust-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_two_goal_env_cfg:G1WalkTwoGoalLateralRobustEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkTwoGoalRobustRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalYaw-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_two_goal_env_cfg:G1WalkTwoGoalYawExpertEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkTwoGoalExpertRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkTwoGoalYawRobust-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_walk_two_goal_env_cfg:G1WalkTwoGoalYawRobustEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkTwoGoalRobustRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

for branch, env_cfg_name in (
    ("Base", "G1WalkAnkleSpacingBaseEnvCfg"),
    ("Lateral", "G1WalkAnkleSpacingLateralEnvCfg"),
    ("Yaw", "G1WalkAnkleSpacingYawEnvCfg"),
):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-G1-ArmHackWalkAnkleSpacing{branch}-v0",
        entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_walk_ankle_spacing_env_cfg:{env_cfg_name}"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:"
                "G1WalkAnkleSpacingRslRlOnPolicyRunnerAmpCfg"
            ),
        },
    )

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkPrecisionSwitch-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_precision_switch_env_cfg:G1WalkPrecisionSwitchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkAnkleSpacingRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkPrecisionSwitch-Play-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_precision_switch_env_cfg:G1WalkPrecisionSwitchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkAnkleSpacingRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkYawForceRobust-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_yaw_force_robust_env_cfg:G1WalkYawForceRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkAnkleSpacingRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ArmHackWalkSwitchOod-v0",
    entry_point="legged_lab.envs:G1WalkPerturbAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_walk_switch_ood_env_cfg:G1WalkSwitchOodEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkAnkleSpacingRslRlOnPolicyRunnerAmpCfg",
    },
)
