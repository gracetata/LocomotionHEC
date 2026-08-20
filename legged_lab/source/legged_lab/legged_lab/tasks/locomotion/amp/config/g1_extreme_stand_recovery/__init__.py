"""Gym registrations for the independent full-body Stand recovery task."""

import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_env_cfg:G1ExtremeStandRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-SmoothSettleV5-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v5_env_cfg:"
            "G1ExtremeStandRecoverySmoothSettleV5EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-SmoothSettleV5-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v5_env_cfg:"
            "G1ExtremeStandRecoverySmoothSettleV5EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-TargetLockV6-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v6_env_cfg:"
            "G1ExtremeStandRecoveryTargetLockV6EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-TargetLockV6-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v6_env_cfg:"
            "G1ExtremeStandRecoveryTargetLockV6EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-RecoveryPreservingV7-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v7_env_cfg:"
            "G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-RecoveryPreservingV7-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v7_env_cfg:"
            "G1ExtremeStandRecoveryRecoveryPreservingV7EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-SupportLockV8-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v8_env_cfg:"
            "G1ExtremeStandRecoverySupportLockV8EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-SupportLockV8-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v8_env_cfg:"
            "G1ExtremeStandRecoverySupportLockV8EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-DynamicLockV9-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v9_env_cfg:"
            "G1ExtremeStandRecoveryDynamicLockV9EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-DynamicLockV9-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v9_env_cfg:"
            "G1ExtremeStandRecoveryDynamicLockV9EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-ConservativeSmoothV10-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v10_env_cfg:"
            "G1ExtremeStandRecoveryConservativeSmoothV10EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "G1ExtremeStandRecoveryConservativeV10RslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-ConservativeSmoothV10-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v10_env_cfg:"
            "G1ExtremeStandRecoveryConservativeSmoothV10EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "G1ExtremeStandRecoveryConservativeV10RslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-V4JerkLimited-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v4_jerk_limited_env_cfg:"
            "G1ExtremeStandRecoveryV4JerkLimitedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "G1ExtremeStandRecoveryV4JerkLimitedRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-V4JerkLimited-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_v4_jerk_limited_env_cfg:"
            "G1ExtremeStandRecoveryV4JerkLimitedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "G1ExtremeStandRecoveryV4JerkLimitedRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_extreme_stand_recovery_env_cfg:G1ExtremeStandRecoveryEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1ExtremeStandRecoveryRslRlOnPolicyRunnerAmpCfg"
        ),
    },
)
