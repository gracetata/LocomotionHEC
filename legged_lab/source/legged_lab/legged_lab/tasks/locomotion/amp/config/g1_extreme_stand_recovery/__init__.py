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
