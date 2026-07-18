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
