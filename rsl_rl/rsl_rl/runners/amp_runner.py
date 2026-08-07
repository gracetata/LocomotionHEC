from __future__ import annotations

import copy
import os
import shutil
import statistics
import time
import torch
import warnings
from collections import deque
from pathlib import Path
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import PPO, PPOAMP
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCommandResidual,
    ActorCriticCNN,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
    resolve_amp_config,
)
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger
from rsl_rl.utils.amp_logger import LoggerAMP
from rsl_rl.runners import OnPolicyRunner


class AMPRunner(OnPolicyRunner):
    
    alg: PPOAMP

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        
        self.logger = LoggerAMP(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            max_episode_length_s=(self.env.max_episode_length*self.env.unwrapped.step_dt),
        )
        self._initialize_static_demo_normalizer()
        self._permanently_frozen_actor_parameters: set[int] = set()
        self._actor_warmup_active: bool | None = None
        self._freeze_actor_hidden_layers()

    def _freeze_actor_hidden_layers(self) -> None:
        if bool(self.cfg.get("freeze_base_actor", False)):
            for parameter in self.alg.policy.actor.parameters():
                parameter.requires_grad_(False)
                self._permanently_frozen_actor_parameters.add(id(parameter))
            print("Froze the complete base actor; command residual adapters remain trainable.")
        if bool(self.cfg.get("freeze_pure_yaw_residual", False)):
            residual = getattr(self.alg.policy, "pure_yaw_command_residual", None)
            if residual is None:
                raise ValueError("freeze_pure_yaw_residual requires a pure-yaw residual module.")
            for parameter in residual.parameters():
                parameter.requires_grad_(False)
                self._permanently_frozen_actor_parameters.add(id(parameter))
            print("Froze the pure-yaw residual; exact carrier bridge remains active.")
        count = int(self.cfg.get("freeze_actor_hidden_layers", 0))
        if count <= 0:
            return
        linear_layers = [
            module for module in self.alg.policy.actor
            if isinstance(module, torch.nn.Linear)
        ]
        hidden_layers = linear_layers[:-1]
        if count > len(hidden_layers):
            raise ValueError(
                f"Cannot freeze {count} actor hidden layers; actor has {len(hidden_layers)}."
            )
        for layer in hidden_layers[:count]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
                self._permanently_frozen_actor_parameters.add(id(parameter))
        print(f"Froze the first {count} actor hidden linear layers for conservative refinement.")

    def _set_actor_warmup_state(self, iteration: int) -> None:
        """Train only the fresh critic during the configured initial iterations."""
        warmup_iterations = int(self.cfg.get("actor_warmup_iterations", 0))
        warmup_active = iteration < warmup_iterations
        if warmup_active == self._actor_warmup_active:
            return

        for name, parameter in self.alg.policy.named_parameters():
            is_actor_parameter = (
                name.startswith("actor.")
                or name.startswith("lateral_command_residual.")
                or name.startswith("pure_yaw_command_residual.")
                or name in {"std", "log_std"}
            )
            if not is_actor_parameter:
                continue
            if warmup_active or id(parameter) in self._permanently_frozen_actor_parameters:
                parameter.requires_grad_(False)
            else:
                parameter.requires_grad_(True)
        self._actor_warmup_active = warmup_active
        if warmup_active:
            print(
                f"Actor warmup active at iteration {iteration}: training the fresh critic only "
                f"until iteration {warmup_iterations}."
            )
        else:
            print(
                f"Actor warmup complete at iteration {iteration}: enabled actor layers except "
                "the permanently frozen hidden prefix."
            )

    def _initialize_static_demo_normalizer(self) -> None:
        amp_cfg = self.alg_cfg.get("amp_cfg", {})
        if amp_cfg.get("normalizer_mode", "policy") != "demo_static":
            return

        num_batches = int(amp_cfg.get("demo_normalizer_init_batches", 8))
        if num_batches <= 0:
            raise ValueError("demo_normalizer_init_batches must be positive when normalizer_mode='demo_static'.")

        normalizer = self.alg.amp_discriminator.disc_obs_normalizer
        normalizer.train()
        with torch.no_grad():
            for _ in range(num_batches):
                obs, _ = self.env.reset()
                obs = obs.to(self.device)
                disc_demo_obs = self.alg.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
                self.alg.amp_discriminator.update_normalization(disc_demo_obs)

        normalizer.until = int(normalizer.count.item())
        print(
            "Initialized frozen AMP discriminator Demo normalizer "
            f"from {normalizer.count.item()} demo frames."
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            self._set_actor_warmup_state(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Extract AMP rewards (only for logging)
                    style_rewards = self.alg.style_rewards
                    total_rewards = self.alg.rewards_lerp
                    env_unwrapped = getattr(self.env, "unwrapped", self.env)
                    if hasattr(env_unwrapped, "record_amp_style_rewards"):
                        env_unwrapped.record_amp_style_rewards(style_rewards.to(self.env.device))
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards, style_rewards, total_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )
            
            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Save AMP model
        saved_dict["amp_discriminator_state_dict"] = self.alg.amp_discriminator.state_dict()
        saved_dict["amp_discriminator_normalizer_state_dict"] = self.alg.amp_discriminator.disc_obs_normalizer.state_dict()
        saved_dict["amp_discriminator_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        torch.save(saved_dict, path)

        checkpoint_output_dir = self.cfg.get("checkpoint_output_dir")
        if checkpoint_output_dir:
            primary_path = Path(path).expanduser().resolve()
            output_root = Path(str(checkpoint_output_dir)).expanduser()
            if not output_root.is_absolute():
                search_roots = [Path.cwd().resolve(), *primary_path.parents]
                project_root = next(
                    (
                        root
                        for root in search_roots
                        if (root / "scripts" / "rsl_rl" / "train.py").is_file()
                        and (root / "source" / "legged_lab").is_dir()
                    ),
                    Path.cwd().resolve(),
                )
                output_root = project_root / output_root

            run_name = primary_path.parent.name
            exported_path = (output_root / run_name / primary_path.name).resolve()
            if exported_path != primary_path:
                exported_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(primary_path, exported_path)
                print(f"Saved dedicated AMP checkpoint copy to: {exported_path}")

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_actor_only = bool(self.cfg.get("load_actor_only", False))
        load_actor_amp_only = bool(self.cfg.get("load_actor_amp_only", False))
        load_policy_only = bool(self.cfg.get("load_policy_only", False))
        if sum((load_actor_only, load_actor_amp_only, load_policy_only)) > 1:
            raise ValueError(
                "load_actor_only, load_actor_amp_only, and load_policy_only are mutually exclusive."
            )

        if load_actor_only or load_actor_amp_only:
            source_state = loaded_dict.get("model_state_dict")
            if not isinstance(source_state, dict):
                raise KeyError("Actor-only AMP checkpoint is missing model_state_dict.")
            source_actor_state = {
                key: value for key, value in source_state.items() if key.startswith("actor.")
            }
            target_actor_state = {
                key: value
                for key, value in self.alg.policy.state_dict().items()
                if key.startswith("actor.")
            }
            if set(source_actor_state) != set(target_actor_state):
                missing = sorted(set(target_actor_state) - set(source_actor_state))
                unexpected = sorted(set(source_actor_state) - set(target_actor_state))
                raise RuntimeError(
                    "Actor-only AMP checkpoint is incompatible with the current actor: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            shape_mismatches = {
                key: (tuple(source_actor_state[key].shape), tuple(target_actor_state[key].shape))
                for key in target_actor_state
                if source_actor_state[key].shape != target_actor_state[key].shape
            }
            if shape_mismatches:
                raise RuntimeError(
                    "Actor-only AMP checkpoint has incompatible actor tensor shapes: "
                    f"{shape_mismatches}"
                )
            load_result = self.alg.policy.load_state_dict(source_actor_state, strict=False)
            # Upstream torch modules return _IncompatibleKeys, while this
            # repository's ActorCritic compatibility wrapper returns a bool.
            # Exact actor key/shape validation above is authoritative in both cases.
            if hasattr(load_result, "unexpected_keys"):
                unexpected_non_actor = [
                    key
                    for key in load_result.unexpected_keys
                    if not key.startswith("actor.")
                ]
                if unexpected_non_actor:
                    raise RuntimeError(
                        "Actor-only AMP load produced unexpected non-actor keys: "
                        f"{unexpected_non_actor}"
                    )
            if load_actor_amp_only:
                self.alg.amp_discriminator.load_state_dict(
                    loaded_dict["amp_discriminator_state_dict"]
                )
                self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
                    loaded_dict["amp_discriminator_normalizer_state_dict"]
                )
            self.current_learning_iteration = 0
            if load_actor_amp_only:
                print(
                    "Loaded actor and AMP discriminator/normalizer; critic, action noise, "
                    f"PPO optimizer, and discriminator optimizer remain fresh: {path}"
                )
            else:
                print(
                    "Loaded actor-only AMP checkpoint; critic, action noise, PPO optimizer, "
                    f"and AMP state remain freshly initialized: {path}"
                )
            return loaded_dict.get("infos", {})

        reset_amp_on_load = bool(self.cfg.get("reset_amp_on_load", False)) and not load_policy_only
        if reset_amp_on_load:
            # Keep actor, critic, PPO optimizer and iteration from a full-state
            # Walk resume, but do not reuse a discriminator trained before the
            # demo DoF name-order fix. State includes the observation normalizer.
            fresh_amp_state = copy.deepcopy(self.alg.amp_discriminator.state_dict())
            fresh_disc_optimizer_state = copy.deepcopy(self.alg.disc_optimizer.state_dict())

        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_policy_only:
            if not bool(self.cfg.get("reset_iteration_on_policy_only_load", True)):
                self.current_learning_iteration = int(loaded_dict.get("iter", 0))
            print(f"Loaded policy-only AMP checkpoint from: {path}")
            return loaded_dict.get("infos", {})

        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load AMP model
        self.alg.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])
        self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
            loaded_dict["amp_discriminator_normalizer_state_dict"]
        )
        # Load optimizer if used
        if load_optimizer:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            # AMP discriminator optimizer
            self.alg.disc_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
            if bool(self.cfg.get("restore_configured_learning_rate_on_load", False)):
                for param_group in self.alg.optimizer.param_groups:
                    param_group["lr"] = self.alg.learning_rate
                print(
                    "Restored configured PPO learning rate after optimizer load: "
                    f"{self.alg.learning_rate:.3e}"
                )
        if reset_amp_on_load:
            self.alg.amp_discriminator.load_state_dict(fresh_amp_state)
            self.alg.disc_optimizer.load_state_dict(fresh_disc_optimizer_state)
            print("Reset AMP discriminator, normalizer, and optimizer after full-state policy resume.")
        # Load current learning iteration
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        return loaded_dict.get("infos", {})

    def train_mode(self):
        super().train_mode()
        if self.alg.freeze_amp_discriminator:
            self.alg.amp_discriminator.eval()
            self.alg.amp_discriminator.disc_obs_normalizer.eval()
        else:
            self.alg.amp_discriminator.train()
            self.alg.amp_discriminator.disc_obs_normalizer.train()
        
    def eval_mode(self):
        super().eval_mode()
        self.alg.amp_discriminator.eval()
        self.alg.amp_discriminator.disc_obs_normalizer.eval()
    
    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        
        # Resolve AMP config
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )
        
        # Initialize AMP discriminator observation buffers
        disc_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs, 
            device=self.device
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs, 
            device=self.device
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPOAMP = alg_class(
            actor_critic, storage, disc_obs_buffer, disc_demo_obs_buffer, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
