"""Walk-only upper-body pose perturbation environment.

This module deliberately keeps pose-reset behavior out of ``g1_perturb_env`` so
the Stand task and an already running Stand process are unaffected.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .g1_perturb_env import G1PerturbAmpEnv, UpperBodyPerturbationCfg


@configclass
class WalkUpperBodyPerturbationCfg(UpperBodyPerturbationCfg):
    """Named fixed-pose settings used only by the ArmHack Walk task."""

    pose_names: list[str] = []
    pose_name: str = "pos2_down"
    initialize_joint_state_on_reset: bool = True


class G1WalkPerturbAmpEnv(G1PerturbAmpEnv):
    """Apply one named arm pose for a whole episode without a reset-time jump."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._walk_pose_reset_ready = False
        self._active_walk_pose_indices: torch.Tensor | None = None
        self._walk_pose_init_max_error_rad: torch.Tensor | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        perturbation_cfg = self._walk_perturbation_cfg()
        if perturbation_cfg is None:
            raise ValueError("G1WalkPerturbAmpEnv requires an enabled pose-set perturbation config.")
        self._validate_walk_pose_cfg(perturbation_cfg)

        self._walk_pose_reset_ready = True
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if perturbation_cfg.initialize_joint_state_on_reset:
            self._initialize_walk_arm_state(all_env_ids)

    def step(self, action: torch.Tensor):
        step_result = super().step(action)
        if self._active_walk_pose_indices is not None:
            log_extras = self.extras.setdefault("log", {})
            log_extras["ArmHack/walk_pose_index_mean"] = torch.mean(
                self._active_walk_pose_indices.to(dtype=torch.float32)
            )
        if self._walk_pose_init_max_error_rad is not None:
            log_extras = self.extras.setdefault("log", {})
            log_extras["ArmHack/walk_pose_init_max_error_rad"] = torch.max(
                self._walk_pose_init_max_error_rad
            )
        return step_result

    def _walk_perturbation_cfg(self) -> WalkUpperBodyPerturbationCfg | None:
        cfg = getattr(self, "_perturbation_cfg", None)
        if cfg is None or not cfg.enabled or cfg.source != "pose_set":
            return None
        if not isinstance(cfg, WalkUpperBodyPerturbationCfg):
            raise TypeError(
                "G1WalkPerturbAmpEnv expects WalkUpperBodyPerturbationCfg, "
                f"but got {type(cfg).__name__}."
            )
        return cfg

    def _validate_walk_pose_cfg(self, cfg: WalkUpperBodyPerturbationCfg) -> None:
        if len(cfg.pose_names) != len(cfg.pose_set):
            raise ValueError("Walk pose_names must have the same length as pose_set.")
        if not cfg.pose_names or len(set(cfg.pose_names)) != len(cfg.pose_names):
            raise ValueError("Walk pose_names must be non-empty and unique.")
        if cfg.pose_name != "random" and cfg.pose_name not in cfg.pose_names:
            choices = ", ".join([*cfg.pose_names, "random"])
            raise ValueError(f"Unknown Walk pose_name={cfg.pose_name!r}; choose one of: {choices}.")

    def _sample_pose_targets(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._pose_set_targets.numel() == 0:
            raise RuntimeError("Walk pose-set perturbations are not initialized.")

        cfg = self._walk_perturbation_cfg()
        if cfg is None:
            # This can only occur while the parent constructor is still creating
            # managers. The normal parent sampler is safe for that phase.
            super()._sample_pose_targets(env_ids)
            return

        self._validate_walk_pose_cfg(cfg)
        if cfg.pose_name == "random":
            if self._pose_probabilities.numel() > 0:
                pose_indices = torch.multinomial(
                    self._pose_probabilities, env_ids.numel(), replacement=True
                )
            else:
                pose_indices = torch.randint(
                    self._pose_set_targets.shape[0], (env_ids.numel(),), device=self.device
                )
        else:
            pose_index = cfg.pose_names.index(cfg.pose_name)
            pose_indices = torch.full(
                (env_ids.numel(),), pose_index, dtype=torch.long, device=self.device
            )

        self._active_pose_targets[env_ids] = self._pose_set_targets[pose_indices]
        if self._active_walk_pose_indices is None:
            self._active_walk_pose_indices = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        self._active_walk_pose_indices[env_ids] = pose_indices

        if self._walk_pose_reset_ready and cfg.initialize_joint_state_on_reset:
            self._initialize_walk_arm_state(env_ids)

    def _initialize_walk_arm_state(self, env_ids: torch.Tensor) -> None:
        """Synchronize physical arm state, actuator targets, and action history."""

        if env_ids.numel() == 0:
            return
        action_term = self._joint_pos_action
        assert action_term is not None
        assert self._upper_action_indices is not None
        assert self._upper_asset_joint_ids is not None

        targets = self._clip_upper_position_targets(
            self._active_pose_targets[env_ids], env_ids=env_ids
        )
        raw_actions, reachable_targets = self._targets_to_local_raw_actions(targets, env_ids)
        self._active_pose_targets[env_ids] = reachable_targets

        robot = self.scene["robot"]
        robot.write_joint_state_to_sim(
            reachable_targets,
            torch.zeros_like(reachable_targets),
            joint_ids=self._upper_asset_joint_ids,
            env_ids=env_ids,
        )
        robot.set_joint_position_target(
            reachable_targets,
            joint_ids=self._upper_asset_joint_ids,
            env_ids=env_ids,
        )

        upper_action_ids = self._upper_action_indices
        action_term._raw_actions[env_ids.unsqueeze(-1), upper_action_ids.unsqueeze(0)] = raw_actions
        action_term._processed_actions[
            env_ids.unsqueeze(-1), upper_action_ids.unsqueeze(0)
        ] = reachable_targets

        term_index = self.action_manager.active_terms.index("joint_pos")
        term_start = sum(self.action_manager.action_term_dim[:term_index])
        manager_action_ids = upper_action_ids + term_start
        manager_env_ids = env_ids.unsqueeze(-1)
        manager_action_ids = manager_action_ids.unsqueeze(0)
        self.action_manager._action[manager_env_ids, manager_action_ids] = raw_actions
        self.action_manager._prev_action[manager_env_ids, manager_action_ids] = raw_actions

        actual_positions = robot.data.joint_pos[env_ids][:, self._upper_asset_joint_ids]
        init_error = torch.max(torch.abs(actual_positions - reachable_targets), dim=1).values
        if self._walk_pose_init_max_error_rad is None:
            self._walk_pose_init_max_error_rad = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        self._walk_pose_init_max_error_rad[env_ids] = init_error

    def _targets_to_local_raw_actions(
        self, targets: torch.Tensor, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the action affine map for only the environments being reset."""

        action_term = self._joint_pos_action
        assert action_term is not None
        upper_action_ids = self._upper_action_indices
        assert upper_action_ids is not None

        if isinstance(action_term._offset, torch.Tensor):
            offset = action_term._offset[env_ids][:, upper_action_ids]
        else:
            offset = torch.full_like(targets, float(action_term._offset))
        if isinstance(action_term._scale, torch.Tensor):
            scale = action_term._scale[env_ids][:, upper_action_ids]
        else:
            scale = torch.full_like(targets, float(action_term._scale))

        safe_scale = torch.where(
            torch.abs(scale) > 1.0e-8, scale, torch.full_like(scale, 1.0e-8)
        )
        raw_actions = (targets - offset) / safe_scale
        if action_term.cfg.clip is not None:
            clip = action_term._clip[env_ids][:, upper_action_ids]
            raw_actions = torch.clamp(raw_actions, min=clip[..., 0], max=clip[..., 1])
        reachable_targets = raw_actions * scale + offset
        return raw_actions, reachable_targets
