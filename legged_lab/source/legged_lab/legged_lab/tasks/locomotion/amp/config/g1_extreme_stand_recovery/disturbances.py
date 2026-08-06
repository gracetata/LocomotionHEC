"""Mutually-exclusive disturbance curriculum for Extreme Stand V5."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import EventTermCfg


class single_body_force_curriculum(ManagerTermBase):
    """Apply one short horizontal force followed by a guaranteed quiet window.

    Each environment owns one state machine.  Exactly one body and one
    horizontal direction can be active at a time; velocity pushes and the old
    independent wrench clocks are disabled by the V5 task configuration.
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._body_ids = list(asset_cfg.body_ids)
        if not self._body_ids:
            raise ValueError("single_body_force_curriculum requires at least one body.")
        self.active_time_left = torch.zeros(env.num_envs, device=env.device)
        self.quiet_time_left = torch.zeros(env.num_envs, device=env.device)
        self.has_disturbed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._force_magnitude = torch.zeros(env.num_envs, device=env.device)
        self._stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._active_body_slot = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._active_direction = torch.zeros((env.num_envs, 2), device=env.device)
        self._all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        self._sample_initial_quiet(self._all_env_ids, cfg.params.get("initial_quiet_range_s", (1.0, 2.0)))

    @property
    def curriculum_stage(self) -> torch.Tensor:
        return self._stage

    @property
    def force_magnitude(self) -> torch.Tensor:
        return self._force_magnitude

    def _uniform(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        if low < 0.0 or high < low:
            raise ValueError(f"Invalid non-negative range: {value_range}")
        return low + (high - low) * torch.rand(count, device=self._asset.device)

    def _sample_initial_quiet(self, env_ids: torch.Tensor, value_range: tuple[float, float]) -> None:
        self.quiet_time_left[env_ids] = self._uniform(len(env_ids), value_range)

    def _clear_wrench(self, env_ids: torch.Tensor) -> None:
        zeros = torch.zeros(
            (len(env_ids), len(self._body_ids), 3),
            device=self._asset.device,
            dtype=self._asset.data.joint_pos.dtype,
        )
        self._asset.set_external_force_and_torque(
            zeros, zeros, body_ids=self._body_ids, env_ids=env_ids, is_global=True
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = self._all_env_ids if env_ids is None else torch.as_tensor(
            env_ids, device=self._asset.device, dtype=torch.long
        )
        self.active_time_left[env_ids] = 0.0
        self.has_disturbed[env_ids] = False
        self._force_magnitude[env_ids] = 0.0
        self._active_direction[env_ids] = 0.0
        initial_range = self.cfg.params.get("initial_quiet_range_s", (1.0, 2.0))
        self._sample_initial_quiet(env_ids, initial_range)
        self._clear_wrench(env_ids)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        tick_s: float = 0.10,
        active_duration_range_s: tuple[float, float] = (0.10, 0.30),
        quiet_duration_range_s: tuple[float, float] = (6.0, 10.0),
        initial_quiet_range_s: tuple[float, float] = (1.0, 2.0),
        force_magnitudes_n: tuple[float, float, float, float] = (10.0, 20.0, 36.0, 45.0),
        stage_step_thresholds: tuple[int, int, int] = (7200, 14400, 24000),
        direction_probabilities: tuple[float, float, float, float] = (0.15, 0.30, 0.20, 0.35),
    ) -> None:
        del asset_cfg, initial_quiet_range_s
        if tick_s <= 0.0:
            raise ValueError("tick_s must be positive.")
        if len(force_magnitudes_n) != 4 or len(stage_step_thresholds) != 3:
            raise ValueError("V5 curriculum requires four force levels and three thresholds.")
        if any(force <= 0.0 for force in force_magnitudes_n):
            raise ValueError("All curriculum forces must be positive.")
        probabilities = torch.tensor(direction_probabilities, device=env.device)
        if probabilities.shape != (4,) or torch.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
            raise ValueError("direction_probabilities must contain four non-negative values.")
        probabilities = probabilities / probabilities.sum()

        # Every callback first clears the selected environments, so a force can
        # never survive a reset or overlap with the next sampled disturbance.
        self._clear_wrench(env_ids)
        self.active_time_left[env_ids] = torch.clamp_min(self.active_time_left[env_ids] - tick_s, 0.0)
        active = self.active_time_left[env_ids] > 0.0
        finished = (~active) & (self._force_magnitude[env_ids] > 0.0)
        finished_ids = env_ids[finished]
        if len(finished_ids) > 0:
            self._force_magnitude[finished_ids] = 0.0
            self._active_direction[finished_ids] = 0.0
            self.quiet_time_left[finished_ids] = self._uniform(len(finished_ids), quiet_duration_range_s)

        quiet_mask = self._force_magnitude[env_ids] <= 0.0
        quiet_ids = env_ids[quiet_mask]
        if len(quiet_ids) > 0:
            self.quiet_time_left[quiet_ids] = torch.clamp_min(
                self.quiet_time_left[quiet_ids] - tick_s, 0.0
            )
        ready_ids = quiet_ids[self.quiet_time_left[quiet_ids] <= 0.0]

        current_step = int(env.common_step_counter)
        thresholds = torch.tensor(stage_step_thresholds, device=env.device)
        stage = int(torch.sum(torch.tensor(current_step, device=env.device) >= thresholds).item())
        if len(ready_ids) > 0:
            self._stage[ready_ids] = stage
            self._force_magnitude[ready_ids] = float(force_magnitudes_n[stage])
            self.active_time_left[ready_ids] = self._uniform(len(ready_ids), active_duration_range_s)
            self.has_disturbed[ready_ids] = True
            self._active_body_slot[ready_ids] = torch.randint(
                len(self._body_ids), (len(ready_ids),), device=env.device
            )
            direction_index = torch.multinomial(probabilities, len(ready_ids), replacement=True)
            # +X forward, -X backward, +Y left, -Y right.  Back/right receive
            # higher default probability because those recovery cases were weak.
            direction_table = torch.tensor(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                device=env.device,
            )
            self._active_direction[ready_ids] = direction_table[direction_index]

        active_ids = env_ids[self.active_time_left[env_ids] > 0.0]
        if len(active_ids) > 0:
            forces = torch.zeros(
                (len(active_ids), len(self._body_ids), 3),
                device=env.device,
                dtype=self._asset.data.joint_pos.dtype,
            )
            rows = torch.arange(len(active_ids), device=env.device)
            forces[rows, self._active_body_slot[active_ids], :2] = (
                self._active_direction[active_ids] * self._force_magnitude[active_ids, None]
            )
            self._asset.set_external_force_and_torque(
                forces,
                torch.zeros_like(forces),
                body_ids=self._body_ids,
                env_ids=active_ids,
                is_global=True,
            )

        # These scalar diagnostics are written to TensorBoard by the standard runner.
        env.extras.setdefault("log", {})["ExtremeStand/disturbance_stage"] = float(stage)
        env.extras["log"]["ExtremeStand/disturbance_force_n"] = self._force_magnitude.mean()
        env.extras["log"]["ExtremeStand/disturbance_active_fraction"] = (
            self.active_time_left > 0.0
        ).float().mean()
