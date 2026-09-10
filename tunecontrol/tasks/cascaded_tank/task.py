"""Cascaded tank controller-tuning problem."""
from dataclasses import asdict
from typing import Any, Dict, Tuple

import torch

from ..base import Task, validate_bounds
from ..module_utils import detach_trajectory
from .config import CascadedTankConfig, CascadedTankNoise
from .objectives import get_cascaded_tank_objective_spec, list_cascaded_tank_objectives
from .plant import CascadedTankSimulator


class CascadedTank(Task):
    config_type = CascadedTankConfig

    def __init__(self, config: CascadedTankConfig | None = None):
        config = CascadedTankConfig() if config is None else config
        if not isinstance(config, CascadedTankConfig):
            raise TypeError("config must be CascadedTankConfig")
        self.config = config
        self.dim = 2
        self.objective = config.objective
        self._objective_spec = get_cascaded_tank_objective_spec(config.objective)
        self._objective_label = self._objective_spec.display_name
        self.is_minimization = self._objective_spec.is_minimization
        self.bounds = torch.tensor([[0.9, 0.01], [4.5, 0.16]], dtype=torch.float64)
        dynamics = asdict(config.dynamics)
        dynamics["initial_state"] = torch.tensor(dynamics["initial_state"], dtype=torch.float64)
        validate_bounds(self.bounds, self.dim)
        self._sim = CascadedTankSimulator(
            duration=config.duration,
            target=config.target,
            noise_std=0.0 if config.noise is None else config.noise.std,
            dynamic_kwargs=dynamics,
        )
        self.data = None

    @property
    def name(self) -> str:  # type: ignore[override]
        label = self._objective_label or self.objective.upper()
        return f"CascadedTank-{label}-2D"

    def _evaluate(self, theta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        kp, ki = float(theta[0]), float(theta[1])
        trajectory = self._sim.simulate(kp, ki, generator=self.generator)
        value_tensor = self._objective_spec.evaluate(self._sim, trajectory)
        self.data = trajectory

        trajectory_torch = detach_trajectory(trajectory, dtype=theta.dtype, device=theta.device)

        info = {
            "theta": theta.detach().clone(),
            "trajectory": trajectory_torch,
        }

        return value_tensor, info

    @classmethod
    def available_configs(cls) -> list[CascadedTankConfig]:
        """Enumerate standard variants; custom configurations are also accepted."""
        return [
            CascadedTankConfig(objective=objective, noise=noise)
            for objective in list_cascaded_tank_objectives()
            for noise in (None, CascadedTankNoise())
        ]
