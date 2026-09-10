"""CartPole controller-tuning problem."""
from dataclasses import asdict
from typing import Any, Dict, Tuple

import torch

from ..base import Task, validate_bounds
from ..module_utils import detach_trajectory
from .config import CartPoleConfig, CartPoleNoise
from .objectives import get_cartpole_objective_spec, list_cartpole_objectives
from .plant import CartPoleSimulator

_CARTPOLE_BOUNDS: Dict[int, Tuple[Tuple[float, float], ...]] = {
    1: ((-10.0, -2.0),),
    2: ((-50.0, -30.0), (-10.0, -2.5)),
    3: ((-8.0, -4.0), (-50.0, -30.0), (-10.0, -2.5)),
    4: ((-3.4, -2.0), (-8.0, -4.0), (-50.0, -30.0), (-10.0, -2.5)),
}


class CartPole(Task):
    config_type = CartPoleConfig

    def __init__(self, config: CartPoleConfig | None = None):
        config = CartPoleConfig() if config is None else config
        if not isinstance(config, CartPoleConfig):
            raise TypeError("config must be CartPoleConfig")
        self.config = config
        self.dim = config.dim
        self.objective = config.objective
        self._objective_spec = get_cartpole_objective_spec(config.objective)
        self._objective_label = self._objective_spec.display_name
        self.is_minimization = self._objective_spec.is_minimization
        self.bounds = torch.tensor(_CARTPOLE_BOUNDS[self.dim], dtype=torch.float64).T
        validate_bounds(self.bounds, self.dim)
        self._sim = CartPoleSimulator(
            dim=self.dim,
            simulation_noise=False if config.noise is None else asdict(config.noise),
        )

    @property
    def name(self) -> str:  # type: ignore[override]
        label = self._objective_label or self.objective.upper()
        return f"CartPole-{label}-{self.dim}D"

    def _evaluate(self, theta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        theta_row = theta.detach().to(dtype=torch.float64).cpu()
        trajectory = self._sim.run_episode(theta_row, generator=self.generator)
        value_tensor = self._objective_spec.evaluate(self._sim, trajectory)
        trajectory_out = detach_trajectory(trajectory, dtype=theta.dtype, device=theta.device)

        info = {
            "theta": theta.detach().clone(),
            "trajectory": trajectory_out,
        }

        return value_tensor, info

    @classmethod
    def available_configs(cls) -> list[CartPoleConfig]:
        """Enumerate standard variants without constructing simulators."""
        return [
            CartPoleConfig(dim=dim, objective=objective, noise=noise)
            for dim in (1, 2, 3, 4)
            for objective in list_cartpole_objectives()
            for noise in (None, CartPoleNoise())
        ]
