"""Task wrapper for the cart-pole benchmark."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch

from ..base import Task
from ..module_utils import detach_trajectory, TaskConfig
from .objectives import (
    CartPoleObjectiveSpec,
    get_cartpole_objective_spec,
    list_cartpole_objectives,
)
from .plant import CartPoleSimulator

__all__ = ["CartPoleTask"]

_CARTPOLE_BOUNDS: Dict[int, Tuple[Tuple[float, float], ...]] = {
    1: ((-10.0, -2.0),),
    2: ((-50.0, -30.0), (-10.0, -2.5)),
    3: ((-8.0, -4.0), (-50.0, -30.0), (-10.0, -2.5)),
    4: ((-3.4, -2.0), (-8.0, -4.0), (-50.0, -30.0), (-10.0, -2.5)),
}


class CartPoleTask(Task):
    """Cart-pole controller tuning benchmark task."""

    @property
    def name(self) -> str:  # type: ignore[override]
        label = self._objective_label or self.objective.upper()
        return f"CartPole-{label}-{self.dim}D"

    def __init__(
        self,
        *,
        dim: int,
        objective: str,
        simulation_noise: Union[Dict[str, float], bool],
    ) -> None:
        obj_key = objective.strip().lower()
        try:
            spec = get_cartpole_objective_spec(obj_key)
        except KeyError as exc:
            raise ValueError(
                f"Unknown objective '{objective}'. "
                f"Available options: {', '.join(list_cartpole_objectives())}."
            ) from exc

        if dim not in _CARTPOLE_BOUNDS:
            raise ValueError(f"Unsupported cart-pole dimension: {dim}")

        self.dim = dim
        self.objective = obj_key
        self._objective_spec = spec
        self._objective_label = spec.display_name
        self.is_minimization = spec.is_minimization
        self._sim = CartPoleSimulator(
            dim=dim,
            simulation_noise=simulation_noise,
        )
        self.simulation_noise = self._sim.simulation_noise
        bounds_tensor = torch.tensor(_CARTPOLE_BOUNDS[dim], dtype=torch.float64).T
        self.bounds = bounds_tensor
        self.config = TaskConfig(
            name="CartPole",
            dim=self.dim,
            bounds=self.bounds,
            is_minimization=self.is_minimization,
            metadata={"objective": self.objective, "simulation_noise": simulation_noise},
        )

    def _evaluate(self, theta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        theta_row = theta.detach().to(dtype=torch.float64).cpu()
        trajectory = self._sim.run_episode(theta_row)
        value_tensor = self._objective_spec.evaluate(self._sim, trajectory)
        trajectory_out = detach_trajectory(trajectory, dtype=theta.dtype, device=theta.device)

        info = {
            "theta": theta.detach().clone(),
            "trajectory": trajectory_out,
        }

        value_tensor = value_tensor.to(dtype=theta.dtype, device=theta.device)
        return value_tensor, info
