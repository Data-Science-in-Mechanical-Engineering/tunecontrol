"""Task wrapper for the cascaded tank benchmark."""

from __future__ import annotations

from typing import Dict

import torch

from ..base import Task
from ..module_utils import detach_trajectory, TaskConfig
from .objectives import (
    CascadedTankObjectiveSpec,
    get_cascaded_tank_objective_spec,
    list_cascaded_tank_objectives,
)
from .plant import CascadedTankSimulator, Trajectory

__all__ = ["CascadedTankTask"]


def _bounds_tensor() -> torch.Tensor:
    lower = torch.tensor([0.9, 0.01], dtype=torch.float64)
    upper = torch.tensor([4.5, 0.16], dtype=torch.float64)
    return torch.stack((lower, upper))


class CascadedTankTask(Task):
    """Cascaded tank PI tuning benchmark task."""

    def __init__(
        self,
        *,
        objective: str,
        noise_std: float,
        duration: float = 1000.0,
        target: float = 4.0,
        dynamic_params: Dict[str, float] | None = None,
    ) -> None:
        obj_key = objective.strip().lower()
        try:
            spec = get_cascaded_tank_objective_spec(obj_key)
        except KeyError as exc:
            raise ValueError(
                f"Unknown objective '{objective}'. "
                f"Available options: {', '.join(list_cascaded_tank_objectives())}."
            ) from exc

        self.dim = 2
        self.objective = obj_key
        self._objective_spec: CascadedTankObjectiveSpec = spec
        self._objective_label = spec.display_name
        self.is_minimization = spec.is_minimization
        self._sim = CascadedTankSimulator(
            duration=duration,
            target=target,
            noise_std=noise_std,
            dynamic_kwargs=dict(dynamic_params or {}),
        )
        self.simulation_noise = float(noise_std)
        self.bounds = _bounds_tensor()
        self.data: Trajectory | None = None
        self.config = TaskConfig(
            name="CascadedTank",
            dim=self.dim,
            bounds=self.bounds,
            is_minimization=self.is_minimization,
            metadata={
                "objective": self.objective,
                "duration": duration,
                "target": target,
                "noise_std": noise_std,
            },
        )

    @property
    def name(self) -> str:  # type: ignore[override]
        label = self._objective_label or self.objective.upper()
        return f"CascadedTank-{label}-2D"

    def setup(self, run_seed: int | None = None) -> None:
        if run_seed is not None:
            torch.manual_seed(run_seed)

    def _evaluate(self, theta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        kp, ki = float(theta[0]), float(theta[1])
        trajectory = self._sim.simulate(kp, ki)
        value_tensor = self._objective_spec.evaluate(self._sim, trajectory)
        self.data = trajectory

        trajectory_torch = detach_trajectory(trajectory, dtype=theta.dtype, device=theta.device)

        info = {
            "theta": theta.detach().clone(),
            "trajectory": trajectory_torch,
        }

        value_tensor = value_tensor.to(dtype=theta.dtype, device=theta.device)
        return value_tensor, info
