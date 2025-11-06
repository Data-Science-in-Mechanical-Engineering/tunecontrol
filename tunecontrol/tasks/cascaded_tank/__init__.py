"""Cascaded tank task package exposing registry utilities and the main task."""

from typing import Dict

from ..base import register_task
from .objectives import (
    CascadedTankObjectiveSpec,
    get_cascaded_tank_objective_spec,
    list_cascaded_tank_objectives,
    register_cascaded_tank_objective,
)
from .plant import CascadedTankSimulator
from .task import CascadedTankTask
from .visualization import plot_episode

__all__ = [
    "CascadedTankTask",
    "CascadedTankSimulator",
    "CascadedTankObjectiveSpec",
    "register_cascaded_tank_objective",
    "get_cascaded_tank_objective_spec",
    "list_cascaded_tank_objectives",
    "plot_episode",
]


_NOISE_VARIANTS: Dict[str, float] = {
    "deterministic": 0.0,
    "default_noise": 0.005,
}


def _register_factories() -> None:
    for objective in list_cascaded_tank_objectives():
        for variant, noise_std in _NOISE_VARIANTS.items():
            key = f"cascaded_tank/2d/{objective}/{variant}"

            @register_task(key)
            def _factory(_objective=objective, _noise=noise_std):
                return CascadedTankTask(objective=_objective, noise_std=_noise)


_register_factories()
