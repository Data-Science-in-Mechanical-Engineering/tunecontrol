"""Cascaded tank task package exposing registry utilities and the main task."""

from .objectives import (
    CascadedTankObjectiveSpec,
    get_cascaded_tank_objective_spec,
    list_cascaded_tank_objectives,
    register_cascaded_tank_objective,
)
from .plant import CascadedTankSimulator
from .task import CascadedTank
from .visualization import plot_episode

__all__ = [
    "CascadedTank",
    "CascadedTankSimulator",
    "CascadedTankObjectiveSpec",
    "register_cascaded_tank_objective",
    "get_cascaded_tank_objective_spec",
    "list_cascaded_tank_objectives",
    "plot_episode",
]

from .config import CascadedTankConfig, CascadedTankNoise, CascadedTankDynamics

__all__ += ["CascadedTankConfig", "CascadedTankNoise", "CascadedTankDynamics"]
