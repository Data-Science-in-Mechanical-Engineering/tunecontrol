"""Cart-pole task package exposing registry utilities and the main task."""

from .objectives import (
    CartPoleObjectiveSpec,
    get_cartpole_objective_spec,
    list_cartpole_objectives,
    register_cartpole_objective,
)
from .plant import CartPoleSimulator
from .task import CartPole
from .visualization import plot_episode

__all__ = [
    "CartPole",
    "CartPoleSimulator",
    "CartPoleObjectiveSpec",
    "register_cartpole_objective",
    "get_cartpole_objective_spec",
    "list_cartpole_objectives",
    "plot_episode",
]

from .config import CartPoleConfig, CartPoleNoise

__all__ += ["CartPoleConfig", "CartPoleNoise"]
