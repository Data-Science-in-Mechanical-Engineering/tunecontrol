"""Cart-pole task package exposing registry utilities and the main task."""

from typing import Dict, Union

from ..base import register_task
from .objectives import (
    CartPoleObjectiveSpec,
    get_cartpole_objective_spec,
    list_cartpole_objectives,
    register_cartpole_objective,
)
from .plant import CartPoleSimulator
from .task import CartPoleTask
from .visualization import plot_episode

__all__ = [
    "CartPoleTask",
    "CartPoleSimulator",
    "CartPoleObjectiveSpec",
    "register_cartpole_objective",
    "get_cartpole_objective_spec",
    "list_cartpole_objectives",
    "plot_episode",
]


_NOISE_VARIANTS: Dict[str, Union[Dict[str, float], bool]] = {
    "deterministic": False,
    "default_noise": {
        "initial_condition_std": 0.0005,
        "process_noise_std": 0.0005,
    },
}


def _register_factories() -> None:
    for dim in (1, 2, 3, 4):
        for objective in list_cartpole_objectives():
            for variant, noise_cfg in _NOISE_VARIANTS.items():
                key = f"cartpole/{dim}d/{objective}/{variant}"

                @register_task(key)
                def _factory(_dim=dim, _objective=objective, _noise=noise_cfg):
                    return CartPoleTask(dim=_dim, objective=_objective, simulation_noise=_noise)


_register_factories()
