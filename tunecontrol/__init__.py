"""Controller-tuning problems with explicit, validated configurations."""
from importlib import import_module

from .registry import available_configs, describe, list_problems, make, register_problem
from .tasks import Task, normalize, unnormalize

_EXPORTS = {
    "CartPole": "cartpole",
    "CartPoleConfig": "cartpole",
    "CartPoleNoise": "cartpole",
    "CascadedTank": "cascaded_tank",
    "CascadedTankConfig": "cascaded_tank",
    "CascadedTankNoise": "cascaded_tank",
    "CascadedTankDynamics": "cascaded_tank",
}
__all__ = ["Task", "make", "list_problems", "describe", "available_configs",
           "register_problem", "normalize", "unnormalize", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".tasks.{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
