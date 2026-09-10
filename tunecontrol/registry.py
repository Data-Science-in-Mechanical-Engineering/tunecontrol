"""Explicit, lazy registration of problem families.

A family implements config_type, available_configs(), and construction from
one validated configuration. Listing families never imports their simulators.
"""
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from importlib import import_module
from typing import Any

from .tasks.base import Task


@dataclass(frozen=True)
class ProblemFamily:
    name: str
    constructor: type[Task] | str
    description: str

    def load(self) -> type[Task]:
        constructor = self.constructor
        if isinstance(constructor, str):
            module, attr = constructor.split(":")
            constructor = getattr(import_module(module), attr)
        if not isinstance(constructor, type) or not issubclass(constructor, Task):
            raise TypeError(f"{self.name}: constructor must be a Task subclass")
        if not isinstance(getattr(constructor, "config_type", None), type):
            raise TypeError(f"{self.name}: family must declare config_type")
        if not is_dataclass(constructor.config_type):
            raise TypeError(f"{self.name}: config_type must be a dataclass")
        if not callable(getattr(constructor, "available_configs", None)):
            raise TypeError(f"{self.name}: family must implement available_configs()")
        return constructor


_FAMILIES: dict[str, ProblemFamily] = {}


def _name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Problem name must be a string")
    name = name.strip().lower()
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Problem names may contain letters, digits, underscores and hyphens")
    return name


def register_problem(name: str, constructor: type[Task] | str, *, description: str = "") -> None:
    """Register a family class or lazy 'module:Class' entry point explicitly."""
    name = _name(name)
    if name in _FAMILIES:
        raise ValueError(f"Problem already registered: {name}")
    family = ProblemFamily(name, constructor, description)
    if isinstance(constructor, str):
        if constructor.count(":") != 1 or not all(constructor.split(":")):
            raise ValueError("Entry point must have the form 'module:Class'")
    else:
        family.load()
    _FAMILIES[name] = family


def list_problems() -> list[str]:
    """List registered family names without importing or constructing them."""
    return sorted(_FAMILIES)


def describe(name: str) -> ProblemFamily:
    """Return registration metadata without constructing a simulator."""
    key = _name(name)
    try:
        return _FAMILIES[key]
    except KeyError:
        raise ValueError(f"Unknown problem {name!r}. Available: {', '.join(list_problems())}") from None


def available_configs(name: str) -> list[Any]:
    """Enumerate a family's standard configurations, without simulations.

    This list is not exhaustive of all custom settings accepted by its config.
    Families decide which combinations are meaningful to enumerate.
    """
    family = describe(name).load()
    configs = list(family.available_configs())
    if any(not isinstance(config, family.config_type) for config in configs):
        raise TypeError(f"{name}: available_configs returned an incorrect configuration type")
    return configs


def make(name: str, config: Any = None) -> Task:
    """Construct a family using its config object, a mapping, or its defaults."""
    family = describe(name).load()
    if config is None:
        config = family.config_type()
    elif isinstance(config, Mapping):
        config = family.config_type(**dict(config))
    if not isinstance(config, family.config_type):
        raise TypeError(f"{name}: config must be {family.config_type.__name__} or a mapping")
    return family(config)


register_problem("cartpole", "tunecontrol.tasks.cartpole:CartPole",
                 description="State-feedback tuning for an inverted pendulum on a cart.")
register_problem("cascaded_tank", "tunecontrol.tasks.cascaded_tank:CascadedTank",
                 description="PI controller tuning for two nonlinear cascaded tanks.")
