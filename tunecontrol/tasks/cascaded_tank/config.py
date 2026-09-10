"""Validated configuration for the cascaded tank family."""
from dataclasses import dataclass
from math import isfinite

from .objectives import list_cascaded_tank_objectives


def _number(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class CascadedTankNoise:
    std: float = 0.005

    def __post_init__(self):
        _number("noise std", self.std)


@dataclass(frozen=True)
class CascadedTankDynamics:
    k1: float = 0.2143
    k2: float = 0.2165
    k3: float = 0.1654
    k4: float = 0.1371
    time_step: float = 4.0
    initial_state: tuple[float, float] = (5.8113, 1.9440)

    def __post_init__(self):
        for name in ("k1", "k2", "k3", "k4", "time_step"):
            _number(name, getattr(self, name), positive=True)
        state = tuple(self.initial_state)
        if len(state) != 2:
            raise ValueError("initial_state must contain two tank levels")
        for value in state:
            _number("initial level", value)
            if value > 10:
                raise ValueError("initial levels must be in [0, 10]")
        object.__setattr__(self, "initial_state", state)


@dataclass(frozen=True)
class CascadedTankConfig:
    objective: str = "sse"
    noise: CascadedTankNoise | None = None
    duration: float = 1000.0
    target: float = 4.0
    dynamics: CascadedTankDynamics = CascadedTankDynamics()

    def __post_init__(self):
        if self.objective not in list_cascaded_tank_objectives():
            raise ValueError(f"Unknown cascaded tank objective: {self.objective!r}")
        _number("duration", self.duration, positive=True)
        _number("target", self.target, positive=True)
        if isinstance(self.noise, dict):
            object.__setattr__(self, "noise", CascadedTankNoise(**self.noise))
        if self.noise is not None and not isinstance(self.noise, CascadedTankNoise):
            raise TypeError("noise must be CascadedTankNoise or None")
        if isinstance(self.dynamics, dict):
            object.__setattr__(self, "dynamics", CascadedTankDynamics(**self.dynamics))
        if not isinstance(self.dynamics, CascadedTankDynamics):
            raise TypeError("dynamics must be CascadedTankDynamics")
