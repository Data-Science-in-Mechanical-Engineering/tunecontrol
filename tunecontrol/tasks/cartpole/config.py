"""Validated configuration for the CartPole family."""
from dataclasses import dataclass
from math import isfinite

from .objectives import list_cartpole_objectives


@dataclass(frozen=True)
class CartPoleNoise:
    initial_condition_std: float = 0.0005
    process_noise_std: float = 0.0005

    def __post_init__(self):
        for value in (self.initial_condition_std, self.process_noise_std):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Noise standard deviations must be real numbers")
            if not isfinite(value) or value < 0:
                raise ValueError("Noise standard deviations must be finite and nonnegative")


@dataclass(frozen=True)
class CartPoleConfig:
    dim: int = 2
    objective: str = "mae"
    noise: CartPoleNoise | None = None

    def __post_init__(self):
        if type(self.dim) is not int or self.dim not in (1, 2, 3, 4):
            raise ValueError("dim must be one of 1, 2, 3, 4")
        if self.objective not in list_cartpole_objectives():
            raise ValueError(f"Unknown CartPole objective: {self.objective!r}")
        if isinstance(self.noise, dict):
            object.__setattr__(self, "noise", CartPoleNoise(**self.noise))
        if self.noise is not None and not isinstance(self.noise, CartPoleNoise):
            raise TypeError("noise must be CartPoleNoise or None")
