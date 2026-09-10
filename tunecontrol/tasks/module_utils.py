"""Shared helpers for task modules."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Mapping

import torch


Trajectory = Dict[str, Any]

__all__ = [
    "ObjectiveSpec",
    "ObjectiveRegistry",
    "Trajectory",
    "detach_trajectory",
]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Metadata describing how to evaluate a task objective."""

    evaluate_fn: Callable[[Any, Trajectory], torch.Tensor | float]
    display_name: str
    is_minimization: bool = True

    def evaluate(self, simulator: Any, trajectory: Trajectory) -> torch.Tensor:
        value = self.evaluate_fn(simulator, trajectory)
        return torch.as_tensor(value).squeeze()


class ObjectiveRegistry:
    """Lightweight registry for task-specific objectives."""

    def __init__(self) -> None:
        self._specs: Dict[str, ObjectiveSpec] = {}

    def register(
        self,
        name: str,
        evaluate_fn: Callable[[Any, Trajectory], torch.Tensor | float],
        *,
        display_name: str | None = None,
        is_minimization: bool = True,
    ) -> ObjectiveSpec:
        key = _normalize(name)
        spec = ObjectiveSpec(
            evaluate_fn=evaluate_fn,
            display_name=display_name or name.upper(),
            is_minimization=is_minimization,
        )
        self._specs[key] = spec
        return spec

    def add(self, name: str, spec: ObjectiveSpec) -> ObjectiveSpec:
        key = _normalize(name)
        self._specs[key] = spec
        return spec

    def get(self, name: str) -> ObjectiveSpec:
        key = _normalize(name)
        if key not in self._specs:
            raise KeyError(f"Unknown objective '{name}'. Available: {', '.join(self.list())}")
        return self._specs[key]

    def list(self) -> list[str]:
        return sorted(self._specs.keys())

    def items(self) -> Iterable[tuple[str, ObjectiveSpec]]:
        return self._specs.items()


def detach_trajectory(
    trajectory: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Trajectory:
    """Detach a trajectory dictionary to the requested dtype/device."""
    out: Trajectory = {}
    for key, value in trajectory.items():
        out[key] = (value.detach().to(dtype=dtype, device=device)
                    if isinstance(value, torch.Tensor) else deepcopy(value))
    return out
