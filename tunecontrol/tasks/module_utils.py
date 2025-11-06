"""Shared helpers for task modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping

import torch


Trajectory = Dict[str, torch.Tensor]

__all__ = [
    "ObjectiveSpec",
    "ObjectiveRegistry",
    "Trajectory",
    "detach_trajectory",
    "TaskConfig",
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
    trajectory: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Trajectory:
    """Detach a trajectory dictionary to the requested dtype/device."""
    out: Trajectory = {}
    for key, value in trajectory.items():
        out[key] = value.detach().to(dtype=dtype, device=device)
    return out


@dataclass(frozen=True)
class TaskConfig:
    """Shared configuration payload for task modules."""

    name: str
    dim: int
    bounds: torch.Tensor  # shape: [2, dim]
    is_minimization: bool = True
    metadata: Dict[str, Any] | None = None
