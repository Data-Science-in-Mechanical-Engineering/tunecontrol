"""Base abstractions and utilities for benchmark tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor


# Global registries for tasks
# - CLASS registry: auto-populated via Task.__init_subclass__ for subclasses
# - FACTORY registry: populated via discovery of zero-arg factory functions or the
#   @register_task decorator.
TASK_CLASS_REGISTRY: Dict[str, Type["Task"]] = {}
TASK_FACTORY_REGISTRY: Dict[str, Callable[..., "Task"]] = {}


def _normalize_key(s: str) -> str:
    """Canonicalize registry keys."""
    return s.strip().lower().replace(" ", "_")


class Task(ABC):
    """Abstract base class for benchmark tasks.

    Tasks expose application-specific objectives to the benchmarking engine.
    Implementers describe the original search space while the engine manages
    normalization and batching.
    """

    name: str
    dim: int
    bounds: Tensor  # shape: [2, dim], lower then upper, original space
    is_minimization: bool

    # Auto-register subclasses by a canonical key
    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return

        keys: list[str] = []
        alias = getattr(cls, "name", None)
        if isinstance(alias, str):
            keys.append(_normalize_key(alias))

        class_key = _normalize_key(cls.__name__)
        if class_key.endswith("task"):
            class_key = class_key[:-4]
        keys.append(class_key)

        for key in keys:
            TASK_CLASS_REGISTRY[key] = cls

    def setup(self, run_seed: int | None = None) -> None:
        """Prepare external resources."""
        return None

    def teardown(self) -> None:
        """Release external resources."""
        return None

    @abstractmethod
    def _evaluate(self, theta: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Evaluate on the original input space; returns value and metadata."""
        raise NotImplementedError

    def evaluate(self, theta: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Evaluate the objective on the original input space."""
        if not isinstance(theta, torch.Tensor):
            raise TypeError("theta must be a torch.Tensor")

        if theta.ndim != 1:
            raise ValueError("theta must be a 1D tensor")

        value, info = self._evaluate(theta)
        if not isinstance(value, torch.Tensor):  # pragma: no cover - defensive
            value = torch.as_tensor(value)

        value = value.to(dtype=theta.dtype, device=theta.device).squeeze()
        return value, info


def register_task(name: str | None = None):
    """Register a factory function or constructor as a task."""

    def _decorator(obj):
        key = _normalize_key(name if name is not None else getattr(obj, "__name__", str(obj)))
        TASK_FACTORY_REGISTRY[key] = obj  # type: ignore[assignment]
        return obj

    return _decorator


def normalize(theta: Tensor, bounds: Tensor | np.ndarray) -> Tensor:
    """Map points from original space to the unit hypercube."""
    if not isinstance(bounds, torch.Tensor):
        b = torch.as_tensor(bounds, dtype=theta.dtype, device=theta.device)
    else:
        b = bounds.to(device=theta.device, dtype=theta.dtype)
    lower, upper = b[0], b[1]
    return (theta - lower) / (upper - lower)


def unnormalize(theta: Tensor, bounds: Tensor | np.ndarray) -> Tensor:
    """Map points from the unit hypercube back to original units."""
    if not isinstance(bounds, torch.Tensor):
        b = torch.as_tensor(bounds, dtype=theta.dtype, device=theta.device)
    else:
        b = bounds.to(device=theta.device, dtype=theta.dtype)
    lower, upper = b[0], b[1]
    return lower + theta * (upper - lower)


# Backwards compatibility alias
register_use_case = register_task
