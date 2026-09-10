"""Base abstractions and utilities for benchmark tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from ..random import new_generator


def validate_bounds(bounds: Tensor, dim: int) -> None:
    """Validate a finite, strictly ordered search box."""
    if type(dim) is not int or dim < 1:
        raise ValueError("dim must be a positive integer")
    if not isinstance(bounds, Tensor) or not bounds.is_floating_point():
        raise TypeError("bounds must be a floating-point tensor")
    if bounds.shape != (2, dim):
        raise ValueError(f"bounds must have shape (2, {dim})")
    if not torch.isfinite(bounds).all():
        raise ValueError("bounds must be finite")
    if not (bounds[1] > bounds[0]).all():
        raise ValueError("Each upper bound must be greater than its lower bound")


class Task(ABC):
    """Abstract base class for benchmark tasks.

    A task evaluates one controller parameter vector in its original units.
    """

    name: str
    dim: int
    bounds: Tensor  # shape: [2, dim], lower then upper, original space
    is_minimization: bool

    @property
    def generator(self) -> torch.Generator:
        """Task-local CPU random stream; usable without calling setup first."""
        if not hasattr(self, "_generator"):
            self._generator = new_generator()
        return self._generator

    def setup(self, run_seed: int | None = None) -> None:
        """Reset the episode random sequence when a seed is supplied.

        Without a seed, preserve the current sequence (or initialize a private
        stream from system entropy on first use). Never seed Torch globally.
        """
        if run_seed is not None:
            if type(run_seed) is not int:
                raise TypeError("run_seed must be an integer or None")
            if not -(2**63) <= run_seed < 2**64:
                raise ValueError("run_seed must be in [-2**63, 2**64 - 1]")
            self.generator.manual_seed(run_seed)
        else:
            _ = self.generator

    def teardown(self) -> None:
        """Release external resources."""
        return None

    @abstractmethod
    def _evaluate(self, theta: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Evaluate on the original input space; returns value and metadata."""
        raise NotImplementedError

    def evaluate(self, theta: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Evaluate one finite floating-point controller vector.

        Bounds define the allowed controller domain, including both endpoints.
        Out-of-bounds controllers raise ValueError before simulation. Nonfinite objective
        results return NaN with the diagnostics dictionary preserved.
        """
        if not isinstance(theta, Tensor) or not theta.is_floating_point():
            raise TypeError("theta must be a real floating-point torch.Tensor")
        validate_bounds(self.bounds, self.dim)
        if theta.shape != (self.dim,):
            raise ValueError(f"theta must have shape ({self.dim},), got {tuple(theta.shape)}")
        if not torch.isfinite(theta).all():
            raise ValueError("theta must contain only finite values")

        bounds = self.bounds.to(dtype=theta.dtype, device=theta.device)
        if ((theta < bounds[0]) | (theta > bounds[1])).any():
            raise ValueError("theta must lie within the declared bounds (inclusive)")

        value, info = self._evaluate(theta)
        if not isinstance(info, dict):
            raise TypeError("Task diagnostics must be a dictionary")
        if isinstance(value, float):
            value = torch.as_tensor(value, dtype=torch.float64)
        else:
            value = torch.as_tensor(value)
        if value.is_complex() or value.dtype == torch.bool:
            raise TypeError("Task objective must be a real number")
        if value.numel() != 1:
            raise ValueError("Task objective must contain exactly one scalar value")
        value = value.to(dtype=theta.dtype, device=theta.device).reshape(())
        if not torch.isfinite(value):
            value = torch.full_like(value, float("nan"))
        return value, info


def _transform_bounds(theta: Tensor, bounds: Tensor | np.ndarray) -> Tensor:
    if not isinstance(theta, Tensor) or not theta.is_floating_point():
        raise TypeError("theta must be a real floating-point torch.Tensor")
    if theta.ndim < 1 or theta.shape[-1] < 1:
        raise ValueError("theta must have a nonempty final parameter dimension")
    if not torch.isfinite(theta).all():
        raise ValueError("theta must contain only finite values")
    original = torch.as_tensor(bounds)
    if original.is_complex() or original.dtype == torch.bool:
        raise TypeError("bounds must be real numbers")
    b = original.to(dtype=theta.dtype, device=theta.device)
    validate_bounds(b, theta.shape[-1])
    return b


def normalize(theta: Tensor, bounds: Tensor | np.ndarray) -> Tensor:
    """Map vectors or batches to the unit cube; allow extrapolation."""
    lower, upper = _transform_bounds(theta, bounds)
    return (theta - lower) / (upper - lower)


def unnormalize(theta: Tensor, bounds: Tensor | np.ndarray) -> Tensor:
    """Map vectors or batches to original units; allow extrapolation."""
    lower, upper = _transform_bounds(theta, bounds)
    return lower + theta * (upper - lower)
