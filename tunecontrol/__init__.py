"""Bayesian optimization benchmark suite.

This package provides a modular framework for evaluating Bayesian
optimization algorithms on pluggable tasks.
"""

from typing import Any

from .tasks import Task, from_name

__all__ = ["make", "Task"]

def make(name: str, /, *args: Any, **overrides: Any) -> Task:
    """Return a registered task by name.

    Args:
        name: Registry key for the desired task (case-insensitive).
        *args: Positional arguments forwarded to the underlying factory.
        **overrides: Keyword arguments forwarded to the underlying factory.

    Returns:
        Task: Instantiated task ready for evaluation via ``task.evaluate``.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty or unknown.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")

    key = name.strip()
    if not key:
        raise ValueError("name must be a non-empty string")

    #try:
    return from_name(key, *args, **overrides)
    #except ValueError as exc:
    #    raise ValueError(f"Unknown use case: {name}") from exc
