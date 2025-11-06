"""Task registry and discovery utilities."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .base import (
    Task,
    normalize,
    unnormalize,
    register_task,
    TASK_CLASS_REGISTRY,
    TASK_FACTORY_REGISTRY,
    _normalize_key,
)

__all__ = [
    "Task",
    "normalize",
    "unnormalize",
    "registry",
    "list_task_names",
    "list_tasks",
    "from_name",
    "register_task",
]


_DISCOVERED = False
logger = logging.getLogger(__name__)


def _auto_register_factories_from_module(mod) -> None:
    """Register zero-argument factory functions defined in ``mod``.

    Args:
        mod: Imported module to inspect for factory functions.
    """
    for attr_name in dir(mod):
        if not attr_name.endswith("Task"):
            continue
        obj = getattr(mod, attr_name)
        if inspect.isclass(obj):
            # Classes register themselves via Task.__init_subclass__.
            continue
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        # Must have no required parameters
        has_required = any(
            p.default is inspect._empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in sig.parameters.values()
        )
        if has_required:
            continue
        key = _normalize_key(attr_name)
        if key == "task":
            continue
        if key not in TASK_FACTORY_REGISTRY:
            TASK_FACTORY_REGISTRY[key] = obj  # type: ignore[assignment]


def _auto_register_zero_arg_class_factories() -> None:
    """Register classes that can be instantiated without arguments."""
    for key, cls in list(TASK_CLASS_REGISTRY.items()):
        if key in TASK_FACTORY_REGISTRY:
            continue
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        has_required = any(
            p.default is inspect._empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in params
        )
        if has_required:
            continue
        try:
            cls()
        except Exception:
            continue
        TASK_FACTORY_REGISTRY[key] = (
            lambda *args, cls=cls, **kwargs: cls(*args, **kwargs)
        )  # type: ignore[assignment]


def _discover() -> None:
    """Import task modules and register discovered factories."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    package_path = Path(__file__).parent
    # Import all submodules to trigger __init_subclass__ and expose factories
    for m in pkgutil.iter_modules([str(package_path)]):
        name = m.name
        if name in {"__init__", "base", "__pycache__"}:
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            logger.warning("Skipping task module %s due to missing optional dependency: %s", name, exc)
            continue
        except ImportError as exc:
            logger.warning("Unable to import task module %s: %s", name, exc)
            continue
        # Auto-register zero-arg factory functions in each module
        _auto_register_factories_from_module(mod)

    _auto_register_zero_arg_class_factories()

    # Add friendly aliases for better CLI UX
    _DISCOVERED = True


def registry() -> Dict[str, Callable[..., Task]]:
    """Return a mapping of registry keys to task factories."""
    _discover()
    # Return a shallow copy to prevent external mutation
    return dict(TASK_FACTORY_REGISTRY)


def list_task_names() -> List[str]:
    """Return display-ready registry keys."""
    names = []
    for key in registry().keys():
        cleaned = key[:-4] if key.endswith("task") else key
        names.append(cleaned)
    # dict.fromkeys preserves order while de-duplicating
    return sorted(dict.fromkeys(names))


def list_tasks() -> List[Task]:
    """Instantiate every discovered zero-argument task."""
    reg = registry()
    tasks: List[Task] = []
    for key in list_task_names():
        factory = reg[key]
        try:
            task = factory()
        except Exception:
            continue
        if isinstance(task, Task):
            tasks.append(task)
    return tasks


def from_name(name: str, *args, **overrides) -> Task:
    """Instantiate a task by registry key.

    Args:
        name: Registry key for the desired task.
        *args: Positional arguments forwarded to the factory.
        **overrides: Keyword arguments forwarded to the factory.

    Returns:
        Task: Instantiated task.

    Raises:
        ValueError: If the name is not registered.
    """
    _discover()
    # normalize legacy ':' to '_'
    key = name.strip().replace(":", "_")

    reg = registry()
    if key in reg:
        fac = reg[key]
        return fac(*args, **overrides)  # type: ignore[misc]
    raise ValueError(f"Unknown task: {name}")
