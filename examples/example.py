"""Example CLI for sampling Bayesian optimization tasks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import torch

from tunecontrol import Task, make
from tunecontrol.tasks import list_task_names, normalize, unnormalize


def _select_use_case(names: list[str], preferred: Iterable[str]) -> Tuple[str, Task]:
    """Select the first task that can be instantiated.

    Args:
        names: Registry keys for available tasks.
        preferred: Candidate names to try before the remaining tasks.

    Returns:
        Tuple[str, Task]: The selected registry key and instantiated task.

    Raises:
        RuntimeError: If no task can be instantiated successfully.
    """
    last_error: Exception | None = None
    ordered = list(dict.fromkeys([*preferred, *names]))
    for name in ordered:
        if name not in names:
            continue
        try:
            task = make(name)
        except Exception as exc:  # pragma: no cover - informative print only
            last_error = exc
            continue
        return name, task
    msg = "No usable tasks found"
    if last_error is not None:
        msg += f": last error was {last_error!r}"
    raise RuntimeError(msg)


def main() -> None:
    """Run an interactive example demonstrating basic task usage."""
    print("=== Step 1: Discover available use cases ===")
    task_names = list_task_names()
    for name in task_names:
        print(f" - {name}")

    print("\n=== Step 2: Initialize a use case ===")
    chosen_name, task = _select_use_case(
        task_names,
        preferred=(
            "cartpole/2d/mae/deterministic",
            "cascaded_tank/2d/logsse/default_noise",
        ),
    )
    print(f"Loaded '{chosen_name}' -> {task.name} (dim={task.dim}, minimization={task.is_minimization})")
    task.setup()

    generator = torch.Generator().manual_seed(0)

    print("\n=== Step 3: Evaluate random samples ===")
    best_value = float("inf") if task.is_minimization else -float("inf")
    best_x = None
    for step in range(10):
        x_norm = torch.rand(task.dim, generator=generator, dtype=torch.float64)
        x = unnormalize(x_norm, task.bounds)
        value_tensor, _ = task.evaluate(x)
        value = float(value_tensor.item())
        improved = value < best_value if task.is_minimization else value > best_value
        tag = " <= best so far" if improved else ""
        print(f"step={step:02d} value={value:+.4f}{tag}")
        if improved:
            best_value = value
            best_x = x

    if best_x is not None:
        print("\n=== Step 4: Inspect best sample ===")
        best_x_norm = normalize(best_x, task.bounds)
        print(f"Best value {best_value:+.4f}")
        print(f"Original space: {best_x.tolist()}")
        print(f"Normalized space: {best_x_norm.tolist()}")

    if getattr(task, "has_known_optimum", False):
        try:
            opt_val = float(task.optimal_value)
            print(f"Known global optimum: {opt_val:+.4f}")
        except Exception:
            pass

    print("\n=== Step 5: Clean up ===")
    task.teardown()
    print("Done.")


if __name__ == "__main__":
    main()
