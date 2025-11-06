# Task Module Architecture

Each benchmark task lives in its own Python subpackage under `tunecontrol/tasks`.
The goal is to keep the public API uniform while allowing every task to manage
its simulator-specific internals.  

The recommended layout for a task package is:

```
tasks/<task_name>/
├── __init__.py         # Registry wiring / factory helpers
├── task.py             # Concrete Task subclass (wraps simulator + objectives)
├── plant.py            # Closed-loop simulation utilities
├── objectives.py       # Objective definitions and registry
├── visualization.py    # (Optional) plotting helpers
└── README.md / docs    # (Optional) task-specific notes
```

## Core components

### `task.py`

* Subclass `tunecontrol.tasks.base.Task`.
* Own the task-level configuration (dimensionality, θ-bounds, noise settings).
* Implement `_evaluate(self, theta: Tensor) -> Tuple[Tensor, Dict[str, Any]]`:
  1. Simulate the closed-loop system for a single parameter vector `theta`.
  2. Compute the objective value via the objective registry.
  3. Build the `info` dictionary (e.g. `{ "theta": ..., "trajectory": ... }`).
* Store the last trajectory for convenience (optional).

### `plant.py`

* Provide deterministic helper classes / functions to simulate a closed-loop
  system. This module must **not** know about task registries.
* Should expose a single simulator class with a `simulate(theta)` method that
  returns a trajectory dictionary (PyTorch tensors).
* Keep physics- / simulator-specific parameters local to this module.

### `objectives.py`

* Define objective metrics as callables operating on a simulator and trajectory.
* Register each metric via the shared `ObjectiveRegistry` helper:
  ```python
  from ..module_utils import OBJECTIVES, ObjectiveRegistry
  ```
* Expose helpers `get_<task>_objective_spec()`, `list_<task>_objectives()`.

### `__init__.py`

* Register factories for the task variants (`register_task`).
* Export the public API (Task class, simulator, objective helpers).
* Keep registration logic declarative (e.g., iterate over objectives/noise modes).

### `visualization.py` (optional)

* Provide plotting utilities that accept trajectory dictionaries.
* Keep imports (e.g., `matplotlib`) local to the functions to avoid hard deps.

## Shared utilities

Use the helpers in `tunecontrol/tasks/module_utils.py` to avoid duplication:

* `ObjectiveRegistry` / `ObjectiveSpec` — consistent objective metadata.
* `Trajectory` — shared type alias for simulator outputs.
* `detach_trajectory()` — convert simulator outputs to the caller's dtype/device.
* `TaskConfig` — optional metadata bundle (name, dim, bounds, extras) you can store on the task instance for discovery/serialization.

## Guidelines

* θ bounds belong to `task.py` (they describe the optimization variables).
* Keep simulators framework-agnostic whenever practical (operate on torch/numpy
  tensors but avoid task registry dependencies).
* Return a scalar torch tensor from each objective evaluation; perform type /
  device conversions in `task.py`.
* It's acceptable to duplicate small pieces of code if it keeps the modules
  easier to read. Aim for clarity first, abstraction second.
