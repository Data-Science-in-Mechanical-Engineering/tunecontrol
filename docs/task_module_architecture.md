# Problem families and configurations

A problem family is a `Task` subclass with a validated configuration dataclass.
The family owns its simulator, objectives, and the combinations worth enumerating.
There is no requirement that all families share dimension or noise options.

## Construct a problem

```python
from tunecontrol import CartPole, CartPoleConfig, CartPoleNoise

config = CartPoleConfig(dim=2, objective="mae", noise=CartPoleNoise())
problem = CartPole(config)
value, info = problem.evaluate(problem.bounds.mean(dim=0))
```

The equivalent registry call is:

```python
import tunecontrol as tc
problem = tc.make("cartpole", config={
    "dim": 2, "objective": "mae",
    "noise": {"initial_condition_std": 0.0005, "process_noise_std": 0.0005},
})
```

`make` accepts a family configuration object, a mapping, or no configuration to
use the family's defaults. Unknown fields and invalid configurations raise errors.
`noise=None` means deterministic. Configurations are frozen dataclasses and support
`dataclasses.asdict` and JSON round-trips; record the package version separately.

## Discover variants

- `tc.list_problems()` lists family names without importing their implementations.
- `tc.describe(name)` returns registration metadata without constructing a simulator.
- `tc.available_configs(name)` imports the selected family and lists standard configurations.
- `CartPole.available_configs()` provides the same list directly in Python.

Configuration enumeration never constructs a simulator. Each family decides which
combinations to enumerate. Custom settings may be valid without appearing in this
list. Enumeration is not a curated or versioned benchmark suite.

## Add a family

Use `examples/creating_custom_task.ipynb` as an executable example. A family needs:

1. A configuration dataclass that validates its values in `__post_init__`.
2. A `Task` subclass with `config_type` pointing to that dataclass.
3. Construction from one configuration, stored as `problem.config`.
4. `available_configs()` returning fresh collections of supported configurations.
5. `name`, `dim`, `bounds` shaped `[2, dim]`, and `is_minimization`.
6. `_evaluate(theta)` returning a scalar tensor and a diagnostics dictionary.

Direct Python construction does not require registration. For discovery, register
explicitly in the contributing package's `registration.py` module:

```python
import tunecontrol as tc
from .problem import MassSpring

tc.register_problem(
    "mass_spring",
    MassSpring,
    description="PD tuning for a mass-spring-damper.",
)
```

A lazy entry point keeps optional simulator dependencies out of discovery:

```python
tc.register_problem(
    "my_robot",
    "my_robot_package.problem:RobotProblem",
    description="Controller tuning for a robot.",
)
```

Users import the external package's registration module before looking up its
family. Subclassing does not register anything. Duplicate names are errors, and
import or construction errors propagate rather than silently hiding a problem.
No scanner, plugin dependency, or central construction change is required.

## Suggested layout

```text
tasks/my_problem/
    config.py          # Validated configuration and noise/dynamics settings
    task.py            # Task subclass, evaluation, available_configs()
    plant.py           # Simulator internals
    objectives.py      # Metrics, optionally using ObjectiveRegistry
    visualization.py   # Plotting helpers; import plotting dependencies locally
    __init__.py        # Public family/configuration exports
```

Simulator internals may differ between families. The shared interface is the task
and its evaluation result, not a universal simulator or controller hierarchy.
Objective registries are local to each family. A newly registered objective may
appear in that family's available configurations after registration.

## Contribution checks

Validate configuration types and combinations, bounds and parameter order, scalar
outputs, noise settings, deterministic repeatability, and meaningful failure
behavior. Check direct construction and registry construction agree. Demonstrate
that discovery does not start simulations and configuration serialization round-trips.
Document the physical model, objective definition, units, and initial conditions.

## Evaluation and randomness contract

`evaluate(theta)` validates the floating-point type, exact `(dim,)` shape, finite
parameters, and finite ordered `[2, dim]` bounds before calling `_evaluate`.
Constructors should also call `validate_bounds` from `tunecontrol.tasks.base` after
setting bounds. Finite controllers outside the search box are permitted; no
clipping occurs. `normalize` and `unnormalize` validate bounds and support batches
and extrapolation.

Return exactly one real objective value and a diagnostics dictionary. The common
interface produces a scalar tensor with the input dtype/device. A nonfinite result
(including overflow during conversion) returns NaN and preserves the original
`info` dictionary for inspection. This does not assign a penalty or define a new
notion of controller failure. Other simulator errors propagate normally.

Use `self.generator` for stochastic simulation. It is a task-local CPU generator
that works even if the caller never invokes `setup`. Thread it through simulator
methods and pass `generator=...` to random draws. Do not use global Torch randomness.
`setup(run_seed=...)` resets this stream; `setup()` preserves it. If a family
provides its own setup method for resources, call `super().setup(run_seed)`.

Tests should compare entire seeded trajectory sequences, check that successive
noisy episodes differ, and confirm independence from optimizer random draws and
other tasks. Record the run seed separately from the physical configuration.

## Trajectory format

Use the same structure for built-in problems and new contributions:

| Field | Meaning |
|---|---|
| `time` | Tensor of shape `(N,)`, timestamps in seconds |
| `states` | Tensor of shape `(N, n_states)`, one sample per row |
| `inputs` | Tensor of shape `(N, n_inputs)`, including a column dimension for a single input |
| `state_names`, `state_units` | Tuples identifying the state columns |
| `input_names`, `input_units` | Tuples identifying the input columns |

Use `None` for an unknown unit rather than inventing a unit. Numerical trajectory
fields returned by built-in tasks use the controller tensor's dtype and device;
metadata is preserved. Family-specific fields such as CartPole's `reference`
may be added. Objective code and plotters consume this same format.

States and inputs in a row correspond to the row's timestamp: the controller
computes that input from that state, before advancing the simulation. Document
which endpoints are recorded and whether the last input drives a transition.
The built-in tasks use uniform sampling, but the timestamps are explicit so a
future problem can document a different sampling scheme.
