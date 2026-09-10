# TuneControl

> A lightweight benchmark suite for **black-box controller tuning**.

<p align="center">
  <img src="https://raw.githubusercontent.com/Data-Science-in-Mechanical-Engineering/tunecontrol/main/docs/bo_example.png" alt="Controller tuning workflow in TuneControl: θ → controller → closed-loop simulator → cost" width="75%"/>
</p>

TuneControl provides reproducible controller-tuning tasks with a consistent API.

---

## Contents
- [Quick Start](#quick-start)
- [Reproducibility and evaluation errors](#reproducibility-and-evaluation-errors)
- [Features at a Glance](#features-at-a-glance)
- [What is controller tuning?](#what-is-controller-tuning)
- [Benchmark problems](#benchmark-problems)
- [Benchmark catalogue](#benchmark-catalogue)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

Currently supports Python 3.12.

**Install from the public repository**
```bash
python -m pip install "tunecontrol @ git+https://github.com/Data-Science-in-Mechanical-Engineering/tunecontrol.git"
```
For the BoTorch examples, use `tunecontrol[botorch]` in place of `tunecontrol` in that command.

Once the first release is published to PyPI, install it with:
```bash
python -m pip install tunecontrol
# Include BoTorch for the Bayesian optimization examples:
python -m pip install 'tunecontrol[botorch]'
```
The base install pulls in `torch>=2.2`, `numpy>=1.26`, `scipy>=1.11`, and `matplotlib`.

For local development, run `python -m pip install -e '.[dev]'` from the repository root.
Maintainers can follow [the release guide](docs/releasing.md) to publish to PyPI.

**Construct a problem explicitly**
```python
from tunecontrol import CartPole, CartPoleConfig

problem = CartPole(CartPoleConfig(dim=2, objective="mae", noise=None))
theta = problem.bounds.mean(dim=0)
value, info = problem.evaluate(theta)
print(value.item(), sorted(info.keys()))
```

**Discover families and standard configurations**
```python
import tunecontrol as tc

for name in tc.list_problems():
    print(name, tc.describe(name).description)
    for config in tc.available_configs(name):
        print(config)

# The registry uses the same validated configurations as direct construction.
problem = tc.make("cartpole", config={"dim": 2, "objective": "mae"})
```

Each family enumerates its meaningful standard variants: 24 for CartPole and 10
for cascaded tanks. Custom configurations are also supported. These are variants
of two physical systems; there is no separately curated benchmark suite.

**Configure noise and save the configuration**
```python
from dataclasses import asdict
import json
from tunecontrol import CartPoleNoise

config = CartPoleConfig(dim=2, objective="mae", noise=CartPoleNoise())
problem = CartPole(config)
saved = json.dumps(asdict(problem.config))
restored = tc.make("cartpole", config=json.loads(saved))
```

`noise=None` selects a deterministic simulation. Noise objects contain the actual
numerical settings, so configuration files do not depend on a string such as
"default". Record the package version alongside the configuration for experiments.

## Reproducibility and evaluation errors

Each problem owns an independent random stream. Seed the problem separately from
your optimizer; successive noisy evaluations advance its stream:

```python
problem.setup(run_seed=42)
first, _ = problem.evaluate(theta)
second, _ = problem.evaluate(theta)
problem.setup(run_seed=42)
replayed_first, _ = problem.evaluate(theta)
```

`setup()` without a seed preserves the current stream; a new problem uses system
entropy. Seeded replay is intended for the same configuration and software/runtime
versions, not guaranteed bitwise across platforms. Simulations currently use CPU
random generators even when results are returned to the input tensor's device.

`evaluate` requires a finite floating-point tensor of shape `(problem.dim,)`.
Search bounds are not clipping rules: finite out-of-bounds controllers may be
explored. Invalid inputs raise `TypeError` or `ValueError` before simulation.
Undefined or nonfinite objectives return `(NaN, info)`, preserving the diagnostics.
Check `torch.isnan(value)` before using a result in an optimizer. No failure penalty is substituted. For example,
a tank rise-time objective is undefined if the threshold is not reached during
the episode.

## Features at a Glance

| Capability | Details |
|------------|---------|
| Uniform evaluate API | Every task uses a single `evaluate(θ)` entry point that returns a scalar objective and metadata payload. |
| Deterministic & noisy twins | Flip between deterministic and noisy variants without changing code—ideal for hardware-style benchmarking. |
| Rich trajectory metadata | Each call returns time-series data (states, inputs, references) for plotting, debugging, and controller diagnostics. |
| Reproducible tasks | Explicit configurations describe each problem; deterministic variants provide repeatable evaluations. |

---

## What is controller tuning?

Controller tuning chooses the parameters `θ` of a **fixed-structure controller** (e.g., state feedback, PI) so that the **closed-loop** system meets a performance objective on a given scenario.

### How TuneControl turns this into a *task*
Each task encapsulates five components behind a clean API:
1. **Decision variables (`θ`)** – a vector of controller parameters with exposed bounds.
2. **Controller structure** – fixed form; only the parameter values change.
3. **Closed-loop simulator** – simulation with optional noise.
4. **Objective `J(θ)`** – a scalar such as MAE, LQR, ITAE; or LogSSE, SSE, quadratic, rise time, overshoot.
5. **Metadata** – full trajectory arrays and diagnostics for plotting/debugging.

### Deterministic vs. noisy
- **Deterministic:** no random disturbances; repeated calls at the same `θ` give identical results.
- **Noisy:** includes process noise and, for CartPole, initial-condition perturbations; repeated calls vary, resembling hardware experiments.

Both variants expose the same interface:
```python
value, info = task.evaluate(theta)
```
Explore objective definitions and modelling details in `docs/cartpole_task_doc.md` and `docs/cascaded_tank_task_doc.md`.

## Benchmark problems

### CartPole
Linear state-feedback control of an inverted pendulum on a cart. Depending on the variant, between one and four feedback gains are optimised.
- **Objectives:** MAE, LQR, ITAE (details in `docs/cartpole_task_doc.md`)
- **Noise variants:** deterministic and noisy (process + initial-condition noise)

### Cascaded Tanks
PI controller tuning for a nonlinear cascaded tank process with soft and hard nonlinearities.
- **Objectives:** LogSSE, SSE, quadratic, rise time, overshoot (details in `docs/cascaded_tank_task_doc.md`)
- **Noise variants:** deterministic and noisy (additive state noise)

---

## Benchmark catalogue (compact)

The catalogue contains problem families; each family owns configuration validation
and enumeration. Listing families does not import their simulators, and enumerating
configurations does not construct simulations.

| Family | Dimensions | Objectives | Noise configurations | Standard variants |
|---|---|---|---|---|
| `cartpole` | 1–4 | `mae`, `lqr`, `itae` | `None`, `CartPoleNoise()` | 24 |
| `cascaded_tank` | 2 | `sse`, `logsse`, `quadratic`, `rise_time`, `overshoot` | `None`, `CascadedTankNoise()` | 10 |

```python
from tunecontrol import CartPole

for config in CartPole.available_configs():
    problem = CartPole(config)
    # Run your optimizer on this problem.
```

The constructor may accept more configurations than those enumerated. For example:

```python
from tunecontrol import CascadedTank, CascadedTankConfig, CascadedTankNoise

problem = CascadedTank(CascadedTankConfig(
    objective="sse", duration=500.0, noise=CascadedTankNoise(std=0.01),
))
```

<p align="center">
  <img src="docs/figures/deterministic_2d_gallery.png" alt="Objective landscape gallery for deterministic 2D tasks"/>

*Objective landscapes for every deterministic 2D task. Darker colours indicate lower cost.*
</p>

---

## Examples

See [the example guide](examples/README.md) for installation and running instructions.

- [Evaluate and plot](examples/quickstart.py): try both built-in problems.
- [Discover and configure](examples/discover_problems.py): list families and save configurations.
- [Bayesian optimization](examples/standard_bo_for_controller_tuning.ipynb): tune one controller gain with BoTorch.
- [Add a problem](examples/creating_custom_task.ipynb): implement a mass-spring-damper family.

---

## Contributing
- Read [the family API and contribution guide](docs/task_module_architecture.md).
- Prototype new benchmarks by following `examples/creating_custom_task.ipynb` and mirroring the doc structure under `docs/`.
- Add narrative docs or figures for new tasks so they appear alongside the existing CartPole and cascaded-tank guides.
- Open an issue or pull request outlining the task, expected objective, and any stochastic settings; include sample trajectories where possible.

---

## Citation
TuneControl was introduced in [*A Decade of Bayesian Optimization for Controller Tuning and Robot Learning: Tutorial, Review, and Future Prospects*](https://arxiv.org/abs/2609.09403). Please cite it if TuneControl supports your research.

```bibtex
@misc{stenger2026decade,
    title = {A Decade of {Bayesian} Optimization for Controller Tuning and Robot Learning: Tutorial, Review, and Future Prospects},
    author = {Stenger, David and Brunzema, Paul and Menn, Johanna and von Rohr, Alexander and Schoellig, Angela P. and Trimpe, Sebastian},
    year = {2026},
    eprint = {2609.09403},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    doi = {10.48550/arXiv.2609.09403},
    url = {https://arxiv.org/abs/2609.09403}
}
```

---

## License
Released under the [MIT License](LICENCE.txt).
