# TuneControl

> A lightweight benchmark suite for **black-box controller tuning**.

<p align="center">
  <img src="docs/bo_example.png" alt="Controller tuning workflow in TuneControl: θ → controller → closed-loop simulator → cost" width="75%"/>
</p>

TuneControl provides reproducible controller-tuning tasks with a consistent API.

---

## Contents
- [Quick Start](#quick-start)
- [Features at a Glance](#features-at-a-glance)
- [What is controller tuning?](#what-is-controller-tuning)
- [Benchmark problems](#benchmark-problems)
- [Benchmark catalogue](#benchmark-catalogue)
- [Workflow recipes](#workflow-recipes)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

Works with Python 3.12+.

**Install**
```bash
pip install tunecontrol
```
Need BoTorch integrations?
```bash
pip install 'tunecontrol[botorch]'
```
The install pulls in `torch>=2.2`, `numpy>=1.26`, `scipy>=1.11`, `matplotlib`, and `tqdm`.

**Evaluate your first task**
```python
import torch
import tunecontrol as tc

task = tc.make("cartpole/2d/mae/deterministic")
theta = (task.bounds[0] + task.bounds[1]) / 2
value, info = task.evaluate(theta)
print(value.item(), sorted(info.keys()))
```

**Discover available registry keys**
```python
import tunecontrol as tc

for name in tc.tasks.list_task_names():
    print(name)
```

## Features at a Glance

| Capability | Details |
|------------|---------|
| Uniform evaluate API | Every task uses a single `evaluate(θ)` entry point that returns a scalar objective and metadata payload. |
| Deterministic & noisy twins | Flip between deterministic and noisy variants without changing code—ideal for hardware-style benchmarking. |
| Rich trajectory metadata | Each call returns time-series data (states, inputs, references) for plotting, debugging, and controller diagnostics. |
| Reproducible tasks | Fixed seeds and fully specified simulators make experiments repeatable across machines. |

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
- **Noisy:** includes process/measurement/initial-condition noise; repeated calls vary, resembling hardware experiments.

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
- **Noise variants:** deterministic and noisy (measurement noise)

---

## Benchmark catalogue (compact)

Benchmarks follow the naming convention `system/dimension/objective/noise`. The tables below enumerate every available task.

### CartPole
| Dimension | Deterministic variants | Noisy variants |
|-----------|-----------------------|----------------|
| 1D | `cartpole/1d/itae/deterministic`<br>`cartpole/1d/lqr/deterministic`<br>`cartpole/1d/mae/deterministic` | `cartpole/1d/itae/default_noise`<br>`cartpole/1d/lqr/default_noise`<br>`cartpole/1d/mae/default_noise` |
| 2D | `cartpole/2d/itae/deterministic`<br>`cartpole/2d/lqr/deterministic`<br>`cartpole/2d/mae/deterministic` | `cartpole/2d/itae/default_noise`<br>`cartpole/2d/lqr/default_noise`<br>`cartpole/2d/mae/default_noise` |
| 3D | `cartpole/3d/itae/deterministic`<br>`cartpole/3d/lqr/deterministic`<br>`cartpole/3d/mae/deterministic` | `cartpole/3d/itae/default_noise`<br>`cartpole/3d/lqr/default_noise`<br>`cartpole/3d/mae/default_noise` |
| 4D | `cartpole/4d/itae/deterministic`<br>`cartpole/4d/lqr/deterministic`<br>`cartpole/4d/mae/deterministic` | `cartpole/4d/itae/default_noise`<br>`cartpole/4d/lqr/default_noise`<br>`cartpole/4d/mae/default_noise` |

### Cascaded Tanks
| Deterministic variants | Noisy variants |
|------------------------|----------------|
| `cascaded_tank/2d/logsse/deterministic`<br>`cascaded_tank/2d/sse/deterministic`<br>`cascaded_tank/2d/quadratic/deterministic`<br>`cascaded_tank/2d/rise_time/deterministic`<br>`cascaded_tank/2d/overshoot/deterministic` | `cascaded_tank/2d/logsse/default_noise`<br>`cascaded_tank/2d/sse/default_noise`<br>`cascaded_tank/2d/quadratic/default_noise`<br>`cascaded_tank/2d/rise_time/default_noise`<br>`cascaded_tank/2d/overshoot/default_noise` |

<p align="center">
  <img src="docs/figures/deterministic_2d_gallery.png" alt="Objective landscape gallery for deterministic 2D tasks"/>

*Objective landscapes for every deterministic 2D task. Darker colours indicate lower cost.*
</p>

---

## Examples
- `examples/quickstart.py` – run `python examples/quickstart.py` to instantiate a task, evaluate mid-domain parameters, and inspect returned metadata.
- `examples/objective_gallery.py` – visualise objective landscapes across deterministic 2D benchmarks; produces figures under `docs/figures/`.
- `standard_bo_for_controller_tuning.ipynb` – step through a full BoTorch loop, from Gaussian-process fitting to acquisition optimisation.
- `examples/creating_custom_task.ipynb` – follow the template for introducing a new benchmark (mass-spring-damper) before contributing your own.

---

## Contributing
- Prototype new benchmarks by following `examples/creating_custom_task.ipynb` and mirroring the doc structure under `docs/`.
- Add narrative docs or figures for new tasks so they appear alongside the existing CartPole and cascaded-tank guides.
- Open an issue or pull request outlining the task, expected objective, and any stochastic settings; include sample trajectories where possible.

---

## Citation
TuneControl was introduced in our paper *Bayesian Optimization for Controller Tuning & Robot Learning: A Tutorial and Review*. Please cite it if TuneControl supports your research.

```bibtex
@article{TBD,
    title = {Bayesian Optimization for Controller Tuning & Robot Learning: A Tutorial and Review},
    author = {TBD},
    journal = {TBD},
    year = {2025}
}
```

---

## License
See `LICENSE` for details.
