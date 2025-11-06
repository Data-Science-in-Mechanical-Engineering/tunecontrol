# Cascaded Tank Task

## Overview
The cascaded-tank benchmark captures a pair of fluid tanks connected in series. A proportional–integral (PI) controller drives a pump voltage so that the lower tank tracks a target level while respecting actuator saturation and nonlinear flow dynamics. TuneControl exposes deterministic and noisy variants of this 2D tuning problem, together with five scalar objectives.

## Decision variables
The decision vector is \(\theta = [k_p, k_i]\), the proportional and integral gains of the PI controller.

| Variable | Description | Bounds |
|----------|-------------|--------|
| \(k_p\) | Proportional gain | (0.9, 4.5) |
| \(k_i\) | Integral gain | (0.01, 0.16) |

## Simulation setup
- **States** \(x = [x_1, x_2]\) represent the fluid levels (cm) in the upper and lower tanks.
- **Discrete dynamics** implement the benchmark equations
  \[
  \begin{aligned}
  x_1(t+h) &= x_1(t) + \tfrac{h}{4}\big(-k_1\sqrt{x_1(t)} + k_4 u(t)\big),\\
  x_2(t+h) &= x_2(t) + \tfrac{h}{4}\big(k_2\sqrt{x_1(t)} - k_3\sqrt{x_2(t)}\big),
  \end{aligned}
  \]
  with parameters \(k_1=0.2143\), \(k_2=0.2165\), \(k_3=0.1654\), \(k_4=0.1371\).
- **Integration** uses an Euler scheme with step \(h = 4\ \text{s}\) over a 1000 s episode (251 samples).
- **Constraints**: levels are clipped to \([0, 10]\) cm, and the control action is lower-bounded at zero to reflect pump limitations.

Returned trajectories include time stamps, both tank levels, and the applied control input.

## Controller
The PI controller acts on the lower tank’s tracking error:
\[
u(t) = k_p (r - x_2(t)) + k_i \int_0^t (r - x_2(\tau))\,d\tau,
\]
with setpoint \(r = 4\ \text{cm}\). The integral state is reset at the beginning of every episode, and \(u(t)\) is clamped to remain nonnegative.

## Objectives
Each factory key selects one of the scalar costs below; all are minimisation objectives evaluated over the episode.

| Objective | Description | Summary |
|-----------|-------------|---------|
| `logsse` | Logarithm of the sum of squared tracking errors | \(\log(\sum (r - x_2)^2)\) for numerical stability |
| `sse` | Sum of squared errors | \(\sum (r - x_2)^2\) |
| `quadratic` | Weighted quadratic penalty on states and control | Combines deviations in \(x_1, x_2\) with pump effort |
| `overshoot` | Peak overshoot relative to the target | \(\max(0, x_2 - r)\) expressed as a percentage |
| `rise_time` | Time to reach 90 % of the setpoint | Duration between 10 % and 90 % threshold crossings |

## Noise models & variants
Registry keys follow `cascaded_tank/2d/{objective}/{noise}`.

| Variant | Description |
|---------|-------------|
| `deterministic` | Sets `noise_std = 0.0`, yielding repeatable, noise-free simulations. |
| `default_noise` | Adds zero-mean Gaussian perturbations with `noise_std = 0.005` to both tank levels at every step. |

Supplying `noise_std` when creating `CascadedTankTask` directly enables custom noise strengths.

## References
- Schoukens, M., Mattson, P., Wigren, T., & Noël, J.-P. (2020). *Cascaded Tanks Benchmark Combining Soft and Hard Nonlinearities.* [Nonlinear Benchmark Repository](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks)
