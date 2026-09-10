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
- **Constraints**: after the dynamics update and process noise, levels are clipped to \([0, 10]\) cm. The control action is lower-bounded at zero to reflect pump limitations.

The initial levels are **5.8113 cm (upper)** and **1.9440 cm (lower)**.
Returned trajectories follow the [common format](task_module_architecture.md#trajectory-format):
`states` has upper/lower level columns (cm), and `inputs` has one pump-voltage
column (V). Proportional and discrete integral gains both have units V/cm under
the accumulated-error convention below.

Samples are equally spaced and each has a timestamp. Defaults are 0, 4, ..., 1000 s:
250 intervals and 251 samples. A duration not divisible by the configured step is
rounded up to a whole interval. Each input is computed from its corresponding
state; the final input is recorded but does not drive another transition.

## Controller
The PI controller uses a discrete accumulated-error state. At sample index \(k\),
\[
e_k = r - x_{2,k}, \qquad I_k = I_{k-1} + e_k, \qquad
u_k = \max(0, k_p e_k + k_i I_k),
\]
with setpoint \(r = 4\ \text{cm}\) and accumulator \(I_{-1}=0\) at the start of
every episode. The current sample's error is included before computing the input.
The gain \(k_i\) multiplies the sum of errors per sample, not a time integral:
there is no sampling-interval factor in the accumulator. The existing gain bounds
use this discrete convention.

## Objectives
The configuration selects one of the scalar costs below; all are minimisation objectives evaluated over the episode.

| Objective | Description | Summary |
|-----------|-------------|---------|
| `logsse` | Logarithm of the sum of squared tracking errors | \(\log(\max(10^{-12}, \tfrac12\sum (r - x_2)^2))\) |
| `sse` | Half the sum of squared tracking errors | \(\tfrac12\sum (r - x_2)^2\) |
| `quadratic` | LQR-style tracking error and pump-effort penalty | \(\frac{1}{M}\sum_{k=0}^{M-1} ((x_{2,k}-r)^2 + u_k^2)\) |
| `overshoot` | Nonnegative peak excess as a percentage of the target | Returns 10.0 for a peak of 4.4 cm at a target of 4 cm; zero if the target is never exceeded |
| `rise_time` | 10–90 % response time relative to the initial-to-target change | Elapsed time between the first samples reaching 10 % and 90 % of that change |

### Quadratic and squared-error conventions

The quadratic objective averages squared lower-tank tracking errors plus squared
pump inputs over the recorded samples, with both weights equal to 1. It does not penalize the upper-tank level.
SSE is half the sum of squared lower-tank tracking errors; LogSSE is the natural
logarithm of that same quantity, floored at 1e-12 before taking the logarithm.
All three objectives include every recorded sample, including the initial and
final samples. The quadratic cost divides by their count (M = 251 by default);
SSE and LogSSE retain sums without averaging. No sampling-interval factor is applied.

### Overshoot convention

Overshoot is 100 times the positive part of the peak lower-tank level minus the
target, divided by the target. The target is strictly positive in the task
configuration. The score is normalized by the absolute target, not by the change
from the initial level. It measures exceedance above the target over the entire
recorded trajectory, including the initial sample; it is not a direction-dependent
undershoot metric for falling commands. Zero overshoot alone does not imply that
the controller reaches the target or tracks it well.

### Rise-time convention

Thresholds are relative to the lower tank's initial level, not zero. With initial
level 1.944 cm and target 4 cm, the 10 % and 90 % levels are 2.1496 cm and 3.7944 cm.
The metric subtracts the first sampled 10 % crossing time from the first sampled
90 % crossing time. It uses the same progress convention for falling responses.
No interpolation or settling requirement is applied; the default 4 s sampling
interval therefore quantizes the result, and a jump across both thresholds in one
sample has rise time zero. An initial level equal to the target also has rise time
zero because no change is commanded. If either threshold is never reached, the
objective is undefined and evaluation returns `(NaN, info)` with diagnostics.

## Noise models & variants
`CascadedTank.available_configs()` enumerates objective and noise combinations.

| Variant | Description |
|---------|-------------|
| `noise=None` | Sets `noise_std = 0.0`, yielding repeatable, noise-free simulations. |
| `noise=CascadedTankNoise()` | Adds zero-mean Gaussian perturbations with `noise_std = 0.005` to both tank levels at every step. |

Supply `CascadedTankNoise(std=...)` in `CascadedTankConfig` for custom noise strengths.
This is process noise: the disturbed, clipped state feeds the next simulation
step and the controller. There is no separate observation-noise model. Noise is
added before clipping to the physical limits, so disturbances near a boundary
are truncated. The square-root flow calculation also guards against negative levels.

## References
- Schoukens, M., Mattson, P., Wigren, T., & Noël, J.-P. (2020). *Cascaded Tanks Benchmark Combining Soft and Hard Nonlinearities.* [Nonlinear Benchmark Repository](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks)
