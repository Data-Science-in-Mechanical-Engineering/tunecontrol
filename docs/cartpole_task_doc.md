# CartPole Task

## Overview
The cart–pole benchmark models an inverted pendulum mounted on a cart that must reject step changes in reference position. A fixed-structure state-feedback controller sets the control input to keep the pole upright while tracking the position command. Variants in TuneControl cover 1–4 decision variables, three objectives, and deterministic or noisy simulations for benchmarking controller-tuning and Bayesian-optimisation routines.

## Decision variables
All variants parameterise the feedback gain vector \(K = [k_x, k_{\dot{x}}, k_\phi, k_{\dot{\phi}}]\). Depending on the requested dimensionality, leading entries fall back to an LQR baseline while the remaining suffix is optimised. Bounds describe the search box directly in gain coordinates (no logarithmic transform); evaluation does not clip gains.

| Dimension | Tuned entries (suffix of \(K\)) | Bounds (lower, upper) |
|-----------|---------------------------------|-----------------------|
| 1D | \(k_{\dot{\phi}}\) | (-10.0, -2.0) |
| 2D | \(k_\phi, k_{\dot{\phi}}\) | (-50.0, -30.0), (-10.0, -2.5) |
| 3D | \(k_{\dot{x}}, k_\phi, k_{\dot{\phi}}\) | (-8.0, -4.0), (-50.0, -30.0), (-10.0, -2.5) |
| 4D | \(k_x, k_{\dot{x}}, k_\phi, k_{\dot{\phi}}\) | (-3.4, -2.0), (-8.0, -4.0), (-50.0, -30.0), (-10.0, -2.5) |

## Simulation setup
- **State vector** \(x = [x, \dot{x}, \phi, \dot{\phi}]\) captures cart position and velocity together with pole angle and angular velocity (radians).
- **Dynamics** follow the nonlinear inverted-pendulum equations with parameters:
  - pole mass \(m_p = 0.0804\ \text{kg}\), pole length \(l = 0.147\ \text{m}\)
  - friction coefficient \(\mu = 2.2\times 10^{-3}\), actuator constants \(T_1 = 1\), \(K = 1\)
- **Integration** uses a fourth-order Runge–Kutta step (implemented via repeated calls to the continuous model) with sample time \(0.02\ \text{s}\) over a \(30\ \text{s}\) horizon.
- **Reference profile (default sampling)**: 0 m from 0 to 2 s, -2 m from 2 to 10 s, +2 m from 10 to 20 s, then 0 m from 20 s onward. Steps take effect at samples 100, 500, and 1000 with the default 0.02 s interval. The simulator currently defines this schedule by sample index.

The deterministic initial state is `[0, 0, 0, 0]`. The actuator uses a first-order
cart-velocity model: acceleration is `(K * u - cart_velocity) / T1`. Its input
unit is not established, so plots label it **Control input** without a physical
unit. The pendulum dynamics include gravity, cart acceleration, and pivot friction.
The listed historical reference does not establish this actuator's calibration.

Default objective and LQR-design weights are `Q = 10 * I₄` and `R = 1`.
The fixed gain prefix in lower-dimensional variants is computed from these weights.
Controller gain units are input units divided by the corresponding state units;
the absolute input unit remains unspecified.

Each simulation returns the [common trajectory format](task_module_architecture.md#trajectory-format).
State columns are cart position (m), cart velocity (m/s), pole angle (rad), and
pole angular velocity (rad/s). The one input column is control input, with unknown
unit (`None`). `reference` contains the cart-position target at each timestamp.
Default timestamps are 0, 0.02, ..., 29.98 s (1500 samples); the 30 s endpoint is
excluded. Inputs are computed from the state and reference at their timestamp;
the final recorded input has no subsequent state transition in the episode.

## Controller
The closed loop applies a full-state feedback control law
\[
u(t) = -K\,\tilde{x}(t),
\]
where \(\tilde{x}\) substitutes the position state with its tracking error relative to the reference. To respect actuator limits, \(u\) is saturated to \([-10, 10]\). For sub-4D variants, untuned entries in \(K\) fall back to the optimal continuous-time LQR solution for the linearised model.

## Objectives
TuneControl exposes three cost functions; each returns a scalar to minimise.

| Objective | Description | Summary |
|-----------|-------------|---------|
| `lqr` | Quadratic cost on state errors and actuator effort | Mean of \(x^\top Q x + u^\top R u\) using simulator weights |
| `itae` | Integral time absolute error with quadratic control penalty | Mean of \(t \cdot \|Q x\|_1 + u^\top R u\) |
| `mae` | Mean absolute tracking error plus control penalty | Mean of \(\|Q x\|_1 + \|R u\|_1\) |

For `itae`, the time weight is elapsed time in seconds from the first trajectory
sample, so the first sample has weight zero. The objective retains the sample
mean of the weighted state error plus the mean quadratic input penalty; it does
not multiply the control penalty by time or apply an additional integration factor.

## Noise models & variants
`CartPole.available_configs()` enumerates dimension, objective and noise combinations. Two noise configurations are bundled:

| Variant | Description |
|---------|-------------|
| `noise=None` | Pure simulation without disturbances; identical evaluations for repeated θ. |
| `noise=CartPoleNoise()` | Gaussian perturbations with `initial_condition_std = 0.0005` applied to the initial state and `process_noise_std = 0.0005` injected at every integration step. |

Supply a `CartPoleNoise` object in `CartPoleConfig` for custom standard deviations.

## References
- Selfridge, O. G., Sutton, R. S., & Barto, A. G. (1985). *Training and Tracking in Robotics.* [Semantic Scholar](https://www.semanticscholar.org/paper/Training-and-Tracking-in-Robotics-Selfridge-Sutton/3f926f229755a617630ff241789bb4ef09f9209c)
