# CartPole Task

## Overview
The cart–pole benchmark models an inverted pendulum mounted on a cart that must reject step changes in reference position. A fixed-structure state-feedback controller applies horizontal forces to keep the pole upright while tracking the position command. Variants in TuneControl cover 1–4 decision variables, three objectives, and deterministic or noisy simulations for benchmarking controller-tuning and Bayesian-optimisation routines.

## Decision variables
All variants parameterise the feedback gain vector \(K = [k_x, k_{\dot{x}}, k_\phi, k_{\dot{\phi}}]\). Depending on the requested dimensionality, leading entries fall back to an LQR baseline while the remaining suffix is optimised. Bounds are enforced directly on the gains (no logarithmic transform).

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
- **Reference profile**: hold at 0 m for the first 2 s, step to -2 m, then to +2 m midway through the episode.

Each simulation returns trajectories for time, states, inputs, and reference signals.

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

## Noise models & variants
Factory keys follow `cartpole/{dim}d/{objective}/{noise}`. Two noise configurations are bundled:

| Variant | Description |
|---------|-------------|
| `deterministic` | Pure simulation without disturbances; identical evaluations for repeated θ. |
| `default_noise` | Gaussian perturbations with `initial_condition_std = 0.0005` applied to the initial state and `process_noise_std = 0.0005` injected at every integration step. |

Custom noise dictionaries can be supplied when instantiating `CartPoleTask` directly.

## References
- Selfridge, O. G., Sutton, R. S., & Barto, A. G. (1985). *Training and Tracking in Robotics.* [Semantic Scholar](https://www.semanticscholar.org/paper/Training-and-Tracking-in-Robotics-Selfridge-Sutton/3f926f229755a617630ff241789bb4ef09f9209c)
