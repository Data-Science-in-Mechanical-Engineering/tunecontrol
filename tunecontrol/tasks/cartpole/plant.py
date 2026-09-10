"""Cart-pole closed-loop simulator utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import torch
from scipy.linalg import solve_continuous_are

from ...random import new_generator
from ..module_utils import Trajectory, detach_trajectory

__all__ = ["CartPoleSimulator"]


DTYPE = torch.float64
Tensor = torch.Tensor


@dataclass
class CartPoleSimulator:
    """Simulate the cart-pole closed-loop system for given controller parameters."""

    dim: int
    simulation_noise: Union[Dict[str, float], bool] = False
    sample_time: float = 0.02
    simulation_time: float = 30.0
    Q_weight: float = 10.0
    R_weight: float = 1.0
    generator: torch.Generator = field(default_factory=new_generator, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 1 <= self.dim <= 4:
            raise ValueError(f"Dimension must be between 1 and 4, got {self.dim}")

        self.Q = torch.eye(4, dtype=DTYPE) * self.Q_weight
        self.R = torch.eye(1, dtype=DTYPE) * self.R_weight

        self._params = {
            "mass_pole": 0.0804,
            "length_pole": 0.147,
            "friction_coef": 2.2e-3,
            "K": 1.0,
            "T1": 1.0,
        }

        self.K_opt = self._compute_lqr_controller()
        self.last_episode: Trajectory | None = None

    @property
    def cost_weights(self) -> Tuple[Tensor, Tensor]:
        return self.Q.clone(), self.R.clone()

    def run_episode(
        self, theta: Tensor, *, generator: torch.Generator | None = None
    ) -> Trajectory:
        theta = theta.to(dtype=DTYPE)
        if theta.ndim != 1 or theta.shape[0] != self.dim:
            raise ValueError(f"theta must be shape ({self.dim},), got {tuple(theta.shape)}")

        controller = self._construct_controller(theta)
        states, inputs, reference, time = self._simulate_system(
            controller, self.generator if generator is None else generator
        )

        trajectory = {
            "time": time,
            "states": states,
            "inputs": inputs.reshape(-1, 1),
            "state_names": ("cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"),
            "state_units": ("m", "m/s", "rad", "rad/s"),
            "input_names": ("control_input",),
            "input_units": (None,),
            "reference": reference,
        }
        self.last_episode = detach_trajectory(trajectory, dtype=DTYPE, device=states.device)
        return trajectory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _construct_controller(self, theta: Tensor) -> Tensor:
        if self.dim == 4:
            gains = theta.view(1, -1)
        else:
            prefix = self.K_opt[:, : 4 - self.dim]
            gains = torch.cat((prefix, theta.view(1, -1)), dim=1)
        return gains.to(dtype=DTYPE)

    def _compute_lqr_controller(self) -> Tensor:
        _, _, A, B = self._linearized_model()
        Q, R = self.Q, self.R

        P = solve_continuous_are(
            A.cpu().numpy(),
            B.cpu().numpy(),
            Q.cpu().numpy(),
            R.cpu().numpy(),
        )
        P_t = torch.as_tensor(P, dtype=DTYPE)
        BT_P = B.transpose(0, 1) @ P_t
        K = torch.linalg.solve(R, BT_P)
        return K

    def _linearized_model(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        T1 = self._params["T1"]
        K = self._params["K"]
        mu_friction = self._params["friction_coef"]
        mp = self._params["mass_pole"]
        l = self._params["length_pole"]
        g = 9.81
        d = 0.005

        J = mp * (l / 2) ** 2 + 1 / 12 * mp * l**2 + 0.25 * mp * (d / 2) ** 2
        mpl_J = mp * l / J

        A = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0 / T1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.5 * mpl_J / T1, 0.5 * mpl_J * g, -mu_friction / J],
            ],
            dtype=DTYPE,
        )
        B = torch.tensor(
            [[0.0], [K / T1], [0.0], [-0.5 * mpl_J * K / T1]],
            dtype=DTYPE,
        )

        em_upper = torch.cat((A, B), dim=1)
        em_lower = torch.zeros((B.shape[1], A.shape[1] + B.shape[1]), dtype=DTYPE)
        em = torch.cat((em_upper, em_lower), dim=0)
        ms = torch.linalg.matrix_exp(em * self.sample_time)
        Ad = ms[: A.shape[0], : A.shape[1]]
        Bd = ms[: A.shape[0], A.shape[1] :]
        return Ad, Bd, A, B

    def _simulate_system(self, K: Tensor, generator: torch.Generator) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n_steps = int(math.ceil(self.simulation_time / self.sample_time))
        time = torch.arange(n_steps, dtype=DTYPE) * self.sample_time
        states = torch.zeros((n_steps, 4), dtype=DTYPE)
        inputs = torch.zeros(n_steps, dtype=DTYPE)
        reference = torch.zeros(n_steps, dtype=DTYPE)

        reference[100:500] = -2.0
        reference[500:1000] = 2.0

        z0 = torch.zeros(4, dtype=DTYPE)
        noise_cfg: Dict[str, float] | None = None
        if isinstance(self.simulation_noise, dict):
            noise_cfg = {k: float(v) for k, v in self.simulation_noise.items()}
            init_std = noise_cfg.get("initial_condition_std", 0.0)
            if init_std:
                z0 = torch.normal(
                    mean=0.0,
                    std=init_std,
                    size=(4,),
                    generator=generator,
                    dtype=DTYPE,
                )

        states[0] = z0
        current_state = z0.clone()
        dt = torch.tensor(self.sample_time, dtype=DTYPE)

        for i in range(n_steps - 1):
            error = current_state.clone()
            error[0] = error[0] - reference[i]
            control = -(K @ error.unsqueeze(1)).squeeze()
            control = torch.clamp(control, min=-10.0, max=10.0)
            inputs[i] = control

            next_state = self._rk4_step(current_state, control, dt)
            if noise_cfg is not None:
                process_noise = noise_cfg.get("process_noise_std", 0.0)
                if process_noise:
                    noise = torch.normal(
                        mean=0.0,
                        std=process_noise,
                        size=current_state.shape,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    next_state = next_state + noise

            states[i + 1] = next_state
            current_state = next_state

        # Compute control for the final state
        final_error = current_state.clone()
        final_error[0] = final_error[0] - reference[-1]
        final_control = torch.clamp(
            -(K @ final_error.unsqueeze(1)).squeeze(),
            min=-10.0,
            max=10.0,
        )
        inputs[-1] = final_control

        return states, inputs, reference, time

    def _rk4_step(self, state: Tensor, control: Tensor, dt: Tensor) -> Tensor:
        k1 = self._inv_pendulum(state, control)
        k2 = self._inv_pendulum(state + 0.5 * dt * k1, control)
        k3 = self._inv_pendulum(state + 0.5 * dt * k2, control)
        k4 = self._inv_pendulum(state + dt * k3, control)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _inv_pendulum(self, state: Tensor, control: Tensor) -> Tensor:
        x, dx, phi, dphi = state

        T1 = self._params["T1"]
        K = self._params["K"]
        mp = self._params["mass_pole"]
        l = self._params["length_pole"]
        mu_friction = self._params["friction_coef"]
        g = 9.81
        d = 0.005
        Jd = mp * (l / 2) ** 2 + 1 / 12 * mp * l**2 + 0.25 * mp * (d / 2) ** 2

        xdt = dx
        xdotdt = 1 / T1 * (K * control - dx)
        phidt = dphi
        phidotdt = (
            0.5 * mp * g * l / Jd * torch.sin(phi)
            - 0.5 * mp * l / Jd * torch.cos(phi) * 1 / T1 * (K * control - dx)
            - mu_friction / Jd * dphi
        )
        return torch.stack([xdt, xdotdt, phidt, phidotdt])
