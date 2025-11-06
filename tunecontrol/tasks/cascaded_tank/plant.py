"""Simulation utilities for the cascaded tank benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

__all__ = [
    "ProportionalIntegralController",
    "CascadedTankDynamic",
    "CascadedTankSimulator",
    "Trajectory",
]


Tensor = torch.Tensor
Trajectory = Dict[str, Tensor]
DTYPE = torch.float64


class ProportionalIntegralController:
    """Simple PI controller operating on torch tensors."""

    def __init__(self, k_p: float = 0.6, k_i: float = 0.01) -> None:
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.integral = torch.zeros(1, dtype=DTYPE)

    def reset(self) -> None:
        self.integral.zero_()

    def get_control_input(self, current_state: Tensor, target_state: Tensor) -> tuple[Tensor, Tensor]:
        error = target_state - current_state
        self.integral = self.integral + error
        control = self.k_p * error + self.k_i * self.integral
        control = torch.clamp(control, min=0.0)
        return control.squeeze(), error.squeeze()


class CascadedTankDynamic:
    """Discrete-time dynamics for the cascaded tank process."""

    def __init__(
        self,
        k1: float = 0.2143,
        k2: float = 0.2165,
        k3: float = 0.1654,
        k4: float = 0.1371,
        initial_state: Tensor | None = None,
        time_step: float = 4.0,
    ) -> None:
        self.k1 = torch.tensor(k1, dtype=DTYPE)
        self.k2 = torch.tensor(k2, dtype=DTYPE)
        self.k3 = torch.tensor(k3, dtype=DTYPE)
        self.k4 = torch.tensor(k4, dtype=DTYPE)
        self.time_step = torch.tensor(time_step, dtype=DTYPE)
        if initial_state is None:
            initial_state = torch.tensor([5.8113, 1.9440], dtype=DTYPE)
        self.initial_state = initial_state.to(dtype=DTYPE).clone()
        self.state = self.initial_state.clone()

    def reset(self) -> None:
        """Reset the internal state to the initial condition."""
        self.state = self.initial_state.clone()

    def set_state(self, state_tensor: Tensor) -> None:
        self.state = state_tensor.to(dtype=DTYPE).clone()

    def make_environment_step(
        self,
        state: Optional[Tensor] = None,
        u_control_input: Tensor | float = 0.0,
        noise_std_dev: float = 0.005,
    ) -> Tensor:
        if state is None:
            state = self.state
        input_state = state.to(dtype=DTYPE).clone()

        u_tensor = torch.as_tensor(u_control_input, dtype=DTYPE)
        next_state = input_state.clone()
        input_state = torch.clamp(input_state, min=0.0)

        next_state[0] = input_state[0] + self.time_step / 4 * (
            -self.k1 * torch.sqrt(input_state[0]) + self.k4 * u_tensor
        )
        next_state[1] = input_state[1] + self.time_step / 4 * (
            self.k2 * torch.sqrt(input_state[0]) - self.k3 * torch.sqrt(input_state[1])
        )

        next_state = torch.clamp(next_state, min=0.0, max=10.0)
        if noise_std_dev:
            noise = torch.randn_like(next_state) * noise_std_dev
            next_state = next_state + noise
        return next_state


@dataclass
class CascadedTankSimulator:
    """Helper simulator for generating trajectories and metrics."""

    duration: float = 1000.0
    target: float = 4.0
    noise_std: float = 0.005
    dynamic_kwargs: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dynamic = CascadedTankDynamic(**self.dynamic_kwargs)
        self.data: Optional[Trajectory] = None

    def simulate(self, k_p: float, k_i: float) -> Trajectory:
        controller = ProportionalIntegralController(k_p=k_p, k_i=k_i)
        controller.reset()

        dynamic = self.dynamic
        dynamic.reset()

        total_steps = int(math.ceil(float(self.duration) / float(dynamic.time_step)))
        target_tensor = torch.tensor(self.target, dtype=DTYPE)

        x1_levels: list[Tensor] = []
        x2_levels: list[Tensor] = []
        control_inputs: list[Tensor] = []
        times: list[Tensor] = []

        time = torch.tensor(0.0, dtype=DTYPE)
        control, _ = controller.get_control_input(dynamic.state[1], target_tensor)

        x1_levels.append(dynamic.state[0].clone())
        x2_levels.append(dynamic.state[1].clone())
        control_inputs.append(control.clone())
        times.append(time.clone())

        for _ in range(total_steps):
            next_state = dynamic.make_environment_step(dynamic.state, control, self.noise_std)
            dynamic.set_state(next_state)

            time = time + dynamic.time_step
            control, _ = controller.get_control_input(dynamic.state[1], target_tensor)

            x1_levels.append(dynamic.state[0].clone())
            x2_levels.append(dynamic.state[1].clone())
            control_inputs.append(control.clone())
            times.append(time.clone())

        self.data = {
            "time": torch.stack(times),
            "x1": torch.stack(x1_levels),
            "x2": torch.stack(x2_levels),
            "u": torch.stack(control_inputs),
        }
        return self.data
