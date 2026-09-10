"""Objective registry for the cart-pole task."""

from __future__ import annotations

from typing import Dict, List

import torch

from ..module_utils import ObjectiveRegistry, ObjectiveSpec, Trajectory
from .plant import CartPoleSimulator

__all__ = [
    "CartPoleObjectiveSpec",
    "register_cartpole_objective",
    "list_cartpole_objectives",
    "get_cartpole_objective_spec",
]


CartPoleObjectiveSpec = ObjectiveSpec
_OBJECTIVES = ObjectiveRegistry()


def _state_error(traj: Trajectory) -> torch.Tensor:
    states = traj["states"]
    reference = traj.get("reference")
    if reference is None:
        return states
    error = states.clone()
    error[:, 0] = error[:, 0] - reference
    return error


def _inputs_column(traj: Trajectory) -> torch.Tensor:
    inputs = traj["inputs"]
    return inputs.view(-1, 1)


def _lqr_cost(sim: CartPoleSimulator, traj: Trajectory) -> torch.Tensor:
    states = _state_error(traj)
    inputs = _inputs_column(traj)
    Q, R = sim.cost_weights
    state_term = torch.einsum("ni,ij,nj->n", states, Q, states)
    input_term = torch.einsum("ni,ij,nj->n", inputs, R, inputs)
    return (state_term + input_term).mean()


def _itae_cost(sim: CartPoleSimulator, traj: Trajectory) -> torch.Tensor:
    states = _state_error(traj)
    inputs = _inputs_column(traj)
    Q, R = sim.cost_weights
    state_error = torch.abs(states @ Q).sum(dim=1)
    time = traj["time"].to(dtype=states.dtype, device=states.device)
    time_weights = time - time[0]
    itae = (time_weights * state_error).mean()
    quadratic_input = torch.einsum("ni,ij,nj->n", inputs, R, inputs).mean()
    return itae + quadratic_input


def _mae_cost(sim: CartPoleSimulator, traj: Trajectory) -> torch.Tensor:
    states = _state_error(traj)
    inputs = _inputs_column(traj)
    Q, R = sim.cost_weights
    state_error = torch.abs(states @ Q).sum(dim=1)
    input_error = torch.abs(inputs @ R).sum(dim=1)
    return torch.mean(state_error + input_error)


def register_cartpole_objective(
    name: str,
    evaluate_fn,
    *,
    display_name: str | None = None,
    is_minimization: bool = True,
) -> CartPoleObjectiveSpec:
    return _OBJECTIVES.register(
        name,
        evaluate_fn,
        display_name=display_name,
        is_minimization=is_minimization,
    )


def list_cartpole_objectives() -> List[str]:
    return _OBJECTIVES.list()


def get_cartpole_objective_spec(name: str) -> CartPoleObjectiveSpec:
    return _OBJECTIVES.get(name)


register_cartpole_objective("lqr", _lqr_cost, display_name="LQR")
register_cartpole_objective("itae", _itae_cost, display_name="ITAE")
register_cartpole_objective("mae", _mae_cost, display_name="MAE")
