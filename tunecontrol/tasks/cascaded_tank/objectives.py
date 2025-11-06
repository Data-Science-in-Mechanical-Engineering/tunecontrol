"""Objective registry for the cascaded tank benchmark."""

from __future__ import annotations

from typing import List

import torch

from ..module_utils import ObjectiveRegistry, ObjectiveSpec, Trajectory
from .plant import CascadedTankSimulator

__all__ = [
    "CascadedTankObjectiveSpec",
    "register_cascaded_tank_objective",
    "list_cascaded_tank_objectives",
    "get_cascaded_tank_objective_spec",
]


CascadedTankObjectiveSpec = ObjectiveSpec
_OBJECTIVES = ObjectiveRegistry()


def _x2_column(traj: Trajectory) -> torch.Tensor:
    return traj["x2"]


def _inputs_column(traj: Trajectory) -> torch.Tensor:
    return traj["u"].view(-1, 1)


def _time_vector(traj: Trajectory) -> torch.Tensor:
    return traj["time"]


def _logsse(sim: CascadedTankSimulator, traj: Trajectory) -> torch.Tensor:
    x2 = _x2_column(traj)
    target = torch.full_like(x2, fill_value=sim.target, dtype=x2.dtype, device=x2.device)
    diff = x2 - target
    sse = 0.5 * torch.sum(diff.square())
    sse = torch.clamp(sse, min=1e-12)
    return torch.log(sse)


def _sse(sim: CascadedTankSimulator, traj: Trajectory) -> torch.Tensor:
    x2 = _x2_column(traj)
    target = torch.full_like(x2, fill_value=sim.target, dtype=x2.dtype, device=x2.device)
    diff = x2 - target
    return 0.5 * torch.sum(diff.square())


def _quadratic(sim: CascadedTankSimulator, traj: Trajectory) -> torch.Tensor:
    x2 = _x2_column(traj).unsqueeze(-1)
    inputs = _inputs_column(traj)
    q = torch.tensor([[1.0]], dtype=x2.dtype, device=x2.device)
    r = torch.tensor([[1.0]], dtype=inputs.dtype, device=inputs.device)
    state_term = torch.einsum("ni,ij,nj->n", x2, q, x2)
    input_term = torch.einsum("ni,ij,nj->n", inputs, r, inputs)
    return torch.sum(state_term + input_term)


def _overshoot(sim: CascadedTankSimulator, traj: Trajectory) -> torch.Tensor:
    x2 = _x2_column(traj)
    target = torch.tensor(sim.target, dtype=x2.dtype, device=x2.device)
    overshoot = (torch.max(x2) - target) / target
    return overshoot


def _rise_time(sim: CascadedTankSimulator, traj: Trajectory) -> torch.Tensor:
    response = _x2_column(traj)
    times = _time_vector(traj)
    target = torch.tensor(sim.target, dtype=response.dtype, device=response.device)
    lower_bound = target * 0.1
    upper_bound = target * 0.9

    lower_indices = torch.nonzero(response >= lower_bound, as_tuple=False)
    upper_indices = torch.nonzero(response >= upper_bound, as_tuple=False)
    if lower_indices.numel() == 0 or upper_indices.numel() == 0:
        return torch.tensor(float("nan"), dtype=response.dtype, device=response.device)
    lower_index = lower_indices[0, 0]
    upper_index = upper_indices[0, 0]
    return times[upper_index] - times[lower_index]


def register_cascaded_tank_objective(
    name: str,
    evaluate_fn,
    *,
    display_name: str | None = None,
    is_minimization: bool = True,
) -> CascadedTankObjectiveSpec:
    return _OBJECTIVES.register(
        name,
        evaluate_fn,
        display_name=display_name,
        is_minimization=is_minimization,
    )


def list_cascaded_tank_objectives() -> List[str]:
    return _OBJECTIVES.list()


def get_cascaded_tank_objective_spec(name: str) -> CascadedTankObjectiveSpec:
    return _OBJECTIVES.get(name)


register_cascaded_tank_objective("logsse", _logsse, display_name="LogSSE")
register_cascaded_tank_objective("sse", _sse, display_name="SSE")
register_cascaded_tank_objective("quadratic", _quadratic, display_name="Quadratic")
register_cascaded_tank_objective("overshoot", _overshoot, display_name="Overshoot")
register_cascaded_tank_objective("rise_time", _rise_time, display_name="RiseTime")
