"""Visualization helpers for cart-pole trajectories."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _to_numpy(array: Sequence[float]) -> np.ndarray:
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def plot_episode(trajectory: Mapping[str, Sequence[float]], title: str = "CartPole Episode") -> None:
    """Plot state and control trajectories from a cart-pole episode."""
    time = _to_numpy(trajectory.get("time", []))
    states = _to_numpy(trajectory.get("states", []))
    inputs = _to_numpy(trajectory.get("inputs", []))
    if time is None or states is None or inputs is None:
        raise ValueError("trajectory must contain 'time', 'states', and 'inputs'")

    reference = trajectory.get("reference")
    reference_np = None if reference is None else _to_numpy(reference)

    x1, x2, x3, x4 = states.T
    u = inputs

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title)

    axs[0].plot(time, x1, label="Cart Position (x1)")
    if reference_np is not None:
        axs[0].plot(time, reference_np, linestyle="--", color="black", label="Reference")
    axs[0].set_ylabel("Position (m)")

    axs[1].plot(time, x2, label="Cart Velocity (x2)", color="orange")
    axs[1].set_ylabel("Velocity (m/s)")

    axs[2].plot(time, x3, label="Pole Angle (x3)", color="green")
    axs[2].set_ylabel("Angle (rad)")

    axs[3].plot(time, x4, label="Pole Angular Velocity (x4)", color="red")
    axs[3].set_ylabel("Angular Velocity (rad/s)")

    axs[4].plot(time, u, label="Control Input (u)", color="purple")
    axs[4].set_ylabel("Force (N)")
    axs[4].set_xlabel("Time (s)")

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
