"""Visualization helpers for cascaded tank trajectories."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:  # Optional torch import for type checking
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is a required dep in practice
    torch = None  # type: ignore[assignment]

__all__ = ["plot_episode"]


def _to_numpy(array: Sequence[float]) -> np.ndarray:
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def plot_episode(trajectory: Mapping[str, Sequence[float]], *, title: str = "Cascaded Tank Episode") -> None:
    """Plot a cascaded tank trajectory using matplotlib."""
    import matplotlib.pyplot as plt  # Local import to keep dependency optional

    time = _to_numpy(trajectory.get("time", []))
    x1 = _to_numpy(trajectory.get("x1", []))
    x2 = _to_numpy(trajectory.get("x2", []))
    u = _to_numpy(trajectory.get("u", []))

    if time.size == 0:
        raise ValueError("Trajectory must include 'time' samples.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(time, x1, "o-", label="x1 (upper)")
    axes[0].set_ylabel("Level")
    axes[0].legend()

    axes[1].plot(time, x2, "o-", label="x2 (lower)")
    axes[1].set_ylabel("Level")
    axes[1].legend()

    axes[2].plot(time, u, "o-", label="u (input)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Input")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
