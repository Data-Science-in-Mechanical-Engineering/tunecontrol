"""Parameter sweep gallery for deterministic 2D tasks."""

from __future__ import annotations

import argparse
import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tunecontrol as tc


TASKS = [
    (family, config)
    for family in tc.list_problems()
    for config in tc.available_configs(family)
    if config.noise is None and getattr(config, "dim", 2) == 2
]
GRID_POINTS = 10


def evaluate_on_grid(task: tc.Task, grid_size: int = GRID_POINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = task.bounds[0].numpy()
    upper = task.bounds[1].numpy()

    theta0 = np.linspace(lower[0], upper[0], grid_size)
    theta1 = np.linspace(lower[1], upper[1], grid_size)
    grid = np.array(list(itertools.product(theta0, theta1)), dtype=np.float64)

    values = []
    for theta_pair in grid:
        theta_tensor = task.bounds.new_tensor(theta_pair)
        value, _ = task.evaluate(theta_tensor)
        values.append(float(value.item()))

    values = np.array(values).reshape(len(theta0), len(theta1))
    mesh0, mesh1 = np.meshgrid(theta0, theta1, indexing="ij")
    return mesh0, mesh1, values


def plot_gallery(tasks: Sequence[tuple[str, object]], *, output: str | None = None, show: bool = True) -> None:
    cols = 4
    rows = int(np.ceil(len(tasks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax, (family, config) in zip(axes.flat, tasks):
        task = tc.make(family, config)
        mesh0, mesh1, values = evaluate_on_grid(task)
        if np.isfinite(values).any():
            contour = ax.contourf(mesh0, mesh1, np.ma.masked_invalid(values), levels=30, cmap="viridis")
            fig.colorbar(contour, ax=ax)
        else:
            ax.text(0.5, 0.5, "No finite evaluations", ha="center", transform=ax.transAxes)
        undefined = np.count_nonzero(~np.isfinite(values))
        suffix = f" ({undefined} undefined)" if undefined else ""
        ax.set_title(f"{family}: {config.objective}{suffix}")
        ax.set_xlabel("theta[0]")
        ax.set_ylabel("theta[1]")

    for ax in axes.flat[len(tasks):]:
        ax.axis("off")

    fig.suptitle("Deterministic 2D Objective Landscapes", fontsize=14)
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot deterministic 2D benchmark landscapes.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the gallery image.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    args = parser.parse_args()

    plot_gallery(TASKS, output=args.output, show=not args.no_show)
