"""Quickstart example: evaluate a single task and inspect metadata."""

from __future__ import annotations

import tunecontrol as tc


def _print_info_keys(info, indent=0):
    """Recursively print the keys of an info dictionary."""
    pad = "  " * indent
    if isinstance(info, dict):
        for key, value in info.items():
            print(f"{pad}- {key}")
            _print_info_keys(value, indent + 1)
    elif isinstance(info, (list, tuple)):
        print(f"{pad}- list[{len(info)}]")
        for item in info:
            _print_info_keys(item, indent + 1)


def main() -> None:
    task = tc.make("cartpole/2d/mae/deterministic")
    theta = (task.bounds[0] + task.bounds[1]) / 2

    value, info = task.evaluate(theta)

    print("Task:", task.name)
    print("Theta:", theta.tolist())
    print("Objective value:", float(value.item()))
    print("Info keys:")
    _print_info_keys(info)


if __name__ == "__main__":
    main()
