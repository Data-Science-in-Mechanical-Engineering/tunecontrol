"""Discover families and save an explicit configuration."""
from dataclasses import asdict
from importlib.metadata import version
import json

import tunecontrol as tc
from tunecontrol import CartPole, CartPoleConfig, CartPoleNoise


def main():
    for family in tc.list_problems():
        print(family, tc.describe(family).description)
        print("Standard configurations:", len(tc.available_configs(family)))

    # Direct Python construction is sufficient for using a known family.
    problem = CartPole(CartPoleConfig(dim=2, objective="mae", noise=CartPoleNoise()))
    record = {"family": "cartpole", "config": asdict(problem.config), "run_seed": 42, "version": version("tunecontrol")}
    serialized = json.dumps(record, indent=2)
    print(serialized)

    # String construction is useful when loading a saved configuration.
    restored = json.loads(serialized)
    replay = tc.make(restored["family"], config=restored["config"])
    replay.setup(run_seed=restored["run_seed"])
    value, _ = replay.evaluate(replay.bounds.mean(dim=0))
    print("Cost:", value.item())


if __name__ == "__main__":
    main()
