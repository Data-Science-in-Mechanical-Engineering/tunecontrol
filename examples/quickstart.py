"""Evaluate and plot both built-in problems: python examples/quickstart.py."""
import matplotlib.pyplot as plt

from tunecontrol import CartPole, CartPoleConfig, CascadedTank, CascadedTankConfig
from tunecontrol.tasks.cartpole import plot_episode as plot_cartpole
from tunecontrol.tasks.cascaded_tank import plot_episode as plot_tanks


def main():
    problems = [
        (CartPole(CartPoleConfig(dim=2, objective="mae")), plot_cartpole),
        (CascadedTank(CascadedTankConfig(objective="quadratic")), plot_tanks),
    ]
    for problem, plot in problems:
        # Bounds identify the search space; their midpoint is just an example.
        theta = problem.bounds.mean(dim=0)
        value, info = problem.evaluate(theta)
        print(problem.config)
        print("Controller:", theta.tolist(), "Cost:", value.item())
        trajectory = info["trajectory"]
        print("State columns:", trajectory["state_names"])
        print("Samples:", len(trajectory["time"]))
        plot(trajectory)
    plt.show()


if __name__ == "__main__":
    main()
