"""Quick smoke checks for cartpole and cascaded tank tasks."""

import torch

import tunecontrol as tc
from tunecontrol.tasks.cartpole import plot_episode as plot_cartpole_episode
from tunecontrol.tasks.cascaded_tank import plot_episode as plot_cascaded_episode


def main() -> None:
    cartpole = tc.make("cartpole/2d/mae/deterministic")
    theta_single = torch.tensor([-40.0, -3.0], dtype=torch.float64)

    cp_value, cp_info = cartpole.evaluate(theta_single)

    print("CartPole value:", cp_value.item())
    print("CartPole info keys:", cp_info.keys())

    plot_cartpole_episode(cp_info["trajectory"])

    cascaded = tc.make("cascaded_tank/2d/rise_time/deterministic")
    ct_theta = torch.tensor([4.0, 0.1], dtype=torch.float64)

    ct_value, ct_info = cascaded.evaluate(ct_theta)

    print("CascadedTank value:", ct_value.item())
    print("CascadedTank info keys:", ct_info.keys())

    plot_cascaded_episode(ct_info["trajectory"])


if __name__ == "__main__":
    main()
