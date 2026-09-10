"""Reference calculations for the agreed tank costs and discrete PI convention."""
import math
from types import SimpleNamespace
import unittest

import torch
import tunecontrol as tc
from tunecontrol.tasks.cascaded_tank.objectives import get_cascaded_tank_objective_spec
from tunecontrol.tasks.cascaded_tank.plant import ProportionalIntegralController
from tunecontrol.tasks.cartpole.plant import CartPoleSimulator


class TankCostTests(unittest.TestCase):
    def score(self, name, levels, inputs, target=4.):
        trajectory = {"states": torch.tensor([[0., level] for level in levels], dtype=torch.float64),
                      "inputs": torch.tensor(inputs, dtype=torch.float64).reshape(-1, 1)}
        return get_cascaded_tank_objective_spec(name).evaluate(SimpleNamespace(target=target), trajectory).item()

    def test_quadratic_tracking_and_effort(self):
        # Squared errors sum to 5 and squared inputs to 14; average over 3 samples.
        self.assertAlmostEqual(self.score("quadratic", [2., 4., 5.], [1., 2., 3.]), 19. / 3.)

    def test_quadratic_prefers_target_at_equal_effort(self):
        at_target = self.score("quadratic", [4., 4.], [2., 2.])
        below_target = self.score("quadratic", [2., 2.], [2., 2.])
        self.assertEqual(at_target, 4.)
        self.assertEqual(below_target, 8.)

    def test_quadratic_uses_configured_target(self):
        self.assertEqual(self.score("quadratic", [1., 3.], [0., 0.], target=2.), 1.)
        self.assertEqual(self.score("quadratic", [3.], [0.], target=3.), 0.)

    def test_repeating_samples_preserves_mean_cost(self):
        levels, inputs = [2., 4., 5.], [1., 2., 3.]
        self.assertAlmostEqual(self.score("quadratic", levels, inputs),
                               self.score("quadratic", levels * 4, inputs * 4))

    def test_single_sample_includes_both_penalties(self):
        self.assertEqual(self.score("quadratic", [2.], [3.]), 13.)

    def test_half_sse_and_natural_log(self):
        self.assertEqual(self.score("sse", [2., 4., 5.], [0., 0., 0.]), 2.5)
        self.assertAlmostEqual(self.score("logsse", [2., 4., 5.], [0., 0., 0.]), math.log(2.5))

    def test_logsse_floor(self):
        self.assertEqual(self.score("sse", [4., 4.], [0., 0.]), 0.)
        self.assertAlmostEqual(self.score("logsse", [4., 4.], [0., 0.]), math.log(1e-12))

    def test_actual_task_scores_tracking_error(self):
        problem = tc.CascadedTank(tc.CascadedTankConfig(objective="quadratic"))
        value, info = problem.evaluate(problem.bounds.mean(0))
        trajectory = info["trajectory"]
        expected = sum((level - 4.) ** 2 + pump ** 2
                       for level, pump in zip(trajectory["states"][:, 1].tolist(), trajectory["inputs"][:, 0].tolist()))
        expected /= len(trajectory["states"][:, 1])
        self.assertAlmostEqual(value.item(), expected, places=8)


class ControllerConventionTests(unittest.TestCase):
    def test_pi_accumulates_current_error_per_sample_and_resets(self):
        controller = ProportionalIntegralController(k_p=2., k_i=.5)
        target = torch.tensor(4., dtype=torch.float64)
        # Errors 2 then 1 produce accumulator values 2 then 3.
        u, _ = controller.get_control_input(torch.tensor(2.), target)
        self.assertEqual(u.item(), 5.)
        u, _ = controller.get_control_input(torch.tensor(3.), target)
        self.assertEqual(u.item(), 3.5)
        controller.reset()
        u, _ = controller.get_control_input(torch.tensor(3.), target)
        self.assertEqual(u.item(), 2.5)

    def test_pi_clamps_negative_pump_input(self):
        controller = ProportionalIntegralController(k_p=2., k_i=.5)
        u, _ = controller.get_control_input(torch.tensor(5.), torch.tensor(4.))
        self.assertEqual(u.item(), 0.)

    def test_default_cartpole_reference_schedule(self):
        trajectory = CartPoleSimulator(dim=2).run_episode(torch.tensor([-40., -6.25]))
        reference = trajectory["reference"]
        self.assertEqual(len(reference), 1500)
        for start, stop, level in ((0, 100, 0.), (100, 500, -2.), (500, 1000, 2.), (1000, 1500, 0.)):
            self.assertTrue(torch.all(reference[start:stop] == level))
        for index, time in ((100, 2.), (500, 10.), (1000, 20.)):
            self.assertEqual(trajectory["time"][index].item(), time)
