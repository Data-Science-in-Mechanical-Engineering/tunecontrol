"""Hand-computed ITAE costs with physical time and unchanged effort penalty."""
from types import SimpleNamespace
import unittest

import torch
import tunecontrol as tc
from tunecontrol.tasks.cartpole.objectives import get_cartpole_objective_spec


class CartPoleITAETests(unittest.TestCase):
    def score(self, times, positions=(5., 6., 7.), inputs=(1., 2., 3.), reference=4.):
        states = torch.zeros((len(times), 4), dtype=torch.float64)
        states[:, 0] = torch.tensor(positions, dtype=torch.float64)
        trajectory = {
            "states": states,
            "inputs": torch.tensor(inputs, dtype=torch.float64),
            "reference": torch.full((len(times),), reference, dtype=torch.float64),
            "time": torch.tensor(times, dtype=torch.float64),
        }
        sim = SimpleNamespace(cost_weights=(10 * torch.eye(4, dtype=torch.float64),
                                           torch.ones((1, 1), dtype=torch.float64)))
        return get_cartpole_objective_spec("itae").evaluate(sim, trajectory).item()

    def test_seconds_with_tracking_error_and_effort(self):
        # Error magnitudes [1,2,3], Q=10: weighted error mean is 1.6/3.
        # Squared inputs [1,4,9] give 14/3, for a total of 5.2.
        self.assertAlmostEqual(self.score([0., .02, .04]), 5.2)

    def test_uses_recorded_times_instead_of_assuming_a_sample_interval(self):
        self.assertAlmostEqual(self.score([0., .1, .4]), 28 / 3)

    def test_elapsed_time_is_independent_of_timestamp_origin(self):
        self.assertAlmostEqual(self.score([10., 10.02, 10.04]), 5.2)

    def test_first_sample_has_zero_time_weight(self):
        self.assertEqual(self.score([0.], positions=[100.], inputs=[2.]), 4.)

    def test_effort_is_not_time_weighted(self):
        self.assertAlmostEqual(self.score([0., 10., 20.], positions=[4., 4., 4.]), 14 / 3)

    def test_actual_midpoint_reference(self):
        problem = tc.CartPole(tc.CartPoleConfig(dim=2, objective="itae"))
        value, _ = problem.evaluate(problem.bounds.mean(0))
        self.assertAlmostEqual(value.item(), 94.2052761407549, places=8)
