"""Hand-computed step responses pin down the tank rise-time convention."""
import unittest

import torch
import tunecontrol as tc
from tunecontrol.tasks.cascaded_tank.objectives import get_cascaded_tank_objective_spec
from tunecontrol.tasks.cascaded_tank.plant import CascadedTankSimulator


class TankRiseTimeTests(unittest.TestCase):
    def score(self, levels, times, target):
        trajectory = {
            "states": torch.tensor([[0., level] for level in levels], dtype=torch.float64),
            "time": torch.tensor(times, dtype=torch.float64),
        }
        return get_cascaded_tank_objective_spec("rise_time").evaluate(
            CascadedTankSimulator(target=target), trajectory,
        )

    def test_nonzero_initial_level(self):
        # Change 2 -> 4: thresholds 2.2 and 3.8, reached at 4 and 12 seconds.
        self.assertEqual(self.score([2, 2.1, 2.3, 3.7, 3.9], [0, 2, 4, 8, 12], 4), 8)

    def test_samples_exactly_at_thresholds(self):
        self.assertEqual(self.score([2, 2.2, 3.8], [0, 4, 12], 4), 8)
        self.assertEqual(self.score([6, 5.8, 4.2], [0, 4, 12], 4), 8)

    def test_default_initial_level(self):
        self.assertEqual(self.score([1.944, 2.14, 2.16, 3.79, 3.80], [0, 4, 8, 12, 16], 4), 8)

    def test_zero_initial_level(self):
        self.assertEqual(self.score([0, 0.5, 3, 3.7], [0, 4, 8, 12], 4), 8)

    def test_falling_response(self):
        # Change 6 -> 4: thresholds 5.8 and 4.2.
        self.assertEqual(self.score([6, 5.9, 5.7, 4.3, 4.1], [0, 2, 4, 8, 12], 4), 8)

    def test_first_crossings_on_nonmonotonic_response(self):
        self.assertEqual(self.score([2, 2.3, 2.0, 3.9, 2.1], [0, 4, 8, 12, 16], 4), 8)

    def test_both_crossings_in_same_sample(self):
        self.assertEqual(self.score([2, 4], [0, 4], 4), 0)

    def test_no_commanded_change(self):
        self.assertEqual(self.score([4, 4.1, 3.9], [0, 4, 8], 4), 0)

    def test_missing_crossings_are_undefined(self):
        for levels, target in (([2, 2.1, 2.15], 4), ([2, 2.4, 3.7], 4), ([6, 5.7, 4.3], 4)):
            with self.subTest(levels=levels):
                self.assertTrue(torch.isnan(self.score(levels, [0, 4, 8], target)))

    def test_actual_task_uses_relative_thresholds(self):
        task = tc.CascadedTank(tc.CascadedTankConfig(objective="rise_time"))
        value, info = task.evaluate(task.bounds.mean(0))
        trajectory = info["trajectory"]
        levels, times = trajectory["states"][:, 1], trajectory["time"]
        first = times[torch.nonzero(levels >= 2.1496)[0, 0]]
        last = times[torch.nonzero(levels >= 3.7944)[0, 0]]
        self.assertGreater(first.item(), 0)
        self.assertEqual(value, last - first)
