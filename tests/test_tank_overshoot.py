"""Hand-computed examples establish percentage units and nonnegative scores."""
import unittest

import torch
import tunecontrol as tc
from tunecontrol.tasks.cascaded_tank.objectives import get_cascaded_tank_objective_spec
from tunecontrol.tasks.cascaded_tank.plant import CascadedTankSimulator


class TankOvershootTests(unittest.TestCase):
    def score(self, levels, target=4., dtype=torch.float64):
        return get_cascaded_tank_objective_spec("overshoot").evaluate(
            CascadedTankSimulator(target=target),
            {"states": torch.tensor([[0., level] for level in levels], dtype=dtype)},
        )

    def test_below_and_at_target(self):
        for levels in ([1.944, 3., 3.6], [1.944, 4., 3.9], [4., 4., 4.]):
            with self.subTest(levels=levels):
                self.assertEqual(self.score(levels).item(), 0.)

    def test_percentage_of_target_and_peak_not_final(self):
        self.assertAlmostEqual(self.score([1.944, 4.4, 4.]).item(), 10.)
        self.assertAlmostEqual(self.score([1., 2.5, 2.], target=2.).item(), 25.)

    def test_initial_level_does_not_change_denominator(self):
        self.assertAlmostEqual(self.score([0., 4.4, 4.]).item(), 10.)
        self.assertAlmostEqual(self.score([3., 4.4, 4.]).item(), 10.)

    def test_initial_peak_is_included(self):
        self.assertEqual(self.score([6., 5., 4.]).item(), 50.)

    def test_scalar_and_dtype(self):
        for dtype in (torch.float32, torch.float64):
            result = self.score([2., 5., 4.], dtype=dtype)
            self.assertEqual(result.shape, ())
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.item(), 25.)

    def test_actual_undertracking_controller_has_zero_score(self):
        problem = tc.CascadedTank(tc.CascadedTankConfig(objective="overshoot"))
        value, info = problem.evaluate(torch.tensor([0.9, 0.01], dtype=torch.float64))
        self.assertLess(info["trajectory"]["states"][:, 1].max().item(), 4.)
        self.assertEqual(value.item(), 0.)
