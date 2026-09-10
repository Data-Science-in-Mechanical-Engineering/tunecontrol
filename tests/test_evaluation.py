"""Regression tests for malformed controllers and objective results."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
import tunecontrol as tc


class Example(tc.Task):
    dim = 2
    bounds = torch.tensor([[0., 0.], [1., 1.]])

    def _evaluate(self, theta):
        return theta.sum(), {"theta": theta.clone()}


class EvaluationTests(unittest.TestCase):
    def test_invalid_inputs_never_reach_simulator(self):
        invalid = [
            [1., 2.], torch.tensor([1, 2]), torch.tensor([True, False]),
            torch.tensor([1j, 2j]), torch.tensor(1.), torch.zeros(1, 2),
            torch.zeros(1), torch.zeros(3), torch.tensor([float("nan"), 1.]),
            torch.tensor([float("inf"), 1.]),
        ]
        for family in (tc.CartPole, tc.CascadedTank):
            task = family()
            with patch.object(task, "_evaluate", side_effect=AssertionError("simulated invalid input")):
                for theta in invalid:
                    with self.subTest(family=family, theta=theta), self.assertRaises((TypeError, ValueError)):
                        task.evaluate(theta)

    def test_invalid_bounds(self):
        task = Example()
        for bounds in (torch.zeros(2, 2), torch.ones(2, 3), torch.tensor([[1., 0.], [0., 1.]]),
                       torch.tensor([[0., 0.], [1., float("inf")]]), torch.tensor([[0, 0], [1, 1]])):
            task.bounds = bounds
            with self.subTest(bounds=bounds), self.assertRaises((TypeError, ValueError)):
                task.evaluate(torch.zeros(2))

    def test_scalar_shape_dtype_and_precision(self):
        task = Example()
        for dtype in (torch.float32, torch.float64):
            theta = torch.tensor([0.25, 0.5], dtype=dtype)
            for output in (0.75, torch.tensor([[0.75]]), 1):
                with patch.object(task, "_evaluate", return_value=(output, {})):
                    value, _ = task.evaluate(theta)
                    self.assertEqual(value.shape, ())
                    self.assertEqual(value.dtype, dtype)
                    self.assertEqual(value.device, theta.device)
        with patch.object(task, "_evaluate", return_value=(1.123456789012345, {})):
            value, _ = task.evaluate(torch.zeros(2, dtype=torch.float64))
            self.assertEqual(value.item(), 1.123456789012345)

    def test_invalid_outputs(self):
        task = Example()
        for output in (torch.ones(2), torch.empty(0), 1j, True):
            with self.subTest(output=output), patch.object(task, "_evaluate", return_value=(output, {})):
                with self.assertRaises((TypeError, ValueError)):
                    task.evaluate(torch.zeros(2))
        with patch.object(task, "_evaluate", return_value=(1., [])):
            with self.assertRaises(TypeError):
                task.evaluate(torch.zeros(2))

    def test_nonfinite_output_preserves_diagnostics(self):
        task = Example()
        info = {"trajectory": {"states": torch.zeros(3, 2)}}
        for output in (float("nan"), float("inf"), -float("inf"), 1e100):
            with self.subTest(output=output), patch.object(task, "_evaluate", return_value=(output, info)):
                theta = torch.zeros(2, dtype=torch.float32)
                value, returned_info = task.evaluate(theta)
                self.assertIs(returned_info, info)
                self.assertTrue(torch.isnan(value))
                self.assertEqual(value.shape, ())
                self.assertEqual(value.dtype, theta.dtype)
                self.assertEqual(value.device, theta.device)

    def test_simulator_errors_are_not_hidden(self):
        task = Example()
        with patch.object(task, "_evaluate", side_effect=RuntimeError("simulation bug")):
            with self.assertRaisesRegex(RuntimeError, "simulation bug"):
                task.evaluate(torch.zeros(2))

    def test_outside_search_box_never_reaches_simulator(self):
        for family in (Example, tc.CartPole, tc.CascadedTank):
            task = family()
            for dtype in (torch.float32, torch.float64):
                bounds = task.bounds.to(dtype=dtype)
                for side, direction in ((0, -float("inf")), (1, float("inf"))):
                    for dim in range(task.dim):
                        theta = bounds.mean(0)
                        theta[dim] = torch.nextafter(bounds[side, dim], theta.new_tensor(direction))
                        with self.subTest(family=family, dtype=dtype, side=side, dim=dim):
                            with patch.object(task, "_evaluate") as evaluate:
                                with self.assertRaisesRegex(ValueError, "within the declared bounds"):
                                    task.evaluate(theta)
                                evaluate.assert_not_called()

    def test_boundary_gains_are_accepted_without_modification(self):
        for family in (Example, tc.CartPole, tc.CascadedTank):
            task = family()
            for dtype in (torch.float32, torch.float64):
                for theta in task.bounds.to(dtype=dtype):
                    original = theta.clone()
                    with patch.object(task, "_evaluate", return_value=(0., {})) as evaluate:
                        task.evaluate(theta)
                        evaluate.assert_called_once()
                        self.assertIs(evaluate.call_args.args[0], theta)
                        torch.testing.assert_close(theta, original)

    def test_transform_validation_and_batched_round_trip(self):
        theta = torch.tensor([[0., 2.], [-3., 8.]], dtype=torch.float64)
        bounds = np.array([[-1., 0.], [1., 4.]])
        torch.testing.assert_close(tc.unnormalize(tc.normalize(theta, bounds), bounds), theta)
        for fn in (tc.normalize, tc.unnormalize):
            for bad in (np.zeros((2, 2)), np.zeros((2, 3)), np.array([[0., 0.], [1., np.nan]])):
                with self.assertRaises(ValueError):
                    fn(theta, bad)
            with self.assertRaises(TypeError):
                fn(torch.tensor([0, 1]), bounds)

    def test_unreached_tank_rise_time_is_reported(self):
        task = tc.CascadedTank(tc.CascadedTankConfig(objective="rise_time", duration=4.))
        value, info = task.evaluate(task.bounds.mean(0))
        self.assertIn("trajectory", info)
        self.assertTrue(torch.isnan(value))
        self.assertEqual(value.dtype, task.bounds.dtype)
