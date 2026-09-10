"""Smoke tests that also run against an installed wheel, outside the checkout."""

import unittest
from importlib.metadata import version

import torch
import tunecontrol as tc


class DistributionTests(unittest.TestCase):
    def test_version(self):
        self.assertTrue(version("tunecontrol"))

    def test_registry_and_evaluation(self):
        expected = {
            f"cartpole/{dim}d/{objective}/{noise}"
            for dim in (1, 2, 3, 4)
            for objective in ("mae", "lqr", "itae")
            for noise in ("deterministic", "default_noise")
        } | {
            f"cascaded_tank/2d/{objective}/{noise}"
            for objective in ("logsse", "sse", "quadratic", "rise_time", "overshoot")
            for noise in ("deterministic", "default_noise")
        }
        self.assertEqual(set(tc.tasks.list_task_names()), expected)
        for name in sorted(expected):
            with self.subTest(task=name):
                task = tc.make(name)
                theta = task.bounds.mean(dim=0)
                task.setup(run_seed=42)
                value, info = task.evaluate(theta)
                self.assertEqual(value.numel(), 1)
                self.assertTrue(torch.isfinite(value).all())
                self.assertIn("trajectory", info)
                if name.endswith("/deterministic"):
                    repeated, _ = task.evaluate(theta)
                    torch.testing.assert_close(value, repeated, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
