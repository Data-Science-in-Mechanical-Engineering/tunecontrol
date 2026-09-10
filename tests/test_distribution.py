"""Exercise every standard configuration from an installed distribution."""
import unittest
from importlib.metadata import version

import torch
import tunecontrol as tc


class DistributionTests(unittest.TestCase):
    def test_version(self):
        self.assertTrue(version("tunecontrol"))

    def test_all_configurations(self):
        self.assertEqual(tc.list_problems(), ["cartpole", "cascaded_tank"])
        for name, count in (("cartpole", 24), ("cascaded_tank", 10)):
            configs = tc.available_configs(name)
            self.assertEqual(len(configs), count)
            for config in configs:
                with self.subTest(family=name, config=config):
                    task = tc.make(name, config)
                    theta = task.bounds.mean(dim=0)
                    value, info = task.evaluate(theta)
                    self.assertEqual(value.ndim, 0)
                    self.assertTrue(torch.isfinite(value))
                    self.assertIn("trajectory", info)
                    self.assertEqual(task.config, config)
                    if config.noise is None:
                        repeated, _ = task.evaluate(theta)
                        torch.testing.assert_close(value, repeated, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
