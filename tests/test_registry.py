"""Family discovery, extension and validated configuration contracts."""
from dataclasses import asdict, dataclass, FrozenInstanceError
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import tunecontrol as tc
from tunecontrol import registry
from tunecontrol.tasks.cartpole.plant import CartPoleSimulator
from tunecontrol.tasks.cascaded_tank.plant import CascadedTankSimulator


class RegistryTests(unittest.TestCase):
    def test_discovery_does_not_construct_simulators(self):
        with patch.object(CartPoleSimulator, "__post_init__", side_effect=AssertionError), \
             patch.object(CascadedTankSimulator, "__post_init__", side_effect=AssertionError):
            tc.list_problems()
            self.assertEqual(len(tc.CartPole.available_configs()), 24)
            self.assertEqual(len(tc.available_configs("cascaded_tank")), 10)

    def test_listing_does_not_import_builtin_simulators(self):
        subprocess.run([sys.executable, "-I", "-c", """
import sys
import tunecontrol as tc
assert tc.list_problems() == ['cartpole', 'cascaded_tank']
assert tc.describe('cartpole').description
assert 'tunecontrol.tasks.cartpole.plant' not in sys.modules
assert 'tunecontrol.tasks.cascaded_tank.plant' not in sys.modules
"""], check=True)

    def test_json_round_trip_and_direct_construction(self):
        for name, family in (("cartpole", tc.CartPole), ("cascaded_tank", tc.CascadedTank)):
            for config in family.available_configs():
                reconstructed = tc.make(name, json.loads(json.dumps(asdict(config))))
                self.assertEqual(reconstructed.config, config)
                self.assertIsInstance(reconstructed, family)
        custom = tc.CascadedTankConfig(
            duration=20, noise=tc.CascadedTankNoise(std=0.02),
            dynamics=tc.CascadedTankDynamics(time_step=2, initial_state=(2., 1.)),
        )
        task = tc.make("cascaded_tank", json.loads(json.dumps(asdict(custom))))
        self.assertEqual(task.config, custom)
        _, info = task.evaluate(task.bounds.mean(0))
        self.assertEqual(info["trajectory"]["time"][-1], 20)
        self.assertEqual(info["trajectory"]["states"][:, 0][0], 2)

    def test_invalid_configuration(self):
        for kwargs in ({"dim": 0}, {"dim": True}, {"objective": "sse"},
                       {"noise": "default"}, {"noise": {"process_noise_std": -1}}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                tc.make("cartpole", kwargs)
        for kwargs in ({"duration": 0}, {"target": float("nan")}, {"noise": {"std": -1}},
                       {"dynamics": {"time_step": 0}}, {"dynamics": {"initial_state": [1]}}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                tc.make("cascaded_tank", kwargs)
        with self.assertRaises(TypeError):
            tc.make("cartpole", tc.CascadedTankConfig())
        with self.assertRaises(TypeError):
            tc.CartPole(tc.CascadedTankConfig())
        with self.assertRaises(TypeError):
            tc.make("cartpole", {"typo": 2})
        with self.assertRaises(ValueError):
            tc.make("unknown")
        with self.assertRaises(ValueError):
            tc.make("invalid family name")

    def test_config_and_instance_independence(self):
        config = tc.CartPoleConfig(noise=tc.CartPoleNoise())
        a, b = tc.CartPole(config), tc.CartPole(config)
        with self.assertRaises(FrozenInstanceError):
            config.dim = 4
        a._sim.simulation_noise["process_noise_std"] = 0.1
        self.assertEqual(b._sim.simulation_noise["process_noise_std"], 0.0005)
        self.assertEqual(config.noise.process_noise_std, 0.0005)
        configs = tc.available_configs("cartpole")
        configs.clear()
        self.assertEqual(len(tc.available_configs("cartpole")), 24)

    def test_register_new_family_without_central_changes(self):
        @dataclass(frozen=True)
        class ExampleConfig:
            offset: float = 1.0

        class Example(tc.Task):
            config_type = ExampleConfig
            dim = 1
            bounds = torch.tensor([[0.], [10.]])
            def __init__(self, config):
                self.config = config
            @classmethod
            def available_configs(cls):
                return [ExampleConfig(1), ExampleConfig(2)]
            def _evaluate(self, theta):
                return theta.sum() + self.config.offset, {}

        with patch.dict(registry._FAMILIES):
            self.assertNotIn("example", tc.list_problems())
            tc.register_problem("example", Example, description="An example family")
            self.assertEqual(len(tc.available_configs("example")), 2)
            value, _ = tc.make("example", {"offset": 3}).evaluate(torch.tensor([2.]))
            self.assertEqual(value.item(), 5)
            with self.assertRaises(ValueError):
                tc.register_problem("example", Example)
            with self.assertRaises(TypeError):
                tc.register_problem("invalid", object)

    def test_lazy_external_registration(self):
        with patch.dict(registry._FAMILIES):
            tc.register_problem("external", "missing_dependency:External")
            self.assertIn("external", tc.list_problems())
            with self.assertRaises(ModuleNotFoundError):
                tc.make("external")


if __name__ == "__main__":
    unittest.main()
