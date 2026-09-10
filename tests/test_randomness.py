"""Episode reproducibility without dependence on optimizer random draws."""
import unittest

import torch
import tunecontrol as tc
from tunecontrol.tasks.cartpole.plant import CartPoleSimulator
from tunecontrol.tasks.cascaded_tank.plant import CascadedTankSimulator, CascadedTankDynamic


class RandomnessTests(unittest.TestCase):
    def assert_episode_equal(self, a, b):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        self.assertEqual(set(a[1]["trajectory"]), set(b[1]["trajectory"]))
        for key in a[1]["trajectory"]:
            if isinstance(a[1]["trajectory"][key], torch.Tensor):
                torch.testing.assert_close(a[1]["trajectory"][key], b[1]["trajectory"][key], rtol=0, atol=0)
            else:
                self.assertEqual(a[1]["trajectory"][key], b[1]["trajectory"][key])

    def test_every_noisy_variant_replays_sequence(self):
        for family in tc.list_problems():
            for config in tc.available_configs(family):
                if config.noise is None:
                    continue
                with self.subTest(family=family, config=config):
                    task = tc.make(family, config)
                    x = task.bounds.mean(0)
                    task.setup(42)
                    first, second = task.evaluate(x), task.evaluate(x)
                    # The stream advances; it is not reseeded on every episode.
                    self.assertTrue(any(not torch.equal(first[1]["trajectory"][k], second[1]["trajectory"][k])
                                        for k in ("states", "inputs")))
                    task.setup(42)
                    self.assert_episode_equal(first, task.evaluate(x))
                    self.assert_episode_equal(second, task.evaluate(x))

    def test_other_tasks_and_global_draws_do_not_change_episode(self):
        for name in tc.list_problems():
            config = next(c for c in tc.available_configs(name) if c.noise is not None)
            a, b = tc.make(name, config), tc.make(name, config)
            a.setup(71)
            b.setup(71)
            x = a.bounds.mean(0)
            reference = b.evaluate(x)
            torch.rand(123)
            b.setup(92)
            b.evaluate(x)
            self.assert_episode_equal(reference, a.evaluate(x))

    def test_construction_setup_and_evaluation_leave_global_rng_unchanged(self):
        for family in tc.list_problems():
            config = next(c for c in tc.available_configs(family) if c.noise is not None)
            for seed in (None, 42):
                before = torch.random.get_rng_state().clone()
                task = tc.make(family, config)
                task.setup(seed)
                task.evaluate(task.bounds.mean(0))
                torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    def test_no_seed_preserves_stream_and_invalid_seeds_fail(self):
        a, b = tc.CascadedTank(), tc.CascadedTank()
        a.setup(42)
        b.setup(42)
        torch.randn(5, generator=a.generator)
        torch.randn(5, generator=b.generator)
        a.setup()
        torch.testing.assert_close(torch.randn(5, generator=a.generator), torch.randn(5, generator=b.generator))
        for seed in (True, 1.5, "42", -(2**63)-1, 2**64):
            with self.assertRaises((TypeError, ValueError)):
                a.setup(seed)

    def test_direct_simulators_do_not_use_global_rng(self):
        before = torch.random.get_rng_state().clone()
        cp = CartPoleSimulator(dim=2, simulation_time=0.1,
                               simulation_noise={"initial_condition_std": .001, "process_noise_std": .001})
        cp.generator.manual_seed(4)
        first = cp.run_episode(torch.tensor([-40., -6.25]))
        cp.generator.manual_seed(4)
        second = cp.run_episode(torch.tensor([-40., -6.25]))
        torch.testing.assert_close(first["states"], second["states"], rtol=0, atol=0)
        tank = CascadedTankSimulator(duration=8.)
        tank.generator.manual_seed(4)
        first = tank.simulate(2., .1)
        tank.generator.manual_seed(4)
        second = tank.simulate(2., .1)
        torch.testing.assert_close(first["states"][:, 1], second["states"][:, 1], rtol=0, atol=0)
        CascadedTankDynamic().make_environment_step()
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
