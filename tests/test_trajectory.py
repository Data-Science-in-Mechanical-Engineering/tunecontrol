"""Shared trajectory structure and physical tank limits."""
import unittest
from unittest.mock import patch

import torch
import tunecontrol as tc
from tunecontrol.tasks.cascaded_tank.plant import CascadedTankDynamic


class TrajectoryTests(unittest.TestCase):
    def test_built_in_shapes_metadata_and_sample_spacing(self):
        for task, state_count, dt in ((tc.CartPole(), 4, .02), (tc.CascadedTank(), 2, 4.)):
            _, info = task.evaluate(task.bounds.mean(0).float())
            trajectory = info['trajectory']
            count = len(trajectory['time'])
            self.assertEqual(trajectory['states'].shape, (count, state_count))
            self.assertEqual(trajectory['inputs'].shape, (count, 1))
            for field, width in (('state', state_count), ('input', 1)):
                self.assertEqual(len(trajectory[field + '_names']), width)
                self.assertEqual(len(trajectory[field + '_units']), width)
            for field in ('time', 'states', 'inputs'):
                self.assertEqual(trajectory[field].dtype, torch.float32)
            torch.testing.assert_close(trajectory['time'].diff(), torch.full((count-1,), dt), atol=3e-6, rtol=1e-4)
            self.assertFalse({'x1', 'x2', 'u', 'control'} & trajectory.keys())

    def test_limits_apply_after_noise_without_preclipping(self):
        dynamic = CascadedTankDynamic(k1=0., k2=0., k3=0., k4=1.)
        # Raw update [11, 1], disturbed state [9, -2], physical result [9, 0].
        with patch('torch.randn', return_value=torch.tensor([-2., -3.], dtype=torch.float64)):
            result = dynamic.make_environment_step(torch.tensor([9., 1.]), 2., 1.)
        torch.testing.assert_close(result, torch.tensor([9., 0.], dtype=torch.float64))
        with patch('torch.randn', return_value=torch.tensor([3., 20.], dtype=torch.float64)):
            result = dynamic.make_environment_step(torch.tensor([9., 1.]), 2., 1.)
        torch.testing.assert_close(result, torch.tensor([10., 10.], dtype=torch.float64))

    def test_noisy_episode_respects_limits(self):
        task = tc.CascadedTank(tc.CascadedTankConfig(noise=tc.CascadedTankNoise(std=20.)))
        task.setup(42)
        _, info = task.evaluate(task.bounds.mean(0))
        states = info['trajectory']['states']
        self.assertTrue(torch.all((states >= 0) & (states <= 10)))
