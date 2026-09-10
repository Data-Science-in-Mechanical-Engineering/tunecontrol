"""Exercise the actual notebook helpers on flat costs and failed evaluations."""
import json
from pathlib import Path
import unittest

import torch
import tunecontrol as tc


class BOExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((Path(__file__).parents[1] / 'examples/standard_bo_for_controller_tuning.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if ''.join(c['source']).startswith('def eval_objective'))
        cls.helpers = {'torch': torch, 'tc': tc, 'device': torch.device('cpu'),
                       'dtype': torch.float64, 'DIM': 1, 'FAILURE_PENALTY': 20.,
                       'bounds': torch.tensor([[0.], [1.]]), 'unnormalize': tc.unnormalize}
        exec(source, cls.helpers)

    def test_constant_and_single_observations(self):
        for costs in (torch.ones(3, 1), torch.tensor([[7.]])):
            scaled, stats = self.helpers['standardize_costs'](costs)
            torch.testing.assert_close(scaled, torch.zeros_like(costs))
            self.assertEqual(stats['std'].item(), 1.)

    def test_variable_observations_keep_standardization(self):
        costs = torch.tensor([[1.], [3.], [7.]])
        scaled, _ = self.helpers['standardize_costs'](costs)
        torch.testing.assert_close(scaled, (costs-costs.mean())/costs.std())

    def test_failure_penalty_keeps_one_observation_per_call(self):
        class FailingExample:
            calls = 0
            def evaluate(self, theta):
                self.calls += 1
                return theta.new_tensor(float('nan') if self.calls == 1 else 2.), {}
        task = FailingExample()
        costs = self.helpers['eval_objective'](torch.tensor([[.1], [.2]]), task)
        self.assertEqual(task.calls, 2)
        torch.testing.assert_close(costs, torch.tensor([20., 2.], dtype=torch.float64))
