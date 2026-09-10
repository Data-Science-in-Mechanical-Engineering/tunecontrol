# Examples

Install from the repository root with `python -m pip install -e '.[botorch]'`.
The Python scripts need only the base package. To open the notebooks, use your
editor's notebook support or install Jupyter separately. Run notebook cells in order.

Commit notebooks after restarting the kernel and running every cell, with text
outputs and plots saved so readers can inspect the results on GitHub.

| Example | Purpose |
|---|---|
| [quickstart.py](quickstart.py) | Construct, evaluate, and plot both built-in problems |
| [discover_problems.py](discover_problems.py) | List families and serialize a noisy configuration |
| [standard_bo_for_controller_tuning.ipynb](standard_bo_for_controller_tuning.ipynb) | Tune one gain with a single, explicit BoTorch loop |
| [creating_custom_task.ipynb](creating_custom_task.ipynb) | Implement a mass-spring-damper family with a configuration and trajectory |

Run the scripts from the repository root:

```bash
python examples/quickstart.py
python examples/discover_problems.py
```

The BO notebook uses a penalty of 20 for failed evaluations of its default
CartPole MAE objective. Adjust it when changing objectives; it is not a package
failure policy. Methodological guidance is in [the paper](https://arxiv.org/abs/2609.09403).

To regenerate the documentation's landscape image, use the separate maintenance script:

```bash
python scripts/objective_gallery.py --no-show --output docs/figures/deterministic_2d_gallery.png
```
