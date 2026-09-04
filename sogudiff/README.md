# SoGuDiff — training and data generation

| File | Purpose |
|---|---|
| `train.py` | Diffusion training. The entry point behind every reported checkpoint |
| `social_cost_torch.py` | Differentiable social cost terms and the style-axis definitions, shared by training and the expert planner |
| `generate_expert_trajectories.py` | Expert demonstrations: sampling-based MPPI planning with multi-modal, style-labelled output |
| `visualize_expert_trajectories.py` | Renders generated demonstrations for inspection |
| `plot_losses.py` | Training-curve plots from a run's logs |

These are scripts run directly rather than an installed package, so
`social_cost_torch` imports by sibling-directory lookup. Run them from the
repository root:

```bash
python sogudiff/train.py --dataset_dir data/expert_trajectories --exp_name sogudiff ...
```

See [../docs/RUNNING.md](../docs/RUNNING.md) for full arguments.
