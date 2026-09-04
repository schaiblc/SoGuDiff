# Pretrained baseline weights

One directory per learned baseline that is loaded through `--model_dir`:
CADRL, LSTM-RL, SARL and RGL. The evaluation harness reads
`<dir>/policy.config` and `<dir>/env.config` for the architecture the
checkpoint was trained under, then loads `rl_model.pth` (or `il_model.pth`
with `--il`).

The `.config` files are tracked in git because they pin those architectures.
The `.pth` weights are fetched by `scripts/download_assets.sh weights`; see
[../../../assets/MANIFEST.md](../../../assets/MANIFEST.md).

```bash
python evaluate.py --policy sarl --model_dir models/sarl --phase test --no_video
```

DSRNN, NaviSTAR, HEIGHT and SoGuDiff do not appear here: they read their
checkpoint paths from `configs/policy.config` instead, because their weights
live with the fork that trained them (`baselines/`) or in `checkpoints/`.
