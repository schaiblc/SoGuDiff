# Model checkpoints

Empty by design. `scripts/download_assets.sh` populates this directory; see
[../assets/MANIFEST.md](../assets/MANIFEST.md) for what lands here and how
large each file is.

`configs/policy.config` points `[sogudiff] ckpt_path` at
`checkpoints/sogudiff_single_axis.pt`, the checkpoint
behind the paper's main results. Training writes here too, under
`checkpoints/<exp_name>/`.
