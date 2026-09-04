# diffusers_unet_1d_condition — vendored, unmodified

A 1D conditional UNet with cross-attention: the counterpart to HuggingFace
Diffusers' `UNet2DConditionModel`, which SoGuDiff uses as its denoising
backbone over trajectory sequences.

## Provenance

| | |
|---|---|
| Vendored from | [lucainiaoge/diffusers-unet-1d-condition](https://github.com/lucainiaoge/diffusers-unet-1d-condition) |
| Which adapts | [huggingface/diffusers](https://github.com/huggingface/diffusers) v0.30.3 (`UNet2DConditionModel`, `resnet.py`, `embeddings.py`, ...) |
| Licence | Apache License 2.0 — see [LICENSE](LICENSE) |
| Modified here | **No.** Every file is byte-identical to upstream |

Each source file keeps its original `Adapted from Huggingface by Tongyu Lu`
header naming the exact upstream file it derives from.

It is vendored rather than pip-installed because it is not published to PyPI,
and pinning it here keeps the architecture fixed alongside the released
checkpoints.

## Usage

`test_unet_1d_condition.ipynb` is upstream's example: it defines a
`UNet1DConditionModel` and runs a forward pass. Nothing in SoGuDiff imports the
notebook — see `crowd_nav/policy/sogudiff.py` and `sogudiff/train.py` for how
the model is actually configured and called.
