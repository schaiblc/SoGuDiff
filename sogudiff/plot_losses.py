"""Plot training and validation loss curves from a saved checkpoint.

Checkpoints written by train.py carry the loss history alongside the weights,
so no separate log file is needed.

    python sogudiff/plot_losses.py checkpoints/my_model/ckpt_step50000_my_model.pt
    python sogudiff/plot_losses.py <ckpt> --out losses.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")          # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", help="Path to a ckpt_step<N>_<exp_name>.pt file.")
    p.add_argument("--out", default=None,
                   help="Output image. Defaults to <checkpoint>_losses.png.")
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    missing = [k for k in ("train_losses", "val_losses") if k not in ckpt]
    if missing:
        raise SystemExit(
            f"{a.checkpoint} has no {', '.join(missing)}. Loss history is saved by "
            "train.py; a checkpoint from another source will not carry it.")

    train, val = ckpt["train_losses"], ckpt["val_losses"]
    print(f"{len(train)} training points, {len(val)} validation points")

    # Validation runs every save_every steps; space its points to match.
    save_every = len(train) // max(1, len(val))
    train_x = np.arange(len(train))
    val_x = np.arange(0, save_every * len(val), save_every)[:len(val)]

    plt.figure()
    plt.plot(train_x, train, label="training", alpha=0.4)
    plt.plot(val_x, val, marker="o", label="validation")
    if "train_loss_avg" in ckpt:
        avg = ckpt["train_loss_avg"]
        plt.plot(val_x[:len(avg)], avg[:len(val_x)], label="training average")

    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("MSE noise-prediction loss")
    plt.legend()
    plt.grid(True)

    out = a.out or os.path.splitext(a.checkpoint)[0] + "_losses.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
