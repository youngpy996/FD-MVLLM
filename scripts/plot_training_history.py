"""Render publication-ready training curves from a result CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Path to training_history.csv")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--best-epoch", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history = pd.read_csv(args.history)
    required = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_f1",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    best_epoch = args.best_epoch
    if best_epoch is None:
        best_epoch = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])

    preferred_style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "seaborn-whitegrid"
    )
    plt.style.use(preferred_style)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)

    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(
        history["epoch"],
        history["val_accuracy"] * 100,
        label="Validation accuracy",
    )
    axes[1].plot(
        history["epoch"],
        history["val_f1"] * 100,
        label="Validation macro-F1",
    )
    axes[1].axvline(
        best_epoch,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch: {best_epoch}",
    )
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Percent")
    axes[1].set_ylim(88, 100.2)
    axes[1].legend()

    figure.suptitle("FD-MVLLM — CWRU 1024-sample experiment")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
