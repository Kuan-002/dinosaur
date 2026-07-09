#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRPO selector training curves.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def read_history(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                parsed[key] = value if key == "elapsed" else float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "visualizations" / "training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_history(run_dir / "history_metrics.csv")
    epochs = [float(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    axes[0].plot(epochs, [float(row["train_acc"]) for row in rows], label="train")
    axes[0].plot(epochs, [float(row["valid_acc"]) for row in rows], label="valid")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[1].plot(epochs, [float(row["valid_loss"]) for row in rows], label="valid")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, [float(row["train_avg_selected"]) for row in rows], label="train")
    axes[2].plot(epochs, [float(row["valid_avg_selected"]) for row in rows], label="valid")
    axes[2].set_title("Avg selected slots")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle(run_dir.name, fontsize=10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
