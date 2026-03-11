#!/usr/bin/env python
"""Publication-quality visualizations for shortcut-learning experiments.

Generates:
  - Training curves (loss, accuracy, worst-group acc per epoch)
  - Method comparison bar charts
  - ECE reliability diagrams
  - Group accuracy heatmaps
  - Batch size sensitivity plots
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.utils import ensure_dir

# Style for publications
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(metrics_path: str, output_dir: str) -> None:
    """Plot loss, accuracy, worst-group accuracy from metrics.json."""
    with open(metrics_path) as f:
        data = json.load(f)
    history = data.get("history", data if isinstance(data, list) else [])
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", color="tab:red", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")

    # Accuracy
    axes[1].plot(epochs, [h["accuracy"] for h in history], "s-", color="tab:blue", label="Avg", linewidth=2)
    axes[1].plot(epochs, [h["worst_group_accuracy"] for h in history], "^-", color="tab:orange", label="Worst-Group", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    # ECE
    if "ece" in history[0]:
        axes[2].plot(epochs, [h["ece"] for h in history], "D-", color="tab:green", linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("ECE")
        axes[2].set_title("Expected Calibration Error")

    # Accuracy gap
    if "accuracy_gap" in history[0]:
        axes[2].plot(epochs, [h["accuracy_gap"] for h in history], "v-", color="tab:purple", linewidth=2, label="Acc Gap")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.pdf")
    plt.savefig(f"{output_dir}/training_curves.png")
    plt.close()
    print(f"Saved training_curves to {output_dir}")


# ---------------------------------------------------------------------------
# 2. Method comparison bar chart
# ---------------------------------------------------------------------------
def plot_method_comparison(results_dir: str, output_dir: str) -> None:
    """Compare metrics across methods from their metrics.json files."""
    results_path = Path(results_dir)
    methods = {}
    for mf in sorted(results_path.glob("*/metrics.json")):
        name = mf.parent.name
        with open(mf) as f:
            data = json.load(f)
        # Use last epoch or best
        history = data.get("history", [])
        if history:
            last = history[-1]
        else:
            last = data
        methods[name] = last

    if not methods:
        print("No metrics.json files found")
        return

    names = list(methods.keys())
    metrics_to_plot = ["accuracy", "worst_group_accuracy", "accuracy_gap", "ece"]
    labels = ["Accuracy ↑", "Worst-Group Acc ↑", "Acc Gap ↓", "ECE ↓"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    colors = sns.color_palette("Set2", len(names))

    for ax, metric, label in zip(axes, metrics_to_plot, labels):
        vals = [methods[n].get(metric, 0) for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison.pdf")
    plt.savefig(f"{output_dir}/method_comparison.png")
    plt.close()
    print(f"Saved method_comparison to {output_dir}")


# ---------------------------------------------------------------------------
# 3. Group accuracy heatmap
# ---------------------------------------------------------------------------
def plot_group_heatmap(results_dir: str, output_dir: str) -> None:
    """Heatmap: rows = methods, columns = groups, cells = accuracy."""
    results_path = Path(results_dir)
    methods = {}
    all_groups = set()
    for mf in sorted(results_path.glob("*/metrics.json")):
        name = mf.parent.name
        with open(mf) as f:
            data = json.load(f)
        history = data.get("history", [])
        last = history[-1] if history else data
        ga = last.get("group_accuracy", {})
        methods[name] = ga
        all_groups.update(ga.keys())

    if not methods or not all_groups:
        return

    groups = sorted(all_groups)
    names = list(methods.keys())
    matrix = np.zeros((len(names), len(groups)))
    for i, n in enumerate(names):
        for j, g in enumerate(groups):
            matrix[i, j] = methods[n].get(g, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), max(4, len(names) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=groups, yticklabels=names,
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Group Accuracy")
    ax.set_xlabel("Group")
    ax.set_ylabel("Method")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/group_accuracy_heatmap.pdf")
    plt.savefig(f"{output_dir}/group_accuracy_heatmap.png")
    plt.close()
    print(f"Saved group_accuracy_heatmap to {output_dir}")


# ---------------------------------------------------------------------------
# 4. Batch size sensitivity
# ---------------------------------------------------------------------------
def plot_batch_sensitivity(results_dir: str, output_dir: str, prefix: str = "bs") -> None:
    """Plot accuracy vs batch size from experiments named bs1, bs4, bs16, etc."""
    results_path = Path(results_dir)
    points = []
    for mf in sorted(results_path.glob(f"{prefix}*/metrics.json")):
        name = mf.parent.name
        try:
            bs = int(name.replace(prefix, ""))
        except ValueError:
            continue
        with open(mf) as f:
            data = json.load(f)
        history = data.get("history", [])
        last = history[-1] if history else data
        points.append((bs, last.get("accuracy", 0), last.get("worst_group_accuracy", 0)))

    if not points:
        return

    points.sort()
    bss = [p[0] for p in points]
    accs = [p[1] for p in points]
    worsts = [p[2] for p in points]

    fig, ax = plt.subplots()
    ax.plot(bss, accs, "o-", label="Avg Accuracy", linewidth=2)
    ax.plot(bss, worsts, "s--", label="Worst-Group Acc", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Batch Size Sensitivity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/batch_sensitivity.pdf")
    plt.savefig(f"{output_dir}/batch_sensitivity.png")
    plt.close()
    print(f"Saved batch_sensitivity to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Generate publication plots")
    p.add_argument("--metrics", default=None, help="Path to a single metrics.json")
    p.add_argument("--results_dir", default="models", help="Dir containing model subdirs")
    p.add_argument("--output_dir", default="results/figures", help="Where to save figures")
    args = p.parse_args()

    ensure_dir(args.output_dir)

    if args.metrics:
        plot_training_curves(args.metrics, args.output_dir)

    plot_method_comparison(args.results_dir, args.output_dir)
    plot_group_heatmap(args.results_dir, args.output_dir)
    plot_batch_sensitivity(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
