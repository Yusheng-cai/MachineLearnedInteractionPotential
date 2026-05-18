from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_relationship(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if not (len(x) == len(y) == len(labels)):
        raise ValueError("x, y, and labels must have matching lengths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.scatter(x, y, color="#2458a7")
    for x_value, y_value, label in zip(x, y, labels, strict=True):
        ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
