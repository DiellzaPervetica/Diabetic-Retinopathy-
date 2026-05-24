"""Metric calculation and plotting helpers."""

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from config import CLASS_NAMES


def save_json(data: Mapping, output_path: Path) -> None:
    """Save a dictionary as pretty JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix using matplotlib only."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = "Blues"
    image = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if normalize:
                text = f"{matrix[i, j]:.2f}"
            else:
                text = f"{int(matrix[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

