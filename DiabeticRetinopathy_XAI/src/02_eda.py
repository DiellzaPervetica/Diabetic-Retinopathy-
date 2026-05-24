"""Exploratory data analysis for the APTOS 2019 training dataset."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR, RANDOM_SEED, ensure_directories
from utils.data_utils import (
    class_distribution_dataframe,
    load_available_train_dataframe,
)
from utils.image_utils import load_image_pil, read_image_dimensions


CLASS_DISTRIBUTION_FIG = FIGURES_DIR / "class_distribution.png"
CLASS_PERCENT_FIG = FIGURES_DIR / "class_distribution_percent.png"
SAMPLE_IMAGES_FIG = FIGURES_DIR / "sample_images_by_class.png"
EDA_SUMMARY_CSV = METRICS_DIR / "eda_summary.csv"


def plot_class_distribution(distribution: pd.DataFrame) -> None:
    """Save class count and percentage charts."""
    labels = distribution["class_name"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, distribution["count"], color="#2F6B8F")
    ax.set_title("APTOS 2019 Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, padding=3)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(CLASS_DISTRIBUTION_FIG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, distribution["percentage"], color="#6A994E")
    ax.set_title("APTOS 2019 Class Percentages")
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in distribution["percentage"]], padding=3)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(CLASS_PERCENT_FIG, dpi=150)
    plt.close(fig)


def plot_sample_images(df: pd.DataFrame, samples_per_class: int = 4) -> None:
    """Save a grid with sample images from each class."""
    rows = len(CLASS_NAMES)
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(rows, cols)

    for label, class_name in CLASS_NAMES.items():
        class_df = df[df["diagnosis"] == label]
        class_samples = class_df.sample(
            n=min(samples_per_class, len(class_df)),
            random_state=RANDOM_SEED,
        )

        for col in range(cols):
            ax = axes[label, col]
            ax.axis("off")
            if col >= len(class_samples):
                continue

            row = class_samples.iloc[col]
            image_path = Path(row["image_path"])
            if not image_path.exists():
                ax.set_title(f"{class_name}\nmissing image", fontsize=9)
                continue

            image = load_image_pil(image_path, normalize=False)
            ax.imshow(image)
            ax.set_title(f"{class_name}\n{row['id_code']}", fontsize=9)

    fig.suptitle("Sample Fundus Images by Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(SAMPLE_IMAGES_FIG, dpi=150)
    plt.close(fig)


def compute_dimension_summary(df: pd.DataFrame, sample_size: int) -> dict[str, float]:
    """Compute basic image dimension statistics for a manageable subset."""
    subset_size = min(sample_size, len(df))
    subset = df.sample(n=subset_size, random_state=RANDOM_SEED)

    widths = []
    heights = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Reading image dimensions"):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        width, height = read_image_dimensions(image_path)
        widths.append(width)
        heights.append(height)

    if not widths:
        return {
            "checked_images": 0,
            "width_min": np.nan,
            "width_mean": np.nan,
            "width_max": np.nan,
            "height_min": np.nan,
            "height_mean": np.nan,
            "height_max": np.nan,
        }

    return {
        "checked_images": len(widths),
        "width_min": int(np.min(widths)),
        "width_mean": float(np.mean(widths)),
        "width_max": int(np.max(widths)),
        "height_min": int(np.min(heights)),
        "height_mean": float(np.mean(heights)),
        "height_max": int(np.max(heights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for APTOS 2019.")
    parser.add_argument(
        "--dimension-sample",
        type=int,
        default=500,
        help="Number of images to inspect for dimension statistics.",
    )
    args = parser.parse_args()

    ensure_directories()
    print("Running exploratory data analysis...\n")

    try:
        df, missing_images = load_available_train_dataframe()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    if not missing_images.empty:
        print(
            f"WARNING: {len(missing_images)} images listed in train.csv are missing. "
            "EDA will use only images that already exist locally."
        )

    distribution = class_distribution_dataframe(df["diagnosis"])
    print("Class distribution:")
    print(distribution.round({"percentage": 2}).to_string(index=False))

    majority_count = distribution["count"].max()
    minority_count = distribution.loc[distribution["count"] > 0, "count"].min()
    imbalance_ratio = majority_count / minority_count if minority_count else np.nan
    print(
        f"\nImbalance signal: the largest class has about {imbalance_ratio:.2f} "
        "times more samples than the smallest non-empty class."
    )

    plot_class_distribution(distribution)
    print(f"Saved: {CLASS_DISTRIBUTION_FIG}")
    print(f"Saved: {CLASS_PERCENT_FIG}")

    plot_sample_images(df)
    print(f"Saved: {SAMPLE_IMAGES_FIG}")

    dimension_summary = compute_dimension_summary(df, args.dimension_sample)

    summary_rows = [
        {"metric": "total_images_in_train_csv", "value": len(df)},
        {"metric": "missing_images", "value": len(missing_images)},
        {"metric": "imbalance_ratio_majority_to_minority", "value": imbalance_ratio},
    ]
    for _, row in distribution.iterrows():
        summary_rows.append(
            {
                "metric": f"class_{int(row['diagnosis'])}_{row['class_name']}_count",
                "value": int(row["count"]),
            }
        )
        summary_rows.append(
            {
                "metric": f"class_{int(row['diagnosis'])}_{row['class_name']}_percentage",
                "value": float(row["percentage"]),
            }
        )
    for key, value in dimension_summary.items():
        summary_rows.append({"metric": key, "value": value})

    pd.DataFrame(summary_rows).to_csv(EDA_SUMMARY_CSV, index=False)
    print(f"Saved EDA summary: {EDA_SUMMARY_CSV}")
    print("\nEDA completed.")


if __name__ == "__main__":
    main()
