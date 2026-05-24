"""Generate Grad-CAM explanations for selected test images."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EFFICIENTNET_BEST_MODEL_PATH,
    GRADCAM_DIR,
    GRADCAM_EXAMPLES_FIG,
    GRADCAM_SELECTED_CASES_CSV,
    PREDICTIONS_CSV,
    TEST_SPLIT_CSV,
    ensure_directories,
)
from utils.data_utils import ensure_image_path_column, load_split_dataframe
from utils.gradcam_utils import (
    find_last_conv_layer,
    make_gradcam_heatmap,
    overlay_heatmap,
    save_overlay,
)
from utils.image_utils import load_image_pil


def normalize_correct_column(series: pd.Series) -> pd.Series:
    """Convert a CSV correct column to booleans."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def select_cases(predictions: pd.DataFrame, num_cases: int) -> pd.DataFrame:
    """Select correct and incorrect examples from multiple classes."""
    predictions = predictions.copy()
    predictions["correct_bool"] = normalize_correct_column(predictions["correct"])

    selected_indices: list[int] = []
    for correctness in [True, False]:
        subset = predictions[predictions["correct_bool"] == correctness]
        for label in CLASS_NAMES:
            candidates = subset[subset["true_label"] == label].sort_values(
                "confidence",
                ascending=False,
            )
            for idx in candidates.index:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    break
            if len(selected_indices) >= num_cases:
                break
        if len(selected_indices) >= num_cases:
            break

    if len(selected_indices) < num_cases:
        remaining = predictions.drop(index=selected_indices, errors="ignore").sort_values(
            "confidence",
            ascending=False,
        )
        selected_indices.extend(remaining.head(num_cases - len(selected_indices)).index.tolist())

    return predictions.loc[selected_indices].head(num_cases).reset_index(drop=True)


def safe_filename(text: str) -> str:
    """Create a simple filename-safe string."""
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text)


def save_combined_grid(records: list[dict]) -> None:
    """Save a combined Grad-CAM figure for the report."""
    if not records:
        return

    cols = min(4, len(records))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.4))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, record in zip(axes.ravel(), records):
        image = plt.imread(record["gradcam_path"])
        ax.imshow(image)
        status = "Correct" if record["correct"] else "Incorrect"
        ax.set_title(
            f"{record['image_id']}\nTrue: {record['true_class']}\n"
            f"Pred: {record['predicted_class']} ({record['confidence']:.2f})\n{status}",
            fontsize=9,
        )

    fig.suptitle("Grad-CAM Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(GRADCAM_EXAMPLES_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations.")
    parser.add_argument("--model-path", type=str, default=str(EFFICIENTNET_BEST_MODEL_PATH))
    parser.add_argument("--predictions-csv", type=str, default=str(PREDICTIONS_CSV))
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--layer-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    ensure_directories()
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    predictions_path = Path(args.predictions_csv)

    if not model_path.exists():
        print(
            f"ERROR: EfficientNet model not found: {model_path}\n"
            "Run python src/05_train_efficientnet.py first."
        )
        sys.exit(1)
    if not predictions_path.exists():
        print(
            f"ERROR: predictions.csv not found: {predictions_path}\n"
            "Run python src/06_evaluate_model.py --model-name efficientnet first."
        )
        sys.exit(1)

    try:
        test_df = load_split_dataframe(TEST_SPLIT_CSV)
    except FileNotFoundError as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    predictions = pd.read_csv(predictions_path)
    test_df = ensure_image_path_column(test_df)
    merged = predictions.merge(
        test_df[["id_code", "image_path", "processed_image_path", "processed_image_exists"]],
        left_on="image_id",
        right_on="id_code",
        how="left",
    )
    merged = merged.drop(columns=["id_code"])
    merged["gradcam_input_path"] = merged.apply(
        lambda row: row["processed_image_path"]
        if bool(row.get("processed_image_exists", False))
        else row["image_path"],
        axis=1,
    )

    selected = select_cases(merged, args.num_cases)
    print(f"Selected {len(selected)} cases for Grad-CAM.")

    # compile=False avoids needing custom training losses during inference.
    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        conv_layer = model.get_layer(args.layer_name) if args.layer_name else find_last_conv_layer(model)
        layer_name = conv_layer.name
        print(f"Using convolutional layer for Grad-CAM: {layer_name}")
    except ValueError as exc:
        print(f"ERROR:\n{exc}")
        print("Tip: run model.summary() and pass --layer-name with a Conv2D layer name.")
        sys.exit(1)

    records = []
    for _, row in selected.iterrows():
        image_path = Path(row["gradcam_input_path"])
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        image = load_image_pil(image_path, normalize=True)
        image_batch = np.expand_dims(image, axis=0)

        try:
            heatmap = make_gradcam_heatmap(
                image_batch=image_batch,
                model=model,
                last_conv_layer_name=layer_name,
                pred_index=int(row["predicted_label"]),
            )
        except ValueError as exc:
            print(f"ERROR while creating Grad-CAM:\n{exc}")
            print("Tip: try passing a different convolutional layer via --layer-name.")
            sys.exit(1)

        overlay = overlay_heatmap(image, heatmap)
        is_correct = bool(row["correct_bool"])
        status = "correct" if is_correct else "incorrect"
        filename = safe_filename(
            f"{row['image_id']}_true_{row['true_label']}_pred_{row['predicted_label']}_{status}.png"
        )
        output_path = GRADCAM_DIR / filename
        save_overlay(overlay, output_path)

        records.append(
            {
                "image_id": row["image_id"],
                "true_label": int(row["true_label"]),
                "true_class": row["true_class"],
                "predicted_label": int(row["predicted_label"]),
                "predicted_class": row["predicted_class"],
                "confidence": float(row["confidence"]),
                "correct": is_correct,
                "gradcam_path": str(output_path),
            }
        )
        print(f"Saved Grad-CAM: {output_path}")

    pd.DataFrame(records).to_csv(GRADCAM_SELECTED_CASES_CSV, index=False)
    save_combined_grid(records)

    print(f"\nSelected Grad-CAM cases saved to: {GRADCAM_SELECTED_CASES_CSV}")
    print(f"Combined Grad-CAM figure saved to: {GRADCAM_EXAMPLES_FIG}")
    print("Grad-CAM generation completed.")


if __name__ == "__main__":
    main()
