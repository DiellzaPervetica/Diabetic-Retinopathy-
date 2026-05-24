"""Evaluate a trained baseline CNN or EfficientNetB0 model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config import (
    BASELINE_MODEL_PATH,
    BATCH_SIZE,
    CLASSIFICATION_REPORT_CSV,
    CLASS_NAMES,
    CONFUSION_MATRIX_FIG,
    CONFUSION_MATRIX_NORMALIZED_FIG,
    EFFICIENTNET_BEST_MODEL_PATH,
    EVALUATION_METRICS_JSON,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_CSV,
    TEST_SPLIT_CSV,
    ensure_directories,
)
from utils.data_utils import create_tf_dataset, labels_to_class_names, load_split_dataframe
from utils.metrics_utils import plot_confusion_matrix, save_json


def resolve_model_path(model_name: str, model_path: str | None) -> Path:
    """Resolve selected model path."""
    if model_path:
        return Path(model_path)
    if model_name == "baseline":
        return BASELINE_MODEL_PATH
    return EFFICIENTNET_BEST_MODEL_PATH


def save_with_latest_copy(dataframe: pd.DataFrame, specific_path: Path, latest_path: Path) -> None:
    """Save a dataframe to a model-specific path and the standard latest path."""
    dataframe.to_csv(specific_path, index=False)
    if specific_path.resolve() != latest_path.resolve():
        dataframe.to_csv(latest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DR model.")
    parser.add_argument(
        "--model-name",
        choices=["efficientnet", "baseline"],
        default="efficientnet",
        help="Which default model path to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional custom .keras model path.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional prefix for model-specific output files.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    ensure_directories()
    model_path = resolve_model_path(args.model_name, args.model_path)
    output_prefix = args.output_prefix or args.model_name
    predictions_csv = METRICS_DIR / f"predictions_{output_prefix}.csv"
    metrics_json = METRICS_DIR / f"evaluation_metrics_{output_prefix}.json"
    report_csv = METRICS_DIR / f"classification_report_{output_prefix}.csv"
    confusion_matrix_fig = FIGURES_DIR / f"confusion_matrix_{output_prefix}.png"
    confusion_matrix_normalized_fig = FIGURES_DIR / f"confusion_matrix_normalized_{output_prefix}.png"
    print(f"Evaluating model: {model_path}\n")

    if not model_path.exists():
        print(
            f"ERROR: Model file not found: {model_path}\n"
            "Train the selected model first."
        )
        sys.exit(1)

    try:
        test_df = load_split_dataframe(TEST_SPLIT_CSV)
    except FileNotFoundError as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    test_ds = create_tf_dataset(test_df, batch_size=args.batch_size, shuffle=False)
    # compile=False keeps evaluation robust when a model was trained with a
    # custom loss such as focal loss. Metrics are computed manually below.
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Generating predictions...")
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_df["diagnosis"].astype(int).to_numpy()
    confidence = np.max(y_prob, axis=1)

    predictions_df = pd.DataFrame(
        {
            "image_id": test_df["id_code"].astype(str).to_numpy(),
            "true_label": y_true,
            "true_class": labels_to_class_names(y_true),
            "predicted_label": y_pred,
            "predicted_class": labels_to_class_names(y_pred),
            "confidence": confidence,
            "correct": y_true == y_pred,
        }
    )
    for label, class_name in CLASS_NAMES.items():
        safe_name = class_name.lower().replace(" ", "_")
        predictions_df[f"prob_{label}_{safe_name}"] = y_prob[:, label]

    save_with_latest_copy(predictions_df, predictions_csv, PREDICTIONS_CSV)
    print(f"Saved predictions: {predictions_csv}")

    labels = list(CLASS_NAMES.keys())
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    roc_auc_macro = None
    roc_auc_weighted = None
    roc_auc_error = None
    try:
        y_true_bin = label_binarize(y_true, classes=labels)
        roc_auc_macro = float(roc_auc_score(y_true_bin, y_prob, average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true_bin, y_prob, average="weighted"))
    except ValueError as exc:
        roc_auc_error = str(exc)
        print(f"ROC-AUC skipped: {roc_auc_error}")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_csv)
    if report_csv.resolve() != CLASSIFICATION_REPORT_CSV.resolve():
        report_df.to_csv(CLASSIFICATION_REPORT_CSV)
    print(f"Saved classification report: {report_csv}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_normalized = np.nan_to_num(cm_normalized)
    plot_confusion_matrix(cm, confusion_matrix_fig, normalize=False, title="Confusion Matrix")
    plot_confusion_matrix(
        cm_normalized,
        confusion_matrix_normalized_fig,
        normalize=True,
        title="Normalized Confusion Matrix",
    )
    plot_confusion_matrix(cm, CONFUSION_MATRIX_FIG, normalize=False, title="Confusion Matrix")
    plot_confusion_matrix(
        cm_normalized,
        CONFUSION_MATRIX_NORMALIZED_FIG,
        normalize=True,
        title="Normalized Confusion Matrix",
    )
    print(f"Saved confusion matrix: {confusion_matrix_fig}")
    print(f"Saved normalized confusion matrix: {confusion_matrix_normalized_fig}")

    metrics = {
        "model_name": args.model_name,
        "model_path": str(model_path),
        "num_test_samples": int(len(test_df)),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "quadratic_weighted_kappa": float(qwk),
        "roc_auc_macro_ovr": roc_auc_macro,
        "roc_auc_weighted_ovr": roc_auc_weighted,
        "roc_auc_note": roc_auc_error,
    }
    save_json(metrics, metrics_json)
    if metrics_json.resolve() != EVALUATION_METRICS_JSON.resolve():
        save_json(metrics, EVALUATION_METRICS_JSON)
    print(f"Saved evaluation metrics: {metrics_json}")

    print("\nEvaluation summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
