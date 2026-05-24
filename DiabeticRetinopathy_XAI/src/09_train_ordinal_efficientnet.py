"""Train an ordinal EfficientNetB0 model for DR grading.

Instead of predicting one of five unrelated classes, this model predicts four
cumulative thresholds:

    diagnosis > 0, diagnosis > 1, diagnosis > 2, diagnosis > 3

The final class is obtained by counting how many thresholds are predicted as
true. This reflects the ordered nature of diabetic retinopathy grades.
"""

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
    precision_recall_fscore_support,
)
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SHAPE,
    NUM_CLASSES,
    ORDINAL_CURVES_FIG,
    ORDINAL_EFFICIENTNET_MODEL_PATH,
    ORDINAL_HISTORY_CSV,
    TEST_SPLIT_CSV,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    METRICS_DIR,
    ensure_directories,
)
from utils.data_utils import create_balanced_tf_dataset, create_tf_dataset, load_split_dataframe
from utils.metrics_utils import save_json
from utils.model_utils import (
    build_data_augmentation,
    build_callbacks,
    history_to_dataframe,
    plot_training_curves,
    save_history,
    set_global_determinism,
)


def to_ordinal_targets(labels: tf.Tensor) -> tf.Tensor:
    """Convert integer labels to cumulative ordinal targets."""
    labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.int32)
    thresholds = tf.range(NUM_CLASSES - 1, dtype=tf.int32)
    return tf.cast(labels > thresholds, tf.float32)


def ordinalize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Map sparse labels to ordinal threshold vectors."""
    return dataset.map(
        lambda image, label: (image, to_ordinal_targets(label)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def build_ordinal_model(
    weights: str | None = "imagenet",
    use_augmentation: bool = True,
) -> keras.Model:
    """Build an EfficientNetB0 model with an ordinal sigmoid head."""
    inputs = keras.Input(shape=IMAGE_SHAPE, name="image")
    x = build_data_augmentation()(inputs) if use_augmentation else inputs
    x = layers.Rescaling(255.0, name="to_0_255_for_efficientnet")(x)
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(NUM_CLASSES - 1, activation="sigmoid", name="ordinal_thresholds")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="ordinal_efficientnetb0_dr")


def ordinal_probabilities_to_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert threshold probabilities to class labels 0-4."""
    return (probabilities >= threshold).sum(axis=1).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ordinal EfficientNetB0.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use equal class sampling while training the ordinal model.",
    )
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    ensure_directories()
    set_global_determinism()

    try:
        train_df = load_split_dataframe(TRAIN_SPLIT_CSV)
        val_df = load_split_dataframe(VAL_SPLIT_CSV)
        test_df = load_split_dataframe(TEST_SPLIT_CSV)
    except FileNotFoundError as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    if args.balanced_sampling:
        train_base_ds = create_balanced_tf_dataset(train_df, batch_size=args.batch_size)
        steps_per_epoch = max(1, len(train_df) // args.batch_size)
        print("Using balanced class sampling for ordinal training batches.")
    else:
        train_base_ds = create_tf_dataset(train_df, batch_size=args.batch_size, shuffle=True)
        steps_per_epoch = None

    train_ds = ordinalize_dataset(train_base_ds)
    val_ds = ordinalize_dataset(
        create_tf_dataset(val_df, batch_size=args.batch_size, shuffle=False)
    )
    test_ds_plain = create_tf_dataset(test_df, batch_size=args.batch_size, shuffle=False)

    weights = None if args.weights == "none" else "imagenet"
    model = build_ordinal_model(weights=weights, use_augmentation=not args.no_augmentation)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    model.summary()

    callbacks = build_callbacks(ORDINAL_EFFICIENTNET_MODEL_PATH, patience=args.patience)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = history_to_dataframe(history, phase="ordinal")
    save_history(history_df, ORDINAL_HISTORY_CSV)
    plot_training_curves(history_df, ORDINAL_CURVES_FIG, title="Ordinal EfficientNetB0 Training")

    print("Evaluating ordinal model on the test split...")
    probabilities = model.predict(test_ds_plain, verbose=1)
    y_pred = ordinal_probabilities_to_labels(probabilities)
    y_true = test_df["diagnosis"].astype(int).to_numpy()

    labels = list(CLASS_NAMES.keys())
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    metrics = {
        "model_name": "ordinal_efficientnet",
        "num_test_samples": int(len(test_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }

    output_metrics = METRICS_DIR / "evaluation_metrics_ordinal_efficientnet.json"
    output_report = METRICS_DIR / "classification_report_ordinal_efficientnet.csv"
    output_predictions = METRICS_DIR / "predictions_ordinal_efficientnet.csv"

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_report)
    save_json(metrics, output_metrics)

    predictions_df = pd.DataFrame(
        {
            "image_id": test_df["id_code"].astype(str),
            "true_label": y_true,
            "predicted_label": y_pred,
            "correct": y_true == y_pred,
        }
    )
    for index in range(NUM_CLASSES - 1):
        predictions_df[f"prob_diagnosis_gt_{index}"] = probabilities[:, index]
    predictions_df.to_csv(output_predictions, index=False)

    print(f"Ordinal model saved to: {ORDINAL_EFFICIENTNET_MODEL_PATH}")
    print(f"Ordinal metrics saved to: {output_metrics}")
    print("Ordinal EfficientNetB0 training completed.")


if __name__ == "__main__":
    main()
