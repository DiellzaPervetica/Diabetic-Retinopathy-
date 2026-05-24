"""Run lightweight stratified cross-validation with EfficientNetB0 features.

Full deep-learning cross-validation is computationally expensive. This script
reduces the cost by extracting frozen EfficientNetB0 features once and then
evaluating a balanced Logistic Regression classifier with Stratified K-Fold.
"""

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    IMAGE_SHAPE,
    METRICS_DIR,
    RANDOM_SEED,
    TRAIN_AVAILABLE_CSV,
    ensure_directories,
)
from utils.data_utils import create_tf_dataset, load_available_train_dataframe
from utils.metrics_utils import save_json


def build_feature_extractor() -> keras.Model:
    """Build a frozen EfficientNetB0 feature extractor."""
    inputs = keras.Input(shape=IMAGE_SHAPE, name="image")
    x = layers.Rescaling(255.0, name="to_0_255_for_efficientnet")(inputs)
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    base_model.trainable = False
    return keras.Model(inputs=inputs, outputs=base_model.output, name="efficientnetb0_features")


def extract_features(df: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Extract EfficientNetB0 features for all images in a dataframe."""
    dataset = create_tf_dataset(df, batch_size=batch_size, shuffle=False)
    feature_extractor = build_feature_extractor()
    return feature_extractor.predict(dataset, verbose=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EfficientNet feature CV.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Recompute EfficientNet features even if a cached matrix exists.",
    )
    args = parser.parse_args()

    ensure_directories()
    df, missing_df = load_available_train_dataframe()
    if missing_df.empty:
        print(f"Using all {len(df)} images for cross-validation.")
    else:
        print(f"Using {len(df)} available images; ignoring {len(missing_df)} missing images.")

    df.to_csv(TRAIN_AVAILABLE_CSV, index=False)
    y = df["diagnosis"].astype(int).to_numpy()
    feature_cache = METRICS_DIR / "cross_validation_efficientnet_feature_matrix.npy"
    if feature_cache.exists() and not args.refresh_features:
        print(f"Loading cached feature matrix: {feature_cache}")
        features = np.load(feature_cache)
        if len(features) != len(df):
            print("Cached feature matrix size does not match the dataframe; recomputing.")
            features = extract_features(df, args.batch_size)
            np.save(feature_cache, features)
    else:
        features = extract_features(df, args.batch_size)
        np.save(feature_cache, features)
        print(f"Cached feature matrix saved to: {feature_cache}")

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(features, y), start=1):
        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_val)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_val, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_val, y_pred, average="weighted", zero_division=0
        )
        row = {
            "fold": fold,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "quadratic_weighted_kappa": cohen_kappa_score(y_val, y_pred, weights="quadratic"),
        }
        fold_rows.append(row)
        print(f"Fold {fold}: accuracy={row['accuracy']:.4f}, f1_macro={row['f1_macro']:.4f}, qwk={row['quadratic_weighted_kappa']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_csv = METRICS_DIR / "cross_validation_efficientnet_features.csv"
    summary_json = METRICS_DIR / "cross_validation_efficientnet_features_summary.json"
    fold_df.to_csv(fold_csv, index=False)

    summary = {
        "folds": int(args.folds),
        "num_images": int(len(df)),
        "method": "Frozen EfficientNetB0 features + balanced LogisticRegression",
    }
    for column in fold_df.columns:
        if column == "fold":
            continue
        summary[f"{column}_mean"] = float(fold_df[column].mean())
        summary[f"{column}_std"] = float(fold_df[column].std(ddof=0))
    save_json(summary, summary_json)

    print(f"Cross-validation fold metrics saved to: {fold_csv}")
    print(f"Cross-validation summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
