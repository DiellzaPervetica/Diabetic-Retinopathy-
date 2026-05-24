"""Dataset helpers for CSV loading, path handling, and tf.data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    PROJECT_ROOT,
    PROCESSED_IMAGES_DIR,
    RANDOM_SEED,
    TRAIN_AVAILABLE_CSV,
    TRAIN_CSV,
    TRAIN_IMAGES_DIR,
    TRAIN_PREPROCESSED_CSV,
    USE_PROCESSED_IMAGES,
)


REQUIRED_TRAIN_COLUMNS = {"id_code", "diagnosis"}


def expected_dataset_message() -> str:
    """Return a clear message showing where the Kaggle files should be."""
    return (
        "Dataset not found in the expected structure.\n\n"
        "Please download the APTOS 2019 Blindness Detection dataset from Kaggle "
        "and place it here:\n\n"
        "data/raw/aptos2019/\n"
        "    train.csv\n"
        "    test.csv\n"
        "    sample_submission.csv\n"
        "    train_images/\n"
        "    test_images/\n"
    )


def validate_train_csv(csv_path: Path = TRAIN_CSV) -> None:
    """Check that train.csv exists and contains the required columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(expected_dataset_message())

    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing_columns = REQUIRED_TRAIN_COLUMNS.difference(columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )


def load_train_dataframe(csv_path: Path = TRAIN_CSV) -> pd.DataFrame:
    """Load train.csv after validating the expected dataset format."""
    validate_train_csv(csv_path)
    df = pd.read_csv(csv_path)
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df


def filter_available_images(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into rows with existing images and missing images.

    This is useful when a dataset download was interrupted. The project can
    still run on the local subset instead of trying to fetch more files.
    """
    with_paths = add_image_path_columns(df)
    exists = with_paths["image_path"].apply(lambda value: Path(value).exists())
    available_df = with_paths.loc[exists].copy().reset_index(drop=True)
    missing_df = with_paths.loc[~exists].copy().reset_index(drop=True)
    return available_df, missing_df


def save_available_train_dataframe(
    output_path: Path = TRAIN_AVAILABLE_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create data/processed/train_available.csv from local image files."""
    df = load_train_dataframe()
    available_df, missing_df = filter_available_images(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    available_df.to_csv(output_path, index=False)
    return available_df, missing_df


def load_available_train_dataframe(
    output_path: Path = TRAIN_AVAILABLE_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or create the dataframe containing only locally available images.

    The file is recomputed each time so it accurately reflects the current
    train_images/ folder, but it is also saved for transparency and reporting.
    """
    return save_available_train_dataframe(output_path)


def processed_image_path_from_id(image_id: str) -> Path:
    """Return the expected preprocessed image path for an APTOS id_code."""
    image_id = str(image_id)
    candidate = PROCESSED_IMAGES_DIR / image_id
    if candidate.suffix:
        return candidate
    return PROCESSED_IMAGES_DIR / f"{image_id}.png"


def add_processed_image_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add processed image paths if preprocessed images exist."""
    result = df.copy()
    result["processed_image_path"] = result["id_code"].apply(
        lambda image_id: str(processed_image_path_from_id(image_id).resolve())
    )
    result["processed_image_exists"] = result["processed_image_path"].apply(
        lambda value: Path(value).exists()
    )
    return result


def preferred_image_path_column(df: pd.DataFrame) -> str:
    """Choose processed images when available; otherwise use original paths."""
    if (
        USE_PROCESSED_IMAGES
        and "processed_image_path" in df.columns
        and df["processed_image_path"].apply(lambda value: Path(value).exists()).all()
    ):
        return "processed_image_path"
    return "image_path"


def image_path_from_id(image_id: str, images_dir: Path = TRAIN_IMAGES_DIR) -> Path:
    """Return the expected image path for an APTOS id_code.

    APTOS 2019 train images are PNG files named <id_code>.png.
    """
    image_id = str(image_id)
    candidate = Path(images_dir) / image_id
    if candidate.suffix:
        return candidate
    return Path(images_dir) / f"{image_id}.png"


def add_image_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and project-relative image path columns to a dataframe."""
    result = df.copy()
    result["image_path"] = result["id_code"].apply(
        lambda image_id: str(image_path_from_id(image_id).resolve())
    )
    result["relative_image_path"] = result["id_code"].apply(
        lambda image_id: str(image_path_from_id(image_id).resolve().relative_to(PROJECT_ROOT))
    )
    return result


def ensure_image_path_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a dataframe has usable image paths.

    Split CSVs include an image_path column. If a project folder is moved after
    the split files were created, old absolute paths may become stale. In that
    case this function rebuilds paths from id_code.
    """
    result = df.copy()
    if "image_path" not in result.columns:
        return add_image_path_columns(result)

    fixed_paths = []
    for _, row in result.iterrows():
        path = Path(str(row["image_path"]))
        if path.exists():
            fixed_paths.append(str(path))
        else:
            fixed_paths.append(str(image_path_from_id(row["id_code"]).resolve()))
    result["image_path"] = fixed_paths
    return result


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    """Load a saved split CSV and make sure image paths are available."""
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}\n"
            "Run python src/03_create_splits.py first."
        )
    df = pd.read_csv(split_csv)
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = ensure_image_path_column(df)
    return add_processed_image_path_columns(df)


def check_missing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose image files do not exist."""
    with_paths = ensure_image_path_column(df)
    exists = with_paths["image_path"].apply(lambda value: Path(value).exists())
    return with_paths.loc[~exists].copy()


def class_distribution_dataframe(
    labels: Iterable[int], include_percent: bool = True
) -> pd.DataFrame:
    """Return class counts and percentages for labels."""
    series = pd.Series(labels, name="diagnosis").astype(int)
    counts = series.value_counts().sort_index()
    rows = []
    total = int(counts.sum())
    for label in sorted(CLASS_NAMES):
        count = int(counts.get(label, 0))
        row = {
            "diagnosis": label,
            "class_name": CLASS_NAMES[label],
            "count": count,
        }
        if include_percent:
            row["percentage"] = (count / total * 100.0) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    shuffle: bool = False,
    seed: int = RANDOM_SEED,
):
    """Create a memory-efficient tf.data dataset from a dataframe.

    The dataframe must contain id_code and diagnosis. If image_path is missing,
    it is created from id_code.
    """
    import tensorflow as tf

    from utils.image_utils import load_and_preprocess_image_tf

    df = add_processed_image_path_columns(ensure_image_path_column(df))
    path_column = preferred_image_path_column(df)
    paths = df[path_column].astype(str).to_numpy()
    labels = df["diagnosis"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        buffer_size = min(len(df), 1024)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image_tf(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_balanced_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
):
    """Create a class-balanced tf.data dataset using equal class sampling.

    This improves class imbalance handling by giving minority classes a higher
    chance to appear in each epoch without loading all images into memory.
    """
    import tensorflow as tf

    from utils.image_utils import load_and_preprocess_image_tf

    df = add_processed_image_path_columns(ensure_image_path_column(df))
    path_column = preferred_image_path_column(df)
    class_datasets = []
    for label in sorted(CLASS_NAMES):
        class_df = df[df["diagnosis"] == label]
        if class_df.empty:
            continue
        paths = class_df[path_column].astype(str).to_numpy()
        labels = class_df["diagnosis"].astype("int32").to_numpy()
        class_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        class_ds = class_ds.shuffle(
            buffer_size=min(len(class_df), 512),
            seed=seed + label,
            reshuffle_each_iteration=True,
        )
        class_ds = class_ds.repeat()
        class_datasets.append(class_ds)

    dataset = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / len(class_datasets)] * len(class_datasets),
        seed=seed,
    )
    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image_tf(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def labels_to_class_names(labels: Iterable[int]) -> list[str]:
    """Convert integer labels to class names."""
    return [CLASS_NAMES[int(label)] for label in labels]


def dataframe_memory_note(df: Optional[pd.DataFrame] = None) -> str:
    """Small helper message for scripts that use generators/tf.data."""
    count = f" ({len(df)} rows)" if df is not None else ""
    return (
        f"Using tf.data for image loading{count}; images are decoded batch by "
        "batch, not loaded fully into memory."
    )
