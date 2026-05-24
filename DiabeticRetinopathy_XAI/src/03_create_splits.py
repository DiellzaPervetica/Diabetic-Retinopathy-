"""Create reproducible stratified train/validation/test splits."""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED,
    SPLITS_DIR,
    TEST_RATIO,
    TEST_SPLIT_CSV,
    TRAIN_RATIO,
    TRAIN_SPLIT_CSV,
    VAL_RATIO,
    VAL_SPLIT_CSV,
    METRICS_DIR,
    ensure_directories,
)
from utils.data_utils import class_distribution_dataframe, load_available_train_dataframe
from utils.data_utils import add_processed_image_path_columns


SPLIT_SUMMARY_CSV = METRICS_DIR / "split_summary.csv"


def validate_ratios() -> None:
    """Validate configured split ratios."""
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1. "
            f"Current total: {total}"
        )


def add_split_summary_rows(split_name: str, split_df: pd.DataFrame) -> pd.DataFrame:
    """Return class distribution rows for one split."""
    distribution = class_distribution_dataframe(split_df["diagnosis"])
    distribution.insert(0, "split", split_name)
    distribution.insert(1, "split_total", len(split_df))
    return distribution


def main() -> None:
    ensure_directories()
    validate_ratios()
    print("Creating stratified train/validation/test splits...\n")

    try:
        df, missing_df = load_available_train_dataframe()
        df = add_processed_image_path_columns(df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    if missing_df.empty:
        print(f"Using all {len(df)} rows because every image exists locally.")
    else:
        print(
            f"Using {len(df)} rows with existing images. "
            f"Ignoring {len(missing_df)} rows whose images are missing."
        )

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=VAL_RATIO + TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=df["diagnosis"],
        )

        val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1.0 - val_fraction_of_temp,
            random_state=RANDOM_SEED,
            stratify=temp_df["diagnosis"],
        )
    except ValueError as exc:
        print(
            "ERROR: Stratified splitting failed. This usually happens when one "
            "class has too few examples to appear in every split.\n"
            f"Original error: {exc}"
        )
        sys.exit(1)

    train_df = train_df.sort_values("id_code").reset_index(drop=True)
    val_df = val_df.sort_values("id_code").reset_index(drop=True)
    test_df = test_df.sort_values("id_code").reset_index(drop=True)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)
    test_df.to_csv(TEST_SPLIT_CSV, index=False)

    print(f"Saved train split: {TRAIN_SPLIT_CSV} ({len(train_df)} rows)")
    print(f"Saved validation split: {VAL_SPLIT_CSV} ({len(val_df)} rows)")
    print(f"Saved test split: {TEST_SPLIT_CSV} ({len(test_df)} rows)")

    summary = pd.concat(
        [
            add_split_summary_rows("train", train_df),
            add_split_summary_rows("validation", val_df),
            add_split_summary_rows("test", test_df),
        ],
        ignore_index=True,
    )
    summary.to_csv(SPLIT_SUMMARY_CSV, index=False)

    print("\nSplit class distributions:")
    print(summary.round({"percentage": 2}).to_string(index=False))
    print(f"\nSaved split summary: {SPLIT_SUMMARY_CSV}")
    print("Splitting completed.")


if __name__ == "__main__":
    main()
