"""Check that the APTOS 2019 dataset is placed correctly."""

import sys
from pathlib import Path

import pandas as pd

from config import (
    METRICS_DIR,
    MISSING_IMAGES_CSV,
    TRAIN_AVAILABLE_CSV,
    TRAIN_CSV,
    TRAIN_IMAGES_DIR,
    ensure_directories,
)
from utils.data_utils import (
    class_distribution_dataframe,
    expected_dataset_message,
    load_available_train_dataframe,
    load_train_dataframe,
)
from utils.image_utils import read_image_dimensions


SUMMARY_PATH = METRICS_DIR / "dataset_check_summary.csv"


def save_summary(rows: list[dict[str, str]]) -> None:
    """Save dataset check summary rows to CSV."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)


def main() -> None:
    ensure_directories()
    print("Checking APTOS 2019 dataset placement...\n")

    try:
        if not TRAIN_IMAGES_DIR.exists():
            raise FileNotFoundError(expected_dataset_message())
        df = load_train_dataframe(TRAIN_CSV)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    print("First rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(list(df.columns))

    distribution = class_distribution_dataframe(df["diagnosis"])
    print("\nOriginal class distribution from train.csv:")
    print(distribution[["diagnosis", "class_name", "count"]].to_string(index=False))

    print("\nClass percentages:")
    print(
        distribution[["diagnosis", "class_name", "percentage"]]
        .round({"percentage": 2})
        .to_string(index=False)
    )

    available_df, missing_images = load_available_train_dataframe()
    available_distribution = class_distribution_dataframe(available_df["diagnosis"])

    print("\nImage existence check:")
    if missing_images.empty:
        print("All image files referenced by train.csv were found.")
        missing_images.to_csv(MISSING_IMAGES_CSV, index=False)
        print(f"Empty missing image list saved to: {MISSING_IMAGES_CSV}")
    else:
        print(f"Missing images found: {len(missing_images)}")
        print(missing_images[["id_code", "diagnosis", "image_path"]].head(20))
        missing_images.to_csv(MISSING_IMAGES_CSV, index=False)
        print(f"Missing image list saved to: {MISSING_IMAGES_CSV}")

    print(f"\nAvailable-image CSV saved to: {TRAIN_AVAILABLE_CSV}")
    print(f"Rows usable for experiments: {len(available_df)}")
    print("\nAvailable-image class distribution:")
    print(
        available_distribution[["diagnosis", "class_name", "count", "percentage"]]
        .round({"percentage": 2})
        .to_string(index=False)
    )

    print("\nDimensions of the first available images:")
    dimension_rows = []
    available_paths = [
        Path(path)
        for path in available_df["image_path"].head(10)
        if Path(path).exists()
    ]
    for path in available_paths[:5]:
        width, height = read_image_dimensions(path)
        dimension_rows.append({"image": path.name, "width": width, "height": height})
        print(f"  {path.name}: width={width}, height={height}")

    summary_rows = [
        {"metric": "train_csv_path", "value": str(TRAIN_CSV)},
        {"metric": "train_images_dir", "value": str(TRAIN_IMAGES_DIR)},
        {"metric": "num_rows", "value": str(len(df))},
        {"metric": "available_rows", "value": str(len(available_df))},
        {"metric": "num_columns", "value": str(df.shape[1])},
        {"metric": "missing_images", "value": str(len(missing_images))},
    ]

    for _, row in distribution.iterrows():
        summary_rows.append(
            {
                "metric": f"class_{int(row['diagnosis'])}_count_{row['class_name']}",
                "value": str(int(row["count"])),
            }
        )
        summary_rows.append(
            {
                "metric": f"class_{int(row['diagnosis'])}_percent_{row['class_name']}",
                "value": f"{row['percentage']:.2f}",
            }
        )

    for _, row in available_distribution.iterrows():
        summary_rows.append(
            {
                "metric": f"available_class_{int(row['diagnosis'])}_count_{row['class_name']}",
                "value": str(int(row["count"])),
            }
        )
        summary_rows.append(
            {
                "metric": f"available_class_{int(row['diagnosis'])}_percent_{row['class_name']}",
                "value": f"{row['percentage']:.2f}",
            }
        )

    for item in dimension_rows:
        summary_rows.append(
            {
                "metric": f"sample_dimensions_{item['image']}",
                "value": f"{item['width']}x{item['height']}",
            }
        )

    save_summary(summary_rows)
    print(f"\nDataset check summary saved to: {SUMMARY_PATH}")
    print("\nDataset check completed.")


if __name__ == "__main__":
    main()
