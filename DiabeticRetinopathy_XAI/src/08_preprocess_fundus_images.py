"""Create lightweight preprocessed 224x224 fundus images.

This script is optional but useful for better fundus-specific preprocessing:
- black-border crop
- CLAHE contrast enhancement
- resize to 224x224

The original images remain untouched. Preprocessed images are written to
data/processed/fundus_224/ and referenced from data/processed/train_preprocessed.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    IMAGE_SIZE,
    PROCESSED_IMAGES_DIR,
    TRAIN_PREPROCESSED_CSV,
    ensure_directories,
)
from utils.data_utils import (
    add_processed_image_path_columns,
    load_available_train_dataframe,
    processed_image_path_from_id,
)
from utils.image_utils import preprocess_fundus_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess fundus images to 224x224.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--no-crop", action="store_true", help="Disable black-border crop.")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE enhancement.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing files.")
    args = parser.parse_args()

    ensure_directories()
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df, missing_df = load_available_train_dataframe()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    if not missing_df.empty:
        print(f"WARNING: {len(missing_df)} original images are missing and will be ignored.")

    records = []
    created = 0
    skipped = 0
    failed = []

    print(f"Preprocessing {len(df)} fundus images into {PROCESSED_IMAGES_DIR}...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing fundus images"):
        image_id = str(row["id_code"])
        source_path = Path(row["image_path"])
        output_path = processed_image_path_from_id(image_id)

        try:
            if output_path.exists() and not args.overwrite:
                skipped += 1
            else:
                preprocess_fundus_image(
                    image_path=source_path,
                    output_path=output_path,
                    image_size=args.image_size,
                    crop_black_borders=not args.no_crop,
                    apply_clahe=not args.no_clahe,
                )
                created += 1

            record = row.to_dict()
            record["processed_image_path"] = str(output_path.resolve())
            record["processed_image_exists"] = output_path.exists()
            records.append(record)
        except Exception as exc:
            failed.append({"id_code": image_id, "image_path": str(source_path), "error": str(exc)})

    processed_df = pd.DataFrame(records)
    processed_df = add_processed_image_path_columns(processed_df)
    processed_df.to_csv(TRAIN_PREPROCESSED_CSV, index=False)

    if failed:
        failed_path = TRAIN_PREPROCESSED_CSV.parent / "preprocessing_failed_images.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"WARNING: {len(failed)} images failed. Details saved to {failed_path}")

    print(f"Created files: {created}")
    print(f"Skipped existing files: {skipped}")
    print(f"Preprocessed CSV saved to: {TRAIN_PREPROCESSED_CSV}")
    print("Fundus preprocessing completed.")


if __name__ == "__main__":
    main()

