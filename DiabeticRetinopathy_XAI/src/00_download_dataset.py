"""Download the APTOS 2019 dataset from the official Kaggle competition.

The project only needs ``train.csv`` and ``train_images/`` because Kaggle test
images do not include public ground-truth labels. If Kaggle credentials are not
available, place the dataset manually under ``data/raw/aptos2019/``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from config import RAW_DATA_DIR, TRAIN_CSV, TRAIN_IMAGES_DIR, ensure_directories


KAGGLE_COMPETITION = "aptos2019-blindness-detection"


def kaggle_credentials_available() -> bool:
    """Return True if ~/.kaggle/kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def kaggle_cli_available() -> bool:
    """Return True if the kaggle command is available."""
    return shutil.which("kaggle") is not None


def expected_dataset_present() -> bool:
    """Return True when the minimum required APTOS files are already present."""
    return TRAIN_CSV.exists() and TRAIN_IMAGES_DIR.exists()


def run_kaggle_download(keep_zip: bool = False) -> bool:
    """Download and extract the dataset with Kaggle CLI."""
    if not kaggle_cli_available():
        print("Kaggle CLI is not installed or not on PATH.")
        return False
    if not kaggle_credentials_available():
        print("Kaggle credentials were not found at ~/.kaggle/kaggle.json.")
        return False

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / f"{KAGGLE_COMPETITION}.zip"
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        KAGGLE_COMPETITION,
        "-p",
        str(RAW_DATA_DIR),
    ]
    print("Trying Kaggle download...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("Kaggle download failed. You may need to accept the competition rules on Kaggle.")
        return False

    if zip_path.exists():
        print("Extracting Kaggle zip file...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(RAW_DATA_DIR)
        if not keep_zip:
            zip_path.unlink()

    return expected_dataset_present()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download APTOS 2019 from Kaggle.")
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded Kaggle zip after extraction.",
    )
    args = parser.parse_args()

    ensure_directories()

    if expected_dataset_present():
        print("Required APTOS 2019 files are already present.")
        return

    if run_kaggle_download(keep_zip=args.keep_zip):
        print("Kaggle dataset download completed.")
        return

    print(
        "Dataset download could not be completed. Install Kaggle CLI, configure "
        "~/.kaggle/kaggle.json, accept the competition rules, or place the dataset "
        "manually under data/raw/aptos2019/."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

