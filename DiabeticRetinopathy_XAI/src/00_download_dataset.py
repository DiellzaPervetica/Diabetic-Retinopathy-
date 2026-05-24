"""Download the APTOS 2019 dataset into the expected local structure.

Primary option:
    Kaggle competition download, if the user has Kaggle CLI credentials.

Fallback option:
    Public Hugging Face mirror:
    https://huggingface.co/datasets/RohanAi/Aptos-blindness-detection

The project only needs train.csv and train_images/ because Kaggle test images do
not include public ground-truth labels. The Hugging Face fallback therefore
downloads only the labeled training images by default to save disk space.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import RAW_DATA_DIR, TRAIN_CSV, TRAIN_IMAGES_DIR, ensure_directories


HF_DATASET_BASE = "https://huggingface.co/datasets/RohanAi/Aptos-blindness-detection/resolve/main"
KAGGLE_COMPETITION = "aptos2019-blindness-detection"


def kaggle_credentials_available() -> bool:
    """Return True if ~/.kaggle/kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def kaggle_cli_available() -> bool:
    """Return True if the kaggle command is available."""
    return shutil.which("kaggle") is not None


def run_kaggle_download() -> bool:
    """Try downloading the dataset with Kaggle CLI."""
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

    return TRAIN_CSV.exists() and TRAIN_IMAGES_DIR.exists()


def download_file(url: str, output_path: Path, retries: int = 3) -> bool:
    """Download one file with simple retry logic."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return False

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    if temporary_path.exists():
        temporary_path.unlink()

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with temporary_path.open("wb") as file:
                    shutil.copyfileobj(response, file, length=1024 * 1024)
            temporary_path.replace(output_path)
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            print(f"Download failed on attempt {attempt}/{retries}: {url}")
            print(f"Reason: {exc}")
            if attempt < retries:
                time.sleep(3)
            else:
                raise
    return False


def download_huggingface_train_only(max_images: int | None = None) -> None:
    """Download train.csv and labeled train images from the public HF mirror."""
    print("Using public Hugging Face mirror for APTOS 2019.")
    print("Source: RohanAi/Aptos-blindness-detection")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    download_file(f"{HF_DATASET_BASE}/train.csv", TRAIN_CSV)

    test_csv_path = RAW_DATA_DIR / "test.csv"
    sample_submission_path = RAW_DATA_DIR / "sample_submission.csv"
    try:
        download_file(f"{HF_DATASET_BASE}/test.csv", test_csv_path)
    except Exception:
        print("Could not download test.csv from the mirror; continuing because it is not used.")

    if test_csv_path.exists() and not sample_submission_path.exists():
        test_df = pd.read_csv(test_csv_path)
        if "id_code" in test_df.columns:
            pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": 0}).to_csv(
                sample_submission_path,
                index=False,
            )

    train_df = pd.read_csv(TRAIN_CSV)
    if max_images is not None:
        train_df = train_df.head(max_images)
        print(f"Downloading only the first {len(train_df)} images because --max-images was set.")

    print(f"Images to download/check: {len(train_df)}")
    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for image_id in tqdm(train_df["id_code"].astype(str), desc="Downloading train images"):
        filename = f"{image_id}.png"
        output_path = TRAIN_IMAGES_DIR / filename
        url = f"{HF_DATASET_BASE}/train_images/{filename}"
        try:
            did_download = download_file(url, output_path)
            if did_download:
                downloaded += 1
            else:
                skipped += 1
        except Exception:
            failed.append(image_id)

    if failed:
        failed_path = RAW_DATA_DIR / "failed_hf_downloads.txt"
        failed_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"WARNING: {len(failed)} images failed. List saved to {failed_path}")

    print(f"Downloaded new images: {downloaded}")
    print(f"Skipped existing images: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download APTOS 2019 dataset.")
    parser.add_argument(
        "--source",
        choices=["auto", "kaggle", "huggingface"],
        default="auto",
        help="Dataset source. auto tries Kaggle first, then Hugging Face.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional small subset for testing the pipeline.",
    )
    args = parser.parse_args()

    ensure_directories()

    if args.source in ["auto", "kaggle"]:
        if run_kaggle_download():
            print("Kaggle dataset download completed.")
            return
        if args.source == "kaggle":
            print("Kaggle source was requested but could not be used.")
            sys.exit(1)

    download_huggingface_train_only(max_images=args.max_images)
    print("Dataset download step completed.")


if __name__ == "__main__":
    main()

