"""Central configuration for the diabetic retinopathy project.

All paths are built with pathlib and are relative to the project folder.
This keeps the project portable across Windows, macOS, and Linux.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "aptos2019"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "fundus_224"

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"
TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"
TEST_IMAGES_DIR = RAW_DATA_DIR / "test_images"

TRAIN_AVAILABLE_CSV = PROCESSED_DIR / "train_available.csv"
TRAIN_PREPROCESSED_CSV = PROCESSED_DIR / "train_preprocessed.csv"

TRAIN_SPLIT_CSV = SPLITS_DIR / "train_split.csv"
VAL_SPLIT_CSV = SPLITS_DIR / "val_split.csv"
TEST_SPLIT_CSV = SPLITS_DIR / "test_split.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
MISSING_IMAGES_CSV = METRICS_DIR / "missing_images.csv"


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
NUM_CLASSES = 5
BATCH_SIZE = 16
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASELINE_EPOCHS = 10
EFFICIENTNET_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
USE_FINE_TUNING = False
USE_PROCESSED_IMAGES = True

LEARNING_RATE = 1e-3
TRANSFER_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_cnn.keras"
EFFICIENTNET_BEST_MODEL_PATH = MODELS_DIR / "efficientnetb0_best.keras"
EFFICIENTNET_FINAL_MODEL_PATH = MODELS_DIR / "efficientnetb0_final.keras"
ORDINAL_EFFICIENTNET_MODEL_PATH = MODELS_DIR / "ordinal_efficientnetb0.keras"

BASELINE_HISTORY_CSV = METRICS_DIR / "baseline_cnn_history.csv"
EFFICIENTNET_HISTORY_CSV = METRICS_DIR / "efficientnetb0_history.csv"
ORDINAL_HISTORY_CSV = METRICS_DIR / "ordinal_efficientnetb0_history.csv"

BASELINE_CURVES_FIG = FIGURES_DIR / "baseline_cnn_training_curves.png"
EFFICIENTNET_CURVES_FIG = FIGURES_DIR / "efficientnetb0_training_curves.png"
ORDINAL_CURVES_FIG = FIGURES_DIR / "ordinal_efficientnetb0_training_curves.png"

PREDICTIONS_CSV = METRICS_DIR / "predictions.csv"
EVALUATION_METRICS_JSON = METRICS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_CSV = METRICS_DIR / "classification_report.csv"
CONFUSION_MATRIX_FIG = FIGURES_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORMALIZED_FIG = FIGURES_DIR / "confusion_matrix_normalized.png"

GRADCAM_SELECTED_CASES_CSV = METRICS_DIR / "gradcam_selected_cases.csv"
GRADCAM_EXAMPLES_FIG = FIGURES_DIR / "gradcam_examples.png"


def ensure_directories() -> None:
    """Create project directories that scripts need for outputs."""
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DIR,
        PROCESSED_IMAGES_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        GRADCAM_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def label_to_name(label: int) -> str:
    """Return the human-readable class name for an integer label."""
    return CLASS_NAMES.get(int(label), f"Unknown ({label})")
