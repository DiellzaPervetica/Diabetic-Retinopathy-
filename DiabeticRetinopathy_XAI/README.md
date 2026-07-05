# DiabeticRetinopathy_XAI

IEEE-style reproducibility artifact for five-class diabetic retinopathy grading on APTOS 2019 fundus images.

Associated manuscript:

**Eyes as Windows to Health: Explainable EfficientNetB0 for Lightweight Five-Class Diabetic Retinopathy Grading in Fundus Images with Clinical Interpretability Support**

Authors: Diellza Pervetica and Fatjeta Gashi, Department of Computer and Software Engineering, University of Prishtina "Hasan Prishtina", Prishtina, Kosovo.

Repository link preserved for citation and paper references:

https://github.com/DiellzaPervetica/Diabetic-Retinopathy-/tree/main/DiabeticRetinopathy_XAI

## Artifact Scope

This repository is a research artifact, not a clinical diagnostic system. It provides a transparent and reproducible workflow for:

- verifying the APTOS 2019 image-label files;
- running exploratory data analysis;
- preprocessing fundus images with black-border handling, local contrast enhancement, and resizing;
- creating a stratified train/validation/test split;
- training a baseline CNN;
- training an EfficientNetB0 transfer-learning classifier;
- training an ordinal EfficientNetB0 variant;
- evaluating global, class-wise, and ordinal metrics;
- producing Grad-CAM visual explanations.

The artifact is designed for auditability and reproducibility under limited computing resources. It should not be used for medical decision-making without independent clinical validation.

## Preservation Note

Do not delete local generated results if they exist on your machine. This repository intentionally ignores large artifacts such as:

- `data/raw/aptos2019/train_images/`
- `data/processed/fundus_224/`
- `data/splits/*.csv`
- `models/*.keras`
- `outputs/`

The committed paper-facing results are preserved in `reports/figures/` and the Markdown summaries in `reports/`. The commands below document how to reproduce the pipeline, but they do not need to be rerun just to inspect the repository.

## Repository Layout

```text
DiabeticRetinopathy_XAI/
|-- CITATION.cff
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/aptos2019/.gitkeep
|   |-- processed/.gitkeep
|   `-- splits/.gitkeep
|-- docs/
|   `-- ARTIFACTS.md
|-- models/
|   `-- .gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- reports/
|   |-- README.md
|   `-- figures/
|-- src/
|   |-- 00_download_dataset.py
|   |-- 01_check_dataset.py
|   |-- 02_eda.py
|   |-- 03_create_splits.py
|   |-- 04_train_baseline_cnn.py
|   |-- 05_train_efficientnet.py
|   |-- 06_evaluate_model.py
|   |-- 07_gradcam_xai.py
|   |-- 08_preprocess_fundus_images.py
|   |-- 09_train_ordinal_efficientnet.py
|   |-- 10_cross_validation.py
|   |-- config.py
|   `-- utils/
```

## Data Availability

Dataset: **APTOS 2019 Blindness Detection**

Public source: https://www.kaggle.com/competitions/aptos2019-blindness-detection

The raw dataset is not redistributed in this repository. To reproduce the experiments, place the labeled training data in:

```text
data/raw/aptos2019/
|-- train.csv
`-- train_images/
```

The official Kaggle test images do not include public labels, so this study uses the labeled training subset and creates a stratified 70/15/15 train/validation/test split.

Class mapping:

| Label | Class |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Class distribution in the labeled subset:

| Class | Images | Percentage |
|---|---:|---:|
| No DR | 1805 | 49.3% |
| Mild | 370 | 10.1% |
| Moderate | 999 | 27.3% |
| Severe | 193 | 5.3% |
| Proliferative DR | 295 | 8.1% |

## Environment

Recommended local environment:

- Python 3.10, 3.11, or compatible TensorFlow-supported Python version
- 16 GB RAM recommended
- GPU optional
- Windows, Linux, or macOS

Install:

```powershell
cd DiabeticRetinopathy_XAI
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file lists the runtime packages used by the scripts. For a formal archival release, freeze the exact execution environment from the machine used for the final run:

```powershell
pip freeze > requirements-lock.txt
```

Do this only when you intentionally create a release snapshot.

## Reproduction Workflow

The pipeline writes generated files to `data/processed/`, `data/splits/`, `models/`, and `outputs/`. These folders are intentionally ignored by git because they can contain large files.

Dataset verification and EDA:

```powershell
python src/01_check_dataset.py
python src/02_eda.py --dimension-sample 500
```

Fundus preprocessing and split creation:

```powershell
python src/08_preprocess_fundus_images.py
python src/03_create_splits.py
```

Baseline CNN:

```powershell
python src/04_train_baseline_cnn.py --epochs 10 --batch-size 16
python src/06_evaluate_model.py --model-name baseline --batch-size 16 --output-prefix baseline
```

EfficientNetB0:

```powershell
python src/05_train_efficientnet.py --epochs 10 --batch-size 8 --fine-tune --fine-tune-epochs 5 --fine-tune-layers 20 --balanced-sampling --focal-loss --patience 4
python src/06_evaluate_model.py --model-name efficientnet --batch-size 8 --output-prefix efficientnet
```

Ordinal EfficientNetB0:

```powershell
python src/09_train_ordinal_efficientnet.py --epochs 10 --batch-size 8 --balanced-sampling --patience 4
```

Lightweight representation-level cross-validation:

```powershell
python src/10_cross_validation.py --folds 3 --batch-size 8
```

Grad-CAM:

```powershell
python src/07_gradcam_xai.py --num-cases 8 --predictions-csv outputs/metrics/predictions_efficientnet.csv
```

## Results Reported in the Manuscript

Test-set metrics from the manuscript:

| Model | Accuracy | Macro F1 | Weighted F1 | QWK | AUC macro |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.5855 | 0.3608 | 0.5766 | 0.5748 | 0.8337 |
| EfficientNetB0 | 0.7709 | 0.5652 | 0.7597 | 0.8796 | 0.9184 |
| Ordinal EfficientNetB0 | 0.7418 | 0.5553 | 0.7394 | 0.8540 | 0.9271 |

Representation-level cross-validation:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.7884 | 0.0067 |
| Macro F1 | 0.6197 | 0.0073 |
| QWK | 0.8505 | 0.0187 |

Class-wise report for EfficientNetB0:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.9636 | 0.9779 | 0.9707 | 271 |
| Mild | 0.4722 | 0.3036 | 0.3696 | 56 |
| Moderate | 0.6706 | 0.7600 | 0.7125 | 150 |
| Severe | 0.3600 | 0.6207 | 0.4557 | 29 |
| Proliferative DR | 0.5263 | 0.2273 | 0.3175 | 44 |

The manuscript identifies EfficientNetB0 as the strongest overall model by accuracy, macro F1, weighted F1, and QWK. The ordinal model has the highest macro AUC, showing that ordinal framing is still informative for diabetic retinopathy severity grading.

## Committed Figures

The following paper-facing figures are committed for inspection without rerunning the pipeline:

- [Class distribution](reports/figures/class_distribution.png)
- [Class distribution percent](reports/figures/class_distribution_percent.png)
- [Sample images by class](reports/figures/sample_images_by_class.png)
- [Baseline CNN training curves](reports/figures/baseline_cnn_training_curves.png)
- [EfficientNetB0 training curves](reports/figures/efficientnetb0_training_curves.png)
- [Ordinal EfficientNetB0 training curves](reports/figures/ordinal_efficientnetb0_training_curves.png)
- [EfficientNetB0 confusion matrix](reports/figures/confusion_matrix_efficientnet_enhanced.png)
- [EfficientNetB0 normalized confusion matrix](reports/figures/confusion_matrix_normalized_efficientnet_enhanced.png)
- [Grad-CAM examples](reports/figures/gradcam_examples.png)

## Script Index

| Script | Purpose |
|---|---|
| `src/00_download_dataset.py` | Optional official Kaggle dataset download helper. |
| `src/01_check_dataset.py` | Verifies labels, image paths, missing files, and dataset summary. |
| `src/02_eda.py` | Generates class distribution plots and sample-image summaries. |
| `src/03_create_splits.py` | Creates stratified train/validation/test CSV files. |
| `src/04_train_baseline_cnn.py` | Trains the baseline CNN. |
| `src/05_train_efficientnet.py` | Trains EfficientNetB0 with transfer learning, optional fine-tuning, balanced sampling, and focal loss. |
| `src/06_evaluate_model.py` | Produces predictions, classification reports, confusion matrices, and metric JSON files. |
| `src/07_gradcam_xai.py` | Generates Grad-CAM overlays and selected-case summaries. |
| `src/08_preprocess_fundus_images.py` | Produces 224 x 224 processed fundus images. |
| `src/09_train_ordinal_efficientnet.py` | Trains the ordinal EfficientNetB0 variant. |
| `src/10_cross_validation.py` | Runs frozen EfficientNetB0 feature extraction with balanced Logistic Regression cross-validation. |

## Reproducibility Controls

- Random seed: `42`
- Split strategy: stratified 70/15/15
- Input size: 224 x 224 RGB
- Main architecture: EfficientNetB0
- Evaluation focus: accuracy, macro F1, weighted F1, QWK, macro AUC, class-wise precision/recall/F1, confusion matrix, Grad-CAM
- Data leakage control: the Kaggle public test set is not used because labels are unavailable; the held-out test subset comes from the labeled training data after stratified splitting

## Limitations

- Results are based only on APTOS 2019 and do not prove generalization to other cameras, clinics, populations, or acquisition settings.
- Minority classes, especially Severe and Proliferative DR, remain difficult because of class imbalance and visual similarity between adjacent stages.
- Images are standardized to 224 x 224 for computational efficiency, which may discard small lesion-level details.
- Grad-CAM is an interpretability aid, not clinical validation.

## Citation

Until a DOI or final publication metadata is assigned, cite the associated manuscript and this repository together:

```bibtex
@misc{pervetica_gashi_dr_xai_artifact,
  title = {Eyes as Windows to Health: Explainable EfficientNetB0 for Lightweight Five-Class Diabetic Retinopathy Grading in Fundus Images with Clinical Interpretability Support},
  author = {Pervetica, Diellza and Gashi, Fatjeta},
  note = {Code artifact: https://github.com/DiellzaPervetica/Diabetic-Retinopathy-/tree/main/DiabeticRetinopathy_XAI}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

No explicit open-source license is currently declared in this artifact. Unless a license is added by the authors, reuse is limited by default copyright rules and by the APTOS/Kaggle dataset terms.

## Contact

For questions about the artifact or manuscript, contact the repository owner or the authors listed in the associated paper.
