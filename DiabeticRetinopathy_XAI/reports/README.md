# Reports

This folder contains paper-facing figures and concise artifact documentation for the diabetic retinopathy grading study.

## Preserved Figures

| Figure | File |
|---|---|
| Class distribution | `figures/class_distribution.png` |
| Class distribution percentage | `figures/class_distribution_percent.png` |
| Representative fundus images | `figures/sample_images_by_class.png` |
| Baseline CNN training curves | `figures/baseline_cnn_training_curves.png` |
| EfficientNetB0 training curves | `figures/efficientnetb0_training_curves.png` |
| Ordinal EfficientNetB0 training curves | `figures/ordinal_efficientnetb0_training_curves.png` |
| EfficientNetB0 confusion matrix | `figures/confusion_matrix_efficientnet_enhanced.png` |
| EfficientNetB0 normalized confusion matrix | `figures/confusion_matrix_normalized_efficientnet_enhanced.png` |
| Grad-CAM examples | `figures/gradcam_examples.png` |

## Manuscript Metrics

| Model | Accuracy | Macro F1 | Weighted F1 | QWK | AUC macro |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.5855 | 0.3608 | 0.5766 | 0.5748 | 0.8337 |
| EfficientNetB0 | 0.7709 | 0.5652 | 0.7597 | 0.8796 | 0.9184 |
| Ordinal EfficientNetB0 | 0.7418 | 0.5553 | 0.7394 | 0.8540 | 0.9271 |

The detailed reproducibility instructions are maintained in the main artifact README: `../README.md`.
