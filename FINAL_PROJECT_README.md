# CheXpert DenseNet-121 Final Project

**BN5212 Group 15 - Final Project**
**Topic:** Automated Detection and Localization of Abnormalities in Chest X-rays using Weakly-Supervised Learning

---

## Project Overview

This project implements a deep learning system for automated chest X-ray analysis using the CheXpert dataset. The system can detect 14 different pathologies and provides explainable AI visualizations through Grad-CAM.

### Key Features
- ✅ Multi-label classification (14 pathologies)
- ✅ DenseNet-121 architecture with ImageNet pretraining
- ✅ Grad-CAM explainability for model interpretability
- ✅ Comprehensive evaluation metrics (AUC-ROC, Precision, Recall, F1)
- ✅ Direct DICOM image loading (no PNG conversion needed)
- ✅ Patient-level data splitting (prevents data leakage)

---

## Dataset

### Statistics
- **Total Images:** 5,534 DICOM files
- **Total Patients:** 1,000
- **Total Studies:** 3,351
- **Dataset Split:**
  - Training: 4,594 images (800 patients)
  - Validation: 499 images (100 patients)
  - Test: 441 images (100 patients)

### Pathologies (14 classes)
1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Lesion
5. Lung Opacity
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

### Label Distribution (Training Set)
- Pneumothorax: 2,529 positive cases
- Pleural Effusion: 2,460 positive cases
- Consolidation: 1,862 positive cases
- Pneumonia: 1,704 positive cases
- Atelectasis: 1,587 positive cases
- Edema: 1,531 positive cases
- Lung Opacity: 1,381 positive cases
- Cardiomegaly: 548 positive cases

---

## Project Structure

```
BN5212/
├── src/                                    # Source code
│   ├── build_dataset_from_dicom.py        # Dataset preparation
│   ├── dicom_dataset.py                   # PyTorch DICOM dataset
│   ├── train_chexpert_densenet.py         # Training script
│   ├── evaluate_model.py                  # Evaluation script
│   ├── gradcam_visualization.py           # Grad-CAM generation
│   └── improved_chexpert_labeler.py       # NLP labeler
├── models/                                 # Saved models
│   └── best_model.pth                     # Best trained model
├── results/                                # Generated results
│   ├── training_curves.png                # Training visualizations
│   ├── training_history.json              # Training metrics
│   ├── evaluation/                        # Evaluation results
│   │   ├── evaluation_metrics.json
│   │   ├── roc_curves.png
│   │   ├── confusion_matrices.png
│   │   └── metrics_comparison.png
│   └── gradcam/                           # Grad-CAM visualizations
│       ├── gradcam_1_*.png
│       └── gradcam_multi_class_comparison.png
├── data/                                   # Labeled data
├── check_training.py                      # Training status checker
└── run_complete_evaluation.py             # Complete evaluation pipeline
```

---

## Implementation Details

### Model Architecture
- **Base Model:** DenseNet-121 (pretrained on ImageNet)
- **Parameters:** ~7 million
- **Final Layer:** Linear(1024 → 14) for multi-label classification
- **Activation:** Sigmoid (for independent multi-label prediction)

### Training Configuration
- **Loss Function:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)
- **Batch Size:** 16
- **Epochs:** 10
- **GPU:** NVIDIA GeForce RTX 4060

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Color jitter (brightness=0.2, contrast=0.2)
- Resize to 224×224
- ImageNet normalization

---

## How to Use

### 1. Check Training Progress
```bash
python check_training.py
```

### 2. Run Complete Evaluation (After Training)
```bash
python run_complete_evaluation.py
```

This will generate:
- Model evaluation metrics (AUC, precision, recall, F1)
- ROC curves for all 14 classes
- Confusion matrices
- Grad-CAM visualizations (10 samples + multi-class comparison)

### 3. Manual Evaluation
```bash
# Model evaluation only
python src/evaluate_model.py

# Grad-CAM visualization only
python src/gradcam_visualization.py
```

---

## Results

### Training Performance
- **Best Validation AUC:** [Will be filled after training]
- **Training Loss:** Decreased from ~0.75 to ~0.38 in Epoch 1
- **Convergence:** Stable training with good loss reduction

### Evaluation Metrics
Located in: `results/evaluation/evaluation_metrics.json`

Metrics include:
- Per-class AUC-ROC scores
- Overall mean AUC
- Precision, Recall, F1-score for each class
- Confusion matrices

### Explainability (Grad-CAM)
Located in: `results/gradcam/`

Visualizations show:
- Where the model focuses for each prediction
- Multi-class comparison on same image
- Heatmap overlays on original X-rays

---

## Key Files

### Data Files
- `E:\download\dataset\train.csv` - Training set (4,594 images)
- `E:\download\dataset\validate.csv` - Validation set (499 images)
- `E:\download\dataset\test.csv` - Test set (441 images)
- `E:\download\dataset\labeled_reports.csv` - All labeled reports

### Model Files
- `models/best_model.pth` - Best model checkpoint
- `results/training_history.json` - Complete training history

### Visualization Files
- `results/training_curves.png` - Training/validation curves
- `results/evaluation/roc_curves.png` - ROC curves (14 classes)
- `results/evaluation/confusion_matrices.png` - Confusion matrices
- `results/evaluation/metrics_comparison.png` - Metrics comparison
- `results/gradcam/*.png` - Grad-CAM heatmaps

---

## Technical Highlights

### 1. Efficient DICOM Loading
- Direct DICOM reading without PNG conversion
- Saves ~3 hours of preprocessing time
- Reduces storage requirements

### 2. Patient-Level Splitting
- Ensures no data leakage
- All images from same patient stay in same split
- Realistic evaluation of generalization

### 3. Multi-Label Classification
- Independent prediction for each pathology
- Handles co-occurring conditions
- BCEWithLogitsLoss for numerical stability

### 4. Explainable AI
- Grad-CAM for visual explanations
- Shows model attention regions
- Builds trust for clinical deployment

---

## Dependencies

```
Python 3.8+
PyTorch 2.0+
torchvision
pydicom
pandas
numpy
scikit-learn
matplotlib
seaborn
opencv-python
tqdm
Pillow
```

---

## References

1. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison (Irvin et al., 2019)
2. Densely Connected Convolutional Networks (Huang et al., 2017)
3. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (Selvaraju et al., 2017)

---

## Authors

**BN5212 Group 15**
Final Project - 2025

---

## Notes

- Training takes approximately 2-3 hours on RTX 4060
- All paths are configured for Windows (use forward slashes for Linux/Mac)
- GPU is highly recommended for training
- Results are automatically saved in `results/` directory

---

## Contact

For questions or issues, please refer to the project documentation or contact the project team.
