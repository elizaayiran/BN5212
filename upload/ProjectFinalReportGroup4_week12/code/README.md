# CheXpert Multi-Label Classification - BN5212 Group 15 Final Project

## Project Overview

This project implements automated detection and localization of abnormalities in chest X-rays using weakly-supervised learning with the CheXpert dataset. We compare two state-of-the-art deep learning architectures (DenseNet-121 and ResNet-50) for multi-label pathology classification.

**Task:** Multi-label classification of 14 pathologies from chest X-ray images

**Dataset:** CheXpert - 5,534 DICOM images from 1,000 patients

**Models:**
- DenseNet-121 (7M parameters)
- ResNet-50 (23.5M parameters)

## Dataset Information

### Statistics
- **Total Images:** 5,534 DICOM files
- **Total Patients:** 1,000 patients
- **Total Studies:** 3,351 studies
- **Data Split (patient-level):**
  - Training: 4,594 images (800 patients)
  - Validation: 499 images (100 patients)
  - Test: 441 images (100 patients)

### 14 Pathology Classes
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

## Project Structure

```
code/
├── README.md                               # This file
├── mainfunctions/                          # Source code (9 core scripts)
│   ├── improved_chexpert_labeler.py        # NLP-based labeling from radiology reports
│   ├── build_dataset_from_dicom.py         # Dataset preparation and splitting
│   ├── dicom_dataset.py                    # PyTorch DICOM dataset class
│   ├── train_chexpert_densenet.py          # DenseNet-121 training script
│   ├── train_resnet50.py                   # ResNet-50 training script
│   ├── evaluate_model.py                   # DenseNet-121 evaluation script
│   ├── evaluate_resnet50.py                # ResNet-50 evaluation script
│   ├── gradcam_visualization.py            # Grad-CAM explainability visualization
│   └── create_model_comparison.py          # Generate model comparison visualizations
├── models/                                 # Trained model weights
│   ├── best_model.pth                      # DenseNet-121 (81 MB, Epoch 4, Val AUC: 0.6813)
│   └── best_model_resnet50.pth             # ResNet-50 (270 MB, Epoch 5, Val AUC: 0.6792)
└── results/                                # All experimental results
    ├── results_densenet121/                # DenseNet-121 test results
    │   ├── evaluation_metrics.json         # Test AUC: 0.6100, Accuracy: 0.8060
    │   ├── training_history_densenet121.json # Training history log
    │   ├── training_curves_densenet121.png # Training/validation curves
    │   ├── roc_curves.png                  # ROC curves for 14 classes
    │   ├── confusion_matrices.png          # Confusion matrices for all classes
    │   └── metrics_comparison.png          # Metrics bar charts
    ├── results_resnet50/                   # ResNet-50 test results
    │   ├── evaluation_metrics.json         # Test AUC: 0.6268, Accuracy: 0.8095
    │   ├── training_history_resnet50.json  # Training history log
    │   ├── training_curves_resnet50.png    # Training/validation curves
    │   ├── roc_curves.png                  # ROC curves for 14 classes
    │   ├── confusion_matrices.png          # Confusion matrices for all classes
    │   └── metrics_comparison.png          # Metrics bar charts
    ├── model_comparison/                   # Model comparison results
    │   ├── architecture_comparison_diagram.png # DenseNet-121 vs ResNet-50 architecture
    │   ├── per_class_auc_comparison.png    # Per-class AUC comparison
    │   ├── overall_metrics_comparison.png  # Overall metrics comparison
    │   ├── training_comparison.png         # Training curves comparison
    │   └── comparison_summary.txt          # Textual comparison summary
    └── gradcam/                            # Grad-CAM visualizations (10 test samples)
        ├── gradcam_1_Pneumothorax.png
        ├── gradcam_2_Atelectasis.png
        ├── gradcam_3_Pneumothorax.png
        ├── gradcam_4_Pneumothorax.png
        ├── gradcam_5_Pleural_Effusion.png
        ├── gradcam_6_Pleural_Effusion.png
        ├── gradcam_7_Pleural_Effusion.png
        ├── gradcam_8_Pleural_Effusion.png
        ├── gradcam_9_Pleural_Effusion.png
        ├── gradcam_10_Pneumothorax.png
        └── gradcam_visualization_clean.png # Grad-CAM methodology demonstration
```

## Requirements

### Python Dependencies
```
Python 3.8+
torch>=2.0.0
torchvision>=0.15.0
pydicom>=2.3.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
opencv-python>=4.7.0
tqdm>=4.64.0
Pillow>=9.4.0
```

### Hardware Requirements
- **GPU:** NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- **RAM:** 16GB+ recommended
- **Storage:** 5GB for dataset + 400MB for models

## Installation

```bash
# Install dependencies
pip install torch torchvision pydicom pandas numpy scikit-learn matplotlib seaborn opencv-python tqdm Pillow

# Verify CUDA availability (for GPU training)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## How to Run

### 1. Data Preprocessing (Already Completed)

The dataset has been preprocessed and split into train/validation/test sets at the patient level.

```bash
# To reproduce data preparation (optional)
python mainfunctions/improved_chexpert_labeler.py      # NLP labeling from reports
python mainfunctions/build_dataset_from_dicom.py       # Create train/val/test splits
```

**Output:**
- `E:\download\dataset\train.csv` (4,594 images)
- `E:\download\dataset\validate.csv` (499 images)
- `E:\download\dataset\test.csv` (441 images)

### 2. Model Training

**Note:** Pre-trained models are already provided in `models/` directory. Training takes 2-3 hours per model on RTX 4060.

#### Train DenseNet-121
```bash
python mainfunctions/train_chexpert_densenet.py
```

**Configuration:**
- Architecture: DenseNet-121 (pretrained on ImageNet)
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (lr=1e-4)
- Batch size: 16
- Epochs: 10
- Data augmentation: Random flip, rotation, color jitter

**Output:** `models/best_model.pth` (84.5 MB)

#### Train ResNet-50
```bash
python mainfunctions/train_resnet50.py
```

**Configuration:**
- Architecture: ResNet-50 (pretrained on ImageNet)
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (lr=1e-4)
- Batch size: 16
- Epochs: 10
- Data augmentation: Random flip, rotation, color jitter

**Output:** `models/best_model_resnet50.pth` (282 MB)

### 3. Model Evaluation

#### Evaluate DenseNet-121 on Test Set
```bash
python mainfunctions/evaluate_model.py
```

**Outputs:**
- `results/results_densenet121/evaluation_metrics.json` - Complete metrics (AUC, Acc, Precision, Recall, F1)
- `results/results_densenet121/roc_curves.png` - ROC curves for all 14 classes
- `results/results_densenet121/confusion_matrices.png` - Confusion matrices for all classes
- `results/results_densenet121/metrics_comparison.png` - Metrics comparison bar chart

**Key Results:**
- Mean AUC: 0.6100
- Mean Accuracy: 0.8060
- Best class: Support Devices (AUC: 0.8100)

#### Evaluate ResNet-50 on Test Set
```bash
python mainfunctions/evaluate_resnet50.py
```

**Outputs:**
- `results/results_resnet50/evaluation_metrics.json`
- `results/results_resnet50/roc_curves.png`
- `results/results_resnet50/confusion_matrices.png`
- `results/results_resnet50/metrics_comparison.png`

**Key Results:**
- Mean AUC: 0.6268
- Mean Accuracy: 0.8095
- Best class: Support Devices (AUC: 0.8264)
- **+0.0168 AUC improvement over DenseNet-121**

### 4. Model Comparison

```bash
python mainfunctions/create_model_comparison.py
```

**Outputs:**
- `results/model_comparison/architecture_comparison_diagram.png` - Architecture comparison diagram
- `results/model_comparison/per_class_auc_comparison.png` - Side-by-side AUC comparison
- `results/model_comparison/overall_metrics_comparison.png` - Overall metrics comparison
- `results/model_comparison/training_comparison.png` - Training curves comparison
- `results/model_comparison/comparison_summary.txt` - Textual summary

### 5. Grad-CAM Explainability Visualization

Generate Grad-CAM heatmaps showing where the model focuses when making predictions:

```bash
python mainfunctions/gradcam_visualization.py
```

**Outputs:**
- `results/gradcam/gradcam_*.png` - Individual Grad-CAM visualizations (10 samples)
- Each visualization shows: Original X-ray | Heatmap | Overlay

**Generated Samples:**
- 5 Pleural Effusion cases (confidence: 0.406-0.682)
- 4 Pneumothorax cases (confidence: 0.655-0.834)
- 1 Atelectasis case (confidence: 0.735)

**Important Note:** The Grad-CAM visualizations show relatively broad activation patterns rather than highly localized features. This is due to:
1. Moderate model performance (test AUC 0.61-0.63)
2. Severe class imbalance in training data
3. Low spatial resolution of DenseNet feature maps (7×7)

These visualizations suggest the model learns **global contextual features** rather than fine-grained pathological details. This is a known limitation that could be addressed with improved training strategies and higher-resolution feature extraction layers.

## Key Results Summary

### Model Performance Comparison

| Model | Parameters | Test AUC | Test Accuracy | Mean F1-Score |
|-------|-----------|----------|---------------|---------------|
| DenseNet-121 | 7.0M | 0.6100 | 0.8060 | 0.2114 |
| ResNet-50 | 23.5M | **0.6268** | **0.8095** | **0.2367** |

### Per-Class Performance (ResNet-50)

| Pathology | AUC | Accuracy |
|-----------|-----|----------|
| Support Devices | **0.8264** | 0.8730 |
| Cardiomegaly | 0.7456 | 0.8458 |
| Edema | 0.7143 | 0.7551 |
| Pleural Effusion | 0.7075 | 0.7596 |
| No Finding | 0.6954 | 0.9592 |

### Key Findings

1. **ResNet-50 outperforms DenseNet-121** by +0.0168 AUC (2.7% relative improvement)
2. **Support Devices** is the best-performing class for both models (AUC > 0.81)
3. Both models achieve high accuracy (>80%) but moderate F1-scores due to class imbalance
4. **5 classes** show zero predictions (all negative), indicating severe class imbalance:
   - No Finding, Enlarged Cardiomediastinum, Lung Lesion, Pleural Other, Fracture
5. Patient-level splitting ensures no data leakage and realistic evaluation

## Implementation Highlights

### 1. Direct DICOM Loading
- No PNG conversion required
- Saves ~3 hours preprocessing time
- Reduces storage requirements

### 2. Patient-Level Data Splitting
- Prevents data leakage (all images from same patient in same split)
- More realistic evaluation of generalization ability

### 3. Multi-Label Classification
- Independent binary prediction for each pathology
- Handles co-occurring conditions naturally
- BCEWithLogitsLoss for numerical stability

### 4. Transfer Learning
- ImageNet pretrained weights for both models
- Faster convergence and better performance

### 5. Explainable AI (Grad-CAM)
- Visual explanations of model predictions
- Helps build trust for clinical deployment
- Identifies relevant anatomical regions

## Limitations

1. **Class Imbalance**: Some pathologies have very few positive samples, leading to zero predictions for 5 classes
2. **Model Performance**: Moderate test AUC (0.61-0.63) indicates room for improvement in distinguishing pathologies
3. **Dataset Size**: Only 1,000 patients (relatively small for medical imaging deep learning)
4. **Single Modality**: Only chest X-rays (no integration with clinical data or radiology reports)
5. **Grad-CAM Localization**: Due to moderate model performance and class imbalance, Grad-CAM visualizations show broad activation patterns rather than highly localized pathological features. This suggests the model learns global contextual features. Higher resolution feature maps and improved model training could enhance localization quality.

## Future Work

1. **Address Class Imbalance**:
   - Use weighted loss functions
   - Apply SMOTE or other oversampling techniques
   - Adjust prediction thresholds per class

2. **Model Improvements**:
   - Try more advanced architectures (Vision Transformers, EfficientNet)
   - Ensemble methods combining multiple models
   - Multi-task learning with auxiliary tasks

3. **Data Augmentation**:
   - More aggressive augmentation for minority classes
   - MixUp or CutMix techniques

4. **Clinical Integration**:
   - Incorporate patient metadata (age, gender, medical history)
   - Multi-modal learning with radiology reports

5. **Explainability Enhancements**:
   - Improve Grad-CAM with higher resolution feature maps (e.g., using intermediate layers)
   - Implement attention mechanisms in model architecture
   - Apply Grad-CAM++ or other advanced visualization techniques
   - Quantify uncertainty in predictions for clinical safety

## References

1. **CheXpert:** Irvin, J., et al. (2019). "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI.

2. **DenseNet:** Huang, G., et al. (2017). "Densely Connected Convolutional Networks." CVPR.

3. **ResNet:** He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

4. **Grad-CAM:** Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.

## Authors

**BN5212 Group 15**
Final Project - October 2025

## Contact

For questions about the code or implementation details, please refer to the inline comments in the source code or contact the project team.

---

**Last Updated:** October 23, 2025
