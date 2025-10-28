# CheXpert Chest X-ray Pathology Detection

Automated detection and localization of 14 pathologies in chest X-rays using weakly-supervised deep learning.

## Project Overview

This project implements a multi-label classification system for chest X-ray analysis using:
- **Dataset**: MIMIC-CXR with 5,534 DICOM images from 1,000 patients
- **Task**: Multi-label classification for 14 CheXpert pathologies
- **Models**: DenseNet-121 (primary) and ResNet-50 (comparison)
- **Labeling**: Automated CheXpert NLP labeler on radiology reports
- **Explainability**: Grad-CAM visualizations

## Dataset Statistics

- **Total Images**: 5,534 DICOM chest X-rays
- **Patients**: 1,000 unique patients
- **Train/Val/Test Split**: 4,594 / 499 / 441 (patient-level split)
- **Labels**: 14 pathology classes from CheXpert labeler

### CheXpert Pathology Classes

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

## Model Performance

### DenseNet-121 (Best Model)

**Test Set Performance:**
- Mean AUC-ROC: **0.6100**
- Mean Accuracy: **0.8060**
- Best Validation AUC: **0.6813** (Epoch 4)

**Top Performing Classes:**
- Support Devices: AUC 0.815
- Pneumothorax: AUC 0.742
- Atelectasis: AUC 0.711
- Cardiomegaly: AUC 0.707

**Model Architecture:**
- Base: DenseNet-121 pretrained on ImageNet
- Parameters: ~7M
- Final Layer: 14-class multi-label classifier
- Loss: Binary Cross-Entropy with Logits

## Project Structure

```
BN5212/
├── src/
│   ├── build_dataset_from_dicom.py    # Data processing pipeline
│   ├── dicom_dataset.py                # PyTorch Dataset for DICOM loading
│   ├── train_chexpert_densenet.py      # DenseNet-121 training
│   ├── train_resnet50.py               # ResNet-50 training
│   ├── evaluate_model.py               # Model evaluation & visualization
│   ├── gradcam_visualization.py        # Grad-CAM explainability
│   └── compare_models.py               # Model comparison
├── models/
│   ├── best_model.pth                  # DenseNet-121 weights (Epoch 4)
│   └── best_model_resnet50.pth         # ResNet-50 weights
├── results/
│   ├── training_curves.png             # Loss and AUC curves
│   ├── training_history.json           # Training logs
│   └── evaluation/
│       ├── roc_curves.png              # ROC curves for 14 classes
│       ├── confusion_matrices.png       # Confusion matrices
│       ├── metrics_comparison.png       # Per-class metrics
│       └── evaluation_metrics.json      # Detailed metrics
└── README.md
```

## Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.5.0
pydicom>=2.3.0
tqdm>=4.65.0
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd BN5212_project/BN5212
```

2. **Install dependencies:**
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow pydicom tqdm
```

3. **Install CheXpert labeler:**
```bash
pip install git+https://github.com/stanfordmlgroup/chexpert-labeler.git
```

## Usage

### 1. Data Processing

Process DICOM images and extract labels from radiology reports:

```bash
python src/build_dataset_from_dicom.py
```

**Input:** Raw DICOM files and radiology reports in `E:\download\dataset`

**Output:**
- `train.csv` - Training set with image paths and labels
- `validate.csv` - Validation set
- `test.csv` - Test set
- `labeled_reports.csv` - CheXpert labels for all reports

### 2. Model Training

#### Train DenseNet-121:
```bash
python src/train_chexpert_densenet.py
```

**Configuration:**
- Epochs: 10
- Batch Size: 16
- Learning Rate: 1e-4
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

**Output:**
- `models/best_model.pth` - Best model checkpoint
- `results/training_curves.png` - Training visualization
- `results/training_history.json` - Training logs

#### Train ResNet-50 (for comparison):
```bash
python src/train_resnet50.py
```

### 3. Model Evaluation

Run comprehensive evaluation on test set:

```bash
python src/evaluate_model.py
```

**Generates:**
- ROC curves for all 14 classes
- Confusion matrices
- Per-class metrics (AUC, Precision, Recall, F1)
- Detailed JSON metrics file

**Output Directory:** `results/evaluation/`

### 4. Grad-CAM Visualization

Generate explainability visualizations:

```bash
python src/gradcam_visualization.py
```

**Generates:** Heatmaps showing which image regions the model focuses on for predictions.

### 5. Model Comparison

Compare DenseNet-121 vs ResNet-50 (after training both):

```bash
python src/compare_models.py
```

**Generates:**
- Training curve comparison
- AUC comparison
- Parameter efficiency analysis

## Key Technical Contributions

1. **Direct DICOM Loading**: Custom PyTorch Dataset eliminates preprocessing bottleneck by loading DICOM files on-the-fly during training.

2. **Patient-Level Data Split**: Ensures no data leakage by splitting at patient level rather than image level.

3. **Automated Label Extraction**: CheXpert NLP labeler automatically extracts 14 pathology labels from free-text radiology reports.

4. **Multi-Label Classification**: Handles 14 independent binary classification tasks simultaneously.

5. **Model Explainability**: Grad-CAM visualizations provide interpretable insights into model predictions.

## Training Details

### Data Augmentation
- Resize: 256x256
- Center Crop: 224x224
- Random Horizontal Flip: 50%
- Normalization: ImageNet statistics

### Loss Function
- Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- Handles uncertain labels (-1) by converting to positive (1)

### Optimization
- Optimizer: Adam
- Learning Rate: 1e-4
- LR Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Early stopping: Based on validation AUC

### Training Time
- DenseNet-121: ~2 hours (10 epochs, RTX 4060)
- ResNet-50: ~2-3 hours (10 epochs, RTX 4060)

## Results Interpretation

### AUC Scores
- **0.8-1.0**: Excellent discrimination
- **0.7-0.8**: Good discrimination
- **0.6-0.7**: Moderate discrimination
- **0.5-0.6**: Poor discrimination

Our model achieves:
- 4 classes with AUC > 0.7 (excellent)
- 6 classes with AUC 0.6-0.7 (good to moderate)
- 4 classes with AUC < 0.6 (challenging classes)

### Challenging Classes
- **Enlarged Cardiomediastinum** (AUC 0.092): Very low prevalence and subtle features
- **Pleural Effusion** (AUC 0.510): Overlapping with other pleural conditions
- **Pneumonia** (AUC 0.572): Variable presentation patterns

### Strong Performance
- **Support Devices** (AUC 0.815): Clear visual features (tubes, lines)
- **Pneumothorax** (AUC 0.742): Distinct air patterns
- **Atelectasis** (AUC 0.711): Well-defined opacity patterns

## Limitations and Future Work

### Current Limitations
1. Limited dataset size (5,534 images)
2. Class imbalance in some pathologies
3. Single view per patient (no multi-view fusion)
4. No ensemble methods

### Future Improvements
1. **Data Augmentation**: Stronger augmentation techniques (CutMix, MixUp)
2. **Architecture**: Vision Transformers, EfficientNet variants
3. **Training**: Longer training (20+ epochs), focal loss for imbalanced classes
4. **Ensemble**: Combine DenseNet + ResNet predictions
5. **Multi-View**: Incorporate both frontal and lateral views
6. **Attention Mechanisms**: Self-attention for better feature extraction

## Citation

If you use this code or methodology, please cite:

```
@misc{chexpert-bn5212,
  author = {BN5212 Group 15},
  title = {Automated Detection of Chest X-ray Pathologies using Deep Learning},
  year = {2025},
  institution = {National University of Singapore}
}
```

## References

1. Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI.
2. Huang, G., et al. (2017). Densely Connected Convolutional Networks. CVPR.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
4. Johnson, A. E., et al. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data.

## License

This project is for educational purposes as part of BN5212 coursework at NUS.

## Contact

For questions or issues, please contact the project team.

---

**Last Updated**: October 22, 2025
**Course**: BN5212 - Deep Learning in Biomedicine
**Institution**: National University of Singapore
