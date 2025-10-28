# BN5212 Final Project Submission Checklist

## ✅ Submission Folder: `upload/ProjectFinalReportGroup15/`

---

## 📂 File Structure Summary

### 📄 Total Files: 41

#### 1. **Report** (1 file)
- ✅ `report.pdf` - Final project report (6 pages, NeurIPS 2025 format)

#### 2. **Source Code** (9 Python files in `code/mainfunctions/`)
- ✅ `improved_chexpert_labeler.py` - NLP-based weak label extraction
- ✅ `build_dataset_from_dicom.py` - Dataset preparation and patient-level splitting
- ✅ `dicom_dataset.py` - PyTorch Dataset class for direct DICOM loading
- ✅ `train_chexpert_densenet.py` - DenseNet-121 training script
- ✅ `train_resnet50.py` - ResNet-50 training script
- ✅ `evaluate_model.py` - DenseNet-121 evaluation on test set
- ✅ `evaluate_resnet50.py` - ResNet-50 evaluation on test set
- ✅ `gradcam_visualization.py` - Grad-CAM explainability visualization
- ✅ `create_model_comparison.py` - Model comparison visualizations

#### 3. **Trained Models** (2 files in `code/models/`)
- ✅ `best_model.pth` - DenseNet-121 (81 MB, Test AUC: 0.610)
- ✅ `best_model_resnet50.pth` - ResNet-50 (270 MB, Test AUC: 0.627)

#### 4. **Results** (28 files in `code/results/`)

##### **DenseNet-121 Results** (6 files in `results_densenet121/`)
- ✅ `evaluation_metrics.json` - Test metrics (AUC: 0.6100, Acc: 0.8060)
- ✅ `training_history_densenet121.json` - Training logs
- ✅ `training_curves_densenet121.png` - Training/validation curves
- ✅ `roc_curves.png` - ROC curves for 14 pathologies
- ✅ `confusion_matrices.png` - Confusion matrices
- ✅ `metrics_comparison.png` - Metrics bar charts

##### **ResNet-50 Results** (6 files in `results_resnet50/`)
- ✅ `evaluation_metrics.json` - Test metrics (AUC: 0.6268, Acc: 0.8095)
- ✅ `training_history_resnet50.json` - Training logs
- ✅ `training_curves_resnet50.png` - Training/validation curves
- ✅ `roc_curves.png` - ROC curves for 14 pathologies
- ✅ `confusion_matrices.png` - Confusion matrices
- ✅ `metrics_comparison.png` - Metrics bar charts

##### **Model Comparison** (5 files in `model_comparison/`)
- ✅ `architecture_comparison_diagram.png` - DenseNet-121 vs ResNet-50 architecture diagram
- ✅ `per_class_auc_comparison.png` - Per-class AUC comparison
- ✅ `overall_metrics_comparison.png` - Overall metrics comparison
- ✅ `training_comparison.png` - Training curves comparison
- ✅ `comparison_summary.txt` - Textual comparison summary

##### **Grad-CAM Visualizations** (11 files in `gradcam/`)
- ✅ `gradcam_1_Pneumothorax.png` through `gradcam_10_Pneumothorax.png` (10 test samples)
- ✅ `gradcam_visualization_clean.png` - Grad-CAM methodology demonstration

#### 5. **Documentation** (1 file)
- ✅ `code/README.md` - Comprehensive project documentation (350 lines)

---

## 📊 Key Statistics

### Dataset
- **Total Images**: 5,534 DICOM files
- **Total Patients**: 1,000 patients
- **Train/Val/Test Split**: 4,594 / 499 / 441 images (80/10/10 at patient level)
- **Classes**: 14 thoracic pathologies (multi-label classification)

### Models
| Model | Parameters | Model Size | Test AUC | Test Accuracy |
|-------|-----------|------------|----------|---------------|
| DenseNet-121 | 7.0M | 81 MB | 0.610 | 0.806 |
| ResNet-50 | 23.5M | 270 MB | **0.627** | **0.810** |

### Performance Highlights
- ✅ ResNet-50 achieves +2.76% higher AUC than DenseNet-121
- ✅ DenseNet-121 has 3.35× fewer parameters (better efficiency)
- ✅ Support Devices: Best performing class (AUC > 0.81 for both models)
- ✅ Patient-level splitting ensures no data leakage

---

## 🔍 Verification Checklist

### ✅ Report Requirements
- [x] 6-page limit (using NeurIPS 2025 format)
- [x] Includes Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion
- [x] All figures and tables properly referenced
- [x] Comparative analysis of DenseNet-121 vs ResNet-50
- [x] Grad-CAM explainability analysis included
- [x] Honest discussion of limitations

### ✅ Code Requirements
- [x] All source code organized in `code/mainfunctions/`
- [x] README with complete documentation
- [x] Requirements.txt or dependency list provided
- [x] Trained model weights included
- [x] All experimental results saved

### ✅ Results Requirements
- [x] Training curves for both models
- [x] Test set evaluation metrics (JSON format)
- [x] ROC curves for all 14 pathologies
- [x] Confusion matrices
- [x] Model comparison visualizations
- [x] Grad-CAM visualizations (10 test samples)

---

## 📦 Submission Files

### Main Submission Folder
```
upload/ProjectFinalReportGroup15/
├── report.pdf                           (Final report, 6 pages)
└── code/
    ├── README.md                        (Project documentation)
    ├── mainfunctions/                   (9 Python scripts)
    ├── models/                          (2 trained models, 351 MB total)
    └── results/                         (28 result files)
        ├── results_densenet121/         (6 files)
        ├── results_resnet50/            (6 files)
        ├── model_comparison/            (5 files)
        └── gradcam/                     (11 files)
```

### Total Submission Size
- **Code + Models + Results**: ~380 MB
- **Report**: ~3 MB
- **Total**: ~383 MB

---

## 🎯 Key Achievements

1. ✅ **Weak Supervision**: Automated label extraction using CheXpert NLP tool
2. ✅ **Multi-Label Classification**: 14 thoracic pathologies
3. ✅ **Direct DICOM Loading**: Custom PyTorch dataset class
4. ✅ **Patient-Level Splitting**: Prevents data leakage
5. ✅ **Transfer Learning**: ImageNet pretrained weights
6. ✅ **Comparative Analysis**: DenseNet-121 vs ResNet-50
7. ✅ **Explainable AI**: Grad-CAM visualizations
8. ✅ **Honest Reporting**: Transparent discussion of limitations

---

## 📝 Important Notes

### ResNet-50 Performance
- ResNet-50 achieved **higher test AUC (0.627 vs 0.610)** and **accuracy (0.810 vs 0.806)**
- However, DenseNet-121 is **3.35× more parameter-efficient** (7M vs 23.5M params)
- The report objectively presents this as a **performance-efficiency trade-off**

### Grad-CAM Limitations
- Visualizations show **broad activation patterns** rather than precise localization
- Report includes **honest discussion** of this limitation
- Attributed to moderate model performance, class imbalance, and low feature map resolution
- This transparency demonstrates **scientific maturity**

### Class Imbalance
- 5 classes show zero predictions on test set (severe imbalance)
- Report discusses this as a **dataset limitation** requiring future work
- Suggests class-balanced loss functions and label uncertainty modeling

---

## ✅ Final Checks Before Submission

- [x] All file paths in README match actual folder structure
- [x] All Python scripts use correct import paths
- [x] Model file sizes match documentation (81 MB and 270 MB)
- [x] All result files present and accessible
- [x] Report PDF renders correctly
- [x] No absolute file paths in code (portable across systems)
- [x] No sensitive information or credentials in code
- [x] All data sources and references properly cited

---

## 🚀 Submission Ready

**Status**: ✅ **READY FOR SUBMISSION**

**Submission Date**: October 24, 2025 (before 2359 SGT)

**Group**: BN5212 Group 15
- Song Yiran (A0294421H)
- Wu Yuhang (A0294425Y)
- Lu Jinzhou (A0329981M)

---

**Last Verified**: October 23, 2025
