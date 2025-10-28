"""
Generate complete evaluation visualizations for ResNet-50
(Same format as DenseNet evaluation folder)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
from datetime import datetime

# Import dataset
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dicom_dataset import CheXpertDICOMDataset

# Paths
BASE_DIR = r"E:\download\BN5212_project\BN5212"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_resnet50.pth")
TEST_CSV = r"E:\download\dataset\test.csv"
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "evaluation_resnet50")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CheXpert classes
CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

class ResNetCheXpert(nn.Module):
    """ResNet-50 for CheXpert"""
    def __init__(self, num_classes=14, pretrained=False):
        super(ResNetCheXpert, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def evaluate_model(model, dataloader, device):
    """Evaluate model and get predictions"""
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            predictions = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    return all_labels, all_predictions

def calculate_metrics(labels, predictions, threshold=0.5):
    """Calculate all metrics"""
    metrics = {}

    per_class_metrics = {}
    for i, class_name in enumerate(CLASSES):
        class_labels = labels[:, i]
        class_preds = predictions[:, i]
        class_binary_preds = (class_preds >= threshold).astype(int)

        try:
            auc_score = roc_auc_score(class_labels, class_preds)
        except:
            auc_score = 0.5

        acc = accuracy_score(class_labels, class_binary_preds)
        prec = precision_score(class_labels, class_binary_preds, zero_division=0)
        rec = recall_score(class_labels, class_binary_preds, zero_division=0)
        f1 = f1_score(class_labels, class_binary_preds, zero_division=0)

        per_class_metrics[class_name] = {
            "auc": float(auc_score),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }

    # Overall metrics
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics["overall_metrics"] = {
        "mean_auc": float(np.mean([m["auc"] for m in per_class_metrics.values()])),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in per_class_metrics.values()])),
        "mean_precision": float(np.mean([m["precision"] for m in per_class_metrics.values()])),
        "mean_recall": float(np.mean([m["recall"] for m in per_class_metrics.values()])),
        "mean_f1": float(np.mean([m["f1"] for m in per_class_metrics.values()]))
    }
    metrics["per_class_metrics"] = per_class_metrics

    return metrics

def plot_roc_curves(labels, predictions):
    """Plot ROC curves for all classes"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.ravel()

    for i, class_name in enumerate(CLASSES):
        class_labels = labels[:, i]
        class_preds = predictions[:, i]

        fpr, tpr, _ = roc_curve(class_labels, class_preds)
        roc_auc = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, color='#e74c3c', lw=2,
                    label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.3)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate', fontsize=9)
        axes[i].set_ylabel('True Positive Rate', fontsize=9)
        axes[i].set_title(class_name, fontsize=10, fontweight='bold')
        axes[i].legend(loc="lower right", fontsize=8)
        axes[i].grid(alpha=0.3, linestyle='--')

    # Remove last 2 empty subplots
    for i in range(len(CLASSES), 16):
        fig.delaxes(axes[i])

    plt.suptitle('ROC Curves - ResNet-50', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] ROC curves saved to {output_path}")

def plot_confusion_matrices(labels, predictions, threshold=0.5):
    """Plot confusion matrices for all classes"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.ravel()

    binary_preds = (predictions >= threshold).astype(int)

    for i, class_name in enumerate(CLASSES):
        class_labels = labels[:, i]
        class_preds = binary_preds[:, i]

        cm = confusion_matrix(class_labels, class_preds)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[i], cbar_kws={'shrink': 0.8})

        axes[i].set_title(class_name, fontsize=10, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=9)
        axes[i].set_xlabel('Predicted Label', fontsize=9)

    # Remove last 2 empty subplots
    for i in range(len(CLASSES), 16):
        fig.delaxes(axes[i])

    plt.suptitle('Confusion Matrices - ResNet-50', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Confusion matrices saved to {output_path}")

def plot_metrics_comparison(metrics):
    """Plot metrics comparison bar chart"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # AUC
    aucs = [metrics['per_class_metrics'][cls]['auc'] for cls in CLASSES]
    axes[0].barh(CLASSES, aucs, color='#e74c3c', alpha=0.8)
    axes[0].set_xlabel('AUC Score', fontsize=11, fontweight='bold')
    axes[0].set_title('AUC by Class', fontsize=12, fontweight='bold')
    axes[0].set_xlim([0, 1])
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')

    # Precision & Recall
    precisions = [metrics['per_class_metrics'][cls]['precision'] for cls in CLASSES]
    recalls = [metrics['per_class_metrics'][cls]['recall'] for cls in CLASSES]

    y_pos = np.arange(len(CLASSES))
    width = 0.35

    axes[1].barh(y_pos - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    axes[1].barh(y_pos + width/2, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(CLASSES)
    axes[1].set_xlabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')

    # F1-Score
    f1_scores = [metrics['per_class_metrics'][cls]['f1'] for cls in CLASSES]
    axes[2].barh(CLASSES, f1_scores, color='#9b59b6', alpha=0.8)
    axes[2].set_xlabel('F1-Score', fontsize=11, fontweight='bold')
    axes[2].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
    axes[2].set_xlim([0, 1])
    axes[2].grid(axis='x', alpha=0.3, linestyle='--')

    plt.suptitle('Performance Metrics Comparison - ResNet-50',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Metrics comparison saved to {output_path}")

def main():
    print("=" * 80)
    print("ResNet-50 Complete Evaluation Visualizations")
    print("=" * 80)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = ResNetCheXpert(num_classes=14, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"[OK] ResNet-50 model loaded (Epoch {checkpoint.get('epoch', 'N/A')})")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = CheXpertDICOMDataset(csv_file=TEST_CSV)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"[OK] Test dataset loaded: {len(test_dataset)} samples")

    # Evaluate
    print("\nRunning inference...")
    labels, predictions = evaluate_model(model, test_loader, device)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(labels, predictions)

    # Save metrics JSON (same format as DenseNet)
    metrics_file = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved to {metrics_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_roc_curves(labels, predictions)
    plot_confusion_matrices(labels, predictions)
    plot_metrics_comparison(metrics)

    print("\n" + "=" * 80)
    print("[SUCCESS] All ResNet-50 evaluation files generated!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
