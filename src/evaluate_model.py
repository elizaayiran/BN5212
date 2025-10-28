"""
Comprehensive Model Evaluation Script
Evaluates trained DenseNet-121 on test set with detailed metrics
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import our dataset
from dicom_dataset import CheXpertDICOMDataset

# Paths
DATASET_DIR = r"E:\download\dataset"
MODEL_PATH = r"E:\download\BN5212_project\BN5212\models\best_model.pth"
OUTPUT_DIR = r"E:\download\BN5212_project\BN5212\results\evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]


class DenseNetCheXpert(nn.Module):
    """DenseNet-121 for CheXpert"""

    def __init__(self, num_classes=14):
        super(DenseNetCheXpert, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate_on_dataset(self, dataloader):
        """
        Evaluate model on a dataset

        Returns:
            predictions: numpy array of predictions
            labels: numpy array of ground truth labels
        """
        all_predictions = []
        all_labels = []

        print("Running inference...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].cpu().numpy()

                outputs = self.model(images)
                predictions = torch.sigmoid(outputs).cpu().numpy()

                all_predictions.append(predictions)
                all_labels.append(labels)

        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        return all_predictions, all_labels

    def calculate_metrics(self, predictions, labels, threshold=0.5):
        """Calculate comprehensive metrics"""

        metrics = {}

        # Per-class AUC
        aucs = []
        for i in range(14):
            try:
                if len(np.unique(labels[:, i])) > 1:
                    auc = roc_auc_score(labels[:, i], predictions[:, i])
                    aucs.append(auc)
                else:
                    aucs.append(np.nan)
            except:
                aucs.append(np.nan)

        metrics['per_class_auc'] = aucs
        metrics['mean_auc'] = np.nanmean(aucs)

        # Binary predictions
        binary_preds = (predictions > threshold).astype(int)

        # Per-class metrics
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(14):
            acc = accuracy_score(labels[:, i], binary_preds[:, i])
            prec = precision_score(labels[:, i], binary_preds[:, i], zero_division=0)
            rec = recall_score(labels[:, i], binary_preds[:, i], zero_division=0)
            f1 = f1_score(labels[:, i], binary_preds[:, i], zero_division=0)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

        metrics['per_class_accuracy'] = accuracies
        metrics['per_class_precision'] = precisions
        metrics['per_class_recall'] = recalls
        metrics['per_class_f1'] = f1_scores

        metrics['mean_accuracy'] = np.mean(accuracies)
        metrics['mean_precision'] = np.mean(precisions)
        metrics['mean_recall'] = np.mean(recalls)
        metrics['mean_f1'] = np.mean(f1_scores)

        return metrics

    def plot_roc_curves(self, predictions, labels):
        """Plot ROC curves for all classes"""

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        for i in range(14):
            if len(np.unique(labels[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                auc = roc_auc_score(labels[:, i], predictions[:, i])

                axes[i].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
                axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(CHEXPERT_LABELS[i])
                axes[i].legend(loc='lower right')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, 'No positive samples',
                           ha='center', va='center')
                axes[i].set_title(CHEXPERT_LABELS[i])

        # Hide extra subplots
        for i in range(14, 16):
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] ROC curves saved to {save_path}")

    def plot_confusion_matrices(self, predictions, labels, threshold=0.5):
        """Plot confusion matrices for all classes"""

        binary_preds = (predictions > threshold).astype(int)

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        for i in range(14):
            cm = confusion_matrix(labels[:, i], binary_preds[:, i])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=axes[i], cbar=False)
            axes[i].set_title(CHEXPERT_LABELS[i])
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        # Hide extra subplots
        for i in range(14, 16):
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Confusion matrices saved to {save_path}")

    def plot_metrics_comparison(self, metrics):
        """Plot comparison of metrics across classes"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # AUC
        axes[0, 0].barh(CHEXPERT_LABELS, metrics['per_class_auc'])
        axes[0, 0].set_xlabel('AUC Score')
        axes[0, 0].set_title('AUC-ROC by Class')
        axes[0, 0].axvline(x=metrics['mean_auc'], color='r',
                          linestyle='--', label=f'Mean: {metrics["mean_auc"]:.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Precision
        axes[0, 1].barh(CHEXPERT_LABELS, metrics['per_class_precision'])
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_title('Precision by Class')
        axes[0, 1].axvline(x=metrics['mean_precision'], color='r',
                          linestyle='--', label=f'Mean: {metrics["mean_precision"]:.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Recall
        axes[1, 0].barh(CHEXPERT_LABELS, metrics['per_class_recall'])
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_title('Recall by Class')
        axes[1, 0].axvline(x=metrics['mean_recall'], color='r',
                          linestyle='--', label=f'Mean: {metrics["mean_recall"]:.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # F1-Score
        axes[1, 1].barh(CHEXPERT_LABELS, metrics['per_class_f1'])
        axes[1, 1].set_xlabel('F1-Score')
        axes[1, 1].set_title('F1-Score by Class')
        axes[1, 1].axvline(x=metrics['mean_f1'], color='r',
                          linestyle='--', label=f'Mean: {metrics["mean_f1"]:.3f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Metrics comparison saved to {save_path}")


def main():
    print("=" * 80)
    print("CheXpert Model Evaluation")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        print("Please train the model first!")
        return

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = DenseNetCheXpert(num_classes=14)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"[OK] Model loaded (Epoch: {epoch})")

    # Create evaluator
    evaluator = ModelEvaluator(model, device)

    # Load test dataset
    print("\nLoading test dataset...")
    test_csv = os.path.join(DATASET_DIR, "test.csv")
    from dicom_dataset import get_data_loaders

    _, _, test_loader = get_data_loaders(
        train_csv=os.path.join(DATASET_DIR, "train.csv"),
        val_csv=os.path.join(DATASET_DIR, "validate.csv"),
        test_csv=test_csv,
        batch_size=16,
        num_workers=0
    )

    # Evaluate
    print("\n" + "-" * 80)
    predictions, labels = evaluator.evaluate_on_dataset(test_loader)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = evaluator.calculate_metrics(predictions, labels)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nOverall Metrics:")
    print(f"  Mean AUC:       {metrics['mean_auc']:.4f}")
    print(f"  Mean Accuracy:  {metrics['mean_accuracy']:.4f}")
    print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
    print(f"  Mean Recall:    {metrics['mean_recall']:.4f}")
    print(f"  Mean F1-Score:  {metrics['mean_f1']:.4f}")

    print(f"\nPer-Class AUC:")
    for i, label in enumerate(CHEXPERT_LABELS):
        auc = metrics['per_class_auc'][i]
        if not np.isnan(auc):
            print(f"  {label:30s}: {auc:.4f}")
        else:
            print(f"  {label:30s}: N/A (no samples)")

    # Generate visualizations
    print("\n" + "-" * 80)
    print("Generating visualizations...")
    evaluator.plot_roc_curves(predictions, labels)
    evaluator.plot_confusion_matrices(predictions, labels)
    evaluator.plot_metrics_comparison(metrics)

    # Save metrics to JSON
    metrics_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_metrics': {
            'mean_auc': float(metrics['mean_auc']),
            'mean_accuracy': float(metrics['mean_accuracy']),
            'mean_precision': float(metrics['mean_precision']),
            'mean_recall': float(metrics['mean_recall']),
            'mean_f1': float(metrics['mean_f1'])
        },
        'per_class_metrics': {
            CHEXPERT_LABELS[i]: {
                'auc': float(metrics['per_class_auc'][i]) if not np.isnan(metrics['per_class_auc'][i]) else None,
                'accuracy': float(metrics['per_class_accuracy'][i]),
                'precision': float(metrics['per_class_precision'][i]),
                'recall': float(metrics['per_class_recall'][i]),
                'f1': float(metrics['per_class_f1'][i])
            }
            for i in range(14)
        }
    }

    metrics_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"[OK] Metrics saved to {metrics_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Evaluation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
