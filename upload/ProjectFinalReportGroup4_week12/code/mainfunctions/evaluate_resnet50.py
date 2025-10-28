"""
Evaluate ResNet-50 model on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json
import os
from tqdm import tqdm

# Import dataset
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dicom_dataset import CheXpertDICOMDataset

# Paths
BASE_DIR = r"E:\download\BN5212_project\BN5212"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_resnet50.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "evaluation_resnet50")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# CheXpert classes
CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

class ResNetCheXpert(nn.Module):
    """ResNet-50 for CheXpert multi-label classification"""

    def __init__(self, num_classes=14, pretrained=False):
        super(ResNetCheXpert, self).__init__()

        # Load ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
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
    """Calculate evaluation metrics"""
    metrics = {}

    # Per-class metrics
    per_class_metrics = {}
    aucs = []

    for i, class_name in enumerate(CLASSES):
        class_labels = labels[:, i]
        class_preds = predictions[:, i]
        class_binary_preds = (class_preds >= threshold).astype(int)

        # AUC
        try:
            auc = roc_auc_score(class_labels, class_preds)
        except:
            auc = 0.5
        aucs.append(auc)

        # Other metrics
        acc = accuracy_score(class_labels, class_binary_preds)
        prec = precision_score(class_labels, class_binary_preds, zero_division=0)
        rec = recall_score(class_labels, class_binary_preds, zero_division=0)
        f1 = f1_score(class_labels, class_binary_preds, zero_division=0)

        per_class_metrics[class_name] = {
            "auc": float(auc),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }

    # Overall metrics
    metrics["overall_metrics"] = {
        "mean_auc": float(np.mean(aucs)),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in per_class_metrics.values()])),
        "mean_precision": float(np.mean([m["precision"] for m in per_class_metrics.values()])),
        "mean_recall": float(np.mean([m["recall"] for m in per_class_metrics.values()])),
        "mean_f1": float(np.mean([m["f1"] for m in per_class_metrics.values()]))
    }

    metrics["per_class_metrics"] = per_class_metrics

    return metrics

def main():
    print("=" * 80)
    print("ResNet-50 Model Evaluation")
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
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"[OK] ResNet-50 model loaded (Epoch: {epoch})")

    # Load test dataset
    print("\nLoading test dataset...")
    TEST_CSV = r"E:\download\dataset\test.csv"
    test_dataset = CheXpertDICOMDataset(csv_file=TEST_CSV)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"[OK] Test dataset loaded: {len(test_dataset)} samples")

    # Evaluate
    print("\nRunning inference...")
    labels, predictions = evaluate_model(model, test_loader, device)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(labels, predictions)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (ResNet-50)")
    print("=" * 80)
    print("\nOverall Metrics:")
    for key, value in metrics["overall_metrics"].items():
        print(f"  {key.replace('_', ' ').title():20s}: {value:.4f}")

    print("\nPer-Class AUC:")
    for class_name in CLASSES:
        auc = metrics["per_class_metrics"][class_name]["auc"]
        print(f"  {class_name:30s}: {auc:.4f}")

    # Save results
    output_file = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Metrics saved to {output_file}")

    print("\n" + "=" * 80)
    print("[SUCCESS] ResNet-50 evaluation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
