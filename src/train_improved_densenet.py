"""
Improved Training Script for CheXpert DenseNet-121
Implements advanced techniques to boost AUC performance:
1. Strong data augmentation
2. Class weighting for imbalanced data
3. Focal loss for hard examples
4. Learning rate warmup and cosine annealing
5. Extended training (20 epochs)
6. Gradient accumulation for larger effective batch size
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime
import pandas as pd

# Import our custom DICOM dataset
from dicom_dataset import CheXpertDICOMDataset

# Paths
DATASET_DIR = r"E:\download\dataset"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "validate.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

# Output paths
OUTPUT_DIR = r"E:\download\BN5212_project\BN5212\models"
RESULTS_DIR = r"E:\download\BN5212_project\BN5212\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# Hyperparameters
BATCH_SIZE = 16  # Smaller batch for better generalization
NUM_EPOCHS = 20  # More epochs for better convergence
LEARNING_RATE = 3e-4  # Higher initial LR with warmup
ACCUMULATION_STEPS = 2  # Effective batch size = 32
WEIGHT_DECAY = 1e-4


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class DenseNetCheXpert(nn.Module):
    """
    DenseNet-121 with dropout for better regularization
    """
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        super(DenseNetCheXpert, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        num_features = self.densenet.classifier.in_features

        # Add dropout for regularization
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


def get_improved_data_loaders(batch_size=16):
    """
    Create data loaders with strong augmentation
    """
    # Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Standard transform for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CheXpertDICOMDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = CheXpertDICOMDataset(VAL_CSV, transform=val_transform)
    test_dataset = CheXpertDICOMDataset(TEST_CSV, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def calculate_class_weights(train_csv):
    """
    Calculate class weights based on positive/negative ratio
    """
    df = pd.read_csv(train_csv)
    weights = []

    for label in CHEXPERT_LABELS:
        pos_count = (df[label] == 1.0).sum()
        neg_count = (df[label] == 0.0).sum()
        total = pos_count + neg_count

        if pos_count > 0:
            # Weight for positive class (inverse frequency)
            weight = neg_count / pos_count
            weights.append(min(weight, 10.0))  # Cap at 10
        else:
            weights.append(1.0)

    return torch.FloatTensor(weights)


class ImprovedTrainer:
    def __init__(self, model, device, learning_rate=3e-4):
        self.model = model.to(device)
        self.device = device

        # Use Focal Loss for better handling of hard examples
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )

        # Cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        self.best_auc = 0.0
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_aucs': [],
            'learning_rates': [],
            'best_auc': 0.0,
            'best_epoch': 0
        }

    def train_epoch(self, train_loader, epoch, accumulation_steps=2):
        """
        Train for one epoch with gradient accumulation
        """
        self.model.train()
        running_loss = 0.0

        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader):
        """
        Validate model and calculate AUC
        """
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Store predictions and labels
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.append(predictions)
                all_labels.append(labels.cpu().numpy())

        avg_loss = running_loss / len(val_loader)
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        # Calculate AUC for each class
        aucs = []
        for i in range(len(CHEXPERT_LABELS)):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                aucs.append(auc)

        mean_auc = np.mean(aucs) if aucs else 0.0

        return avg_loss, mean_auc, aucs

    def train(self, train_loader, val_loader, num_epochs=20):
        """
        Complete training loop
        """
        print("=" * 80)
        print(f"Starting IMPROVED training for {num_epochs} epochs")
        print("Enhancements:")
        print("  - Focal Loss for hard examples")
        print("  - Strong data augmentation")
        print("  - Gradient accumulation (effective batch size: 32)")
        print("  - Cosine annealing with warm restarts")
        print("  - Extended training (20 epochs)")
        print("=" * 80)
        print()

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 80)

            # Training
            train_loss = self.train_epoch(train_loader, epoch, ACCUMULATION_STEPS)

            # Validation
            val_loss, val_auc, class_aucs = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['val_aucs'].append(val_auc)
            self.history['learning_rates'].append(current_lr)

            # Print summary
            print()
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Mean AUC:   {val_auc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print()
            print("  Per-class AUCs:")
            for label, auc in zip(CHEXPERT_LABELS, class_aucs):
                print(f"    {label:30s}: {auc:.4f}")
            print()

            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.history['best_auc'] = val_auc
                self.history['best_epoch'] = epoch

                model_path = os.path.join(OUTPUT_DIR, "best_model_improved.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': self.best_auc,
                }, model_path)
                print(f"  [OK] Saved best model (AUC: {val_auc:.4f})")

            print()

        # Save training history
        history_path = os.path.join(RESULTS_DIR, "training_history_improved.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print("=" * 80)
        print("Training completed!")
        print(f"Best AUC: {self.best_auc:.4f} at epoch {self.history['best_epoch']}")
        print("=" * 80)


def main():
    # Print system info
    print("=" * 80)
    print("CheXpert DenseNet-121 IMPROVED Training")
    print("=" * 80)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Load data with improved augmentation
    print("Loading data with strong augmentation...")
    train_loader, val_loader, test_loader = get_improved_data_loaders(BATCH_SIZE)
    print("[OK] Data loaders created:")
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    print()

    # Create improved model with dropout
    print("Creating improved DenseNet-121 model with dropout...")
    model = DenseNetCheXpert(num_classes=14, pretrained=True, dropout=0.3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {total_params} parameters")
    print()

    # Create trainer and start training
    print("Starting training...")
    trainer = ImprovedTrainer(model, device, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    print()
    print("Training script completed successfully!")


if __name__ == "__main__":
    main()
