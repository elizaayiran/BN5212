"""
ResNet-50 Training Script for CheXpert
For comparison with DenseNet-121 as mentioned in midterm presentation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import our custom DICOM dataset
from dicom_dataset import get_data_loaders

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
    "Pleural Other", "Fracture", "Support Devices"
]


class ResNetCheXpert(nn.Module):
    """ResNet-50 for CheXpert multi-label classification"""

    def __init__(self, num_classes=14, pretrained=True):
        super(ResNetCheXpert, self).__init__()

        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CheXpertTrainer:
    """Trainer for CheXpert model"""

    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device

        # Multi-label loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )

        # Track history
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                # Store predictions and labels for metrics
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(torch.sigmoid(outputs).cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        all_labels = np.vstack(all_labels)
        all_predictions = np.vstack(all_predictions)

        # Calculate AUC for each class
        aucs = []
        for i in range(14):
            try:
                if len(np.unique(all_labels[:, i])) > 1:
                    auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                    aucs.append(auc)
                else:
                    aucs.append(0.0)
            except:
                aucs.append(0.0)

        mean_auc = np.mean(aucs)

        self.val_losses.append(avg_loss)
        self.val_aucs.append(mean_auc)

        return avg_loss, mean_auc, aucs

    def train(self, train_loader, val_loader, num_epochs=10):
        """Full training loop"""
        best_auc = 0.0
        best_epoch = 0

        print("=" * 80)
        print(f"Starting ResNet-50 training for {num_epochs} epochs")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, mean_auc, class_aucs = self.validate(val_loader, epoch)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Mean AUC:   {mean_auc:.4f}")

            # Print per-class AUCs
            print(f"\n  Per-class AUCs:")
            for label, auc in zip(CHEXPERT_LABELS, class_aucs):
                print(f"    {label:30s}: {auc:.4f}")

            # Save best model
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_epoch = epoch
                model_path = os.path.join(OUTPUT_DIR, "best_model_resnet50.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mean_auc': mean_auc,
                    'class_aucs': class_aucs,
                }, model_path)
                print(f"\n  [OK] Saved best ResNet-50 model (AUC: {mean_auc:.4f})")

        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best model: Epoch {best_epoch}, AUC: {best_auc:.4f}")
        print("=" * 80)

        return best_auc, best_epoch

    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('ResNet-50: Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # AUC curve
        ax2.plot(epochs, self.val_aucs, 'g-', label='Val Mean AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title('ResNet-50: Validation Mean AUC')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, 'training_curves_resnet50.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Training curves saved to: {save_path}")
        plt.close()


def main():
    print("=" * 80)
    print("CheXpert ResNet-50 Training")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        TRAIN_CSV, VAL_CSV, TEST_CSV,
        batch_size=16,
        num_workers=0
    )

    # Create model
    print("\nCreating ResNet-50 model...")
    model = ResNetCheXpert(num_classes=14, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {total_params} parameters")
    print(f"     (ResNet-50 has ~4x more parameters than DenseNet-121)")

    # Create trainer
    trainer = CheXpertTrainer(model, device, learning_rate=1e-4)

    # Train
    print("\nStarting training...")
    best_auc, best_epoch = trainer.train(
        train_loader, val_loader,
        num_epochs=10
    )

    # Plot results
    print("\nGenerating training curves...")
    trainer.plot_training_curves()

    # Save training history
    history = {
        'model': 'ResNet-50',
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_aucs': trainer.val_aucs,
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'total_parameters': total_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    history_path = os.path.join(RESULTS_DIR, 'training_history_resnet50.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Training history saved to: {history_path}")

    print("\n" + "=" * 80)
    print("All done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
