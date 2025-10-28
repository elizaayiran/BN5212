"""
Simple script to check training progress
Run this anytime to see the current status
"""

import os
import json
from pathlib import Path

# Paths
RESULTS_DIR = r"E:\download\BN5212_project\BN5212\results"
MODEL_DIR = r"E:\download\BN5212_project\BN5212\models"

print("=" * 80)
print("CheXpert Training Status Check")
print("=" * 80)

# Check if training has started
history_file = os.path.join(RESULTS_DIR, "training_history.json")
model_file = os.path.join(MODEL_DIR, "best_model.pth")

if os.path.exists(history_file):
    print("\n[OK] Training history found!")

    with open(history_file, 'r') as f:
        history = json.load(f)

    print(f"\nTraining Summary:")
    print(f"  Completed Epochs: {len(history['train_losses'])}")
    print(f"  Best AUC:        {history['best_auc']:.4f}")
    print(f"  Best Epoch:      {history['best_epoch']}")
    print(f"  Timestamp:       {history['timestamp']}")

    print(f"\nLoss History:")
    for i, (train_loss, val_loss) in enumerate(zip(history['train_losses'], history['val_losses']), 1):
        print(f"  Epoch {i}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, AUC={history['val_aucs'][i-1]:.4f}")

else:
    print("\n[INFO] Training in progress or not yet started")
    print("       Training history will be saved after completion")

# Check for best model
if os.path.exists(model_file):
    print(f"\n[OK] Best model checkpoint found at:")
    print(f"     {model_file}")

    # Get file size
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"     Size: {size_mb:.1f} MB")
else:
    print(f"\n[INFO] No model checkpoint yet")
    print("       Will be saved after first epoch")

# Check for training curves
curves_file = os.path.join(RESULTS_DIR, "training_curves.png")
if os.path.exists(curves_file):
    print(f"\n[OK] Training curves available at:")
    print(f"     {curves_file}")
else:
    print(f"\n[INFO] Training curves not yet generated")
    print("       Will be created after training completes")

print("\n" + "=" * 80)
print("To view real-time progress, the training script shows:")
print("  - Progress bar with batch number")
print("  - Current loss value")
print("  - Estimated time remaining")
print("=" * 80)
