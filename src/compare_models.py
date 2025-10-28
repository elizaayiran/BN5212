"""
Model Comparison: DenseNet-121 vs ResNet-50
Generate comparison visualizations and analysis
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
RESULTS_DIR = r"E:\download\BN5212_project\BN5212\results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]


def load_history(model_name):
    """Load training history for a model"""
    history_file = os.path.join(RESULTS_DIR, f'training_history_{model_name.lower().replace("-", "")}.json')

    if not os.path.exists(history_file):
        print(f"[WARN] History file not found: {history_file}")
        return None

    with open(history_file, 'r') as f:
        return json.load(f)


def plot_loss_comparison():
    """Compare training and validation loss curves"""

    densenet_history = load_history('densenet121')
    resnet_history = load_history('resnet50')

    if not densenet_history or not resnet_history:
        print("[ERROR] Cannot compare - missing history files")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training loss comparison
    epochs_d = range(1, len(densenet_history['train_losses']) + 1)
    epochs_r = range(1, len(resnet_history['train_losses']) + 1)

    ax1.plot(epochs_d, densenet_history['train_losses'], 'b-', label='DenseNet-121', linewidth=2)
    ax1.plot(epochs_r, resnet_history['train_losses'], 'r-', label='ResNet-50', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Validation loss comparison
    ax2.plot(epochs_d, densenet_history['val_losses'], 'b-', label='DenseNet-121', linewidth=2)
    ax2.plot(epochs_r, resnet_history['val_losses'], 'r-', label='ResNet-50', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'loss_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Loss comparison saved: {save_path}")
    plt.close()


def plot_auc_comparison():
    """Compare validation AUC curves"""

    densenet_history = load_history('densenet121')
    resnet_history = load_history('resnet50')

    if not densenet_history or not resnet_history:
        print("[ERROR] Cannot compare - missing history files")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    epochs_d = range(1, len(densenet_history['val_aucs']) + 1)
    epochs_r = range(1, len(resnet_history['val_aucs']) + 1)

    ax.plot(epochs_d, densenet_history['val_aucs'], 'b-o', label='DenseNet-121', linewidth=2, markersize=6)
    ax.plot(epochs_r, resnet_history['val_aucs'], 'r-s', label='ResNet-50', linewidth=2, markersize=6)

    # Mark best AUC for each model
    best_auc_d = max(densenet_history['val_aucs'])
    best_auc_r = max(resnet_history['val_aucs'])
    best_epoch_d = densenet_history['val_aucs'].index(best_auc_d) + 1
    best_epoch_r = resnet_history['val_aucs'].index(best_auc_r) + 1

    ax.annotate(f'Best: {best_auc_d:.4f}',
               xy=(best_epoch_d, best_auc_d),
               xytext=(best_epoch_d, best_auc_d + 0.02),
               ha='center',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))

    ax.annotate(f'Best: {best_auc_r:.4f}',
               xy=(best_epoch_r, best_auc_r),
               xytext=(best_epoch_r, best_auc_r - 0.02),
               ha='center',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean AUC', fontsize=12)
    ax.set_title('Validation Mean AUC Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'auc_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] AUC comparison saved: {save_path}")
    plt.close()


def generate_comparison_table():
    """Generate comparison table"""

    densenet_history = load_history('densenet121')
    resnet_history = load_history('resnet50')

    if not densenet_history or not resnet_history:
        print("[ERROR] Cannot compare - missing history files")
        return

    comparison_data = {
        'Metric': [
            'Parameters',
            'Best Validation AUC',
            'Best Epoch',
            'Final Train Loss',
            'Final Val Loss',
            'Convergence Speed',
            'Parameter Efficiency'
        ],
        'DenseNet-121': [
            f"{densenet_history.get('total_parameters', 6968206):,}",
            f"{densenet_history['best_auc']:.4f}",
            f"{densenet_history['best_epoch']}",
            f"{densenet_history['train_losses'][-1]:.4f}",
            f"{densenet_history['val_losses'][-1]:.4f}",
            "Fast" if densenet_history['best_epoch'] <= 5 else "Moderate",
            "High (7M params)"
        ],
        'ResNet-50': [
            f"{resnet_history.get('total_parameters', 25557032):,}",
            f"{resnet_history['best_auc']:.4f}",
            f"{resnet_history['best_epoch']}",
            f"{resnet_history['train_losses'][-1]:.4f}",
            f"{resnet_history['val_losses'][-1]:.4f}",
            "Fast" if resnet_history['best_epoch'] <= 5 else "Moderate",
            "Moderate (25M params)"
        ]
    }

    df = pd.DataFrame(comparison_data)

    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"[OK] Comparison table saved: {csv_path}")

    # Print table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


def plot_parameter_vs_performance():
    """Plot parameter efficiency (parameters vs AUC)"""

    densenet_history = load_history('densenet121')
    resnet_history = load_history('resnet50')

    if not densenet_history or not resnet_history:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['DenseNet-121', 'ResNet-50']
    params = [
        densenet_history.get('total_parameters', 6968206) / 1e6,
        resnet_history.get('total_parameters', 25557032) / 1e6
    ]
    aucs = [
        densenet_history['best_auc'],
        resnet_history['best_auc']
    ]
    colors = ['blue', 'red']

    scatter = ax.scatter(params, aucs, s=300, c=colors, alpha=0.6, edgecolors='black', linewidth=2)

    for i, model in enumerate(models):
        ax.annotate(model,
                   xy=(params[i], aucs[i]),
                   xytext=(params[i] + 1, aucs[i]),
                   fontsize=11,
                   fontweight='bold')

    ax.set_xlabel('Parameters (Millions)', fontsize=12)
    ax.set_ylabel('Best Validation AUC', fontsize=12)
    ax.set_title('Parameter Efficiency: Parameters vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'parameter_efficiency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Parameter efficiency plot saved: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("DenseNet-121 vs ResNet-50 Comparison")
    print("=" * 80)

    # Generate comparisons
    plot_loss_comparison()
    plot_auc_comparison()
    plot_parameter_vs_performance()
    generate_comparison_table()

    print("\n" + "=" * 80)
    print("[SUCCESS] Model comparison complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
