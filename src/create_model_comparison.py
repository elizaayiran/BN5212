"""
Create comprehensive model comparison visualizations and summary
DenseNet-121 vs ResNet-50
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
BASE_DIR = r"E:\download\BN5212_project\BN5212\results"
DENSENET_METRICS = os.path.join(BASE_DIR, "evaluation", "evaluation_metrics.json")
RESNET_METRICS = os.path.join(BASE_DIR, "evaluation_resnet50", "evaluation_metrics.json")
DENSENET_HISTORY = os.path.join(BASE_DIR, "training_history.json")
RESNET_HISTORY = os.path.join(BASE_DIR, "training_history_resnet50.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CheXpert classes
CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

def load_metrics():
    """Load evaluation metrics for both models"""
    with open(DENSENET_METRICS, 'r') as f:
        densenet = json.load(f)
    with open(RESNET_METRICS, 'r') as f:
        resnet = json.load(f)
    return densenet, resnet

def load_training_history():
    """Load training history for both models"""
    with open(DENSENET_HISTORY, 'r') as f:
        densenet = json.load(f)
    with open(RESNET_HISTORY, 'r') as f:
        resnet = json.load(f)
    return densenet, resnet

def create_per_class_comparison(densenet_metrics, resnet_metrics):
    """Create per-class AUC comparison bar chart"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract per-class AUCs
    densenet_aucs = [densenet_metrics['per_class_metrics'][cls]['auc'] for cls in CLASSES]
    resnet_aucs = [resnet_metrics['per_class_metrics'][cls]['auc'] for cls in CLASSES]

    x = np.arange(len(CLASSES))
    width = 0.35

    bars1 = ax.bar(x - width/2, densenet_aucs, width, label='DenseNet-121', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, resnet_aucs, width, label='ResNet-50', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Pathology Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class AUC Comparison: DenseNet-121 vs ResNet-50', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only show label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "per_class_auc_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Per-class comparison saved to {output_path}")

def create_overall_metrics_comparison(densenet_metrics, resnet_metrics):
    """Create overall metrics comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['Mean AUC', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1']
    densenet_values = [
        densenet_metrics['overall_metrics']['mean_auc'],
        densenet_metrics['overall_metrics']['mean_accuracy'],
        densenet_metrics['overall_metrics']['mean_precision'],
        densenet_metrics['overall_metrics']['mean_recall'],
        densenet_metrics['overall_metrics']['mean_f1']
    ]
    resnet_values = [
        resnet_metrics['overall_metrics']['mean_auc'],
        resnet_metrics['overall_metrics']['mean_accuracy'],
        resnet_metrics['overall_metrics']['mean_precision'],
        resnet_metrics['overall_metrics']['mean_recall'],
        resnet_metrics['overall_metrics']['mean_f1']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, densenet_values, width, label='DenseNet-121', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, resnet_values, width, label='ResNet-50', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "overall_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Overall metrics comparison saved to {output_path}")

def create_training_curves_comparison(densenet_history, resnet_history):
    """Create training curves comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Validation AUC curves
    epochs_densenet = range(1, len(densenet_history['val_aucs']) + 1)
    epochs_resnet = range(1, len(resnet_history['val_aucs']) + 1)

    ax1.plot(epochs_densenet, densenet_history['val_aucs'], 'o-',
             label='DenseNet-121', color='#3498db', linewidth=2, markersize=6)
    ax1.plot(epochs_resnet, resnet_history['val_aucs'], 's-',
             label='ResNet-50', color='#e74c3c', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Validation AUC During Training', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')

    # Mark best epochs
    ax1.axvline(x=densenet_history['best_epoch'], color='#3498db', linestyle=':', alpha=0.5)
    ax1.axvline(x=resnet_history['best_epoch'], color='#e74c3c', linestyle=':', alpha=0.5)

    # Validation Loss curves
    ax2.plot(epochs_densenet, densenet_history.get('val_losses', []), 'o-',
             label='DenseNet-121', color='#3498db', linewidth=2, markersize=6)
    ax2.plot(epochs_resnet, resnet_history.get('val_losses', []), 's-',
             label='ResNet-50', color='#e74c3c', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss During Training', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "training_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Training curves comparison saved to {output_path}")

def generate_summary_report(densenet_metrics, resnet_metrics, densenet_history, resnet_history):
    """Generate text summary report"""
    summary = []
    summary.append("=" * 80)
    summary.append("MODEL COMPARISON SUMMARY: DenseNet-121 vs ResNet-50")
    summary.append("=" * 80)
    summary.append("")

    # Model specifications
    summary.append("MODEL SPECIFICATIONS")
    summary.append("-" * 80)
    summary.append(f"DenseNet-121:")
    summary.append(f"  - Parameters: ~7M")
    summary.append(f"  - Architecture: Dense connectivity with feature reuse")
    summary.append(f"  - Best Validation AUC: {densenet_history['best_auc']:.4f} (Epoch {densenet_history['best_epoch']})")
    summary.append("")
    summary.append(f"ResNet-50:")
    summary.append(f"  - Parameters: {resnet_history['total_parameters']:,} (~23.5M)")
    summary.append(f"  - Architecture: Residual learning with skip connections")
    summary.append(f"  - Best Validation AUC: {resnet_history['best_auc']:.4f} (Epoch {resnet_history['best_epoch']})")
    summary.append("")

    # Test set performance
    summary.append("TEST SET PERFORMANCE")
    summary.append("-" * 80)
    summary.append(f"{'Metric':<20} {'DenseNet-121':<15} {'ResNet-50':<15} {'Difference':<15}")
    summary.append("-" * 80)

    for metric_key, metric_name in [
        ('mean_auc', 'Mean AUC'),
        ('mean_accuracy', 'Mean Accuracy'),
        ('mean_precision', 'Mean Precision'),
        ('mean_recall', 'Mean Recall'),
        ('mean_f1', 'Mean F1-Score')
    ]:
        d_val = densenet_metrics['overall_metrics'][metric_key]
        r_val = resnet_metrics['overall_metrics'][metric_key]
        diff = r_val - d_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        summary.append(f"{metric_name:<20} {d_val:<15.4f} {r_val:<15.4f} {diff_str:<15}")

    summary.append("")

    # Per-class top performers
    summary.append("TOP 5 PERFORMING CLASSES (by AUC)")
    summary.append("-" * 80)

    # DenseNet top 5
    densenet_class_aucs = [(cls, densenet_metrics['per_class_metrics'][cls]['auc']) for cls in CLASSES]
    densenet_class_aucs.sort(key=lambda x: x[1], reverse=True)

    summary.append("DenseNet-121 Top 5:")
    for i, (cls, auc) in enumerate(densenet_class_aucs[:5], 1):
        summary.append(f"  {i}. {cls:<30} AUC: {auc:.4f}")
    summary.append("")

    # ResNet top 5
    resnet_class_aucs = [(cls, resnet_metrics['per_class_metrics'][cls]['auc']) for cls in CLASSES]
    resnet_class_aucs.sort(key=lambda x: x[1], reverse=True)

    summary.append("ResNet-50 Top 5:")
    for i, (cls, auc) in enumerate(resnet_class_aucs[:5], 1):
        summary.append(f"  {i}. {cls:<30} AUC: {auc:.4f}")
    summary.append("")

    # Key findings
    summary.append("KEY FINDINGS")
    summary.append("-" * 80)
    auc_improvement = resnet_metrics['overall_metrics']['mean_auc'] - densenet_metrics['overall_metrics']['mean_auc']
    summary.append(f"1. ResNet-50 achieves {auc_improvement:.4f} higher mean AUC than DenseNet-121")
    summary.append(f"   ({resnet_metrics['overall_metrics']['mean_auc']:.4f} vs {densenet_metrics['overall_metrics']['mean_auc']:.4f})")
    summary.append("")
    summary.append("2. Both models perform best on:")
    summary.append(f"   - Support Devices (DenseNet: {densenet_metrics['per_class_metrics']['Support Devices']['auc']:.4f}, ResNet: {resnet_metrics['per_class_metrics']['Support Devices']['auc']:.4f})")
    summary.append("")
    summary.append("3. ResNet-50 uses 3.4x more parameters but achieves modest performance gain")
    summary.append("")
    summary.append("4. Training efficiency:")
    summary.append(f"   - DenseNet-121 best epoch: {densenet_history['best_epoch']}/10")
    summary.append(f"   - ResNet-50 best epoch: {resnet_history['best_epoch']}/10")
    summary.append("")

    summary.append("=" * 80)

    # Save to file
    output_path = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))

    # Also print to console
    print('\n'.join(summary))
    print(f"\n[OK] Summary report saved to {output_path}")

def main():
    print("=" * 80)
    print("Creating Model Comparison Visualizations")
    print("=" * 80)

    # Load data
    print("\nLoading metrics and training history...")
    densenet_metrics, resnet_metrics = load_metrics()
    densenet_history, resnet_history = load_training_history()
    print("[OK] Data loaded")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_per_class_comparison(densenet_metrics, resnet_metrics)
    create_overall_metrics_comparison(densenet_metrics, resnet_metrics)
    create_training_curves_comparison(densenet_history, resnet_history)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(densenet_metrics, resnet_metrics, densenet_history, resnet_history)

    print("\n" + "=" * 80)
    print("[SUCCESS] Model comparison complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
