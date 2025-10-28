"""
Generate standalone ResNet-50 test results table
"""

import json
import os

# Paths
RESNET_METRICS = r"E:\download\BN5212_project\BN5212\results\evaluation_resnet50\evaluation_metrics.json"
OUTPUT_FILE = r"E:\download\BN5212_project\BN5212\results\resnet50_test_results.txt"

# CheXpert classes
CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

def generate_results_table():
    """Generate ResNet-50 test results table"""

    # Load metrics
    with open(RESNET_METRICS, 'r') as f:
        metrics = json.load(f)

    lines = []
    lines.append("=" * 100)
    lines.append("ResNet-50 Test Set Results")
    lines.append("=" * 100)
    lines.append("")

    # Overall metrics
    lines.append("OVERALL METRICS")
    lines.append("-" * 100)
    lines.append(f"Mean AUC:       {metrics['overall_metrics']['mean_auc']:.4f}")
    lines.append(f"Mean Accuracy:  {metrics['overall_metrics']['mean_accuracy']:.4f}")
    lines.append(f"Mean Precision: {metrics['overall_metrics']['mean_precision']:.4f}")
    lines.append(f"Mean Recall:    {metrics['overall_metrics']['mean_recall']:.4f}")
    lines.append(f"Mean F1-Score:  {metrics['overall_metrics']['mean_f1']:.4f}")
    lines.append("")

    # Per-class metrics table
    lines.append("PER-CLASS METRICS")
    lines.append("-" * 100)
    lines.append(f"{'Pathology':<35} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    lines.append("-" * 100)

    for cls in CLASSES:
        cls_metrics = metrics['per_class_metrics'][cls]
        lines.append(
            f"{cls:<35} "
            f"{cls_metrics['auc']:<10.4f} "
            f"{cls_metrics['accuracy']:<10.4f} "
            f"{cls_metrics['precision']:<10.4f} "
            f"{cls_metrics['recall']:<10.4f} "
            f"{cls_metrics['f1']:<10.4f}"
        )

    lines.append("")
    lines.append("=" * 100)

    # Print to console
    output = '\n'.join(lines)
    print(output)

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(output)

    print(f"\n[OK] Results saved to: {OUTPUT_FILE}")

    # Also create a simple CSV version
    csv_output = OUTPUT_FILE.replace('.txt', '.csv')
    with open(csv_output, 'w') as f:
        f.write("Pathology,AUC,Accuracy,Precision,Recall,F1-Score\n")
        for cls in CLASSES:
            cls_metrics = metrics['per_class_metrics'][cls]
            f.write(f"{cls},{cls_metrics['auc']:.4f},{cls_metrics['accuracy']:.4f},"
                   f"{cls_metrics['precision']:.4f},{cls_metrics['recall']:.4f},"
                   f"{cls_metrics['f1']:.4f}\n")

    print(f"[OK] CSV version saved to: {csv_output}")

if __name__ == "__main__":
    generate_results_table()
