"""
Complete Evaluation Pipeline
Run this after training completes to generate all evaluation results
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("CheXpert Complete Evaluation Pipeline")
print("=" * 80)

# Check if model exists
MODEL_PATH = r"E:\download\BN5212_project\BN5212\models\best_model.pth"
if not os.path.exists(MODEL_PATH):
    print("\n[ERROR] Trained model not found!")
    print(f"Expected location: {MODEL_PATH}")
    print("\nPlease complete training first by running:")
    print("  python src/train_chexpert_densenet.py")
    sys.exit(1)

print("\n[OK] Model found, proceeding with evaluation...")

# Step 1: Model Evaluation
print("\n" + "=" * 80)
print("Step 1: Running Model Evaluation")
print("=" * 80)

try:
    from src import evaluate_model
    evaluate_model.main()
    print("\n[OK] Model evaluation complete!")
except Exception as e:
    print(f"\n[ERROR] Model evaluation failed: {e}")
    print("Continuing with Grad-CAM generation...")

# Step 2: Grad-CAM Visualization
print("\n" + "=" * 80)
print("Step 2: Generating Grad-CAM Visualizations")
print("=" * 80)

try:
    from src import gradcam_visualization
    gradcam_visualization.generate_gradcam_samples(num_samples=10)
    gradcam_visualization.generate_multi_class_comparison()
    print("\n[OK] Grad-CAM generation complete!")
except Exception as e:
    print(f"\n[ERROR] Grad-CAM generation failed: {e}")

# Summary
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print("\nGenerated Results:")
print("\n1. Model Evaluation:")
print("   Location: E:\\download\\BN5212_project\\BN5212\\results\\evaluation\\")
print("   Files:")
print("     - evaluation_metrics.json    (Detailed metrics)")
print("     - roc_curves.png             (ROC curves for all classes)")
print("     - confusion_matrices.png     (Confusion matrices)")
print("     - metrics_comparison.png     (Metrics comparison)")

print("\n2. Grad-CAM Visualizations:")
print("   Location: E:\\download\\BN5212_project\\BN5212\\results\\gradcam\\")
print("   Files:")
print("     - gradcam_1_*.png to gradcam_10_*.png  (Individual samples)")
print("     - gradcam_multi_class_comparison.png   (Multi-class comparison)")

print("\n3. Training History:")
print("   Location: E:\\download\\BN5212_project\\BN5212\\results\\")
print("   Files:")
print("     - training_curves.png        (Training/validation curves)")
print("     - training_history.json      (Training history)")

print("\n" + "=" * 80)
print("All evaluation results are ready for your presentation!")
print("=" * 80)
