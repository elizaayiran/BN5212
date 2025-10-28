"""
Grad-CAM Visualization for CheXpert DenseNet-121
Generates heatmaps showing where the model focuses for making predictions
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pydicom
from torchvision import transforms
from scipy import ndimage

# Import our dataset
from dicom_dataset import CheXpertDICOMDataset

# Paths
DATASET_DIR = r"E:\download\dataset"
MODEL_PATH = r"E:\download\BN5212_project\BN5212\models\best_model.pth"
OUTPUT_DIR = r"E:\download\BN5212_project\BN5212\results\gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]


class DenseNetCheXpert(nn.Module):
    """DenseNet-121 for CheXpert with Grad-CAM support"""

    def __init__(self, num_classes=14):
        super(DenseNetCheXpert, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

        # Store features and gradients for Grad-CAM
        self.features = None
        self.gradients = None

    def forward(self, x):
        # Get features from the last conv layer
        features = self.densenet.features(x)

        # Register hook to capture gradients
        if features.requires_grad:
            h = features.register_hook(self.save_gradients)

        self.features = features

        # Continue forward pass
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)

        return out

    def save_gradients(self, grad):
        """Hook to save gradients"""
        self.gradients = grad


class GradCAM:
    """Grad-CAM implementation"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def generate_cam(self, image, class_idx):
        """
        Generate Grad-CAM for a specific class

        Args:
            image: Input image tensor [1, 3, 224, 224]
            class_idx: Index of the class to visualize

        Returns:
            cam: Grad-CAM heatmap
        """
        # Forward pass
        image = image.to(self.device)
        output = self.model(image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        class_score = output[:, class_idx]
        class_score.backward()

        # Get gradients and features
        gradients = self.model.gradients  # [1, 1024, 7, 7]
        features = self.model.features     # [1, 1024, 7, 7]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, 1024, 1, 1]

        # Weighted combination of features
        cam = torch.sum(weights * features, dim=1, keepdim=True)  # [1, 1, 7, 7]
        cam = nn.functional.relu(cam)  # ReLU to keep positive influences

        # Normalize
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to match input image size
        cam = cv2.resize(cam, (224, 224))

        return cam

    def visualize(self, image, cam, original_image, title="Grad-CAM"):
        """
        Visualize Grad-CAM heatmap overlayed on original image

        Args:
            image: Preprocessed image tensor
            cam: Grad-CAM heatmap
            original_image: Original PIL image
            title: Plot title

        Returns:
            fig: matplotlib figure
        """
        # Convert preprocessed image back to displayable format
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap / 255.0

        # Overlay heatmap on image
        overlayed = 0.6 * img_np + 0.4 * heatmap
        overlayed = np.clip(overlayed, 0, 1)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


def load_model(model_path, device='cuda'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")

    model = DenseNetCheXpert(num_classes=14)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded (AUC: {checkpoint['mean_auc']:.4f})")
    return model


def generate_gradcam_samples(num_samples=10):
    """Generate Grad-CAM visualizations for sample images"""

    print("=" * 80)
    print("Generating Grad-CAM Visualizations")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        print("Please train the model first!")
        return

    model = load_model(MODEL_PATH, device)

    # Create Grad-CAM object
    grad_cam = GradCAM(model, device)

    # Load validation dataset
    val_csv = os.path.join(DATASET_DIR, "validate.csv")
    dataset = CheXpertDICOMDataset(val_csv)

    print(f"\nGenerating {num_samples} Grad-CAM visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Generate visualizations for random samples
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        labels = sample['labels']
        subject_id = sample['subject_id']
        study_id = sample['study_id']

        print(f"\n[{i+1}/{num_samples}] Processing {subject_id}/{study_id}")

        # Get predictions
        with torch.no_grad():
            output = model(image.to(device))
            predictions = torch.sigmoid(output).cpu().numpy()[0]

        # Find positive labels
        positive_labels = [(idx, CHEXPERT_LABELS[idx], predictions[idx])
                          for idx in range(14)
                          if labels[idx] == 1.0 and predictions[idx] > 0.3]

        if len(positive_labels) == 0:
            print("  No confident positive predictions, skipping...")
            continue

        # Generate Grad-CAM for top predicted label
        top_label = max(positive_labels, key=lambda x: x[2])
        class_idx, class_name, confidence = top_label

        print(f"  Top prediction: {class_name} (confidence: {confidence:.3f})")

        # Generate CAM
        cam = grad_cam.generate_cam(image, class_idx)

        # Visualize
        title = f"{class_name} (Confidence: {confidence:.3f})\n{subject_id}/{study_id}"
        fig = grad_cam.visualize(image[0], cam, None, title)

        # Save
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{i+1}_{class_name.replace(' ', '_')}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  [OK] Saved to {save_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Grad-CAM generation complete!")
    print(f"Generated {num_samples} visualizations in {OUTPUT_DIR}")
    print("=" * 80)


def generate_multi_class_comparison(sample_idx=0):
    """Generate comparison of Grad-CAM for multiple classes on same image"""

    print("\n" + "=" * 80)
    print("Generating Multi-Class Grad-CAM Comparison")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH, device)
    grad_cam = GradCAM(model, device)

    # Load sample
    val_csv = os.path.join(DATASET_DIR, "validate.csv")
    dataset = CheXpertDICOMDataset(val_csv)
    sample = dataset[sample_idx]

    image = sample['image'].unsqueeze(0)
    labels = sample['labels']

    # Get predictions
    with torch.no_grad():
        output = model(image.to(device))
        predictions = torch.sigmoid(output).cpu().numpy()[0]

    # Get top 4 predictions
    top_indices = np.argsort(predictions)[::-1][:4]

    # Generate CAM for each
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, class_idx in enumerate(top_indices):
        cam = grad_cam.generate_cam(image, class_idx)

        # Overlay
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        overlayed = 0.6 * img_np + 0.4 * heatmap
        overlayed = np.clip(overlayed, 0, 1)

        axes[idx].imshow(overlayed)
        axes[idx].set_title(f"{CHEXPERT_LABELS[class_idx]}\nConf: {predictions[class_idx]:.3f}",
                           fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('Multi-Class Grad-CAM Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'gradcam_multi_class_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Multi-class comparison saved to {save_path}")


if __name__ == "__main__":
    # Generate Grad-CAM visualizations
    generate_gradcam_samples(num_samples=10)

    # Generate multi-class comparison
    generate_multi_class_comparison()

    print("\n[OK] All visualizations complete!")
