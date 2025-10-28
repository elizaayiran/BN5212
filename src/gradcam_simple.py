"""
Simplified Grad-CAM Visualization for CheXpert DenseNet-121
No cv2 dependency - uses matplotlib and PIL only
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import os
import pandas as pd

# Paths
DATASET_DIR = r"E:\download\dataset"
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
MODEL_PATH = r"E:\download\BN5212_project\BN5212\models\best_model.pth"
OUTPUT_DIR = r"E:\download\BN5212_project\BN5212\results\gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

class DenseNetCheXpert(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNetCheXpert, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, class_idx):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Calculate weights
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

def load_dicom_image(dicom_path):
    """Load and preprocess DICOM image"""
    import pydicom

    # Read DICOM
    ds = pydicom.dcmread(dicom_path)
    image_array = ds.pixel_array.astype(float)

    # Normalize to 0-255
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8) * 255.0
    image = Image.fromarray(image_array.astype(np.uint8))
    image = image.convert('RGB')
    image = image.resize((224, 224), Image.BILINEAR)

    return image

def apply_colormap(cam, colormap='jet'):
    """Apply colormap to CAM using matplotlib"""
    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Apply colormap
    colored_cam = cmap(cam)[:, :, :3]  # Remove alpha channel

    return colored_cam

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    # Convert PIL image to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0

    # Resize heatmap to match image size if needed
    if heatmap.shape[:2] != img_array.shape[:2]:
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((img_array.shape[1], img_array.shape[0]), Image.BILINEAR)
        heatmap = np.array(heatmap_pil).astype(np.float32) / 255.0

    # Overlay
    superimposed = heatmap * alpha + img_array * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)

    return superimposed

def generate_gradcam_samples(num_samples=5):
    """Generate Grad-CAM visualizations for sample images"""
    print("=" * 80)
    print("Generating Grad-CAM Visualizations")
    print("=" * 80)

    # Load test data
    print("\nLoading test dataset...")
    df_test = pd.read_csv(TEST_CSV)
    print(f"[OK] Loaded {len(df_test)} test images")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseNetCheXpert(num_classes=14, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("[OK] Model loaded")

    # Get target layer (last conv layer in DenseNet)
    target_layer = model.densenet.features[-1]

    # Create Grad-CAM instance
    gradcam = SimpleGradCAM(model, target_layer)

    # Select diverse samples with positive labels
    print("\nSelecting sample images...")
    selected_samples = []

    # Try to get one sample for each major pathology
    target_classes = ["Cardiomegaly", "Pneumothorax", "Atelectasis", "Edema", "Support Devices"]

    for target_class in target_classes:
        if target_class in df_test.columns:
            positive_samples = df_test[df_test[target_class] == 1.0]
            if len(positive_samples) > 0:
                selected_samples.append((positive_samples.iloc[0], target_class))
                if len(selected_samples) >= num_samples:
                    break

    print(f"[OK] Selected {len(selected_samples)} samples")

    # Generate visualizations
    print("\nGenerating Grad-CAM heatmaps...")

    fig, axes = plt.subplots(len(selected_samples), 3, figsize=(15, 5 * len(selected_samples)))
    if len(selected_samples) == 1:
        axes = axes.reshape(1, -1)

    for idx, (sample, pathology) in enumerate(selected_samples):
        print(f"Processing sample {idx+1}/{len(selected_samples)}: {pathology}...")

        # Load image
        dicom_path = sample['dicom_path']
        image = load_dicom_image(dicom_path)

        # Prepare input tensor
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Generate CAM
        class_idx = CHEXPERT_LABELS.index(pathology)
        cam = gradcam.generate_cam(input_tensor, class_idx)

        # Resize CAM to match image size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0

        # Apply colormap
        heatmap = apply_colormap(cam_resized, colormap='jet')

        # Overlay
        overlay = overlay_heatmap(image, heatmap, alpha=0.4)

        # Plot
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Original\n{pathology}', fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title(f'Grad-CAM Heatmap', fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Overlay', fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'gradcam_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Grad-CAM visualizations saved to: {output_path}")
    print("=" * 80)
    print("[SUCCESS] Grad-CAM generation complete!")
    print("=" * 80)

if __name__ == "__main__":
    generate_gradcam_samples(num_samples=5)
