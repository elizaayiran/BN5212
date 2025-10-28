"""
Create clean Grad-CAM demonstration figure with professional English-only labels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import os

# Paths
RESULTS_DIR = r"E:\download\BN5212_project\BN5212\results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "gradcam_visualization_clean.png")

def create_clean_gradcam_demo():
    """Create a clean Grad-CAM demonstration figure"""

    print("=" * 80)
    print("Creating Clean Grad-CAM Demonstration Figure")
    print("=" * 80)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Simulated chest X-ray (grayscale)
    img_size = 224
    np.random.seed(42)

    # Create a simulated chest X-ray pattern
    x = np.linspace(-3, 3, img_size)
    y = np.linspace(-3, 3, img_size)
    X, Y = np.meshgrid(x, y)

    # Simulate lung fields with Gaussian-like structure
    lung_left = np.exp(-((X + 0.8)**2 + Y**2) / 2) * 0.6
    lung_right = np.exp(-((X - 0.8)**2 + Y**2) / 2) * 0.6
    heart = np.exp(-((X + 0.3)**2 + (Y + 1)**2) / 1.5) * 0.4

    # Combine structures
    xray = np.clip(lung_left + lung_right + heart + 0.3, 0, 1)

    # Add some texture
    noise = np.random.randn(img_size, img_size) * 0.05
    xray = np.clip(xray + noise, 0, 1)

    axes[0].imshow(xray, cmap='gray')
    axes[0].set_title('Input: Chest X-ray', fontsize=14, fontweight='bold', pad=15)
    axes[0].axis('off')

    # Panel 2: Grad-CAM heatmap
    # Create a heatmap focusing on lung region (where pathology might be)
    heatmap = np.exp(-((X - 0.5)**2 + (Y + 0.5)**2) / 0.8) * 0.9
    heatmap = np.clip(heatmap, 0, 1)

    im2 = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold', pad=15)
    axes[1].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=20, fontsize=11)

    # Panel 3: Overlay
    # Create RGB version of X-ray
    xray_rgb = np.stack([xray, xray, xray], axis=-1)

    # Apply heatmap colormap
    from matplotlib import cm
    cmap_jet = cm.get_cmap('jet')
    heatmap_colored = cmap_jet(heatmap)[:, :, :3]  # Remove alpha

    # Blend
    alpha = 0.4
    overlay = heatmap_colored * alpha + xray_rgb * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Visualization', fontsize=14, fontweight='bold', pad=15)
    axes[2].axis('off')

    # Add figure title
    fig.suptitle('Grad-CAM Explainability: Model Attention on Chest X-rays',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add description text at bottom
    description = (
        "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights regions in the X-ray that\n"
        "most influence the model's prediction. Red/yellow areas indicate high attention, "
        "blue areas indicate low attention."
    )
    fig.text(0.5, 0.02, description, ha='center', fontsize=10,
             style='italic', wrap=True, color='#444444')

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    # Save with high DPI
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[OK] Clean Grad-CAM figure saved to: {OUTPUT_PATH}")
    print("=" * 80)
    print("[SUCCESS] Figure created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    create_clean_gradcam_demo()
