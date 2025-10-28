"""
Run Grad-CAM visualization with proper environment setup
"""
import os
import sys

# Set environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run
from gradcam_visualization import generate_gradcam_samples

if __name__ == '__main__':
    generate_gradcam_samples(num_samples=10)
