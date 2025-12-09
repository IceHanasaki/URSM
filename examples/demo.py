"""
Simple demo script showing basic usage of URSM.

This script demonstrates:
1. Loading a pre-trained model
2. Running inference on a stereo pair
3. Visualizing results
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from ursm.models import URSMNet
from ursm.utils import visualize_disparity, visualize_calibration


def create_sample_stereo_pair():
    """
    Create a simple synthetic stereo pair for demonstration.
    In practice, you would load real images.
    """
    # Create a simple pattern with depth variation
    height, width = 480, 640
    
    # Left image: gradient pattern
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        left_img[i, :, :] = int(255 * i / height)
    
    # Right image: shifted version (simulating disparity)
    right_img = np.zeros_like(left_img)
    shift = 20  # pixels
    right_img[:, shift:, :] = left_img[:, :-shift, :]
    
    return left_img, right_img


def prepare_images(left_img, right_img):
    """
    Convert images to tensors for model input.
    
    Args:
        left_img: numpy array [H, W, 3] in [0, 255]
        right_img: numpy array [H, W, 3] in [0, 255]
    
    Returns:
        Tuple of tensors [1, 3, H, W] in [0, 1]
    """
    # Normalize to [0, 1]
    left = torch.from_numpy(left_img).float() / 255.0
    right = torch.from_numpy(right_img).float() / 255.0
    
    # Permute to [C, H, W] and add batch dimension
    left = left.permute(2, 0, 1).unsqueeze(0)
    right = right.permute(2, 0, 1).unsqueeze(0)
    
    return left, right


def main():
    """Main demo function."""
    print("="*60)
    print("URSM Demo - Unrectified Stereo Matching")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating URSM model...")
    model = URSMNet(
        max_disparity=96,  # Smaller for demo
        feature_channels=16,  # Smaller for demo
        refine_iterations=2,  # Fewer iterations for demo
        estimate_calibration=True
    )
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model created with {num_params:.2f}M parameters")
    
    # Create sample stereo pair
    print("\nCreating sample stereo pair...")
    left_img, right_img = create_sample_stereo_pair()
    print(f"Image size: {left_img.shape[0]}x{left_img.shape[1]}")
    
    # Prepare images
    left_tensor, right_tensor = prepare_images(left_img, right_img)
    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(left_tensor, right_tensor)
    
    # Extract outputs
    disparity = outputs['disparity']
    uncertainty = outputs['uncertainty']
    calibration = outputs.get('calibration', None)
    
    print(f"Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")
    
    # Print calibration if available
    if calibration is not None:
        print("\nEstimated calibration parameters:")
        v_offset = calibration['vertical_offset'][0, 0].item()
        rotation = calibration['rotation'][0].cpu().numpy()
        scale = calibration['scale'][0, 0].item()
        
        print(f"  Vertical offset: {v_offset:.2f} pixels")
        print(f"  Rotation (R,P,Y): [{rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}] degrees")
        print(f"  Scale factor: {scale:.4f}")
    
    # Save visualizations
    print("\nSaving visualizations...")
    os.makedirs('demo_output', exist_ok=True)
    
    # Save input images
    Image.fromarray(left_img).save('demo_output/left.png')
    Image.fromarray(right_img).save('demo_output/right.png')
    print("  Saved: demo_output/left.png")
    print("  Saved: demo_output/right.png")
    
    # Save disparity
    visualize_disparity(
        disparity[0],
        save_path='demo_output/disparity.png',
        title='Predicted Disparity'
    )
    print("  Saved: demo_output/disparity.png")
    
    # Save uncertainty
    visualize_disparity(
        uncertainty[0],
        save_path='demo_output/uncertainty.png',
        title='Uncertainty Map',
        cmap='viridis'
    )
    print("  Saved: demo_output/uncertainty.png")
    
    # Save calibration
    if calibration is not None:
        visualize_calibration(
            calibration,
            save_path='demo_output/calibration.png'
        )
        print("  Saved: demo_output/calibration.png")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("Check the 'demo_output' directory for results.")
    print("="*60)


if __name__ == '__main__':
    main()
