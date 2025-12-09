"""
Demo script for Foundation Stereo with monocular priors.

This script demonstrates how Foundation Stereo leverages pre-trained
monocular depth estimation to improve stereo matching accuracy.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from ursm.models import FoundationStereo
from ursm.utils import visualize_disparity, create_comparison_plot


def create_sample_scene():
    """
    Create a synthetic scene with depth variation.
    """
    height, width = 480, 640
    
    # Create left image with multiple depth planes
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (far)
    left_img[:, :, :] = [100, 100, 150]
    
    # Middle ground objects
    left_img[150:300, 200:400, :] = [150, 200, 150]
    
    # Foreground object (close)
    left_img[200:280, 280:360, :] = [200, 150, 100]
    
    # Very close object
    left_img[100:180, 100:200, :] = [255, 200, 200]
    
    # Create right image with appropriate disparities
    right_img = np.zeros_like(left_img)
    
    # Background (disparity = 5)
    right_img[:, 5:, :] = left_img[:, :-5, :]
    
    # Override middle ground (disparity = 15)
    shift = 15
    right_img[150:300, (200-shift):(400-shift), :] = [150, 200, 150]
    
    # Override foreground (disparity = 30)
    shift = 30
    right_img[200:280, (280-shift):(360-shift), :] = [200, 150, 100]
    
    # Override very close (disparity = 50)
    shift = 50
    right_img[100:180, (100-shift):(200-shift), :] = [255, 200, 200]
    
    return left_img, right_img


def create_synthetic_monocular_depth(left_img):
    """
    Create synthetic monocular depth that roughly matches scene structure.
    
    In practice, this would come from a pre-trained model like DPT or MiDaS.
    """
    height, width = left_img.shape[:2]
    
    # Create depth map based on image content
    depth = np.ones((height, width), dtype=np.float32) * 10.0
    
    # Closer objects have smaller depth values
    depth[150:300, 200:400] = 5.0  # Middle ground
    depth[200:280, 280:360] = 2.5  # Foreground
    depth[100:180, 100:200] = 1.5  # Very close
    
    # Add some noise to make it realistic
    depth += np.random.randn(height, width) * 0.2
    depth = np.clip(depth, 0.5, 20.0)
    
    return depth


def prepare_images(left_img, right_img, depth=None):
    """Convert images to tensors for model input."""
    left = torch.from_numpy(left_img).float() / 255.0
    right = torch.from_numpy(right_img).float() / 255.0
    
    left = left.permute(2, 0, 1).unsqueeze(0)
    right = right.permute(2, 0, 1).unsqueeze(0)
    
    if depth is not None:
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        return left, right, depth_tensor
    
    return left, right


def main():
    """Main demo function."""
    print("="*70)
    print("Foundation Stereo Demo - Monocular Prior-Guided Stereo Matching")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating Foundation Stereo model...")
    print("Key features:")
    print("  ✓ Monocular depth priors from pre-trained models")
    print("  ✓ Prior-guided cost volume (focused search)")
    print("  ✓ Confidence-weighted fusion")
    print("  ✓ Multi-scale feature integration")
    
    model = FoundationStereo(
        max_disparity=96,
        feature_channels=16,
        use_monocular_prior=True,
        search_range=24
    )
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel created with {num_params:.2f}M parameters")
    
    # Create sample scene
    print("\nCreating sample scene with known depth structure...")
    left_img, right_img = create_sample_scene()
    print(f"Image size: {left_img.shape[0]}x{left_img.shape[1]}")
    print("Scene contains multiple depth planes:")
    print("  - Background (far, disparity ~5px)")
    print("  - Middle ground (mid, disparity ~15px)")
    print("  - Foreground (close, disparity ~30px)")
    print("  - Very close object (disparity ~50px)")
    
    # Create synthetic monocular depth
    print("\nGenerating synthetic monocular depth prior...")
    print("(In practice, this would come from DPT, MiDaS, or Depth Anything)")
    monocular_depth = create_synthetic_monocular_depth(left_img)
    
    # Prepare images
    left_tensor, right_tensor, depth_tensor = prepare_images(
        left_img, right_img, monocular_depth
    )
    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)
    depth_tensor = depth_tensor.to(device)
    
    # Run inference with monocular prior
    print("\nRunning inference WITH monocular prior...")
    with torch.no_grad():
        outputs_with_prior = model(
            left_tensor, right_tensor,
            monocular_depth=depth_tensor,
            baseline=0.1,
            focal_length=500.0
        )
    
    # Run inference without monocular prior (for comparison)
    print("Running inference WITHOUT monocular prior...")
    model.use_monocular_prior = False
    with torch.no_grad():
        outputs_without_prior = model(left_tensor, right_tensor)
    model.use_monocular_prior = True
    
    # Extract outputs
    disparity_with = outputs_with_prior['disparity']
    disparity_without = outputs_without_prior['disparity']
    disparity_prior = outputs_with_prior['disparity_prior']
    prior_confidence = outputs_with_prior['prior_confidence']
    
    print(f"\nResults:")
    print(f"  With prior - Disparity range: [{disparity_with.min():.2f}, {disparity_with.max():.2f}]")
    print(f"  Without prior - Disparity range: [{disparity_without.min():.2f}, {disparity_without.max():.2f}]")
    print(f"  Monocular prior - Disparity range: [{disparity_prior.min():.2f}, {disparity_prior.max():.2f}]")
    print(f"  Prior confidence: mean={prior_confidence.mean():.3f}, std={prior_confidence.std():.3f}")
    
    # Compute improvement
    diff = torch.abs(disparity_with - disparity_without).mean()
    print(f"  Difference (with vs without prior): {diff:.3f} pixels")
    
    # Save visualizations
    print("\nSaving visualizations...")
    os.makedirs('demo_output_foundation', exist_ok=True)
    
    # Save input images
    Image.fromarray(left_img).save('demo_output_foundation/left.png')
    Image.fromarray(right_img).save('demo_output_foundation/right.png')
    print("  Saved: demo_output_foundation/left.png")
    print("  Saved: demo_output_foundation/right.png")
    
    # Save monocular depth prior
    depth_vis = (monocular_depth - monocular_depth.min()) / (monocular_depth.max() - monocular_depth.min())
    depth_vis = (depth_vis * 255).astype(np.uint8)
    Image.fromarray(depth_vis, mode='L').save('demo_output_foundation/monocular_depth.png')
    print("  Saved: demo_output_foundation/monocular_depth.png")
    
    # Save disparity from monocular prior
    visualize_disparity(
        disparity_prior[0],
        save_path='demo_output_foundation/disparity_prior.png',
        title='Disparity Prior (from Monocular Depth)'
    )
    print("  Saved: demo_output_foundation/disparity_prior.png")
    
    # Save prior confidence
    visualize_disparity(
        prior_confidence[0],
        save_path='demo_output_foundation/prior_confidence.png',
        title='Prior Confidence',
        cmap='viridis'
    )
    print("  Saved: demo_output_foundation/prior_confidence.png")
    
    # Save disparity WITH prior
    visualize_disparity(
        disparity_with[0],
        save_path='demo_output_foundation/disparity_with_prior.png',
        title='Disparity WITH Monocular Prior'
    )
    print("  Saved: demo_output_foundation/disparity_with_prior.png")
    
    # Save disparity WITHOUT prior
    visualize_disparity(
        disparity_without[0],
        save_path='demo_output_foundation/disparity_without_prior.png',
        title='Disparity WITHOUT Monocular Prior'
    )
    print("  Saved: demo_output_foundation/disparity_without_prior.png")
    
    # Save comparison
    create_comparison_plot(
        left_tensor[0],
        right_tensor[0],
        disparity_with[0],
        save_path='demo_output_foundation/comparison.png'
    )
    print("  Saved: demo_output_foundation/comparison.png")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("\nKey insights:")
    print("  ✓ Monocular priors provide strong initial estimates")
    print("  ✓ Prior-guided search focuses computational resources")
    print("  ✓ Confidence weighting balances prior and stereo cues")
    print("  ✓ Especially useful in textureless or ambiguous regions")
    print("\nCompare outputs:")
    print("  - disparity_prior.png: Initial estimate from monocular")
    print("  - disparity_with_prior.png: Refined with stereo + prior")
    print("  - disparity_without_prior.png: Stereo only (for comparison)")
    print("="*70)


if __name__ == '__main__':
    main()
