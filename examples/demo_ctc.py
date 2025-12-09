"""
Demo script for URSM-CTC (CTC/HMM-based 1D alignment).

This script demonstrates the CTC-based approach to stereo matching
that uses 1D alignment along epipolar curves with blank tokens for occlusions.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from ursm.models import URSMNetCTC
from ursm.utils import visualize_disparity, create_comparison_plot


def create_sample_stereo_pair():
    """
    Create a synthetic stereo pair with occlusions for demonstration.
    """
    height, width = 480, 640
    
    # Create left image with vertical stripes and objects
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(height):
        left_img[i, :, :] = int(128 + 127 * np.sin(i / height * 4 * np.pi))
    
    # Add a "close object" (high disparity)
    left_img[200:300, 100:200, :] = [255, 0, 0]  # Red square
    
    # Add a "far object" (low disparity)
    left_img[150:250, 350:450, :] = [0, 255, 0]  # Green square
    
    # Create right image with disparity
    right_img = np.zeros_like(left_img)
    
    # Background (small shift)
    shift_bg = 5
    right_img[:, shift_bg:, :] = left_img[:, :-shift_bg, :]
    
    # Close object (large shift = high disparity = 30 pixels)
    shift_close = 30
    right_img[200:300, (100-shift_close):(200-shift_close), :] = [255, 0, 0]
    
    # Far object (small shift = low disparity = 10 pixels)
    shift_far = 10
    right_img[150:250, (350-shift_far):(450-shift_far), :] = [0, 255, 0]
    
    # Create occlusion (visible in left but not in right)
    left_img[100:150, 250:300, :] = [0, 0, 255]  # Blue occluded region
    # This region won't appear correctly in right image (occlusion)
    
    return left_img, right_img


def prepare_images(left_img, right_img):
    """Convert images to tensors for model input."""
    left = torch.from_numpy(left_img).float() / 255.0
    right = torch.from_numpy(right_img).float() / 255.0
    
    left = left.permute(2, 0, 1).unsqueeze(0)
    right = right.permute(2, 0, 1).unsqueeze(0)
    
    return left, right


def main():
    """Main demo function."""
    print("="*70)
    print("URSM-CTC Demo - CTC/HMM-based 1D Alignment for Stereo Matching")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating URSM-CTC model...")
    print("Key features:")
    print("  ✓ 1D alignment along epipolar curves (reduced complexity)")
    print("  ✓ CTC with blank tokens (natural occlusion handling)")
    print("  ✓ Monotonic constraint (forward-only matching)")
    print("  ✓ Sparse parity check error correction")
    
    model = URSMNetCTC(
        max_disparity=96,
        feature_channels=16,
        hidden_dim=64,
        blank_threshold=0.3,
        use_error_correction=True,
        correction_iterations=3
    )
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel created with {num_params:.2f}M parameters")
    
    # Create sample stereo pair with occlusions
    print("\nCreating sample stereo pair with occlusions...")
    left_img, right_img = create_sample_stereo_pair()
    print(f"Image size: {left_img.shape[0]}x{left_img.shape[1]}")
    print("Scene contains:")
    print("  - Close object (red square, high disparity ~30px)")
    print("  - Far object (green square, low disparity ~10px)")
    print("  - Occluded region (blue, visible in left only)")
    
    # Prepare images
    left_tensor, right_tensor = prepare_images(left_img, right_img)
    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)
    
    # Run inference
    print("\nRunning inference with CTC alignment...")
    with torch.no_grad():
        outputs = model(left_tensor, right_tensor)
    
    # Extract outputs
    disparity = outputs['disparity']
    disparity_ctc = outputs['disparity_ctc']
    occlusion_mask = outputs['occlusion_mask']
    confidence = outputs['confidence']
    
    print(f"\nResults:")
    print(f"  Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")
    print(f"  Occlusion ratio: {(1 - occlusion_mask.mean()):.2%}")
    print(f"  Mean confidence: {confidence.mean():.4f}")
    
    # Compare before and after error correction
    diff = torch.abs(disparity - disparity_ctc).mean()
    print(f"  Error correction change: {diff:.4f} pixels (mean absolute)")
    
    # Save visualizations
    print("\nSaving visualizations...")
    os.makedirs('demo_output_ctc', exist_ok=True)
    
    # Save input images
    Image.fromarray(left_img).save('demo_output_ctc/left.png')
    Image.fromarray(right_img).save('demo_output_ctc/right.png')
    print("  Saved: demo_output_ctc/left.png")
    print("  Saved: demo_output_ctc/right.png")
    
    # Save CTC raw disparity
    visualize_disparity(
        disparity_ctc[0],
        save_path='demo_output_ctc/disparity_ctc_raw.png',
        title='Disparity (CTC Output - Before Correction)'
    )
    print("  Saved: demo_output_ctc/disparity_ctc_raw.png")
    
    # Save corrected disparity
    visualize_disparity(
        disparity[0],
        save_path='demo_output_ctc/disparity_corrected.png',
        title='Disparity (After Parity Check Correction)'
    )
    print("  Saved: demo_output_ctc/disparity_corrected.png")
    
    # Save occlusion mask
    visualize_disparity(
        occlusion_mask[0],
        save_path='demo_output_ctc/occlusion_mask.png',
        title='Occlusion Mask (1=valid, 0=occluded)',
        cmap='RdYlGn'
    )
    print("  Saved: demo_output_ctc/occlusion_mask.png")
    
    # Save confidence map
    confidence_img = confidence[0].squeeze(-1) if confidence.dim() == 4 else confidence[0]
    visualize_disparity(
        confidence_img,
        save_path='demo_output_ctc/confidence.png',
        title='Confidence Map',
        cmap='viridis'
    )
    print("  Saved: demo_output_ctc/confidence.png")
    
    # Create comparison plot
    create_comparison_plot(
        left_tensor[0],
        right_tensor[0],
        disparity[0],
        save_path='demo_output_ctc/comparison.png'
    )
    print("  Saved: demo_output_ctc/comparison.png")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("\nKey takeaways:")
    print("  ✓ CTC naturally handles occlusions via blank tokens")
    print("  ✓ 1D alignment is more efficient than 2D matching")
    print("  ✓ Monotonic constraint prevents impossible matches")
    print("  ✓ Parity check correction improves geometric consistency")
    print("\nCheck the 'demo_output_ctc' directory for results.")
    print("Compare 'disparity_ctc_raw.png' vs 'disparity_corrected.png'")
    print("to see the effect of error correction.")
    print("="*70)


if __name__ == '__main__':
    main()
