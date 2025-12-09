"""
Inference script for URSM.

Example usage:
    python scripts/inference.py --checkpoint outputs/best/checkpoint.pth \
                                --left left.png --right right.png \
                                --output disparity.png
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ursm.models import URSMNet
from ursm.utils import visualize_disparity, visualize_calibration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with URSM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--left', type=str, required=True,
                       help='Path to left image')
    parser.add_argument('--right', type=str, required=True,
                       help='Path to right image')
    parser.add_argument('--output', type=str, default='disparity.png',
                       help='Output path for disparity map')
    parser.add_argument('--save_calibration', action='store_true',
                       help='Save calibration parameters visualization')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f'Loading model from {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = URSMNet(
        max_disparity=192,
        feature_channels=32,
        refine_iterations=3,
        estimate_calibration=True
    )
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel models
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print('Model loaded successfully')
    return model


def load_image(image_path):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    
    # Convert to tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def run_inference(model, left_img, right_img, device):
    """Run inference on stereo pair."""
    with torch.no_grad():
        outputs = model(left_img, right_img)
    
    return outputs


def save_disparity(disparity, output_path):
    """Save disparity map as image."""
    # Visualize and save
    disparity_colored = visualize_disparity(
        disparity[0],
        save_path=output_path,
        title='Predicted Disparity'
    )
    
    # Also save raw disparity as numpy file
    disparity_np = disparity[0].detach().cpu().numpy()
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, disparity_np)
    
    print(f'Disparity map saved to {output_path}')
    print(f'Raw disparity saved to {npy_path}')


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load images
    print(f'Loading images...')
    print(f'  Left: {args.left}')
    print(f'  Right: {args.right}')
    
    left_img = load_image(args.left).to(device)
    right_img = load_image(args.right).to(device)
    
    print(f'Image size: {left_img.shape[2]}x{left_img.shape[3]}')
    
    # Run inference
    print('Running inference...')
    outputs = run_inference(model, left_img, right_img, device)
    
    # Extract results
    disparity = outputs['disparity']
    uncertainty = outputs.get('uncertainty', None)
    calibration = outputs.get('calibration', None)
    
    print(f'Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]')
    
    # Save disparity
    save_disparity(disparity, args.output)
    
    # Save uncertainty if available
    if uncertainty is not None:
        uncertainty_path = args.output.replace('.png', '_uncertainty.png')
        visualize_disparity(
            uncertainty[0],
            save_path=uncertainty_path,
            title='Uncertainty Map',
            cmap='viridis'
        )
        print(f'Uncertainty map saved to {uncertainty_path}')
    
    # Save calibration if requested
    if args.save_calibration and calibration is not None:
        calibration_path = args.output.replace('.png', '_calibration.png')
        visualize_calibration(calibration, save_path=calibration_path)
        print(f'Calibration visualization saved to {calibration_path}')
        
        # Print calibration parameters
        print('\nEstimated Calibration Parameters:')
        print(f'  Vertical offset: {calibration["vertical_offset"][0, 0].item():.2f} px')
        print(f'  Rotation (roll, pitch, yaw): {calibration["rotation"][0].cpu().numpy()}')
        print(f'  Scale factor: {calibration["scale"][0, 0].item():.4f}')
    
    print('\nInference completed!')


if __name__ == '__main__':
    main()
