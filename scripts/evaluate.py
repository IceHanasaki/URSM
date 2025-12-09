"""
Evaluation script for URSM.

Example usage:
    python scripts/evaluate.py --checkpoint outputs/best/checkpoint.pth --data_dir /path/to/test/data
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ursm.models import URSMNet
from ursm.datasets import StereoDataset, UnrectifiedAugmentation
from ursm.utils import evaluate_stereo, visualize_disparity, create_comparison_plot


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate URSM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f'Loading model from {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with default parameters
    # You may need to adjust these based on your checkpoint
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


def create_test_dataloader(data_dir, batch_size):
    """Create test dataloader."""
    test_augmentation = UnrectifiedAugmentation(train=False)
    
    test_dataset = StereoDataset(
        root_dir=data_dir,
        split='test',
        transform=test_augmentation,
        load_disparity=True,
        dataset_type='custom'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader


def evaluate_model(model, test_loader, device, output_dir, save_visualizations):
    """Evaluate model on test set."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {
        'epe': [],
        'bad_1': [],
        'bad_2': [],
        'bad_3': [],
        'd1_error': [],
        'acc_1': [],
        'acc_2': [],
        'acc_3': []
    }
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader, desc='Evaluating')):
            left_img = sample['left'].to(device)
            right_img = sample['right'].to(device)
            gt_disparity = sample['disparity'].to(device)
            
            # Forward pass
            outputs = model(left_img, right_img)
            pred_disparity = outputs['disparity']
            
            # Compute metrics
            metrics = evaluate_stereo(pred_disparity, gt_disparity)
            
            # Accumulate metrics
            for key in all_metrics.keys():
                all_metrics[key].append(metrics[key])
            
            # Save visualizations
            if save_visualizations:
                vis_dir = os.path.join(output_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                save_path = os.path.join(vis_dir, f'sample_{batch_idx:04d}.png')
                create_comparison_plot(
                    left_img[0],
                    right_img[0],
                    pred_disparity[0],
                    gt_disparity[0],
                    save_path=save_path
                )
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics.keys():
        avg_metrics[key] = np.mean(all_metrics[key])
    
    return avg_metrics


def print_metrics(metrics):
    """Print evaluation metrics."""
    print('\n' + '='*50)
    print('Evaluation Results:')
    print('='*50)
    print(f'End-Point Error (EPE):        {metrics["epe"]:.4f} px')
    print(f'Bad Pixels @ 1px:             {metrics["bad_1"]:.2f}%')
    print(f'Bad Pixels @ 2px:             {metrics["bad_2"]:.2f}%')
    print(f'Bad Pixels @ 3px:             {metrics["bad_3"]:.2f}%')
    print(f'D1 Error (KITTI):             {metrics["d1_error"]:.2f}%')
    print(f'Accuracy @ 1px:               {metrics["acc_1"]:.2f}%')
    print(f'Accuracy @ 2px:               {metrics["acc_2"]:.2f}%')
    print(f'Accuracy @ 3px:               {metrics["acc_3"]:.2f}%')
    print('='*50 + '\n')


def save_metrics(metrics, output_dir):
    """Save metrics to file."""
    import json
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f'Metrics saved to {metrics_path}')


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create test dataloader
    print(f'Loading test data from {args.data_dir}')
    test_loader = create_test_dataloader(args.data_dir, args.batch_size)
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Evaluate model
    print('Starting evaluation...')
    metrics = evaluate_model(
        model, test_loader, device, args.output_dir, args.save_visualizations
    )
    
    # Print and save results
    print_metrics(metrics)
    save_metrics(metrics, args.output_dir)
    
    print('Evaluation completed!')


if __name__ == '__main__':
    main()
