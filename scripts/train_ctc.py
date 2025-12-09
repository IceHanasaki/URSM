"""
Training script for URSM-CTC (CTC/HMM-based 1D alignment).

Example usage:
    python scripts/train_ctc.py --config configs/ctc_default.yaml --gpu 0
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ursm.models import URSMNetCTC, URSMNetCTCLoss
from ursm.datasets import StereoDataset, UnrectifiedAugmentation
from ursm.utils import compute_epe, compute_bad_pixels


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train URSM-CTC model')
    parser.add_argument('--config', type=str, default='configs/ctc_default.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='outputs_ctc',
                       help='Output directory for checkpoints and logs')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    # Training dataset
    train_augmentation = UnrectifiedAugmentation(
        vertical_offset_range=tuple(config['augmentation']['vertical_offset_range']),
        rotation_range=tuple(config['augmentation']['rotation_range']),
        scale_range=tuple(config['augmentation']['scale_range']),
        photometric=config['augmentation']['photometric'],
        train=True
    )
    
    train_dataset = StereoDataset(
        root_dir=config['data']['train_dir'],
        split='train',
        transform=train_augmentation,
        load_disparity=True,
        dataset_type=config['data']['dataset_type']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Validation dataset
    val_augmentation = UnrectifiedAugmentation(train=False)
    
    val_dataset = StereoDataset(
        root_dir=config['data']['val_dir'],
        split='val',
        transform=val_augmentation,
        load_disparity=True,
        dataset_type=config['data']['dataset_type']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['val_batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config, device):
    """Create URSM-CTC model."""
    model = URSMNetCTC(
        max_disparity=config['model']['max_disparity'],
        feature_channels=config['model']['feature_channels'],
        hidden_dim=config['model']['hidden_dim'],
        blank_threshold=config['model']['blank_threshold'],
        use_error_correction=config['model']['use_error_correction'],
        correction_iterations=config['model']['correction_iterations']
    )
    
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model


def create_optimizer(model, config):
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['lr_milestones'],
        gamma=config['training']['lr_gamma']
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, loss_fn, config, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_ctc_loss = 0
    total_smooth_loss = 0
    
    for batch_idx, sample in enumerate(train_loader):
        # Move data to device
        left_img = sample['left'].to(device)
        right_img = sample['right'].to(device)
        gt_disparity = sample['disparity'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(left_img, right_img, target_disparity=gt_disparity)
        
        # Compute losses
        losses = loss_fn(outputs, gt_disparity, left_img)
        
        loss = losses['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for CTC stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        if 'ctc_loss' in losses:
            total_ctc_loss += losses['ctc_loss'].item()
        if 'smooth_loss' in losses:
            total_smooth_loss += losses['smooth_loss'].item()
        
        # Logging
        global_step = epoch * len(train_loader) + batch_idx
        
        if batch_idx % config['training']['log_interval'] == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'CTC: {losses.get("ctc_loss", 0):.4f} '
                  f'Smooth: {losses.get("smooth_loss", 0):.4f}')
            
            if writer is not None:
                writer.add_scalar('train/total_loss', loss.item(), global_step)
                for key, value in losses.items():
                    if key != 'total_loss' and torch.is_tensor(value):
                        writer.add_scalar(f'train/{key}', value.item(), global_step)
    
    # Return average losses
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'ctc_loss': total_ctc_loss / num_batches,
        'smooth_loss': total_smooth_loss / num_batches
    }


def validate(model, val_loader, device, epoch, writer):
    """Validate the model."""
    model.eval()
    
    total_epe = 0
    total_bad3 = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            left_img = sample['left'].to(device)
            right_img = sample['right'].to(device)
            gt_disparity = sample['disparity'].to(device)
            
            # Forward pass
            outputs = model(left_img, right_img)
            
            # Compute metrics
            pred_disparity = outputs['disparity']
            epe = compute_epe(pred_disparity, gt_disparity)
            bad3 = compute_bad_pixels(pred_disparity, gt_disparity, threshold=3.0)
            
            total_epe += epe
            total_bad3 += bad3
            total_samples += 1
    
    # Average metrics
    avg_epe = total_epe / total_samples
    avg_bad3 = total_bad3 / total_samples
    
    print(f'Validation - EPE: {avg_epe:.4f}, Bad3: {avg_bad3:.2f}%')
    
    if writer is not None:
        writer.add_scalar('val/epe', avg_epe, epoch)
        writer.add_scalar('val/bad3', avg_bad3, epoch)
    
    return {'epe': avg_epe, 'bad3': avg_bad3}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Create dataloaders
    print('Creating dataloaders...')
    train_loader, val_loader = create_dataloaders(config)
    print(f'Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}')
    
    # Create model
    print('Creating URSM-CTC model...')
    model = create_model(config, device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # Create optimizer
    optimizer, scheduler = create_optimizer(model, config)
    
    # Create loss function
    loss_fn = URSMNetCTCLoss(
        ctc_weight=config['loss']['ctc_weight'],
        smooth_weight=config['loss']['smooth_weight'],
        occlusion_weight=config['loss']['occlusion_weight']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print('Starting training...')
    best_epe = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f'\nEpoch {epoch}/{config["training"]["num_epochs"]}')
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            config, device, epoch, writer
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch, writer)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if epoch % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args.output_dir)
        
        # Save best model
        if val_metrics['epe'] < best_epe:
            best_epe = val_metrics['epe']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                          os.path.join(args.output_dir, 'best'))
            print(f'New best model! EPE: {best_epe:.4f}')
    
    print('Training completed!')
    writer.close()


if __name__ == '__main__':
    main()
