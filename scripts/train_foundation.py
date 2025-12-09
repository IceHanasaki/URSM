"""
Training script for Foundation Stereo with monocular priors.

Example usage:
    python scripts/train_foundation.py --config configs/foundation_default.yaml --gpu 0
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

from ursm.models import FoundationStereo
from ursm.datasets import StereoDataset, UnrectifiedAugmentation
from ursm.losses import DisparityLoss, CalibrationAwareSmoothness
from ursm.utils import compute_epe, compute_bad_pixels


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Foundation Stereo model')
    parser.add_argument('--config', type=str, default='configs/foundation_default.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='outputs_foundation',
                       help='Output directory for checkpoints and logs')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config):
    """Create training and validation dataloaders."""
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
    """Create Foundation Stereo model."""
    model = FoundationStereo(
        max_disparity=config['model']['max_disparity'],
        feature_channels=config['model']['feature_channels'],
        use_monocular_prior=config['model']['use_monocular_prior'],
        monocular_model_name=config['model']['monocular_model_name'],
        search_range=config['model']['search_range']
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


def create_losses(config, device):
    """Create loss functions."""
    disparity_loss = DisparityLoss(
        loss_type=config['loss']['disparity_loss_type'],
        uncertainty_weight=config['loss']['uncertainty_weight'],
        max_disparity=config['model']['max_disparity']
    ).to(device)
    
    smoothness_loss = CalibrationAwareSmoothness(
        alpha=config['loss']['smoothness_alpha'],
        edge_aware=config['loss']['edge_aware']
    ).to(device)
    
    return disparity_loss, smoothness_loss


def train_epoch(model, train_loader, optimizer, disparity_loss_fn, smoothness_loss_fn,
                config, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_disparity_loss = 0
    total_smoothness_loss = 0
    
    for batch_idx, sample in enumerate(train_loader):
        left_img = sample['left'].to(device)
        right_img = sample['right'].to(device)
        gt_disparity = sample['disparity'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(left_img, right_img)
        
        # Compute losses
        disparity_loss = disparity_loss_fn(outputs, gt_disparity)
        smoothness_loss = smoothness_loss_fn(outputs['disparity'], left_img)
        
        # Prior consistency loss (if using monocular prior)
        if 'disparity_prior' in outputs and config['loss'].get('prior_weight', 0) > 0:
            prior_weight = config['loss']['prior_weight']
            prior_confidence = outputs['prior_confidence']
            
            # Weighted L1 loss between prediction and prior
            prior_diff = torch.abs(outputs['disparity'] - outputs['disparity_prior'])
            prior_loss = (prior_diff * prior_confidence).mean()
            
            total_loss_val = disparity_loss + smoothness_loss + prior_weight * prior_loss
        else:
            prior_loss = torch.tensor(0.0)
            total_loss_val = disparity_loss + smoothness_loss
        
        # Backward pass
        total_loss_val.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_val.item()
        total_disparity_loss += disparity_loss.item()
        total_smoothness_loss += smoothness_loss.item()
        
        # Logging
        global_step = epoch * len(train_loader) + batch_idx
        
        if batch_idx % config['training']['log_interval'] == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {total_loss_val.item():.4f} '
                  f'Disp: {disparity_loss.item():.4f} '
                  f'Smooth: {smoothness_loss.item():.4f} '
                  f'Prior: {prior_loss.item():.4f}')
            
            if writer is not None:
                writer.add_scalar('train/total_loss', total_loss_val.item(), global_step)
                writer.add_scalar('train/disparity_loss', disparity_loss.item(), global_step)
                writer.add_scalar('train/smoothness_loss', smoothness_loss.item(), global_step)
                if prior_loss.item() > 0:
                    writer.add_scalar('train/prior_loss', prior_loss.item(), global_step)
    
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'disparity_loss': total_disparity_loss / num_batches,
        'smoothness_loss': total_smoothness_loss / num_batches
    }


def validate(model, val_loader, disparity_loss_fn, device, epoch, writer):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_epe = 0
    total_bad3 = 0
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            left_img = sample['left'].to(device)
            right_img = sample['right'].to(device)
            gt_disparity = sample['disparity'].to(device)
            
            # Forward pass
            outputs = model(left_img, right_img)
            
            # Compute loss
            loss = disparity_loss_fn(outputs, gt_disparity)
            total_loss += loss.item()
            
            # Compute metrics
            pred_disparity = outputs['disparity']
            epe = compute_epe(pred_disparity, gt_disparity)
            bad3 = compute_bad_pixels(pred_disparity, gt_disparity, threshold=3.0)
            
            total_epe += epe
            total_bad3 += bad3
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_epe = total_epe / num_batches
    avg_bad3 = total_bad3 / num_batches
    
    print(f'Validation - Loss: {avg_loss:.4f}, EPE: {avg_epe:.4f}, Bad3: {avg_bad3:.2f}%')
    
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/epe', avg_epe, epoch)
        writer.add_scalar('val/bad3', avg_bad3, epoch)
    
    return {'loss': avg_loss, 'epe': avg_epe, 'bad3': avg_bad3}


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
    print('Creating Foundation Stereo model...')
    model = create_model(config, device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # Create optimizer
    optimizer, scheduler = create_optimizer(model, config)
    
    # Create losses
    disparity_loss_fn, smoothness_loss_fn = create_losses(config, device)
    
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
            model, train_loader, optimizer, disparity_loss_fn,
            smoothness_loss_fn, config, device, epoch, writer
        )
        
        # Validate
        val_metrics = validate(model, val_loader, disparity_loss_fn, device, epoch, writer)
        
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
