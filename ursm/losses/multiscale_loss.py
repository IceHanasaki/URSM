"""
Multi-scale loss for stereo matching.

Computes loss at multiple resolutions to help with training stability
and to capture both fine and coarse details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that computes disparity loss at multiple resolutions.
    
    Args:
        scales (list): List of scale factors (default: [1.0, 0.5, 0.25])
        weights (list): Weights for each scale (default: [1.0, 0.5, 0.25])
        base_loss (nn.Module): Base loss function to use at each scale
    """
    
    def __init__(self, scales=None, weights=None, base_loss=None):
        super(MultiScaleLoss, self).__init__()
        
        if scales is None:
            scales = [1.0, 0.5, 0.25]
        if weights is None:
            weights = [1.0, 0.5, 0.25]
        
        assert len(scales) == len(weights), "Scales and weights must have same length"
        
        self.scales = scales
        self.weights = weights
        self.base_loss = base_loss
    
    def downsample_disparity(self, disparity, scale):
        """
        Downsample disparity map to a specific scale.
        
        Args:
            disparity (torch.Tensor): Disparity map [B, 1, H, W]
            scale (float): Scale factor
        
        Returns:
            torch.Tensor: Downsampled disparity
        """
        if scale == 1.0:
            return disparity
        
        # Downsample using bilinear interpolation
        downsampled = F.interpolate(
            disparity,
            scale_factor=scale,
            mode='bilinear',
            align_corners=True
        )
        
        # Scale disparity values accordingly
        downsampled = downsampled * scale
        
        return downsampled
    
    def forward(self, outputs_pyramid, target_disparity):
        """
        Compute multi-scale loss.
        
        Args:
            outputs_pyramid (list or dict): Either a list of outputs at different scales,
                                          or a single output dict (will create pyramid)
            target_disparity (torch.Tensor): Ground truth disparity [B, 1, H, W]
        
        Returns:
            torch.Tensor: Total multi-scale loss
        """
        total_loss = 0.0
        
        # If outputs_pyramid is a dict, create pyramid from it
        if isinstance(outputs_pyramid, dict):
            pred_disparity = outputs_pyramid['disparity']
            outputs_list = []
            
            for scale in self.scales:
                if scale == 1.0:
                    outputs_list.append(outputs_pyramid)
                else:
                    # Downsample prediction
                    downsampled_pred = self.downsample_disparity(pred_disparity, scale)
                    outputs_list.append({'disparity': downsampled_pred})
            
            outputs_pyramid = outputs_list
        
        # Compute loss at each scale
        for scale, weight, outputs in zip(self.scales, self.weights, outputs_pyramid):
            # Downsample target to current scale
            target_scale = self.downsample_disparity(target_disparity, scale)
            
            # Compute loss at this scale
            if self.base_loss is not None:
                scale_loss = self.base_loss(outputs, target_scale)
            else:
                # Default to L1 loss
                pred_scale = outputs['disparity']
                valid_mask = (target_scale > 0).float()
                scale_loss = torch.abs(pred_scale - target_scale) * valid_mask
                scale_loss = scale_loss.sum() / (valid_mask.sum() + 1e-8)
            
            # Add weighted loss
            total_loss += weight * scale_loss
        
        # Normalize by total weight
        total_weight = sum(self.weights)
        total_loss = total_loss / total_weight
        
        return total_loss
