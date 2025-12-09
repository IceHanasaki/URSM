"""
Disparity loss with uncertainty weighting.

Implements robust disparity loss that accounts for prediction uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisparityLoss(nn.Module):
    """
    Disparity loss with uncertainty weighting.
    
    Computes loss between predicted and ground truth disparity,
    weighted by prediction uncertainty. Areas with high uncertainty
    receive lower loss weight.
    
    Args:
        loss_type (str): Type of loss ('l1', 'smooth_l1', 'l2') (default: 'smooth_l1')
        uncertainty_weight (bool): Use uncertainty for weighting (default: True)
        max_disparity (float): Maximum valid disparity for masking (default: None)
    """
    
    def __init__(
        self,
        loss_type='smooth_l1',
        uncertainty_weight=True,
        max_disparity=None
    ):
        super(DisparityLoss, self).__init__()
        
        self.loss_type = loss_type
        self.uncertainty_weight = uncertainty_weight
        self.max_disparity = max_disparity
    
    def compute_loss(self, pred, target, mask=None):
        """
        Compute base disparity loss.
        
        Args:
            pred (torch.Tensor): Predicted disparity [B, 1, H, W]
            target (torch.Tensor): Ground truth disparity [B, 1, H, W]
            mask (torch.Tensor, optional): Valid pixel mask [B, 1, H, W]
        
        Returns:
            torch.Tensor: Loss value
        """
        if self.loss_type == 'l1':
            loss = torch.abs(pred - target)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'l2':
            loss = (pred - target) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()
    
    def forward(self, outputs, target_disparity):
        """
        Compute disparity loss.
        
        Args:
            outputs (dict): Model outputs containing:
                - 'disparity': Predicted disparity [B, 1, H, W]
                - 'uncertainty': Uncertainty map [B, 1, H, W] (optional)
            target_disparity (torch.Tensor): Ground truth disparity [B, 1, H, W]
        
        Returns:
            torch.Tensor: Loss value
        """
        pred_disparity = outputs['disparity']
        uncertainty = outputs.get('uncertainty', None)
        
        # Create valid pixel mask
        # Mask out invalid disparity values (typically 0 or very large)
        valid_mask = (target_disparity > 0) & (target_disparity < float('inf'))
        
        if self.max_disparity is not None:
            valid_mask = valid_mask & (target_disparity < self.max_disparity)
        
        valid_mask = valid_mask.float()
        
        # Compute base loss
        loss = torch.abs(pred_disparity - target_disparity)
        
        if self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_disparity, target_disparity, reduction='none')
        elif self.loss_type == 'l2':
            loss = (pred_disparity - target_disparity) ** 2
        
        # Apply uncertainty weighting if available
        if self.uncertainty_weight and uncertainty is not None:
            # Uncertainty-weighted loss: loss / (2 * uncertainty) + log(uncertainty)
            # This encourages the model to be certain when it's correct
            # and uncertain when it might be wrong
            loss = loss / (2 * uncertainty + 1e-8) + torch.log(uncertainty + 1e-8)
        
        # Apply valid mask
        loss = loss * valid_mask
        
        # Return mean loss over valid pixels
        total_loss = loss.sum() / (valid_mask.sum() + 1e-8)
        
        return total_loss
