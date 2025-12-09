"""
Calibration-aware smoothness loss.

Encourages smooth disparity predictions while respecting image edges
and accounting for potential calibration errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationAwareSmoothness(nn.Module):
    """
    Calibration-aware smoothness loss for disparity maps.
    
    Encourages disparity smoothness in regions with similar appearance,
    while allowing discontinuities at object boundaries. Takes into
    account calibration uncertainty.
    
    Args:
        alpha (float): Weight for smoothness loss (default: 0.1)
        edge_aware (bool): Use edge-aware weighting (default: True)
    """
    
    def __init__(self, alpha=0.1, edge_aware=True):
        super(CalibrationAwareSmoothness, self).__init__()
        
        self.alpha = alpha
        self.edge_aware = edge_aware
    
    def compute_image_gradients(self, image):
        """
        Compute image gradients.
        
        Args:
            image (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
            tuple: (grad_x, grad_y) gradients in x and y directions
        """
        # Compute gradients using Sobel-like filters
        grad_x = image[:, :, :, :-1] - image[:, :, :, 1:]
        grad_y = image[:, :, :-1, :] - image[:, :, 1:, :]
        
        return grad_x, grad_y
    
    def compute_disparity_gradients(self, disparity):
        """
        Compute disparity gradients.
        
        Args:
            disparity (torch.Tensor): Disparity map [B, 1, H, W]
        
        Returns:
            tuple: (grad_x, grad_y) disparity gradients
        """
        grad_x = disparity[:, :, :, :-1] - disparity[:, :, :, 1:]
        grad_y = disparity[:, :, :-1, :] - disparity[:, :, 1:, :]
        
        return grad_x, grad_y
    
    def forward(self, disparity, image, calibration_uncertainty=None):
        """
        Compute calibration-aware smoothness loss.
        
        Args:
            disparity (torch.Tensor): Predicted disparity [B, 1, H, W]
            image (torch.Tensor): Input image [B, 3, H, W]
            calibration_uncertainty (torch.Tensor, optional): Calibration uncertainty
        
        Returns:
            torch.Tensor: Smoothness loss value
        """
        # Compute disparity gradients
        disp_grad_x, disp_grad_y = self.compute_disparity_gradients(disparity)
        
        # Compute smoothness (L1 norm of gradients)
        smoothness_x = torch.abs(disp_grad_x)
        smoothness_y = torch.abs(disp_grad_y)
        
        # Edge-aware weighting
        if self.edge_aware:
            # Compute image gradients
            img_grad_x, img_grad_y = self.compute_image_gradients(image)
            
            # Compute gradient magnitude
            img_grad_x = torch.mean(torch.abs(img_grad_x), dim=1, keepdim=True)
            img_grad_y = torch.mean(torch.abs(img_grad_y), dim=1, keepdim=True)
            
            # Weight smoothness by inverse of image gradients
            # Strong image edges -> low smoothness penalty
            weight_x = torch.exp(-img_grad_x)
            weight_y = torch.exp(-img_grad_y)
            
            smoothness_x = smoothness_x * weight_x
            smoothness_y = smoothness_y * weight_y
        
        # Account for calibration uncertainty
        if calibration_uncertainty is not None:
            # Higher uncertainty -> higher smoothness penalty
            # (we trust smoothness more when calibration is uncertain)
            uncertainty_weight = 1.0 + calibration_uncertainty
            smoothness_x = smoothness_x * uncertainty_weight
            smoothness_y = smoothness_y * uncertainty_weight
        
        # Combine smoothness in both directions
        smoothness_loss = self.alpha * (smoothness_x.mean() + smoothness_y.mean())
        
        return smoothness_loss
