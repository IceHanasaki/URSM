"""
Adaptive cost volume construction for unrectified stereo matching.

Builds cost volumes that adapt to calibration parameters, enabling
robust matching even with imperfect calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCostVolume(nn.Module):
    """
    Adaptive cost volume that handles calibration variations.
    
    Instead of simple horizontal matching, this module:
    1. Adapts search regions based on estimated calibration
    2. Uses correlation-based matching with learned features
    3. Aggregates costs over multiple scales
    
    Args:
        max_disparity (int): Maximum disparity range (default: 192)
        feature_channels (int): Number of feature channels (default: 32)
        cost_aggregation (bool): Enable cost aggregation (default: True)
    """
    
    def __init__(
        self,
        max_disparity=192,
        feature_channels=32,
        cost_aggregation=True
    ):
        super(AdaptiveCostVolume, self).__init__()
        
        self.max_disparity = max_disparity
        self.feature_channels = feature_channels
        self.cost_aggregation = cost_aggregation
        
        # Cost aggregation network
        if self.cost_aggregation:
            self.cost_aggregator = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
            )
    
    def compute_correlation_cost(self, features_left, features_right, disparity):
        """
        Compute correlation cost for a specific disparity level.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            disparity (int): Disparity level to compute cost for
        
        Returns:
            torch.Tensor: Cost map [B, H, W]
        """
        batch_size, channels, height, width = features_left.shape
        
        if disparity == 0:
            # No shift for disparity 0
            shifted_right = features_right
        else:
            # Shift right features by disparity
            shifted_right = torch.zeros_like(features_right)
            shifted_right[:, :, :, disparity:] = features_right[:, :, :, :-disparity]
        
        # Compute correlation (dot product)
        cost = torch.sum(features_left * shifted_right, dim=1)  # [B, H, W]
        
        return cost
    
    def build_cost_volume_standard(self, features_left, features_right):
        """
        Build standard cost volume with horizontal search only.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
        
        Returns:
            torch.Tensor: Cost volume [B, D, H, W]
        """
        batch_size, channels, height, width = features_left.shape
        device = features_left.device
        
        # Initialize cost volume
        cost_volume = torch.zeros(
            batch_size, self.max_disparity, height, width,
            dtype=features_left.dtype, device=device
        )
        
        # Compute cost for each disparity level
        for d in range(self.max_disparity):
            cost_volume[:, d, :, :] = self.compute_correlation_cost(
                features_left, features_right, d
            )
        
        return cost_volume
    
    def build_cost_volume_adaptive(
        self, features_left, features_right, calibration_params
    ):
        """
        Build adaptive cost volume considering calibration parameters.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            calibration_params (dict): Estimated calibration parameters
        
        Returns:
            torch.Tensor: Adaptive cost volume [B, D, H, W]
        """
        batch_size, channels, height, width = features_left.shape
        device = features_left.device
        
        # Extract calibration parameters
        vertical_offset = calibration_params.get('vertical_offset', None)
        
        # For adaptive matching, we adjust the search region
        # based on vertical offset
        if vertical_offset is not None:
            # Apply vertical correction to right features
            grid_y = torch.linspace(-1, 1, height, device=device).view(1, height, 1).expand(batch_size, height, width)
            grid_x = torch.linspace(-1, 1, width, device=device).view(1, 1, width).expand(batch_size, height, width)
            
            # Apply vertical offset
            offset_normalized = vertical_offset.view(batch_size, 1, 1) * (2.0 / height)
            grid_y = grid_y + offset_normalized
            
            grid = torch.stack([grid_x, grid_y], dim=-1)
            features_right_corrected = F.grid_sample(
                features_right, grid, mode='bilinear',
                padding_mode='border', align_corners=True
            )
        else:
            features_right_corrected = features_right
        
        # Build standard cost volume with corrected features
        cost_volume = self.build_cost_volume_standard(
            features_left, features_right_corrected
        )
        
        return cost_volume
    
    def forward(self, features_left, features_right, calibration_params=None):
        """
        Construct cost volume.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            calibration_params (dict, optional): Calibration parameters
        
        Returns:
            torch.Tensor: Cost volume [B, D, H, W]
        """
        # Build cost volume (adaptive if calibration params provided)
        if calibration_params is not None:
            cost_volume = self.build_cost_volume_adaptive(
                features_left, features_right, calibration_params
            )
        else:
            cost_volume = self.build_cost_volume_standard(
                features_left, features_right
            )
        
        # Aggregate costs if enabled
        if self.cost_aggregation:
            # Add channel dimension for 3D convolution
            cost_volume_input = cost_volume.unsqueeze(1)  # [B, 1, D, H, W]
            
            # Apply 3D convolution for cost aggregation
            aggregated_cost = self.cost_aggregator(cost_volume_input)
            
            # Remove channel dimension
            cost_volume = aggregated_cost.squeeze(1)  # [B, D, H, W]
        
        return cost_volume
