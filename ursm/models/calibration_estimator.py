"""
Calibration parameter estimation module.

Estimates calibration parameters (vertical offset, rotation, scale) from
stereo image features to handle imperfect camera calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationEstimator(nn.Module):
    """
    Estimates calibration parameters between stereo image pairs.
    
    This module predicts calibration discrepancies that occur due to:
    - Vertical misalignment (baseline is not perfectly horizontal)
    - Rotational misalignment (cameras not perfectly aligned)
    - Scale differences (focal length variations)
    
    Args:
        feature_channels (int): Number of input feature channels (default: 32)
        hidden_channels (int): Hidden layer channels (default: 128)
    """
    
    def __init__(self, feature_channels=32, hidden_channels=128):
        super(CalibrationEstimator, self).__init__()
        
        self.feature_channels = feature_channels
        
        # Feature aggregation layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Correlation-based feature extraction
        self.correlation_conv = nn.Sequential(
            nn.Conv2d(feature_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Global context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Calibration parameter prediction heads
        # Vertical offset (in pixels)
        self.vertical_offset_head = nn.Linear(hidden_channels // 2, 1)
        
        # Rotation angles (roll, pitch, yaw in degrees)
        self.rotation_head = nn.Linear(hidden_channels // 2, 3)
        
        # Scale factor (relative scale difference)
        self.scale_head = nn.Linear(hidden_channels // 2, 1)
        
        # Uncertainty estimation for calibration parameters
        self.uncertainty_head = nn.Linear(hidden_channels // 2, 5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features_left, features_right):
        """
        Estimate calibration parameters from stereo features.
        
        Args:
            features_left (torch.Tensor): Left image features [B, C, H, W]
            features_right (torch.Tensor): Right image features [B, C, H, W]
        
        Returns:
            dict: Calibration parameters containing:
                - 'vertical_offset': Vertical misalignment in pixels [B, 1]
                - 'rotation': Rotation angles (roll, pitch, yaw) [B, 3]
                - 'scale': Relative scale factor [B, 1]
                - 'uncertainty': Uncertainty for each parameter [B, 5]
        """
        batch_size = features_left.shape[0]
        
        # Concatenate features for correlation analysis
        combined_features = torch.cat([features_left, features_right], dim=1)
        
        # Extract correlation features
        correlation_features = self.correlation_conv(combined_features)
        
        # Global average pooling
        global_features = self.global_pool(correlation_features)
        global_features = global_features.view(batch_size, -1)
        
        # Encode global context
        context = self.context_encoder(global_features)
        
        # Scale factor range constants
        SCALE_MIN = 0.8
        SCALE_RANGE = 0.4  # (0.8 to 1.2)
        
        # Predict calibration parameters
        vertical_offset = self.vertical_offset_head(context)  # [B, 1]
        rotation = self.rotation_head(context)  # [B, 3]
        scale = torch.sigmoid(self.scale_head(context))  # [B, 1], range (0, 1)
        scale = SCALE_MIN + SCALE_RANGE * scale  # Map to range (0.8, 1.2)
        
        # Estimate uncertainty
        uncertainty = F.softplus(self.uncertainty_head(context))  # [B, 5]
        
        # Prepare output dictionary
        calibration_params = {
            'vertical_offset': vertical_offset,
            'rotation': rotation,
            'scale': scale,
            'uncertainty': uncertainty
        }
        
        return calibration_params
    
    def apply_calibration(self, image, calibration_params):
        """
        Apply estimated calibration correction to an image.
        
        Args:
            image (torch.Tensor): Input image [B, C, H, W]
            calibration_params (dict): Calibration parameters
        
        Returns:
            torch.Tensor: Corrected image [B, C, H, W]
        """
        batch_size, channels, height, width = image.shape
        device = image.device
        
        # Extract parameters
        vertical_offset = calibration_params['vertical_offset']
        rotation = calibration_params['rotation']
        scale = calibration_params['scale']
        
        # Create affine transformation matrix
        # This is a simplified version; full implementation would include
        # proper rotation and scaling transformations
        
        # For now, apply vertical offset correction
        grid_y = torch.linspace(-1, 1, height, device=device).view(1, height, 1).expand(batch_size, height, width)
        grid_x = torch.linspace(-1, 1, width, device=device).view(1, 1, width).expand(batch_size, height, width)
        
        # Apply vertical offset (normalized to [-1, 1] range)
        offset_normalized = vertical_offset.view(batch_size, 1, 1) * (2.0 / height)
        grid_y = grid_y + offset_normalized
        
        # Stack into sampling grid
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Apply transformation
        corrected_image = F.grid_sample(
            image, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return corrected_image
