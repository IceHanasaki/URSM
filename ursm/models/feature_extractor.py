"""
Feature extraction module with robustness to calibration errors.

Uses a hierarchical architecture to capture multi-scale features that are
invariant to small calibration perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class FeatureExtractor(nn.Module):
    """
    Hierarchical feature extractor for stereo images.
    
    Extracts robust multi-scale features that are less sensitive to
    calibration errors through the use of:
    - Residual connections for better gradient flow
    - Multi-scale feature pyramid
    - Larger receptive fields to capture geometric context
    
    Args:
        output_channels (int): Number of output feature channels (default: 32)
        num_blocks (list): Number of residual blocks at each scale (default: [2, 2, 2])
    """
    
    def __init__(self, output_channels=32, num_blocks=None):
        super(FeatureExtractor, self).__init__()
        
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        
        self.output_channels = output_channels
        
        # Initial convolution
        self.conv_start = nn.Sequential(
            nn.Conv2d(3, output_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks at different scales
        self.layer1 = self._make_layer(output_channels, output_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(output_channels, output_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(output_channels * 2, output_channels * 2, num_blocks[2], stride=1)
        
        # Feature fusion and output projection
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=False)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer of residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Extract features from input image.
        
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Extracted features [B, C, H/4, W/4]
        """
        # Initial feature extraction
        x = self.conv_start(x)  # [B, C, H/2, W/2]
        
        # Hierarchical feature extraction
        x = self.layer1(x)      # [B, C, H/2, W/2]
        x = self.layer2(x)      # [B, 2C, H/4, W/4]
        x = self.layer3(x)      # [B, 2C, H/4, W/4]
        
        # Feature fusion to output channels
        features = self.conv_fusion(x)  # [B, C, H/4, W/4]
        
        return features
