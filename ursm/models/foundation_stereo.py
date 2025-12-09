"""
Foundation Stereo: Post-calibration stereo matching with monocular priors.

Inspired by Foundation/Monster Stereo (CVPR 2025), this module leverages
pre-trained monocular depth estimation models to provide strong priors
for stereo matching, particularly useful in challenging scenarios.

Key features:
- Integration of monocular depth priors
- Multi-scale feature fusion
- Prior-guided cost volume construction
- Confidence-weighted refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class MonocularPriorEncoder(nn.Module):
    """
    Encodes monocular depth predictions into features for stereo matching.
    
    Args:
        depth_channels (int): Number of input channels from depth model (default: 1)
        feature_channels (int): Number of output feature channels (default: 64)
    """
    
    def __init__(self, depth_channels=1, feature_channels=64):
        super(MonocularPriorEncoder, self).__init__()
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Confidence estimation for depth prior
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, depth_prior):
        """
        Encode monocular depth prior.
        
        Args:
            depth_prior (torch.Tensor): Monocular depth prediction [B, 1, H, W]
        
        Returns:
            Tuple of (depth_features, confidence)
        """
        depth_features = self.depth_encoder(depth_prior)
        confidence = self.confidence_head(depth_features)
        return depth_features, confidence


class PriorGuidedCostVolume(nn.Module):
    """
    Constructs cost volume guided by monocular depth priors.
    
    Instead of uniform search across all disparities, this module focuses
    the search around the disparity predicted by monocular depth.
    
    Args:
        max_disparity (int): Maximum disparity range (default: 192)
        feature_channels (int): Number of feature channels (default: 32)
        search_range (int): Search range around prior (default: 32)
    """
    
    def __init__(self, max_disparity=192, feature_channels=32, search_range=32):
        super(PriorGuidedCostVolume, self).__init__()
        
        self.max_disparity = max_disparity
        self.feature_channels = feature_channels
        self.search_range = search_range
        
        # Cost aggregation with prior awareness
        self.cost_aggregator = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )
    
    def depth_to_disparity(self, depth, baseline=0.1, focal_length=1000.0):
        """
        Convert monocular depth to disparity estimate.
        
        Args:
            depth (torch.Tensor): Depth map [B, 1, H, W]
            baseline (float): Camera baseline
            focal_length (float): Camera focal length
        
        Returns:
            torch.Tensor: Disparity estimate [B, 1, H, W]
        """
        # Disparity = baseline * focal_length / depth
        disparity = (baseline * focal_length) / (depth + 1e-6)
        disparity = torch.clamp(disparity, 0, self.max_disparity)
        return disparity
    
    def build_prior_guided_cost_volume(
        self,
        features_left,
        features_right,
        disparity_prior,
        prior_confidence
    ):
        """
        Build cost volume focused around disparity prior.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            disparity_prior (torch.Tensor): Disparity from depth prior [B, 1, H, W]
            prior_confidence (torch.Tensor): Confidence in prior [B, 1, H, W]
        
        Returns:
            torch.Tensor: Cost volume [B, D, H, W]
        """
        batch_size, channels, height, width = features_left.shape
        device = features_left.device
        
        # For each pixel, search around the prior disparity
        cost_volume = torch.zeros(
            batch_size, self.max_disparity, height, width,
            dtype=features_left.dtype, device=device
        )
        
        # Full correlation for compatibility
        for d in range(self.max_disparity):
            if d == 0:
                shifted_right = features_right
            else:
                shifted_right = torch.zeros_like(features_right)
                shifted_right[:, :, :, d:] = features_right[:, :, :, :-d]
            
            # Compute correlation cost
            cost = torch.sum(features_left * shifted_right, dim=1)
            
            # Weight by prior confidence
            # High confidence -> focus on disparities near prior
            # Low confidence -> uniform search
            prior_weight = self._compute_prior_weight(
                d, disparity_prior, prior_confidence
            )
            cost = cost * prior_weight
            
            cost_volume[:, d, :, :] = cost
        
        return cost_volume
    
    def _compute_prior_weight(self, disparity_level, disparity_prior, confidence):
        """
        Compute weight based on distance from prior and confidence.
        
        Args:
            disparity_level (int): Current disparity being evaluated
            disparity_prior (torch.Tensor): Prior disparity [B, 1, H, W]
            confidence (torch.Tensor): Confidence [B, 1, H, W]
        
        Returns:
            torch.Tensor: Weight map [B, H, W]
        """
        # Distance from prior
        distance = torch.abs(disparity_level - disparity_prior)
        
        # Gaussian weighting centered at prior
        sigma = self.search_range / 3.0
        prior_weight = torch.exp(-distance**2 / (2 * sigma**2))
        
        # Blend with uniform based on confidence
        # High confidence -> use prior weight
        # Low confidence -> uniform weight
        weight = confidence * prior_weight + (1 - confidence) * 1.0
        
        return weight.squeeze(1)  # [B, H, W]
    
    def forward(
        self,
        features_left,
        features_right,
        disparity_prior,
        prior_confidence
    ):
        """
        Construct prior-guided cost volume.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            disparity_prior (torch.Tensor): Disparity prior [B, 1, H, W]
            prior_confidence (torch.Tensor): Confidence [B, 1, H, W]
        
        Returns:
            torch.Tensor: Aggregated cost volume [B, D, H, W]
        """
        # Build prior-guided cost volume
        cost_volume = self.build_prior_guided_cost_volume(
            features_left, features_right, disparity_prior, prior_confidence
        )
        
        # Apply 3D cost aggregation
        cost_volume_input = cost_volume.unsqueeze(1)  # [B, 1, D, H, W]
        aggregated_cost = self.cost_aggregator(cost_volume_input)
        cost_volume = aggregated_cost.squeeze(1)  # [B, D, H, W]
        
        return cost_volume


class FoundationStereo(nn.Module):
    """
    Foundation Stereo: Stereo matching with monocular depth priors.
    
    This model combines traditional stereo matching with strong monocular
    priors from pre-trained depth estimation models. The monocular prior
    provides:
    - Initial disparity estimate
    - Confidence map for guiding search
    - Semantic understanding of scene
    
    Args:
        max_disparity (int): Maximum disparity (default: 192)
        feature_channels (int): Feature dimension (default: 32)
        use_monocular_prior (bool): Enable monocular prior (default: True)
        monocular_model_name (str): Name of monocular depth model (default: 'dpt_hybrid')
        search_range (int): Search range around prior (default: 32)
    """
    
    def __init__(
        self,
        max_disparity=192,
        feature_channels=32,
        use_monocular_prior=True,
        monocular_model_name='dpt_hybrid',
        search_range=32
    ):
        super(FoundationStereo, self).__init__()
        
        self.max_disparity = max_disparity
        self.use_monocular_prior = use_monocular_prior
        self.monocular_model_name = monocular_model_name
        
        # Import feature extractor from existing URSM
        from .feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(output_channels=feature_channels)
        
        # Monocular prior encoder
        if self.use_monocular_prior:
            self.prior_encoder = MonocularPriorEncoder(
                depth_channels=1,
                feature_channels=64
            )
            
            self.prior_guided_cost_volume = PriorGuidedCostVolume(
                max_disparity=max_disparity,
                feature_channels=feature_channels,
                search_range=search_range
            )
        
        # Disparity regression head
        self.disparity_head = nn.Sequential(
            nn.Conv2d(max_disparity, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(max_disparity, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def get_monocular_depth(self, image):
        """
        Get monocular depth prediction.
        
        In practice, this would call a pre-trained monocular depth model
        like DPT, MiDaS, or Depth Anything. For now, we provide a placeholder
        that can be replaced with actual model inference.
        
        Args:
            image (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Depth prediction [B, 1, H, W]
        """
        # Placeholder: In real implementation, call pre-trained model
        # Example: depth = self.monocular_model(image)
        
        # For now, return dummy depth (will be replaced by actual model)
        batch_size, _, height, width = image.shape
        device = image.device
        
        # Simple heuristic: use image intensity as rough depth proxy
        # (This is just for demonstration - replace with real model!)
        depth = torch.mean(image, dim=1, keepdim=True)
        depth = F.interpolate(depth, size=(height, width), mode='bilinear', align_corners=True)
        
        return depth
    
    def forward(
        self,
        left_img,
        right_img,
        monocular_depth=None,
        baseline=0.1,
        focal_length=1000.0
    ):
        """
        Forward pass of Foundation Stereo.
        
        Args:
            left_img (torch.Tensor): Left image [B, 3, H, W]
            right_img (torch.Tensor): Right image [B, 3, H, W]
            monocular_depth (torch.Tensor, optional): Pre-computed depth [B, 1, H, W]
            baseline (float): Camera baseline
            focal_length (float): Camera focal length
        
        Returns:
            dict: Outputs containing:
                - 'disparity': Final disparity map [B, 1, H, W]
                - 'uncertainty': Uncertainty map [B, 1, H, W]
                - 'depth_prior': Monocular depth prior [B, 1, H, W]
                - 'disparity_prior': Disparity from depth [B, 1, H, W]
                - 'prior_confidence': Confidence in prior [B, 1, H, W]
        """
        batch_size, _, height, width = left_img.shape
        
        # Extract stereo features
        features_left = self.feature_extractor(left_img)
        features_right = self.feature_extractor(right_img)
        
        outputs = {
            'features_left': features_left,
            'features_right': features_right
        }
        
        # Get monocular depth prior
        if self.use_monocular_prior:
            if monocular_depth is None:
                # Get monocular depth prediction from left image
                monocular_depth = self.get_monocular_depth(left_img)
            
            # Resize depth to match feature resolution
            depth_resized = F.interpolate(
                monocular_depth,
                size=(features_left.shape[2], features_left.shape[3]),
                mode='bilinear',
                align_corners=True
            )
            
            # Encode depth prior
            depth_features, prior_confidence = self.prior_encoder(depth_resized)
            
            # Convert depth to disparity prior
            disparity_prior = self.prior_guided_cost_volume.depth_to_disparity(
                depth_resized, baseline, focal_length
            )
            
            # Build prior-guided cost volume
            cost_volume = self.prior_guided_cost_volume(
                features_left,
                features_right,
                disparity_prior,
                prior_confidence
            )
            
            outputs['depth_prior'] = monocular_depth
            outputs['disparity_prior'] = F.interpolate(
                disparity_prior, size=(height, width),
                mode='bilinear', align_corners=True
            )
            outputs['prior_confidence'] = F.interpolate(
                prior_confidence, size=(height, width),
                mode='bilinear', align_corners=True
            )
        else:
            # Standard cost volume without prior
            cost_volume = self._build_standard_cost_volume(
                features_left, features_right
            )
        
        # Regress disparity from cost volume
        disparity = self.disparity_head(cost_volume)
        
        # Upsample to input resolution
        disparity = F.interpolate(
            disparity, size=(height, width),
            mode='bilinear', align_corners=True
        )
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(cost_volume)
        uncertainty = F.interpolate(
            uncertainty, size=(height, width),
            mode='bilinear', align_corners=True
        )
        
        outputs['disparity'] = disparity
        outputs['uncertainty'] = uncertainty
        outputs['cost_volume'] = cost_volume
        
        return outputs
    
    def _build_standard_cost_volume(self, features_left, features_right):
        """Build standard correlation cost volume."""
        batch_size, channels, height, width = features_left.shape
        device = features_left.device
        
        cost_volume = torch.zeros(
            batch_size, self.max_disparity, height, width,
            dtype=features_left.dtype, device=device
        )
        
        for d in range(self.max_disparity):
            if d == 0:
                shifted_right = features_right
            else:
                shifted_right = torch.zeros_like(features_right)
                shifted_right[:, :, :, d:] = features_right[:, :, :, :-d]
            
            cost_volume[:, d, :, :] = torch.sum(
                features_left * shifted_right, dim=1
            )
        
        return cost_volume
    
    def load_monocular_model(self, model_name='dpt_hybrid'):
        """
        Load pre-trained monocular depth model.
        
        Supports models like:
        - DPT (Dense Prediction Transformer)
        - MiDaS
        - Depth Anything
        
        Args:
            model_name (str): Name of monocular model
        
        Returns:
            nn.Module: Loaded monocular depth model
        """
        # Placeholder for loading actual pre-trained models
        # In practice, this would load models from torch.hub or transformers
        
        # Example for DPT:
        # model = torch.hub.load('intel-isl/MiDaS', model_name)
        # return model
        
        print(f"Note: Monocular model '{model_name}' should be loaded here.")
        print("Replace get_monocular_depth() with actual model inference.")
        return None
