"""
URSMNet: Main network architecture for unrectified stereo matching.

This network handles stereo matching with imperfect calibration by:
1. Extracting robust features from both views
2. Estimating calibration parameters
3. Building adaptive cost volumes
4. Refining disparity predictions with uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeatureExtractor
from .calibration_estimator import CalibrationEstimator
from .cost_volume import AdaptiveCostVolume
from .disparity_refinement import DisparityRefinement


class URSMNet(nn.Module):
    """
    Unrectified Stereo Matching Network.
    
    A foundation model that addresses stereo matching under imperfect calibration
    by jointly estimating calibration parameters and disparity maps.
    
    Args:
        max_disparity (int): Maximum disparity range (default: 192)
        feature_channels (int): Number of feature channels (default: 32)
        refine_iterations (int): Number of refinement iterations (default: 3)
        estimate_calibration (bool): Whether to estimate calibration parameters (default: True)
    """
    
    def __init__(
        self,
        max_disparity=192,
        feature_channels=32,
        refine_iterations=3,
        estimate_calibration=True
    ):
        super(URSMNet, self).__init__()
        
        self.max_disparity = max_disparity
        self.feature_channels = feature_channels
        self.refine_iterations = refine_iterations
        self.estimate_calibration = estimate_calibration
        
        # Feature extraction for left and right images
        self.feature_extractor = FeatureExtractor(
            output_channels=feature_channels
        )
        
        # Calibration parameter estimation
        if self.estimate_calibration:
            self.calibration_estimator = CalibrationEstimator(
                feature_channels=feature_channels
            )
        
        # Adaptive cost volume construction
        self.cost_volume = AdaptiveCostVolume(
            max_disparity=max_disparity,
            feature_channels=feature_channels
        )
        
        # Disparity refinement with uncertainty
        self.disparity_refinement = DisparityRefinement(
            feature_channels=feature_channels,
            iterations=refine_iterations
        )
    
    def forward(self, left_img, right_img, calibration_params=None):
        """
        Forward pass of URSMNet.
        
        Args:
            left_img (torch.Tensor): Left image tensor [B, 3, H, W]
            right_img (torch.Tensor): Right image tensor [B, 3, H, W]
            calibration_params (dict, optional): Known calibration parameters
        
        Returns:
            dict: Dictionary containing:
                - 'disparity': Predicted disparity map [B, 1, H, W]
                - 'uncertainty': Uncertainty map [B, 1, H, W]
                - 'calibration': Estimated calibration parameters (if enabled)
                - 'features_left': Left image features
                - 'features_right': Right image features
        """
        batch_size, _, height, width = left_img.shape
        
        # Extract features from both images
        features_left = self.feature_extractor(left_img)
        features_right = self.feature_extractor(right_img)
        
        # Estimate calibration parameters if enabled
        estimated_calibration = None
        if self.estimate_calibration and calibration_params is None:
            estimated_calibration = self.calibration_estimator(
                features_left, features_right
            )
            calibration_params = estimated_calibration
        
        # Build adaptive cost volume
        cost_volume = self.cost_volume(
            features_left,
            features_right,
            calibration_params
        )
        
        # Initial disparity estimation from cost volume
        # Use soft argmin for differentiable disparity
        disparity_values = torch.arange(
            0, self.max_disparity,
            dtype=torch.float32,
            device=left_img.device
        ).view(1, self.max_disparity, 1, 1)
        
        # Compute probability distribution over disparities
        prob_volume = F.softmax(-cost_volume, dim=1)
        initial_disparity = torch.sum(
            prob_volume * disparity_values,
            dim=1,
            keepdim=True
        )
        
        # Refine disparity with context and uncertainty
        refined_outputs = self.disparity_refinement(
            initial_disparity,
            features_left,
            cost_volume
        )
        
        # Prepare output dictionary
        outputs = {
            'disparity': refined_outputs['disparity'],
            'uncertainty': refined_outputs['uncertainty'],
            'features_left': features_left,
            'features_right': features_right,
            'cost_volume': cost_volume,
            'prob_volume': prob_volume
        }
        
        if estimated_calibration is not None:
            outputs['calibration'] = estimated_calibration
        
        return outputs
    
    def extract_features(self, image):
        """
        Extract features from a single image.
        
        Args:
            image (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Extracted features
        """
        return self.feature_extractor(image)
