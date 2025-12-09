"""
URSM-CTC: Unrectified Stereo Matching with CTC/HMM 1D Alignment.

This is an alternative architecture that replaces 2D cost volume matching
with CTC-based 1D alignment along epipolar curves, combined with sparse
parity check error correction.

Key innovations:
1. Transform 2D matching → 1D ordered matching along epipolar curves
2. CTC/HMM with blank tokens for natural occlusion handling
3. Monotonic (forward-only) constraint eliminates impossible matches
4. Sparse parity check error correction for geometric consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeatureExtractor
from .ctc_alignment import CTCAlignmentModule
from .parity_check_correction import SparseParityCheckCorrection


class URSMNetCTC(nn.Module):
    """
    URSM with CTC-based 1D alignment.
    
    This architecture directly performs robust matching on uncalibrated images
    without explicit calibration estimation, using:
    - 1D alignment along epipolar curves (reduced computational cost)
    - CTC with blank tokens (natural occlusion handling)
    - Sparse parity check error correction (geometric consistency)
    
    Args:
        max_disparity (int): Maximum disparity range (default: 192)
        feature_channels (int): Number of feature channels (default: 32)
        hidden_dim (int): Hidden dimension for CTC alignment (default: 128)
        blank_threshold (float): Threshold for blank token (default: 0.3)
        use_error_correction (bool): Enable parity check correction (default: True)
        correction_iterations (int): Number of BP iterations (default: 5)
    """
    
    def __init__(
        self,
        max_disparity=192,
        feature_channels=32,
        hidden_dim=128,
        blank_threshold=0.3,
        use_error_correction=True,
        correction_iterations=5
    ):
        super(URSMNetCTC, self).__init__()
        
        self.max_disparity = max_disparity
        self.feature_channels = feature_channels
        self.use_error_correction = use_error_correction
        
        # Feature extraction (shared with original URSM)
        self.feature_extractor = FeatureExtractor(
            output_channels=feature_channels
        )
        
        # CTC-based 1D alignment module
        self.ctc_alignment = CTCAlignmentModule(
            feature_channels=feature_channels,
            hidden_dim=hidden_dim,
            max_disparity=max_disparity,
            blank_threshold=blank_threshold
        )
        
        # Sparse parity check error correction (optional)
        if self.use_error_correction:
            self.error_correction = SparseParityCheckCorrection(
                max_disparity=max_disparity,
                check_density=0.1,
                num_iterations=correction_iterations,
                smoothness_weight=1.0
            )
    
    def forward(self, left_img, right_img, target_disparity=None):
        """
        Forward pass of URSM-CTC.
        
        Args:
            left_img (torch.Tensor): Left image [B, 3, H, W]
            right_img (torch.Tensor): Right image [B, 3, H, W]
            target_disparity (torch.Tensor, optional): Ground truth for training
        
        Returns:
            dict: Outputs containing:
                - 'disparity': Final disparity map [B, 1, H, W]
                - 'disparity_ctc': Raw CTC output [B, 1, H, W]
                - 'occlusion_mask': Detected occlusions [B, 1, H, W]
                - 'confidence': Confidence scores
                - 'ctc_loss': CTC loss (if target provided)
        """
        batch_size, _, height, width = left_img.shape
        
        # Extract features from both images
        features_left = self.feature_extractor(left_img)
        features_right = self.feature_extractor(right_img)
        
        # CTC-based 1D alignment
        ctc_outputs = self.ctc_alignment(
            features_left,
            features_right,
            target_disparity=target_disparity
        )
        
        disparity_ctc = ctc_outputs['disparity']
        occlusion_mask = ctc_outputs['occlusion_mask']
        confidence = ctc_outputs['confidence']
        
        # Apply error correction if enabled
        if self.use_error_correction:
            # Reshape confidence for error correction
            if confidence.dim() == 4 and confidence.shape[-1] == 1:
                confidence_for_correction = confidence
            else:
                # confidence is [B, H, W, 1], reshape to [B, H*W]
                conf_flat = confidence.view(batch_size, -1)
                confidence_for_correction = conf_flat
            
            disparity_corrected, correction_confidence = self.error_correction(
                disparity_ctc,
                confidence_for_correction
            )
        else:
            disparity_corrected = disparity_ctc
            correction_confidence = None
        
        # Prepare outputs
        outputs = {
            'disparity': disparity_corrected,
            'disparity_ctc': disparity_ctc,
            'occlusion_mask': occlusion_mask,
            'confidence': confidence,
            'features_left': features_left,
            'features_right': features_right,
            'log_probs': ctc_outputs['log_probs']
        }
        
        if correction_confidence is not None:
            outputs['correction_confidence'] = correction_confidence
        
        if 'ctc_loss' in ctc_outputs:
            outputs['ctc_loss'] = ctc_outputs['ctc_loss']
        
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


class URSMNetCTCLoss(nn.Module):
    """
    Combined loss for URSM-CTC training.
    
    Combines:
    1. CTC alignment loss (from CTC module)
    2. Disparity smoothness loss
    3. Occlusion consistency loss
    
    Args:
        ctc_weight (float): Weight for CTC loss (default: 1.0)
        smooth_weight (float): Weight for smoothness loss (default: 0.1)
        occlusion_weight (float): Weight for occlusion loss (default: 0.05)
    """
    
    def __init__(
        self,
        ctc_weight=1.0,
        smooth_weight=0.1,
        occlusion_weight=0.05
    ):
        super(URSMNetCTCLoss, self).__init__()
        
        self.ctc_weight = ctc_weight
        self.smooth_weight = smooth_weight
        self.occlusion_weight = occlusion_weight
    
    def compute_smoothness_loss(self, disparity, image):
        """
        Compute edge-aware smoothness loss.
        
        Args:
            disparity (torch.Tensor): Disparity map [B, 1, H, W]
            image (torch.Tensor): Image for edge detection [B, 3, H, W]
        
        Returns:
            torch.Tensor: Smoothness loss
        """
        # Compute disparity gradients
        disp_grad_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        disp_grad_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])
        
        # Compute image gradients for edge-awareness
        img_grad_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        img_grad_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Edge-aware weighting
        weight_x = torch.exp(-img_grad_x)
        weight_y = torch.exp(-img_grad_y)
        
        # Weighted smoothness
        smooth_loss = (disp_grad_x * weight_x).mean() + (disp_grad_y * weight_y).mean()
        
        return smooth_loss
    
    def compute_occlusion_loss(self, occlusion_mask, disparity):
        """
        Encourage occlusion mask to be consistent.
        
        Args:
            occlusion_mask (torch.Tensor): Occlusion mask [B, 1, H, W]
            disparity (torch.Tensor): Disparity map [B, 1, H, W]
        
        Returns:
            torch.Tensor: Occlusion consistency loss
        """
        # Occlusions should be spatially sparse but clustered
        # Penalize isolated occlusions
        
        # Compute spatial variance of occlusion mask
        occ_grad_x = torch.abs(occlusion_mask[:, :, :, :-1] - occlusion_mask[:, :, :, 1:])
        occ_grad_y = torch.abs(occlusion_mask[:, :, :-1, :] - occlusion_mask[:, :, 1:, :])
        
        # Penalize high frequency changes (encourages clustering)
        occlusion_loss = (occ_grad_x.mean() + occ_grad_y.mean())
        
        return occlusion_loss
    
    def forward(self, outputs, target_disparity, image):
        """
        Compute combined loss.
        
        Args:
            outputs (dict): Model outputs
            target_disparity (torch.Tensor): Ground truth disparity
            image (torch.Tensor): Input image (for smoothness)
        
        Returns:
            dict: Loss components
        """
        losses = {}
        total_loss = 0.0
        
        # CTC loss (from alignment module)
        if 'ctc_loss' in outputs:
            ctc_loss = outputs['ctc_loss']
            losses['ctc_loss'] = ctc_loss
            total_loss += self.ctc_weight * ctc_loss
        
        # Smoothness loss
        disparity = outputs['disparity']
        smooth_loss = self.compute_smoothness_loss(disparity, image)
        losses['smooth_loss'] = smooth_loss
        total_loss += self.smooth_weight * smooth_loss
        
        # Occlusion consistency loss
        if 'occlusion_mask' in outputs:
            occlusion_mask = outputs['occlusion_mask']
            occ_loss = self.compute_occlusion_loss(occlusion_mask, disparity)
            losses['occlusion_loss'] = occ_loss
            total_loss += self.occlusion_weight * occ_loss
        
        losses['total_loss'] = total_loss
        
        return losses
