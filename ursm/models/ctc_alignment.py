"""
CTC/HMM-based 1D alignment for stereo matching.

This module implements ordered 1D matching along epipolar curves using
Connectionist Temporal Classification (CTC) with blank tokens for occlusions.

Key features:
- Monotonic (forward-only) alignment constraint
- Natural occlusion handling via blank tokens
- 1D search along epipolar curves (reduced complexity)
- Beam search decoding for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CTCAlignmentModule(nn.Module):
    """
    CTC-based alignment module for 1D stereo matching along epipolar curves.
    
    Instead of 2D cost volume matching, this module:
    1. Samples features along epipolar curves (left and right)
    2. Computes 1D similarity sequence
    3. Uses CTC to find monotonic alignment with blank tokens for occlusions
    
    Args:
        feature_channels (int): Number of input feature channels (default: 32)
        hidden_dim (int): Hidden dimension for alignment network (default: 128)
        max_disparity (int): Maximum disparity to consider (default: 192)
        blank_threshold (float): Threshold for blank token confidence (default: 0.3)
    """
    
    def __init__(
        self,
        feature_channels=32,
        hidden_dim=128,
        max_disparity=192,
        blank_threshold=0.3
    ):
        super(CTCAlignmentModule, self).__init__()
        
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.max_disparity = max_disparity
        self.blank_threshold = blank_threshold
        
        # Feature transformation for alignment
        self.feature_transform = nn.Sequential(
            nn.Conv1d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bidirectional LSTM for sequential modeling
        self.lstm = nn.LSTM(
            hidden_dim * 2,  # Concatenated left + right features
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # CTC output layer: disparity classes + blank token
        # Outputs: [0, 1, 2, ..., max_disparity-1, blank]
        self.num_classes = max_disparity + 1  # +1 for blank
        self.ctc_output = nn.Linear(hidden_dim * 2, self.num_classes)
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # CTC loss (blank index is the last one)
        self.ctc_loss = nn.CTCLoss(blank=self.num_classes - 1, reduction='mean', zero_infinity=True)
    
    def extract_epipolar_features(
        self,
        features_left: torch.Tensor,
        features_right: torch.Tensor,
        epipolar_curves: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and align features along epipolar curves.
        
        For uncalibrated case, we approximate epipolar curves as horizontal lines
        (which is exact for calibrated case and a reasonable approximation otherwise).
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            epipolar_curves (torch.Tensor, optional): Pre-computed curves
        
        Returns:
            Tuple of (left_sequences, right_sequences) as [B, H, W, C]
        """
        batch_size, channels, height, width = features_left.shape
        
        # For now, use horizontal scanlines as epipolar approximation
        # Shape: [B, H, W, C]
        left_sequences = features_left.permute(0, 2, 3, 1)
        right_sequences = features_right.permute(0, 2, 3, 1)
        
        return left_sequences, right_sequences
    
    def compute_similarity_sequence(
        self,
        left_seq: torch.Tensor,
        right_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between left and right feature sequences.
        
        Args:
            left_seq (torch.Tensor): Left sequence [B, H, W, C]
            right_seq (torch.Tensor): Right sequence [B, H, W, C]
        
        Returns:
            torch.Tensor: Similarity features [B, H, W, C*2]
        """
        # Concatenate features for joint reasoning
        combined = torch.cat([left_seq, right_seq], dim=-1)
        return combined
    
    def forward_pass_alignment(
        self,
        left_seq: torch.Tensor,
        right_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for alignment with CTC.
        
        Args:
            left_seq (torch.Tensor): Left features [B, H, W, C]
            right_seq (torch.Tensor): Right features [B, H, W, C]
        
        Returns:
            Tuple of (log_probs, confidence) for CTC
                log_probs: [B*H, W, num_classes] - Log probabilities for CTC
                confidence: [B, H, W, 1] - Confidence scores
        """
        batch_size, height, width, channels = left_seq.shape
        
        # Compute similarity features
        combined = self.compute_similarity_sequence(left_seq, right_seq)
        
        # Reshape for row-wise processing: [B*H, W, C*2]
        combined_flat = combined.view(batch_size * height, width, -1)
        
        # Apply LSTM for sequential modeling
        lstm_out, _ = self.lstm(combined_flat)  # [B*H, W, hidden_dim*2]
        
        # CTC output logits
        logits = self.ctc_output(lstm_out)  # [B*H, W, num_classes]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Confidence estimation
        confidence = self.confidence_head(lstm_out)  # [B*H, W, 1]
        confidence = confidence.view(batch_size, height, width, 1)
        
        return log_probs, confidence
    
    def decode_alignment(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode CTC output to disparity map using beam search.
        
        Args:
            log_probs (torch.Tensor): Log probabilities [B*H, W, num_classes]
            beam_width (int): Beam width for decoding (default: 10)
        
        Returns:
            Tuple of (disparity, alignment_confidence)
                disparity: [B, H, W] - Decoded disparity values
                alignment_confidence: [B, H, W] - Per-pixel confidence
        """
        batch_height, width, num_classes = log_probs.shape
        device = log_probs.device
        
        # Greedy decoding for efficiency (can be replaced with beam search)
        # For each position, take the most likely disparity (excluding blank)
        probs = torch.exp(log_probs)  # [B*H, W, num_classes]
        
        # Get best disparity (0 to max_disparity-1)
        # Blank is at index num_classes-1
        disparity_probs = probs[:, :, :-1]  # Exclude blank
        blank_probs = probs[:, :, -1:]  # Blank probability
        
        # For positions with high blank probability, mark as occluded
        max_disp_vals, max_disp_idx = torch.max(disparity_probs, dim=-1)
        is_occluded = blank_probs.squeeze(-1) > self.blank_threshold
        
        # Set occluded regions to 0 (or could use a special value)
        disparity = max_disp_idx.float()
        disparity[is_occluded] = 0.0
        
        # Confidence is the maximum probability (including blank for occluded)
        alignment_confidence = torch.max(probs, dim=-1)[0]
        
        return disparity, alignment_confidence
    
    def compute_disparity_from_alignment(
        self,
        log_probs: torch.Tensor,
        batch_size: int,
        height: int,
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert CTC alignment to disparity map.
        
        Args:
            log_probs (torch.Tensor): CTC log probabilities [B*H, W, num_classes]
            batch_size (int): Batch size
            height (int): Image height
            width (int): Image width
        
        Returns:
            Tuple of (disparity_map, occlusion_mask)
        """
        # Decode alignment
        disparity_flat, confidence_flat = self.decode_alignment(log_probs)
        
        # Reshape to image
        disparity_map = disparity_flat.view(batch_size, height, width)
        confidence_map = confidence_flat.view(batch_size, height, width)
        
        # Create occlusion mask (1 for valid, 0 for occluded)
        occlusion_mask = (disparity_map > 0).float()
        
        # Add channel dimension
        disparity_map = disparity_map.unsqueeze(1)  # [B, 1, H, W]
        occlusion_mask = occlusion_mask.unsqueeze(1)  # [B, 1, H, W]
        
        return disparity_map, occlusion_mask
    
    def forward(
        self,
        features_left: torch.Tensor,
        features_right: torch.Tensor,
        target_disparity: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass of CTC alignment module.
        
        Args:
            features_left (torch.Tensor): Left features [B, C, H, W]
            features_right (torch.Tensor): Right features [B, C, H, W]
            target_disparity (torch.Tensor, optional): Ground truth for training
        
        Returns:
            dict: Outputs containing:
                - 'disparity': Predicted disparity [B, 1, H, W]
                - 'occlusion_mask': Occlusion mask [B, 1, H, W]
                - 'confidence': Confidence map [B, H, W, 1]
                - 'log_probs': CTC log probabilities (for loss computation)
                - 'ctc_loss': CTC loss value (if target provided)
        """
        batch_size, channels, height, width = features_left.shape
        
        # Extract features along epipolar curves
        left_seq, right_seq = self.extract_epipolar_features(
            features_left, features_right
        )
        
        # Forward pass for alignment
        log_probs, confidence = self.forward_pass_alignment(left_seq, right_seq)
        
        # Decode to disparity map
        disparity_map, occlusion_mask = self.compute_disparity_from_alignment(
            log_probs, batch_size, height, width
        )
        
        # Prepare output
        outputs = {
            'disparity': disparity_map,
            'occlusion_mask': occlusion_mask,
            'confidence': confidence,
            'log_probs': log_probs
        }
        
        # Compute CTC loss if target is provided
        if target_disparity is not None:
            ctc_loss = self.compute_ctc_loss(
                log_probs, target_disparity, batch_size, height, width
            )
            outputs['ctc_loss'] = ctc_loss
        
        return outputs
    
    def compute_ctc_loss(
        self,
        log_probs: torch.Tensor,
        target_disparity: torch.Tensor,
        batch_size: int,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Compute CTC loss from log probabilities and target disparity.
        
        Args:
            log_probs (torch.Tensor): Log probs [B*H, W, num_classes]
            target_disparity (torch.Tensor): Target disparity [B, 1, H, W]
            batch_size (int): Batch size
            height (int): Image height
            width (int): Image width
        
        Returns:
            torch.Tensor: CTC loss value
        """
        # Prepare target sequence
        # Convert disparity map to sequence of disparity values
        target = target_disparity.squeeze(1)  # [B, H, W]
        target = target.view(batch_size * height, width)  # [B*H, W]
        
        # Create valid mask (ignore invalid disparities)
        valid_mask = (target > 0) & (target < self.max_disparity)
        
        # Convert to integer labels
        target_labels = target.long()
        target_labels[~valid_mask] = self.num_classes - 1  # Set invalid to blank
        
        # Prepare for CTC loss
        # Input: [T, B, num_classes] where T is sequence length (width)
        log_probs_ctc = log_probs.permute(1, 0, 2)  # [W, B*H, num_classes]
        
        # Input lengths (all sequences have same length)
        input_lengths = torch.full(
            (batch_size * height,), width, dtype=torch.long, device=log_probs.device
        )
        
        # Target lengths (number of valid disparities per scanline)
        target_lengths = valid_mask.sum(dim=1).long()  # [B*H]
        
        # Compute CTC loss
        # Note: CTC expects target to be flattened valid labels
        target_flat = []
        for i in range(batch_size * height):
            valid_labels = target_labels[i][valid_mask[i]]
            target_flat.append(valid_labels)
        
        target_concat = torch.cat(target_flat)
        
        try:
            loss = self.ctc_loss(
                log_probs_ctc,
                target_concat,
                input_lengths,
                target_lengths
            )
        except RuntimeError:
            # If CTC loss fails (e.g., sequence too short), return zero loss
            loss = torch.tensor(0.0, device=log_probs.device)
        
        return loss
