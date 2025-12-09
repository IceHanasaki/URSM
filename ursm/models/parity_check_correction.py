"""
Sparse parity check error correction for stereo matching.

Inspired by Low-Density Parity-Check (LDPC) codes, this module uses
belief propagation to correct errors in disparity estimates by exploiting
spatial consistency constraints.

Key features:
- Sparse parity check matrix for computational efficiency
- Belief propagation for iterative error correction
- Integration with CTC alignment confidence scores
- Geometric consistency enforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SparseParityCheckCorrection(nn.Module):
    """
    Sparse parity check error correction for disparity maps.
    
    Uses belief propagation on a sparse graph to correct disparity errors
    by enforcing geometric consistency constraints (e.g., smoothness, ordering).
    
    Args:
        max_disparity (int): Maximum disparity value (default: 192)
        check_density (float): Density of parity check matrix (default: 0.1)
        num_iterations (int): Number of belief propagation iterations (default: 5)
        smoothness_weight (float): Weight for smoothness constraint (default: 1.0)
    """
    
    def __init__(
        self,
        max_disparity=192,
        check_density=0.1,
        num_iterations=5,
        smoothness_weight=1.0
    ):
        super(SparseParityCheckCorrection, self).__init__()
        
        self.max_disparity = max_disparity
        self.check_density = check_density
        self.num_iterations = num_iterations
        self.smoothness_weight = smoothness_weight
        
        # Learnable parameters for belief propagation
        self.message_weight = nn.Parameter(torch.tensor(1.0))
        self.damping_factor = nn.Parameter(torch.tensor(0.5))
    
    def create_parity_check_graph(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sparse parity check graph for spatial consistency.
        
        The graph connects neighboring pixels with parity check constraints:
        - Horizontal neighbors should have similar disparity (smoothness)
        - Vertical neighbors should have similar disparity (smoothness)
        - Ordering constraint: left-to-right disparity should be non-increasing
        
        Args:
            height (int): Image height
            width (int): Image width
            device (torch.device): Device for tensors
        
        Returns:
            Tuple of (edge_index, edge_type)
                edge_index: [2, num_edges] - Graph connectivity
                edge_type: [num_edges] - Type of constraint (0: h-smooth, 1: v-smooth, 2: order)
        """
        edges = []
        edge_types = []
        
        num_pixels = height * width
        
        # Horizontal smoothness constraints
        for i in range(height):
            for j in range(width - 1):
                idx1 = i * width + j
                idx2 = i * width + (j + 1)
                
                # Sample with probability check_density
                if torch.rand(1).item() < self.check_density:
                    edges.append([idx1, idx2])
                    edge_types.append(0)  # Horizontal smoothness
        
        # Vertical smoothness constraints
        for i in range(height - 1):
            for j in range(width):
                idx1 = i * width + j
                idx2 = (i + 1) * width + j
                
                if torch.rand(1).item() < self.check_density:
                    edges.append([idx1, idx2])
                    edge_types.append(1)  # Vertical smoothness
        
        # Ordering constraints (left-to-right non-increasing)
        for i in range(height):
            for j in range(0, width - 2, 2):  # Sample every other pair
                idx1 = i * width + j
                idx2 = i * width + (j + 2)
                
                if torch.rand(1).item() < self.check_density * 0.5:
                    edges.append([idx1, idx2])
                    edge_types.append(2)  # Ordering constraint
        
        if len(edges) == 0:
            # Fallback: at least some edges
            edges.append([0, 1])
            edge_types.append(0)
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
        
        return edge_index, edge_type
    
    def compute_parity_check_violations(
        self,
        disparity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute parity check violations for each edge.
        
        Args:
            disparity (torch.Tensor): Disparity map [B, 1, H, W]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            edge_type (torch.Tensor): Edge types [num_edges]
        
        Returns:
            torch.Tensor: Violation scores [B, num_edges]
        """
        batch_size = disparity.shape[0]
        num_edges = edge_index.shape[1]
        device = disparity.device
        
        # Flatten disparity
        disp_flat = disparity.view(batch_size, -1)  # [B, H*W]
        
        # Get disparities at edge endpoints
        src_disp = disp_flat[:, edge_index[0]]  # [B, num_edges]
        dst_disp = disp_flat[:, edge_index[1]]  # [B, num_edges]
        
        # Compute violations based on edge type
        violations = torch.zeros(batch_size, num_edges, device=device)
        
        # Type 0 & 1: Smoothness (horizontal and vertical)
        smooth_mask = (edge_type == 0) | (edge_type == 1)
        if smooth_mask.any():
            # Violation is the absolute difference
            violations[:, smooth_mask] = torch.abs(
                src_disp[:, smooth_mask] - dst_disp[:, smooth_mask]
            )
        
        # Type 2: Ordering constraint (src >= dst)
        order_mask = edge_type == 2
        if order_mask.any():
            # Violation if src < dst (should be non-increasing)
            violations[:, order_mask] = F.relu(
                dst_disp[:, order_mask] - src_disp[:, order_mask]
            )
        
        return violations
    
    def belief_propagation_step(
        self,
        beliefs: torch.Tensor,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of belief propagation for error correction.
        
        Args:
            beliefs (torch.Tensor): Current beliefs [B, H*W, max_disp]
            messages (torch.Tensor): Messages on edges [B, num_edges, max_disp]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            edge_type (torch.Tensor): Edge types [num_edges]
            confidence (torch.Tensor): Confidence scores [B, H*W]
        
        Returns:
            Tuple of (updated_beliefs, updated_messages)
        """
        batch_size, num_pixels, num_disparities = beliefs.shape
        num_edges = edge_index.shape[1]
        device = beliefs.device
        
        # Compute new messages from source to destination
        new_messages = torch.zeros_like(messages)
        
        for e in range(num_edges):
            src_idx = edge_index[0, e]
            dst_idx = edge_index[1, e]
            etype = edge_type[e]
            
            # Get source beliefs
            src_belief = beliefs[:, src_idx, :]  # [B, max_disp]
            
            # Compute message based on edge type
            if etype == 0 or etype == 1:  # Smoothness
                # Message favors similar disparities
                # Create compatibility matrix (Gaussian kernel)
                disp_range = torch.arange(num_disparities, device=device).float()
                disp_diff = (disp_range.unsqueeze(1) - disp_range.unsqueeze(0)).abs()
                compatibility = torch.exp(-disp_diff**2 / (2 * self.smoothness_weight**2))
                
                # Message is belief convolved with compatibility
                message = torch.matmul(src_belief.unsqueeze(1), compatibility).squeeze(1)
                
            elif etype == 2:  # Ordering
                # Message enforces non-increasing constraint
                # For dst, only allow disparities <= src disparity
                src_max_disp = torch.argmax(src_belief, dim=-1)  # [B]
                message = torch.zeros(batch_size, num_disparities, device=device)
                
                for b in range(batch_size):
                    max_d = int(src_max_disp[b].item())
                    if max_d < num_disparities:
                        message[b, :max_d+1] = 1.0
                
                message = message * src_belief  # Combine with source belief
            
            else:
                message = src_belief
            
            # Normalize message
            message = message / (message.sum(dim=-1, keepdim=True) + 1e-8)
            
            new_messages[:, e, :] = message
        
        # Update beliefs by combining messages
        new_beliefs = beliefs.clone()
        
        # For each node, aggregate incoming messages
        for dst_idx in range(num_pixels):
            # Find incoming edges
            incoming_mask = edge_index[1] == dst_idx
            
            if incoming_mask.any():
                incoming_messages = new_messages[:, incoming_mask, :]  # [B, num_in, max_disp]
                
                # Multiply messages (in log space for stability)
                log_beliefs = torch.log(beliefs[:, dst_idx, :] + 1e-8)
                log_messages = torch.log(incoming_messages + 1e-8).sum(dim=1)
                
                # Combine with confidence
                conf_weight = confidence[:, dst_idx].unsqueeze(-1)  # [B, 1]
                
                # Update belief
                log_belief_new = (
                    conf_weight * log_beliefs + 
                    (1 - conf_weight) * self.message_weight * log_messages
                )
                
                new_beliefs[:, dst_idx, :] = torch.exp(log_belief_new)
                
                # Normalize
                new_beliefs[:, dst_idx, :] = new_beliefs[:, dst_idx, :] / (
                    new_beliefs[:, dst_idx, :].sum(dim=-1, keepdim=True) + 1e-8
                )
        
        # Damping: blend old and new messages
        damping = torch.clamp(self.damping_factor, 0.0, 1.0)
        new_messages = damping * messages + (1 - damping) * new_messages
        
        return new_beliefs, new_messages
    
    def forward(
        self,
        disparity: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparse parity check error correction to disparity map.
        
        Args:
            disparity (torch.Tensor): Initial disparity [B, 1, H, W]
            confidence (torch.Tensor): Confidence scores [B, H, W, 1] or [B, H*W]
        
        Returns:
            Tuple of (corrected_disparity, correction_confidence)
                corrected_disparity: [B, 1, H, W]
                correction_confidence: [B, 1, H, W]
        """
        batch_size, _, height, width = disparity.shape
        device = disparity.device
        num_pixels = height * width
        
        # Prepare confidence
        if confidence.dim() == 4:
            confidence_flat = confidence.view(batch_size, -1)  # [B, H*W]
        else:
            confidence_flat = confidence.view(batch_size, -1)
        
        # Create parity check graph
        edge_index, edge_type = self.create_parity_check_graph(height, width, device)
        num_edges = edge_index.shape[1]
        
        # Initialize beliefs as one-hot at current disparity
        beliefs = torch.zeros(batch_size, num_pixels, self.max_disparity, device=device)
        
        disp_flat = disparity.view(batch_size, -1).long()  # [B, H*W]
        disp_flat = torch.clamp(disp_flat, 0, self.max_disparity - 1)
        
        # Set initial beliefs
        for b in range(batch_size):
            beliefs[b, torch.arange(num_pixels), disp_flat[b]] = 1.0
        
        # Add uncertainty based on confidence
        uncertainty = 1.0 - confidence_flat.unsqueeze(-1)  # [B, H*W, 1]
        beliefs = (1 - uncertainty) * beliefs + uncertainty / self.max_disparity
        
        # Initialize messages
        messages = torch.ones(batch_size, num_edges, self.max_disparity, device=device) / self.max_disparity
        
        # Run belief propagation
        for iteration in range(self.num_iterations):
            beliefs, messages = self.belief_propagation_step(
                beliefs, messages, edge_index, edge_type, confidence_flat
            )
        
        # Extract corrected disparity (MAP estimate)
        corrected_disp_flat = torch.argmax(beliefs, dim=-1).float()  # [B, H*W]
        
        # Reshape to image
        corrected_disparity = corrected_disp_flat.view(batch_size, 1, height, width)
        
        # Compute correction confidence (max belief value)
        correction_confidence = torch.max(beliefs, dim=-1)[0]  # [B, H*W]
        correction_confidence = correction_confidence.view(batch_size, 1, height, width)
        
        return corrected_disparity, correction_confidence
