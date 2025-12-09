"""
Disparity refinement module with uncertainty estimation.

Refines initial disparity estimates using context features and provides
uncertainty estimates for each pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRU(nn.Module):
    """Convolutional GRU for iterative refinement."""
    
    def __init__(self, hidden_dim, input_dim):
        super(ConvGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Gates
        self.conv_gates = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim * 2,
            kernel_size=3, padding=1
        )
        self.conv_candidate = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim,
            kernel_size=3, padding=1
        )
    
    def forward(self, h, x):
        """
        Forward pass of ConvGRU.
        
        Args:
            h (torch.Tensor): Hidden state [B, H, H, W]
            x (torch.Tensor): Input [B, I, H, W]
        
        Returns:
            torch.Tensor: Updated hidden state [B, H, H, W]
        """
        combined = torch.cat([h, x], dim=1)
        
        # Compute gates
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        # Compute candidate
        combined_reset = torch.cat([reset_gate * h, x], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        
        # Update hidden state
        h_new = (1 - update_gate) * h + update_gate * candidate
        
        return h_new


class DisparityRefinement(nn.Module):
    """
    Iterative disparity refinement with uncertainty estimation.
    
    Uses a recurrent architecture to progressively refine disparity
    estimates while providing per-pixel uncertainty measures.
    
    Args:
        feature_channels (int): Number of feature channels (default: 32)
        hidden_dim (int): Hidden dimension for GRU (default: 128)
        iterations (int): Number of refinement iterations (default: 3)
    """
    
    def __init__(self, feature_channels=32, hidden_dim=128, iterations=3):
        super(DisparityRefinement, self).__init__()
        
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.iterations = iterations
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Disparity encoder
        self.disparity_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ConvGRU for iterative refinement
        self.gru = ConvGRU(
            hidden_dim=hidden_dim,
            input_dim=hidden_dim + 64  # context + disparity features
        )
        
        # Disparity update head
        self.disparity_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Uncertainty in [0, 1]
        )
    
    def forward(self, initial_disparity, features, cost_volume):
        """
        Refine disparity estimates iteratively.
        
        Args:
            initial_disparity (torch.Tensor): Initial disparity [B, 1, H, W]
            features (torch.Tensor): Image features [B, C, H, W]
            cost_volume (torch.Tensor): Cost volume [B, D, H, W]
        
        Returns:
            dict: Refined outputs containing:
                - 'disparity': Final disparity map [B, 1, H, W]
                - 'uncertainty': Uncertainty map [B, 1, H, W]
                - 'disparity_history': List of intermediate disparities
        """
        batch_size, _, height, width = initial_disparity.shape
        
        # Encode context from features
        context = self.context_encoder(features)
        
        # Initialize hidden state
        h = torch.zeros(
            batch_size, self.hidden_dim, height, width,
            dtype=features.dtype, device=features.device
        )
        
        # Current disparity
        disparity = initial_disparity
        disparity_history = [disparity]
        
        # Iterative refinement
        for i in range(self.iterations):
            # Encode current disparity
            disparity_features = self.disparity_encoder(disparity)
            
            # Concatenate context and disparity features
            gru_input = torch.cat([context, disparity_features], dim=1)
            
            # Update hidden state
            h = self.gru(h, gru_input)
            
            # Predict disparity update
            disparity_delta = self.disparity_head(h)
            
            # Update disparity
            disparity = disparity + disparity_delta
            
            # Clamp disparity to valid range
            disparity = torch.clamp(disparity, min=0.0)
            
            disparity_history.append(disparity)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(h)
        
        # Prepare outputs
        outputs = {
            'disparity': disparity,
            'uncertainty': uncertainty,
            'disparity_history': disparity_history
        }
        
        return outputs
