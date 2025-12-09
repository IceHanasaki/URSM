"""
Geometric utilities for stereo vision.
"""

import torch
import torch.nn.functional as F
import numpy as np


def warp_image(image, disparity, direction='right_to_left'):
    """
    Warp image using disparity map.
    
    Args:
        image (torch.Tensor): Image to warp [B, C, H, W]
        disparity (torch.Tensor): Disparity map [B, 1, H, W]
        direction (str): Warping direction ('right_to_left' or 'left_to_right')
    
    Returns:
        torch.Tensor: Warped image [B, C, H, W]
    """
    batch_size, channels, height, width = image.shape
    device = image.device
    
    # Create coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
    
    # Compute disparity offset in normalized coordinates
    disparity_normalized = disparity * (2.0 / width)
    
    if direction == 'right_to_left':
        # Shift left (negative disparity)
        grid_x = grid_x - disparity_normalized
    else:  # left_to_right
        # Shift right (positive disparity)
        grid_x = grid_x + disparity_normalized
    
    # Stack grid
    grid = torch.cat([grid_x, grid_y], dim=1)  # [B, 2, H, W]
    grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    # Warp image
    warped = F.grid_sample(
        image, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return warped


def compute_fundamental_matrix(
    calibration_params,
    baseline=1.0,
    focal_length=1000.0
):
    """
    Compute fundamental matrix from calibration parameters.
    
    Args:
        calibration_params (dict): Calibration parameters containing:
            - 'vertical_offset': Vertical offset
            - 'rotation': Rotation angles
        baseline (float): Camera baseline (default: 1.0)
        focal_length (float): Focal length (default: 1000.0)
    
    Returns:
        torch.Tensor: Fundamental matrix [B, 3, 3]
    """
    batch_size = calibration_params['vertical_offset'].shape[0]
    device = calibration_params['vertical_offset'].device
    
    # Extract parameters
    vertical_offset = calibration_params['vertical_offset']  # [B, 1]
    rotation = calibration_params['rotation']  # [B, 3]
    
    # Initialize fundamental matrix
    F = torch.zeros(batch_size, 3, 3, device=device)
    
    # Simplified fundamental matrix for rectified stereo with vertical offset
    # F = [0, 0, v_offset]
    #     [0, 0, -1      ]
    #     [-v_offset, 1, 0]
    
    F[:, 0, 2] = vertical_offset.squeeze(1)
    F[:, 1, 2] = -1.0
    F[:, 2, 0] = -vertical_offset.squeeze(1)
    F[:, 2, 1] = 1.0
    
    return F


def compute_epipolar_error(
    points_left,
    points_right,
    fundamental_matrix
):
    """
    Compute epipolar error for point correspondences.
    
    Args:
        points_left (torch.Tensor): Points in left image [B, N, 2]
        points_right (torch.Tensor): Points in right image [B, N, 2]
        fundamental_matrix (torch.Tensor): Fundamental matrix [B, 3, 3]
    
    Returns:
        torch.Tensor: Epipolar errors [B, N]
    """
    batch_size, num_points, _ = points_left.shape
    
    # Convert to homogeneous coordinates
    points_left_h = torch.cat([
        points_left,
        torch.ones(batch_size, num_points, 1, device=points_left.device)
    ], dim=2)  # [B, N, 3]
    
    points_right_h = torch.cat([
        points_right,
        torch.ones(batch_size, num_points, 1, device=points_right.device)
    ], dim=2)  # [B, N, 3]
    
    # Compute epipolar lines: l = F * p_left
    epipolar_lines = torch.bmm(
        fundamental_matrix,
        points_left_h.transpose(1, 2)
    ).transpose(1, 2)  # [B, N, 3]
    
    # Compute distance: d = |p_right^T * l| / sqrt(l_x^2 + l_y^2)
    numerator = torch.abs(torch.sum(points_right_h * epipolar_lines, dim=2))
    denominator = torch.sqrt(
        epipolar_lines[:, :, 0]**2 + epipolar_lines[:, :, 1]**2 + 1e-8
    )
    
    epipolar_error = numerator / denominator
    
    return epipolar_error


def triangulate_points(
    disparity,
    baseline=1.0,
    focal_length=1000.0,
    min_depth=0.1,
    max_depth=100.0
):
    """
    Triangulate 3D points from disparity map.
    
    Args:
        disparity (torch.Tensor): Disparity map [B, 1, H, W]
        baseline (float): Camera baseline (default: 1.0)
        focal_length (float): Focal length (default: 1000.0)
        min_depth (float): Minimum valid depth (default: 0.1)
        max_depth (float): Maximum valid depth (default: 100.0)
    
    Returns:
        torch.Tensor: Depth map [B, 1, H, W]
    """
    # Depth = (baseline * focal_length) / disparity
    depth = (baseline * focal_length) / (disparity + 1e-8)
    
    # Clip to specified range
    depth = torch.clamp(depth, min=min_depth, max=max_depth)
    
    return depth


def create_occlusion_mask(left_disparity, right_disparity, threshold=1.0):
    """
    Create occlusion mask using left-right consistency check.
    
    Args:
        left_disparity (torch.Tensor): Left disparity map [B, 1, H, W]
        right_disparity (torch.Tensor): Right disparity map [B, 1, H, W]
        threshold (float): Consistency threshold (default: 1.0)
    
    Returns:
        torch.Tensor: Occlusion mask [B, 1, H, W], 1 for non-occluded
    """
    # Warp right disparity to left view
    right_disparity_warped = warp_image(
        right_disparity,
        left_disparity,
        direction='right_to_left'
    )
    
    # Compute consistency error
    consistency_error = torch.abs(left_disparity + right_disparity_warped)
    
    # Create mask (1 for consistent, 0 for occluded)
    occlusion_mask = (consistency_error < threshold).float()
    
    return occlusion_mask
