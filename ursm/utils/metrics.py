"""
Evaluation metrics for stereo matching.
"""

import torch
import numpy as np


def compute_epe(pred_disparity, gt_disparity, mask=None):
    """
    Compute End-Point Error (EPE) - average L1 distance.
    
    Args:
        pred_disparity (torch.Tensor or np.ndarray): Predicted disparity
        gt_disparity (torch.Tensor or np.ndarray): Ground truth disparity
        mask (torch.Tensor or np.ndarray, optional): Valid pixel mask
    
    Returns:
        float: Mean EPE over valid pixels
    """
    # Convert to torch tensors if needed
    if isinstance(pred_disparity, np.ndarray):
        pred_disparity = torch.from_numpy(pred_disparity)
    if isinstance(gt_disparity, np.ndarray):
        gt_disparity = torch.from_numpy(gt_disparity)
    
    # Compute absolute error
    error = torch.abs(pred_disparity - gt_disparity)
    
    # Create mask for valid pixels
    MAX_VALID_DISPARITY = 1e6  # Large finite value instead of infinity
    if mask is None:
        mask = (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    else:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask & (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    
    # Compute mean error over valid pixels
    if mask.sum() > 0:
        epe = (error * mask.float()).sum() / mask.float().sum()
    else:
        epe = torch.tensor(0.0)
    
    return epe.item()


def compute_bad_pixels(pred_disparity, gt_disparity, threshold=3.0, mask=None):
    """
    Compute percentage of bad pixels (error > threshold).
    
    Args:
        pred_disparity (torch.Tensor or np.ndarray): Predicted disparity
        gt_disparity (torch.Tensor or np.ndarray): Ground truth disparity
        threshold (float): Error threshold (default: 3.0 pixels)
        mask (torch.Tensor or np.ndarray, optional): Valid pixel mask
    
    Returns:
        float: Percentage of bad pixels (0-100)
    """
    # Convert to torch tensors if needed
    if isinstance(pred_disparity, np.ndarray):
        pred_disparity = torch.from_numpy(pred_disparity)
    if isinstance(gt_disparity, np.ndarray):
        gt_disparity = torch.from_numpy(gt_disparity)
    
    # Compute absolute error
    error = torch.abs(pred_disparity - gt_disparity)
    
    # Create mask for valid pixels
    MAX_VALID_DISPARITY = 1e6  # Large finite value instead of infinity
    if mask is None:
        mask = (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    else:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask & (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    
    # Compute bad pixels
    bad_pixels = (error > threshold) & mask
    
    if mask.sum() > 0:
        bad_percentage = (bad_pixels.float().sum() / mask.float().sum() * 100).item()
    else:
        bad_percentage = 0.0
    
    return bad_percentage


def compute_d1_error(pred_disparity, gt_disparity, mask=None):
    """
    Compute D1 error metric (KITTI benchmark).
    
    Percentage of pixels where error > 3 pixels AND error > 5% of ground truth.
    
    Args:
        pred_disparity (torch.Tensor or np.ndarray): Predicted disparity
        gt_disparity (torch.Tensor or np.ndarray): Ground truth disparity
        mask (torch.Tensor or np.ndarray, optional): Valid pixel mask
    
    Returns:
        float: D1 error percentage (0-100)
    """
    # Convert to torch tensors if needed
    if isinstance(pred_disparity, np.ndarray):
        pred_disparity = torch.from_numpy(pred_disparity)
    if isinstance(gt_disparity, np.ndarray):
        gt_disparity = torch.from_numpy(gt_disparity)
    
    # Compute absolute error
    error = torch.abs(pred_disparity - gt_disparity)
    
    # Create mask for valid pixels
    MAX_VALID_DISPARITY = 1e6  # Large finite value instead of infinity
    if mask is None:
        mask = (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    else:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask & (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    
    # D1 error thresholds (KITTI standard)
    D1_THRESHOLD_ABS = 3.0
    D1_THRESHOLD_REL = 0.05
    
    # D1 error: (error > 3) AND (error / gt > 0.05)
    threshold_abs = D1_THRESHOLD_ABS
    threshold_rel = D1_THRESHOLD_REL
    
    bad_pixels = ((error > threshold_abs) & (error / (gt_disparity + 1e-8) > threshold_rel)) & mask
    
    if mask.sum() > 0:
        d1_error = (bad_pixels.float().sum() / mask.float().sum() * 100).item()
    else:
        d1_error = 0.0
    
    return d1_error


def compute_threshold_accuracy(pred_disparity, gt_disparity, threshold=1.0, mask=None):
    """
    Compute percentage of pixels within threshold.
    
    Args:
        pred_disparity (torch.Tensor or np.ndarray): Predicted disparity
        gt_disparity (torch.Tensor or np.ndarray): Ground truth disparity
        threshold (float): Error threshold (default: 1.0 pixel)
        mask (torch.Tensor or np.ndarray, optional): Valid pixel mask
    
    Returns:
        float: Accuracy percentage (0-100)
    """
    # Convert to torch tensors if needed
    if isinstance(pred_disparity, np.ndarray):
        pred_disparity = torch.from_numpy(pred_disparity)
    if isinstance(gt_disparity, np.ndarray):
        gt_disparity = torch.from_numpy(gt_disparity)
    
    # Compute absolute error
    error = torch.abs(pred_disparity - gt_disparity)
    
    # Create mask for valid pixels
    MAX_VALID_DISPARITY = 1e6  # Large finite value instead of infinity
    if mask is None:
        mask = (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    else:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask & (gt_disparity > 0) & (gt_disparity < MAX_VALID_DISPARITY)
    
    # Compute good pixels
    good_pixels = (error <= threshold) & mask
    
    if mask.sum() > 0:
        accuracy = (good_pixels.float().sum() / mask.float().sum() * 100).item()
    else:
        accuracy = 0.0
    
    return accuracy


def evaluate_stereo(pred_disparity, gt_disparity, mask=None):
    """
    Compute all standard stereo metrics.
    
    Args:
        pred_disparity (torch.Tensor or np.ndarray): Predicted disparity
        gt_disparity (torch.Tensor or np.ndarray): Ground truth disparity
        mask (torch.Tensor or np.ndarray, optional): Valid pixel mask
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'epe': compute_epe(pred_disparity, gt_disparity, mask),
        'bad_3': compute_bad_pixels(pred_disparity, gt_disparity, 3.0, mask),
        'bad_2': compute_bad_pixels(pred_disparity, gt_disparity, 2.0, mask),
        'bad_1': compute_bad_pixels(pred_disparity, gt_disparity, 1.0, mask),
        'd1_error': compute_d1_error(pred_disparity, gt_disparity, mask),
        'acc_1': compute_threshold_accuracy(pred_disparity, gt_disparity, 1.0, mask),
        'acc_2': compute_threshold_accuracy(pred_disparity, gt_disparity, 2.0, mask),
        'acc_3': compute_threshold_accuracy(pred_disparity, gt_disparity, 3.0, mask),
    }
    
    return metrics
