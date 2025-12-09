"""
Visualization utilities for disparity maps and calibration parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def apply_colormap(disparity, vmin=None, vmax=None, cmap='turbo'):
    """
    Apply colormap to disparity map.
    
    Args:
        disparity (np.ndarray or torch.Tensor): Disparity map [H, W]
        vmin (float, optional): Minimum value for colormap
        vmax (float, optional): Maximum value for colormap
        cmap (str): Matplotlib colormap name (default: 'turbo')
    
    Returns:
        np.ndarray: RGB image [H, W, 3] with values in [0, 255]
    """
    # Convert to numpy if tensor
    if torch.is_tensor(disparity):
        disparity = disparity.detach().cpu().numpy()
    
    # Remove channel dimension if present
    if disparity.ndim == 3:
        disparity = disparity[0]
    
    # Set range
    if vmin is None:
        vmin = np.percentile(disparity[disparity > 0], 2)
    if vmax is None:
        vmax = np.percentile(disparity[disparity > 0], 98)
    
    # Normalize
    disparity_norm = (disparity - vmin) / (vmax - vmin + 1e-8)
    disparity_norm = np.clip(disparity_norm, 0, 1)
    
    # Apply colormap
    cmap_fn = plt.cm.get_cmap(cmap)
    disparity_colored = cmap_fn(disparity_norm)[:, :, :3]  # Remove alpha
    
    # Convert to uint8
    disparity_colored = (disparity_colored * 255).astype(np.uint8)
    
    return disparity_colored


def visualize_disparity(
    disparity,
    save_path=None,
    title='Disparity Map',
    vmin=None,
    vmax=None,
    cmap='turbo'
):
    """
    Visualize disparity map.
    
    Args:
        disparity (np.ndarray or torch.Tensor): Disparity map
        save_path (str, optional): Path to save visualization
        title (str): Title for the plot
        vmin (float, optional): Minimum value for colormap
        vmax (float, optional): Maximum value for colormap
        cmap (str): Matplotlib colormap name
    
    Returns:
        np.ndarray: Colored disparity map
    """
    disparity_colored = apply_colormap(disparity, vmin, vmax, cmap)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(disparity_colored)
    ax.set_title(title)
    ax.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    return disparity_colored


def visualize_calibration(calibration_params, save_path=None):
    """
    Visualize calibration parameters.
    
    Args:
        calibration_params (dict): Calibration parameters containing:
            - 'vertical_offset': Vertical offset [B, 1]
            - 'rotation': Rotation angles [B, 3]
            - 'scale': Scale factor [B, 1]
            - 'uncertainty': Uncertainty [B, 5]
        save_path (str, optional): Path to save visualization
    
    Returns:
        None
    """
    # Extract parameters
    vertical_offset = calibration_params['vertical_offset'].detach().cpu().numpy()
    rotation = calibration_params['rotation'].detach().cpu().numpy()
    scale = calibration_params['scale'].detach().cpu().numpy()
    uncertainty = calibration_params.get('uncertainty', None)
    
    if uncertainty is not None:
        uncertainty = uncertainty.detach().cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vertical offset
    axes[0, 0].bar(['Vertical Offset'], [vertical_offset[0, 0]])
    axes[0, 0].set_title('Vertical Offset (pixels)')
    axes[0, 0].set_ylabel('Pixels')
    
    # Rotation angles
    rotation_labels = ['Roll', 'Pitch', 'Yaw']
    axes[0, 1].bar(rotation_labels, rotation[0])
    axes[0, 1].set_title('Rotation Angles (degrees)')
    axes[0, 1].set_ylabel('Degrees')
    
    # Scale factor
    axes[1, 0].bar(['Scale'], [scale[0, 0]])
    axes[1, 0].set_title('Scale Factor')
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', label='Perfect calibration')
    axes[1, 0].legend()
    
    # Uncertainty
    if uncertainty is not None:
        uncertainty_labels = ['V. Offset', 'Roll', 'Pitch', 'Yaw', 'Scale']
        axes[1, 1].bar(uncertainty_labels, uncertainty[0])
        axes[1, 1].set_title('Parameter Uncertainties')
        axes[1, 1].set_ylabel('Uncertainty')
    else:
        axes[1, 1].text(0.5, 0.5, 'No uncertainty data',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Parameter Uncertainties')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def create_comparison_plot(
    left_img, right_img, disparity_pred, disparity_gt=None, save_path=None
):
    """
    Create comparison plot showing images and disparity.
    
    Args:
        left_img (np.ndarray or torch.Tensor): Left image
        right_img (np.ndarray or torch.Tensor): Right image
        disparity_pred (np.ndarray or torch.Tensor): Predicted disparity
        disparity_gt (np.ndarray or torch.Tensor, optional): Ground truth disparity
        save_path (str, optional): Path to save the plot
    
    Returns:
        None
    """
    # Convert tensors to numpy
    if torch.is_tensor(left_img):
        left_img = left_img.detach().cpu().permute(1, 2, 0).numpy()
    if torch.is_tensor(right_img):
        right_img = right_img.detach().cpu().permute(1, 2, 0).numpy()
    
    # Normalize images to [0, 1]
    left_img = np.clip(left_img, 0, 1)
    right_img = np.clip(right_img, 0, 1)
    
    # Apply colormap to disparity
    disparity_pred_colored = apply_colormap(disparity_pred)
    
    # Create figure
    if disparity_gt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images and disparities
    axes[0].imshow(left_img)
    axes[0].set_title('Left Image')
    axes[0].axis('off')
    
    axes[1].imshow(right_img)
    axes[1].set_title('Right Image')
    axes[1].axis('off')
    
    axes[2].imshow(disparity_pred_colored)
    axes[2].set_title('Predicted Disparity')
    axes[2].axis('off')
    
    if disparity_gt is not None:
        disparity_gt_colored = apply_colormap(disparity_gt)
        axes[3].imshow(disparity_gt_colored)
        axes[3].set_title('Ground Truth Disparity')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
