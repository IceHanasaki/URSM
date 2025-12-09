# URSM Examples

This directory contains example scripts demonstrating how to use URSM.

## Demo Script

The `demo.py` script provides a simple demonstration of URSM:

```bash
python examples/demo.py
```

This will:
1. Create a synthetic stereo pair
2. Initialize the URSM model
3. Run inference
4. Save visualizations to `demo_output/`

### Expected Output

The demo creates several visualization files:
- `left.png`, `right.png`: Input stereo pair
- `disparity.png`: Predicted disparity map
- `uncertainty.png`: Uncertainty estimates
- `calibration.png`: Estimated calibration parameters

## Custom Usage

### Basic Inference

```python
import torch
from ursm.models import URSMNet

# Create model
model = URSMNet(
    max_disparity=192,
    feature_channels=32,
    refine_iterations=3,
    estimate_calibration=True
)

# Load your stereo images (as tensors [B, 3, H, W])
left_img = ...  # Load left image
right_img = ... # Load right image

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(left_img, right_img)

# Get results
disparity = outputs['disparity']  # [B, 1, H, W]
uncertainty = outputs['uncertainty']  # [B, 1, H, W]
calibration = outputs['calibration']  # dict
```

### With Known Calibration

If you have known calibration parameters:

```python
known_calibration = {
    'vertical_offset': torch.tensor([[2.5]]),  # 2.5 pixels
    'rotation': torch.tensor([[0.5, -0.3, 0.1]]),  # degrees
    'scale': torch.tensor([[1.02]])  # 2% scale difference
}

outputs = model(left_img, right_img, known_calibration)
```

### Training

```python
from ursm.losses import DisparityLoss, CalibrationAwareSmoothness

# Create loss functions
disparity_loss = DisparityLoss(loss_type='smooth_l1')
smoothness_loss = CalibrationAwareSmoothness(alpha=0.1)

# Training loop
for batch in dataloader:
    left, right, gt_disparity = batch
    
    # Forward pass
    outputs = model(left, right)
    
    # Compute losses
    loss_disp = disparity_loss(outputs, gt_disparity)
    loss_smooth = smoothness_loss(outputs['disparity'], left)
    loss = loss_disp + loss_smooth
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

## More Examples

For complete training and evaluation examples, see:
- `scripts/train.py`: Full training script
- `scripts/evaluate.py`: Evaluation script
- `scripts/inference.py`: Inference script

## Tips

1. **Memory Management**: Use smaller batch sizes if you encounter out-of-memory errors
2. **Speed vs Quality**: Reduce `refine_iterations` for faster inference
3. **Calibration**: Enable `estimate_calibration=True` when calibration is uncertain
4. **Visualization**: Use the utilities in `ursm.utils.visualization` for easy visualization
