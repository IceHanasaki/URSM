# Getting Started with URSM

This guide will help you get started with URSM for robust stereo matching.

## Prerequisites

Before you begin, ensure you have:
- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- Basic understanding of stereo vision

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/IceHanasaki/URSM.git
cd URSM
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### Step 3: Verify Installation

```bash
python -c "import ursm; print(ursm.__version__)"
```

Expected output: `0.1.0`

## Quick Start

### 1. Run Inference on Sample Images

```bash
# Download sample stereo pair (or use your own)
python scripts/inference.py \
    --checkpoint path/to/checkpoint.pth \
    --left examples/left.png \
    --right examples/right.png \
    --output results/disparity.png
```

This will generate:
- `results/disparity.png`: Disparity visualization
- `results/disparity.npy`: Raw disparity values
- `results/disparity_uncertainty.png`: Uncertainty map

### 2. Visualize Results

Open the generated images to see:
- **Disparity**: Depth map (warmer colors = closer)
- **Uncertainty**: Confidence (darker = more certain)

## Understanding the Output

### Disparity Map

The disparity map represents depth:
- High disparity (warm colors) = close objects
- Low disparity (cool colors) = far objects
- Zero disparity (black) = invalid/occluded

### Uncertainty Map

The uncertainty map shows confidence:
- Low values (dark) = confident prediction
- High values (bright) = uncertain prediction

Common reasons for high uncertainty:
- Textureless regions
- Occlusions
- Reflective surfaces
- Calibration ambiguity

### Calibration Parameters

If calibration estimation is enabled:
- **Vertical Offset**: Misalignment in pixels
- **Rotation**: Camera rotation errors
- **Scale**: Focal length difference

## Next Steps

### Prepare Your Data

For training, organize your data:

```
data/
├── train/
│   ├── left/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   ├── right/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── disparity/  (optional)
│       ├── 000001.pfm
│       ├── 000002.pfm
│       └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Customize Configuration

Edit `configs/default.yaml`:

```yaml
# Adjust for your data
data:
  train_dir: "path/to/your/data/train"
  val_dir: "path/to/your/data/val"

# Adjust for your GPU memory
training:
  batch_size: 2  # Reduce if out of memory

# Adjust disparity range
model:
  max_disparity: 256  # Increase for close objects
```

### Start Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --gpu 0 \
    --output_dir outputs/my_experiment
```

Monitor training:
```bash
tensorboard --logdir outputs/my_experiment/logs
```

### Evaluate Model

After training:

```bash
python scripts/evaluate.py \
    --checkpoint outputs/my_experiment/best/checkpoint_epoch_100.pth \
    --data_dir data/test \
    --save_visualizations
```

## Common Issues

### Out of Memory

If you get CUDA out of memory errors:

1. Reduce batch size:
   ```yaml
   training:
     batch_size: 1
   ```

2. Reduce image resolution:
   ```python
   # In your data loader, add resize
   transform = transforms.Resize((384, 768))
   ```

3. Reduce model capacity:
   ```yaml
   model:
     feature_channels: 16  # Instead of 32
     max_disparity: 128    # Instead of 192
   ```

### Slow Training

To speed up training:

1. Use mixed precision:
   ```python
   # Add to training script
   scaler = torch.cuda.amp.GradScaler()
   ```

2. Increase batch size (if memory allows)

3. Use more workers:
   ```yaml
   training:
     num_workers: 16
   ```

### Poor Results

If results are poor:

1. Check data quality:
   - Are images well-aligned?
   - Is ground truth disparity correct?

2. Adjust augmentation:
   ```yaml
   augmentation:
     vertical_offset_range: [-5, 5]  # Reduce if too aggressive
   ```

3. Train longer:
   ```yaml
   training:
     num_epochs: 200
   ```

## Tips for Best Results

### 1. Data Quality

- Use high-quality stereo pairs
- Ensure reasonable baseline (5-15cm for indoor, 0.5-1m for outdoor)
- Avoid extreme lighting conditions

### 2. Calibration Simulation

Match augmentation to expected deployment:
- Indoor robots: Small offsets (±2-5 pixels)
- Outdoor vehicles: Larger offsets (±10-20 pixels)

### 3. Model Capacity

Balance capacity with data:
- Large dataset → large model
- Small dataset → small model (avoid overfitting)

### 4. Hyperparameters

Start with defaults, then adjust:
- Learning rate: 0.0001 works well
- Batch size: As large as GPU allows
- Refinement iterations: 3 is good balance

## Example Workflows

### Workflow 1: Quick Prototype

```bash
# 1. Use pre-trained model
wget http://example.com/ursm_base.pth

# 2. Test on your images
python scripts/inference.py \
    --checkpoint ursm_base.pth \
    --left my_left.png \
    --right my_right.png \
    --output my_disparity.png

# 3. Evaluate quality visually
```

### Workflow 2: Fine-tune on Custom Data

```bash
# 1. Prepare your data
# (organize as shown above)

# 2. Start from pre-trained
python scripts/train.py \
    --config configs/default.yaml \
    --resume ursm_base.pth \
    --output_dir outputs/finetuned

# 3. Evaluate on test set
python scripts/evaluate.py \
    --checkpoint outputs/finetuned/best/checkpoint.pth \
    --data_dir data/test
```

### Workflow 3: Train from Scratch

```bash
# 1. Train on synthetic data (SceneFlow)
python scripts/train.py \
    --config configs/sceneflow.yaml \
    --output_dir outputs/sceneflow

# 2. Fine-tune on real data (KITTI)
python scripts/train.py \
    --config configs/kitti.yaml \
    --resume outputs/sceneflow/best/checkpoint.pth \
    --output_dir outputs/kitti

# 3. Deploy
python scripts/inference.py \
    --checkpoint outputs/kitti/best/checkpoint.pth \
    --left real_scene_left.png \
    --right real_scene_right.png
```

## Resources

### Datasets

- **SceneFlow**: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
- **KITTI Stereo 2015**: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php
- **Middlebury**: https://vision.middlebury.edu/stereo/

### Further Reading

- [Architecture Details](ARCHITECTURE.md)
- [Training Guide](TRAINING.md)
- [API Reference](API.md)

## Getting Help

If you encounter issues:

1. Check [Common Issues](#common-issues) above
2. Search existing GitHub issues
3. Open a new issue with:
   - Error message
   - Configuration used
   - System information (GPU, CUDA version)
   - Minimal reproducible example

## Next Steps

Now that you're set up:

1. ✅ Run inference on sample images
2. ✅ Understand the outputs
3. ⬜ Prepare your own data
4. ⬜ Train a custom model
5. ⬜ Deploy to your application

Happy matching! 🎯
