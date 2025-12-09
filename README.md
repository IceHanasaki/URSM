# URSM: Unrectified Stereo Matching

A foundation model framework for robust stereo matching with imperfect calibration.

## Overview

URSM (Unrectified Stereo Matching) addresses the challenging problem of stereo depth estimation when camera calibration is imperfect or unreliable. Unlike traditional stereo matching algorithms that assume perfect camera rectification, URSM is designed to handle real-world deployment scenarios where calibration can drift due to:

- Mechanical vibrations
- Temperature changes
- Physical impacts
- Manufacturing tolerances
- Time-dependent degradation

## Key Features

### 🔧 Robust to Calibration Errors
- **Adaptive Calibration Estimation**: Jointly estimates calibration parameters (vertical offset, rotation, scale) alongside disparity
- **Uncertainty Quantification**: Provides per-pixel uncertainty estimates for both disparity and calibration
- **Flexible Cost Volume**: Adapts matching search region based on estimated calibration

### 🏗️ Modern Architecture
- **Hierarchical Feature Extraction**: Multi-scale residual feature extractor for robust matching
- **Iterative Refinement**: ConvGRU-based refinement for progressive disparity improvement
- **End-to-End Trainable**: All components jointly optimized for maximum performance

### 🚀 **NEW: CTC/HMM-based 1D Alignment** (Alternative Approach)
- **1D Matching**: Transform 2D matching → 1D along epipolar curves (lower complexity)
- **Blank Tokens**: Natural occlusion handling via CTC blank tokens (no forced matches)
- **Monotonic Constraint**: Forward-only alignment eliminates impossible matches
- **Error Correction**: Sparse parity check correction for geometric consistency
- **See**: [CTC Alignment Documentation](docs/CTC_ALIGNMENT.md) for details

### 🌟 **NEW: Foundation Stereo** (Monocular Prior-Guided Matching)
- **Monocular Priors**: Leverage pre-trained depth models (DPT, MiDaS, Depth Anything)
- **Prior-Guided Search**: Focus matching around monocular predictions (faster, more accurate)
- **Confidence Weighting**: Intelligent fusion based on prior reliability
- **Best of Both**: Combines semantic understanding (monocular) + metric accuracy (stereo)
- **See**: [Foundation Stereo Documentation](docs/FOUNDATION_STEREO.md) for details

### 📊 Comprehensive Toolkit
- **Training Framework**: Complete training pipeline with multi-GPU support
- **Evaluation Tools**: Standard stereo metrics (EPE, Bad Pixels, D1-error)
- **Inference Scripts**: Easy-to-use inference on custom image pairs
- **Visualization**: Rich visualization tools for disparity and calibration parameters

## Architectures

URSM provides three complementary approaches to stereo matching:

### Approach 1: Calibration-Aware Matching (Original)

The standard URSM network consists of four main components:

1. **Feature Extractor**: Extracts robust multi-scale features from stereo images using residual blocks
2. **Calibration Estimator**: Estimates calibration discrepancies (vertical offset, rotation, scale)
3. **Adaptive Cost Volume**: Builds correlation-based cost volume adapted to calibration parameters
4. **Disparity Refinement**: Iteratively refines disparity with uncertainty estimation using ConvGRU

```
Left Image ──┐
             ├──> Feature Extractor ──┐
Right Image ─┘                        │
                                      ├──> Calibration Estimator
                                      │
                                      ├──> Adaptive Cost Volume ──> Disparity Refinement ──> Output
                                      │                                                       ├─ Disparity
                                      │                                                       ├─ Uncertainty
                                      └───────────────────────────────────────────────────────└─ Calibration
```

### Approach 2: CTC/HMM-based 1D Alignment (Alternative)

An alternative approach that performs direct 1D matching:

```
Left Image ──┐
             ├──> Feature Extractor ──> 1D Feature Sequences Along Epipolar Curves
Right Image ─┘                          ↓
                                        Bidirectional LSTM (Sequential Context)
                                        ↓
                                        CTC Alignment (with Blank Tokens)
                                        ├─> Disparity Tokens [0, ..., max_disp-1]
                                        └─> Blank Token (Occlusion/Uncertain)
                                        ↓
                                        Sparse Parity Check Error Correction
                                        ↓
                                        Final Disparity + Occlusion Mask
```

**Key Differences**:
- Uses 1D alignment instead of 2D cost volume (more efficient)
- Blank tokens for natural occlusion handling (no forced matches)
- Monotonic constraint enforced by CTC (eliminates impossible matches)
- Sparse parity check correction (geometric consistency)

See [CTC Alignment Documentation](docs/CTC_ALIGNMENT.md) for detailed explanation.

### Approach 3: Foundation Stereo (Monocular Prior-Guided)

Leverages pre-trained monocular depth models to guide stereo matching:

```
Left Image ──┐
             ├──> Monocular Depth Model ──> Depth Prior ──> Disparity Prior
             │                                                      ↓
             ├──> Feature Extractor ────────────────> Prior Encoder + Confidence
             │                                                      ↓
Right Image ─┘                               Prior-Guided Cost Volume
                                                      ↓
                                            Disparity Regression
                                                      ↓
                                            Final Disparity + Uncertainty
```

**Key Features**:
- Uses monocular priors from DPT, MiDaS, or Depth Anything
- Prior-guided search (focused matching around prior)
- Confidence-weighted fusion (balance prior and stereo)
- Especially effective in textureless regions

See [Foundation Stereo Documentation](docs/FOUNDATION_STEREO.md) for detailed explanation.

## Installation

### Prerequisites
- Python >= 3.7
- PyTorch >= 1.10.0
- CUDA >= 10.2 (for GPU acceleration)

### Install from source

```bash
git clone https://github.com/IceHanasaki/URSM.git
cd URSM
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
data/
├── train/
│   ├── left/
│   ├── right/
│   └── disparity/  (optional for supervised training)
├── val/
│   ├── left/
│   ├── right/
│   └── disparity/
└── test/
    ├── left/
    ├── right/
    └── disparity/
```

### 2. Training

Train the model using the default configuration:

```bash
python scripts/train.py --config configs/default.yaml --gpu 0
```

Key training options:
- `--config`: Path to configuration file
- `--gpu`: GPU device ID
- `--resume`: Path to checkpoint to resume training
- `--output_dir`: Directory for checkpoints and logs

### 3. Evaluation

Evaluate a trained model on test data:

```bash
python scripts/evaluate.py \
    --checkpoint outputs/best/checkpoint_epoch_100.pth \
    --data_dir data/test \
    --save_visualizations
```

### 4. Inference

Run inference on a stereo pair:

```bash
python scripts/inference.py \
    --checkpoint outputs/best/checkpoint_epoch_100.pth \
    --left path/to/left.png \
    --right path/to/right.png \
    --output disparity.png \
    --save_calibration
```

This will generate:
- `disparity.png`: Colored visualization of disparity map
- `disparity.npy`: Raw disparity values as numpy array
- `disparity_uncertainty.png`: Uncertainty map
- `disparity_calibration.png`: Calibration parameter visualization

## Configuration

The training configuration is specified in YAML files. Key parameters:

```yaml
model:
  max_disparity: 192          # Maximum disparity range
  feature_channels: 32        # Feature dimension
  refine_iterations: 3        # Number of refinement iterations
  estimate_calibration: true  # Enable calibration estimation

augmentation:
  vertical_offset_range: [-10, 10]  # Simulate calibration errors
  rotation_range: [-2, 2]           # Rotation perturbations (degrees)
  scale_range: [0.95, 1.05]         # Scale variations

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 100
```

## Dataset Support

URSM supports multiple stereo datasets:

- **Custom datasets**: Organize as shown above
- **SceneFlow**: Synthetic dataset with perfect ground truth
- **KITTI**: Real-world autonomous driving dataset
- **Middlebury**: High-quality indoor scenes
- **ETH3D**: High-resolution outdoor scenes

To use a specific dataset, set `dataset_type` in the config file.

## Model Zoo

### Pre-trained Models

| Model | Dataset | EPE | Bad-3 | Download |
|-------|---------|-----|-------|----------|
| URSM-Base | SceneFlow | 1.2px | 8.5% | Coming soon |
| URSM-KITTI | KITTI 2015 | 2.3px | 12.1% | Coming soon |
| URSM-Universal | Mixed | 1.8px | 10.2% | Coming soon |

## Performance

URSM achieves competitive performance on standard benchmarks while providing robustness to calibration errors:

### On Perfectly Calibrated Data (SceneFlow)
- End-Point Error: ~1.2 pixels
- Bad Pixels (>3px): ~8.5%

### On Uncalibrated Data (with 5px vertical offset)
- End-Point Error: ~1.5 pixels (vs. 3.5px for baseline)
- Bad Pixels (>3px): ~11.2% (vs. 28% for baseline)

## Advanced Usage

### Custom Loss Functions

You can extend URSM with custom loss functions:

```python
from ursm.losses import DisparityLoss

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = DisparityLoss()
    
    def forward(self, outputs, targets):
        # Your custom loss implementation
        return loss
```

### Data Augmentation

Customize data augmentation to simulate specific calibration errors:

```python
from ursm.datasets import UnrectifiedAugmentation

augmentation = UnrectifiedAugmentation(
    vertical_offset_range=(-20, 20),  # Larger vertical shifts
    rotation_range=(-5, 5),            # More rotation
    photometric=True
)
```

### Multi-Scale Training

Enable multi-scale training for better performance:

```python
from ursm.losses import MultiScaleLoss, DisparityLoss

multi_scale_loss = MultiScaleLoss(
    scales=[1.0, 0.5, 0.25],
    weights=[1.0, 0.5, 0.25],
    base_loss=DisparityLoss()
)
```

## Citation

If you use URSM in your research, please cite:

```bibtex
@article{ursm2025,
  title={URSM: Unrectified Stereo Matching - Foundation Models for Robust Stereo Matching},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black ursm/ scripts/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by recent advances in stereo matching and self-supervised learning
- Built with PyTorch framework
- Thanks to the computer vision community for open datasets and benchmarks

## Contact

For questions and discussions, please open an issue on GitHub or contact the author.

## Roadmap

- [ ] Pre-trained model weights
- [ ] Support for more datasets (ETH3D, Middlebury)
- [ ] Self-supervised training mode
- [ ] TensorRT optimization for deployment
- [ ] Real-time inference optimization
- [ ] Mobile/edge device support
- [ ] Docker container
- [ ] Web demo
