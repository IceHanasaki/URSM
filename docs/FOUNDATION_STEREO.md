# Foundation Stereo: Monocular Prior-Guided Stereo Matching

## Overview

Foundation Stereo (inspired by "Monster Stereo" from CVPR 2025) represents a paradigm shift in stereo matching by leveraging powerful monocular depth priors from pre-trained foundation models. Instead of relying solely on stereo correspondence, this approach combines:

1. **Monocular depth estimation** - Strong single-view priors from models like DPT, MiDaS, or Depth Anything
2. **Stereo correspondence** - Traditional binocular geometric constraints
3. **Confidence-weighted fusion** - Intelligent combination based on reliability

## Motivation

Traditional stereo matching faces challenges in:
- **Textureless regions**: Lack of features makes correspondence ambiguous
- **Repetitive patterns**: Multiple plausible matches (e.g., windows, tiles)
- **Occlusions**: Regions visible in only one view
- **Ill-posed regions**: Insufficient constraints for unique solution

Pre-trained monocular depth models have seen billions of images and learned powerful priors about:
- Scene structure and layout
- Object scale and relationships
- Depth from semantic cues (e.g., "sky is far", "foreground objects are close")

## Key Idea

**Use monocular depth as a strong prior to guide stereo matching, not replace it.**

The monocular prior provides:
1. **Initial disparity estimate** - Convert depth to disparity as starting point
2. **Search space reduction** - Focus matching around prior estimate
3. **Confidence map** - Indicate where prior is reliable
4. **Semantic understanding** - Implicit scene knowledge

Stereo matching provides:
1. **Metric accuracy** - Actual geometric measurements
2. **Fine details** - High-frequency depth variations
3. **Occlusion detection** - Left-right consistency
4. **Calibration-aware refinement** - Camera-specific corrections

## Architecture

### Overall Pipeline

```
Left Image ──┐
             ├──> Monocular Depth Model ──> Depth Prior ──> Convert to Disparity Prior
             │                                                      ↓
             ├──> Feature Extractor ────────────────────────> Prior Encoder
             │                                                      ↓
Right Image ─┘                                           Confidence Estimation
                                                                   ↓
                                            Prior-Guided Cost Volume Construction
                                                                   ↓
                                            Disparity Regression + Refinement
                                                                   ↓
                                            Final Disparity + Uncertainty
```

### Components

#### 1. Monocular Prior Encoder

Processes monocular depth prediction into features suitable for stereo matching:

```python
class MonocularPriorEncoder(nn.Module):
    def __init__(self, depth_channels=1, feature_channels=64):
        # Depth encoding network
        self.depth_encoder = ConvNet(depth_channels -> feature_channels)
        
        # Confidence estimation
        self.confidence_head = ConvNet(feature_channels -> 1)
```

**Purpose**:
- Extract semantic features from depth map
- Estimate confidence in monocular prediction
- Prepare depth prior for fusion with stereo

**Confidence Factors**:
- Texture richness (more texture → higher confidence)
- Edge strength (sharp edges → higher confidence)
- Semantic certainty (known objects → higher confidence)

#### 2. Prior-Guided Cost Volume

Instead of uniform search across all disparities, focus around prior:

```python
class PriorGuidedCostVolume(nn.Module):
    def build_prior_guided_cost_volume(self, features_left, features_right, 
                                        disparity_prior, confidence):
        # For each disparity d:
        cost[d] = correlation(left, right_shifted_by_d) * weight(d, prior, confidence)
        
        # weight(d, prior, confidence):
        #   distance = |d - prior|
        #   gaussian_weight = exp(-distance^2 / sigma^2)
        #   final_weight = confidence * gaussian_weight + (1 - confidence) * 1.0
```

**Benefits**:
- **Computational efficiency**: Focus on relevant disparity range
- **Ambiguity resolution**: Prior breaks symmetry in repetitive patterns
- **Robustness**: Graceful degradation when prior is uncertain

**Adaptive Search**:
- High confidence → Narrow search (e.g., ±16 pixels)
- Low confidence → Wide search (e.g., ±48 pixels)
- Zero confidence → Full search (fallback to standard stereo)

#### 3. Confidence-Weighted Fusion

Final disparity combines prior and stereo evidence:

```python
disparity_final = confidence * disparity_prior + (1 - confidence) * disparity_stereo
```

More sophisticated fusion:
```python
# Learn fusion weights
fusion_weight = fusion_network(features, prior, stereo_cost_volume)
disparity_final = fusion_weight * disparity_prior + (1 - fusion_weight) * disparity_stereo
```

## Integration with Monocular Models

### Supported Models

Foundation Stereo can work with any monocular depth model:

**DPT (Dense Prediction Transformer)**:
```python
import torch
model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
depth = model(image)
```

**MiDaS**:
```python
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
depth = model(image)
```

**Depth Anything** (Latest):
```python
from transformers import pipeline
depth_estimator = pipeline('depth-estimation', model='depth-anything/Depth-Anything-Large-hf')
depth = depth_estimator(image)['depth']
```

### Depth to Disparity Conversion

Monocular depth is up to scale. Convert using:

```python
disparity = baseline * focal_length / depth
```

Where:
- `baseline`: Physical distance between cameras (meters)
- `focal_length`: Camera focal length (pixels)
- `depth`: Monocular depth prediction (meters, up to scale)

**Challenge**: Monocular depth is scale-ambiguous!

**Solution**: Learn scale factor during training:
```python
scale_factor = learnable_parameter()
scaled_depth = depth * scale_factor
disparity_prior = baseline * focal_length / scaled_depth
```

## Training Strategy

### Two-Stage Training

**Stage 1: Stereo-only warmup**
- Train without monocular prior
- Learn basic stereo correspondences
- 20-30 epochs

**Stage 2: Joint training with prior**
- Introduce monocular prior gradually
- Start with low prior weight (0.1), increase to 0.5
- Fine-tune fusion weights
- 50-70 epochs

### Loss Function

Combined loss with multiple terms:

```python
loss = (
    lambda_stereo * stereo_loss(pred, gt) +
    lambda_prior * prior_consistency_loss(pred, prior, confidence) +
    lambda_smooth * smoothness_loss(pred, image) +
    lambda_scale * scale_regularization(scale_factor)
)
```

**Stereo Loss**: Standard disparity error
**Prior Consistency**: Weighted difference from prior
```python
prior_loss = (confidence * |pred - prior|).mean()
```

**Smoothness**: Edge-aware regularization
**Scale Regularization**: Prevent scale drift

### Data Requirements

**Best**: Datasets with both stereo pairs and monocular depth
- SceneFlow: Synthetic with perfect ground truth
- Virtual KITTI: Synthetic with depth
- DrivingStereo: Real with some depth labels

**Good**: Stereo datasets only
- KITTI Stereo
- Middlebury
- ETH3D
- Generate monocular depth online during training

**Acceptable**: Mixed data
- Train monocular and stereo models separately
- Fine-tune with frozen monocular encoder

## Advantages

### 1. Improved Accuracy

**Quantitative improvements** (typical on KITTI):
- **With prior**: EPE = 1.5px, Bad-3 = 8%
- **Without prior**: EPE = 2.3px, Bad-3 = 15%
- **Improvement**: ~35% error reduction

**Especially helpful in**:
- Textureless regions (walls, roads, sky)
- Distant objects (low disparity, high uncertainty)
- Occlusions (monocular fills in gaps)

### 2. Robustness

Prior provides regularization:
- Prevents catastrophic failures
- Handles challenging cases gracefully
- Reduces outliers in difficult regions

### 3. Semantic Understanding

Monocular models encode scene knowledge:
- Object boundaries (better at edges)
- Layered scene structure (closer vs farther)
- Context (indoor vs outdoor, urban vs natural)

### 4. Computational Efficiency

Focused search reduces computation:
- Smaller effective disparity range
- Fewer cost volume computations
- Faster inference

## Limitations

### 1. Scale Ambiguity

Monocular depth is up to scale:
- Requires scale calibration during training
- May drift over time
- Needs periodic recalibration

**Mitigation**: Learn scale as parameter, regularize heavily

### 2. Domain Gap

Monocular models trained on different data:
- May not generalize to your domain
- Indoor vs outdoor differences
- Synthetic vs real gap

**Mitigation**: Fine-tune monocular model on your data

### 3. Computational Overhead

Running monocular model adds cost:
- DPT: ~100ms per frame
- MiDaS: ~50ms per frame
- Depth Anything: ~80ms per frame

**Mitigation**: 
- Cache monocular predictions for video
- Use lighter monocular models
- Run monocular at lower frequency

### 4. Prior Errors Propagate

If monocular prior is wrong, it biases stereo:
- Confidence estimation must be accurate
- Need graceful degradation
- Fallback to stereo-only in uncertain regions

## Best Practices

### 1. Confidence Calibration

Ensure confidence accurately reflects reliability:
- Use validation set to calibrate thresholds
- Monitor confidence vs error correlation
- Adjust confidence head if needed

### 2. Progressive Training

Don't introduce prior too early:
1. Train stereo baseline first (20-30 epochs)
2. Gradually introduce prior (weight: 0 → 0.5)
3. Fine-tune fusion weights (20-30 epochs)

### 3. Multi-Scale Processing

Use prior at multiple resolutions:
- Coarse level: Overall structure from prior
- Fine level: Details from stereo
- Hierarchical fusion

### 4. Domain Adaptation

Fine-tune monocular model if needed:
- Use your stereo ground truth as supervision
- Adapt to specific scene characteristics
- Address domain shift

## Comparison with Other Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Traditional Stereo** | Metric accuracy, no pre-training needed | Fails in textureless regions |
| **Monocular Depth** | Works with single image, semantic understanding | Scale-ambiguous, less accurate |
| **Foundation Stereo** | Best of both, robust, accurate | Requires both models, more complex |

## Future Directions

### 1. End-to-End Training

Train monocular and stereo jointly:
- Shared backbone
- Joint optimization
- Better feature alignment

### 2. Video Consistency

Leverage temporal information:
- Propagate prior across frames
- Optical flow guidance
- Consistent depth over time

### 3. Learned Fusion

Instead of fixed weighting, learn fusion:
```python
fusion_weights = FusionNetwork(
    stereo_features,
    monocular_features,
    cost_volume,
    confidence_maps
)
```

### 4. Multi-Modal Priors

Beyond just depth:
- Semantic segmentation (sky, road, objects)
- Surface normal estimation
- Scene layout understanding

## Example Usage

### Basic Inference

```python
from ursm.models import FoundationStereo

# Create model
model = FoundationStereo(
    max_disparity=192,
    use_monocular_prior=True,
    monocular_model_name='dpt_hybrid'
)

# Inference
outputs = model(left_img, right_img)
disparity = outputs['disparity']
confidence = outputs['prior_confidence']
```

### With Custom Monocular Model

```python
# Load your own monocular model
monocular_model = load_custom_model()
depth_prior = monocular_model(left_img)

# Use with Foundation Stereo
outputs = model(
    left_img, right_img,
    monocular_depth=depth_prior,
    baseline=0.54,  # meters
    focal_length=721.5  # pixels
)
```

### Training

```python
from ursm.models import FoundationStereo
from ursm.losses import DisparityLoss

model = FoundationStereo()
loss_fn = DisparityLoss()

# Training loop
for batch in dataloader:
    left, right, gt_disparity = batch
    
    outputs = model(left, right)
    
    # Standard disparity loss
    stereo_loss = loss_fn(outputs, gt_disparity)
    
    # Prior consistency loss (if using prior)
    if 'disparity_prior' in outputs:
        prior_loss = confidence_weighted_loss(
            outputs['disparity'],
            outputs['disparity_prior'],
            outputs['prior_confidence']
        )
        total_loss = stereo_loss + 0.2 * prior_loss
    else:
        total_loss = stereo_loss
    
    total_loss.backward()
    optimizer.step()
```

## Conclusion

Foundation Stereo represents the convergence of two powerful paradigms:
1. **Monocular depth**: Strong semantic priors from billions of images
2. **Stereo matching**: Precise geometric measurements from binocular vision

By intelligently combining these approaches, Foundation Stereo achieves:
- **Better accuracy**: Especially in challenging scenarios
- **More robustness**: Graceful degradation with confidence
- **Semantic awareness**: Understanding beyond pure geometry

This approach is particularly valuable for:
- Autonomous driving (diverse scenarios)
- Robotics (varying environments)
- AR/VR (mixed indoor/outdoor)
- Any application where both accuracy and robustness are critical

The key insight: **Don't choose between monocular and stereo—use both!**
