# CTC/HMM-based 1D Alignment for Stereo Matching

## Overview

This document describes the alternative approach to stereo matching in URSM that uses **CTC/HMM-based 1D alignment** instead of traditional 2D cost volume matching. This approach directly handles uncalibrated stereo pairs without explicit calibration estimation.

## Motivation

Traditional stereo matching approaches have limitations:

1. **High Computational Cost**: 2D brute-force matching across the entire image is expensive
2. **Calibration Dependency**: Requires explicit calibration or assumes perfect rectification
3. **Occlusion Handling**: Often forces matches even in occluded regions
4. **No Temporal Consistency**: Treats each frame independently

## Key Innovations

### 1. Transform 2D → 1D Matching

Instead of searching the entire 2D image space, we:

- **Extract epipolar curves** from uncalibrated images
- **Sample features along curves** at regular intervals
- **Perform 1D matching** along each curve independently
- **Reduce search space** from O(W²) to O(W) per scanline

**Benefits**:
- Lower computational cost
- Natural geometric constraint
- Easier to enforce temporal consistency

### 2. CTC with Blank Tokens

We use Connectionist Temporal Classification (CTC) with a special "blank" token:

**What is CTC?**
- Originally developed for speech recognition
- Allows variable-length input-output alignment
- Handles insertions, deletions, and repetitions
- Includes "blank" token for uncertain/missing labels

**For Stereo Matching**:
- **Blank token** = occlusion or low-confidence region
- **Non-blank tokens** = disparity values [0, max_disparity-1]
- **CTC loss** trains the network to:
  - Predict correct disparities where confident
  - Output blank for occluded/uncertain regions
  - Learn monotonic alignment naturally

**Benefits**:
- Natural occlusion handling
- No forced matches in difficult areas
- Reduces contamination from erroneous supervision
- More stable training

### 3. Monotonic Constraint

CTC enforces a "forward-only" alignment rule:

**What this means**:
- Correspondences must move left-to-right (or right-to-left)
- Cannot jump backwards or cross
- Eliminates many impossible matches

**Why it helps**:
- Stereo correspondences naturally follow epipolar ordering
- Reduces ambiguity and false matches
- Smaller effective search space
- More stable convergence

### 4. Sparse Parity Check Error Correction

Inspired by LDPC (Low-Density Parity-Check) codes from information theory:

**Concept**:
- Define sparse geometric constraints (smoothness, ordering)
- Use belief propagation to enforce consistency
- Correct local errors through global reasoning

**Components**:
1. **Parity Check Graph**: Sparse graph of pixel relationships
2. **Belief Propagation**: Iterative message passing algorithm
3. **Geometric Constraints**:
   - Horizontal smoothness
   - Vertical smoothness
   - Ordering (left-to-right non-increasing)

**Benefits**:
- Corrects isolated errors
- Enforces geometric consistency
- Computationally efficient (sparse operations)
- Improves robustness

## Architecture Details

### Overall Pipeline

```
Left Image ──┐
             ├──> Feature Extractor ──> Features Along Epipolar Curves
Right Image ─┘                          ↓
                                        1D Feature Sequences
                                        ↓
                                        Bidirectional LSTM
                                        ↓
                                        CTC Output Layer
                                        ├─> Disparity Tokens [0, ..., max_disp-1]
                                        └─> Blank Token (occlusion)
                                        ↓
                                        CTC Decoding
                                        ↓
                                        Sparse Parity Check Correction
                                        ↓
                                        Final Disparity + Occlusion Mask
```

### Component Details

#### 1. Feature Extraction
- Same as original URSM (ResNet-based)
- Multi-scale features at 1/4 resolution
- Robust to photometric variations

#### 2. Epipolar Feature Sampling
```python
# For each scanline (horizontal row)
for y in range(height):
    left_features[y, :, :] = features_left[:, :, y, :]
    right_features[y, :, :] = features_right[:, :, y, :]
    # Now we have 1D sequences for matching
```

Note: For uncalibrated images, we approximate epipolar curves as horizontal lines. For better accuracy, actual epipolar curves can be computed from estimated fundamental matrix.

#### 3. CTC Alignment Module

**Input**: Concatenated left-right feature sequences
**Processing**:
1. LSTM encoding: Captures sequential context
2. CTC output layer: Produces log probabilities for each class
3. Classes: [d0, d1, ..., d_max, blank]

**Output**: 
- Log probabilities: [B*H, W, num_classes]
- Decoded disparity: [B, H, W]
- Occlusion mask: [B, H, W]

**CTC Loss**:
```python
# Automatically handles:
# - Variable length alignments
# - Multiple paths to same alignment
# - Blank token insertions
loss = CTC_Loss(log_probs, target_labels, input_lengths, target_lengths)
```

#### 4. Sparse Parity Check Correction

**Graph Construction**:
```python
# Create sparse constraint graph
edges = []
for each pixel pair (i, j):
    if neighbors and random() < density:
        edges.append((i, j, constraint_type))
```

**Constraint Types**:
- Type 0: Horizontal smoothness (|d[i] - d[j]| should be small)
- Type 1: Vertical smoothness (|d[i] - d[j]| should be small)
- Type 2: Ordering (d[left] >= d[right])

**Belief Propagation**:
```python
for iteration in range(num_iterations):
    # Compute messages from neighbors
    messages = compute_messages(beliefs, graph, constraints)
    
    # Update beliefs
    new_beliefs = combine_beliefs_and_messages(
        initial_beliefs, messages, confidence
    )
    
    # Damping for stability
    messages = damping * old_messages + (1 - damping) * messages
```

**Output**: Corrected disparity with improved consistency

## Training

### Loss Function

Combined loss with three components:

```python
total_loss = (
    ctc_weight * CTC_loss +           # Alignment accuracy
    smooth_weight * Smoothness_loss +  # Spatial consistency
    occlusion_weight * Occlusion_loss  # Occlusion clustering
)
```

**CTC Loss** (automatic from PyTorch):
- Handles variable alignments
- Maximizes probability of correct sequence
- Blank token learned automatically

**Smoothness Loss** (edge-aware):
- Penalizes disparity gradients
- Weighted by image edges
- Encourages piecewise smooth disparity

**Occlusion Loss**:
- Penalizes isolated occlusions
- Encourages spatial clustering
- Makes occlusion mask more coherent

### Training Tips

1. **Gradient Clipping**: CTC can have large gradients
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
   ```

2. **Warm-up**: Start with lower learning rate for CTC stability

3. **Curriculum Learning**: Train on easier examples first
   - Start with well-calibrated data
   - Gradually add calibration noise

4. **Data Augmentation**: Still beneficial
   - Photometric variations
   - Geometric perturbations
   - Realistic occlusion patterns

## Inference

### Standard Inference

```python
model.eval()
with torch.no_grad():
    outputs = model(left_img, right_img)
    
    disparity = outputs['disparity']  # Final disparity
    disparity_ctc = outputs['disparity_ctc']  # Before correction
    occlusion_mask = outputs['occlusion_mask']  # 1=valid, 0=occluded
    confidence = outputs['confidence']  # Per-pixel confidence
```

### Decoding Options

**Greedy Decoding** (default, fast):
```python
# For each position, take most likely disparity
disparity = argmax(probs[:, :, :-1])  # Exclude blank
# Mark as occluded if blank probability > threshold
occluded = probs[:, :, -1] > blank_threshold
```

**Beam Search** (better quality, slower):
```python
# Maintain top-k paths at each step
# Expand with disparity transitions and blanks
# Return highest probability path
```

**Viterbi Decoding** (HMM-style):
```python
# Use dynamic programming with transition probabilities
# Enforce monotonic constraint
# Return maximum likelihood path
```

## Advantages Over Traditional Approaches

### Computational Efficiency

| Approach | Complexity | Memory |
|----------|-----------|--------|
| 2D Cost Volume | O(W² × H × D) | O(W × H × D) |
| CTC 1D Alignment | O(W × H × D) | O(W × D) |
| **Speedup** | **~W times faster** | **~H times less** |

### Occlusion Handling

**Traditional**:
- Forced matching everywhere
- Post-hoc occlusion detection
- Error propagation from bad matches

**CTC Approach**:
- Blank token for uncertain regions
- Natural during training and inference
- No forced matches in occluded areas

### Robustness to Calibration

**Traditional**:
- Requires good calibration
- Degrades with calibration errors
- Needs explicit correction

**CTC Approach**:
- Works on uncalibrated images
- Learns to find correspondences along curves
- Robust to moderate calibration errors

### Geometric Consistency

**Traditional**:
- Post-processing for consistency
- May conflict with data term
- Hard to balance

**CTC Approach**:
- Monotonic constraint built-in
- Parity check correction enforces consistency
- Better geometry-data tradeoff

## Limitations and Future Work

### Current Limitations

1. **Epipolar Approximation**: Currently uses horizontal scanlines
   - Could improve with accurate epipolar curves
   - Requires fundamental matrix estimation

2. **Greedy Decoding**: Uses simple greedy decoding
   - Beam search would improve quality
   - Viterbi would be more principled

3. **Fixed Parity Graph**: Graph structure is fixed
   - Could learn graph structure
   - Adaptive density based on scene

4. **Single Scale**: Operates at single resolution
   - Multi-scale could improve accuracy
   - Coarse-to-fine refinement

### Future Enhancements

1. **Learned Epipolar Estimation**
   ```python
   # Jointly learn epipolar curves and matching
   epipolar_curves = epipolar_network(features_left, features_right)
   alignment = ctc_alignment(features, epipolar_curves)
   ```

2. **Attention Mechanism**
   ```python
   # Attend to relevant features along curves
   attended_features = attention(queries=left_seq, keys=right_seq)
   ```

3. **Temporal Consistency** (for video)
   ```python
   # Propagate beliefs across frames
   beliefs_t = update(beliefs_{t-1}, features_t, optical_flow)
   ```

4. **Learned Error Correction**
   ```python
   # Learn correction parameters
   correction = neural_network(disparity_ctc, features, confidence)
   ```

5. **End-to-End Calibration**
   ```python
   # Jointly optimize alignment and calibration
   disparity, calibration = model(left, right)
   loss = ctc_loss(disparity) + calibration_loss(calibration)
   ```

## Comparison with Original URSM

| Aspect | Original URSM | URSM-CTC |
|--------|---------------|----------|
| **Matching** | 2D cost volume | 1D CTC alignment |
| **Calibration** | Explicit estimation | Implicit in alignment |
| **Occlusion** | Uncertainty estimate | Blank tokens |
| **Complexity** | O(W² × H × D) | O(W × H × D) |
| **Memory** | High | Lower |
| **Training** | Disparity loss | CTC + smoothness |
| **Inference** | Soft-argmin | CTC decoding + BP |

## Example Usage

### Basic Training

```python
from ursm.models import URSMNetCTC, URSMNetCTCLoss

# Create model
model = URSMNetCTC(
    max_disparity=192,
    feature_channels=32,
    hidden_dim=128,
    blank_threshold=0.3,
    use_error_correction=True
)

# Create loss
loss_fn = URSMNetCTCLoss(
    ctc_weight=1.0,
    smooth_weight=0.1,
    occlusion_weight=0.05
)

# Training loop
for batch in dataloader:
    left, right, gt_disparity = batch
    
    outputs = model(left, right, target_disparity=gt_disparity)
    losses = loss_fn(outputs, gt_disparity, left)
    
    losses['total_loss'].backward()
    optimizer.step()
```

### Inference with Error Correction

```python
model.eval()
with torch.no_grad():
    outputs = model(left_img, right_img)
    
    # Get results
    disparity = outputs['disparity']  # After correction
    disparity_raw = outputs['disparity_ctc']  # Before correction
    occlusion = outputs['occlusion_mask']
    
    # Visualize
    visualize_disparity(disparity, save_path='disparity.png')
    visualize_disparity(occlusion, save_path='occlusion.png')
```

## Conclusion

The CTC/HMM-based 1D alignment approach offers a compelling alternative to traditional stereo matching:

- **More efficient**: 1D search vs 2D
- **Better occlusion handling**: Blank tokens
- **Stronger constraints**: Monotonic alignment
- **More robust**: Parity check correction

This approach is particularly well-suited for:
- Real-time applications (lower complexity)
- Uncalibrated scenarios (no explicit calibration)
- Occluded scenes (natural handling)
- Edge devices (lower memory)

While both approaches have merit, the CTC-based method provides a fresh perspective on the stereo matching problem by treating it as a sequence alignment task rather than a correlation problem.
