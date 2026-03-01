---
layout: post
title: "Training a Waifu Diffusion Model with Patch Diffusion and Rectified Flow"
date: 2026-03-01
description: "How to train a data-efficient diffusion model on corrupted anime face data using CIELAB space, patch cropping, and modern transformer techniques."
tags: [diffusion-models, generative-ai, flow-matching, efficient-learning]
categories: [ML-Engineering]
giscus_comments: true
---

## Introduction

Training generative models on small, imbalanced datasets is notoriously difficult—but it's doable with the right tricks. I trained a **130M-parameter diffusion model** on just 10,000 anime faces, 90% of them **monochrome**, that still generates coherent colorful images. This post covers the key techniques: **patch diffusion**, **rectified flow**, and **CIELAB color space decoupling**.

**Weights & Code**: [ruwwww/waifu_diffusion](https://huggingface.co/ruwwww/waifu_diffusion)

---

## Part 1: Diffusion & Flow Matching Primer

### Diffusion Models: The Basics

Diffusion models learn to reverse a noise corruption process. You start with data $x_1$ and gradually add noise:

$$x_t = \sqrt{\bar{\alpha}_t} x_1 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The model learns to predict the noise $\epsilon_\theta(x_t, t)$ at each step, allowing deterministic sampling through **many denoising steps** (often 50–1000).

### Rectified Flow: Straight-Line Paths

**Rectified flow** simplifies this by learning a velocity field along a _straight path_ from noise to data:

$$x_t = (1-t) x_0 + t x_1, \quad t \in [0, 1]$$

Instead of predicting noise, the model learns velocity:
$$v_\theta(x_t, t)$$

**Why it's better**: Straight paths require fewer steps (30–50 work well), and the linear time mapping is more natural for learning.

In practice, we predict the **clean image** $\hat{x}_1$ and derive velocity:
$$v = \frac{\hat{x}_1 - x_t}{1 - t}$$

---

## Part 2: Handling Imbalanced Data with CIELAB

### The Dataset Problem

We use [Anime-Face-Dataset-10k](https://huggingface.co/datasets/amirali900/Anime-Face-Dataset-10k):

- 10,000 native 80×80 images
- **90% corrupted to monochrome**
- 10% kept in color

{% include figure.liquid path="assets/img/posts/waifu-diffusion/monochrome-corruption.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Monochrome corruption examples: 4 pairs showing original color images (left) and their transformed grayscale versions (right)
</div>

### Why CIELAB Works Better Than RGB

Rather than train in RGB, we convert to **CIE L\*a\*b\*** color space:

$$L \in [0, 100] \quad \text{(luminance)}, \quad a \in [-128, 128] \quad \text{(green-red)}, \quad b \in [-128, 128] \quad \text{(blue-yellow)}$$

**Key insight**: L\*a\*b\* decouples structure (L) from color (a, b). For monochrome images, we zero out the chroma channels and mask gradients:

```python
# For monochrome images
if idx not in color_indices:
    lab_tensor[1:, :, :] = 0.0     # Zero chroma
    mask[1:, :, :] = 0.0            # No gradient flow
```

This lets the model learn structural features from all 10k samples while learning color specifically from the 1k color samples without interference.

---

## Part 3: Patch Diffusion for Data Augmentation

### The Strategy

With only 10k samples, training on full 80×80 images is risky. We use **random patch cropping** during training:

```python
patch_sizes = [40, 64, 80]  # Variable size patches
full_image_prob = 0.20

if random.random() < full_image_prob:
    x = x_full  # Full image
else:
    size = random.choice(patch_sizes)
    top = random.randint(0, (80 - size) // patch_size)
    left = random.randint(0, (80 - size) // patch_size)
    x = x_full[:, :, top*4:top*4+size, left*4:left*4+size]
```

A 40×40 patch can appear at up to 21 positions, effectively multiplying dataset size. We use **Vision Rotary Embeddings** to ensure spatial consistency across patches.

{% include figure.liquid path="assets/img/posts/waifu-diffusion/patch-diagram.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Patch diffusion strategy: A 40×40 patch can be cropped from multiple positions in the 80×80 image, effectively augmenting the dataset
</div>

---

## Part 4: Model Architecture

### The JiT Transformer

We use a modern **Diffusion Transformer (DiT-B)** with ~130M parameters:

```python
model = JiT(
    input_size=80,
    patch_size=4,           # 80×80 → 20×20 token grid
    hidden_size=768,
    depth=12,
    num_heads=12,
    in_context_len=0        # Unconditional
)
```

**Key modern techniques**:

- **AdaLN**: Timestep-modulated layer norm for expressive conditioning
- **SwiGLU**: Swish-gated feedforward layers (better than standard MLPs)
- **RMSNorm**: Stable layer normalization for mixed-precision training
- **Vision RoPE**: 2D rotary positional embeddings for patch-aware spatial reasoning
- **Scaled Dot-Product Attention**: Memory-efficient attention

The attention mechanism applies Vision RoPE with patch coordinates, ensuring smooth spatial transitions:

```python
q = rope(q, top_idx=top_idx, left_idx=left_idx)
k = rope(k, top_idx=top_idx, left_idx=left_idx)
```

---

## Part 5: Training Strategy

### Loss Function & Gradient Masking

We train the velocity field with **masked MSE loss**. The model outputs **x-pred** (clean image prediction), which we convert to **v-pred** for both loss computation and sampling:

$$\mathcal{L} = \mathbb{E} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \odot \mathbf{m} \right]$$

where $v_\theta = \hat{x}_1 - x_t$ is derived from the model's x-prediction.

```python
def masked_mse_loss(pred_x1, target_x1, mask):
    return (F.mse_loss(pred_x1, target_x1, reduction='none') * mask).mean()

# Training loop converts x-pred to v-pred:
v_pred = pred_x1 - x0
v_target = x1 - x0
loss = ((v_pred - v_target) ** 2 * mask).mean()
```

The mask prevents gradients from flowing through masked chroma channels in monochrome images.

### Handling Data Imbalance: Oversampling

We oversample colored images by 3x using `WeightedRandomSampler`:

```python
weights = [3.0 if i in color_indices else 1.0 for i in range(len(dataset))]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
```

### Training Details

- **Epochs**: 1280
- **Batch size**: 256 (accumulated from 4×64)
- **Learning rate**: 3e-4 (AdamW)
- **Mixed precision**: fp16 with gradient scaling
- **Compilation**: `torch.compile` for 1.5–2x speedup

---

## Part 6: Results

### Generation Sampling

Sampling uses **Euler integration** over 50 steps:

```python
@torch.no_grad()
def generate(model, device, steps=50):
    xt = torch.randn((1, 3, 80, 80), device=device)
    y = torch.zeros(1, dtype=torch.long, device=device)

    for step in range(steps):
        t_val = step / steps
        t = torch.tensor(t_val, device=device)
        pred_x1 = model(xt, t, y, top_idx=0, left_idx=0)

        v = (pred_x1 - xt) / max(1.0 - t_val, 1e-2)
        xt = xt + v / steps

    return pred_x1
```

### Generated Images

Despite training on 90% monochrome data, the model generates vibrant, coherent faces:

{% include figure.liquid path="assets/img/posts/waifu-diffusion/generated-samples.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Three generated anime face samples from the model, all at native 80×80 resolution
</div>

### Avoiding Memorization: LPIPS Validation

We verify each generated image is novel by computing **LPIPS distance** to the nearest training sample. Typical values:

- Same image: 0.0
- Very similar: 0.05–0.15
- Somewhat distinct: 0.15–0.25
- Our generated samples: **≥ 0.25–0.3** ✓

The consistently higher LPIPS scores confirm the model generates novel faces rather than memorizing training data.

{% include figure.liquid path="assets/img/posts/waifu-diffusion/nearest-neighbor.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Nearest neighbor validation (merged panel): each pair shows a generated image and its closest training sample with LPIPS distance ≥ 0.25
</div>

### Generation Trajectory

The model smoothly transitions from noise to structure to detail:

{% include figure.liquid path="assets/img/posts/waifu-diffusion/trajectory.gif" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Generation trajectory showing 10 frames from pure noise (t=0) to final image (t=1) over 50 sampling steps
</div>

---

## Part 7: Key Takeaways

1. **CIELAB decoupling is powerful**: Separating structure from color lets you learn from partial/corrupted data gracefully.

2. **Patch diffusion multiplies your data**: Random cropping with spatial embeddings is a simple but effective augmentation.

3. **Rectified flow is simpler & faster**: Straight-line paths with velocity matching need far fewer steps than traditional DDPM.

4. **Modern components matter**: AdaLN, RMSNorm, Vision RoPE, and torch.compile all contribute to efficiency.

5. **Oversampling works**: Weighted sampling of rare color samples prevents the model from ignoring them.

---

## Part 8: Code & Model

**Model weights**: [ruwwww/waifu_diffusion](https://huggingface.co/ruwwww/waifu_diffusion)  
**File**: `waifu_diffusion_1280_bs256.safetensors` (130M params)

**Quick inference**:

```python
from safetensors.torch import load_file
from skimage import color
import numpy as np

model = JiT(input_size=80, patch_size=4, in_channels=3,
            hidden_size=768, depth=12, num_heads=12, num_classes=1)
state_dict = load_file("waifu_diffusion_1280_bs256.safetensors")
model.load_state_dict(state_dict)
model.to(device).eval()

# Generate
xt = torch.randn((1, 3, 80, 80), device=device)
y = torch.zeros(1, dtype=torch.long, device=device)
for step in range(50):
    t = torch.tensor(step / 50, device=device)
    pred_x1 = model(xt, t, y, top_idx=0, left_idx=0)
    v = (pred_x1 - xt) / max(1.0 - step/50, 1e-2)
    xt = xt + v / 50

# Convert CIELAB → RGB
lab = torch.clamp(pred_x1[0], -1, 1).cpu().numpy()
L = (lab[0] + 1) * 50
a = lab[1] * 128
b = lab[2] * 128
rgb = color.lab2rgb(np.stack([L, a, b], axis=-1))
```

Full training code is in the repository.

---

## References

- **Rectified Flow**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **Vision RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

_March 2026_ | Model: [ruwwww/waifu_diffusion](https://huggingface.co/ruwwww/waifu_diffusion)
