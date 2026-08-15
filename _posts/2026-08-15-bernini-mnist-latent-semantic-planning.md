---
layout: post
title: "Inside ByteDance's Video Models: Building a Generative MLLM Planner from Scratch"
date: 2026-08-15
description: "ByteDance's video models (Seedance / Seaweed) are rumored to scale up to 200B parameters. Here is how their generative MLLM planning architecture actually works, and how we built a toy reproduction from scratch."
tags: [generative-ai, mllm, diffusion-models, computer-vision, deep-learning]
categories: [ML-Engineering, Generative-AI]
giscus_comments: false
---

## 1. Scaling Vision to 200B: The Next Frontier in Generative AI

When ByteDance showcased their latest flagship video generation models (Seedance / Seaweed), the AI world witnessed a quantum leap in prompt understanding, complex physics simulation, and temporal consistency. Industry insights point to a bold architectural direction: scaling generative vision models up to **~200 Billion parameters**.

This signals a transformative turning point for generative media: **How do we push visual generation to the scale of frontier language models without diffusion instability or extreme compute bottlenecks?**

The answer lies in a powerful paradigm shift highlighted in ByteDance's **Bernini** research: **Unlocking Large Language Models as Latent Semantic Planners.**

Rather than treating image or video synthesis as a brute-force diffusion problem, this architecture blends the cognitive reasoning of MLLMs with the continuous precision of Flow Matching. To explore and teach the inner mechanics of this next-generation vision stack, I built **[Bernini-MNIST](https://github.com/ruwwww/bernini-mnist)**—an end-to-end toy implementation powered by `Qwen3-0.6B`, continuous ViT latents, and 2D flow rendering.

**Weights & Code**: [🤗 Hugging Face Hub](https://huggingface.co/ruwwww/bernini-mnist) · [GitHub Repository](https://github.com/ruwwww/bernini-mnist)

---

## 2. The Core Idea: Split Planning from Painting

Traditional diffusion models try to do everything in a single neural network: understand text, plan the scene layout, simulate physics, and paint high-res pixels.

Bernini splits the task into two specialized stages:

{% include figure.liquid path="assets/img/posts/bernini-mnist/bernini-mnist-architecture.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    The decoupled system: The Planner MLLM plans the visual blueprint in abstract feature space, and the Diffusion Renderer paints the physical pixels.
</div>

1. **The Planner MLLM**: A Large Language Model (`Qwen3-0.6B`) works in abstract visual space. Given a prompt (e.g. digit class `8`), it plans a sequence of **16 continuous visual tokens** representing a $4 \times 4$ spatial grid.
2. **The Diffusion Renderer**: A lightweight convolutional denoiser (`ConvResnet`) takes those 16 planned tokens and paints the final clean pixels.

---

## 3. How the Planner Works: ViT + Qwen Attention

Instead of discrete tokens (like VQ-VAE codebooks that round values and cause visual glitches), Bernini uses **continuous visual vectors** extracted from a Vision Transformer (ViT):

{% include figure.liquid path="assets/img/posts/bernini-mnist/bernini-mnist-attention.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    How the visual tokens flow into Qwen: A 28×28 image is split into 16 continuous patch vectors, which feed directly into Qwen's bidirectional self-attention alongside the class prompt.
</div>

* A pretrained Vision Transformer splits an image into a $4 \times 4$ grid ($16$ continuous vectors).
* The sequence `[Class Prompt, Visual Tokens 1..16]` is passed into **Qwen's self-attention layers**.
* Qwen handles all the heavy lifting: learning how the patches relate to each other in 2D space.
* The output head on top of Qwen is just a lightweight MLP that predicts velocities for continuous diffusion.

---

## 4. How Inference Works: Step-by-Step Masked Refinement

When generating a new image from scratch, the system uses an iterative refinement loop based on **MaskGIT**:

{% include figure.liquid path="assets/img/posts/bernini-mnist/bernini-mnist-inference.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    The generation loop: Qwen progressively unmasks and refines the 16 visual tokens step-by-step, and then hands the complete blueprint to the ConvFlow denoiser.
</div>

1. **Step 1 (All Masked)**: All 16 token slots start as `[MASK]`. Qwen looks at the class label and predicts a rough draft for all patches.
2. **Step 2 (Commit Core Strokes)**: The model commits the high-confidence center tokens (e.g. the solid body of the digit) and keeps peripheral patches masked.
3. **Step 3 (Context-Aware Boundary Polish)**: The committed center tokens are fed back into Qwen's attention. Because Qwen can see the committed center, it easily predicts the matching boundary curves.
4. **Step 4 (Final Painting)**: Once all 16 continuous tokens are unmasked, they are sent to the 2D ConvFlow Denoiser, which denoises random noise into clean pixels in 30 steps.

---

## 5. Visual Results

Let's look at how the generated samples compare to real digits:

### Real vs. Reconstructed vs. Generated

{% include figure.liquid path="assets/img/posts/bernini-mnist/comparison_real_rec_gen.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Top row: Real MNIST test images. Middle row: Reconstruction from real ViT tokens (testing the renderer in isolation). Bottom row: Pure end-to-end generation from class labels (Qwen Planner → ConvFlow Renderer).
</div>

* **Row 2 (Reconstruction)**: Proves that 16 continuous tokens ($4 \times 4$ grid) carry all the geometric detail needed to paint clean, solid digits.
* **Row 3 (Pure Generation)**: Shows that Qwen successfully generates smooth, unbroken handwritten digits from scratch with zero background noise.

---

### The 10×10 Generation Matrix

{% include figure.liquid path="assets/img/posts/bernini-mnist/evaluation_grid.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    10×10 generation gallery across all digit classes (0 to 9) using 4-step progressive refinement.
</div>

* Notice the natural diversity across samples: different stroke slants, loops, and handwriting curvatures for every digit class.

---

### Continuous Denoising Trajectory Gallery (Classes 0–9)

To see the continuous flow matching process in action, here is the step-by-step denoising trajectory for every digit class from 0 to 9 across 30 Euler ODE steps (from pure Gaussian static at $t=0.0$ on the left to clean digits at $t=1.0$ on the right):

{% include figure.liquid path="assets/img/posts/bernini-mnist/ode_trajectories_all_classes.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
    Step-by-step continuous denoising trajectories across classes 0 to 9 over 30 Euler ODE steps ($t=0.0 \to 1.0$). Pure Gaussian noise is progressively integrated into clean, crisp handwritten strokes guided by the semantic plan.
</div>

---

## 6. Takeaways

This 2-stage generative paradigm shows why ByteDance's vision stack is so compelling:

1. **Planning vs. Painting**: Letting LLMs handle structure and reasoning while letting diffusion handle pixels makes scaling to hundreds of billions of parameters feasible.
2. **Continuous Latents**: Working in continuous float feature space avoids codebook quantization artifacts.
3. **Simplicity**: The LLM handles cross-token attention, while the flow head and renderer stay lightweight and fast.

Check out the full repository and open-source checkpoints:
* **GitHub Repository**: [https://github.com/ruwwww/bernini-mnist](https://github.com/ruwwww/bernini-mnist)
* **Hugging Face Model Hub**: [https://huggingface.co/ruwwww/bernini-mnist](https://huggingface.co/ruwwww/bernini-mnist)
