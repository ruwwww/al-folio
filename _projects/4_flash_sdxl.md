---
layout: page
title: Flash-SDXL
description: High-throughput end-to-end web serving engine and inference optimization pipeline for SDXL models
img: assets/img/3.jpg
importance: 4
category: ML Systems & Compilers
github: https://github.com/ruwwww/flash-sdxl
---

**Flash-SDXL** is an end-to-end web application and high-performance inference engine optimized for serving the **Stable Diffusion XL (SDXL)** model with minimal latency and high concurrency.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <a href="https://github.com/ruwwww/flash-sdxl" target="_blank" class="btn btn-primary btn-sm">GitHub Repository</a>
    </div>
</div>

### Features & Architecture

- **Serving Pipeline Optimization:** Streamlined execution backend integrating FlashAttention kernels, FP16/BF16 mixed-precision inference, and TensorRT compilation for sub-second generation.
- **Dynamic Request Scheduling:** Asynchronous job queues and worker pools designed to prevent GPU VRAM starvation under concurrent user load.
- **Full-Stack Application:** Modern TypeScript web frontend paired with a modular Python API backend for prompt configuration, seed management, and real-time generation streaming.
