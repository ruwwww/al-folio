---
layout: page
title: Axon
description: A minimal deep learning framework built in C++20 from scratch with eager autograd, AVX2 SIMD CPU backend, and GGML-style quantization
img: assets/img/9.jpg
importance: 2
category: ML Systems & Compilers
github: https://github.com/ruwwww/Axon
---

**Axon** is a minimal deep learning framework in **C++20** that implements a complete training and inference pipeline with **zero external ML framework dependencies**.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <a href="https://github.com/ruwwww/Axon" target="_blank" class="btn btn-primary btn-sm">GitHub Repository</a>
    </div>
</div>

### Architectural Highlights

- **C++20 Core & Eager Autograd:** Polymorphic `Node` computational graph recording reverse-mode automatic differentiation natively in C++20.
- **AVX2 SIMD CPU Acceleration:** Hand-tuned vectorization kernels via `KernelRegistry` for dense matrix multiplications (GEMM) and fused elementwise activations.
- **GGML-Style Quantization:** Custom integer quantization routines enabling low-latency CPU inference without third-party runtimes.
- **End-to-End Deep Learning Pipeline:** Complete implementations of Tensor abstractions, Neural Network Modules (`Linear`, `Conv2d`, `BatchNorm`), Optimizers (`SGD`, `AdamW`), and custom binary model serialization.
