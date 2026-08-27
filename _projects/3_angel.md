---
layout: page
title: Angel
description: Next-generation distributed LLM RL framework with VERL-style 3D HybridEngine architecture (In Development)
img: assets/img/7.jpg
importance: 3
category: ML Systems & Compilers
github: https://github.com/ruwwww/angel
---

**Angel** is the architectural successor to Yoshida, designed to achieve **VERL-style 3D HybridEngine scalability** while maintaining pure Python modularity.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <span class="badge badge-warning">Active Development</span>
        <a href="https://github.com/ruwwww/angel" target="_blank" class="btn btn-outline-primary btn-sm ml-2">GitHub Repository</a>
    </div>
</div>

### Architectural Goals

- **Zero-Copy In-GPU Weight Sharing:** Direct memory pointer swapping between PyTorch FSDP training ranks and inference execution backends without moving parameters across PCIe or disk.
- **Asynchronous Train/Rollout Pipelining:** Double-buffered generation queues where the rollout engine generates batch $D_{t+1}$ on an asynchronous CUDA stream while the training ranks compute the backward pass on batch $D_t$.
- **Space-Multiplexed GPU Partitioning:** Support for hybrid cluster allocation (e.g. 6 training GPUs + 2 dedicated vLLM rollout GPUs) with background NCCL weight broadcasts.
- **Online Speculative Decoding:** Integrating Multi-Token Prediction (MTP) draft training during RL execution to double rollout generation throughput.
