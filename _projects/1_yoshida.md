---
layout: page
title: Yoshida
description: Production-grade General LLM Reinforcement Learning & Post-Training Framework
img: assets/img/12.jpg
importance: 1
category: ML Systems & Compilers
github: https://github.com/ruwwww/yoshida
---

**Yoshida** is a high-performance, zero-lock-in General LLM Reinforcement Learning (RL) and Post-Training framework built from first principles in **native PyTorch** (no TRL, no Ray, no Deepspeed-RL lock-in).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <a href="https://github.com/ruwwww/yoshida" target="_blank" class="btn btn-primary btn-sm">GitHub Repository</a>
    </div>
</div>

### Key Capabilities

- **Mathematical Formulations:** Exact native implementations of **DeepSeek-R1 style GRPO** with Schulman $k_3$ non-negative KL penalty, **Online PPO (Actor-Critic)** with Generalized Advantage Estimation ($\text{GAE}(\gamma, \lambda)$), **DPO** with reference-model memory swapping, and **Bradley-Terry Reward Modeling**.
- **High-Throughput Rollout Subsystem:** Colocated in-process **vLLM Engine** delivering **3,300+ tokens/second** on a single GPU with PagedAttention and continuous batching, backed by a zero-dependency Native PyTorch FlashAttention-2 fallback engine.
- **Cutting-Edge Spectral Optimizers:** Built-in **Dion3** (Gram Newton-Schulz with Fractional Row Selection — up to $6\times$ lighter than standard Muon), **Muon**, and **Fused AdamW**.
- **Multi-GPU & Cluster Scaling:** Native `torchrun` support for **FSDP (ZeRO-3)** parameter and optimizer state sharding, coupled with production SLURM HPC batch templates for 16+ GPU nodes.

```bash
# Launch GRPO with colocated vLLM & Dion3 optimizer
python train.py \
  --method grpo \
  --rollout_backend vllm \
  --rollout_gpu_utilization 0.35 \
  --optimizer dion3 \
  --group_size 4
```
