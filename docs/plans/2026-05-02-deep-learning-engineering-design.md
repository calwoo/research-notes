# Design: Deep Learning Engineering Concept Notes

**Date:** 2026-05-02
**Topic slug:** `deep-learning-engineering`
**Category:** `concepts`
**Multi-note:** yes

## Scope

This cluster covers practical techniques for improving training efficiency, inference speed, and memory utilization in modern deep learning systems — with emphasis on large language models and transformer-based architectures. The focus is on techniques that are both widely used in production and mathematically interesting: not just "what to do" but why each technique works, what it costs, and how it interacts with other techniques.

The entry point is weight tying, which is a clean, self-contained example of a technique that trades parameter count for a structural prior (the input and output embedding operate in the same semantic space). From there the cluster branches into memory-efficiency techniques (gradient checkpointing, mixed precision, quantization), inference acceleration (KV cache, speculative decoding, continuous batching), and parameter-efficient fine-tuning (LoRA, adapters).

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/deep-learning-engineering/overview.md` | Topic index, subtopic map, dependency graph, master references |
| `concepts/deep-learning-engineering/weight-tying.md` | Weight tying in language models — parameter sharing between input embedding and output projection |
| `concepts/deep-learning-engineering/gradient-checkpointing.md` | Activation recomputation: trading compute for memory |
| `concepts/deep-learning-engineering/mixed-precision.md` | FP16/BF16 training, loss scaling, and numerical stability |
| `concepts/deep-learning-engineering/kv-cache.md` | Key-value caching for autoregressive inference |
| `concepts/deep-learning-engineering/speculative-decoding.md` | Draft-then-verify decoding for latency reduction |
| `concepts/deep-learning-engineering/quantization.md` | PTQ, QAT, GPTQ, AWQ — reducing bit-width at inference |
| `concepts/deep-learning-engineering/lora.md` | Low-rank adaptation — parameter-efficient fine-tuning |
| `concepts/deep-learning-engineering/flash-attention.md` | IO-aware attention — fusing softmax to avoid HBM round-trips |

## Note Structure (weight-tying.md)

1. **Motivation** — parameter count in LMs; why embeddings dominate
2. **The Technique** — formal definition: $W_{\text{out}} = W_{\text{emb}}^\top$; where it appears in the forward pass
3. **Why It Works** — geometric argument: embeddings and logits live in the same semantic space; Press & Wolf (2017) empirical analysis
4. **Formal Analysis** — effect on gradient flow; tied vs. untied gradient decomposition
5. **When It Hurts** — very large vocabularies; domain-specific output distributions; factored embedding alternatives
6. **Variations** — partial tying, factored embeddings (ALBERT), output projection scaling
7. Inline exercises distributed throughout

## Planned Subtopics (multi-note)

| File | Description |
|------|-------------|
| `weight-tying.md` | Parameter sharing between input embedding and output projection |
| `gradient-checkpointing.md` | Recompute activations on backward pass to reduce peak memory |
| `mixed-precision.md` | FP16/BF16 training with loss scaling; master weight copies |
| `kv-cache.md` | Cache past K/V tensors during autoregressive generation |
| `speculative-decoding.md` | Small draft model + large verifier for parallel token acceptance |
| `quantization.md` | Post-training and quantization-aware training; calibration methods |
| `lora.md` | Freeze base weights; train low-rank $\Delta W = BA$ |
| `flash-attention.md` | Tiled SRAM-resident attention; IO complexity analysis |

## References

- Press & Wolf (2017) — "Using the Output Embedding to Improve Language Models" (weight tying)
- Vaswani et al. (2017) — Transformer (uses weight tying)
- Chen et al. (2016) — "Training Deep Nets with Sublinear Memory Cost" (gradient checkpointing)
- Micikevicius et al. (2018) — "Mixed Precision Training"
- Pope et al. (2023) — "Efficiently Scaling Transformer Inference" (KV cache analysis)
- Leviathan et al. (2023) — "Fast Inference from Transformers via Speculative Decoding"
- Dao et al. (2022) — "FlashAttention"
- Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023) — "GPTQ: Accurate Post-Training Quantization"
