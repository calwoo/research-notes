# Plan: Matryoshka Representation Learning Paper Note

## Task

Write a paper note at `papers/matryoshka-representation-learning.md` for the MRL paper (Kusupati et al., NeurIPS 2022).

## Sources Consulted

1. https://arxiv.org/html/2205.13147 — arXiv HTML version of the original paper
2. https://ar5iv.labs.arxiv.org/html/2205.13147 — ar5iv HTML rendering
3. https://aniketrege.github.io/blog/2024/mrl/ — pedagogical blog post
4. https://arxiv.org/abs/2402.14776 — 2D Matryoshka Sentence Embeddings (follow-up)
5. https://arxiv.org/abs/2407.20243 — Matryoshka-Adaptor (follow-up)
6. Web searches for OpenAI text-embedding-3, MatFormer, and multimodal extensions

## Sections Planned

1. Motivation and Background (capacity-vs-cost tradeoff, Matryoshka doll analogy)
2. The MRL Objective (formal loss, nesting constraint, MRL-E weight tying)
3. Training MRL Models (training loop modification, c_m weights, contrastive/MLM extensions)
4. Inference-Time Flexibility (adaptive classification, adaptive retrieval funnel, Pareto frontier)
5. Theoretical Grounding (why prefixes beat PCA, interpolation property, information bottleneck)
6. Empirical Results (ImageNet classification, retrieval mAP, few-shot, robustness)
7. Extensions and Follow-up (2D-MRL, Matryoshka-Adaptor, M3, OpenAI text-embedding-3, MatFormer)
8. References

## Inline Exercises (12 planned)

Distributed across sections: 8 mathematical derivations + 4 computational/empirical exercises.

## Figures

Fetched from https://arxiv.org/html/2205.13147v4/ using direct image URLs:
- x3.png — MRL overview (inference + training diagram)
- x9.png — ImageNet-1K linear classification accuracy
- x12.png — 1-NN accuracy interpolation across scales
- x13.png — Adaptive classification Pareto curve
- x14.png — mAP@10 retrieval comparison

## Status

Completed. File at `papers/matryoshka-representation-learning.md`.
