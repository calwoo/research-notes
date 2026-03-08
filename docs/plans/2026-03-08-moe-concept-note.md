# Plan: Mixture of Experts Concept Note

## Goal

Write a comprehensive concept note on Mixture of Experts (MoE) for the knowledge repository at `concepts/mixture-of-experts/note.md`.

## Scope

Sections:
1. Motivation and Background — sparse conditional computation, parameter-efficiency argument, historical lineage Jacobs 1991 → Shazeer 2017 → Switch → Expert-Choice → Mixtral
2. MoE Layer: Formal Definition — notation, soft gating, hard top-k gating, relation to ensembles
3. Sparsely-Gated MoE (Shazeer 2017) — noisy top-k gating, load imbalance problem
4. Load Balancing — importance loss, load loss (CV-based), capacity factor, token dropping
5. Switch Transformer (Fedus 2021) — top-1 routing, simplified auxiliary loss, scaling behavior
6. Expert-Choice Routing — inverted selection, formal spec, guaranteed load balance, bipartite matching
7. Mixtral and Dense-Sparse Tradeoffs — architecture, FLOP-matched comparison, MoE scaling laws
8. Training Dynamics and Collapse — expert collapse, router z-loss, entropy regularization, initialization

## Sources

- Jacobs et al. (1991) — original soft MoE
- Shazeer et al. (2017) arXiv:1701.06538
- Lepikhin et al. (2021) arXiv:2006.16668 (GShard)
- Fedus et al. (2021) arXiv:2101.03961 (Switch Transformers)
- Zoph et al. (2022) arXiv:2202.08906 (ST-MoE)
- Zhou et al. (2022) arXiv:2202.09368 (Expert Choice)
- Jiang et al. (2024) arXiv:2401.04088 (Mixtral)
- Mu and Lin (2025) arXiv:2503.07137 (MoE Survey) — confirmed relevant, included in references

## Status

Completed.
