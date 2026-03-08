# Design: Mixture of Experts Concept Note

**Date:** 2026-03-08
**Topic slug:** `mixture-of-experts`
**Category:** `concepts`

## Scope

Mixture of Experts (MoE) is a class of neural network architectures in which a learned routing function selectively activates a sparse subset of specialized sub-networks ("experts") for each input, keeping computational cost constant as model capacity scales. This note covers MoE from first principles: the formal routing objective, the sparsity mechanisms that make training tractable, load balancing as an auxiliary optimization problem, and the major architectural instantiations (token-level MoE in Transformers, expert-choice vs. token-choice routing, hierarchical MoE).

The focus is on the mathematical structure of MoE — routing as a categorical distribution, the softmax-top-k approximation, load balancing loss derivation, and the capacity factor tradeoff — rather than engineering details. The note will trace the evolution from Jacobs et al. 1991 (soft MoE) through Shazeer 2017 (sparsely-gated MoE) to Switch Transformer and Mixtral, showing how each paper addressed a concrete mathematical or optimization problem left open by its predecessor.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/mixture-of-experts/note.md` | Main research note |
| `concepts/mixture-of-experts/exercises.md` | Problem set |
| `concepts/mixture-of-experts/solutions.md` | Full answer key |

## Note Structure

1. **Motivation and Background** — why sparse conditional computation; the parameter-efficiency argument; history from soft gating (1991) to hard sparse gating (2017)
2. **The MoE Layer: Formal Definition** — general MoE formulation; the gating network; soft vs. hard (top-k) gating; relation to ensemble methods
3. **Sparsely-Gated MoE (Shazeer 2017)** — top-k gating; noisy top-k for exploration; sparsity and the load imbalance problem
4. **Load Balancing** — the importance loss; the load loss; derivation of auxiliary loss terms; capacity factor and token dropping
5. **Switch Transformer (Fedus 2021)** — top-1 routing; simplification argument; scaling behavior; expert parallelism
6. **Expert-Choice Routing** — token-choice vs. expert-choice; guaranteed load balance; relation to optimal transport
7. **Mixtral and Modern Dense-Sparse Tradeoffs** — Mixtral 8x7B architecture; practical scaling laws for MoE; FLOP-matched comparison to dense models
8. **Training Dynamics and Collapse** — expert collapse; entropy regularization; router z-loss; initialization strategies
9. **References**

## Exercise Structure

1. **Derivation problems** — soft gating gradient derivation; top-k as argmax relaxation; load balancing loss gradient; capacity factor token-drop probability; expert-choice as a bipartite matching
2. **Conceptual questions** — MoE vs. ensemble vs. product of experts; why top-1 routing works as well as top-k; expert collapse mechanism; when MoE outperforms dense at matched FLOPs
3. **Implementation sketches** — top-k gating with noisy exploration; distributed expert dispatch (all-to-all); capacity factor token masking

## References

- Jacobs et al. 1991 — Adaptive mixtures of local experts (original soft MoE)
- Shazeer et al. 2017 — Outrageously Large Neural Networks (sparsely-gated MoE): https://arxiv.org/abs/1701.06538
- Lepikhin et al. 2021 — GShard: Scaling Giant Models with Conditional Computation: https://arxiv.org/abs/2006.16668
- Fedus et al. 2021 — Switch Transformers: Scaling to Trillion Parameter Models: https://arxiv.org/abs/2101.03961
- Zhou et al. 2022 — Mixture-of-Experts with Expert Choice Routing: https://arxiv.org/abs/2202.09368
- Jiang et al. 2024 — Mixtral of Experts: https://arxiv.org/abs/2401.04088
