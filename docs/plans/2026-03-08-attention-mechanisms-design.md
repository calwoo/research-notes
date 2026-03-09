# Design: Attention Mechanisms Concept Note

**Date:** 2026-03-08
**Topic slug:** `attention-mechanisms`
**Category:** `concepts`

## Scope

This note covers the full arc of attention mechanisms from standard softmax attention through modern
linear attention variants. The primary reference is the YouTube video "Attention Mechanisms" by
Umar Jamil (pUCWwGR5WmQ), supplemented by primary papers.

The topic naturally splits into two logical files:

1. **Standard attention** (`note.md`) — The mathematical foundations: query/key/value projections,
   dot-product attention, causal masking, multi-head attention, and KV caching. This is the
   bedrock that all modern transformer-based LLMs build on.

2. **Linear attention** (`linear-attention.md`) — The evolution beyond softmax: linear attention as
   a recurrent model with matrix-valued hidden state, chunkwise-parallel training, decay/gating
   mechanisms (RetNet, Mamba-2, GLA), and the neural memory / fast-weight perspective connecting
   linear attention to online linear regression.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/attention-mechanisms/note.md` | Standard softmax attention: Q/K/V formalism, causal masking, multi-head, KV cache |
| `concepts/attention-mechanisms/linear-attention.md` | Linear attention: recurrent form, chunkwise-parallel, gating, neural memory |
| `concepts/attention-mechanisms/exercises.md` | Problem set spanning both notes |
| `concepts/attention-mechanisms/solutions.md` | Full answer key |

## Note Structure

### `note.md` — Standard Attention

1. Introduction and motivation
2. Single-head scaled dot-product attention
   2.1 Query, key, value projections
   2.2 Attention score computation and causal masking
   2.3 Softmax normalization and weighted aggregation
   2.4 Output projection
3. Matrix form of attention
4. Multi-head attention
5. KV caching
   5.1 Motivation: avoiding redundant computation
   5.2 Memory growth with sequence length
   5.3 Variants: GQA, MLA, sparse attention (brief)
6. References

### `linear-attention.md` — Linear Attention

1. Motivation: beyond quadratic/linear memory tradeoffs
2. Linear attention: removing softmax
   2.1 Factoring out the query vector
   2.2 The state matrix S_t and recurrence relation
   2.3 Comparison with standard attention
3. Training efficiency challenges
4. Chunkwise-parallel form
   4.1 Intra-chunk parallel computation
   4.2 Inter-chunk recurrent state passing
   4.3 Output decomposition
5. Decay and gating mechanisms
   5.1 Constant decay (RetNet, Lightning Attention)
   5.2 Scalar data-dependent decay (Mamba-2, RWKV)
   5.3 Vector/matrix data-dependent gates (GLA, HGRN2, RWKV-6)
6. Neural memory perspective
   6.1 Associative memory in standard attention
   6.2 Linear attention as online linear regression
   6.3 Connection to fast weights / Hebbian learning
   6.4 The delta rule
7. References

## Exercise Structure

1. **Mathematical Development** (~16 problems)
   - Derive the scaled dot-product attention formula from first principles
   - Show that softmax attention is equivalent to a normalized weighted associative lookup
   - Prove the matrix form equivalence for the causal case
   - Derive the linear attention recurrence relation from the factored sum
   - Show memory complexity of KV cache vs linear attention state
   - Derive the chunkwise-parallel output decomposition
   - Prove equivalence of chunkwise-parallel and recurrent forms
   - Derive the gradient update rule for online linear regression with negative dot-product loss
   - Show the connection between the delta rule and linear attention state update

2. **Algorithmic Applications** (~6 problems)
   - Pseudocode for KV cache prefill and decode stages
   - Complexity analysis: softmax attention vs linear attention in training and inference
   - Implement chunkwise-parallel attention in pseudocode
   - Analyze trade-offs of scalar vs vector gating mechanisms

## References

- Vaswani et al., "Attention Is All You Need" (2017) — original transformer paper
- Katharopoulos et al., "Transformers are RNNs" (2020) — linear attention formulation
- Sun et al., "Retentive Network" (2023) — RetNet with constant decay
- Dao & Gu, "Transformers are SSMs" (2024) — Mamba-2, chunkwise-parallel SSD
- Yang et al., "Gated Linear Attention" (2024) — GLA with vector gating
- Schmidhuber, "Learning to Control Fast-Weight Memories" (1992) — fast weights origin
- YouTube: "Attention Mechanisms" by Umar Jamil — https://youtu.be/pUCWwGR5WmQ
