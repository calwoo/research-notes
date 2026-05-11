# Design: Rotary Position Embeddings (RoPE) and Variants

**Date:** 2026-05-10
**Topic slug:** `rotary-embeddings`
**Category:** `concepts/deep-learning-engineering`
**Multi-note:** no

## Scope

This note covers Rotary Position Embedding (RoPE) from first principles — the rotation-matrix construction, its complex-number formulation, and the key properties that make it the dominant positional encoding in modern LLMs. The note then derives the context-length extension problem and treats two main solutions: NTK-aware frequency scaling and YaRN (Yet Another RoPE extensioN), including the temperature scaling and "mixes of interpolations" tricks in YaRN.

The mathematical treatment is thorough: we derive the RoPE operation as a block-diagonal rotation, show why it encodes *relative* position via the dot-product of two rotated vectors, and analyze the long-term decay property. For the extension variants we work through the NTK lens (high-frequency vs. low-frequency components) and explain what YaRN adds on top.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/deep-learning-engineering/rotary-embeddings.md` | Full concept note: RoPE derivation, NTK interpolation, YaRN |

## Note Structure

1. **Introduction** — motivation for positional encodings; why relative > absolute; overview of RoPE
2. **RoPE Derivation** — formal construction as block-diagonal rotation matrices; complex-number shorthand; frequency schedule
3. **Key Properties** — relative position in the dot product; long-term decay; equivariance to sequence shifts
4. **The Context Extension Problem** — what breaks at inference-time when sequence length exceeds training length; perplexity explosion
5. **NTK-Aware Scaling** — neural tangent kernel interpretation; base-frequency rescaling; why it works for low-frequency components but not high
6. **YaRN** — dynamic interpolation: "ramp" function partitioning frequencies; NTK interp for high-freq, linear interp for low-freq; temperature scaling of attention logits; fine-tuning recipe
7. **Comparison and Practical Guidance** — summary table of methods; recommended approach per use case

Exercises and solutions inline after each major section.

## References

- Su et al. (2021) — RoFormer: Enhanced Transformer with Rotary Position Embedding ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))
- bloc97 (2023) — NTK-Aware Scaled RoPE (Reddit/GitHub blog post)
- Peng et al. (2023) — YaRN: Efficient Context Window Extension of Large Language Models ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071))
- Chen et al. (2023) — Extending Context Window of Large Language Models via Positional Interpolation ([arXiv:2306.15595](https://arxiv.org/abs/2306.15595))
