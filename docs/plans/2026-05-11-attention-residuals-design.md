# Design: Attention Residuals Paper Note

**Date:** 2026-05-11
**Topic slug:** `attention-residuals`
**Category:** `papers`
**Multi-note:** no (single paper)

## Scope

This note covers the Kimi Team's Attention Residuals paper (arXiv:2603.15031, March 2026), which proposes replacing the fixed unit-weight residual accumulation in PreNorm Transformers with learned softmax attention over preceding layer outputs. The note emphasizes mathematical depth: a formal analysis of PreNorm dilution (O(√L) hidden-state growth), a derivation of the gradient amplification mechanism via normalization Jacobians, the time-depth duality framing, and the structured-matrix unification of residual variants.

The secondary goal is to provide strong intuition for *why* standard PreNorm causes disproportionate gradients in early layers — the normalization Jacobian argument — since this is the core motivation for AttnRes and is underexplained in most presentations of the work.

## Files to Create

| File | Purpose |
|------|---------|
| `papers/attention-residuals.md` | Full paper note with TL;DR, derivations, exercises inline |
| `papers/figures/attention-residuals/` | Figures extracted from the arxiv source |

## Note Structure

1. **The Residual Bottleneck** — fixed-weight accumulation, unrolled recurrence, limitations
2. **PreNorm Dilution: Formal Analysis** — O(√l) growth argument, dilution ratio, empirical manifestation (output magnitude growth)
3. **Why Gradients Blow Up in Early Layers** — backward pass through normalization, Jacobian scaling with ‖h_l‖, cumulative amplification
4. **The Time-Depth Duality** — residuals as depth-axis RNNs, the parallel to sequence-side attention
5. **Attention Residuals (AttnRes)** — formal mechanism, pseudo-query wl, RMSNorm on keys
6. **Block AttnRes** — block compression, inter/intra-block attention, efficiency O(Nd)
7. **Residuals as Structured Matrices** — depth mixing matrix M, semiseparable rank, linear vs. softmax depth attention
8. **Training Dynamics Analysis** — empirical output/gradient magnitude, how AttnRes breaks the dilution cycle
9. **Learned AttnRes Patterns** — attention sinks, diagonal dominance, layer specialization
10. **Experimental Results** — scaling laws, ablations, benchmarks
11. **When Does AttnRes Win?** — Ziming Liu's blog: structured vs. memorization tasks

## References

- [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (Xiong et al., 2020) — PreNorm analysis
- [SiameseNorm](https://arxiv.org/abs/2602.08064) (Tianyu Li et al., 2026) — formal PreNorm dilution analysis
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015)
- [Hyper-Connections](https://arxiv.org/abs/2409.19606) (Zhu et al., 2025)
- [DenseFormer](https://arxiv.org/abs/2402.02622) (Pagliardini et al., 2024)
- [When does Kimi's "Attention Residuals" Work?](https://kindxiaoming.github.io/blog/2026/attention-residual/) (Ziming Liu, 2026)
