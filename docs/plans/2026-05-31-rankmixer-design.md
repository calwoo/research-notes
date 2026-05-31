# Design: RankMixer & TokenMixer-Large Paper Notes

**Date:** 2026-05-31
**Topic slug:** `rankmixer`
**Category:** `papers`
**Multi-note:** yes (2-paper cluster)

## Scope

RankMixer (Zhu et al., CIKM 2025) and its direct follow-up TokenMixer-Large (Jiang et al., 2026) are ByteDance papers that replace quadratic self-attention with hardware-aware token-mixing in large-scale industrial ranking models. RankMixer diagnoses why Transformer-based rankers underperform on GPU (low MFU ~4.5%) and redesigns the interaction layer as multi-head token mixing (inspired by MLP-Mixer) with per-token FFNs, scaling from ~10M to 1B dense parameters and introducing a sparse MoE variant. TokenMixer-Large pushes the same design to 7B–15B parameters using mixing-and-reverting operations, inter-layer residuals, and a Sparse Per-token MoE.

Together they document a coherent scaling trajectory for industrial recommenders and serve as the primary exemplar of applying vision-architecture ideas (MLP-Mixer) to CTR/re-ranking at production scale.

## Files to Create

| File | Purpose |
|------|---------|
| `papers/rankmixer/rankmixer.md` | Full paper note on RankMixer: architecture, MFU analysis, scaling experiments, MoE variant |
| `papers/rankmixer/tokenmixer-large.md` | Full paper note on TokenMixer-Large: mixing-and-reverting, inter-layer residuals, 7B–15B scale |

## Note Structure

### rankmixer.md

1. **Header** — title, authors/venue, TL;DR table, Relations, TOC
2. **Background & Motivation** — DLRM baseline, why Transformers have low MFU in RecSys, the "memory-bound" vs "compute-bound" dichotomy
3. **Architecture** — multi-head token mixing module; per-token FFN; overall model design diagram
4. **Hardware Efficiency Analysis** — MFU derivation, arithmetic intensity, comparison to attention; why mixing is compute-bound
5. **Scaling Experiments** — dense scaling (10M → 1B), training dynamics, offline metrics
6. **Sparse MoE Variant** — routing design, expert capacity, online A/B results
7. **Discussion & Limitations**

### tokenmixer-large.md

1. **Header** — title, authors/venue, TL;DR table, Relations, TOC
2. **Background** — brief recap of RankMixer and what it left on the table
3. **Architecture Innovations** — mixing-and-reverting operations, inter-layer residuals, architectural changes vs RankMixer
4. **Sparse Per-Token MoE** — design differences from RankMixer's MoE, token-level vs sequence-level routing
5. **Scaling to 7B–15B** — training setup, compute budget, offline scaling curves
6. **Online Experiments** — +2.98% GMV result, deployment details
7. **Discussion & Limitations**

## References

- RankMixer: https://arxiv.org/abs/2507.15551
- TokenMixer-Large: https://arxiv.org/abs/2602.06563
- MLP-Mixer: https://arxiv.org/abs/2105.01601
- DLRM: https://arxiv.org/abs/1906.00091
- DCN V2: https://arxiv.org/abs/2008.13535
- DHEN: https://arxiv.org/abs/2203.11014
- Wukong: https://arxiv.org/abs/2403.02545
- Scaling Laws for NLMs: https://arxiv.org/abs/2001.08361
