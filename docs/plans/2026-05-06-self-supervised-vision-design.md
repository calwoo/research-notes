# Design: Self-Supervised Vision (BYOL, Barlow Twins, VICReg) Concept Note

**Date:** 2026-05-06
**Topic slug:** `self-supervised-vision`
**Category:** `concepts`
**Multi-note:** no

## Scope

This note gives a pedagogical account of the post-contrastive wave of self-supervised visual representation learning, tracing the intellectual arc from contrastive baselines (SimCLR, MoCo) through three landmark papers that each solve the representational collapse problem via a fundamentally different mechanism:

1. **BYOL** (Grill et al., 2020) — asymmetric online/target network pair with momentum update; no negatives required.
2. **Barlow Twins** (Zbontar et al., 2021) — drives the cross-correlation matrix of twin embeddings toward the identity, decorrelating output dimensions.
3. **VICReg** (Bardes et al., 2021) — applies three explicit per-branch regularizers (variance, invariance, covariance) without weight sharing or batch normalization.

The note emphasizes rigorous mathematical derivation of each objective, formal analysis of *why* collapse is avoided in each case, and the conceptual lineage connecting all three.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/self-supervised-vision/ssl-vision.md` | Single comprehensive concept note |

## Note Structure

1. **Background: the self-supervised setup** — augmentation pipelines, the joint embedding framework, formal notation
2. **The collapse problem** — why trivially constant representations satisfy naive objectives; necessary conditions to avoid collapse
3. **Contrastive baselines** — SimCLR and MoCo as motivation; NT-Xent loss, queue/momentum encoder tricks; the large-batch bottleneck
4. **BYOL: bootstrapping without negatives** — online/target architecture, EMA update rule, predictor asymmetry, formal analysis of why collapse doesn't occur (stop-gradient + EMA dynamics)
5. **Barlow Twins: redundancy reduction** — cross-correlation matrix objective, connection to Horace Barlow's redundancy reduction hypothesis in neuroscience, whitening interpretation
6. **VICReg: explicit regularization** — three-term loss derivation, per-branch application, relaxation of batch normalization and weight-sharing requirements
7. **Unified perspective** — taxonomy of collapse-prevention mechanisms: contrastive (negatives), architectural (asymmetry/stop-gradient), spectral (redundancy reduction/covariance), explicit (variance regularization)
8. **Exercises** — inline after each section

## References

- [BYOL] Grill et al., "Bootstrap Your Own Latent," NeurIPS 2020 — https://arxiv.org/abs/2006.07733
- [Barlow Twins] Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction," ICML 2021 — https://arxiv.org/abs/2103.03230
- [VICReg] Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning," ICLR 2022 — https://arxiv.org/abs/2105.04906
- [SimCLR] Chen et al., "A Simple Framework for Contrastive Learning," ICML 2020 — https://arxiv.org/abs/2002.05709
- [MoCo] He et al., "Momentum Contrast for Unsupervised Visual Representation Learning," CVPR 2020 — https://arxiv.org/abs/1911.05722
