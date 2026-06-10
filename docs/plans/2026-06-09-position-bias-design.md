# Design: Position Bias in Search Ranking — Concept Note

**Date:** 2026-06-09
**Topic slug:** `position-bias`
**Category:** `concepts`
**Multi-note:** no

## Scope

This note covers the theory and practice of position bias in search and recommendation ranking systems. Position bias arises because users are more likely to examine (and click) items shown at higher ranks regardless of their true relevance, making raw click logs a noisy training signal for learning-to-rank models. The note develops the formal framework from click models and the Examination Hypothesis, derives the canonical Inverse Propensity Scoring (IPS) estimator for unbiased learning-to-rank (ULTR), covers propensity estimation strategies (randomization, regression-EM, intervention harvesting, DLA), and surveys industrial deployment techniques including PAL, doubly robust estimators, and trust/selection bias extensions.

The mathematical treatment emphasizes causal structure: clicks are treated as potential outcomes, propensities as counterfactual probabilities, and unbiasedness is defined in terms of expected risk over the logging policy. Formal derivations are included for the IPS gradient, the Regression-EM propensity update, and the DLA dual objective. Industrial case studies from Google, Huawei, eBay, and Baidu ground the theory.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/position-bias/position-debiasing.md` | Single note covering the full landscape: click models, IPS/ULTR, propensity estimation, doubly robust methods, extensions, industrial deployments |

## Note Structure

1. **Motivation and Problem Setup** — informal description of position bias, why raw clicks mislead rankers
2. **Click Models and the Examination Hypothesis** — position-based model (PBM), cascade model, UBM; formal definition of propensity
3. **Causal Framework** — potential outcomes notation, logging policy, MNAR data, identifiability conditions
4. **Inverse Propensity Scoring (IPS) Estimator** — derivation of the IPS loss, unbiasedness proof, variance issues and clipping
5. **Propensity Estimation** — result randomization, Regression-EM (Wang et al. 2018), intervention harvesting (Fang et al. 2019, Agarwal et al. 2019)
6. **Dual Learning Algorithm (DLA)** — joint ranker + propensity training, dual objective, convergence intuition
7. **Doubly Robust Estimators** — imputation model, DR loss, bias-variance tradeoff vs pure IPS
8. **Extensions: Trust Bias, Selection Bias, Context-Dependent Propensity** — TrustPBM, CPBM, MNAR selection
9. **Industrial Deployment** — PAL (Huawei), Google personal search (Regression-EM), eBay eCommerce, Baidu ULTR at scale
10. **Evaluation Under Bias** — counterfactual offline evaluation, IPS-DCG, comparison to online A/B

## References

- Joachims et al. (2005, 2017) — click bias fundamentals and founding ULTR paper
- Craswell et al. (2008) — Examination Hypothesis and click position-bias models
- Schnabel et al. (2016) — IPS for recommendations (potential outcomes framework)
- Wang et al. (2018) — Regression-EM for propensity estimation (Google)
- Ai et al. (2018) — Dual Learning Algorithm (DLA)
- Agarwal et al. (2019a,b) — counterfactual LTR framework and extremum propensity estimator
- Fang et al. (2019) — CPBM, intervention harvesting
- Guo et al. (2019) — PAL (Huawei, production CTR debiasing)
- Agarwal et al. (2019c) — TrustPBM (Google, trust bias)
- Hu et al. (2019) — Unbiased LambdaMART
- Aslanyan & Porwal (2019) — eBay eCommerce debiasing
- Oosterhuis (2022) — Doubly Robust ULTR
- Chen et al. (2022) — Vectorization-based ULTR
- Hager et al. (2024) — Baidu large-scale ULTR empirical study
