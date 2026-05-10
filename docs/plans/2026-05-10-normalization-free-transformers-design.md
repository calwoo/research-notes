# Design: Normalization-Free Transformers Concept Note

**Date:** 2026-05-10
**Topic slug:** `normalization-free-transformers`
**Category:** `concepts/deep-learning-engineering`
**Multi-note:** no

## Scope

This note synthesizes two papers — *Transformers without Normalization* (Zhu et al., 2025; DyT) and *Stronger Normalization-Free Transformers* (Chen et al., 2025; Derf) — into a unified conceptual treatment of pointwise functions as drop-in replacements for LayerNorm/RMSNorm. The pedagogical arc moves from empirical observation (LayerNorm ≈ tanh in trained models) through the formal theory of what properties a pointwise replacement needs, to the design of Derf as the best-known instance of that theory.

The note does not rehash basic LayerNorm mechanics (those belong in the planned `normalization.md`). It assumes the reader knows what LayerNorm does and asks: *why does a simple tanh work just as well, and how do we find an even better function?*

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/deep-learning-engineering/normalization-free-transformers.md` | Unified concept note covering DyT, the four-property theory, and Derf |

## Note Structure

1. **Background: What Normalization Is Doing** — brief recap of LayerNorm/RMSNorm mechanics, focusing on the S-curve behavior that motivates the rest
2. **The Tanh Observation** — empirical finding from DyT: trained LayerNorm produces tanh-shaped input-output curves; formal explanation via varying token variances
3. **Dynamic Tanh (DyT)** — formal definition `DyT(x) = γ ⊙ tanh(αx) + β`; how α tracks 1/std during training; drop-in replacement mechanics; experimental results summary
4. **Toward a Theory: Four Properties** — zero-centeredness, boundedness, center sensitivity, monotonicity; ablation evidence for each from the Derf paper
5. **Dynamic erf (Derf)** — formal definition `Derf(x) = γ · erf(αx + s) + β`; why erf satisfies the four properties better than tanh; function search methodology; experimental results
6. **Why Do Pointwise Functions Work At All?** — the generalization argument (higher training loss, better validation); what statistics-free computation buys you
7. **Limitations** — BN incompatibility, LLM α₀ sensitivity, scope restrictions

Inline exercises after each major section (Sections 2, 3, 4, 5). Total target: ~4 exercises.

## References

| Paper | Authors | Link |
|-------|---------|------|
| Transformers without Normalization | Zhu, Chen, He, LeCun, Liu (2025) | https://arxiv.org/abs/2503.10622 |
| Stronger Normalization-Free Transformers | Chen, Lu, Zhu, Sun, Liu (2025) | https://arxiv.org/abs/2512.10938 |
