# Design: Spectral Bias and Fourier Kernel Remedies

**Date:** 2026-06-22
**Topic slug:** `spectral-bias`
**Category:** `concepts`
**Multi-note:** no

## Scope

Neural networks trained with gradient descent exhibit a strong inductive bias toward low-frequency functions — they fit coarse, smooth structure before capturing fine detail. This *spectral bias* (also called *frequency principle*) is not incidental: it emerges from the eigenspectrum of the Neural Tangent Kernel (NTK) and governs convergence rates across the entire frequency spectrum. The note will derive this formally, then show how lifting inputs through a Fourier feature mapping reshapes the NTK eigenspectrum so that high-frequency components are learnable, connecting to Rahimi & Recht random kitchen sinks as the approximation mechanism.

The note covers: the empirical phenomenon; its formal characterization via the NTK and kernel regression; the relationship to the stationary RBF kernel; the Fourier feature map construction; convergence rate analysis before and after the mapping; and practical implications for implicit neural representations (NeRF, occupancy networks).

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/spectral-bias/spectral-bias-fourier-kernels.md` | Single note covering diagnosis and remedy |

## Note Structure

1. **Introduction** — empirical phenomenon, why it matters for implicit neural representations
2. **The Spectral Bias Phenomenon** — definition, empirical evidence from Rahaman et al.
3. **NTK Analysis of Spectral Bias** — NTK as a kernel, eigendecomposition, convergence rates per frequency mode
4. **Fourier Feature Maps** — Bochner's theorem, random Fourier features (Rahimi & Recht), the positional encoding as a deterministic variant
5. **NTK Spectrum After Lifting** — how the feature map changes the kernel and equalizes eigenvalues
6. **Convergence Rate Analysis** — formal statement of improved high-frequency learning
7. **Applications** — NeRF positional encodings, SIREN, implicit neural representations
8. **Exercises** (inline after each section)

## References

- Rahaman et al. (2019) "On the Spectral Bias of Neural Networks" — ICML 2019
- Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" — NeurIPS 2020
- Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines" — NIPS 2007
- Mildenhall et al. (2020) "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
