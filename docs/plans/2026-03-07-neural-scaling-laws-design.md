# Design: Neural Scaling Laws Research Note

**Date:** 2026-03-07
**Topic slug:** `neural-scaling-laws`
**Category:** `concepts`

## Scope

A single unified concept note covering neural scaling laws, anchored on:
- Kaplan et al. (2020) — original OpenAI scaling laws
- Hoffmann et al. (2022) — Chinchilla compute-optimal revision

Style: intuition-first, then full mathematical derivations.

## Files to Create

| File | Purpose |
|------|---------|
| `notes/concepts/neural-scaling-laws.md` | Main research note |
| `exercises/concepts/neural-scaling-laws.md` | Problem set (derivations, conceptual, implementation sketches) |
| `solutions/concepts/neural-scaling-laws.md` | Full answer key |

## Note Structure

### 1. Motivation
- Empirical observation: loss vs. scale is linear in log-log space
- Intuition from statistical physics and power laws in nature
- Why this matters practically for compute budgeting

### 2. Mathematical Setup
- Loss decomposition: L = L_inf + A/N^alpha + B/D^beta
- Irreducible loss L_inf (Bayes error floor)
- Parameter-dependent term A/N^alpha
- Data-dependent term B/D^beta
- Justification for the additive, power-law structure

### 3. Kaplan et al. (2020)
- Empirical scaling laws: L(N), L(D), L(C_min) as univariate power laws
- Compute approximation: C ≈ 6ND
- Efficient frontier derivation: minimize L(N,D) subject to C = 6ND
- Kaplan's result: N ∝ C^0.73, D ∝ C^0.27
- Interpretation: parameters should scale ~3x faster than data

### 4. Chinchilla (2022)
- IsoFLOP analysis: fix C, sweep N and D jointly
- Empirical finding: equal scaling of N and D
- Analytical derivation via Lagrangian optimization:
  - Minimize L(N, D) = E + A/N^alpha + B/D^beta
  - Subject to: 6ND = C
  - Result: N* ∝ C^0.5, D* ∝ C^0.5
- The "20 tokens per parameter" rule
- Why Kaplan was wrong: undertrained models biased exponent estimates

### 5. Fitting Methodology
- Log-linear regression for power-law exponent estimation
- Parametric fit via L-BFGS: fit E, A, B, alpha, beta jointly
- Chinchilla's three cross-validation approaches
- Confidence and reliability of exponent estimates

### 6. References
Table with columns: Reference Name, Brief Summary, Link

## Exercise Structure

Following repo conventions (in order):
1. **Derivation problems** — re-derive compute-optimal scaling from Lagrangian; show the power-law loss decomposition implies specific exponent relationships
2. **Conceptual questions** — interpret alpha vs beta magnitudes; explain why L_inf matters; contrast Kaplan vs Chinchilla conclusions
3. **Implementation sketches** — pseudocode for IsoFLOP sweep; log-linear fitting procedure

## Decisions

- Single concept note (not split by paper) — the two papers tell one evolving story
- Intuition + derivations (not just results, not just formal proofs)
- Full three-tree output: notes + exercises + solutions
