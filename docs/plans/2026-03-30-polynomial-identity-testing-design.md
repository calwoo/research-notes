# Design: Polynomial Identity Testing Concept Note

**Date:** 2026-03-30
**Topic slug:** `polynomial-identity-testing`
**Category:** `concepts`

## Scope

This note covers the randomized Polynomial Identity Testing (PIT) problem as introduced in Stanford CS 265 (Randomized Algorithms and Probabilistic Analysis), Lecture 1. The primary goal is to give a mathematically rigorous treatment of the Schwartz-Zippel lemma and its proof by induction on the number of variables, along with the surrounding algorithmic context (Las Vegas vs Monte Carlo, error amplification).

The note departs from the standard separate `exercises.md` format: exercises are embedded **inline** throughout the note, immediately following the material they test. This matches the interactive classroom style of CS 265.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/randomized-algorithms/polynomial-identity-testing.md` | Main note with inline exercises |

## Note Structure

1. **Introduction and Motivation** — Why PIT is a non-trivial problem; complexity of naively expanding monomials; applications (perfect matching, primality)
2. **Computational Model** — Formal definition of randomized algorithms; binary-tree view of computation; consequence that probabilities are integer multiples of $1/2^k$
3. **Las Vegas vs Monte Carlo** — Definitions; 1-sided vs 2-sided error; error amplification lemmas with proofs
4. **The PIT Problem** — Formal statement; why deterministic algorithms are hard
5. **Schwartz-Zippel Theorem** — Algorithm description; theorem statement; full proof by induction on number of variables
6. **Discussion and Open Problems** — Derandomization hardness (Kabanets-Impagliazzo); connection to circuit lower bounds

Inline exercises appear after each major section, with `[!TIP]-` solution callouts.

## References

- CS 265 Lecture 1 notes (Valiant/Wootters): https://web.stanford.edu/class/archive/cs/cs265/cs265.1212/Lectures/Lecture1/l1.pdf
- CS 265 Class 1 solutions: https://web.stanford.edu/class/archive/cs/cs265/cs265.1212/Lectures/Lecture1/class1-sols.pdf
- Kabanets & Impagliazzo (2004): Derandomizing polynomial identity tests means proving circuit lower bounds
- Raz & Shpilka (2005): Deterministic PIT in noncommutative models
