# Design: Neuroalgebraic Geometry Concept Note

**Date:** 2026-04-27
**Topic slug:** `neuroalgebraic-geometry`
**Category:** `concepts/algebraic-geometry`
**Multi-note:** yes

## Scope

Neuroalgebraic geometry studies neural networks and statistical learning through the lens of algebraic geometry and related fields (real algebraic geometry, algebraic statistics, commutative algebra). The core observation is that neural network parameter spaces and function spaces carry rich algebraic structure: parameter spaces are semi-algebraic sets, functional equivalence classes are orbits of algebraic group actions, and loss landscapes are algebraic hypersurfaces. This algebraic structure governs expressivity, generalization, optimization geometry, and Bayesian inference.

The topic spans several interconnected threads: (1) Watanabe's *singular learning theory*, which replaces the classical Fisher information framework with resolution of singularities to obtain sharp Bayesian asymptotics for singular models; (2) algebraic characterizations of neural network expressivity via polynomial maps, tensor decompositions, and circuit complexity; (3) the geometry of loss landscapes — critical points, saddle structure, and symmetry — through stratified Morse theory and algebraic topology; and (4) connections to algebraic statistics (graphical models, exponential families, maximum likelihood estimation on varieties).

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/algebraic-geometry/neuroalgebraic-geometry/overview.md` | Topic index, subtopic map, dependency graph, master references |
| `concepts/algebraic-geometry/neuroalgebraic-geometry/singular-learning-theory.md` | Watanabe's RLCT, resolution of singularities, Bayesian asymptotics |
| `concepts/algebraic-geometry/neuroalgebraic-geometry/expressivity-and-complexity.md` | Polynomial maps, tensor decompositions, algebraic circuit complexity |
| `concepts/algebraic-geometry/neuroalgebraic-geometry/loss-landscape-geometry.md` | Critical point structure, saddle topology, symmetry and quotient spaces |
| `concepts/algebraic-geometry/neuroalgebraic-geometry/algebraic-statistics-connections.md` | Graphical models, exponential families, MLE on varieties, identifiability |

## Note Structure (singular-learning-theory.md — first note)

1. **Introduction** — Why classical asymptotics fail for neural networks; singular models and the KL geometry
2. **Parameter space as an algebraic variety** — Realizing the model map as a polynomial/Nash map; fiber structure; symmetry groups
3. **Resolution of singularities** — Hironaka's theorem, normal crossings, the resolution map; real log canonical threshold (RLCT) as birational invariant
4. **Free energy and the RLCT** — Watanabe's asymptotic expansion of the Bayes free energy; λ (RLCT) replaces d/2 in the BIC; derivation of the main theorem
5. **Phase transitions and learning coefficients** — How λ changes with model architecture; examples for shallow networks, ReLU networks, matrix factorization
6. **Implications for generalization** — WBIC, widely applicable BIC; RLCT and model selection; connection to double descent
7. **Worked examples** — Explicit RLCT computations for a 1-hidden-layer tanh network and a rank-1 matrix factorization model
8. Inline exercises distributed after each section

## Planned Subtopics

| File | Description |
|------|-------------|
| `singular-learning-theory.md` | Watanabe's RLCT framework: resolution of singularities, Bayes free energy asymptotics, phase transitions |
| `expressivity-and-complexity.md` | Polynomial circuit complexity, VC dimension via Milnor-Thom, tensor rank and neural network depth |
| `loss-landscape-geometry.md` | Morse theory on loss surfaces, symmetry-induced degeneracies, saddle point structure |
| `algebraic-statistics-connections.md` | ML estimation on varieties, graphical models, identifiability in latent variable models |

## References

- Watanabe (2009) *Algebraic Geometry and Statistical Learning Theory* (Cambridge)
- Watanabe homepage: https://sites.google.com/view/sumiowatanabe/home
- arxiv 2501.18915 — recent survey on AG and deep learning
- arxiv 2211.10049 — TBD after fetch
- arxiv 2010.11560 — TBD after fetch
- arxiv 2406.10234 — TBD after fetch
- IPAM Workshop: Algebraic Geometry — A Window to Machine Learning: https://www.ipam.ucla.edu/programs/workshops/algebraic-geometry-a-window-to-machine-learning/
