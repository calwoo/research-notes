# Design: Symmetric Monoidal Categories Concept Note

**Date:** 2026-04-19
**Topic slug:** `symmetric-monoidal-categories`
**Category:** `concepts/category-theory`
**Multi-note:** no

## Scope

This note introduces symmetric monoidal categories from the ground up, assuming familiarity with categories, functors, and natural transformations. The treatment follows Riehl's *Category Theory in Context* with supplementary material from Mac Lane and nLab.

The note covers the full definition (tensor product, associator, unitors, braiding, symmetry), the coherence theorem, monoidal functors, and the key examples that appear throughout algebra, topology, and homotopy theory.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/category-theory/symmetric-monoidal-categories/note.md` | Single expository note with inline exercises |

## Note Structure

1. **Monoidal Categories** — bifunctor ⊗, associator α, left/right unitors λ, η; pentagon and triangle axioms; statement of Mac Lane's coherence theorem
2. **The Coherence Theorem** — strictification; why every monoidal category is monoidally equivalent to a strict one
3. **Examples** — (Set, ×, 1), (Ab, ⊗, ℤ), (Vect_k, ⊗, k), (Ch(R), ⊗, R), (Top, ×, *), (End(C), ∘, id)
4. **Braided and Symmetric Monoidal Categories** — braiding β, hexagon axioms; symmetry σ²=id; braided vs symmetric distinction
5. **Monoidal Functors** — lax/strong/strict monoidal functors; monoidal natural transformations; examples (forgetful functors, free constructions)
6. **Algebras in a Monoidal Category** — monoids, commutative monoids; connection to operads; algebras over commutative rings as a special case
7. **Duals and Compact Closed Categories** — left/right duals, pivotal and compact closed structure; trace; connection to topological field theory

## References

- Emily Riehl, *Category Theory in Context* (primary)
- Saunders Mac Lane, *Categories for the Working Mathematician*, Ch. VII
- nLab: symmetric monoidal category, braided monoidal category, coherence theorem
- Etingof et al., *Tensor Categories* (for algebras in monoidal categories)
- Baez, lecture notes: https://math.ucr.edu/home/baez/qg-winter2001/definitions.pdf
- Baez & Stay, *Physics, Topology, Logic and Computation: a Rosetta Stone*: https://math.ucr.edu/home/baez/rosetta/rosetta_topos_web.pdf
- Baez, Rosetta Stone blog: https://johncarlosbaez.wordpress.com/2021/05/28/symmetric-monoidal-categories-a-rosetta-stone/
- Baez, Cartesian to SMC blog (2024): https://golem.ph.utexas.edu/category/2024/02/from_cartesian_to_symmetric_mo.html
