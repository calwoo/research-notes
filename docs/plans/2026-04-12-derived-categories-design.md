# Design: Derived Categories Concept Note

**Date:** 2026-04-12
**Topic slug:** `derived-categories` (under `concepts/category-theory/`)
**Category:** `concepts`
**Multi-note:** yes

## Scope

This cluster covers derived categories and triangulated categories starting from the abstract algebraic theory and progressing to the geometric picture. The treatment is mathematically rigorous, favoring formal definitions, commutative diagrams, and complete proofs of key results over informal exposition. The target reader knows abelian categories, exact sequences, chain complexes, and basic homological algebra (Ext, Tor), but no algebraic geometry is assumed for the first three notes.

The cluster builds from the axiomatic theory of triangulated categories, through the construction of the derived category D(A) via localization of the homotopy category K(A) at quasi-isomorphisms, to derived functors (right/left derived via injective/projective resolutions), and finally to geometric applications (derived categories of sheaves, Grothendieck's six functors). t-structures, perverse sheaves, and the BBD decomposition theorem are deferred to a separate future note cluster.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/category-theory/derived-categories/overview.md` | Cluster index: notes table, subtopic map, dependency graph, master references |
| `concepts/category-theory/derived-categories/triangulated-categories.md` | Axioms of triangulated categories, distinguished triangles, octahedral axiom, examples (K(A)) |
| `concepts/category-theory/derived-categories/construction.md` | Chain complexes, homotopy category K(A), localization at quasi-isomorphisms, existence of D(A) |
| `concepts/category-theory/derived-categories/derived-functors.md` | Right/left derived functors via resolutions, composition (Grothendieck spectral sequence), examples |
| `concepts/category-theory/derived-categories/geometric.md` | D(X) for schemes/sheaves, six functors (f*, f*, f!, f!, ⊗, RHom), proper base change |

## Note Structure (first note: triangulated-categories.md)

1. **Motivation** — What homotopy theory forces on us; maps between complexes up to homotopy lose exactness in the naive sense; why we need a replacement for short exact sequences
2. **Additive and Pre-Triangulated Categories** — Additive categories, shift functor, cones; definition of a pre-triangulated category
3. **Triangulated Categories: Axioms (TR1–TR4)** — The four axioms stated precisely with the octahedral axiom in full; careful discussion of what TR4 says geometrically
4. **The Octahedral Axiom** — Detailed treatment, mnemonic diagrams, why it is independent from TR1–TR3
5. **Morphisms of Triangulated Categories** — Exact functors, natural transformations, equivalences
6. **The Homotopy Category K(A)** — Construction, proof it is triangulated, exact triangles = cone sequences
7. **Localization and the Verdier Quotient** — Gabriel-Zisman localization, multiplicative systems, Ore conditions, Verdier's quotient theorem
8. **Exercises** — Inline after each section

## Planned Subtopics (multi-note)

| File | Description |
|------|-------------|
| `triangulated-categories.md` | Axiomatic theory: TR1–TR4, homotopy category, Verdier quotient |
| `construction.md` | Explicit construction of D(A): injective/projective resolutions, boundedness conditions (D+, D−, Db) |
| `derived-functors.md` | RF and LF via resolutions, δ-functors, Grothendieck spectral sequence, examples (sheaf cohomology, Tor) |
| `geometric.md` | D(X) for ringed spaces/schemes, six functor formalism, proper base change, projection formula |

## References

- Huybrechts, "Fourier-Mukai Transforms in Algebraic Geometry" (arxiv.org/pdf/0704.1009 or the book)
- Caldararu or similar arxiv.org/pdf/math/0001045
- Merrick Cai lecture notes: merrickcai.com/pdfs_notes/Derived%20Categories.pdf
- Akhil Mathew blog post on BBD: amathew.wordpress.com/2011/06/23/trying-to-understand-bbd/
- Weibel, "An Introduction to Homological Algebra"
- Kashiwara–Schapira, "Sheaves on Manifolds"
- Gelfand–Manin, "Methods of Homological Algebra"
- Verdier, "Des catégories dérivées des catégories abéliennes" (original source)
