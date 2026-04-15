# Design: Classical Algebraic Geometry Concept Note

**Date:** 2026-04-14
**Topic slug:** `classical-algebraic-geometry`
**Category:** `concepts`
**Multi-note:** no

## Scope

A single reference note following the arc of Harvard Math 137 (Brooke Ullery, 2020) and Fulton's *Algebraic Curves*. The goal is a fast-reference document listing the main definitions, theorems, and key results across all 24 lecture sections — not a textbook, not a set of exercises. Proofs are omitted unless the proof idea is itself geometrically illuminating (e.g., the Nullstellensatz proof via the Rabinowitsch trick, or the Bézout proof via intersection numbers). This is the kind of note you read before an exam or a qual to quickly reload the key facts.

The mathematical emphasis is on the algebraic ↔ geometric dictionary: every algebraic object (ideal, ring map, DVR, valuation) should be immediately paired with its geometric meaning (variety, morphism, smooth point on a curve, order of vanishing). Style follows the repo conventions: rigorous definitions, Obsidian callouts for key theorems, no hand-waving.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/classical-algebraic-geometry/note.md` | Single reference note covering all 24 Math 137 sections |

## Note Structure

1. **Affine Varieties** — algebraic sets, Zariski topology, Hilbert Basis Theorem, Nullstellensatz, radical ideals
2. **Coordinate Rings and Morphisms** — coordinate ring, regular functions, morphisms, algebra–geometry duality
3. **Rational Functions and Local Rings** — rational functions, stalks $\mathcal{O}_{X,P}$, maximal ideal, localization
4. **Affine Plane Curves** — tangent lines, multiplicity of a point, branches at a singularity
5. **Discrete Valuation Rings** — DVR definition, uniformizer, valuation $v_P$, smooth points on curves
6. **Intersection Numbers** — Fulton's local definition $(C \cdot D)_P = \dim_k \mathcal{O}_P/(f,g)$, properties, examples
7. **Projective Space and Varieties** — $\mathbb{P}^n$, homogeneous coordinates, standard affine cover, projective algebraic sets, homogeneous coordinate ring
8. **Morphisms of Projective Varieties** — regular maps, Veronese, Segre, quasiprojective varieties
9. **Projective Plane Curves** — degree, genus formula (preview), tangent lines in projective coordinates
10. **Linear Systems** — complete linear system $|D|$, base locus, the map $\phi_D$
11. **Bézout's Theorem** — statement, proof sketch via intersection numbers, applications
12. **Abstract Varieties** — gluing construction, function field, examples ($\mathbb{P}^n$ as abstract variety)
13. **Rational Maps and Dimension** — rational maps, birational equivalence, transcendence degree, fiber dimension theorem
14. **Blowing Up** — blow-up of a point in $\mathbb{A}^2$, exceptional divisor, strict transform, resolution of singularities

## References

- Fulton, *Algebraic Curves* (free PDF) — primary source for Math 137
- Ullery, Math 137 lecture notes (24 sections, Harvard 2020)
- Shafarevich, *Basic Algebraic Geometry* Vol 1 — for additional depth on affine/projective geometry
- Reid, *Undergraduate Algebraic Geometry* — for geometric intuition and pictures
