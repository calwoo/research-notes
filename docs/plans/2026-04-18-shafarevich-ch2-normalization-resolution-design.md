# Design: Normalization and Resolution of Curve Singularities

**Date:** 2026-04-18
**Topic slug:** `shafarevich-ch2-normalization-resolution`
**Category:** `concepts/algebraic-geometry`
**Multi-note:** no

## Scope

This note covers Shafarevich *Basic Algebraic Geometry* Vol 1, §II.4–5: the algebraic and geometric approaches to resolving singularities of algebraic curves. It is the natural sequel to `shafarevich-ch2-local-properties.md`, which established what a singular point *is* (via the tangent space and local ring). This note answers: how do we *fix* singularities?

The algebraic approach (§II.4) passes to the normalization: replacing the coordinate ring $k[X]$ with its integral closure in $k(X)$ produces a birational finite morphism $\nu: \tilde{X} \to X$ from a normal (in dim 1: smooth) variety. The geometric approach (§II.5) blows up the singular point and takes the strict transform, which separates tangent directions and reduces multiplicity at each step. The resolution theorem asserts every curve over a perfect field has a unique smooth projective model, and the two approaches converge on it.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/algebraic-geometry/shafarevich-ch2-normalization-resolution.md` | Single note on normalization, blowing up, and resolution |

## Note Structure

1. **Introduction** — motivation: two ways to desingularize a curve; algebraic vs geometric
2. **Normal Varieties and Integral Closure**
   - Normal domain: integrally closed in its fraction field
   - Normal variety: local rings are normal domains
   - Equivalent conditions in dimension 1: normal = smooth for curves
   - Example: $k[t^2, t^3]$ vs its integral closure $k[t]$ for the cusp
3. **The Normalization**
   - Construction: $\tilde{X}$ corresponds to integral closure of $k[X]$ in $k(X)$
   - The normalization map $\nu: \tilde{X} \to X$: birational + finite morphism
   - Noether normalization (statement + use): every affine variety of dim $d$ admits a finite surjective map to $\mathbb{A}^d$
   - Normalization of the cuspidal cubic: $\mathbb{A}^1 \to V(y^2 - x^3)$, $t \mapsto (t^2, t^3)$
   - Normalization of the nodal cubic: two branches, normalization separates them
   - A curve is smooth iff it is normal (in dimension 1: regular = normal = integrally closed)
4. **Blowing Up**
   - Definition: $\mathrm{Bl}_P X \subset X \times \mathbb{P}^{n-1}$; universal property
   - The exceptional divisor $E \cong \mathbb{P}^{n-1}$; the blow-down map
   - Strict transform of a curve $C$ under blowing up at $P$
   - Multiplicity reduction: $\mathrm{mult}_{\tilde{P}} \tilde{C} < \mathrm{mult}_P C$
5. **Resolving Curve Singularities**
   - Resolution of the node $V(y^2 - x^2(x+1))$: one blow-up separates the two branches
   - Resolution of the cusp $V(y^2 - x^3)$ ($A_2$ singularity): requires two blow-ups; explicit charts
   - Termination: successive blow-ups eventually produce a smooth curve
6. **The Smooth Projective Model**
   - Existence: every curve over a perfect field has a smooth projective model
   - Uniqueness up to isomorphism
   - Equivalence with normalization for curves: the normalization of any projective model = smooth model
7. **Exercises** (inline after each section)
   - Mathematical Development: ~16–18 problems (integral closure computations, normalization of explicit curves, strict transform calculations, blow-up charts)
   - Algorithmic Applications: ~5–7 problems (explicit blow-up algorithms, multiplicity tracking)

## References

- Shafarevich, *Basic Algebraic Geometry* Vol 1, §II.4–5 (primary)
- Fulton, *Algebraic Curves*, Ch 7 (blowing up and resolution)
- Reid, *Undergraduate Algebraic Geometry*, Ch 7 §7.1–7.2 (geometric intuition)
- Atiyah–MacDonald, *Introduction to Commutative Algebra*, Ch 5 (integral dependence, Noether normalization)
- Eisenbud, *Commutative Algebra with a View Toward Algebraic Geometry*, Ch 4 (integral closure)
