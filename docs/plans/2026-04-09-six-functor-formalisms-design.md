# Design: Six Functor Formalisms Concept Note

**Date:** 2026-04-09
**Topic slug:** `six-functor-formalisms`
**Category:** `concepts/category-theory`

## Scope

This note introduces the theory of 6-functor formalisms, following Lecture I of Scholze's lecture notes (arXiv:2510.26269). The goal is a self-contained, graduate-level exposition that builds up from concrete geometric motivation — sheaves on locally compact Hausdorff spaces — and arrives at a precise categorical definition of what a 6-functor formalism is.

The note should be expository in character: it derives each functor from first principles, carefully motivates adjunctions and base change, and includes inline exercises at points where the reader should pause to verify small results or computations. The audience is a graduate student comfortable with homological algebra and derived categories, but not necessarily with ∞-categories.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/category-theory/six-functor-formalisms/note.md` | Main expository note |

## Note Structure

1. **Introduction and Motivation** — What do we want from "cohomology functors"? Why does a single derived pushforward fail to capture all the structure?
2. **The Derived Category of Sheaves** — Setup: locally compact Hausdorff spaces, abelian sheaves, derived categories D(X, Z). Brief reminder of what derived categories are.
3. **The First Two Functors: f* and f*** — Pullback/pushforward adjunction. Proper Base Change theorem. Interpretation of RΓ.
4. **The Next Two Functors: ⊗ and Hom** — Tensor product of sheaves, internal Hom, adjunction. Künneth formula and the Projection Formula for proper maps.
5. **The Exceptional Functors: f! and f!** — Proper pushforward with compact support, exceptional inverse image. General base change and projection formula without properness. Verdier duality (local and global forms).
6. **What Is a Six-Functor Formalism?** — Rough definition: category C, class E of morphisms, association X ↦ D(X), six functors, the key compatibilities (base change, projection formula, adjunctions). Mention Ayoub's early formalization and the difficulty of higher coherences.
7. **The Betti Cohomology Example** — The pro-étale algebraic space X_Betti and the equivalence D(X, Z) ≅ D_qc(X_Betti).
8. **References**

## Inline Exercise Plan

Exercises are placed inline as `> [!EXAMPLE]` or plain exercise blocks right after the relevant result:
- After f* ⊣ f*: Verify the unit/counit maps explicitly for X = pt
- After Proper Base Change: Specialize to a fiber square to recover the stalk formula
- After Künneth: Derive RΓ(X × Y, Z) ≅ RΓ(X, Z) ⊗ RΓ(Y, Z) for compact X, Y using Projection Formula
- After Projection Formula: Verify the adjointness map defining the projection formula morphism
- After Verdier Duality: Verify that for X compact oriented d-manifold, ωX/pt ≅ Z[d]

## References

- Peter Scholze, "Six-Functor Formalisms", arXiv:2510.26269 (primary source)
- Lurie, "Higher Algebra" (for ∞-category background)
- Kashiwara–Schapira, "Sheaves on Manifolds" (classical sheaf theory)
- Mann, "A p-adic 6-functor formalism in rigid analytic geometry" (Mann's definition)
