# Implementation Plan: Category Theory Concept Notes

**Date:** 2026-03-27
**Design doc:** `docs/plans/2026-03-27-category-theory-design.md`

## Tasks

1. **Create topic directory** `concepts/category-theory/`
2. **Write `01-categories-functors-natural-transformations.md`** — Sheet 1 content with exercises 1–14 embedded inline
3. **Write `02-adjoints-representables.md`** — Sheet 2 content with exercises 1–15 embedded inline
4. **Write `03-limits-colimits.md`** — Sheet 3 content with exercises 1–16 embedded inline
5. **Write `04-adjoint-functor-theorems-monads.md`** — Sheet 4 content with exercises 1–17 embedded inline
6. **Final cross-check** — TOC anchors correct, all exercises present in order, notation consistent across files
7. **Commit** with message `feat: add category-theory concept notes (Leinster Part III)`

## Exercise Sources

### Sheet 1 (14 exercises)
1. Three examples each of: categories, functors, natural transformations, adjunctions
2. Left+right inverse → isomorphism; functors preserve isos
3. Subcategories of a poset; subcategories of a group
4. Opposite category of a group G; G ≅ G^op; poset/monoid not iso to dual
5. Objects in Hty isomorphic to the one-element space
6. Center functor Z: Gp → Gp?
7. Natural transformations are natural isos iff each component is an iso
8. Pointed sets Set*; Set* ≅ 1/Set; Set* equivalent to Par
9. Mat_k equivalent to FDVect_k
10. Functors F: A × B → C from families of induced functors
11. Adjunctions; left adjoints preserve initial objects; right adjoints preserve terminal
12–13. Adjunctions between posets/powersets; unit/counit; triangle identities
14. Sym and Ord functors — pointwise iso but not naturally iso

### Sheet 2 (15 exercises)
1. Representable functors and representations
2. Five examples of representable functors
3. Yoneda embedding: injective on objects, full, faithful, characterizes representability
4. Full and faithful functors reflect isomorphisms
5. Topological spaces, presheaves, chains of adjoint functors
6. Adjunctions on functor categories [C,S]
7. Cayley embedding; small categories embed into Set
8. Adjunctions restrict to equivalences on fixed-point categories
9. Six conditions for idempotent adjunctions; application to preorders
10. Comma categories and characterizing adjoints via initial/terminal objects
11. Yoneda Lemma: statement, proof, three deductions
12–15. Advanced: functors on group actions, quantifiers as adjoints, mates, arrow categories

### Sheet 3 (16 exercises)
1. Define limit and colimit; uniqueness; three examples each
2. Limit cone and colimit cocone in Set
3. C(A, B×C) ≅ C(A,B) × C(A,C) naturally (direct proof from products)
4. Preservation, reflection, creation of limits; examples; F creates + D has → C has and F preserves
5. Projective objects; injective objects; regular injective; ℝ regular injective in NVS
6. k[X]* ≅ k[[X]] categorically
7. Regular monic, split monic (and duals); implications; examples in Ab, Top, Set
8. Terminal + binary products + equalizers → all finite limits; pullbacks → all finite limits; pushouts and colimits
9. Widget and Chad categories; forgetful functors create limits
10. C has all I-shaped colimits iff Δ: C → [I,C] has a left adjoint
11. Limits in [C^op, Set] computed pointwise; monics and epics
12. H_A preserves limits; Yoneda H^− preserves limits
13. Left adjoints preserve colimits; right adjoints preserve limits; Gp → Set has no right adjoint
14. Every presheaf is a colimit of representables; cartesian closed; Yoneda preserves products/exponentials
15. Closure under composition and pullback stability for six classes of morphisms
16. Optional extended project on homological algebra

### Sheet 4 (17 exercises)
1. Left adjoint to infima-preserving map on posets (explicit construction)
2. Monads: definition, adjunction-induced; monadic functors are faithful and reflect isos
3. Absolute coequalizer pairs; preservation/reflection/creation equivalences
4. T-algebra parallel pair forms reflexive split coequalizer
5. Set-adjoint ⟺ representable ⟺ limit-preserving (with coproduct/completeness hypotheses)
6. General Adjoint Functor Theorem: statement; initial objects; proof outline
7. Well-powered categories; Sub: C^op → Set
8. Special Adjoint Functor Theorem (assuming GAFT)
9. GAFT applied to Widget and Chad; Widget is monadic over Set
10. U_T creates limits; creates coequalizers for absolute pairs; every T-algebra is coequalizer of free algebras
11. Idempotent monads; every algebra is free
12. [C^op, Set] monadicity; creates limits; reflects isos; presheaves as colimits
13. Huge Monadicity Theorem: five equivalent conditions
14. Comparison functor K and when unit/counit are isos
15. Geometric realization via SAFT on simplex category Δ
16. Right adjoint to monad has comonad structure; coalgebras ≅ algebras
17. True/false: fields, posets, totally ordered sets, lattices, topological groups — monadic over Set or Top?
