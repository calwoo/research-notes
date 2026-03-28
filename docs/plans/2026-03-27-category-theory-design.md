# Design: Category Theory Concept Note

**Date:** 2026-03-27
**Topic slug:** `category-theory`
**Category:** `concepts`

## Scope

This note series covers the full content of Tom Leinster's Part III Category Theory course (Cambridge, Michaelmas 2000), structured around the four exercise sheets at https://webhomes.maths.ed.ac.uk/~tl/categories/index.html. The treatment emphasizes rigorous mathematical definitions, formal notation, and complete derivations. The primary goal is to serve as a self-study guide where each exercise appears (in order) in a callout box, preceded by enough exposition to solve it.

The mathematical arc follows the course's functors-first pedagogy: categories and functors → natural transformations → adjoints → representables and Yoneda → limits/colimits → adjoint functor theorems → monads. This ordering reflects the fact that most constructions (limits, colimits, free objects) are instances of adjunctions.

## Files to Create

| File | Purpose | Source |
|------|---------|--------|
| `concepts/category-theory/01-categories-functors-natural-transformations.md` | Definitions, examples, equivalence; Sheet 1 exercises 1–14 | Sheet 1 |
| `concepts/category-theory/02-adjoints-representables.md` | Adjunctions in depth, Yoneda lemma, representable functors; Sheet 2 exercises 1–15 | Sheet 2 |
| `concepts/category-theory/03-limits-colimits.md` | Limits, colimits, preservation/reflection/creation; Sheet 3 exercises 1–16 | Sheet 3 |
| `concepts/category-theory/04-adjoint-functor-theorems-monads.md` | GAFT, SAFT, monads, Eilenberg-Moore, monadicity; Sheet 4 exercises 1–17 | Sheet 4 |

## Note Structure (per file)

### File 1: Categories, Functors, Natural Transformations
1. Categories — definition, examples (Set, Grp, Top, posets, monoids, homotopy category)
2. Functors — definition, forgetful/free/hom-functors, composition
3. Special morphisms — monics, epics, isomorphisms, initial/terminal objects
4. Subcategories, opposite categories
5. Natural transformations — definition, components, naturality squares
6. Natural isomorphisms and equivalence of categories
7. Products of categories
8. Adjoints — first encounter, unit/counit, triangle identities

### File 2: Adjoints and Representables
1. Adjunctions in depth — four equivalent definitions
2. Comma categories
3. Representable functors — definition and examples
4. The Yoneda embedding
5. The Yoneda Lemma — statement and proof
6. Corollaries of Yoneda
7. Idempotent adjunctions and reflections

### File 3: Limits and Colimits
1. Diagrams and cones
2. Limits — definition, uniqueness, examples
3. Colimits — definition, examples
4. Limits in Set
5. Preservation, reflection, and creation of limits
6. Projective and injective objects
7. Building limits from products and equalizers
8. Limits in functor categories
9. Adjoint preservation of limits/colimits
10. The density theorem (every presheaf is a colimit of representables)

### File 4: Adjoint Functor Theorems and Monads
1. Motivation — when does a functor have an adjoint?
2. The General Adjoint Functor Theorem (GAFT)
3. The Special Adjoint Functor Theorem (SAFT)
4. Applications of GAFT/SAFT
5. Monads — definition, adjunction-induced monads
6. Eilenberg-Moore algebras
7. The Kleisli category
8. The Beck Monadicity Theorem
9. Examples — monadic categories over Set

## Exercise Format

Each exercise appears in an Obsidian callout:

```
> [!NOTE] Exercise N
> (exercise text)
```

Exercises are embedded inline, immediately after the exposition that establishes the prerequisites for that exercise.

## References

- Tom Leinster, Part III Category Theory course notes and exercise sheets (Cambridge, 2000): https://webhomes.maths.ed.ac.uk/~tl/categories/index.html
- Mac Lane, S. *Categories for the Working Mathematician* (Springer, 1971)
- Borceux, F. *Handbook of Categorical Algebra* (Cambridge UP, 1994)
- McLarty, C. *Elementary Categories, Elementary Toposes* (Oxford UP, 1992)
- Lawvere & Schanuel, *Conceptual Mathematics* (Cambridge UP, 1997)
