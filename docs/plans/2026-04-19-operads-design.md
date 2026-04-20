# Design: Operads Concept Notes

**Date:** 2026-04-19
**Topic slug:** `operads`
**Category:** `concepts/category-theory/operads`
**Multi-note:** yes

## Scope

Operads are the algebraic devices that encode families of multilinear operations with a compatible notion of composition. They unify the theories of associative, commutative, and Lie algebras under a single framework, and provide the correct language for homotopy-coherent algebraic structures ($A_\infty$, $L_\infty$), the bar/cobar duality machinery, and deformation theory. This note cluster covers operads from first principles through Koszul duality, with applications to the probability operad and entropy.

The treatment is grounded in Loday–Vallette *Algebraic Operads* (the canonical reference), supplemented by Markl *Operads and PROPs*, Fresse *Modules over Operads and Functors*, and Leinster *Entropy and Diversity* for the probabilistic application.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/category-theory/operads/overview.md` | Index, subtopic map, dependency graph, master references |
| `concepts/category-theory/operads/definitions.md` | First note: symmetric sequences, composition product, operad as monoid, key examples |
| `concepts/category-theory/operads/algebras-modules.md` | Algebras over an operad, free algebras, modules, enveloping algebra, derivations, Kähler differentials |
| `concepts/category-theory/operads/koszul-duality.md` | Bar/cobar constructions, twisting morphisms, Koszul criterion, O∞-algebras |

## Note Structure: definitions.md

1. **Symmetric Sequences** — collections $\{\mathcal{O}(n)\}_{n\geq 0}$ with $S_n$-actions; the category $\mathbf{SymSeq}$
2. **The Composition Product** — the substitution monoidal product $\mathcal{O} \circ \mathcal{P}$; the monoidal category $(\mathbf{SymSeq}, \circ, \mathbf{1})$; non-symmetry of $\circ$
3. **Operads as Monoids** — operad = monoid in $(\mathbf{SymSeq}, \circ)$; unit $\eta$, composition $\gamma$; equivalent partial-composition formulation ($\circ_i$ axioms: associativity, unit, equivariance)
4. **Morphisms and the Category Op** — operad morphisms; operads as full subcategory of monoids in SymSeq
5. **Key Examples** — $\mathrm{Ass}$, $\mathrm{Com}$, $\mathrm{Lie}$, $\mathrm{End}_V$; the maps $\mathrm{Lie} \to \mathrm{Ass} \to \mathrm{Com}$; topological operads and the little disks operad $\mathcal{D}_n$; the probability operad $\mathcal{P}$
6. **Colored Operads and Multicategories** — colored operads with multiple object types; small categories and symmetric monoidal categories as colored operads

## Planned Subtopics

| File | Description |
|------|-------------|
| `definitions.md` | Symmetric sequences, composition product, operad-as-monoid, examples |
| `algebras-modules.md` | Algebras over an operad, free algebras, modules, enveloping algebra, Kähler differentials, $A_\infty$-algebras |
| `koszul-duality.md` | Bar/cobar constructions, twisting morphisms, Koszul duality, $\mathcal{O}_\infty$-algebras from Koszul duality |

## References

- Loday & Vallette, *Algebraic Operads* (Springer, 2012) — primary reference, Ch. 1–5 for definitions
- Markl, *Operads and PROPs* (arXiv:math/0601129)
- Fresse, *Modules over Operads and Functors* (Springer LNM 1967)
- Leinster, *Entropy and Diversity* (arXiv:2012.02113) Ch. 2–3 — for the probability operad
