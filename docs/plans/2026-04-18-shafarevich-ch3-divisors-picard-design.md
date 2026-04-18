# Design: Divisors and the Picard Group

**Date:** 2026-04-18
**Topic slug:** `shafarevich-ch3-divisors-picard`
**Category:** `concepts/algebraic-geometry`
**Multi-note:** no

## Scope

This note covers the divisor theory of smooth projective curves and the associated Picard group, corresponding to Weeks 11–12 of the Phase I curriculum (Shafarevich §III.1). It picks up where `shafarevich-ch2-local-properties.md` left off — DVRs and valuations are used immediately to define the order of vanishing of a rational function at a smooth point — and sets the stage for the Riemann-Roch theorem (Week 14).

The note develops two parallel threads. The *additive thread* (Weil divisors): formal integer-linear combinations of codimension-1 subvarieties, principal divisors from rational functions, the divisor class group $\mathrm{Pic}(X) = \mathrm{Div}(X)/\mathrm{PDiv}(X)$, and explicit computations of $\mathrm{Pic}(\mathbb{P}^1)$ and $\mathrm{Pic}(\mathbb{A}^1)$. The *linear-algebra thread* (linear systems): the complete linear system $|D|$ as a projective space of effective divisors, the base locus, and the map $\phi_D: X \dashrightarrow \mathbb{P}(H^0(X, \mathcal{O}(D)))^\vee$ to projective space; base-point-free, very ample, and ample divisors; illustrative examples via $\mathcal{O}(n)$ on $\mathbb{P}^1$ and the Veronese embedding.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/algebraic-geometry/shafarevich-ch3-divisors-picard.md` | Single note on divisors, Pic, and linear systems |

## Note Structure

1. **Introduction** — motivation: keeping track of zeros and poles of rational functions; divisors encode this data globally
2. **Weil Divisors on Curves**
   - Smooth projective curve $X$; every point $P$ gives a DVR $(\mathcal{O}_{X,P}, v_P)$
   - Divisor $D = \sum_{P \in X} n_P[P]$ with finite support; $\mathrm{Div}(X)$ is a free abelian group
   - Degree map $\deg: \mathrm{Div}(X) \to \mathbb{Z}$; effective divisors $D \geq 0$
3. **Rational Functions and Principal Divisors**
   - For $f \in k(X)^\times$: $\mathrm{div}(f) = \sum_P v_P(f)[P]$ (zeros minus poles)
   - Key theorem: $\deg \mathrm{div}(f) = 0$ for any $f \in k(X)^\times$ on a projective curve
   - $\mathrm{PDiv}(X) \subset \mathrm{Div}^0(X)$; product formula
   - Worked example: $\mathrm{div}((x-a)/(x-b))$ on $\mathbb{P}^1$
4. **The Picard Group**
   - Linear equivalence: $D \sim D'$ iff $D - D' = \mathrm{div}(f)$ for some $f$
   - $\mathrm{Pic}(X) = \mathrm{Div}(X)/\mathrm{PDiv}(X)$; $\mathrm{Pic}^0(X) = \ker(\deg)/ \mathrm{PDiv}(X)$
   - Splitting: $\mathrm{Pic}(X) \cong \mathbb{Z} \oplus \mathrm{Pic}^0(X)$ for connected projective curves
   - $\mathrm{Pic}(\mathbb{P}^1) \cong \mathbb{Z}$: every degree-0 divisor on $\mathbb{P}^1$ is principal
   - $\mathrm{Pic}(\mathbb{A}^1) = 0$: every divisor on $\mathbb{A}^1$ is principal
   - Preview: $\mathrm{Pic}^0(E) \cong E$ for an elliptic curve (group law via Riemann-Roch)
5. **Line Bundles and the Divisor–Line Bundle Correspondence**
   - $\mathcal{O}(D)$: the line bundle (rank-1 locally free sheaf) associated to $D$
   - Sections $H^0(X, \mathcal{O}(D))$: rational functions $f$ with $\mathrm{div}(f) + D \geq 0$
   - Isomorphism of groups: $\mathrm{Pic}(X) \cong \{\text{line bundles on } X\}/\text{iso}$
   - $\mathcal{O}(n)$ on $\mathbb{P}^1$: sections are homogeneous polynomials of degree $n$; $h^0 = n+1$
6. **Linear Systems and Maps to Projective Space**
   - Complete linear system $|D| = \mathbb{P}(H^0(X, \mathcal{O}(D)))$: projective space of effective divisors $\sim D$
   - Base locus $\mathrm{Bs}(|D|) = \bigcap_{D' \in |D|} \mathrm{supp}(D')$
   - The rational map $\phi_D: X \dashrightarrow \mathbb{P}^{h^0(D)-1}$; it is a morphism iff $\mathrm{Bs}(|D|) = \emptyset$
   - Very ample: $\phi_D$ is a closed immersion; ample: some multiple is very ample
   - Example: $\mathcal{O}(1)$ on $\mathbb{P}^1$ gives identity; $\mathcal{O}(n)$ gives degree-$n$ Veronese $\mathbb{P}^1 \hookrightarrow \mathbb{P}^n$
   - Example: $|2[P]|$ on a smooth conic
7. **Exercises** (inline throughout, ~16–18 mathematical + 5–7 algorithmic)

## References

- Shafarevich, *Basic Algebraic Geometry* Vol 1, §III.1 (primary)
- Fulton, *Algebraic Curves*, Ch 8 (divisors and linear series)
- Hartshorne, *Algebraic Geometry*, §II.6–7 (Weil and Cartier divisors, line bundles)
- Reid, *Undergraduate Algebraic Geometry*, Ch 9 §9.1–9.3
- Silverman, *Arithmetic of Elliptic Curves*, Ch II (Pic^0 of an elliptic curve)
