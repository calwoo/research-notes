# Design: Mackey Functors Concept Note

**Date:** 2026-04-03
**Topic slug:** `mackey-functors`
**Category:** `concepts/equivariant-stable-homotopy`

## Scope

This note covers Mackey functors — the correct coefficient objects for equivariant cohomology theories — from both the algebraic and categorical perspectives. It sits immediately after `g-spaces-and-equivariant-maps.md` in the dependency graph and is a prerequisite for `g-spectra.md`, where Mackey functors appear as the homotopy groups $\pi_*^G$ of genuine $G$-spectra.

The note will develop: coefficient systems and why they are insufficient; the definition of a Mackey functor via the Lindner/Dress double-functor formulation and via the box product; the Burnside category $\mathcal{A}(G)$ (both the classical and $\infty$-categorical Barwick versions); key algebraic structure (Green functors, Tambara functors); and worked examples ($\underline{\mathbb{Z}}$, $\underline{A}$, $RO(G)$-related examples for $G = C_2$). The $\infty$-categorical perspective (spectral Mackey functors as excisive functors on the effective Burnside $\infty$-category) will be treated in a final section.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/equivariant-stable-homotopy/mackey-functors.md` | Main research note |

## Note Structure

1. **From Coefficient Systems to Mackey Functors** — Recall coefficient systems (contravariant functors $\mathcal{O}_G^{op} \to \mathbf{Ab}$); explain why the transfer maps for Bredon cohomology force covariant functoriality too; motivate Mackey functors as functors on the span category.

2. **The Burnside Category** — Define the *Burnside category* $\mathcal{A}(G)$: objects are finite $G$-sets, morphisms are spans (correspondences) of finite $G$-sets modulo isomorphism. Composition via fiber product. The enriched version with $\mathrm{Hom}$-groups given by the Grothendieck group of $G$-sets over a product.

3. **Mackey Functors: Formal Definition** — Define a Mackey functor as an additive functor $\mathcal{A}(G) \to \mathbf{Ab}$ (Lindner's theorem). Unpack into the double-functor data: restriction maps $\mathrm{res}_H^K$, transfer (induction) maps $\mathrm{tr}_H^K$, and conjugation maps $c_g$, together with the Mackey double coset formula.

4. **The Mackey Double Coset Formula** — Derive the Mackey formula $\mathrm{res}_K^G \circ \mathrm{tr}_H^G = \sum_{[g] \in K \backslash G/H} \mathrm{tr}_{K \cap {}^gH}^K \circ c_g \circ \mathrm{res}_{g^{-1}Kg \cap H}^H$ from the pullback of spans.

5. **Key Examples** — Constant Mackey functor $\underline{A}$; Burnside ring Mackey functor $\underline{A}(G)$; fixed-point Mackey functor $\underline{\pi}_n(X)$ for a $G$-space $X$; representation ring $\underline{R}(G)$; $C_2$-examples worked out explicitly.

6. **The Box Product and Closed Structure** — The box product $M \mathbin{\square} N$ of Mackey functors (Day convolution on spans); Green functors as commutative monoids; Tambara functors (incorporating norms); $\mathrm{RO}(G)$-graded Mackey functors.

7. **Projective Mackey Functors and Resolutions** — Projective generators; the Mackey functor $M_H = A(G, H) = \mathbb{Z}[G/H, -]$; relationship to the Burnside ring; global dimension and resolutions.

8. **Spectral Mackey Functors: Barwick's Theorem** — The effective Burnside $\infty$-category $\mathcal{A}_\infty(G)$; spectral Mackey functors as excisive functors $\mathcal{A}_\infty(G) \to \mathbf{Sp}$; Barwick's theorem that genuine $G$-spectra $\simeq$ spectral Mackey functors; homotopy groups as classical Mackey functors.

9. **References**

## Exercise Structure (if later requested)

1. **Derivation problems** — prove the Mackey formula from the pullback of spans; classify Mackey functors for $G = C_2$; compute the box product $\underline{\mathbb{Z}} \mathbin{\square} \underline{\mathbb{Z}}$; Green functor structure on $\underline{A}(G)$; Tambara polynomial functors.
2. **Algorithmic applications** — compute $\mathrm{res}$/$\mathrm{tr}$ tables for $G = C_p$ and $G = C_{p^2}$; write pseudocode for the Burnside ring multiplication table; compute projective resolutions for small $G$.

## References

- Webb 2000: "A Guide to Mackey Functors" — https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/WebbMF.pdf
- Blumberg 2017 (notes by Debray), §3: https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf
- Barwick 2014: "Spectral Mackey Functors and Equivariant Algebraic K-Theory" — https://arxiv.org/abs/1404.0108
- Dress 1973: "Contributions to the Theory of Induced Representations" (Batelle conference proceedings)
- May et al. 1996 Alaska notes, §IX–X: https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf
- Thévenaz–Webb 1995: "The structure of Mackey functors" (TAMS)
