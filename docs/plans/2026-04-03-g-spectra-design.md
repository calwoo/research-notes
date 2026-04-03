# Design: Genuine G-Spectra Concept Note

**Date:** 2026-04-03
**Topic slug:** `g-spectra`
**Category:** `concepts/equivariant-stable-homotopy`

## Scope

This note develops genuine $G$-spectra — the stable equivariant homotopy category — starting from the unstable foundations already laid in `g-spaces-and-equivariant-maps.md` and the algebraic structures in `mackey-functors.md`. The central insight is that naive stabilization (inverting $S^1$) and genuine stabilization (inverting all representation spheres $S^V$) yield genuinely different categories with different formal properties, and that the genuine theory is the one that supports a rich stable homotopy theory.

The note covers: the equivariant stable category (naive and genuine, and the precise sense in which they differ); $G$-universes and the indexing problem; orthogonal $G$-spectra as the modern model; the stable model structures; the smash product and closed symmetric monoidal structure; suspension spectra $\Sigma^\infty_+ X$; and the key structural theorem that homotopy groups of genuine $G$-spectra are Mackey functors. The note ends with change-of-universe and the cofiber/fiber sequence machinery needed for later notes.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/equivariant-stable-homotopy/g-spectra.md` | Main research note |

## Note Structure

1. **Stabilization and the Need for Genuine Spectra** — review naive stabilization; representation spheres $S^V$; why inverting only $S^1$ gives an inadequate theory; what genuine G-spectra fix (RO(G)-graded homotopy groups, Wirthmuller isomorphism, etc.)

2. **G-Universes and Indexing** — definition of a $G$-universe $\mathcal{U}$; complete vs. incomplete universes; the trivial universe (naive) vs. complete universe (genuine); inner product spaces, $\mathcal{L}(\mathcal{U}, \mathcal{U}')$ spaces of isometric embeddings; why the indexing matters for smash products

3. **Lewis–May–Steinberger G-Spectra** — classical definition: a $G$-prespectrum indexed on finite-dimensional sub-inner-product-spaces $V \subset \mathcal{U}$; the structure maps $\Sigma^{W-V} E_V \to E_W$; the $\Omega$-spectrum condition; the stabilization adjunction

4. **Orthogonal G-Spectra** — modern definition via orthogonal spectra with $G$-action; the indexing category $\mathbf{I}_G$ (finite-dimensional real inner product spaces with $G$-action); the Day convolution smash product; comparison with LMS spectra; why orthogonal spectra are the preferred model

5. **Naive vs. Genuine: The Two Model Structures** — the naive model structure (weak equivalences are $\pi_*^e$-isos); the genuine model structure (weak equivalences are $\pi_*^H$-isos for all $H \leq G$); these give genuinely different homotopy categories — not just different presentations of the same thing; the forgetful functor and its adjoints

6. **Homotopy Groups as Mackey Functors** — definition of $\underline{\pi}_n(E)$ as a Mackey functor: $\underline{\pi}_n(E)(G/H) = \pi_n(E^H)$; restriction maps from restriction of group; transfer maps from the stable transfer; the Mackey axiom from the double coset formula in stable homotopy; why this fails for naive spectra

7. **Suspension Spectra and Generators** — $\Sigma^\infty_+ X$ for a $G$-space $X$; the counit of $(\Sigma^\infty_+, \Omega^\infty)$; the orbit and fixed-point functors on spectra ($\tilde{E}G \wedge -$, $(-)^G$, $(-)^{hG}$) previewed here; the sphere spectrum $\mathbb{S}$ and representation spheres $S^V$

8. **The Smash Product and Closed Structure** — the symmetric monoidal structure $(\mathrm{Sp}^G, \wedge, \mathbb{S})$; the internal Hom functor $F(-, -)$; the suspension isomorphism $\Sigma^V E \simeq S^V \wedge E$; equivariant Spanier-Whitehead duality

9. **Change of Universe and Change of Group** — restriction $i^*: \mathrm{Sp}^G(\mathcal{U}) \to \mathrm{Sp}^G(\mathcal{U}')$ and its adjoints; change of group $i^*: \mathrm{Sp}^G \to \mathrm{Sp}^H$ for $H \leq G$; the geometric fixed-point functor $\Phi^H$ previewed; Lewis's theorem that naive and genuine are inequivalent

10. **References**

## References

- Blumberg 2017 (Debray notes) §2: https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf
- Lewis, May, Steinberger 1986: Springer LNM 1213 (classical reference for LMS spectra)
- Schwede 2020 lectures: https://www.math.uni-bonn.de/~schwede/equivariant.pdf
- Malkiewich draft (344pp): https://people.math.binghamton.edu/malkiewich/G_spectra.pdf
- Adams 1984 prerequisites: https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/prerequisites_for_carlsson.pdf
- Greenlees–May 1995: https://www.math.uchicago.edu/~may/PAPERS/Newthird.pdf
- May et al. Alaska notes §XIII: https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf
