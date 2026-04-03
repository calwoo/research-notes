# Implementation Plan: Mackey Functors Note

**Date:** 2026-04-03
**Design doc:** `docs/plans/2026-04-03-mackey-functors-design.md`
**Output:** `concepts/equivariant-stable-homotopy/mackey-functors.md`

## Tasks

1. **Research sources** — fetch Webb 2000, Blumberg §3, Barwick 2014 abstract/intro, Alaska §IX–X; extract key definitions and theorems to use as anchors.
2. **Write §1: From Coefficient Systems to Mackey Functors** — motivation from Bredon cohomology; why covariance + contravariance forces span category.
3. **Write §2: The Burnside Category** — objects, span morphisms, fiber-product composition, enrichment over $\mathbf{Ab}$.
4. **Write §3: Mackey Functors Formal Definition** — additive functor on $\mathcal{A}(G)$; unpack into Lindner data; double coset formula stated here.
5. **Write §4: The Mackey Double Coset Formula** — derive from pullback of spans with diagram.
6. **Write §5: Key Examples** — $\underline{\mathbb{Z}}$, $\underline{A}(G)$, fixed-point Mackey functor, $C_2$-examples with explicit tables.
7. **Write §6: Box Product and Green/Tambara Functors** — Day convolution; monoid = Green functor; Tambara norms.
8. **Write §7: Projective Mackey Functors** — generators $M_H$, global dimension, resolutions.
9. **Write §8: Spectral Mackey Functors** — effective Burnside $\infty$-category; Barwick's equivalence; connection to genuine $G$-spectra.
10. **Write References table** — full table with links from the design doc.
11. **Review** — verify TOC wikilinks, notation consistency ($\mathrm{res}$/$\mathrm{tr}$/$c_g$ uniform throughout), callouts present in each section, no LaTeX in headings.
12. **Fetch figures** — run image-extractor agent on cited arXiv sources.
13. **Commit** — `git commit -m "feat: add mackey-functors concept note"`.
