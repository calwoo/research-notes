# Algebraic Geometry Curriculum Design
*Date: 2026-04-13*

## Overview

A self-study curriculum for algebraic geometry and arithmetic geometry, starting from undergraduate algebra (groups, rings, modules) and culminating at Berkeley/Harvard PhD qualifying exam level. The focus throughout is on understanding the **intuition between algebraic constructions and their geometric avatars** — every definition should be paired with a geometric picture.

### Target Outcomes

- Understand and work through problems from:
  - Ritvik Ramkumar's Berkeley qual syllabus (Hartshorne + Eisenbud, 2017)
  - Will Fisher's Berkeley qual transcript (Hartshorne + category theory, 2024)
  - Harvard graduate AG qualifying problem collection
- Comfortable with: scheme theory, sheaf cohomology, Riemann-Roch, Hurwitz, elliptic curves over Q/F_q, Weil conjectures (statement)

### Profile

| Dimension | Value |
|---|---|
| Starting background | Undergraduate algebra (Dummit & Foote level) |
| Time commitment | 5–8 hrs/week (~6.5 avg) |
| Target flavor | Classical Hartshorne + Eisenbud program |
| Arithmetic depth | Moderate (elliptic curves over Q, curves over F_q, Weil conjectures at statement level) |
| Learning style | Interleaved reading + exercises |

### Curriculum Structure

| Phase | Theme | Duration |
|---|---|---|
| 1 | Classical Varieties (Shafarevich Vol 1) | ~20 weeks |
| 2 | Scheme Theory (Hartshorne I-II + Mumford) | ~22 weeks |
| 3 | Cohomology and Curves (Hartshorne III-IV) | ~22 weeks |
| 4 | Arithmetic Geometry (Silverman + Milne) | ~18 weeks |
| **Total** | | **~82 weeks (~1.6 years)** |

**Note on commutative algebra:** No dedicated phase. The required CA (localization, primary decomposition, Noether normalization, Krull dimension, Serre's criterion, Artin-Rees, Cohen-Macaulay) is woven in at point-of-need throughout Phases 1–3, with A&M chapter references given inline. The student is expected to self-study the relevant A&M sections as they arise.

---

## Phase 1: Classical Varieties (~20 weeks)

### Texts

| Role | Text |
|---|---|
| Primary | Shafarevich, *Basic Algebraic Geometry* Vol 1 (*Varieties in Projective Space*) |
| Intuition companion | Reid, *Undergraduate Algebraic Geometry* (skim for pictures/intuition) |
| Problem supplement | Gathmann, *Algebraic Geometry* lecture notes (free) |

### Goal

Build a **geometric dictionary** before scheme theory arrives. Every concept introduced here will have a scheme-theoretic analogue in Phase 2 — Phase 1 is about understanding what those analogues are supposed to mean geometrically.

Key objects to internalize:
- Affine/projective varieties and their defining ideals
- Morphisms, rational maps, birational equivalence
- Local ring at a point $\mathcal{O}_{X,P}$ — this is the stalk of the structure sheaf before you know what a sheaf is
- Smoothness via the Jacobian criterion
- Function field $k(X)$ — the algebraic avatar of "generic geometry"
- Divisors on curves and the classical Riemann-Roch theorem

### Week-by-Week Plan

| Week | Topic | Shafarevich § | CA prereqs (A&M ch) | Key problems |
|---|---|---|---|---|
| 1–2 | Affine varieties, regular functions, morphisms, $\mathbb{A}^n$ | I.1–I.2 | Ch 1–2 (rings, ideals) | I.1 #1–7 |
| 3–4 | Zariski topology; irreducibility; function field $k(X)$ | I.2–I.3 | Ch 3 (localization) | I.2 #1–6 |
| 5–6 | Projective varieties; homogeneous coordinates; $\mathbb{P}^n$ | I.4 | Ch 1 (graded rings) | I.4 #1–8 |
| 7–8 | Quasiprojective varieties; products; Segre embedding | I.5–I.6 | Ch 2 (modules) | I.5 #1–5 |
| 9–10 | Regular and rational maps; birational equivalence | I.3, I.6 | Ch 3 | I.6 #1–6 |
| 11–12 | Local ring at a point; tangent space; smoothness | II.1–II.2 | Ch 3, 11 (DVRs) | II.1 #1–6 |
| 13–14 | Local structure of morphisms; ramification; fiber dimension | II.3 | Ch 9 (dimension) | II.3 #1–5 |
| 15–16 | Normalization; resolution of singularities on curves | II.4–II.5 | Ch 5 (integral ext), 9 | II.5 #1–4 |
| 17–18 | Divisors on curves; principal divisors; divisor class group | III.1 | Ch 9 (Dedekind domains) | III.1 #1–6 |
| 19–20 | Differential forms; geometric genus; Riemann-Roch (classical) | III.3–III.4 | — | III.3 #1–5 |

### Key Geometric Intuitions to Extract

1. **Local ring = stalk.** $\mathcal{O}_{X,P}$ is the ring of rational functions on $X$ that are defined near $P$. This is exactly the stalk of the structure sheaf — before you know sheaves exist.
2. **Function field = generic point.** $k(X)$ is the "value at the generic point." Scheme theory makes this precise by adding non-closed points to the space.
3. **Zariski topology is coarser than you think.** Dense open subsets are enormous; this is why algebraic geometry can't use classical topology naively.
4. **Divisors on curves.** A divisor is a formal sum of points; the Riemann-Roch theorem says that the "expected" number of independent sections of $\mathcal{O}(D)$ is $\deg D + 1 - g$, corrected by the genus. Every later cohomological statement is a generalization of this.

---

## Phase 2: Scheme Theory (~22 weeks)

### Texts

| Role | Text |
|---|---|
| Primary | Hartshorne, *Algebraic Geometry*, Chapters I–II |
| Geometric companion | Mumford, *The Red Book of Varieties and Schemes*, Ch I–II |
| Supplementary reference | Vakil, *The Rising Sea* (free; use when Hartshorne is too terse) |

### Goal

**Translation:** Every concept from Phase 1 has a scheme-theoretic analogue. This phase is structured around that dictionary. The key conceptual leap is that points are now prime ideals (not just maximal ideals) — this is what lets schemes unify geometry over any ring.

### The Translation Dictionary

| Classical (Phase 1) | Scheme-theoretic (Phase 2) |
|---|---|
| Affine variety $V(I) \subset \mathbb{A}^n$ | Affine scheme $\text{Spec}(A)$ |
| Closed points $\mathfrak{m} \in \text{MaxSpec}(A)$ | All primes $\mathfrak{p} \in \text{Spec}(A)$ |
| Regular function on $U$ | Section $\mathcal{O}_X(U)$ |
| Local ring $\mathcal{O}_{X,P}$ | Stalk $\mathcal{O}_{X,\mathfrak{p}}$ |
| Morphism of varieties | Morphism of locally ringed spaces |
| Projective variety $V_+(I) \subset \mathbb{P}^n$ | $\text{Proj}(S)$ for graded ring $S$ |
| Line bundle on $X$ | Invertible sheaf $\mathcal{L} \in \text{Pic}(X)$ |
| Divisor class group | $\text{Pic}(X) \cong H^1(X, \mathcal{O}_X^*)$ |
| Cotangent space at $P$ | Stalk $\Omega_{X/k,P}$ of sheaf of Kähler differentials |

### Week-by-Week Plan

| Week | Topic | Source |
|---|---|---|
| 1–2 | Spec $A$; Zariski topology; structure sheaf $\mathcal{O}_X$ | Hartshorne II.1–2, Mumford I.1 |
| 3–4 | Sheaves: presheaves, sheafification, stalks, exactness | Hartshorne II.1, Vakil Ch 2 |
| 5–6 | Locally ringed spaces; definition of schemes; first examples | Hartshorne II.2, Mumford I.2 |
| 7–8 | Morphisms of schemes; open/closed subschemes; affine morphisms | Hartshorne II.2–3 |
| 9–10 | Proj construction; projective schemes; global Proj | Hartshorne II.2, II.5 |
| 11–12 | Fiber products; base change; examples (fibers of a morphism) | Hartshorne II.3, Mumford I.4 |
| 13–14 | Separated morphisms; proper morphisms; valuative criteria | Hartshorne II.4 |
| 15–16 | Quasi-coherent and coherent sheaves; $\widetilde{M}$ construction | Hartshorne II.5 |
| 17–18 | Invertible sheaves; Weil divisors; Cartier divisors; Picard group | Hartshorne II.6–7 |
| 19–20 | Kähler differentials $\Omega_{X/k}$; smooth/regular morphisms | Hartshorne II.8 |
| 21–22 | Blowing up; exceptional divisors; blow-up of an ideal sheaf | Hartshorne II.7 |

### Key Problems (Hartshorne Ch II)

Hartshorne exercises are the qualifying exams in disguise. Prioritize:
- II.1.22 (gluing sheaves), II.2.14 (sheaves on Spec), II.2.15
- II.3.3–3.5 (fiber products), II.3.12 (affine morphisms)
- II.5.1, II.5.17 (coherent sheaves), II.5.18
- II.6.1–6.6 (divisors), II.6.11, II.7.3 (blowing up)

### Key Geometric Intuitions to Extract

1. **Generic points.** The scheme $\text{Spec}(\mathbb{Z})$ has a "generic point" corresponding to $(0)$, plus closed points for each prime $p$. A family of schemes over $\text{Spec}(\mathbb{Z})$ encodes one scheme over each field $\mathbb{F}_p$ and one over $\mathbb{Q}$ simultaneously.
2. **Separation = Hausdorff analogue.** $f: X \to Y$ is separated iff the diagonal $\Delta: X \to X \times_Y X$ is a closed immersion. This prevents "doubled lines."
3. **Proper = universally closed + finite type + separated.** The correct algebraic analogue of compactness. Projective morphisms are the main examples.
4. **Coherent sheaves = algebraic vector bundles (locally).** A coherent sheaf on a scheme is, locally, the sheaf associated to a finitely generated module.

---

## Phase 3: Cohomology and Curves (~22 weeks)

### Texts

| Role | Text |
|---|---|
| Primary | Hartshorne, *Algebraic Geometry*, Chapters III–IV |
| Cohomology intuition | Serre, *Faisceaux Algébriques Cohérents* (FAC) |
| Curves supplement | Miranda, *Algebraic Curves and Riemann Surfaces* |

### Goal

Make the machinery pay off. Sheaf cohomology makes Riemann-Roch precise; curves are where every theorem is sharpest and most computable.

### Week-by-Week Plan

| Week | Topic | Geometric meaning | Source |
|---|---|---|---|
| 1–3 | Derived functors; sheaf cohomology; injective resolutions | $H^i(X, \mathcal{F})$ as derived $\Gamma(X, -)$ | Hartshorne III.1–2 |
| 4–6 | Čech cohomology; cohomology of $\mathbb{P}^n$; $H^i(\mathbb{P}^n, \mathcal{O}(m))$ | Explicit computation via standard affine cover | Hartshorne III.4–5 |
| 7–8 | Serre duality: statement + applications | $H^0(\mathcal{L})$ dual to $H^n(\omega \otimes \mathcal{L}^{-1})$ | Hartshorne III.7 |
| 9–10 | Cohomological criterion for ampleness; Serre vanishing | Higher cohomology vanishes for ample twists | Hartshorne III.5, III.7 |
| 11–12 | Flat morphisms; cohomology and base change | How do fibers of a family vary cohomologically? | Hartshorne III.9, III.12 |
| 13–14 | Riemann-Roch: $\ell(D) - \ell(K - D) = \deg D + 1 - g$ | Counts independent sections; corrected by genus | Hartshorne IV.1 |
| 15–16 | Linear systems on curves; very ample divisors; embeddings in $\mathbb{P}^n$ | Linear series $|D|$ gives map $X \to \mathbb{P}^n$ | Hartshorne IV.1–2 |
| 17–18 | Clifford's theorem; Castelnuovo bound; low-degree curves in $\mathbb{P}^3$ | Genus bounds from degree and embedding | Hartshorne IV.3 |
| 19–20 | Elliptic curves (scheme-theoretic): group law, $j$-invariant, Weierstrass | Group structure via $\text{Pic}^0(E)$; $j$ classifies over $\bar{k}$ | Hartshorne IV.4 |
| 21–22 | Hyperelliptic curves; canonical embeddings; genus 2–4 classification | How the canonical map $\phi_{K_X}$ behaves by genus | Hartshorne IV.5–6 |

### Key Problems

- Hartshorne III.4.1–4.5 (Čech cohomology)
- III.5.1–5.5 (projective space cohomology, Hilbert polynomials)
- IV.1.1–1.7 (Riemann-Roch applications)
- IV.4.1–4.10 (elliptic curves)
- All problems in the Harvard qual collection that fall under these topics

### Key Geometric Intuitions to Extract

1. **$H^0$ = sections, $H^1$ = obstructions.** $H^0(X, \mathcal{L})$ = global sections of a line bundle; $H^1(X, \mathcal{L})$ = obstruction to gluing local sections globally. Riemann-Roch is literally a formula for $h^0 - h^1$.
2. **Serre duality is Poincaré duality for algebraic varieties.** The dualizing sheaf $\omega_X$ plays the role of the orientation sheaf.
3. **Elliptic curves are groups because genus 1.** For a curve of genus $g$, the Abel-Jacobi map $X \to \text{Pic}^0(X)$ is an embedding when $g \geq 1$. When $g = 1$, $\text{Pic}^0(X) \cong X$ as varieties — so $X$ is its own Jacobian, which is why it's a group.

---

## Phase 4: Arithmetic Geometry (~18 weeks)

### Texts

| Role | Text |
|---|---|
| Primary | Silverman, *The Arithmetic of Elliptic Curves* (AEC) |
| Finite fields + zeta functions | Ireland-Rosen, *A Classical Introduction to Modern Number Theory*, Ch 8–11 |
| Weil conjectures | Milne, *Lectures on Étale Cohomology*, Ch 1–2 (for statement + intuition) |

### Goal

Add the number-theoretic dimension: elliptic curves over $\mathbb{Q}$, $\mathbb{F}_q$, and $\mathbb{Z}$. The Weil conjectures as a unifying theme. BSD as the motivating open problem.

### Week-by-Week Plan

| Week | Topic | Key ideas | Source |
|---|---|---|---|
| 1–3 | Elliptic curves over any field: Weierstrass, smoothness, group law | Group structure from Riemann-Roch over any field | Silverman AEC Ch 1–2 |
| 4–6 | Isogenies; torsion subgroups; $E[n] \cong (\mathbb{Z}/n)^2$ over $\bar{k}$ | Isogenies = group-scheme morphisms; Tate module | Silverman Ch 3 |
| 7–9 | Elliptic curves over $\mathbb{Q}$: Mordell-Weil theorem | $E(\mathbb{Q}) \cong \mathbb{Z}^r \oplus E(\mathbb{Q})_{\text{tors}}$ | Silverman Ch 4–8 |
| 10–11 | Torsion over $\mathbb{Q}$: Nagell-Lutz; Mazur's torsion theorem (statement) | The 15 finite groups that occur as $E(\mathbb{Q})_{\text{tors}}$ | Silverman Ch 7 |
| 12–13 | Elliptic curves over $\mathbb{F}_q$: Hasse's theorem; point counting; Frobenius | $|E(\mathbb{F}_q)| = q + 1 - t$, $|t| \leq 2\sqrt{q}$ | Silverman Ch 5, Ireland-Rosen Ch 8 |
| 14–15 | Zeta functions of curves over $\mathbb{F}_q$; rationality; functional equation | $Z(X/\mathbb{F}_q, T)$ rational; relates to Weil conjectures | Ireland-Rosen Ch 11, Milne Ch 1 |
| 16–17 | Weil conjectures (statement): rationality, functional equation, Riemann hypothesis | Eigenvalues of Frobenius on $H^i_{\text{ét}}$ have absolute value $q^{i/2}$ | Milne Ch 1–2 |
| 18 | BSD conjecture (statement); $L$-functions; rank and analytic behavior | $\text{ord}_{s=1} L(E,s) = \text{rank}(E(\mathbb{Q}))$ | Silverman Appendix C |

### Key Geometric Intuitions to Extract

1. **The group law on an elliptic curve comes from Riemann-Roch.** Two points $P, Q$ sum to $R$ iff $P + Q + R \sim 3O$ in $\text{Pic}^0(E)$ — the group structure is the algebraic avatar of complex torus addition.
2. **Frobenius is the arithmetic avatar of the fundamental class.** Over $\mathbb{F}_q$, the Frobenius endomorphism $\phi_q: x \mapsto x^q$ plays the role of a deck transformation; its eigenvalues on étale cohomology encode point counts via the Weil conjectures.
3. **The Weil conjectures say varieties over $\mathbb{F}_q$ "look like" compact Kähler manifolds.** The Riemann hypothesis component is the deepest — it says the "cohomological weight" of each piece of $H^i$ is exactly $i$, as in the Hodge decomposition.

---

## Milestones and Checkpoints

| Milestone | When | Assessment |
|---|---|---|
| Nullstellensatz + classical projective space | End of Phase 1, Wk 6 | Work through Gathmann Ch 1–3 problem set |
| Divisors and classical Riemann-Roch | End of Phase 1 | Prove RR for a genus 2 curve from Shafarevich |
| First scheme examples and fiber products | Phase 2, Wk 12 | Hartshorne II.3 exercises |
| Coherent sheaves and Picard group | Phase 2, Wk 18 | Hartshorne II.6 exercises |
| Čech cohomology computation | Phase 3, Wk 6 | Compute $H^i(\mathbb{P}^2, \mathcal{O}(m))$ for several values |
| Riemann-Roch applications | Phase 3, Wk 14 | Solve 5 problems from Harvard qual list |
| Elliptic curves over Q | Phase 4, Wk 9 | Prove Mordell-Weil for a specific curve |
| **Full qual readiness** | End of Phase 4 | Work all problems in Harvard qual collection and both Berkeley transcripts |

---

## References

| Text | Role |
|---|---|
| Shafarevich, *Basic Algebraic Geometry* Vol 1 | Phase 1 primary |
| Reid, *Undergraduate Algebraic Geometry* | Phase 1 companion |
| Gathmann, *Algebraic Geometry* (free notes) | Problem supplement |
| Atiyah-Macdonald, *Introduction to Commutative Algebra* | CA reference (woven in) |
| Eisenbud, *Commutative Algebra with a View Toward AG* | CA supplement (geometric commentary) |
| Hartshorne, *Algebraic Geometry* | Phase 2–3 primary |
| Mumford, *The Red Book of Varieties and Schemes* | Phase 2 companion |
| Vakil, *The Rising Sea* (free) | Phase 2 reference/exercises |
| Serre, *Faisceaux Algébriques Cohérents* (FAC) | Phase 3 cohomology |
| Miranda, *Algebraic Curves and Riemann Surfaces* | Phase 3 curves supplement |
| Silverman, *The Arithmetic of Elliptic Curves* | Phase 4 primary |
| Ireland-Rosen, *A Classical Introduction to Modern Number Theory* | Phase 4 finite fields |
| Milne, *Lectures on Étale Cohomology* (free) | Phase 4 Weil conjectures |

---

## Qualifying Exam Sources

| Source | URL |
|---|---|
| Ritvik Ramkumar (Berkeley, 2017) | https://math.berkeley.edu/~ritvik/Qualifying_Exam_Syllabus_and_Transcript.pdf |
| Will Fisher (Berkeley, 2024) | https://math.berkeley.edu/~willfisher/papers/Qual_Transcript.pdf |
| Harvard AG Qualifying Problem Collection | https://www.math.harvard.edu/media/alggeom.pdf |
