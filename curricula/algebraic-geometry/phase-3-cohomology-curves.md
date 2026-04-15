# Phase III — Cohomology and Curves

*Weeks 43–64 · ~143 hrs · Hartshorne Chapters III–IV*

> **Goal:** Make the machinery pay off. Sheaf cohomology is the tool that makes Riemann-Roch precise, gives Serre duality, and lets you compute everything. Curves are where every theorem is sharpest and most computable. By the end, you should be able to work all curve-related problems in all three qualifying exam sources.

**Primary text:** Hartshorne, *Algebraic Geometry*, Chapters III–IV
**Cohomology supplement:** Serre, *Faisceaux Algébriques Cohérents* (FAC) — the foundational paper; read after Hartshorne III for historical context
**Curves supplement:** Miranda, *Algebraic Curves and Riemann Surfaces* — bridges the analytic and algebraic viewpoints

---

## Phase Bridge: Cohomology as Obstruction Theory

The passage from Phase II to Phase III is conceptual: Phase II built the language (schemes, sheaves, morphisms), Phase III makes the language compute. The key upgrade is sheaf cohomology, which converts geometric questions into linear algebra.

| Phase II construction | Phase III payoff |
|---|---|
| Quasi-coherent sheaf $\mathcal{F}$ on $X$ | $H^i(X, \mathcal{F})$: measures "holes" in $\mathcal{F}$ |
| Line bundle $\mathcal{L} = \mathcal{O}(D)$ | $h^0(\mathcal{L})$: dimension of global sections; gives $\ell(D)$ |
| Kähler differentials $\Omega_{X/k}$ | $\omega_X = \Omega_{X/k}^{\dim X}$: the dualizing sheaf for Serre duality |
| Morphism $f: X \to Y$ | $R^i f_* \mathcal{F}$: higher direct images |
| Hilbert polynomial $P_X(m)$ | $P_X(m) = \chi(\mathcal{O}_X(m))$: now provably a polynomial by finite-dimensionality of cohomology |

---

## Weeks 43–58 — Core Theory and Curves

### Week 43 — Derived Functors

**CA prerequisite:** A&M Ch 2 (Hom and tensor), Ch 6 (chain complexes — not in A&M, use Weibel §2.1–2.3 or Hartshorne App A).

**Concepts to understand:**

- [ ] Left-exact functors and their failure to be exact: the prototype is $\text{Hom}(M, -)$ and $\Gamma(X, -)$
- [ ] Injective objects and injective resolutions: every module/sheaf embeds into an injective
- [ ] Derived functor $R^i F$: apply $F$ to an injective resolution, take cohomology
- [ ] $R^0 F = F$ (exactness on the right)
- [ ] Long exact sequence in derived functors: $0 \to F(A) \to F(B) \to F(C) \to R^1 F(A) \to \cdots$

**Reading:**

- [ ] Hartshorne §III.1 *(~3 hrs)*
- [ ] Weibel, *Introduction to Homological Algebra*, Ch 2 §2.1–2.3 (for derived functors cleanly) *(~2 hrs)*

**Problems:**

- [ ] Hartshorne III.1.1, III.1.2, III.1.4
- [ ] Show that Ext and Tor are the derived functors of Hom and tensor for modules

---

### Week 44 — Sheaf Cohomology

**Concepts to understand:**

- [ ] Sheaf cohomology $H^i(X, \mathcal{F}) = R^i \Gamma(X, \mathcal{F})$: derived functor of global sections
- [ ] Geometric meaning: $H^0$ = global sections; $H^1$ = obstruction to gluing; higher cohomology = higher obstructions
- [ ] Flasque (flabby) sheaves: surjective restriction maps; acyclic for $\Gamma$
- [ ] Injective sheaves are flasque; use flasque resolutions to compute
- [ ] $H^i(X, \mathcal{F}) = 0$ for $i > \dim X$ (Grothendieck vanishing — statement)

**Reading:**

- [ ] Hartshorne §III.2 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne III.2.1, III.2.2, III.2.3, III.2.5, III.2.7

---

### Week 45 — Cohomology of Noetherian Affine Schemes

**Concepts to understand:**

- [ ] Serre's theorem: $H^i(X, \mathcal{F}) = 0$ for all $i > 0$ and quasi-coherent $\mathcal{F}$ iff $X$ is affine (Noetherian)
- [ ] Geometric meaning: affine schemes have "no holes" — all cohomology vanishes
- [ ] Criterion for affineness via $H^1$: if $H^1(X, \mathcal{I}) = 0$ for all coherent ideal sheaves $\mathcal{I}$, then $X$ is affine
- [ ] Application: $\mathbb{P}^n$ is not affine (because $H^n(\mathbb{P}^n, \mathcal{O}(-n-1)) \neq 0$)

**Reading:**

- [ ] Hartshorne §III.3 *(~2.5 hrs)*

**Problems:**

- [ ] Hartshorne III.3.1, III.3.2, III.3.3, III.3.4
- [ ] Harvard qual: "How can you tell if a scheme is affine?" — work Ogus questions on affineness criteria

---

### Week 46 — Čech Cohomology

**Concepts to understand:**

- [ ] Čech complex for an open cover $\mathfrak{U} = \{U_i\}$: $\check{C}^p(\mathfrak{U}, \mathcal{F}) = \prod_{i_0 < \cdots < i_p} \mathcal{F}(U_{i_0} \cap \cdots \cap U_{i_p})$
- [ ] Čech cohomology $\check{H}^p(\mathfrak{U}, \mathcal{F})$: cohomology of the Čech complex
- [ ] Agreement with sheaf cohomology: $\check{H}^p(\mathfrak{U}, \mathcal{F}) \cong H^p(X, \mathcal{F})$ when the cover is acyclic (e.g., affine cover on a separated scheme)
- [ ] Practical value: Čech cohomology is computable directly from a cover

**Reading:**

- [ ] Hartshorne §III.4 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne III.4.1, III.4.2, III.4.3, III.4.4, III.4.5
- [ ] Compute $H^1(\mathbb{P}^1, \mathcal{O})$ directly using the standard affine cover $\{U_0, U_1\}$

---

### Week 47 — Cohomology of Projective Space

**Concepts to understand:**

- [ ] Key computation: $H^i(\mathbb{P}^n_k, \mathcal{O}(m))$ for all $i$ and $m$:
  - $H^0(\mathbb{P}^n, \mathcal{O}(m)) = k[x_0, \ldots, x_n]_m$ (degree-$m$ forms) for $m \geq 0$, $0$ for $m < 0$
  - $H^n(\mathbb{P}^n, \mathcal{O}(m)) = 0$ for $m \geq -n$, and dual to $H^0(\mathcal{O}(-m-n-1))$ for $m < -n$
  - $H^i = 0$ for $0 < i < n$
- [ ] This computation is the engine behind Riemann-Roch and Serre duality
- [ ] Hilbert polynomial $P_X(m) = \chi(\mathcal{O}_X(m)) = \sum (-1)^i h^i(\mathcal{O}_X(m))$

**Reading:**

- [ ] Hartshorne §III.5 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne III.5.1, III.5.2, III.5.3, III.5.5
- [ ] Compute $H^i(\mathbb{P}^2, \mathcal{O}(m))$ for $m = -4, -3, -2, -1, 0, 1, 2$

> **Geometric insight:** The "duality" in the computation of $H^n(\mathbb{P}^n, \mathcal{O}(m))$ vs $H^0(\mathcal{O}(-m-n-1))$ is the first appearance of Serre duality. The dualizing sheaf of $\mathbb{P}^n$ is $\omega_{\mathbb{P}^n} = \mathcal{O}(-n-1)$.

---

### Week 48 — Serre Duality

**Concepts to understand:**

- [ ] Dualizing sheaf $\omega_X$: the unique coherent sheaf such that $H^n(X, \omega_X) \cong k$ and Serre duality holds
- [ ] Serre duality: for $X$ smooth projective of dimension $n$ over $k$, $H^i(X, \mathcal{F}) \times H^{n-i}(X, \omega_X \otimes \mathcal{F}^\vee) \to H^n(X, \omega_X) \cong k$ is a perfect pairing
- [ ] Consequence: $h^i(X, \mathcal{F}) = h^{n-i}(X, \omega_X \otimes \mathcal{F}^\vee)$
- [ ] For a smooth projective curve: $\omega_X = \Omega_{X/k}$ (the canonical sheaf)
- [ ] For $\mathbb{P}^n$: $\omega_{\mathbb{P}^n} = \mathcal{O}(-n-1)$; verified by the cohomology computation of Week 47

**Reading:**

- [ ] Hartshorne §III.7 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne III.7.1, III.7.2, III.7.3, III.7.5
- [ ] Use Serre duality to show $h^1(X, \mathcal{O}) = g$ for a smooth projective curve of genus $g$

---

### Week 49 — Vanishing Theorems

**Concepts to understand:**

- [ ] Cohomological criterion for ampleness: $\mathcal{L}$ is ample iff for every coherent $\mathcal{F}$, $H^i(X, \mathcal{F} \otimes \mathcal{L}^n) = 0$ for $i > 0$ and $n \gg 0$
- [ ] Serre vanishing: if $\mathcal{L}$ is ample, $H^i(X, \mathcal{F} \otimes \mathcal{L}^n) = 0$ for all $i > 0$ and $n \gg 0$
- [ ] Nakano vanishing (characteristic 0): $H^q(X, \Omega^p_X \otimes \mathcal{L}) = 0$ for $p + q > \dim X$ and $\mathcal{L}$ ample
- [ ] Grothendieck vanishing: $H^i(X, \mathcal{F}) = 0$ for $i > \dim X$

**Reading:**

- [ ] Hartshorne §III.5 (ampleness), §III.7 (vanishing) *(~2.5 hrs)*

**Problems:**

- [ ] Hartshorne III.5.2, III.5.4
- [ ] Show $\mathcal{O}(1)$ is ample on $\mathbb{P}^n$ using the vanishing criterion

---

### Week 50 — Flat Morphisms

**CA prerequisite:** A&M Ch 10 (flatness, local criterion) — read §10.1–10.3 this week.

**Concepts to understand:**

- [ ] Flat module: $M$ is flat over $A$ iff $- \otimes_A M$ is exact
- [ ] Flat morphism $f: X \to Y$: $\mathcal{O}_{X,x}$ is flat over $\mathcal{O}_{Y,f(x)}$ for all $x$
- [ ] Geometric meaning: flat morphism = "continuously varying family of schemes"
- [ ] Flat families: the fibers vary "nicely" — Hilbert polynomial is constant
- [ ] Miracle flatness: if $X, Y$ are regular and $f$ is surjective with equidimensional fibers, then $f$ is flat

**Reading:**

- [ ] Hartshorne §III.9 *(~3 hrs)*
- [ ] A&M Ch 10 §10.1–10.3 *(~2 hrs)*

**Problems:**

- [ ] Hartshorne III.9.1, III.9.2, III.9.3, III.9.4

---

### Week 51 — Cohomology and Base Change

**Concepts to understand:**

- [ ] For a flat proper morphism $f: X \to Y$ and coherent $\mathcal{F}$: the function $y \mapsto h^i(X_y, \mathcal{F}_y)$ is upper semicontinuous
- [ ] Cohomology and base change theorem: if $y \mapsto h^i(X_y, \mathcal{F}_y)$ is constant, then $R^i f_* \mathcal{F}$ is locally free and commutes with base change
- [ ] Hilbert scheme (preview): flat proper families of subschemes have a moduli space — the Hilbert scheme parameterizes them
- [ ] Application to linear series: a linear system on the generic fiber extends to all fibers when flatness holds

**Reading:**

- [ ] Hartshorne §III.12 *(~2.5 hrs)*

**Problems:**

- [ ] Hartshorne III.12.1, III.12.2, III.12.4

---

### Week 52 — The Riemann-Roch Theorem (Scheme-Theoretic)

**Concepts to understand:**

- [ ] Euler characteristic $\chi(\mathcal{F}) = \sum_i (-1)^i h^i(X, \mathcal{F})$: the alternating sum of cohomology dimensions
- [ ] Riemann-Roch for curves: $\chi(\mathcal{O}(D)) = \deg D + 1 - g$; equivalently $h^0 - h^1 = \deg D + 1 - g$
- [ ] Serre duality gives $h^1(\mathcal{O}(D)) = h^0(\omega_X(-D)) = h^0(\mathcal{O}(K - D))$, recovering the classical form $\ell(D) - \ell(K-D) = \deg D + 1 - g$
- [ ] Riemann-Roch for surfaces (Noether's formula): $\chi(\mathcal{L}) = \frac{1}{2} \mathcal{L} \cdot (\mathcal{L} - K) + \chi(\mathcal{O}_X)$
- [ ] The sheaf-theoretic proof of RR is essentially the computation of the Hilbert polynomial of $\mathcal{O}_X(D)$

**Reading:**

- [ ] Hartshorne §IV.1 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne IV.1.1, IV.1.2, IV.1.3, IV.1.4, IV.1.5, IV.1.6, IV.1.7
- [ ] Work all Harvard qual problems involving Riemann-Roch

> **Milestone:** You should now be able to prove Riemann-Roch from scratch and apply it to compute $\ell(D)$ for any divisor $D$ on any smooth projective curve, given the genus.

---

### Week 53 — Linear Systems and Embeddings

**Concepts to understand:**

- [ ] A divisor $D$ on a curve $X$ is base-point-free iff $h^0(D - P) = h^0(D) - 1$ for all points $P$
- [ ] $D$ is very ample iff it separates points ($h^0(D - P - Q) = h^0(D) - 2$ for all $P \neq Q$) and separates tangent directions
- [ ] For $\deg D > 2g$: $D$ is base-point-free; for $\deg D > 2g + 1$ (or $\deg D \geq 2g + 1$ for $g \geq 1$): $D$ is very ample
- [ ] The canonical map $\phi_K: X \to \mathbb{P}^{g-1}$ (when $g \geq 2$) is an embedding iff $X$ is non-hyperelliptic

**Reading:**

- [ ] Hartshorne §IV.1 (continued), §IV.2 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne IV.2.1, IV.2.2, IV.2.3, IV.2.5, IV.2.7
- [ ] Show that every genus-2 curve is hyperelliptic (the canonical map is 2:1 onto $\mathbb{P}^1$)

---

### Week 54 — Clifford's Theorem and Castelnuovo's Bound

**Concepts to understand:**

- [ ] Special divisors: $D$ is special iff $h^1(D) = h^0(K - D) > 0$, i.e., $\deg D \leq 2g - 2$
- [ ] Clifford's theorem: for a special divisor $D$ on a curve of genus $g$, $h^0(D) \leq \frac{\deg D}{2} + 1$; equality iff $D = 0$ or $D = K$ or $X$ is hyperelliptic
- [ ] Castelnuovo's bound: if $X \hookrightarrow \mathbb{P}^3$ is a smooth curve of degree $d \geq 3$ not lying in a plane, then $g \leq \frac{1}{4}(d-1)^2 + 1$ (for $d$ odd: $g \leq \frac{1}{4}d(d-2) + 1$)
- [ ] Application: degree 3, genus 1 curves in $\mathbb{P}^3$ lie in planes (this is Ritvik's qual question)

**Reading:**

- [ ] Hartshorne §IV.3 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne IV.3.1, IV.3.2, IV.3.3, IV.3.4
- [ ] Work Ritvik's qual question: show a degree-3, genus-1 curve in $\mathbb{P}^3$ lies in a plane (two ways: Castelnuovo and direct RR)

---

### Week 55 — Elliptic Curves: Scheme-Theoretic Group Structure

**Concepts to understand:**

- [ ] An elliptic curve is a smooth projective curve of genus 1 with a marked point $O$: $(E, O)$
- [ ] The group law via $\text{Pic}^0(E)$: $P + Q = R$ iff $[P] - [O] + [Q] - [O] \sim [R] - [O]$ in $\text{Pic}^0(E)$
- [ ] $\text{Pic}^0(E) \cong E$ as varieties (Abel's theorem for genus 1)
- [ ] Consequence: $E$ is a group scheme — there is a group law morphism $E \times E \to E$
- [ ] The $j$-invariant: $j = 1728 \cdot \frac{4a^3}{4a^3 + 27b^2}$; two elliptic curves are isomorphic over $\bar{k}$ iff they have the same $j$

**Reading:**

- [ ] Hartshorne §IV.4 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne IV.4.1, IV.4.2, IV.4.4, IV.4.6, IV.4.7, IV.4.10
- [ ] Harvard qual: "What is the significance of the Jacobian? What is it in the case of genus 1?" — write a careful answer

---

### Week 56 — The Multiplication-by-$n$ Map and Torsion

**Concepts to understand:**

- [ ] The isogeny $[n]: E \to E$ (multiplication by $n$): a finite morphism of degree $n^2$
- [ ] The $n$-torsion subgroup $E[n] = \ker([n])$: as a group scheme, $E[n] \cong (\mathbb{Z}/n)^2$ over $\bar{k}$ when $\text{char}(k) \nmid n$
- [ ] Ritvik's qual: unramified maps of elliptic curves = isogenies; $[2]: E \to E$ as the unramified example
- [ ] Kernel of $[2]$: the 2-torsion points, computed by finding roots of $y = 0$ in Weierstrass form; $|E[2]| = 4$
- [ ] Hurwitz's theorem for $[n]: E \to E$: $\deg R = 0$ (unramified) iff $g(E) = 1$ — confirmed by Hurwitz

**Reading:**

- [ ] Hartshorne §IV.4 (continued) *(~2 hrs)*
- [ ] Work through Ritvik's qual transcript, questions by Eisenbud and Auroux on $[2]: E \to E$ *(~1.5 hrs)*

**Problems:**

- [ ] Show $[2]: E \to E$ is unramified for $\text{char}(k) \neq 2$ by computing the ramification divisor via Hurwitz
- [ ] Compute $E[2]$ explicitly for $E: y^2 = x(x-1)(x+1)$

---

### Weeks 57–58 — Curves of Higher Genus

**Concepts to understand:**

- [ ] Hyperelliptic curves of genus $g$: $y^2 = f(x)$ with $\deg f = 2g+1$ or $2g+2$ (char $\neq 2$); always admit a 2:1 map to $\mathbb{P}^1$
- [ ] Genus 2 curves: always hyperelliptic; $y^2 = f(x)$ with $\deg f = 5$ or $6$
- [ ] Genus 3 curves: either hyperelliptic, or the canonical embedding is a smooth plane quartic in $\mathbb{P}^2$
- [ ] Genus 4 curves: the canonical embedding lands in $\mathbb{P}^3$ as the intersection of a quadric and a cubic surface
- [ ] Genus 5 curves: the canonical embedding lands in $\mathbb{P}^4$ as the intersection of three quadric hypersurfaces

**Reading:**

- [ ] Hartshorne §IV.5, §IV.6 *(~3 hrs)*

**Problems:**

- [ ] Hartshorne IV.5.1, IV.5.2, IV.6.1, IV.6.2, IV.6.3, IV.6.4
- [ ] Show a smooth plane quartic has genus 3 via the genus formula AND via Hurwitz (degree-4 map to $\mathbb{P}^1$)

---

## Weeks 59–64 — Phase III Consolidation and Qual Practice

Use these six weeks for consolidation, returning to weak areas, and intensive qual problem work.

**Week 59:** Work all Harvard qual problems involving curves, genus, and Riemann-Roch
- [ ] Compute genus of specific curves by multiple methods (genus formula, Hurwitz, RR)
- [ ] "Find the arithmetic genus of $y^3 = x^2 z$" — work through Frenkel's question

**Week 60:** Work all Harvard qual problems involving cohomology, line bundles, and Pic
- [ ] "$H^1$ and line bundles" — Wodzicki questions
- [ ] "Serre's affineness criterion" — Ogus questions
- [ ] "Is the complement of a hypersurface in $\mathbb{P}^2$ affine?" — Poonen question

**Week 61:** Work through Ritvik's qual transcript (algebraic geometry section) completely
- [ ] Write full solutions to all questions, including the ones solved verbally

**Week 62:** Work through Will Fisher's qual transcript (algebraic geometry section) completely
- [ ] Morphism types, Picard groups, Cartier vs Weil, Bezout, scheme-theoretic intersection

**Week 63:** Read Serre's FAC (Faisceaux Algébriques Cohérents) §1–3
- [ ] Understand the historical context: FAC introduced coherent sheaves and their cohomology
- [ ] See how Serre's original proofs compare to Hartshorne's presentation

**Week 64:** Write a self-assessment
- [ ] List every question from all three qual sources you can now answer confidently
- [ ] List gaps; prioritize for review during Phase IV where possible

> **Phase III Milestone:** You should now be able to follow every question in the algebraic geometry sections of Ritvik's and Will Fisher's Berkeley qual transcripts, and work the majority of the Harvard qual collection.
