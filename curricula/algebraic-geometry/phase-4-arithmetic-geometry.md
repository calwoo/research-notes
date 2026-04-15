# Phase IV — Arithmetic Geometry

*Weeks 65–82 · ~117 hrs · Silverman + Milne*

> **Goal:** Add the number-theoretic dimension. Elliptic curves defined over $\mathbb{Q}$, $\mathbb{F}_q$, and $\mathbb{Z}$; the Mordell-Weil theorem; the Hasse bound; and the Weil conjectures as a grand synthesis. By the end, you should understand the statement of BSD and why it's hard, and be comfortable with how algebraic geometry over non-algebraically-closed fields works.

**Primary text:** Silverman, *The Arithmetic of Elliptic Curves* (AEC)
**Number theory supplement:** Ireland-Rosen, *A Classical Introduction to Modern Number Theory*, Ch 8–11
**Weil conjectures:** Milne, *Lectures on Étale Cohomology* (free), Ch 1–2

---

## Phase Bridge: From Geometry to Arithmetic

Phase III closed with the full scheme-theoretic treatment of curves over an algebraically closed field. Phase IV removes that assumption: the base field is now $\mathbb{Q}$, $\mathbb{F}_q$, or a number field. The key new phenomenon is that the Galois group $G_k = \text{Gal}(\bar{k}/k)$ acts everywhere.

| Phase III result | Phase IV upgrade |
|---|---|
| $E[n](\bar{k}) \cong (\mathbb{Z}/n)^2$ | $G_k$ acts on $E[n]$: the mod-$n$ Galois representation |
| $\text{Pic}^0(E) \cong E$ | $E(\mathbb{Q})$ is a finitely generated abelian group (Mordell-Weil) |
| $|E(\mathbb{F}_q)|$ is finite | Hasse bound: $|q + 1 - |E(\mathbb{F}_q)|| \leq 2\sqrt{q}$ |
| Zeta function = generating series | Weil conjectures: rationality, functional equation, Riemann hypothesis |
| Riemann-Roch over $\bar{k}$ | $L$-functions encode global arithmetic of $E/\mathbb{Q}$ |

---

### Weeks 65–67 — Elliptic Curves over Arbitrary Fields

**Concepts to understand:**

- [ ] Weierstrass equations and the discriminant: $\Delta = -16(4a^3 + 27b^2) \neq 0$ for smoothness
- [ ] Short Weierstrass form ($\text{char} \neq 2, 3$): $y^2 = x^3 + ax + b$
- [ ] The group law over any field $k$: the chord-and-tangent construction works over any field
- [ ] The group $E(k)$: $k$-rational points form an abelian group
- [ ] Isogenies over non-algebraically-closed fields: a finite morphism $\phi: E_1 \to E_2$ preserving the identity

**Reading:**

- [ ] Silverman AEC, Ch 1 (§1–4), Ch 2 (§1–4) *(~5 hrs)*

**Problems:**

- [ ] Silverman AEC: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.5, 2.6, 2.8, 2.11

---

### Weeks 68–69 — Isogenies and the Tate Module

**Concepts to understand:**

- [ ] Isogeny: nonzero morphism $\phi: E_1 \to E_2$ of elliptic curves (automatically surjective over $\bar{k}$)
- [ ] Degree of an isogeny: degree as a finite morphism; $\deg([n]) = n^2$
- [ ] The $n$-torsion $E[n](\bar{k}) \cong (\mathbb{Z}/n)^2$ for $\text{char}(k) \nmid n$
- [ ] Tate module: $T_\ell(E) = \varprojlim E[\ell^n] \cong \mathbb{Z}_\ell^2$ (as abelian groups over $\bar{k}$)
- [ ] Galois acts on $T_\ell(E)$: the $\ell$-adic representation $\rho_\ell: G_k \to \text{GL}_2(\mathbb{Z}_\ell)$

**Reading:**

- [ ] Silverman AEC, Ch 3 (§1–7) *(~4 hrs)*

**Problems:**

- [ ] Silverman AEC: 3.1, 3.2, 3.5, 3.6, 3.7, 3.10

---

### Weeks 70–72 — Mordell-Weil Theorem

**Concepts to understand:**

- [ ] Mordell-Weil theorem: $E(\mathbb{Q}) \cong \mathbb{Z}^r \oplus E(\mathbb{Q})_{\text{tors}}$ (finitely generated abelian group)
- [ ] The rank $r$: the number of independent generators of infinite order; can be 0, 1, 2, ...
- [ ] Proof strategy: (1) $E(\mathbb{Q})/2E(\mathbb{Q})$ is finite (weak Mordell-Weil); (2) infinite descent via heights
- [ ] Weil height function $h: E(\mathbb{Q}) \to \mathbb{R}$: measures arithmetic complexity of a rational point
- [ ] Canonical (Néron-Tate) height $\hat{h}$: the unique quadratic form approximating $h$; positive definite on $E(\mathbb{Q}) \otimes \mathbb{R}$

**Reading:**

- [ ] Silverman AEC, Ch 4 (§1–5: heights), Ch 8 (§1–2: Mordell-Weil) *(~6 hrs)*

**Problems:**

- [ ] Silverman AEC: 4.1, 4.3, 4.4, 8.1, 8.2, 8.5
- [ ] For $E: y^2 = x^3 - x$, show $E(\mathbb{Q})_{\text{tors}} \cong \mathbb{Z}/2 \times \mathbb{Z}/2$ and find a generator of infinite order if $r = 1$

> **Milestone:** State and sketch the proof of Mordell-Weil from scratch. Identify where each of the two main steps (weak M-W and descent) uses the height function.

---

### Weeks 73–74 — Torsion Subgroups over $\mathbb{Q}$

**Concepts to understand:**

- [ ] Nagell-Lutz theorem: if $E: y^2 = x^3 + ax + b$ with $a, b \in \mathbb{Z}$, then $E(\mathbb{Q})_{\text{tors}}$ has coordinates in $\mathbb{Z}$ and $y = 0$ or $y^2 \mid \Delta$
- [ ] Using Nagell-Lutz to find all torsion points: a finite computation
- [ ] Mazur's torsion theorem (statement): $E(\mathbb{Q})_{\text{tors}}$ is one of exactly 15 groups:
  - $\mathbb{Z}/n\mathbb{Z}$ for $n = 1, 2, \ldots, 10, 12$
  - $\mathbb{Z}/2 \times \mathbb{Z}/2n$ for $n = 1, 2, 3, 4$
- [ ] Reduction mod $p$: for good primes $p$, torsion injects into $\tilde{E}(\mathbb{F}_p)$

**Reading:**

- [ ] Silverman AEC, Ch 7 (§1–3: torsion) *(~3 hrs)*

**Problems:**

- [ ] Silverman AEC: 7.1, 7.2, 7.3, 7.5, 7.7
- [ ] Find all rational torsion points of $E: y^2 = x^3 - x$ using Nagell-Lutz

---

### Weeks 75–76 — Elliptic Curves over Finite Fields

**Concepts to understand:**

- [ ] $E(\mathbb{F}_q)$: the group of $\mathbb{F}_q$-rational points; a finite abelian group
- [ ] Hasse's theorem: $|E(\mathbb{F}_q)| = q + 1 - t$ where $|t| \leq 2\sqrt{q}$ (the "trace of Frobenius")
- [ ] The Frobenius endomorphism $\phi_q: (x,y) \mapsto (x^q, y^q)$: an isogeny of degree $q$
- [ ] Characteristic polynomial of Frobenius: $T^2 - tT + q$; roots $\alpha, \bar{\alpha}$ with $|\alpha| = \sqrt{q}$
- [ ] $|E(\mathbb{F}_{q^n})| = q^n + 1 - \alpha^n - \bar{\alpha}^n$ for all $n$

**Reading:**

- [ ] Silverman AEC, Ch 5 (§1–4: finite fields) *(~4 hrs)*
- [ ] Ireland-Rosen, Ch 8 §1–3 *(~2 hrs)*

**Problems:**

- [ ] Silverman AEC: 5.1, 5.2, 5.3, 5.4, 5.7, 5.9
- [ ] For $E: y^2 = x^3 + x$ over $\mathbb{F}_p$, compute $|E(\mathbb{F}_p)|$ for $p = 5, 7, 11, 13$

---

### Weeks 77–78 — Zeta Functions of Curves

**Concepts to understand:**

- [ ] Zeta function of a curve $X/\mathbb{F}_q$: $Z(X, T) = \exp\left(\sum_{n=1}^\infty |X(\mathbb{F}_{q^n})| \frac{T^n}{n}\right)$
- [ ] For a smooth projective curve of genus $g$ over $\mathbb{F}_q$:
  - $Z(X, T) = \frac{P(T)}{(1-T)(1-qT)}$ where $P(T) = \prod_{i=1}^{2g}(1 - \alpha_i T)$ is a polynomial of degree $2g$
  - Rationality: $Z(X,T) \in \mathbb{Q}(T)$
  - Functional equation: $Z(X, \frac{1}{qT}) = q^{1-g} T^{2-2g} Z(X, T)$
  - Riemann hypothesis: $|\alpha_i| = \sqrt{q}$ for all $i$
- [ ] For elliptic curves: $P(T) = 1 - tT + qT^2$ where $t$ is the trace of Frobenius

**Reading:**

- [ ] Ireland-Rosen, Ch 11 §1–4 *(~3 hrs)*
- [ ] Milne, *Lectures on Étale Cohomology*, Ch 1 §1.1–1.3 (introduction) *(~2 hrs)*

**Problems:**

- [ ] Compute $Z(E, T)$ for $E: y^2 = x^3 + x$ over $\mathbb{F}_5$ by counting points for $n = 1, 2, 3$
- [ ] Verify the functional equation for this example

---

### Weeks 79–80 — The Weil Conjectures

**Concepts to understand:**

- [ ] The Weil conjectures (1949) for a smooth projective variety $X/\mathbb{F}_q$ of dimension $n$:
  1. **Rationality:** $Z(X, T) = \prod_{i=0}^{2n} P_i(T)^{(-1)^{i+1}}$, each $P_i \in \mathbb{Z}[T]$
  2. **Functional equation:** $Z(X, \frac{1}{q^n T}) = \pm q^{n\chi/2} T^\chi Z(X, T)$ where $\chi = \chi_{\text{top}}(X)$
  3. **Riemann hypothesis:** roots of $P_i$ have absolute value $q^{-i/2}$
  4. **Betti numbers:** $\deg P_i = b_i$, the $i$-th Betti number of the "corresponding" complex variety
- [ ] Weil's proof for curves (1948): using the Riemann-Roch theorem and a positivity argument
- [ ] Grothendieck's proof strategy: $\ell$-adic cohomology $H^i_{\text{ét}}(X_{\bar{\mathbb{F}}_q}, \mathbb{Q}_\ell)$ plays the role of $H^i_{\text{sing}}$; Frobenius acts on it; the Lefschetz fixed-point formula gives the zeta function
- [ ] Deligne's proof of the Riemann hypothesis (1974): the deepest result; uses weights and Hard Lefschetz

**Reading:**

- [ ] Milne, *Lectures on Étale Cohomology*, Ch 1–2 (for statement, intuition, and Weil's proof for curves) *(~4 hrs)*
- [ ] Ireland-Rosen, Ch 11 §5–6 (Weil's proof for curves) *(~2 hrs)*

**Problems:**

- [ ] Verify the Weil conjectures for $\mathbb{P}^n_{\mathbb{F}_q}$ directly: compute $Z(\mathbb{P}^n, T)$ and check all four properties
- [ ] Verify the Weil conjectures for a smooth plane conic over $\mathbb{F}_q$ (genus 0, so $\mathbb{P}^1$ after base change)
- [ ] Sketch Weil's proof of the Riemann hypothesis for curves: where does Riemann-Roch enter?

> **Geometric insight:** The Weil conjectures say that the zeta function of a variety over $\mathbb{F}_q$ "behaves like" the zeta function of a compact complex manifold. This is the deep bridge between arithmetic and geometry: number theory over $\mathbb{F}_q$ is secretly topology of the corresponding complex variety.

---

### Week 81 — The BSD Conjecture

**Concepts to understand:**

- [ ] The $L$-function of an elliptic curve $E/\mathbb{Q}$: $L(E, s) = \prod_{p \nmid N} \frac{1}{1 - a_p p^{-s} + p^{1-2s}} \prod_{p \mid N} (\text{bad factors})$
  where $a_p = p + 1 - |E(\mathbb{F}_p)|$
- [ ] Modularity theorem (Wiles-Taylor-Wiles, 1995): every elliptic curve over $\mathbb{Q}$ is modular — $L(E,s) = L(f, s)$ for a modular form $f$
- [ ] BSD conjecture: $\text{ord}_{s=1} L(E, s) = \text{rank}(E(\mathbb{Q}))$
- [ ] Birch-Swinnerton-Dyer: numerical evidence by computing $|E(\mathbb{F}_p)|$ for many primes
- [ ] Why it is hard: the rank is "global" (points over $\mathbb{Q}$) while $L(E,s)$ is "local" (product over all primes)

**Reading:**

- [ ] Silverman AEC, Appendix C (BSD) *(~2 hrs)*
- [ ] Silverman, *The Arithmetic of Elliptic Curves*, Ch 5 §5 (introduction to $L$-functions) *(~2 hrs)*

**Problems:**

- [ ] For $E: y^2 = x^3 - x$ (rank 0), verify numerically that $L(E, 1) \neq 0$
- [ ] For $E: y^2 = x^3 - x^2 - 2x$ (rank 1), observe numerically that $L(E, s)$ has a simple zero at $s = 1$

---

### Week 82 — Final Qual Preparation

Use this week for a full mock qualifying exam.

**Mock exam checklist:**

- [ ] Work the complete Ritvik Ramkumar transcript from memory (algebraic geometry section)
- [ ] Work the complete Will Fisher transcript from memory (algebraic geometry section)
- [ ] Work all Harvard qualifying problems — aim for confidence on $> 80\%$
- [ ] Write clean answers to:
  - [ ] State and prove Riemann-Roch from scratch
  - [ ] State Serre duality and deduce $h^1(\mathcal{O}_E) = 1$ for an elliptic curve
  - [ ] Define Weil divisors, Cartier divisors, and $\text{Pic}(X)$; give an example where Weil $\neq$ Cartier
  - [ ] State Hurwitz's theorem and apply it to classify unramified covers of elliptic curves
  - [ ] Define a scheme. Give an example that is not a variety. Give an example of a non-separated scheme.
  - [ ] State the Weil conjectures and verify them for $\mathbb{P}^1_{\mathbb{F}_q}$

> **Final Milestone:** A successful qual preparation means you can respond to unexpected follow-up questions, not just recite definitions. The measure is: given any statement in the transcripts, can you reconstruct the *why* — the geometric picture — without having seen that specific question before?
