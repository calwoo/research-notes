# Phase II — Scheme Theory
*Weeks 21–42 · ~143 hrs*

> **Goal:** Translate every concept from Phase I into scheme language. The central conceptual upgrade: points are prime ideals (not just maximal ideals), and varieties become the special case of schemes over an algebraically closed field. By the end, you should be comfortable with Spec, Proj, coherent sheaves, the Picard group, Kähler differentials, and blowing up — and you should be able to follow the algebraic geometry section of both Berkeley qual transcripts.

**Primary text:** Hartshorne, *Algebraic Geometry*, Chapters I–II
**Geometric companion:** Mumford, *The Red Book of Varieties and Schemes*, Chapters I–II (read alongside Hartshorne — Mumford is the geometric conscience of this phase)
**Supplementary:** Vakil, *The Rising Sea* (free; use when Hartshorne is too terse or skips details)

### The Translation Dictionary

| Classical (Phase I) | Scheme-theoretic (Phase II) |
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

---

### Week 21 — Spec A and the Zariski Topology

**Concepts to understand:**

- [ ] $\text{Spec}(A)$: the set of prime ideals of a commutative ring $A$
- [ ] Why primes, not just maximal ideals? Points of $\text{Spec}(\mathbb{Z})$: $(p)$ for each prime $p$, plus the generic point $(0)$
- [ ] Zariski topology on $\text{Spec}(A)$: closed sets $V(I) = \{\mathfrak{p} \supseteq I\}$; basis of open sets $D(f) = \{\mathfrak{p} \nmid f\}$
- [ ] $\text{Spec}(A)$ is a sober topological space: every irreducible closed set has a unique generic point
- [ ] Functoriality: ring map $A \to B$ induces continuous map $\text{Spec}(B) \to \text{Spec}(A)$

**Reading:**

- [ ] Hartshorne §II.2 (first 6 pages) *(~2 hrs)*
- [ ] Mumford Red Book §I.1 *(~2 hrs)*
- [ ] Vakil *The Rising Sea*, Ch 3 §3.1–3.4 *(~1.5 hrs)*

**Problems:**

- [ ] Hartshorne II.2.2, II.2.3
- [ ] Describe $\text{Spec}(\mathbb{Z})$, $\text{Spec}(\mathbb{C}[t])$, $\text{Spec}(\mathbb{C}[x,y]/(xy))$ as topological spaces with generic points

> **Geometric insight:** $\text{Spec}(\mathbb{Z})$ is a curve! It has one closed point for each prime $p$ and one generic point. A "function" on it is an integer. This is the origin of arithmetic geometry: number theory is literally algebraic geometry over $\text{Spec}(\mathbb{Z})$.

---

### Week 22 — The Structure Sheaf

**Concepts to understand:**

- [ ] Sheaf of rings $\mathcal{O}_X$ on $\text{Spec}(A)$: $\mathcal{O}_X(D(f)) = A_f$ (localization at $f$)
- [ ] Stalk $\mathcal{O}_{X,\mathfrak{p}} = A_\mathfrak{p}$ (localization at prime $\mathfrak{p}$) — this is the local ring at a point
- [ ] Translation: $\mathcal{O}_{X,\mathfrak{m}} = A_\mathfrak{m}$ for a maximal ideal $\mathfrak{m}$ is the classical local ring from Phase I
- [ ] $\mathcal{O}_X$ is a sheaf of local rings: each stalk is a local ring
- [ ] Global sections: $\Gamma(\text{Spec}(A), \mathcal{O}) = A$

**Reading:**

- [ ] Hartshorne §II.2 (structure sheaf construction) *(~2.5 hrs)*
- [ ] Vakil Ch 2 §2.1–2.5 (sheaf axioms, stalks) *(~2 hrs)*

**Problems:**

- [ ] Hartshorne II.1.1, II.1.2, II.1.15 (sheaf axioms)
- [ ] Show that $\mathcal{O}_X(D(f)) = A_f$ is well-defined (i.e., consistent on overlaps)

---

### Week 23 — Presheaves and Sheaves

**Concepts to understand:**

- [ ] Presheaf: assignment $U \mapsto \mathcal{F}(U)$ with restriction maps
- [ ] Sheaf axioms: identity (a section is determined by its germs) and gluing (compatible local sections glue uniquely)
- [ ] Stalk $\mathcal{F}_P = \varinjlim_{U \ni P} \mathcal{F}(U)$: the "germ" at $P$
- [ ] Sheafification of a presheaf: the universal sheaf approximation
- [ ] Morphism of sheaves; kernel, image, cokernel as sheaves (exactness is at the level of stalks)

**Reading:**

- [ ] Hartshorne §II.1 *(~2 hrs)*
- [ ] Vakil Ch 2 §2.1–2.7 *(~2.5 hrs)*

**Problems:**

- [ ] Hartshorne II.1.1, II.1.3, II.1.6, II.1.8, II.1.14, II.1.22 (gluing sheaves — important)

---

### Week 24 — Locally Ringed Spaces and Schemes

**Concepts to understand:**

- [ ] Locally ringed space: topological space $X$ with a sheaf of rings $\mathcal{O}_X$ such that each stalk is a local ring
- [ ] Morphism of locally ringed spaces: continuous map + sheaf map respecting local ring structure at each stalk
- [ ] Affine scheme: $(X, \mathcal{O}_X) \cong (\text{Spec}(A), \mathcal{O}_{\text{Spec}(A)})$ for some ring $A$
- [ ] Scheme: locally ringed space that has an open cover by affine schemes
- [ ] The category of affine schemes is equivalent to the opposite of the category of commutative rings

**Reading:**

- [ ] Hartshorne §II.2 (schemes) *(~3 hrs)*
- [ ] Mumford Red Book §I.2–I.3 *(~2 hrs)*

**Problems:**

- [ ] Hartshorne II.2.1, II.2.2, II.2.4, II.2.7, II.2.9
- [ ] Construct the "line with a doubled origin" as a non-affine scheme by gluing two copies of $\text{Spec}(k[t])$

---

### Week 25 — First Examples of Schemes

**Concepts to understand:**

- [ ] Reduced, integral, irreducible schemes: the scheme-theoretic analogues of varieties
- [ ] $\text{Spec}(k[x]/(x^2))$: the "fat point" — a non-reduced scheme with one topological point but two scheme-theoretic points
- [ ] $\text{Spec}(k[x,y]/(xy))$: two lines meeting at the origin — the union in scheme theory
- [ ] Generic point of an integral scheme: the unique point $\eta$ with $\overline{\{\eta\}} = X$
- [ ] Scheme over $S$: a scheme with a structural morphism $X \to S$; $k$-schemes for varieties

**Reading:**

- [ ] Hartshorne §II.2 (examples throughout) *(~2 hrs)*
- [ ] Vakil Ch 3 §3.2–3.4 (examples of Spec) *(~2 hrs)*

**Problems:**

- [ ] Hartshorne II.2.3, II.2.6, II.2.11, II.2.14, II.2.15
- [ ] Show $\text{Spec}(k[x,y]/(x^2, xy, y^2))$ has one topological point; what is its local ring?

---

### Week 26 — Morphisms of Schemes

**Concepts to understand:**

- [ ] Open and closed immersions; open and closed subschemes
- [ ] Affine morphisms: $f: X \to Y$ such that $f^{-1}(V)$ is affine for every affine $V \subset Y$
- [ ] Morphisms of finite type: locally of finite type (finitely generated ring map on affines)
- [ ] Quasi-compact and quasi-separated morphisms
- [ ] $\text{Hom}_S(T, X)$: $T$-valued points of $X$ over $S$; the functor-of-points perspective

**Reading:**

- [ ] Hartshorne §II.3 (morphisms) *(~3 hrs)*
- [ ] Vakil Ch 7 §7.1–7.3 *(~2 hrs)*

**Problems:**

- [ ] Hartshorne II.3.1, II.3.2, II.3.4, II.3.5, II.3.6, II.3.12

---

### Week 27 — The Proj Construction

**Concepts to understand:**

- [ ] Graded ring $S = \bigoplus_{d \geq 0} S_d$ and the homogeneous prime spectrum $\text{Proj}(S)$
- [ ] $\text{Proj}(S)$: set of homogeneous primes not containing the irrelevant ideal $S_+ = \bigoplus_{d \geq 1} S_d$
- [ ] Open cover: $D_+(f) = \{\mathfrak{p} \nmid f\} \cong \text{Spec}(S_{(f)})$ for homogeneous $f \in S_1$
- [ ] $\text{Proj}(k[x_0, \ldots, x_n]) = \mathbb{P}^n_k$: the scheme-theoretic projective space
- [ ] Twisting sheaves $\mathcal{O}(n)$ on $\mathbb{P}^n$: $\mathcal{O}(n)(D_+(f)) = S_{(f)}$, sections in degree $n$

**Reading:**

- [ ] Hartshorne §II.2 (Proj), §II.5 (twisting sheaves) *(~3 hrs)*
- [ ] Mumford Red Book §II.4 *(~1.5 hrs)*

**Problems:**

- [ ] Hartshorne II.2.10, II.2.13, II.2.14
- [ ] Show $\text{Proj}(k[x,y]) \cong \mathbb{P}^1_k$. Identify $\mathcal{O}(1)$ with the line bundle of degree-1 functions.

---

### Week 28 — Fiber Products and Base Change

**Concepts to understand:**

- [ ] Fiber product $X \times_S Y$ in the category of $S$-schemes: the categorical product over $S$
- [ ] For affines: $\text{Spec}(A) \times_{\text{Spec}(R)} \text{Spec}(B) = \text{Spec}(A \otimes_R B)$
- [ ] Geometric fiber $X_s = X \times_S \text{Spec}(\kappa(s))$ over a point $s \in S$
- [ ] Base change: properties of morphisms are preserved under base change (separated, finite type, proper, flat)
- [ ] Scheme-theoretic intersection: $V(I) \cap V(J) = \text{Spec}(A/I \otimes_A A/J) = \text{Spec}(A/(I+J))$

**Reading:**

- [ ] Hartshorne §II.3 (fiber products) *(~3 hrs)*
- [ ] Mumford Red Book §I.4 *(~1.5 hrs)*

**Problems:**

- [ ] Hartshorne II.3.3, II.3.4, II.3.5, II.3.15, II.3.18
- [ ] Compute the scheme-theoretic intersection of the hyperbola $xy = 1$ and the circle $x^2 + y^2 = 2$ in $\mathbb{A}^2_\mathbb{R}$. What does the local ring at a tangent intersection look like?

> **Geometric insight:** This is the Will Fisher qual question. The scheme-theoretic intersection at a tangent point is $k[x]/(x^2)$ — the dual numbers. The scheme "remembers" the tangency that the set-theoretic intersection misses.

---

### Week 29 — Separated and Proper Morphisms

**Concepts to understand:**

- [ ] Separated morphism: the diagonal $\Delta: X \to X \times_S X$ is a closed immersion
- [ ] Geometric meaning: any two morphisms from a reduced scheme that agree on a dense open agree everywhere (no "doubling")
- [ ] Proper morphism: separated + finite type + universally closed
- [ ] Valuative criterion for separatedness: for any DVR $R$ with fraction field $K$, any $K$-point extends to at most one $R$-point
- [ ] Valuative criterion for properness: extends to exactly one $R$-point

**Reading:**

- [ ] Hartshorne §II.4 *(~3 hrs)*
- [ ] Vakil Ch 10 §10.1–10.3 *(~1.5 hrs)*

**Problems:**

- [ ] Hartshorne II.4.1, II.4.2, II.4.3, II.4.6, II.4.7, II.4.11
- [ ] Show that $\mathbb{P}^n_k \to \text{Spec}(k)$ is proper. Why does this capture "compactness"?
- [ ] Show the "line with doubled origin" is not separated

---

### Week 30 — Quasi-Coherent Sheaves

**CA prerequisite:** A&M Ch 2 (modules) and Ch 3 (localization) — should be already read.

**Concepts to understand:**

- [ ] $\mathcal{O}_X$-module: sheaf of abelian groups with compatible $\mathcal{O}_X$-module structure on each open set
- [ ] Quasi-coherent sheaf on $\text{Spec}(A)$: $\widetilde{M}$ for an $A$-module $M$, with $\widetilde{M}(D(f)) = M_f$
- [ ] Equivalence: $\text{QCoh}(\text{Spec}(A)) \simeq A\text{-Mod}$ (quasi-coherent sheaves = modules)
- [ ] Coherent sheaf: quasi-coherent and locally presented by a finitely generated module
- [ ] The sheaf associated to an ideal $I \subset A$: the ideal sheaf $\widetilde{I} \hookrightarrow \mathcal{O}_X$

**Reading:**

- [ ] Hartshorne §II.5 (first half) *(~3 hrs)*
- [ ] Vakil Ch 13 §13.1–13.3 *(~1.5 hrs)*

**Problems:**

- [ ] Hartshorne II.5.1, II.5.2, II.5.3, II.5.4, II.5.7

---

### Week 31 — Locally Free Sheaves and Vector Bundles

**Concepts to understand:**

- [ ] Locally free sheaf of rank $r$: $\mathcal{F}$ such that $\mathcal{F}|_U \cong \mathcal{O}_U^r$ for some cover
- [ ] Geometric avatar: a rank-$r$ vector bundle $E \to X$ corresponds to a locally free sheaf $\mathcal{E}$ via $E = \text{Spec}(\text{Sym}^\bullet \mathcal{E}^\vee)$
- [ ] Invertible sheaf (line bundle): locally free of rank 1
- [ ] Operations: $\mathcal{F} \otimes \mathcal{G}$, $\mathcal{H}om(\mathcal{F}, \mathcal{G})$, dual $\mathcal{F}^\vee$
- [ ] Pullback $f^* \mathcal{F}$ and pushforward $f_* \mathcal{F}$ of quasi-coherent sheaves

**Reading:**

- [ ] Hartshorne §II.5 (continued) *(~3 hrs)*

**Problems:**

- [ ] Hartshorne II.5.8, II.5.12, II.5.14, II.5.16, II.5.17, II.5.18
- [ ] Show $\mathcal{O}(n) \otimes \mathcal{O}(m) \cong \mathcal{O}(n+m)$ on $\mathbb{P}^1$

---

### Week 32 — Invertible Sheaves and the Picard Group

**Concepts to understand:**

- [ ] Picard group $\text{Pic}(X)$: invertible sheaves on $X$ up to isomorphism, with tensor product
- [ ] $\text{Pic}(\mathbb{P}^n_k) \cong \mathbb{Z}$: generated by $\mathcal{O}(1)$; degree map
- [ ] $\text{Pic}(\mathbb{P}^1 \times \mathbb{P}^1) \cong \mathbb{Z}^2$: generated by the pullbacks of $\mathcal{O}(1)$ along the two projections
- [ ] The exponential sequence (over $\mathbb{C}$): $0 \to \mathbb{Z} \to \mathcal{O} \to \mathcal{O}^* \to 0$ gives $\text{Pic}(X) \hookrightarrow H^2(X, \mathbb{Z})$
- [ ] Global sections functor $H^0(X, \mathcal{L})$: dimensions to be computed via RR in Phase III

**Reading:**

- [ ] Hartshorne §II.6 (Weil divisors, Cartier divisors, Pic) *(~3 hrs)*

**Problems:**

- [ ] Hartshorne II.6.1, II.6.2, II.6.4, II.6.5, II.6.6, II.6.11

> **Milestone:** Will Fisher's qual asked: "What is $\text{Pic}(\mathbb{P}^n)$? What is $\text{Pic}(\mathbb{P}^1 \times \mathbb{P}^1)$?" You should now be able to answer these — and explain what "degree" means in terms of line bundles and pullbacks.

---

### Week 33 — Weil and Cartier Divisors

**Concepts to understand:**

- [ ] Weil divisor on a normal scheme: formal $\mathbb{Z}$-linear combination of codimension-1 irreducible closed subsets
- [ ] Cartier divisor: a global section of $\mathcal{K}^*/\mathcal{O}^*$ — locally a rational function up to regular invertible function
- [ ] Comparison map: $\text{CaCl}(X) \to \text{Cl}(X)$ (Cartier to Weil class groups); isomorphism when $X$ is smooth
- [ ] On a singular variety: Weil divisors can fail to be Cartier (the ruling on the quadric cone is the standard example)
- [ ] $\text{Pic}(X) \cong \text{CaCl}(X)$: invertible sheaves and Cartier divisors are the same thing

**Reading:**

- [ ] Hartshorne §II.6 (continued) *(~3 hrs)*

**Problems:**

- [ ] Hartshorne II.6.3, II.6.7, II.6.8, II.6.9, II.6.10
- [ ] Show the ruling on the quadric cone $V(xy - z^2) \subset \mathbb{A}^3$ is a Weil but not Cartier divisor

---

### Week 34 — Maps to Projective Space

**Concepts to understand:**

- [ ] Theorem: maps $X \to \mathbb{P}^n_A$ over $\text{Spec}(A)$ correspond to: a line bundle $\mathcal{L}$ on $X$ together with $n+1$ global sections generating $\mathcal{L}$
- [ ] A line bundle $\mathcal{L}$ is globally generated (base-point-free) iff the corresponding map to projective space is a morphism (not just rational)
- [ ] Very ample line bundle: the corresponding map is a closed immersion
- [ ] Ample line bundle: some tensor power is very ample

**Reading:**

- [ ] Hartshorne §II.7 (intro), §II.5 (global generation) *(~3 hrs)*

**Problems:**

- [ ] Hartshorne II.7.1, II.7.2, II.7.3 (blowing up)
- [ ] Show the Veronese map $\nu_d: \mathbb{P}^n \to \mathbb{P}^N$ corresponds to $\mathcal{O}(d)$

---

### Week 35 — Kähler Differentials and the Cotangent Sheaf

**Concepts to understand:**

- [ ] Kähler differentials $\Omega_{B/A}$: the $B$-module generated by symbols $db$ for $b \in B$, subject to Leibniz rule
- [ ] Sheaf of differentials $\Omega_{X/S}$: the quasi-coherent sheaf associated to the diagonal $\Delta: X \to X \times_S X$
- [ ] For smooth varieties over $k$: $\Omega_{X/k}$ is locally free of rank $\dim X$ (the cotangent bundle)
- [ ] Geometric avatar: the stalk $\Omega_{X/k,P}$ is the cotangent space $\mathfrak{m}_P/\mathfrak{m}_P^2$ from Phase I
- [ ] Conormal sequence: $I/I^2 \to \Omega_{X/S}|_Y \to \Omega_{Y/S} \to 0$ for a closed subscheme $Y \hookrightarrow X$

**Reading:**

- [ ] Hartshorne §II.8 *(~3 hrs)*
- [ ] Vakil Ch 21 §21.1–21.3 *(~1.5 hrs)*

**Problems:**

- [ ] Hartshorne II.8.1, II.8.2, II.8.3, II.8.6, II.8.8
- [ ] Compute $\Omega_{\mathbb{P}^1/k}$ and show $\omega_{\mathbb{P}^1} = \mathcal{O}(-2)$

---

### Week 36 — Smooth Morphisms and the Jacobian Criterion

**Concepts to understand:**

- [ ] Smooth morphism $f: X \to Y$ of relative dimension $n$: $\Omega_{X/Y}$ is locally free of rank $n$, and $f$ is flat
- [ ] Geometric avatar: smooth $\Leftrightarrow$ the fibers are smooth varieties of the expected dimension
- [ ] Jacobian criterion for smoothness: $X = V(f_1, \ldots, f_r) \subset \mathbb{A}^n$ is smooth at $P$ iff the Jacobian matrix has rank $n - \dim X$ at $P$
- [ ] Relative tangent bundle: $T_{X/Y} = \mathcal{H}om(\Omega_{X/Y}, \mathcal{O}_X)$
- [ ] Étale morphisms: smooth of relative dimension 0; the algebraic analogue of local isomorphisms

**Reading:**

- [ ] Hartshorne §II.8 (continued, smooth) *(~2 hrs)*
- [ ] Vakil Ch 21 §21.4–21.5 *(~2 hrs)*

**Problems:**

- [ ] Hartshorne II.8.9, II.8.10
- [ ] Show a morphism of smooth curves is étale iff it is unramified

---

### Week 37 — Blowing Up

**Concepts to understand:**

- [ ] Blow-up of $\text{Spec}(A)$ along ideal $I$: $\text{Bl}_I X = \text{Proj}(\bigoplus_{n \geq 0} I^n)$
- [ ] The exceptional divisor: $E = \text{Proj}(\bigoplus_{n \geq 0} I^n/I^{n+1}) \hookrightarrow \text{Bl}_I X$; $E \cong \mathbb{P}(I/I^2)$ when $I/I^2$ is locally free
- [ ] The blow-up map $\pi: \text{Bl}_I X \to X$ is proper and an isomorphism away from $V(I)$
- [ ] The strict transform of a closed subscheme $Y$: the closure of $\pi^{-1}(Y \setminus V(I))$
- [ ] Universal property: blow-up is the initial scheme over $X$ in which $I \cdot \mathcal{O}$ is an invertible sheaf

**Reading:**

- [ ] Hartshorne §II.7 (blowing up) *(~3 hrs)*

**Problems:**

- [ ] Hartshorne II.7.3, II.7.4, II.7.5, II.7.9
- [ ] Re-derive the blow-up of $\mathbb{A}^2$ at the origin from the scheme-theoretic definition. Identify the exceptional divisor as $\mathbb{P}^1$. Verify this matches the classical construction from Week 12.

> **Phase II Milestone:** Open Ritvik's qual transcript. The questions on projective morphisms, blow-ups, Picard groups, and fiber products should now be fully followable. Open Will Fisher's transcript — the questions on morphism types (finite, proper, separated, finite-type) and divisors should be accessible.

---

### Weeks 38–42 — Phase II Consolidation and Qual Practice

Spend these five weeks working through Hartshorne Chapter I (algebraic varieties) and revisiting the weakest areas from Weeks 21–37. Use the Berkeley and Harvard qual problems as a diagnostic.

**Week 38:** Hartshorne Ch I — algebraic varieties as a review + contrast with schemes
- [ ] Hartshorne I.1 (affine varieties), I.2 (projective varieties), I.3 (morphisms)
- [ ] Exercises: I.1.1–1.5, I.2.1–2.7, I.3.1–3.5

**Week 39:** Hartshorne Ch I (continued)
- [ ] Hartshorne I.4 (rational maps), I.5 (nonsingular varieties), I.6 (nonsingular curves)
- [ ] Exercises: I.4.1–4.5, I.5.1–5.3, I.6.1–6.4

**Week 40:** Work Ritvik's qual transcript problems
- [ ] Projective morphisms and blowing up (Olsson questions)
- [ ] Primary decomposition of monomial ideals (Eisenbud questions)
- [ ] Noether normalization (rational quartic example)

**Week 41:** Work Will Fisher's qual transcript problems
- [ ] Morphism types: hyperbola projection (Gaetz questions)
- [ ] Divisors, Pic, comparison map (Teleman questions)
- [ ] Barr-Beck (skip — category theory not in scope)

**Week 42:** Work Harvard qual collection
- [ ] All problems tagged "schemes," "morphisms," "divisors," "Pic"
- [ ] Identify any Phase III topics needed (cohomology); flag for Phase III
