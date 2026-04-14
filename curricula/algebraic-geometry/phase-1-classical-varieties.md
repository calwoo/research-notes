# Phase I — Classical Varieties
*Weeks 1–20 · ~130 hrs*

> **Goal:** Build the geometric dictionary before scheme theory arrives. Every object defined here has a scheme-theoretic avatar in Phase II — Phase I is about seeing and internalizing the geometry first. By the end, affine and projective varieties, morphisms, smoothness, divisors, and the Riemann-Roch theorem should feel concrete and familiar.

**Primary text:** Shafarevich, *Basic Algebraic Geometry* Vol 1 (*Varieties in Projective Space*)
**Intuition companion:** Reid, *Undergraduate Algebraic Geometry* (skim for pictures; do not do exercises from both)
**Problem supplement:** Gathmann, *Algebraic Geometry* lecture notes (free PDF)

---

### Week 1 — Affine Varieties and the Nullstellensatz

**CA prerequisite:** Read A&M Ch 1 (rings and ideals) before starting.

**Concepts to understand:**
- [ ] Affine $n$-space $\mathbb{A}^n_k$ over a field $k$ and algebraic subsets $V(f_1, \ldots, f_r) \subset \mathbb{A}^n$
- [ ] The ideal $I(X)$ of a subset $X \subset \mathbb{A}^n$: functions vanishing on $X$
- [ ] Hilbert Basis Theorem: every ideal in $k[x_1, \ldots, x_n]$ is finitely generated
- [ ] Weak Nullstellensatz: if $I \subsetneq k[x_1, \ldots, x_n]$ and $k$ is algebraically closed, $V(I) \neq \emptyset$
- [ ] Strong Nullstellensatz: $I(V(J)) = \sqrt{J}$ — the algebraic↔geometric dictionary
- [ ] Radical ideals correspond bijectively (order-reversing) to algebraic subsets

**Reading:**
- [ ] Shafarevich Vol 1, §I.1 *(~3 hrs)*
- [ ] Reid, Ch 1–2 (skim for pictures and intuition) *(~1 hr)*

**Problems:**
- [ ] Shafarevich I.1 #1, 2, 3, 5, 7
- [ ] Show that $V(x^2 - y, x^3 - z) \subset \mathbb{A}^3$ is isomorphic to $\mathbb{A}^1$

> **Milestone:** State and explain (with geometric examples) why $I(V(J)) = \sqrt{J}$, and why the hypothesis that $k$ be algebraically closed is necessary.

---

### Week 2 — Regular Functions and Morphisms

**CA prerequisite:** A&M Ch 2 (modules, exact sequences).

**Concepts to understand:**
- [ ] Coordinate ring $k[X] = k[x_1, \ldots, x_n]/I(X)$ as the ring of regular functions on $X$
- [ ] A regular function $f: X \to k$ is an element of $k[X]$
- [ ] Morphism of affine varieties = ring homomorphism on coordinate rings (contravariantly)
- [ ] Isomorphism of varieties; the category of affine varieties is opposite to the category of finitely generated reduced $k$-algebras
- [ ] Dominant morphisms; the pullback $f^*: k[Y] \to k[X]$

**Reading:**
- [ ] Shafarevich Vol 1, §I.2 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich I.2 #1, 2, 3, 5, 6
- [ ] Show the map $t \mapsto (t^2, t^3)$ gives an isomorphism $\mathbb{A}^1 \cong V(y^2 - x^3)$ as topological spaces but NOT as varieties. What is the coordinate ring of the cusp?

---

### Week 3 — The Zariski Topology and Irreducibility

**CA prerequisite:** A&M Ch 3 (localization) — read §3.1–3.4 this week.

**Concepts to understand:**
- [ ] Zariski topology on $\mathbb{A}^n$: closed sets are algebraic subsets
- [ ] Irreducible topological spaces: not a union of two proper closed subsets
- [ ] An algebraic set $X$ is irreducible iff $I(X)$ is prime
- [ ] Decomposition into irreducible components (corresponds to primary decomposition)
- [ ] Noetherian property: every descending chain of closed sets stabilizes
- [ ] Function field $k(X)$ of an irreducible variety: fraction field of $k[X]$

**Reading:**
- [ ] Shafarevich Vol 1, §I.2 (continued), §I.3 intro *(~3 hrs)*
- [ ] A&M Ch 4, §4.1–4.5 (primary decomposition — read alongside for the algebraic picture) *(~2 hrs)*

**Problems:**
- [ ] Shafarevich I.2 #4, 7, 8
- [ ] Show $V(xy) \subset \mathbb{A}^2$ is reducible and find its irreducible components
- [ ] Find the primary decomposition of $(xy, x^2) \subset k[x,y]$ and interpret geometrically

---

### Week 4 — Rational Functions and the Local Ring at a Point

**Concepts to understand:**
- [ ] Rational function on $X$: equivalence class of pairs $(U, f)$ with $f$ regular on open $U$
- [ ] Local ring $\mathcal{O}_{X,P}$: rational functions defined at $P$ — this is the stalk of the structure sheaf
- [ ] Maximal ideal $\mathfrak{m}_P \subset \mathcal{O}_{X,P}$: functions vanishing at $P$
- [ ] $\mathcal{O}_{X,P} \cong k[X]_{\mathfrak{m}_P}$ (localization at the maximal ideal of $P$)
- [ ] Regular functions on $X$ embed into $\mathcal{O}_{X,P}$ for every $P$

**Reading:**
- [ ] Shafarevich Vol 1, §I.3 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich I.3 #1, 2, 4, 5
- [ ] For $X = V(y - x^2) \subset \mathbb{A}^2$, compute $\mathcal{O}_{X,(0,0)}$ explicitly

> **Geometric insight:** $\mathcal{O}_{X,P}$ is the algebraic avatar of "functions on an infinitesimally small neighborhood of $P$." When you later define the structure sheaf of a scheme, its stalks will be exactly these local rings — so this object is already familiar.

---

### Week 5 — Projective Space and Projective Varieties

**CA prerequisite:** Skim A&M Ch 1 discussion of graded rings (not in A&M explicitly — use Shafarevich §I.4 or Reid Ch 4 for this).

**Concepts to understand:**
- [ ] Projective $n$-space $\mathbb{P}^n_k$: lines through origin in $\mathbb{A}^{n+1}$; homogeneous coordinates $[x_0 : \cdots : x_n]$
- [ ] Projective algebraic set $V_+(F_1, \ldots, F_r)$: common zeros of homogeneous polynomials
- [ ] Homogeneous ideal $I_+(X)$: generated by homogeneous polynomials vanishing on $X$
- [ ] Homogeneous coordinate ring $S(X) = k[x_0, \ldots, x_n]/I_+(X)$: graded ring, NOT the ring of functions
- [ ] Standard affine cover: $\mathbb{P}^n = U_0 \cup \cdots \cup U_n$ with $U_i = \{x_i \neq 0\} \cong \mathbb{A}^n$

**Reading:**
- [ ] Shafarevich Vol 1, §I.4 *(~3.5 hrs)*
- [ ] Reid, Ch 4 (for alternative motivation) *(~1 hr)*

**Problems:**
- [ ] Shafarevich I.4 #1, 2, 4, 5, 7, 8
- [ ] Show the twisted cubic $C = \{[t^3 : t^2 s : ts^2 : s^3]\} \subset \mathbb{P}^3$ is a smooth projective variety. Find its ideal.

---

### Week 6 — Projective Varieties: Maps and Products

**Concepts to understand:**
- [ ] Regular maps between projective varieties
- [ ] The Veronese embedding $\nu_d: \mathbb{P}^n \hookrightarrow \mathbb{P}^N$ and its image
- [ ] The Segre embedding $\sigma: \mathbb{P}^m \times \mathbb{P}^n \hookrightarrow \mathbb{P}^{(m+1)(n+1)-1}$ — makes products projective
- [ ] $\mathbb{P}^1 \times \mathbb{P}^1$ as a quadric surface in $\mathbb{P}^3$ via Segre
- [ ] Quasiprojective varieties: open subsets of projective varieties; the correct general notion

**Reading:**
- [ ] Shafarevich Vol 1, §I.5–I.6 (quasiprojective varieties, products) *(~3 hrs)*

**Problems:**
- [ ] Shafarevich I.5 #1, 2, 3, 4
- [ ] Shafarevich I.6 #1, 2
- [ ] Show $\mathbb{P}^1 \times \mathbb{P}^1 \cong V(xw - yz) \subset \mathbb{P}^3$ via Segre. Compute the class of the diagonal.

> **Milestone:** Compute $\text{Pic}(\mathbb{P}^1)$ and $\text{Pic}(\mathbb{P}^1 \times \mathbb{P}^1)$ using divisors (you will rederive these via Picard groups in Phase II).

---

### Week 7 — Rational Maps and Birational Equivalence

**Concepts to understand:**
- [ ] Rational map $f: X \dashrightarrow Y$: defined on a dense open subset
- [ ] Domain of definition: the largest open set on which $f$ extends to a regular map
- [ ] Birational equivalence: rational maps in both directions that are inverse on dense opens
- [ ] $\mathbb{P}^n$ and $\mathbb{A}^n$ are NOT isomorphic but ARE birationally equivalent
- [ ] The function field $k(X)$ is a birational invariant

**Reading:**
- [ ] Shafarevich Vol 1, §I.3 (rational maps), §I.6 (birational maps) *(~3 hrs)*

**Problems:**
- [ ] Shafarevich I.6 #3, 4, 5, 6
- [ ] Show that the projection $\pi: \mathbb{P}^2 \dashrightarrow \mathbb{P}^1$ from a point is a rational map. Where is it not defined?

---

### Week 8 — Dimension of Varieties

**CA prerequisite:** A&M Ch 8–9 (Krull dimension, going-up, going-down).

**Concepts to understand:**
- [ ] Dimension of an affine variety $X$: Krull dimension of $k[X]$, equivalently transcendence degree of $k(X)/k$
- [ ] Theorem: $\dim X = \text{trdeg}_k k(X)$
- [ ] $\dim \mathbb{A}^n = \dim \mathbb{P}^n = n$
- [ ] Fiber dimension theorem: if $f: X \to Y$ is dominant, then $\dim f^{-1}(y) \geq \dim X - \dim Y$ for all $y \in \overline{f(X)}$, with equality on a dense open
- [ ] Hypersurfaces have codimension 1; the principal ideal theorem

**Reading:**
- [ ] Shafarevich Vol 1, §I.6 (dimension) *(~2 hrs)*
- [ ] A&M Ch 8–9 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich I.6 #7, 8
- [ ] A&M 8.1, 8.3, 9.1, 9.5
- [ ] Show $\dim k[x_1, \ldots, x_n] = n$ using the chain $(0) \subset (x_1) \subset (x_1, x_2) \subset \cdots$

---

### Week 9 — Tangent Spaces and Smoothness

**CA prerequisite:** A&M Ch 11 (discrete valuation rings) — read §11.1 this week.

**Concepts to understand:**
- [ ] Tangent space $T_{X,P}$ at a point: $(\mathfrak{m}_P/\mathfrak{m}_P^2)^\vee$ — the dual of the cotangent space
- [ ] Jacobian criterion: if $X = V(f_1, \ldots, f_m) \subset \mathbb{A}^n$, then $T_{X,P}$ is the kernel of the Jacobian matrix $(\partial f_i/\partial x_j)(P)$
- [ ] A point $P \in X$ is smooth (nonsingular) iff $\dim T_{X,P} = \dim X$
- [ ] An irreducible variety is smooth iff its smooth locus is open and dense
- [ ] Smooth $\Rightarrow$ local ring is regular; $\dim \mathcal{O}_{X,P} = \dim X$ at smooth points

**Reading:**
- [ ] Shafarevich Vol 1, §II.1–II.2 *(~3.5 hrs)*

**Problems:**
- [ ] Shafarevich II.1 #1, 2, 3, 4, 5
- [ ] Find the singular locus of $V(y^2 - x^2(x+1)) \subset \mathbb{A}^2$ (nodal cubic)
- [ ] Find the singular locus of $V(y^2 - x^3) \subset \mathbb{A}^2$ (cuspidal cubic)

---

### Week 10 — Local Structure of Morphisms

**Concepts to understand:**
- [ ] Local ring map induced by a morphism $f: X \to Y$ at a point: $\mathcal{O}_{Y, f(P)} \to \mathcal{O}_{X,P}$
- [ ] Finite morphisms: the ring map $k[Y] \to k[X]$ makes $k[X]$ a finite $k[Y]$-module
- [ ] Ramification: when a finite map to a smooth curve is not étale at a point
- [ ] Fiber dimension theorem revisited: generic smoothness and the structure of generic fibers
- [ ] Chevalley's theorem: the image of a constructible set is constructible

**Reading:**
- [ ] Shafarevich Vol 1, §II.3 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich II.3 #1, 2, 3, 4

---

### Week 11 — Normalization

**CA prerequisite:** A&M Ch 5 (integral dependence, Noether normalization) — read this week.

**Concepts to understand:**
- [ ] Normal domain: integrally closed in its fraction field
- [ ] Normalization $\tilde{X}$ of a variety $X$: the variety corresponding to the integral closure of $k[X]$ in $k(X)$
- [ ] Normalization is a birational finite morphism $\nu: \tilde{X} \to X$
- [ ] Noether normalization: every affine variety $X$ of dimension $d$ admits a finite surjective map $X \to \mathbb{A}^d$
- [ ] A curve is normal iff it is smooth (in dimension 1: regular = normal)

**Reading:**
- [ ] Shafarevich Vol 1, §II.4 *(~2.5 hrs)*
- [ ] A&M Ch 5 *(~2.5 hrs)*

**Problems:**
- [ ] Shafarevich II.4 #1, 2, 3
- [ ] A&M 5.1, 5.4, 5.16, 5.22
- [ ] Compute the normalization of the cuspidal cubic $V(y^2 - x^3) \subset \mathbb{A}^2$

> **Milestone:** Explain geometrically why normalization "resolves" the cusp: the integral closure separates branches that pass through a singular point.

---

### Week 12 — Resolution of Curve Singularities

**Concepts to understand:**
- [ ] Blowing up a point in $\mathbb{A}^2$: $\text{Bl}_0 \mathbb{A}^2 \subset \mathbb{A}^2 \times \mathbb{P}^1$; the exceptional divisor $E \cong \mathbb{P}^1$
- [ ] Strict transform of a curve $C$ under blowing up: the closure of $\pi^{-1}(C \setminus \{0\})$
- [ ] Resolution of the node $V(y^2 - x^2(x+1))$ and the cusp $V(y^2 - x^3)$ by successive blowing up
- [ ] Every curve over a perfect field has a smooth projective model (resolution of singularities in dimension 1)
- [ ] The smooth projective model is unique up to isomorphism

**Reading:**
- [ ] Shafarevich Vol 1, §II.4–II.5 *(~3 hrs)*
- [ ] Reid, Ch 7 (§7.1–7.2 on blowing up, for extra intuition) *(~1 hr)*

**Problems:**
- [ ] Shafarevich II.5 #1, 2, 3, 4
- [ ] Resolve the $A_2$ singularity $V(y^2 - x^3) \subset \mathbb{A}^2$ by two blow-ups. Describe the exceptional locus.

---

### Week 13 — Discrete Valuation Rings and Curves

**CA prerequisite:** A&M Ch 9 (DVRs) and Ch 11 (completions) — read Ch 9 this week.

**Concepts to understand:**
- [ ] Discrete valuation ring (DVR): a PID with a unique nonzero prime ideal; has a uniformizer $t$ with $\mathfrak{m} = (t)$
- [ ] Valuation $v: k(X)^\times \to \mathbb{Z}$: the order of vanishing/pole of a rational function at a smooth point of a curve
- [ ] Smooth points of a curve correspond bijectively to DVRs inside $k(X)$
- [ ] Over a smooth projective curve: every rational function has only finitely many zeros and poles

**Reading:**
- [ ] Shafarevich Vol 1, §III.1 (intro), §II.5 (continuation) *(~2 hrs)*
- [ ] A&M Ch 9 *(~2.5 hrs)*

**Problems:**
- [ ] A&M 9.1, 9.2, 9.3
- [ ] Show the local ring of $\mathbb{P}^1$ at any point is a DVR, and identify the uniformizer

---

### Week 14 — Divisors on Curves

**Concepts to understand:**
- [ ] Divisor on a smooth projective curve $X$: $D = \sum_{P \in X} n_P [P]$ with $n_P \in \mathbb{Z}$, finitely many nonzero
- [ ] Degree of a divisor: $\deg D = \sum n_P$
- [ ] Principal divisor $(f)$ of a rational function $f \in k(X)^\times$: zeros minus poles with multiplicity
- [ ] Divisor class group (Picard group) $\text{Pic}(X) = \text{Div}(X)/\text{PDiv}(X)$
- [ ] $\text{Pic}^0(X)$ = degree-0 part; $\text{Pic}(X) \cong \mathbb{Z} \oplus \text{Pic}^0(X)$ for connected curves

**Reading:**
- [ ] Shafarevich Vol 1, §III.1 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich III.1 #1, 2, 3, 4, 5
- [ ] Show $\text{Pic}(\mathbb{A}^1) = 0$ and $\text{Pic}(\mathbb{P}^1) \cong \mathbb{Z}$

---

### Week 15 — Linear Systems and Maps to Projective Space

**Concepts to understand:**
- [ ] Divisor of a section: $\text{div}(s)$ for $s \in H^0(X, \mathcal{O}(D))$
- [ ] Linear system $|D|$: the projective space of effective divisors linearly equivalent to $D$
- [ ] Complete linear system and the corresponding map $\phi_D: X \dashrightarrow \mathbb{P}(H^0(X, \mathcal{O}(D)))^\vee$
- [ ] Base locus: points where all sections vanish; $\phi_D$ is a morphism iff base locus is empty
- [ ] A divisor $D$ is very ample iff $\phi_D$ is a closed immersion

**Reading:**
- [ ] Shafarevich Vol 1, §III.1 (continued) *(~2 hrs)*
- [ ] Reid, Ch 9 §9.1–9.3 *(~2 hrs)*

**Problems:**
- [ ] Show that $\mathcal{O}(1)$ on $\mathbb{P}^1$ corresponds to the identity map $\mathbb{P}^1 \to \mathbb{P}^1$
- [ ] Find all effective divisors linearly equivalent to $2[P]$ on a smooth conic $C \subset \mathbb{P}^2$

---

### Week 16 — Differential Forms on Curves

**Concepts to understand:**
- [ ] Module of Kähler differentials $\Omega_{k[X]/k}$ for an affine variety; the sheaf $\Omega_{X/k}$
- [ ] For a smooth curve: $\Omega_{X/k}$ is a line bundle (rank-1 locally free sheaf), called the canonical bundle $\omega_X$
- [ ] Differential form $\omega \in H^0(X, \omega_X)$: a "1-form" on the curve
- [ ] Divisor of a differential form: $(\omega) = \sum v_P(\omega) [P]$ where $v_P$ is the order of vanishing in local coordinates
- [ ] Canonical class $K_X$: the divisor class of any nonzero differential form

**Reading:**
- [ ] Shafarevich Vol 1, §III.3 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich III.3 #1, 2, 3
- [ ] On $\mathbb{P}^1$ with coordinate $t$, compute $\text{div}(dt)$. What is $\deg K_{\mathbb{P}^1}$?

---

### Week 17 — Genus and the Riemann-Roch Theorem (Classical)

**Concepts to understand:**
- [ ] Geometric genus $g$ of a smooth projective curve: $g = \dim H^0(X, \omega_X)$ (holomorphic differentials)
- [ ] Riemann-Roch theorem: $\ell(D) - \ell(K - D) = \deg D + 1 - g$
  - where $\ell(D) = \dim H^0(X, \mathcal{O}(D))$
- [ ] Special cases: $\ell(K) = g$, $\deg K = 2g - 2$
- [ ] Consequence: if $\deg D > 2g - 2$, then $\ell(D) = \deg D + 1 - g$
- [ ] Genus formula for a smooth plane curve of degree $d$: $g = \binom{d-1}{2}$

**Reading:**
- [ ] Shafarevich Vol 1, §III.4 *(~3 hrs)*
- [ ] Reid, Ch 9 §9.4–9.7 *(~1 hr)*

**Problems:**
- [ ] Shafarevich III.4 #1, 2, 3, 4
- [ ] Compute $g$ for a smooth quartic in $\mathbb{P}^2$. Use RR to find $\ell(K)$.
- [ ] Show: a smooth curve of genus 0 over $\bar{k}$ is isomorphic to $\mathbb{P}^1$

> **Milestone:** Work through 5 problems from the Harvard qualifying exam collection that involve Riemann-Roch, Hurwitz, or divisors.

---

### Week 18 — Hurwitz's Theorem

**Concepts to understand:**
- [ ] A finite morphism of smooth projective curves $f: X \to Y$ of degree $n$
- [ ] Ramification point: where the local degree of $f$ is $> 1$; the ramification index $e_P$
- [ ] Ramification divisor: $R = \sum_P (e_P - 1)[P]$ on $X$
- [ ] Hurwitz's theorem: $2g(X) - 2 = n(2g(Y) - 2) + \deg R$ (for separable $f$)
- [ ] Purely inseparable maps in characteristic $p$: Hurwitz gives $g(X) = g(Y)$

**Reading:**
- [ ] Shafarevich Vol 1, §III.4 (Hurwitz) *(~2 hrs)*
- [ ] Fulton, *Algebraic Curves*, Ch 7 (for Riemann-Hurwitz with proof) *(~2 hrs)*

**Problems:**
- [ ] Show every map $f: X \to \mathbb{P}^1$ of degree $n$ from a genus-$g$ curve has exactly $2g + 2n - 2$ branch points (over $\mathbb{C}$)
- [ ] Deduce the genus formula for a smooth plane curve from Hurwitz by projecting from a point
- [ ] Harvard qual: "Let's talk about Riemann-Hurwitz..." — work through the Ogus questions from the problem set

---

### Week 19 — Elliptic Curves: Classical Picture

**Concepts to understand:**
- [ ] An elliptic curve is a smooth projective curve of genus 1 with a marked point $O$
- [ ] Weierstrass form: $y^2 = x^3 + ax + b$ (char $\neq 2, 3$); smoothness iff discriminant $\Delta \neq 0$
- [ ] Group law: $P + Q + R = 0$ iff $P, Q, R$ are collinear; $O$ is the identity
- [ ] The $j$-invariant $j(E) = 1728 \cdot \frac{4a^3}{4a^3 + 27b^2}$: classifies elliptic curves over $\bar{k}$ up to isomorphism
- [ ] Over $\mathbb{C}$: $E \cong \mathbb{C}/\Lambda$ for a lattice $\Lambda = \mathbb{Z} + \mathbb{Z}\tau$; the group law is addition in $\mathbb{C}/\Lambda$

**Reading:**
- [ ] Shafarevich Vol 1, §III.3 (elliptic curves), §III.4 *(~2 hrs)*
- [ ] Silverman AEC, Ch 1 §1–3 (preview; you'll return to this in Phase IV) *(~2 hrs)*

**Problems:**
- [ ] Harvard qual: "What are the involutions of an elliptic curve over $\mathbb{C}$?" Work through the McMullen questions
- [ ] Show the group law is associative using Riemann-Roch (sketch): $\text{Pic}^0(E) \cong E$ as sets

---

### Week 20 — Hyperelliptic Curves and Review

**Concepts to understand:**
- [ ] Hyperelliptic curve: double cover of $\mathbb{P}^1$ branched at $2g + 2$ points (for genus $g$)
- [ ] Canonical map $\phi_K: X \to \mathbb{P}^{g-1}$: defined when $g \geq 2$, base-point-free when $X$ is not hyperelliptic
- [ ] Hyperelliptic curves of genus 2: always $y^2 = f(x)$ with $\deg f = 5$ or $6$
- [ ] Review: the algebraic↔geometric dictionary built in Phase I
- [ ] Every classical concept paired with its scheme-theoretic avatar (preview of Phase II)

**Reading:**
- [ ] Shafarevich Vol 1, §III.5 *(~2 hrs)*
- [ ] Review your Phase I notes; read the spec document for the Phase II translation table *(~3 hrs)*

**Problems:**
- [ ] Show a genus-2 curve is always hyperelliptic using the canonical map
- [ ] Work 3 more problems from the Harvard qual collection; identify which Phase II concepts they preview

> **Phase I Milestone:** You should now be able to: (1) define varieties, morphisms, divisors, and the Picard group from scratch; (2) state and apply Riemann-Roch and Hurwitz; (3) work problems from the Harvard qual involving divisors, Pic, and curves. Open a copy of Ritvik's qual transcript — the question on Hurwitz's theorem should now be followable end-to-end.
