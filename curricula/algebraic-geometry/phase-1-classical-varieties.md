# Phase I — Classical Varieties

*Weeks 1–20 · ~130 hrs*

> **Goal:** Build the geometric dictionary before scheme theory arrives. Every object defined here has a scheme-theoretic avatar in Phase II — Phase I is about seeing and internalizing the geometry first. By the end, affine and projective varieties, morphisms, smoothness, divisors, and the Riemann-Roch theorem should feel concrete and familiar.

**Weeks 1–6 primary:** Harvard Math 137 (*Algebraic Geometry*, Brooke Ullery), 24 lectures + 11 problem sets.
Lectures and problem sets: [https://people.math.harvard.edu/~bullery/math137/](https://people.math.harvard.edu/~bullery/math137/)
Primary text for Math 137: Fulton, *Algebraic Curves* (free at [http://www.math.lsa.umich.edu/~wfulton/CurveBook.pdf](http://www.math.lsa.umich.edu/~wfulton/CurveBook.pdf))

**Weeks 7–20 primary:** Shafarevich, *Basic Algebraic Geometry* Vol 1, Chapters II–III
**Intuition companion:** Reid, *Undergraduate Algebraic Geometry* (skim for pictures)

> **Why this structure:** Math 137 replaces Shafarevich §I entirely with a more exercise-dense, pedagogically sequenced treatment using Fulton. It adds two topics Phase I would otherwise lack — intersection numbers and Bézout's theorem — which pay dividends in Phase III. Shafarevich §II–III (smoothness through elliptic curves) has no equivalent in Math 137 and is covered in Weeks 7–20.

---

## Weeks 1–6 — Harvard Math 137

The 24 lectures and 11 problem sets map naturally to a 6-week block at ~5–8 hrs/week. Work through the lecture notes sequentially; the problem sets are the primary exercise source for this block.

**Lecture notes** (all PDFs at `https://people.math.harvard.edu/~bullery/math137/`):

| Lec | Title | Phase I week analogue |
|-----|-------|-----------------------|
| 1 | [What is algebraic geometry?](https://people.math.harvard.edu/~bullery/math137/Section%201_%20What%20is%20algebraic%20geometry.pdf) | orientation |
| 2 | [Algebraic sets](https://people.math.harvard.edu/~bullery/math137/Section%202_%20Algebraic%20sets.pdf) | Wk 1 |
| 3 | [The ideal of a subset of affine space](https://people.math.harvard.edu/~bullery/math137/Section%203_%20The%20ideal%20of%20a%20subset%20of%20affine%20space.pdf) | Wk 1 |
| 4 | [Irreducibility and the Hilbert Basis Theorem](https://people.math.harvard.edu/~bullery/math137/Section%204_%20Irreducibility%20and%20the%20Hilbert%20Basis%20Theorem.pdf) | Wk 3 |
| 5 | [Hilbert's Nullstellensatz](https://people.math.harvard.edu/~bullery/math137/Section%205_%20Hilberts%20Nullstellensatz.pdf) | Wk 1 |
| 6 | [Algebra detour](https://people.math.harvard.edu/~bullery/math137/Section%206_%20Algebra%20detour.pdf) | CA supplement |
| 7 | [Affine varieties and coordinate rings](https://people.math.harvard.edu/~bullery/math137/Section%207_%20Affine%20varieties%20and%20coordinate%20rings.pdf) | Wk 2 |
| 8 | [Regular maps](https://people.math.harvard.edu/~bullery/math137/Section%208_%20Regular%20maps.pdf) | Wk 2 |
| 9 | [Rational functions and local rings](https://people.math.harvard.edu/~bullery/math137/Section%209_%20Rational%20functions%20and%20local%20rings.pdf) | Wk 4 |
| 10 | [Affine plane curves](https://people.math.harvard.edu/~bullery/math137/Section%2010_%20Affine%20plane%20curves.pdf) | Wk 1–2 |
| 11 | [Discrete valuation rings and multiplicities](https://people.math.harvard.edu/~bullery/math137/Section%2011_%20Discrete%20valuation%20rings%20and%20multiplicities.pdf) | Wk 13 |
| 12 | [Intersection numbers](https://people.math.harvard.edu/~bullery/math137/Section%2012_%20Intersection%20numbers.pdf) | *bonus — not in Shafarevich Phase I* |
| 13 | [Projective space](https://people.math.harvard.edu/~bullery/math137/Section%2013_%20Projective%20space.pdf) | Wk 5 |
| 14 | [Projective algebraic sets](https://people.math.harvard.edu/~bullery/math137/Section%2014_%20Projective%20algebraic%20sets.pdf) | Wk 5 |
| 15 | [Homogeneous coordinate rings and rational functions](https://people.math.harvard.edu/~bullery/math137/Section%2015_%20Homogeneous%20coordinate%20rings%20and%20rational%20functions.pdf) | Wk 5 |
| 16 | [Affine and projective varieties](https://people.math.harvard.edu/~bullery/math137/Section%2016_%20Affine%20and%20projective%20varieties.pdf) | Wk 5–6 |
| 17 | [Morphisms of projective varieties](https://people.math.harvard.edu/~bullery/math137/Section%2017_%20Morphism%20of%20projective%20varieties.pdf) | Wk 6 |
| 18 | [Projective plane curves](https://people.math.harvard.edu/~bullery/math137/Section%2018_%20Projective%20plane%20curves.pdf) | Wk 6 |
| 19 | [Linear systems of curves](https://people.math.harvard.edu/~bullery/math137/Section%2019_%20Linear%20systems%20of%20curves.pdf) | Wk 15 |
| 20 | [Bézout's Theorem](https://people.math.harvard.edu/~bullery/math137/Section%2020_%20Bezouts%20Theorem.pdf) | *bonus — not in Shafarevich Phase I* |
| 21 | [Abstract varieties](https://people.math.harvard.edu/~bullery/math137/Section%2021_%20Abstract%20varieties.pdf) | Wk 6 |
| 22 | [Rational maps and dimension](https://people.math.harvard.edu/~bullery/math137/Section%2022_%20Rational%20maps%20and%20dimension.pdf) | Wks 7–8 |
| 23 | [Rational maps of curves](https://people.math.harvard.edu/~bullery/math137/Section%2023_%20Rational%20maps%20of%20curves.pdf) | Wk 7 |
| 24 | [Blowing up a point in the plane](https://people.math.harvard.edu/~bullery/math137/Section%2024_%20Blowing%20up%20a%20point%20in%20the%20plane.pdf) | Wk 12 |

### Week 1 — Math 137 Lectures 1–8

**Concepts to understand:**
- [ ] What algebraic geometry studies: solution sets of polynomial equations as geometric objects
- [ ] Algebraic sets $V(f_1, \ldots, f_r) \subset \mathbb{A}^n$; the ideal $I(X)$ of a subset
- [ ] Hilbert Basis Theorem: every ideal in $k[x_1, \ldots, x_n]$ is finitely generated
- [ ] Weak and strong Nullstellensatz: $I(V(J)) = \sqrt{J}$ over algebraically closed $k$
- [ ] Irreducibility; algebraic set is irreducible iff $I(X)$ is prime
- [ ] Coordinate ring $k[X] = k[x_1, \ldots, x_n]/I(X)$; regular functions; morphisms
- [ ] Algebra–geometry duality: affine varieties $\leftrightarrow$ finitely generated reduced $k$-algebras

**Lectures:** 1–8
**Problem sets:** PS1 (due Feb 5), PS2 (due Feb 12)

---

### Week 2 — Math 137 Lectures 9–12

**Concepts to understand:**
- [ ] Rational functions and local rings $\mathcal{O}_{X,P}$: functions defined near $P$
- [ ] Affine plane curves: tangent lines, multiplicity of a point, branches
- [ ] Discrete valuation rings (DVRs): uniformizers, valuation $v_P(f)$ = order of vanishing
- [ ] Intersection number $(C \cdot D)_P$ at a point: via $\dim_k \mathcal{O}_P/(f, g)$
- [ ] Properties of intersection numbers: symmetry, additivity, invariance under linear equivalence

**Lectures:** 9–12
**Problem sets:** PS3 (due Feb 19)

> **Bonus from Math 137:** Intersection numbers (Lec 12) are Fulton's key tool — not covered in Shafarevich at this stage. They foreshadow intersection theory in Phase III and will make Bézout's theorem feel inevitable.

---

### Week 3 — Math 137 Lectures 13–16

**Concepts to understand:**
- [ ] Projective $n$-space $\mathbb{P}^n_k$: homogeneous coordinates $[x_0 : \cdots : x_n]$
- [ ] Projective algebraic sets: common zeros of homogeneous polynomials
- [ ] Homogeneous coordinate ring $S(X)$: graded ring, NOT the ring of functions on $X$
- [ ] Standard affine cover: $\mathbb{P}^n = U_0 \cup \cdots \cup U_n$ with $U_i \cong \mathbb{A}^n$
- [ ] Affine and projective varieties as a unified notion; quasiprojective varieties

**Lectures:** 13–16
**Problem sets:** PS4 (due Feb 26)

---

### Week 4 — Math 137 Lectures 17–20

**Concepts to understand:**
- [ ] Morphisms of projective varieties: defined by homogeneous polynomials of the same degree
- [ ] Projective plane curves: degree, tangent lines, singular points, intersection with lines
- [ ] Linear systems of plane curves of degree $d$: a projective space parameterizing curves
- [ ] Bézout's Theorem: two projective plane curves of degrees $d$ and $e$ with no common component meet in exactly $de$ points (counted with intersection multiplicity)

**Lectures:** 17–20
**Problem sets:** PS5 (due Mar 4), PS6 (due Mar 11)

> **Bonus from Math 137:** Bézout's theorem (Lec 20) with proof. This is used repeatedly in the Harvard qualifying exam collection.

---

### Week 5 — Math 137 Lectures 21–24

**Concepts to understand:**
- [ ] Abstract varieties: gluing affine pieces via transition maps; the correct intrinsic notion
- [ ] Rational maps $f: X \dashrightarrow Y$: defined on a dense open subset; domain of definition
- [ ] Birational equivalence: rational inverse in both directions; $k(X)$ is a birational invariant
- [ ] Dimension via transcendence degree: $\dim X = \text{trdeg}_k k(X)$
- [ ] Fiber dimension theorem: generic fiber has dimension $\dim X - \dim Y$
- [ ] Blowing up a point in $\mathbb{A}^2$: $\text{Bl}_0 \mathbb{A}^2 \subset \mathbb{A}^2 \times \mathbb{P}^1$; exceptional divisor $E \cong \mathbb{P}^1$

**Lectures:** 21–24
**Problem sets:** PS7 (due Mar 27), PS8 (due Apr 3)

---

### Week 6 — Math 137 Consolidation

Complete remaining problem sets and consolidate.

**Problem sets:** PS9 (due Apr 10), PS10 (due Apr 17), PS11 (due Apr 29)

**Consolidation checklist:**
- [ ] For every affine object (coordinate ring, regular map, rational function, local ring), identify its projective analogue
- [ ] Work through the Harvard qual collection: all problems tagged "affine variety," "projective variety," "morphism," "rational map"
- [ ] State Bézout's theorem and use it to compute intersection numbers for 3 explicit pairs of plane curves
- [ ] State and prove the Nullstellensatz from scratch without notes

> **Math 137 Milestone:** You should now be fluent in the language of classical algebraic geometry — affine and projective varieties, morphisms, rational maps, dimension, and intersection numbers. The bridge to Shafarevich §II (which opens with tangent spaces) requires only the language of local rings, which Math 137 covered in Lec 9.

---

## Weeks 7–20 — Shafarevich Vol 1, Chapters II–III

From here the primary text is Shafarevich, *Basic Algebraic Geometry* Vol 1. Chapters I of Shafarevich is now fully replaced by Math 137 — begin directly at Chapter II.

> **Note on overlap:** Math 137 covered DVRs (Lec 11) and blowing up (Lec 24). Weeks 12–13 below revisit these in greater depth; treat them as consolidation rather than new material.

---

### Week 7 — Tangent Spaces and Smoothness

**CA prerequisite:** A&M Ch 11 (discrete valuation rings) — read §11.1 this week.

**Concepts to understand:**
- [ ] Tangent space $T_{X,P}$ at a point: $(\mathfrak{m}_P/\mathfrak{m}_P^2)^\vee$ — dual of the cotangent space
- [ ] Jacobian criterion: if $X = V(f_1, \ldots, f_m) \subset \mathbb{A}^n$, then $T_{X,P} = \ker(\partial f_i/\partial x_j)(P)$
- [ ] A point $P \in X$ is smooth (nonsingular) iff $\dim T_{X,P} = \dim X$
- [ ] Smooth $\Rightarrow$ local ring is regular; regular local ring has $\dim_k \mathfrak{m}/\mathfrak{m}^2 = \dim \mathcal{O}_{X,P}$
- [ ] The smooth locus is open and dense in any irreducible variety

**Reading:**
- [ ] Shafarevich Vol 1, §II.1–II.2 *(~3.5 hrs)*

**Problems:**
- [ ] Shafarevich II.1 #1, 2, 3, 4, 5
- [ ] Find the singular locus of $V(y^2 - x^2(x+1)) \subset \mathbb{A}^2$ (nodal cubic)
- [ ] Find the singular locus of $V(y^2 - x^3) \subset \mathbb{A}^2$ (cuspidal cubic)

---

### Week 8 — Local Structure of Morphisms

**Concepts to understand:**
- [ ] Local ring map induced by $f: X \to Y$ at a point: $\mathcal{O}_{Y, f(P)} \to \mathcal{O}_{X,P}$
- [ ] Finite morphisms: $k[Y] \to k[X]$ makes $k[X]$ a finite $k[Y]$-module
- [ ] Ramification: when a finite map to a smooth curve is not étale at a point
- [ ] Fiber dimension theorem revisited: generic smoothness and structure of generic fibers
- [ ] Chevalley's theorem: the image of a constructible set is constructible

**Reading:**
- [ ] Shafarevich Vol 1, §II.3 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich II.3 #1, 2, 3, 4

---

### Week 9 — Normalization

**CA prerequisite:** A&M Ch 5 (integral dependence, Noether normalization) — read this week.

**Concepts to understand:**
- [ ] Normal domain: integrally closed in its fraction field
- [ ] Normalization $\tilde{X}$ of $X$: variety corresponding to the integral closure of $k[X]$ in $k(X)$
- [ ] Normalization is a birational finite morphism $\nu: \tilde{X} \to X$
- [ ] Noether normalization: every affine variety of dimension $d$ admits a finite surjective map to $\mathbb{A}^d$
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

### Week 10 — Resolution of Curve Singularities

*Math 137 Lec 24 introduced blowing up; this week goes further to resolution.*

**Concepts to understand:**
- [ ] Blowing up revisited: strict transform of a curve $C$ under blowing up at $P$
- [ ] Resolution of the node $V(y^2 - x^2(x+1))$ and the cusp $V(y^2 - x^3)$ by successive blowing up
- [ ] Every curve over a perfect field has a smooth projective model (resolution in dimension 1)
- [ ] The smooth projective model is unique up to isomorphism

**Reading:**
- [ ] Shafarevich Vol 1, §II.4–II.5 *(~3 hrs)*
- [ ] Reid, Ch 7 §7.1–7.2 *(~1 hr)*

**Problems:**
- [ ] Shafarevich II.5 #1, 2, 3, 4
- [ ] Resolve the $A_2$ singularity $V(y^2 - x^3)$ by two blow-ups. Describe the exceptional locus.

---

### Week 11 — Divisors on Curves

*Math 137 Lec 11 covered DVRs and valuations. This week builds the divisor theory on top of that.*

**Concepts to understand:**
- [ ] Divisor on a smooth projective curve $X$: $D = \sum_{P \in X} n_P [P]$ with finitely many nonzero $n_P$
- [ ] Degree of a divisor: $\deg D = \sum n_P$
- [ ] Principal divisor $(f)$ of $f \in k(X)^\times$: zeros minus poles with multiplicity (using the DVR valuation)
- [ ] Divisor class group (Picard group) $\text{Pic}(X) = \text{Div}(X)/\text{PDiv}(X)$
- [ ] $\text{Pic}^0(X)$ = degree-0 part; $\text{Pic}(X) \cong \mathbb{Z} \oplus \text{Pic}^0(X)$ for connected curves

**Reading:**
- [ ] Shafarevich Vol 1, §III.1 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich III.1 #1, 2, 3, 4, 5
- [ ] Show $\text{Pic}(\mathbb{A}^1) = 0$ and $\text{Pic}(\mathbb{P}^1) \cong \mathbb{Z}$

---

### Week 12 — Linear Systems and Maps to Projective Space

**Concepts to understand:**
- [ ] Linear system $|D|$: the projective space of effective divisors linearly equivalent to $D$
- [ ] Complete linear system and the map $\phi_D: X \dashrightarrow \mathbb{P}(H^0(X, \mathcal{O}(D)))^\vee$
- [ ] Base locus: points where all sections vanish; $\phi_D$ is a morphism iff base locus is empty
- [ ] A divisor $D$ is very ample iff $\phi_D$ is a closed immersion
- [ ] Connection to Math 137 Lec 19 (linear systems of plane curves): the same projective space, now intrinsically defined

**Reading:**
- [ ] Shafarevich Vol 1, §III.1 (continued) *(~2 hrs)*
- [ ] Reid, Ch 9 §9.1–9.3 *(~2 hrs)*

**Problems:**
- [ ] Show that $\mathcal{O}(1)$ on $\mathbb{P}^1$ corresponds to the identity map $\mathbb{P}^1 \to \mathbb{P}^1$
- [ ] Find all effective divisors linearly equivalent to $2[P]$ on a smooth conic $C \subset \mathbb{P}^2$

---

### Week 13 — Differential Forms on Curves

**Concepts to understand:**
- [ ] Module of Kähler differentials $\Omega_{k[X]/k}$; the sheaf $\Omega_{X/k}$
- [ ] For a smooth curve: $\Omega_{X/k}$ is a line bundle, the canonical bundle $\omega_X$
- [ ] Divisor of a differential form: $(\omega) = \sum v_P(\omega) [P]$ via the DVR valuation at each point
- [ ] Canonical class $K_X$: the divisor class of any nonzero differential form
- [ ] On $\mathbb{P}^1$: $K_{\mathbb{P}^1} \sim -2[P]$, so $\deg K_{\mathbb{P}^1} = -2$

**Reading:**
- [ ] Shafarevich Vol 1, §III.3 *(~3 hrs)*

**Problems:**
- [ ] Shafarevich III.3 #1, 2, 3
- [ ] On $\mathbb{P}^1$ with coordinate $t$, compute $\text{div}(dt)$. What is $\deg K_{\mathbb{P}^1}$?

---

### Week 14 — Genus and the Riemann-Roch Theorem

**Concepts to understand:**
- [ ] Geometric genus $g$: $g = \dim H^0(X, \omega_X)$ (holomorphic differentials)
- [ ] Riemann-Roch theorem: $\ell(D) - \ell(K - D) = \deg D + 1 - g$
- [ ] Special cases: $\ell(K) = g$, $\deg K = 2g - 2$
- [ ] For $\deg D > 2g - 2$: $\ell(D) = \deg D + 1 - g$ (no correction term)
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

### Week 15 — Hurwitz's Theorem

**Concepts to understand:**
- [ ] A finite morphism of smooth projective curves $f: X \to Y$ of degree $n$
- [ ] Ramification index $e_P$ at $P$; ramification divisor $R = \sum_P (e_P - 1)[P]$ on $X$
- [ ] Hurwitz's theorem: $2g(X) - 2 = n(2g(Y) - 2) + \deg R$ (for separable $f$)
- [ ] Application: every map $f: X \to \mathbb{P}^1$ of degree $n$ from a genus-$g$ curve has exactly $2g + 2n - 2$ branch points (over $\mathbb{C}$)
- [ ] Purely inseparable maps in characteristic $p$: $g(X) = g(Y)$

**Reading:**
- [ ] Shafarevich Vol 1, §III.4 (Hurwitz) *(~2 hrs)*
- [ ] Fulton, *Algebraic Curves*, Ch 7 *(~2 hrs)*

**Problems:**
- [ ] Deduce the genus formula for a smooth plane curve from Hurwitz by projecting from a point
- [ ] Harvard qual: "Let's talk about Riemann-Hurwitz..." — work through the Ogus questions from the problem set

---

### Week 16 — Elliptic Curves: Classical Picture

**Concepts to understand:**
- [ ] An elliptic curve is a smooth projective curve of genus 1 with a marked point $O$
- [ ] Weierstrass form: $y^2 = x^3 + ax + b$ (char $\neq 2, 3$); smoothness iff $\Delta \neq 0$
- [ ] Group law: $P + Q + R = 0$ iff $P, Q, R$ are collinear; $O$ is the identity
- [ ] The $j$-invariant $j(E) = 1728 \cdot \frac{4a^3}{4a^3 + 27b^2}$: classifies $E$ over $\bar{k}$ up to isomorphism
- [ ] Over $\mathbb{C}$: $E \cong \mathbb{C}/\Lambda$ for a lattice $\Lambda$; group law is addition in $\mathbb{C}/\Lambda$

**Reading:**
- [ ] Shafarevich Vol 1, §III.3 (elliptic curves), §III.4 *(~2 hrs)*
- [ ] Silverman AEC, Ch 1 §1–3 (preview; return in Phase IV) *(~2 hrs)*

**Problems:**
- [ ] Harvard qual: "What are the involutions of an elliptic curve over $\mathbb{C}$?" — work the McMullen questions
- [ ] Show the group law is associative using Riemann-Roch (sketch): $\text{Pic}^0(E) \cong E$ as sets

---

### Weeks 17–18 — Hyperelliptic Curves and Phase I Consolidation

**Concepts to understand:**
- [ ] Hyperelliptic curve of genus $g$: double cover of $\mathbb{P}^1$ branched at $2g + 2$ points
- [ ] Canonical map $\phi_K: X \to \mathbb{P}^{g-1}$ (for $g \geq 2$): base-point-free when $X$ is not hyperelliptic
- [ ] Every genus-2 curve is hyperelliptic; the canonical map is 2:1 onto $\mathbb{P}^1$
- [ ] Review: the full algebraic $\leftrightarrow$ geometric dictionary built in Phase I
- [ ] Every classical concept paired with its scheme-theoretic avatar (preview of Phase II)

**Reading:**
- [ ] Shafarevich Vol 1, §III.5 *(~2 hrs)*
- [ ] Review Phase I notes; read the Phase II translation table *(~3 hrs)*

**Problems:**
- [ ] Show a genus-2 curve is always hyperelliptic using the canonical map
- [ ] Work 3 more problems from the Harvard qual collection; identify which Phase II concepts they preview

---

### Weeks 19–20 — Phase I Qual Practice

Use these two weeks for intensive qualifying exam problem work on Phase I material.

**Week 19:** Work Harvard qual problems involving affine/projective varieties, morphisms, and Bézout
- [ ] "Is $\mathbb{P}^1 \times \mathbb{P}^1$ a projective variety? Prove it." — use Segre
- [ ] "Find the explicit equation of the image of $\mathbb{P}^1 \times \mathbb{P}^1$ under Segre"
- [ ] "Show a hypersurface of degree $d$ has degree $d$" — use Hilbert polynomial
- [ ] "Is the twisted cubic the set-theoretic/scheme-theoretic intersection of two surfaces?" — Ogus questions

**Week 20:** Work Harvard qual problems involving curves, genus, and Riemann-Roch
- [ ] "Find the arithmetic genus of $y^3 = x^2 z$" — Frenkel
- [ ] "Calculate $H^0(\mathbb{P}^1, \Omega^1)$" — Poonen (preview of cohomology)
- [ ] "Describe Weil divisors and Cartier divisors on curves"
- [ ] All McMullen elliptic curve questions

> **Phase I Milestone:** You should now be able to: (1) define varieties, morphisms, divisors, and the Picard group from scratch; (2) state and apply Riemann-Roch and Hurwitz; (3) compute intersection numbers and apply Bézout; (4) work problems from the Harvard qual involving divisors, Pic, and curves. Open Ritvik's qual transcript — the question on Hurwitz's theorem should now be followable end-to-end.
