# Divisors and the Picard Group

*Authors: Based on Shafarevich §III.1, Hartshorne §II.6–7, Fulton Ch. 8*

## Relations

**Builds on:** [[concepts/algebraic-geometry/shafarevich-ch2-local-properties|Shafarevich Ch. II: Local Properties]], [[concepts/algebraic-geometry/shafarevich-ch2-normalization-resolution|Normalization and Resolution]]
**Extended by:** *(Riemann-Roch note — no note yet)*
**Concepts used:** [[concepts/algebraic-geometry/note|Classical Algebraic Geometry]]

---

> Throughout, $k$ denotes an algebraically closed field and $X$ is a smooth projective curve over $k$. For background on DVRs and the local ring $\mathcal{O}_{X,P}$, see [[concepts/algebraic-geometry/shafarevich-ch2-local-properties|Local Properties]]; for normalization and the smooth projective model, see [[concepts/algebraic-geometry/shafarevich-ch2-normalization-resolution|Normalization and Resolution]].

## Table of Contents

- [[#1. Introduction|1. Introduction]]
- [[#2. Weil Divisors on Curves|2. Weil Divisors on Curves]]
  - [[#2.1 The DVR at a Smooth Point|2.1 The DVR at a Smooth Point]]
  - [[#2.2 The Divisor Group|2.2 The Divisor Group]]
  - [[#2.3 Effective Divisors and the Degree Map|2.3 Effective Divisors and the Degree Map]]
- [[#3. Rational Functions and Principal Divisors|3. Rational Functions and Principal Divisors]]
  - [[#3.1 Order of Vanishing|3.1 Order of Vanishing]]
  - [[#3.2 Principal Divisors and the Key Degree Theorem|3.2 Principal Divisors and the Key Degree Theorem]]
  - [[#3.3 Worked Example on P^1|3.3 Worked Example on P^1]]
- [[#4. The Picard Group|4. The Picard Group]]
  - [[#4.1 Linear Equivalence|4.1 Linear Equivalence]]
  - [[#4.2 Definition and Basic Structure|4.2 Definition and Basic Structure]]
  - [[#4.3 Computations: P^1, A^1, and Elliptic Curves|4.3 Computations: P^1, A^1, and Elliptic Curves]]
- [[#5. Line Bundles and the Divisor-Line Bundle Correspondence|5. Line Bundles and the Divisor-Line Bundle Correspondence]]
  - [[#5.1 Rank-1 Locally Free Sheaves|5.1 Rank-1 Locally Free Sheaves]]
  - [[#5.2 The Sheaf O(D)|5.2 The Sheaf O(D)]]
  - [[#5.3 Global Sections and the Identification Pic(X) = Line Bundles|5.3 Global Sections and the Identification Pic(X) = Line Bundles]]
- [[#6. Linear Systems and Maps to Projective Space|6. Linear Systems and Maps to Projective Space]]
  - [[#6.1 Complete Linear Systems|6.1 Complete Linear Systems]]
  - [[#6.2 Base Locus and the Map phi_D|6.2 Base Locus and the Map phi_D]]
  - [[#6.3 Amplitude and Very Ampleness|6.3 Amplitude and Very Ampleness]]
  - [[#6.4 Examples: Veronese and Conics|6.4 Examples: Veronese and Conics]]
- [[#7. Algorithmic Applications|7. Algorithmic Applications]]
- [[#References|References]]

---

## 1. Introduction 📐

A rational function $f \in k(X)^\times$ on a projective curve $X$ cannot be regular everywhere if it is non-constant: wherever it is non-zero and finite, it eventually vanishes somewhere and diverges somewhere else. The key insight of divisor theory is to *track these zeros and poles globally*, as a single algebraic object.

A *divisor* on $X$ is a formal finite $\mathbb{Z}$-linear combination of closed points. Adding up the vanishing orders gives a map $f \mapsto \mathrm{div}(f)$, and two divisors are declared *linearly equivalent* if they differ by such a principal divisor. The group of equivalence classes is the *Picard group* $\mathrm{Pic}(X)$, the central invariant of this note.

The theory has three rewards:

1. **Classification of line bundles.** There is a canonical isomorphism $\mathrm{Pic}(X) \cong \{\text{line bundles on }X\}/{\cong}$.
2. **Maps to projective space.** Every morphism $X \to \mathbb{P}^r$ (up to linear equivalence) arises from a divisor $D$ with $h^0(D) = r+1$ and empty base locus.
3. **Group law on elliptic curves.** For a smooth elliptic curve $E$, the map $P \mapsto [P] - [O]$ identifies $E(k)$ with $\mathrm{Pic}^0(E)$, realizing the mysterious chord-tangent law as group addition in the Picard group.

> [!INFO] Historical context
> The notion of divisor on an algebraic curve is the algebraic analogue of the divisor of a meromorphic function on a compact Riemann surface. Abel's theorem (1826) and Jacobi's inversion theorem together characterize which divisors of degree zero are principal — a question that Riemann-Roch answers in the algebraic setting.

---

## 2. Weil Divisors on Curves 🔑

### 2.1 The DVR at a Smooth Point

Fix $X$ a smooth projective curve over $k = \bar{k}$. Recall from [[concepts/algebraic-geometry/shafarevich-ch2-local-properties|Local Properties]] that at each closed point $P \in X$, the local ring $\mathcal{O}_{X,P}$ is a *discrete valuation ring* (DVR): a local PID with maximal ideal $\mathfrak{m}_P = (t_P)$ for some *local parameter* $t_P$ (a uniformizer). The associated *valuation*

$$v_P : k(X)^\times \longrightarrow \mathbb{Z}$$

is defined by $v_P(f) = n$ when $f = t_P^n \cdot u$ with $u \in \mathcal{O}_{X,P}^\times$. By convention $v_P(0) = +\infty$.

**Key properties:**

- $v_P(fg) = v_P(f) + v_P(g)$,
- $v_P(f + g) \geq \min(v_P(f), v_P(g))$, with equality when $v_P(f) \neq v_P(g)$,
- $v_P(f) > 0 \Leftrightarrow P$ is a zero of $f$; $v_P(f) < 0 \Leftrightarrow P$ is a pole of $f$; $v_P(f) = 0 \Leftrightarrow f(P) \in k^\times$.

> [!NOTE] Finiteness of zeros and poles
> For any $f \in k(X)^\times$, the set $\{P \in X : v_P(f) \neq 0\}$ is finite. This follows because $f$ defines a rational map $X \dashrightarrow \mathbb{P}^1$ which, by the smoothness of $X$, extends to a regular morphism $X \to \mathbb{P}^1$; the preimage of $0$ and $\infty$ are each finite.

### 2.2 The Divisor Group

**Definition (Divisor).** A *Weil divisor* on $X$ is a formal sum

$$D = \sum_{P \in X} n_P [P], \quad n_P \in \mathbb{Z},$$

where $n_P = 0$ for all but finitely many $P$. The *divisor group* is

$$\mathrm{Div}(X) = \bigoplus_{P \in X} \mathbb{Z} \cdot [P],$$

with the group law given by pointwise addition of coefficients: $(D + D')_P = n_P + n_P'$.

The coefficient $n_P$ is called the *multiplicity* of $D$ at $P$. The *support* of $D$ is $\mathrm{supp}(D) = \{P : n_P \neq 0\}$.

### 2.3 Effective Divisors and the Degree Map

**Definition (Effective divisor).** A divisor $D = \sum n_P [P]$ is *effective*, written $D \geq 0$, if $n_P \geq 0$ for all $P$. We write $D \geq D'$ if $D - D' \geq 0$.

**Definition (Degree).** The *degree map* is the group homomorphism

$$\deg : \mathrm{Div}(X) \longrightarrow \mathbb{Z}, \quad \deg\!\left(\sum n_P [P]\right) = \sum_{P} n_P.$$

The kernel $\mathrm{Div}^0(X) = \ker(\deg)$ is the subgroup of *degree-zero divisors*.

> [!EXAMPLE] Simple examples
> - The zero divisor $0 \in \mathrm{Div}(X)$ has $n_P = 0$ for all $P$; $\deg(0) = 0$.
> - A single point $[P]$ is an effective divisor of degree 1.
> - The divisor $2[P] - [Q]$ has $\deg = 1$, support $\{P, Q\}$, and is not effective (since $n_Q = -1$).
> - On $\mathbb{P}^1$, the "hyperplane class" is any $[P]$; all degree-$n$ effective divisors on $\mathbb{P}^1$ consist of $n$ points (with multiplicity).

> [!QUESTION] Exercise 1: Degree and Support
> *This problem establishes fluency with the basic operations on divisors.*
>
> > **Prerequisites:** [[#2.3 Effective Divisors and the Degree Map|2.3 Effective Divisors and the Degree Map]]
>
> Let $X = \mathbb{P}^1$. (a) Write down all effective divisors of degree 2 whose support has exactly one point. (b) Show that $\mathrm{Div}^0(\mathbb{P}^1)$ is not finitely generated as an abelian group. (c) For an arbitrary smooth projective curve $X$ of genus $g$, how many points does a "general" effective divisor of degree $d$ have in its support?

> [!TIP]- Solution to Exercise 1
> **Key insight:** Effective divisors of degree 2 on $\mathbb{P}^1$ are either two distinct points or one point with multiplicity 2.
>
> **Sketch:**
> (a) The only effective divisors of degree 2 with a single-point support are $2[P]$ for any $P \in \mathbb{P}^1$. There are $|\mathbb{P}^1| = |k|$ such divisors (one per point).
>
> (b) $\mathrm{Div}^0(\mathbb{P}^1) = \bigoplus_{P \in \mathbb{P}^1} \mathbb{Z} \cdot [P] / \langle \sum [P] = 0 \rangle$ (informally). More precisely, for any finite set $S \subset \mathbb{P}^1$, the elements $\{[P] - [Q] : P \in S\}$ are linearly independent in $\mathrm{Div}^0$. Since $|k| = \infty$ (as $k = \bar{k}$), no finite generating set suffices.
>
> (c) A general effective divisor of degree $d$ on a curve has exactly $d$ points in its support (all with multiplicity 1). Special divisors (with multiplicities) form a closed subset of the $d$-th symmetric product $X^{(d)}$.

> [!QUESTION] Exercise 2: Divisors on a Nodal Curve
> *This problem explores what goes wrong when the smoothness hypothesis is dropped.*
>
> > **Prerequisites:** [[#2.2 The Divisor Group|2.2 The Divisor Group]]
>
> Let $C \subset \mathbb{P}^2$ be the nodal cubic $y^2 z = x^2(x+z)$, which has a node at $P = [0:0:1]$. The local ring $\mathcal{O}_{C,P}$ is NOT a DVR. (a) Verify this by showing $\mathfrak{m}_P = (x/z, y/z)$ requires two generators. (b) Explain why the definition of $v_P(f)$ breaks down at $P$. (c) The normalization $\tilde{C} \cong \mathbb{P}^1$ separates the node into two smooth points $\tilde{P}_1, \tilde{P}_2$. How should we define $\mathrm{Div}(\tilde{C})$ versus $\mathrm{Div}(C)$?

> [!TIP]- Solution to Exercise 2
> **Key insight:** A DVR requires the maximal ideal to be principal. At a node, two branches meet, so the maximal ideal needs two generators.
>
> **Sketch:**
> (a) In affine coordinates $(x,y) = (x/z, y/z)$, the nodal cubic becomes $y^2 = x^2(x+1)$. At the origin, both $x$ and $y$ vanish. The maximal ideal is $(x,y)$, but the relation $y^2 = x^2(x+1)$ is irreducible (not a product of linear factors over $\mathcal{O}_{C,P}$), so $\mathfrak{m}_P$ cannot be generated by a single element — confirming $\mathcal{O}_{C,P}$ is not a DVR.
>
> (b) Without a uniformizer, there is no canonical $v_P : \mathcal{O}_{C,P}^\times \to \mathbb{Z}$ with the valuation property: the two branches through $P$ give two distinct "orders of vanishing," and there is no way to reconcile them into a single integer.
>
> (c) $\mathrm{Div}(\tilde{C}) = \bigoplus_{Q \in \mathbb{P}^1} \mathbb{Z} \cdot [Q]$ is well-defined since $\tilde{C}$ is smooth. $\mathrm{Div}(C)$ can be defined using only the smooth points $C \setminus \{P\}$, or by working on $\tilde{C}$ and imposing the "compatibility at $\tilde{P}_1, \tilde{P}_2$" condition. The Picard group of $C$ and $\tilde{C}$ differ: $\mathrm{Pic}^0(\tilde{C}) = 0$ while $\mathrm{Pic}^0(C) \cong k^\times$.

---

## 3. Rational Functions and Principal Divisors 📐

### 3.1 Order of Vanishing

For a nonzero rational function $f \in k(X)^\times$, the *order of vanishing* of $f$ at $P$ is $v_P(f) \in \mathbb{Z}$. The finiteness property of Section 2.1 ensures that $v_P(f) \neq 0$ for only finitely many $P$, so we may form:

**Definition (Principal divisor).** The *principal divisor* of $f \in k(X)^\times$ is

$$\mathrm{div}(f) = \sum_{P \in X} v_P(f)\, [P] \;\in\; \mathrm{Div}(X).$$

We decompose $\mathrm{div}(f) = \mathrm{div}(f)_+ - \mathrm{div}(f)_-$ where

$$\mathrm{div}(f)_+ = \sum_{v_P(f) > 0} v_P(f)\,[P], \quad \mathrm{div}(f)_- = \sum_{v_P(f) < 0} (-v_P(f))\,[P]$$

are the *zero divisor* and *pole divisor* of $f$, both effective.

The map $\mathrm{div}: k(X)^\times \to \mathrm{Div}(X)$ is a group homomorphism (since $v_P(fg) = v_P(f) + v_P(g)$).

### 3.2 Principal Divisors and the Key Degree Theorem

**Theorem (Degree of a principal divisor is zero).** For any smooth projective curve $X$ and any $f \in k(X)^\times$,

$$\deg(\mathrm{div}(f)) = 0.$$

*Proof sketch.* The rational function $f$ defines a regular morphism $\varphi: X \to \mathbb{P}^1$ (extending the rational map, using smoothness). We must show $|\varphi^{-1}(0)| = |\varphi^{-1}(\infty)|$ counted with multiplicity.

For a finite morphism $\varphi: X \to \mathbb{P}^1$ of degree $d$, every point $Q \in \mathbb{P}^1$ has the property that

$$\sum_{P \in \varphi^{-1}(Q)} e_P = d,$$

where $e_P = v_P(\varphi^*(t_Q))$ is the *ramification index* at $P$ (here $t_Q$ is a local parameter at $Q$). Applying this with $Q = 0$ gives $\deg(\mathrm{div}(f)_+) = d$, and with $Q = \infty$ gives $\deg(\mathrm{div}(f)_-) = d$. Hence

$$\deg(\mathrm{div}(f)) = \deg(\mathrm{div}(f)_+) - \deg(\mathrm{div}(f)_-) = d - d = 0. \quad \square$$

**Corollary.** The image of $\mathrm{div}: k(X)^\times \to \mathrm{Div}(X)$ lies entirely within $\mathrm{Div}^0(X)$.

The subgroup of principal divisors is

$$\mathrm{PDiv}(X) = \{\mathrm{div}(f) : f \in k(X)^\times\} \subset \mathrm{Div}^0(X).$$

> [!WARNING] Non-principality on non-smooth or non-projective varieties
> The degree-zero statement fails for non-projective varieties. On $\mathbb{A}^1$, the function $f = x$ has $\mathrm{div}(x) = [0]$, which has degree 1. Projectivity (compactness) is essential to ensure the pole at infinity is accounted for.

### 3.3 Worked Example on P^1

On $\mathbb{P}^1$ with affine coordinate $x$, the function field is $k(x)$. Every nonzero $f \in k(x)$ is a ratio of polynomials. Write

$$f = c \cdot \frac{\prod_i (x - a_i)^{m_i}}{\prod_j (x - b_j)^{n_j}}, \quad c \in k^\times.$$

Then:
- $v_{[a_i]}(f) = m_i$ (zero of order $m_i$),
- $v_{[b_j]}(f) = -n_j$ (pole of order $n_j$),
- $v_{[\infty]}(f) = \deg(\text{denominator}) - \deg(\text{numerator}) = \sum n_j - \sum m_i$.

So

$$\mathrm{div}(f) = \sum_i m_i [a_i] - \sum_j n_j [b_j] + \left(\sum_j n_j - \sum_i m_i\right)[\infty].$$

Two canonical examples:

1. $\mathrm{div}(x) = [0] - [\infty]$. Zero at $0$, pole at $\infty$, degree $= 0$.
2. $\mathrm{div}\!\left(\dfrac{x-a}{x-b}\right) = [a] - [b]$ for $a \neq b$. This shows $[a] - [b]$ is a principal divisor on $\mathbb{P}^1$.

> [!EXAMPLE] Every degree-0 divisor on P^1 is principal
> Any divisor $D = \sum n_P [P]$ of degree zero on $\mathbb{P}^1$ is $\mathrm{div}(f)$ for $f = \prod_P (x - P)^{n_P}$ (where we set $x - \infty = 1$, and the $[\infty]$ contribution balances automatically by the degree condition). Hence $\mathrm{Pic}^0(\mathbb{P}^1) = 0$.

> [!QUESTION] Exercise 3: Computing Principal Divisors on P^1
> *This problem develops the mechanical skill of reading off zeros and poles from a rational function.*
>
> > **Prerequisites:** [[#3.1 Order of Vanishing|3.1 Order of Vanishing]]
>
> Compute $\mathrm{div}(f)$ for each of the following on $\mathbb{P}^1$ with coordinate $x$, and verify in each case that $\deg(\mathrm{div}(f)) = 0$:
> (a) $f = x^2 - 1$, (b) $f = x^3/(x^2+1)$ (assume $\mathrm{char}(k) \neq 2$), (c) $f = (x^2-1)/(x^2-4)$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** Factor the numerator and denominator, read off zeros/poles in $\mathbb{A}^1$, then compute the order at $\infty$ from the degree difference.
>
> **Sketch:**
> (a) $f = (x-1)(x+1)$. Zeros: $[1]$ and $[-1]$, each multiplicity 1. Pole at $\infty$ of order 2 (numerator degree 2 > denominator degree 0). So $\mathrm{div}(f) = [1] + [-1] - 2[\infty]$. Degree: $1+1-2=0$. ✓
>
> (b) $f = x^3/(x^2+1)$. Zeros: $[0]$ with multiplicity 3. Poles: $[i]$ and $[-i]$ each multiplicity 1, and $[\infty]$ with order $3-2=1$. So $\mathrm{div}(f) = 3[0] - [i] - [-i] - [\infty]$. Degree: $3-1-1-1=0$. ✓
>
> (c) $f = (x-1)(x+1)/((x-2)(x+2))$. $\mathrm{div}(f) = [1]+[-1]-[2]-[-2]$. Order at $\infty$: equal degrees ($2-2=0$), so no zero or pole there. Degree: $1+1-1-1=0$. ✓

> [!QUESTION] Exercise 4: Degree Theorem via a Linear Map
> *This problem gives a direct proof of $\deg(\mathrm{div}(f))=0$ in the simplest non-trivial case.*
>
> > **Prerequisites:** [[#3.2 Principal Divisors and the Key Degree Theorem|3.2 Principal Divisors and the Key Degree Theorem]]
>
> Let $f = (ax+b)/(cx+d) \in k(x)$ with $ad - bc \neq 0$ (a Möbius transformation). (a) Explicitly compute $\mathrm{div}(f)$ as a divisor on $\mathbb{P}^1$. (b) Verify $\deg(\mathrm{div}(f)) = 0$ directly, without invoking the general theorem. (c) The corresponding morphism $\varphi: \mathbb{P}^1 \to \mathbb{P}^1$ has degree 1; reconcile this with the proof sketch in §3.2.

> [!TIP]- Solution to Exercise 4
> **Key insight:** A degree-1 morphism $\mathbb{P}^1 \to \mathbb{P}^1$ is an isomorphism; every point has exactly one preimage with ramification index 1.
>
> **Sketch:**
> (a) $f = (ax+b)/(cx+d)$. The unique zero of the numerator is $x_0 = -b/a$ (if $a \neq 0$), the unique zero of the denominator is $x_1 = -d/c$ (if $c \neq 0$). Order at $\infty$: degrees are equal, so $v_\infty(f) = 1 - 1 = 0$. Hence $\mathrm{div}(f) = [x_0] - [x_1]$, which has degree $1 - 1 = 0$.
>
> (b) Directly: $\deg(\mathrm{div}(f)) = 1 + (-1) = 0$.
>
> (c) Since $\varphi$ has degree 1, every point $Q \in \mathbb{P}^1$ has $\sum_{P \in \varphi^{-1}(Q)} e_P = 1$, i.e., a unique preimage with ramification index 1. The proof gives $\deg(\mathrm{div}(f)_+) = 1 = \deg(\mathrm{div}(f)_-)$.

---

## 4. The Picard Group 🔑

### 4.1 Linear Equivalence

**Definition (Linear equivalence).** Two divisors $D, D' \in \mathrm{Div}(X)$ are *linearly equivalent*, written $D \sim D'$, if $D - D' = \mathrm{div}(f)$ for some $f \in k(X)^\times$.

Linear equivalence is an equivalence relation on $\mathrm{Div}(X)$ (reflexivity: $\mathrm{div}(1) = 0$; symmetry: $\mathrm{div}(f^{-1}) = -\mathrm{div}(f)$; transitivity: $\mathrm{div}(fg) = \mathrm{div}(f) + \mathrm{div}(g)$).

> [!NOTE] Geometric meaning
> $D \sim D'$ means: $D$ and $D'$ are "cut out by the same geometric data up to a change of rational function." On $\mathbb{P}^1$, any two points are linearly equivalent (since $[P] - [Q] = \mathrm{div}((x-P)/(x-Q))$), so there is essentially one "degree-1 class."

### 4.2 Definition and Basic Structure

**Definition (Picard group).** The *Picard group* of $X$ is the quotient

$$\mathrm{Pic}(X) = \mathrm{Div}(X) / \mathrm{PDiv}(X).$$

The *degree-zero Picard group* is

$$\mathrm{Pic}^0(X) = \mathrm{Div}^0(X) / \mathrm{PDiv}(X).$$

Since $\mathrm{PDiv}(X) \subset \mathrm{Div}^0(X)$ (by the degree theorem), the degree map descends to a well-defined surjective homomorphism $\deg: \mathrm{Pic}(X) \to \mathbb{Z}$, giving the short exact sequence

$$0 \longrightarrow \mathrm{Pic}^0(X) \longrightarrow \mathrm{Pic}(X) \xrightarrow{\;\deg\;} \mathbb{Z} \longrightarrow 0.$$

This sequence is *split*: any choice of base point $P_0 \in X$ gives a section $\mathbb{Z} \to \mathrm{Pic}(X)$ by $n \mapsto n[P_0]$. Hence

$$\mathrm{Pic}(X) \cong \mathbb{Z} \oplus \mathrm{Pic}^0(X).$$

**The factor $\mathbb{Z}$ records the degree; the factor $\mathrm{Pic}^0(X)$ records the "finer" geometry of the curve.**

> [!NOTE] Dependence on base point
> The splitting $\mathrm{Pic}(X) \cong \mathbb{Z} \oplus \mathrm{Pic}^0(X)$ depends on the choice of $P_0$. Changing $P_0$ to $P_0'$ changes the section by $[P_0'] - [P_0] \in \mathrm{Pic}^0(X)$, giving a different (but isomorphic) splitting.

### 4.3 Computations: P^1, A^1, and Elliptic Curves

**Proposition.** $\mathrm{Pic}(\mathbb{P}^1) \cong \mathbb{Z}$.

*Proof.* By the example in §3.3, every degree-zero divisor on $\mathbb{P}^1$ is principal, so $\mathrm{Pic}^0(\mathbb{P}^1) = 0$. The exact sequence gives $\mathrm{Pic}(\mathbb{P}^1) \cong \mathbb{Z}$, generated by the class of any point $[P]$. $\square$

**Proposition.** $\mathrm{Pic}(\mathbb{A}^1) = 0$.

*Proof.* Every divisor on $\mathbb{A}^1$ has the form $D = \sum_{i=1}^r n_i [a_i]$ for distinct $a_i \in k$. Set $f = \prod_{i=1}^r (x - a_i)^{n_i} \in k[x] \subset k(x)^\times$. This is a polynomial (regular on $\mathbb{A}^1$), and $\mathrm{div}(f) = D$ since there is no "point at infinity" on $\mathbb{A}^1$. Every divisor is principal, so $\mathrm{Pic}(\mathbb{A}^1) = 0$. $\square$

> [!INFO] Elliptic curves: a preview
> For a smooth elliptic curve $E$ with a chosen base point $O \in E$, there is a group isomorphism $\mathrm{Pic}^0(E) \cong E(k)$ given by
> $$P \longmapsto [P] - [O].$$
> The group law on $E$ (the chord-tangent construction) corresponds exactly to addition in $\mathrm{Pic}^0(E)$. This identification is the content of Abel's theorem for elliptic curves and will be proved in the Riemann-Roch note using Riemann-Roch's theorem and its corollaries.

> [!QUESTION] Exercise 5: Pic of Affine Space
> *This problem generalizes the computation $\mathrm{Pic}(\mathbb{A}^1)=0$ to higher dimensions.*
>
> > **Prerequisites:** [[#4.3 Computations: P^1, A^1, and Elliptic Curves|4.3 Computations: P^1, A^1, and Elliptic Curves]]
>
> Show that $\mathrm{Pic}(\mathbb{A}^n) = 0$ for all $n \geq 1$. (*Hint:* $k[\mathbb{A}^n] = k[x_1,\ldots,x_n]$ is a UFD; use the fact that the Picard group of a UFD domain is trivial.)

> [!TIP]- Solution to Exercise 5
> **Key insight:** The Picard group of a normal variety equals the class group, and for a UFD, every Weil divisor is principal.
>
> **Sketch:** The coordinate ring of $\mathbb{A}^n$ is $k[x_1,\ldots,x_n]$, a UFD. Every irreducible polynomial $p \in k[x_1,\ldots,x_n]$ defines a prime Weil divisor $V(p)$. Since the ring is a UFD, every Weil divisor (codimension-1 irreducible subvariety) is the zero locus of a single irreducible polynomial, hence is $\mathrm{div}(p)$ for that polynomial $p$. Thus every Weil divisor is principal, so $\mathrm{Cl}(\mathbb{A}^n) = 0$. For smooth varieties, $\mathrm{Pic} = \mathrm{Cl}$, so $\mathrm{Pic}(\mathbb{A}^n) = 0$.

> [!QUESTION] Exercise 6: Pic of Projective Space
> *This problem establishes $\mathrm{Pic}(\mathbb{P}^n) \cong \mathbb{Z}$, generated by a hyperplane.*
>
> > **Prerequisites:** [[#4.2 Definition and Basic Structure|4.2 Definition and Basic Structure]]
>
> (a) Show that every Weil divisor on $\mathbb{P}^n$ is linearly equivalent to $d \cdot [H]$ for some integer $d$ and any hyperplane $H \cong \mathbb{P}^{n-1}$. (*Hint:* Use homogeneous polynomials and the degree.) (b) Show that $[H] \not\sim 0$, i.e., no hyperplane is principal. Conclude that $\mathrm{Pic}(\mathbb{P}^n) \cong \mathbb{Z}$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** Rational functions on $\mathbb{P}^n$ are ratios of homogeneous polynomials of the same degree; their divisors always have degree zero.
>
> **Sketch:**
> (a) A prime divisor on $\mathbb{P}^n$ is an irreducible hypersurface $V(F)$ for a homogeneous polynomial $F$ of some degree $d$. For any hyperplane $H = V(L)$ (degree 1), we have $\mathrm{div}(F/L^d) = [V(F)] - d[H]$, so $[V(F)] \sim d[H]$. By linearity, every Weil divisor is equivalent to $d[H]$ for some $d$.
>
> (b) If $[H] = \mathrm{div}(f)$ for some $f \in k(\mathbb{P}^n)^\times$, then $f = G/H'$ for homogeneous $G, H'$ of the same degree. But $\mathrm{div}(G/H') = [V(G)] - [V(H')]$, which has degree $\deg G - \deg H' = 0$ as a divisor, while $[H]$ has degree 1. Contradiction. Hence the map $\deg: \mathrm{Pic}(\mathbb{P}^n) \to \mathbb{Z}$ is an isomorphism.

> [!QUESTION] Exercise 7: 2-Torsion in Pic^0 of an Elliptic Curve
> *This problem identifies specific torsion elements in the Picard group of an elliptic curve.*
>
> > **Prerequisites:** [[#4.3 Computations: P^1, A^1, and Elliptic Curves|4.3 Computations: P^1, A^1, and Elliptic Curves]]
>
> Let $E: y^2 = (x-e_1)(x-e_2)(x-e_3)$ be an elliptic curve over $k$ with $\mathrm{char}(k) \neq 2$, and let $O = [\infty]$ be the base point. Let $P_i = (e_i, 0)$ be the three 2-torsion points. (a) Show that $2[P_i] \sim 2[O]$, so $[P_i] - [O]$ has order dividing 2 in $\mathrm{Pic}^0(E)$. (b) Show that $[P_i] \not\sim [O]$ (i.e., $P_i \neq O$ in $E(k)$). (c) Conclude that $\{[O], P_1, P_2, P_3\}$ contribute distinct elements of order exactly 2 in $\mathrm{Pic}^0(E)$, so $(\mathbb{Z}/2\mathbb{Z})^2 \hookrightarrow \mathrm{Pic}^0(E)[2]$.

> [!TIP]- Solution to Exercise 7
> **Key insight:** The function $x - e_i$ has a double zero at $P_i$ and a double pole at $O$ (since $O$ is a flex point at infinity with local parameter $1/x$).
>
> **Sketch:**
> (a) The function $f_i = x - e_i \in k(E)^\times$ has $\mathrm{div}(f_i) = 2[P_i] - 2[O]$ (double zero at $P_i$, double pole at $O$ counted by the Weierstrass model). So $2[P_i] \sim 2[O]$, i.e., $2([P_i]-[O]) = 0$ in $\mathrm{Pic}^0(E)$.
>
> (b) If $[P_i] \sim [O]$, there would be a rational function $g: E \to \mathbb{P}^1$ with a simple zero at $P_i$ and a simple pole at $O$ (i.e., $\mathrm{div}(g) = [P_i]-[O]$). Such $g$ has degree 1, making $E \cong \mathbb{P}^1$. But $E$ has genus 1 $\neq$ 0, contradiction.
>
> (c) The three differences $[P_i]-[O]$ are distinct nonzero elements of order 2. Together with 0, they form a subgroup isomorphic to $(\mathbb{Z}/2\mathbb{Z})^2$ inside $\mathrm{Pic}^0(E)[2] \cong E[2]$.

---

## 5. Line Bundles and the Divisor-Line Bundle Correspondence 🔑

### 5.1 Rank-1 Locally Free Sheaves

A *line bundle* on $X$ (equivalently, a *rank-1 locally free sheaf* or *invertible sheaf*) is a sheaf $\mathcal{L}$ of $\mathcal{O}_X$-modules such that there exists an open cover $\{U_i\}$ of $X$ with $\mathcal{L}|_{U_i} \cong \mathcal{O}_{U_i}$ for each $i$.

The set of isomorphism classes of line bundles on $X$ forms an abelian group under tensor product:

- **Multiplication:** $[\mathcal{L}] \cdot [\mathcal{L}'] = [\mathcal{L} \otimes_{\mathcal{O}_X} \mathcal{L}']$
- **Identity:** $[\mathcal{O}_X]$ (the structure sheaf)
- **Inverse:** $[\mathcal{L}]^{-1} = [\mathcal{L}^{-1}] = [\mathcal{H}om(\mathcal{L}, \mathcal{O}_X)]$ (the dual sheaf)

This group is also denoted $\mathrm{Pic}(X)$, and the fundamental theorem below justifies this notation.

### 5.2 The Sheaf O(D)

**Definition (Sheaf of a divisor).** For a divisor $D \in \mathrm{Div}(X)$, define the sheaf $\mathcal{O}(D)$ by

$$\mathcal{O}(D)(U) = \{f \in k(X)^\times : \mathrm{div}(f)|_U + D|_U \geq 0\} \cup \{0\}$$

for each open $U \subset X$. Concretely, $f$ belongs to $\mathcal{O}(D)(U)$ if and only if, for every $P \in U$, $v_P(f) \geq -n_P$ (where $D = \sum n_P [P]$).

In words: $\mathcal{O}(D)$ consists of rational functions that are "allowed to have poles only where $D$ has positive multiplicity, and only up to the order prescribed by $D$."

> [!NOTE] Local trivialization of O(D)
> Near any point $P \in X$, choose an open $U_P \ni P$ and a rational function $g_P$ with $\mathrm{div}(g_P)|_{U_P} = -D|_{U_P}$ (possible since $D$ has integer coefficients and $X$ is smooth). Then $\mathcal{O}(D)|_{U_P} \cong \mathcal{O}_{U_P}$ via $f \mapsto f/g_P$. This confirms $\mathcal{O}(D)$ is an invertible sheaf.

### 5.3 Global Sections and the Identification Pic(X) = Line Bundles

The *space of global sections* of $\mathcal{O}(D)$ is

$$H^0(X, \mathcal{O}(D)) = \{f \in k(X)^\times : \mathrm{div}(f) + D \geq 0\} \cup \{0\}.$$

This is a finite-dimensional $k$-vector space. We write $h^0(D) = \dim_k H^0(X, \mathcal{O}(D))$.

**Key structural facts:**

1. $\mathcal{O}(D) \otimes \mathcal{O}(D') \cong \mathcal{O}(D + D')$.
2. $\mathcal{O}(D)^{-1} \cong \mathcal{O}(-D)$.
3. $\mathcal{O}(0) = \mathcal{O}_X$ (sections of $\mathcal{O}(0)$ are regular functions; on a projective variety, these are constants).

**Theorem (Divisor–line bundle correspondence).** The map

$$\mathrm{Div}(X) \longrightarrow \{\text{line bundles on }X\}, \quad D \longmapsto \mathcal{O}(D),$$

is a surjective group homomorphism with kernel $\mathrm{PDiv}(X)$. Hence it induces an isomorphism

$$\mathrm{Pic}(X) \xrightarrow{\;\sim\;} \{\text{isomorphism classes of line bundles on }X\}.$$

*Proof sketch.* Surjectivity: any line bundle $\mathcal{L}$ embeds into the constant sheaf $k(X)$ (since $X$ is integral), giving a rational section $s$; then $\mathcal{L} \cong \mathcal{O}(D)$ where $D = -\mathrm{div}(s)$. The kernel calculation: $\mathcal{O}(D) \cong \mathcal{O}(D')$ iff they share a rational section differing by a unit in $k(X)^\times$, iff $D - D' = \mathrm{div}(f)$. $\square$

**Example: $\mathcal{O}(n)$ on $\mathbb{P}^1$.** Take $D = n[\infty]$ on $\mathbb{P}^1$. Then

$$H^0(\mathbb{P}^1, \mathcal{O}(n)) = \{f \in k(x) : v_\infty(f) \geq -n,\text{ no other poles}\}.$$

Since $v_\infty(x^j) = -j$, the sections are exactly polynomials $a_0 + a_1 x + \cdots + a_n x^n$ of degree at most $n$. So $h^0(\mathcal{O}(n)) = n + 1$.

**$H^0(\mathbb{P}^1, \mathcal{O}(n)) = 0$ for $n < 0$** (no rational function can satisfy $v_\infty(f) \geq -n > 0$ without being zero).

> [!QUESTION] Exercise 8: Sections of O(D) for Explicit D
> *This problem builds fluency with the global sections formula.*
>
> > **Prerequisites:** [[#5.3 Global Sections and the Identification Pic(X) = Line Bundles|5.3 Global Sections and the Identification Pic(X) = Line Bundles]]
>
> On $\mathbb{P}^1$ with coordinate $x$, compute $H^0(\mathbb{P}^1, \mathcal{O}(D))$ and its dimension for: (a) $D = 2[0] + [1] - [\infty]$, (b) $D = -[0] + 3[\infty]$, (c) $D = [0] - [1]$ (degree 0).

> [!TIP]- Solution to Exercise 8
> **Key insight:** $f \in H^0(\mathbb{P}^1, \mathcal{O}(D))$ iff $v_P(f) \geq -n_P$ for all $P$, i.e., $f$ has at most a pole of order $n_P$ at each $P$ where $n_P > 0$ and is required to vanish to order $|n_P|$ where $n_P < 0$.
>
> **Sketch:**
> (a) $D = 2[0]+[1]-[\infty]$: $f$ must have $v_0(f)\geq -2$, $v_1(f)\geq -1$, $v_\infty(f)\geq 1$, and be regular elsewhere. In terms of $x$: $f = p(x)/(x^2(x-1))$ where $p$ is a polynomial, and $v_\infty(f)\geq 1$ forces $\deg p \leq 1$. So $f = (ax+b)/(x^2(x-1))$ — but we also need $v_\infty(f)\geq 1$, meaning $\deg(\text{denom}) - \deg(\text{num}) \geq 1$, i.e., $3 - \deg p \geq 1$, so $\deg p \leq 2$. But also checking: $f=(ax^2+bx+c)/(x^2(x-1))$ has $v_\infty = 3-2=1\geq 1$. So $H^0 = \{(ax^2+bx+c)/(x^2(x-1))\}$, dimension 3.
>
> (b) $D = -[0]+3[\infty]$: $v_0(f)\geq 1$ (zero at 0), $v_\infty(f)\geq -3$, regular elsewhere. So $f = x \cdot q(x)$ for $q$ a polynomial with $\deg(xq)\leq 3$, giving $f \in \{ax, bx^2, cx^3\}$, dimension 3.
>
> (c) $D=[0]-[1]$: $v_0(f)\geq -1$, $v_1(f)\geq 1$, regular elsewhere, $v_\infty(f)\geq 0$. So $f$ has a zero at 1, at most a simple pole at 0, and no other poles. $f = a(x-1)/x$ for $a\in k$. Dimension 1.

> [!QUESTION] Exercise 9: Tensor Product Rule for Line Bundles
> *This problem verifies the tensor product formula $\mathcal{O}(D)\otimes\mathcal{O}(D')\cong\mathcal{O}(D+D')$ directly.*
>
> > **Prerequisites:** [[#5.2 The Sheaf O(D)|5.2 The Sheaf O(D)]]
>
> (a) Show directly from the definition that if $f \in \mathcal{O}(D)(U)$ and $g \in \mathcal{O}(D')(U)$, then $fg \in \mathcal{O}(D+D')(U)$. (b) Conclude that there is a natural map $\mathcal{O}(D) \otimes \mathcal{O}(D') \to \mathcal{O}(D+D')$, and verify it is an isomorphism.

> [!TIP]- Solution to Exercise 9
> **Key insight:** Valuations are additive: $v_P(fg) = v_P(f) + v_P(g)$.
>
> **Sketch:**
> (a) If $f\in\mathcal{O}(D)(U)$ then $v_P(f)\geq -n_P$ for all $P\in U$; if $g\in\mathcal{O}(D')(U)$ then $v_P(g)\geq -n_P'$. Since $v_P(fg) = v_P(f)+v_P(g) \geq -n_P - n_P' = -(n_P+n_P')$, we have $fg\in\mathcal{O}(D+D')(U)$.
>
> (b) The natural map $\mathcal{O}(D)(U)\times\mathcal{O}(D')(U)\to\mathcal{O}(D+D')(U)$, $(f,g)\mapsto fg$, is bilinear and factors through the tensor product. To see it's an isomorphism, work locally: on a sufficiently small open $U$ containing $P$, choose a local uniformizer $t$ at $P$, and write $f = t^a u$, $g = t^b v$ with $u,v$ units; then $fg = t^{a+b}(uv)$, and the map is $t^a\mathcal{O}_U \otimes t^b\mathcal{O}_U \xrightarrow{\sim} t^{a+b}\mathcal{O}_U$ via multiplication, an isomorphism.

> [!QUESTION] Exercise 10: h^0 Computations via the Riemann-Roch Bound
> *This problem uses the Riemann-Roch theorem (stated without proof) to compute h^0 for explicit divisors on curves of low genus.*
>
> > **Prerequisites:** [[#5.3 Global Sections and the Identification Pic(X) = Line Bundles|5.3 Global Sections and the Identification Pic(X) = Line Bundles]]
>
> Riemann-Roch states: for a smooth projective curve of genus $g$ and a divisor $D$, $h^0(D) - h^0(K-D) = \deg(D) + 1 - g$, where $K$ is the canonical divisor. Using this (without proof), compute $h^0(D)$ for: (a) $D = n[P]$ on $\mathbb{P}^1$ (genus 0), (b) $D = n[P]$ on an elliptic curve $E$ (genus 1) for $n = 0, 1, 2, 3$.

> [!TIP]- Solution to Exercise 10
> **Key insight:** On $\mathbb{P}^1$, the canonical divisor has degree $-2$ (genus 0), so $h^0(K-D)=0$ for $\deg D>-2$. On an elliptic curve, $K\sim 0$, so Riemann-Roch becomes $h^0(D)-h^0(-D) = \deg D$.
>
> **Sketch:**
> (a) $g=0$, $\deg K = -2$. For $D=n[P]$ with $n\geq 0$: $\deg(K-D) = -2-n < 0$, so $h^0(K-D)=0$. Riemann-Roch: $h^0(n[P]) = n+1$. Matches our direct computation.
>
> (b) $g=1$, $K\sim 0$. Riemann-Roch: $h^0(D)-h^0(-D)=\deg D$. For $n=0$: $D=0$, $h^0(0)=1$ (constants). For $n=1$: $h^0([P])-h^0(-[P])=1$; $h^0(-[P])=0$ (degree $-1$), so $h^0([P])=1$. For $n=2$: $h^0(2[P])=2$. For $n=3$: $h^0(3[P])=3$.

> [!QUESTION] Exercise 11: O(D) for a Negative Divisor
> *This problem establishes when the space of global sections is trivial.*
>
> > **Prerequisites:** [[#5.3 Global Sections and the Identification Pic(X) = Line Bundles|5.3 Global Sections and the Identification Pic(X) = Line Bundles]]
>
> Let $X$ be a smooth projective curve. (a) Show that if $\deg(D) < 0$ then $H^0(X, \mathcal{O}(D)) = 0$. (b) Show that if $D \sim D'$ then $h^0(D) = h^0(D')$. (c) Show that $h^0(0) = 1$ (the only global sections of $\mathcal{O}_X$ are constants).

> [!TIP]- Solution to Exercise 11
> **Key insight:** A nonzero section $f \in H^0(X,\mathcal{O}(D))$ satisfies $\mathrm{div}(f)+D \geq 0$, so $\deg(\mathrm{div}(f)+D) = \deg(D) \geq 0$ (since $\deg(\mathrm{div}(f))=0$).
>
> **Sketch:**
> (a) If $f \neq 0$ were in $H^0(X,\mathcal{O}(D))$, then $\deg(\mathrm{div}(f)+D) \geq 0$ (as an effective divisor). But $\deg(\mathrm{div}(f)+D) = \deg D < 0$, contradiction.
>
> (b) If $D \sim D'$, then $D' = D + \mathrm{div}(g)$ for some $g\in k(X)^\times$. The map $f \mapsto f/g$ is an isomorphism $H^0(X,\mathcal{O}(D)) \xrightarrow{\sim} H^0(X,\mathcal{O}(D'))$: indeed, $\mathrm{div}(f)+D\geq 0$ iff $\mathrm{div}(f/g)+D'\geq 0$.
>
> (c) $H^0(X,\mathcal{O}_X) = \{f\in k(X)^\times : \mathrm{div}(f)\geq 0\}\cup\{0\} = $ regular functions on $X$. By projectivity of $X$, any regular function is constant. So $H^0(X,\mathcal{O}_X) = k$, dimension 1.

> [!QUESTION] Exercise 12: Dual Sheaf and O(-D)
> *This problem derives the duality $\mathcal{O}(D)^\vee \cong \mathcal{O}(-D)$ from first principles.*
>
> > **Prerequisites:** [[#5.2 The Sheaf O(D)|5.2 The Sheaf O(D)]]
>
> For a divisor $D$ on a smooth projective curve $X$: (a) Show that $\mathcal{H}om(\mathcal{O}(D), \mathcal{O}_X) \cong \mathcal{O}(-D)$ as $\mathcal{O}_X$-modules. (b) Deduce that $\mathrm{Pic}(X)$ is indeed a group under $\otimes$ (i.e., every element has an inverse).

> [!TIP]- Solution to Exercise 12
> **Key insight:** Multiplication by $f \in \mathcal{O}(D)(U)$ maps into $\mathcal{O}_X(U)$ precisely when $f \in \mathcal{O}(-D)(U)$.
>
> **Sketch:**
> (a) A morphism of $\mathcal{O}_X$-modules $\phi: \mathcal{O}(D) \to \mathcal{O}_X$ over an open $U$ is determined by where the "canonical section" $1 \in k(X)^\times$ (locally trivializing $\mathcal{O}(D)$) maps to. Say $\phi_U(1) = h \in \mathcal{O}_X(U)$. For $\phi$ to be a well-defined map $\mathcal{O}(D)(U) \to \mathcal{O}_X(U)$, we need $f \cdot h \in \mathcal{O}_X(U)$ for all $f\in\mathcal{O}(D)(U)$. This forces $v_P(h) \geq -v_P(f) \geq n_P = v_P(D)$ for all $P$, i.e., $h\in\mathcal{O}(-D)(U)$. So $\mathcal{H}om(\mathcal{O}(D),\mathcal{O}_X)\cong\mathcal{O}(-D)$.
>
> (b) From (a): $\mathcal{O}(D) \otimes \mathcal{O}(-D) \cong \mathcal{O}(D) \otimes \mathcal{O}(D)^\vee \cong \mathcal{O}_X$ (using the natural evaluation pairing for rank-1 locally free sheaves). So $[\mathcal{O}(-D)]$ is the inverse of $[\mathcal{O}(D)]$ in $\mathrm{Pic}(X)$.

---

## 6. Linear Systems and Maps to Projective Space 📐

### 6.1 Complete Linear Systems

**Definition (Linear system).** A *linear system* on $X$ is a set of the form $\mathfrak{d} = \{D' \in \mathrm{Div}(X) : D' \geq 0,\, D' \sim D\}$ for some divisor $D$, i.e., a set of effective divisors linearly equivalent to $D$.

The *complete linear system* associated to $D$ is

$$|D| = \{D' \in \mathrm{Div}(X) : D' \geq 0,\, D' \sim D\}.$$

There is a canonical bijection

$$|D| \;\longleftrightarrow\; \mathbb{P}(H^0(X, \mathcal{O}(D))),$$

given by $\mathrm{div}(f) + D \leftrightarrow [f]$ for $f \in H^0(X, \mathcal{O}(D)) \setminus \{0\}$. The *dimension* of $|D|$ is

$$\dim|D| = h^0(D) - 1.$$

> [!NOTE] Projective space of sections
> The bijection $|D| \leftrightarrow \mathbb{P}(H^0(X,\mathcal{O}(D)))$ comes from the fact that two nonzero sections $f, f' \in H^0(X,\mathcal{O}(D))$ give the same effective divisor (i.e., $\mathrm{div}(f)+D = \mathrm{div}(f')+D$) iff $\mathrm{div}(f/f')=0$ iff $f/f' \in k^\times$ (the only global regular functions on a projective variety are constants). So divisors correspond to lines in $H^0(X,\mathcal{O}(D))$.

### 6.2 Base Locus and the Map phi_D

**Definition (Base locus).** The *base locus* of $|D|$ is

$$\mathrm{Bs}(|D|) = \bigcap_{D' \in |D|} \mathrm{supp}(D').$$

Equivalently, $P \in \mathrm{Bs}(|D|)$ iff every nonzero section $s \in H^0(X, \mathcal{O}(D))$ vanishes at $P$.

**Definition (Map to projective space).** Choose a basis $s_0, \ldots, s_r \in H^0(X, \mathcal{O}(D))$. Define a (possibly rational) map

$$\phi_D : X \dashrightarrow \mathbb{P}^r, \quad P \longmapsto [s_0(P) : s_1(P) : \cdots : s_r(P)].$$

More precisely, after choosing a local trivialization of $\mathcal{O}(D)$ near $P$, the $s_i(P)$ are elements of $k$; the ratio $[s_0(P):\cdots:s_r(P)]$ is well-defined as an element of $\mathbb{P}^r$ as long as not all $s_i$ vanish at $P$.

**Theorem.** $\phi_D$ extends to a regular morphism $X \to \mathbb{P}^r$ if and only if $\mathrm{Bs}(|D|) = \emptyset$.

> [!WARNING] Base-point-freeness is necessary for a morphism
> If $P \in \mathrm{Bs}(|D|)$, then $s_i(P) = 0$ for all $i$, and $\phi_D$ is undefined at $P$. Removing base points (e.g., by passing to a subsystem or a different $D'$) is often necessary before constructing a morphism.

### 6.3 Amplitude and Very Ampleness

**Definition (Very ample).** A divisor $D$ is *very ample* if:

1. $\mathrm{Bs}(|D|) = \emptyset$ (so $\phi_D$ is a morphism), and
2. $\phi_D : X \hookrightarrow \mathbb{P}^r$ is a closed immersion (i.e., an embedding).

Condition 2 means $\phi_D$ is injective on points and on tangent spaces: $\phi_D(P) \neq \phi_D(Q)$ for $P \neq Q$ (separates points), and $d\phi_D|_P: T_{X,P} \hookrightarrow T_{\mathbb{P}^r, \phi_D(P)}$ is injective (separates tangent vectors).

**Definition (Ample).** A divisor $D$ is *ample* if $nD$ is very ample for some $n \gg 0$.

> [!NOTE] Serre's criterion
> On a projective variety, $D$ is ample if and only if $H^i(X, \mathcal{F} \otimes \mathcal{O}(nD)) = 0$ for all $i > 0$, all coherent sheaves $\mathcal{F}$, and all sufficiently large $n$ (depending on $\mathcal{F}$). This is the cohomological characterization of ampleness.

### 6.4 Examples: Veronese and Conics

**Example 1: $\mathcal{O}(1)$ on $\mathbb{P}^1$.** Take $D = [\infty]$, so $\mathcal{O}(D) = \mathcal{O}(1)$. A basis for $H^0(\mathbb{P}^1, \mathcal{O}(1))$ is $\{1, x\}$ (constants and linear polynomials, dimension 2). The map $\phi_D: \mathbb{P}^1 \to \mathbb{P}^1$ sends $[s:t] \mapsto [s:t]$, which is the identity. This is a closed immersion, so $\mathcal{O}(1)$ is very ample on $\mathbb{P}^1$.

**Example 2: The degree-$n$ Veronese embedding.** Take $D = n[\infty]$ on $\mathbb{P}^1$, so $\mathcal{O}(D) = \mathcal{O}(n)$. A basis for $H^0(\mathbb{P}^1, \mathcal{O}(n))$ is $\{1, x, x^2, \ldots, x^n\}$ (dimension $n+1$). In homogeneous coordinates $[s:t]$:

$$\nu_n : \mathbb{P}^1 \hookrightarrow \mathbb{P}^n, \quad [s:t] \longmapsto [s^n : s^{n-1}t : s^{n-2}t^2 : \cdots : t^n].$$

This is the *degree-$n$ Veronese embedding*. It is a closed immersion for all $n \geq 1$, so $\mathcal{O}(n)$ is very ample on $\mathbb{P}^1$.

**Why is $\nu_n$ a closed immersion?** Injectivity on points: $[s:t] \neq [s':t']$ implies some ratio $s/t \neq s'/t'$, hence $(s/t)^n \neq (s'/t')^n$ or $(s/t)^{n-1} \neq (s'/t')^{n-1}$, so the images in $\mathbb{P}^n$ differ. Injectivity on tangent vectors: the differential $d\nu_n$ at $[1:0]$ maps $\partial/\partial(t/s)|_{[1:0]}$ to $(0, 1, 0, \ldots, 0)$, which is nonzero.

**Example 3: A conic as a Veronese image.** Let $C \subset \mathbb{P}^2$ be a smooth conic (say $xz = y^2$). By the classification of conics, $C \cong \mathbb{P}^1$. Under this isomorphism, the hyperplane class on $C$ (i.e., $\mathcal{O}_C(1) = \mathcal{O}_{\mathbb{P}^2}(1)|_C$) corresponds to $\mathcal{O}_{\mathbb{P}^1}(2)$. The complete linear system $|2[P]|$ on $\mathbb{P}^1$ has $h^0 = 3$ and gives exactly the Veronese $\nu_2: \mathbb{P}^1 \hookrightarrow \mathbb{P}^2$, whose image is $C$. So $C$ is "the degree-2 Veronese of $\mathbb{P}^1$."

> [!QUESTION] Exercise 13: Base Locus of a Linear System
> *This problem computes the base locus for an explicit subsystem.*
>
> > **Prerequisites:** [[#6.2 Base Locus and the Map phi_D|6.2 Base Locus and the Map phi_D]]
>
> On $\mathbb{P}^1$, consider the linear system $\mathfrak{d} = \{D' \in |2[\infty]| : D' \geq [0]\}$ (the subsystem of $|2[\infty]|$ consisting of effective divisors that contain $[0]$ with multiplicity at least 1). (a) Identify $\mathfrak{d}$ with a projective subspace of $\mathbb{P}(H^0(\mathbb{P}^1,\mathcal{O}(2)))$. (b) Compute $\mathrm{Bs}(\mathfrak{d})$. (c) What morphism $\phi: \mathbb{P}^1 \to \mathbb{P}^1$ does $\mathfrak{d}$ define?

> [!TIP]- Solution to Exercise 13
> **Key insight:** The condition $D' \geq [0]$ forces every section to vanish at 0, cutting out a hyperplane in the space of sections.
>
> **Sketch:**
> (a) $H^0(\mathbb{P}^1,\mathcal{O}(2)) = \mathrm{span}\{1, x, x^2\}$. The condition $D' \geq [0]$ means the section $f$ with $\mathrm{div}(f)+2[\infty] = D' \geq [0]$ must have $v_0(f) \geq 1$, so $f$ vanishes at $x=0$. The sections in question are $f \in \{ax + bx^2 : a, b \in k\} = x \cdot k[x]_{\leq 1}$. So $\mathfrak{d} \leftrightarrow \mathbb{P}(\mathrm{span}\{x, x^2\}) \cong \mathbb{P}^1$.
>
> (b) The sections in $\mathfrak{d}$ are $ax + bx^2 = x(a+bx)$, all vanishing at $x=0$ (i.e., $[0]$) and nowhere else simultaneously (the other zero is $-a/b$, varying). So $\mathrm{Bs}(\mathfrak{d}) = \{[0]\}$.
>
> (c) With basis $\{x, x^2\}$, the map is $[s:t] \mapsto [s \cdot t: s^2 \cdot t^2]$... more carefully in affine coordinates, $x \mapsto [x : x^2] = [1 : x]$ (dividing by $x$), giving the identity map $\mathbb{P}^1 \to \mathbb{P}^1$, $[s:t]\mapsto [s:t]$, well-defined away from $[0]$. Since $\mathrm{Bs}\neq\emptyset$, this is only a rational map.

> [!QUESTION] Exercise 14: Very Ampleness Criterion for a Degree-3 Divisor on an Elliptic Curve
> *This problem verifies that a degree-3 divisor on an elliptic curve embeds it in P^2.*
>
> > **Prerequisites:** [[#6.3 Amplitude and Very Ampleness|6.3 Amplitude and Very Ampleness]], [[#5.3 Global Sections and the Identification Pic(X) = Line Bundles|5.3 Global Sections and the Identification Pic(X) = Line Bundles]]
>
> Let $E$ be a smooth elliptic curve (genus 1) with base point $O$. (a) Using Riemann-Roch (from Exercise 10), show $h^0(3[O]) = 3$. (b) Show $\mathrm{Bs}(|3[O]|) = \emptyset$. (c) The resulting map $\phi_{3[O]}: E \to \mathbb{P}^2$ is a closed immersion; what is the degree of its image?

> [!TIP]- Solution to Exercise 14
> **Key insight:** A degree-$d$ divisor on a curve of genus $g$ is very ample if $d \geq 2g+1$ (by the Riemann-Roch theorem). For $g=1$, this means $d \geq 3$.
>
> **Sketch:**
> (a) From Exercise 10(b): $h^0(3[O]) = 3$ by Riemann-Roch.
>
> (b) Base-point-freeness: for any $P \in E$, we want a section $s \in H^0(E,\mathcal{O}(3[O]))$ that doesn't vanish at $P$. The divisor $3[O]-[P]$ has degree 2, and $h^0(3[O]-[P]) = h^0(2[O]+(O-P)) = 2$ (by Riemann-Roch, $h^0 = \deg+1-g+h^0(K-D) = 2+1-1+0=2$). So the evaluation map $H^0(\mathcal{O}(3[O])) \to k_P$ is not identically zero, meaning some $s$ doesn't vanish at $P$.
>
> (c) The image $\phi_{3[O]}(E) \subset \mathbb{P}^2$ is a smooth plane curve of degree 3 (since $\mathcal{O}(3[O])$ corresponds to the pullback of $\mathcal{O}_{\mathbb{P}^2}(1)$ restricted to the image, and the degree of the image equals $\deg(3[O]) = 3$). A smooth plane cubic has genus 1, consistent with $E$ having genus 1.

> [!QUESTION] Exercise 15: Linear Systems on P^1 and the Degree-Dimension Formula
> *This problem establishes the formula $\dim|D| = \deg(D)$ for $\deg(D) \geq 0$ on $\mathbb{P}^1$.*
>
> > **Prerequisites:** [[#6.1 Complete Linear Systems|6.1 Complete Linear Systems]], [[#5.3 Global Sections and the Identification Pic(X) = Line Bundles|5.3 Global Sections and the Identification Pic(X) = Line Bundles]]
>
> Let $X = \mathbb{P}^1$. (a) Show that for any divisor $D$ on $\mathbb{P}^1$ with $\deg(D) = d \geq 0$, we have $h^0(D) = d+1$. (b) Conclude that $\dim|D| = d$ for $d \geq 0$. (c) Give a geometric interpretation: what does a general element of $|D|$ look like?

> [!TIP]- Solution to Exercise 15
> **Key insight:** On $\mathbb{P}^1$, every divisor of degree $d\geq 0$ is linearly equivalent to $d[\infty]$, and $H^0(\mathbb{P}^1,\mathcal{O}(d[\infty]))$ is spanned by $\{1,x,\ldots,x^d\}$.
>
> **Sketch:**
> (a) By the example in §3.3, every degree-0 divisor on $\mathbb{P}^1$ is principal. So $D \sim d[\infty]$ for any $D$ with $\deg D = d$. By Exercise 11(b), $h^0(D) = h^0(d[\infty])$. A section of $\mathcal{O}(d[\infty])$ is a rational function $f$ with $v_\infty(f) \geq -d$ and no other poles — i.e., a polynomial of degree $\leq d$. This vector space has dimension $d+1$.
>
> (b) $\dim|D| = h^0(D)-1 = d$.
>
> (c) A general element of $|D|$ is $\mathrm{div}(f) + d[\infty]$ for a degree-$d$ polynomial $f = c\prod_{i=1}^d (x-a_i)$ with distinct roots $a_i$. Geometrically, it is a set of $d$ distinct points in $\mathbb{A}^1 \subset \mathbb{P}^1$ (plus possibly a contribution at $\infty$ if $d[\infty]$ has extra multiplicity).

> [!QUESTION] Exercise 16: Pencils of Divisors and Rational Maps
> *This problem analyzes a 1-dimensional linear system (a pencil) and the associated map.*
>
> > **Prerequisites:** [[#6.2 Base Locus and the Map phi_D|6.2 Base Locus and the Map phi_D]]
>
> A *pencil* is a 1-dimensional linear system, i.e., $\dim|D| = 1$ so $h^0(D) = 2$. (a) On $\mathbb{P}^1$, identify all pencils: what are the possible divisors $D$ that give a pencil? (b) For $X$ a smooth projective curve of genus $g \geq 1$ and a pencil $|D|$ with $\deg D = 1$, show the map $\phi_D: X \to \mathbb{P}^1$ is an isomorphism (so $X \cong \mathbb{P}^1$, hence $g = 0$, contradiction). Conclude that no curve of genus $\geq 1$ has a degree-1 pencil. (c) What is the minimum degree of a divisor on an elliptic curve $E$ (genus 1) that gives a morphism $E \to \mathbb{P}^1$?

> [!TIP]- Solution to Exercise 16
> **Key insight:** A degree-1 pencil defines a degree-1 map to $\mathbb{P}^1$, hence an isomorphism, which forces genus 0.
>
> **Sketch:**
> (a) By Exercise 15, $\dim|D| = \deg D$ on $\mathbb{P}^1$, so $\dim|D|=1$ iff $\deg D=1$. Any $D$ of degree 1 gives a pencil; all are equivalent to $[\infty]$.
>
> (b) If $h^0(D)=2$ and $\deg D=1$, the map $\phi_D: X\to\mathbb{P}^1$ has degree $\deg D=1$ (the degree of a map equals $\deg D$ for a base-point-free system). A degree-1 morphism of curves is an isomorphism. But $\mathbb{P}^1$ has genus 0, while $X$ has genus $g\geq 1$; isomorphic curves have the same genus. Contradiction.
>
> (c) For $E$ of genus 1, Riemann-Roch gives $h^0(D)=\deg D$ for $\deg D\geq 1$. A pencil requires $h^0(D)=2$, so $\deg D=2$. The map $\phi_D: E\to\mathbb{P}^1$ has degree 2 (a *degree-2 map* or "hyperelliptic involution"). So the minimum degree is 2.

---

## 7. Algorithmic Applications 💻

This section gives algorithmic treatments of the main constructions. Code is written as executable Python pseudocode.

> [!QUESTION] Exercise 17: Computing div(f) on P^1
> *This problem implements the principal divisor algorithm for rational functions on P^1.*
>
> > **Prerequisites:** [[#3.1 Order of Vanishing|3.1 Order of Vanishing]]
>
> Write a Python function `compute_div(f_num, f_den, points)` that, given polynomials `f_num` and `f_den` representing a rational function $f = f_{\mathrm{num}}/f_{\mathrm{den}}$ on $\mathbb{P}^1$, and a list of affine points, computes $\mathrm{div}(f)$ as a dictionary `{point: multiplicity}` including the multiplicity at $\infty$.

> [!TIP]- Solution to Exercise 17
> **Key insight:** The multiplicity of $f$ at $P = a$ is $\mathrm{ord}_{x=a}(f_{\mathrm{num}}) - \mathrm{ord}_{x=a}(f_{\mathrm{den}})$, where $\mathrm{ord}_{x=a}(p)$ is the multiplicity of $a$ as a root of the polynomial $p$. The multiplicity at $\infty$ is $\deg(f_{\mathrm{den}}) - \deg(f_{\mathrm{num}})$.
>
> ```python
> from sympy import Poly, roots, Symbol, degree
>
> def multiplicity_at_point(poly, point, x):
>     """Return the multiplicity of 'point' as a root of poly."""
>     p = Poly(poly, x)
>     count = 0
>     while True:
>         q, r = p.div(Poly(x - point, x))
>         if r != 0:
>             break
>         p = q
>         count += 1
>     return count
>
> def compute_div(f_num, f_den, affine_points, x=Symbol('x')):
>     """
>     Compute div(f_num / f_den) on P^1 as a dict {point: order}.
>     'infty' key holds the order at infinity.
>     Positive order = zero, negative order = pole.
>     """
>     div = {}
>     for P in affine_points:
>         ord_num = multiplicity_at_point(f_num, P, x)
>         ord_den = multiplicity_at_point(f_den, P, x)
>         v = ord_num - ord_den
>         if v != 0:
>             div[P] = v
>     # Order at infinity: deg(den) - deg(num)
>     v_inf = degree(Poly(f_den, x)) - degree(Poly(f_num, x))
>     if v_inf != 0:
>         div['infty'] = v_inf
>     # Sanity check: total degree should be 0
>     assert sum(div.values()) == 0, "deg(div(f)) != 0!"
>     return div
>
> # Example: f = x*(x-1)/(x-2)^2 on P^1
> # Expected: div(f) = [0] + [1] - 2[2] + 0[infty]
> # (deg(num)=2, deg(den)=2, so v_inf=0)
> ```

> [!QUESTION] Exercise 18: Computing the Complete Linear System |D|
> *This problem implements an algorithm for finding the complete linear system.*
>
> > **Prerequisites:** [[#6.1 Complete Linear Systems|6.1 Complete Linear Systems]]
>
> Given a divisor $D = \sum n_i [a_i]$ on $\mathbb{P}^1$ with $\deg(D) = d \geq 0$ and a basis $\{s_0, \ldots, s_r\}$ of $H^0(\mathbb{P}^1, \mathcal{O}(D))$ (as polynomials of degree $\leq d$), write a function `effective_divisors_in_system(basis, D, num_samples)` that samples random elements of $|D|$ by taking random linear combinations of the basis sections.

> [!TIP]- Solution to Exercise 18
> **Key insight:** Each nonzero linear combination $f = \sum \lambda_i s_i$ of basis sections gives an effective divisor $\mathrm{div}(f) + D \in |D|$.
>
> ```python
> import numpy as np
> from sympy import Poly, Symbol, factor_list, prod
>
> def sample_linear_system(basis_polys, D_dict, num_samples=10, x=Symbol('x')):
>     """
>     Sample num_samples effective divisors from |D|.
>     basis_polys: list of sympy expressions forming a basis of H^0(O(D))
>     D_dict: dict {point_or_infty: multiplicity} for the fixed divisor D
>     Returns a list of dicts {point: multiplicity} for each sample.
>     """
>     samples = []
>     for _ in range(num_samples):
>         coeffs = np.random.randn(len(basis_polys))
>         f = sum(c * p for c, p in zip(coeffs, basis_polys))
>         # Compute div(f) + D to get the effective divisor D' ~ D
>         f_poly = Poly(f, x)
>         # Find roots of f (zeros in affine part)
>         zero_div = {}
>         for root, mult in factor_list(f_poly.as_expr())[1]:
>             # root is a linear factor (x - a), mult is multiplicity
>             a = -root.coeff(x, 0)  # extract the root value
>             zero_div[float(a)] = mult
>         # D' = div(f)_+ + (D + div(f)_-)  but for P^1 just return zeros of f
>         # combined with the contribution from D's positive part
>         # (simplification: effective_D' = div(f) + D, so only show nonzero entries)
>         effective = dict(D_dict)  # start with D
>         for pt, v in zero_div.items():
>             effective[pt] = effective.get(pt, 0) + v
>         # Remove zero-multiplicity entries
>         effective = {k: v for k, v in effective.items() if v != 0}
>         samples.append(effective)
>     return samples
>
> # Example usage for |2*[infty]| on P^1: basis = {1, x, x^2}
> from sympy.abc import x
> basis = [1, x, x**2]
> D = {'infty': 2}
> samples = sample_linear_system(basis, D, num_samples=5)
> ```

> [!QUESTION] Exercise 19: Computing the Base Locus
> *This problem implements an algorithm for finding the base locus of a linear system.*
>
> > **Prerequisites:** [[#6.2 Base Locus and the Map phi_D|6.2 Base Locus and the Map phi_D]]
>
> Given a basis $\{s_0,\ldots,s_r\}$ of $H^0(X,\mathcal{O}(D))$ as polynomials (for $X=\mathbb{P}^1$), write a function `base_locus(basis)` that computes the base locus as the set of common zeros of all basis elements.

> [!TIP]- Solution to Exercise 19
> **Key insight:** The base locus is $\mathrm{Bs}(|D|) = \{P : s(P) = 0 \text{ for all } s \in H^0(X,\mathcal{O}(D))\}$, which equals the set of common roots of all basis polynomials. This can be computed via the GCD.
>
> ```python
> from sympy import gcd, Poly, solve, Symbol
>
> def base_locus(basis_polys, x=Symbol('x')):
>     """
>     Compute the base locus (in affine P^1) of the linear system
>     spanned by basis_polys.
>     Returns a list of affine base points (roots of gcd of all basis elements).
>     'infty' is a base point iff all basis polys have the same degree
>     and leading coefficients sum to 0 (handled separately).
>     """
>     if not basis_polys:
>         return []
>     # Compute GCD of all basis polynomials
>     g = Poly(basis_polys[0], x)
>     for p in basis_polys[1:]:
>         g = gcd(g, Poly(p, x))
>     # Roots of g are the affine base points
>     affine_base_pts = solve(g.as_expr(), x)
>
>     # Check if infty is a base point:
>     # infty is a base point iff all sections vanish at infty,
>     # i.e., all basis polys have degree < deg(D) at infty
>     # For O(n)[infty], sections are polys of degree <= n;
>     # infty is a base point iff all sections have degree < n.
>     max_deg = max(Poly(p, x).degree() for p in basis_polys)
>     infty_base = all(Poly(p, x).degree() < max_deg for p in basis_polys)
>
>     result = list(affine_base_pts)
>     if infty_base:
>         result.append('infty')
>     return result
>
> # Example: basis = {x, x^2} for the subsystem from Exercise 11
> from sympy.abc import x
> print(base_locus([x, x**2]))  # Expected: [0]
> ```

> [!QUESTION] Exercise 20: The Map phi_D in Coordinates
> *This problem implements the map to projective space associated to a linear system.*
>
> > **Prerequisites:** [[#6.2 Base Locus and the Map phi_D|6.2 Base Locus and the Map phi_D]], [[#6.4 Examples: Veronese and Conics|6.4 Examples: Veronese and Conics]]
>
> Implement `phi_D(point, basis_polys)` that computes the image of an affine point $P = a$ under $\phi_D: \mathbb{P}^1 \to \mathbb{P}^r$. Then use this to verify the degree-2 Veronese $\nu_2: [s:t]\mapsto[s^2:st:t^2]$ is injective.

> [!TIP]- Solution to Exercise 20
> **Key insight:** Evaluate each basis section at the point and return the result as a projective coordinate vector. Injectivity is checked by confirming distinct inputs give non-proportional outputs.
>
> ```python
> from sympy import Symbol, simplify, Rational
> from itertools import combinations
>
> def phi_D(affine_point, basis_polys, x=Symbol('x')):
>     """
>     Evaluate phi_D at an affine point.
>     Returns a list of values [s0(P), s1(P), ..., sr(P)] in P^r.
>     """
>     return [float(p.subs(x, affine_point)) for p in basis_polys]
>
> def projective_equal(v1, v2, tol=1e-9):
>     """Check if two vectors represent the same point in P^r."""
>     # v1 ~ v2 iff v1[i]*v2[j] == v1[j]*v2[i] for all i,j
>     n = len(v1)
>     for i, j in combinations(range(n), 2):
>         if abs(v1[i]*v2[j] - v1[j]*v2[i]) > tol:
>             return False
>     return True
>
> def verify_veronese_injective(degree=2, num_test_points=50):
>     """
>     Sample random affine points and check that nu_d is injective on them.
>     Basis for O(d) on P^1: {1, x, x^2, ..., x^d}.
>     """
>     from sympy.abc import x
>     import numpy as np
>     basis = [x**i for i in range(degree + 1)]
>
>     test_pts = np.random.uniform(-10, 10, num_test_points)
>     images = [phi_D(p, basis) for p in test_pts]
>
>     injective = True
>     for i in range(num_test_points):
>         for j in range(i + 1, num_test_points):
>             if abs(test_pts[i] - test_pts[j]) > 1e-9:  # distinct points
>                 if projective_equal(images[i], images[j]):
>                     injective = False
>                     print(f"Injectivity fails at {test_pts[i]}, {test_pts[j]}")
>     if injective:
>         print(f"Degree-{degree} Veronese appears injective on {num_test_points} test points.")
>     return injective
>
> # Run the verification
> verify_veronese_injective(degree=2, num_test_points=50)
> ```

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Shafarevich, *Basic Algebraic Geometry* Vol. 1, §III.1 | Primary source; develops divisors on curves, linear systems, and Picard groups in the classical algebraic geometry setting | [SpringerLink](https://link.springer.com/book/10.1007/978-3-642-37956-7) |
| Hartshorne, *Algebraic Geometry*, §II.6–7 | Definitive modern treatment of Weil divisors, Cartier divisors, and invertible sheaves; proves the divisor–line bundle correspondence in full generality | [Springer](https://link.springer.com/book/10.1007/978-1-4757-3849-0) |
| Fulton, *Algebraic Curves* | Accessible introduction to divisors on curves with emphasis on the Riemann-Roch theorem; free PDF available | [Fulton's website](http://www.math.lsa.umich.edu/~wfulton/CurveBook.pdf) |
| Silverman, *Arithmetic of Elliptic Curves*, Ch. II §1–3 | Divisors on curves in the context of elliptic curves; explains the Picard group and its identification with the curve's group law | [Springer](https://link.springer.com/book/10.1007/978-0-387-09494-6) |
| Milne, *Algebraic Geometry* lecture notes, Ch. 12 | Freely available lecture notes covering divisors, intersection theory, and Picard groups; very complete and up-to-date | [jmilne.org](https://www.jmilne.org/math/CourseNotes/AG.pdf) |
| Reid, *Undergraduate Algebraic Geometry*, Ch. 9 | Elementary treatment of linear systems and maps to projective space, with worked examples | [Cambridge](https://www.cambridge.org/core/books/undergraduate-algebraic-geometry/F5E57E60A75C56F96B0CBC66B7C55A4B) |
| Wikipedia, *Divisor (algebraic geometry)* | Concise summary of Weil vs. Cartier divisors, Picard group, and the class group | [Wikipedia](https://en.wikipedia.org/wiki/Divisor_(algebraic_geometry)) |
| Vakil, *Foundations of Algebraic Geometry*, Ch. 27–32 | Stanford lecture notes with a thorough treatment of divisors, line bundles, and linear systems in scheme-theoretic language | [math.stanford.edu](https://math.stanford.edu/~vakil/0708-216/216class27.pdf) |
