# Hochschild Cohomology

## Table of Contents

- [[#1. The Hochschild Cochain Complex|1. The Hochschild Cochain Complex]]
  - [[#1.1 Setup and Definitions|1.1 Setup and Definitions]]
  - [[#1.2 The Coboundary Operator|1.2 The Coboundary Operator]]
  - [[#1.3 Nilpotency and Cohomology Groups|1.3 Nilpotency and Cohomology Groups]]
  - [[#1.4 Connection to Operadic Cohomology|1.4 Connection to Operadic Cohomology]]
- [[#2. Low-Degree Interpretations|2. Low-Degree Interpretations]]
  - [[#2.1 HH0: The Center|2.1 HH0: The Center]]
  - [[#2.2 HH1: Outer Derivations|2.2 HH1: Outer Derivations]]
  - [[#2.3 HH2: Extensions and Infinitesimal Deformations|2.3 HH2: Extensions and Infinitesimal Deformations]]
  - [[#2.4 HH3: Obstruction Theory|2.4 HH3: Obstruction Theory]]
- [[#3. The Cup Product and Gerstenhaber Algebra|3. The Cup Product and Gerstenhaber Algebra]]
  - [[#3.1 The Cup Product|3.1 The Cup Product]]
  - [[#3.2 The Pre-Lie Product and Gerstenhaber Bracket|3.2 The Pre-Lie Product and Gerstenhaber Bracket]]
  - [[#3.3 The Gerstenhaber Algebra Axioms|3.3 The Gerstenhaber Algebra Axioms]]
- [[#4. The Deligne Conjecture|4. The Deligne Conjecture]]
  - [[#4.1 Statement|4.1 Statement]]
  - [[#4.2 Proofs and the E2 Structure|4.2 Proofs and the E2 Structure]]
  - [[#4.3 Corollary: Gerstenhaber Structure on Cohomology|4.3 Corollary: Gerstenhaber Structure on Cohomology]]
- [[#5. Cyclic Cohomology|5. Cyclic Cohomology]]
  - [[#5.1 The Cyclic Operator and Cyclic Cochains|5.1 The Cyclic Operator and Cyclic Cochains]]
  - [[#5.2 The Connes B-Operator|5.2 The Connes B-Operator]]
  - [[#5.3 The SBI Long Exact Sequence|5.3 The SBI Long Exact Sequence]]
  - [[#5.4 Cyclic Cohomology for Commutative Algebras|5.4 Cyclic Cohomology for Commutative Algebras]]
- [[#6. The Hochschild-Kostant-Rosenberg Theorem|6. The Hochschild-Kostant-Rosenberg Theorem]]
  - [[#6.1 Statement|6.1 Statement]]
  - [[#6.2 The HKR Map|6.2 The HKR Map]]
  - [[#6.3 Consequences|6.3 Consequences]]
  - [[#6.4 Failure for Singular Varieties|6.4 Failure for Singular Varieties]]
- [[#7. Hochschild Cohomology and Deformation Theory|7. Hochschild Cohomology and Deformation Theory]]
  - [[#7.1 Formal Deformations and the Maurer-Cartan Equation|7.1 Formal Deformations and the Maurer-Cartan Equation]]
  - [[#7.2 Order-by-Order Expansion|7.2 Order-by-Order Expansion]]
  - [[#7.3 Rigidity and Formality|7.3 Rigidity and Formality]]
- [[#8. Operadic Cohomology as the General Framework|8. Operadic Cohomology as the General Framework]]
  - [[#8.1 The Two-Sided Bar Resolution for Ass|8.1 The Two-Sided Bar Resolution for Ass]]
  - [[#8.2 Analogues for Com and Lie|8.2 Analogues for Com and Lie]]
  - [[#8.3 Closing the Loop: the Deformation Complex of Ass|8.3 Closing the Loop: the Deformation Complex of Ass]]
- [[#References|References]]

---

## 1. The Hochschild Cochain Complex 📐

### 1.1 Setup and Definitions

Throughout this note, $k$ denotes a commutative ring (often a field of characteristic zero for the finer results), $A$ a unital associative $k$-algebra, and $M$ an $A$-*bimodule* — that is, a $k$-module equipped with compatible left and right $A$-actions satisfying $(a \cdot m) \cdot b = a \cdot (m \cdot b)$ for all $a, b \in A$, $m \in M$.

The central object of this note lives in [[concepts/category-theory/operads/koszul-duality|Koszul Duality and the Bar Construction]] §8 as the special case $\mathcal{O} = \mathrm{Ass}$ of operadic cohomology. We give the fully explicit, classical presentation here.

**Definition (Hochschild cochain group).** For $n \geq 0$, the *$n$-th Hochschild cochain group* of $A$ with coefficients in $M$ is

$$C^n(A, M) := \mathrm{Hom}_k(A^{\otimes n}, M),$$

where $A^{\otimes 0} := k$ by convention, so $C^0(A, M) = M$.

An element $f \in C^n(A, M)$ is called an *$n$-cochain*; it is a $k$-linear map $f : A^{\otimes n} \to M$.

> [!NOTE] Comparison with the bar complex
> The Hochschild cochain complex $C^\bullet(A, M)$ is the linear dual of the Hochschild *chain* complex $C_\bullet(A, M)$, where $C_n(A, M) = M \otimes_k A^{\otimes n}$. The chain complex computes Hochschild *homology* $HH_\bullet(A, M)$, while we focus on the cohomological side here.

### 1.2 The Coboundary Operator

**Definition (Hochschild coboundary).** The *coboundary map* $\delta : C^n(A, M) \to C^{n+1}(A, M)$ is defined by

$$(\delta f)(a_1, \ldots, a_{n+1}) = a_1 \cdot f(a_2, \ldots, a_{n+1}) + \sum_{i=1}^{n} (-1)^i f(a_1, \ldots, a_i a_{i+1}, \ldots, a_{n+1}) + (-1)^{n+1} f(a_1, \ldots, a_n) \cdot a_{n+1}.$$

In degrees 0 and 1 this reads explicitly:
- **Degree 0:** $(\delta m)(a) = a \cdot m - m \cdot a$ for $m \in M = C^0(A, M)$.
- **Degree 1:** $(\delta f)(a_1, a_2) = a_1 \cdot f(a_2) - f(a_1 a_2) + f(a_1) \cdot a_2$ for $f \in C^1(A,M)$.

> [!INFO] Sign convention and indexing
> Some sources (including Weibel §9.1) index the Hochschild cochain groups as $C^n(A,M) = \mathrm{Hom}_k(A^{\otimes n}, M)$ and write the coboundary with input variables labeled $a_0, \ldots, a_{n-1}$. We follow the convention that is more natural for deformation theory (inputs labeled $a_1, \ldots, a_n$), which makes the Maurer-Cartan equation in §7 cleaner. The two conventions differ only in notation.

### 1.3 Nilpotency and Cohomology Groups

**Proposition ($\delta^2 = 0$).** The Hochschild coboundary satisfies $\delta \circ \delta = 0$, making $(C^\bullet(A,M), \delta)$ a cochain complex.

*Proof sketch.* One computes $(\delta^2 f)(a_1, \ldots, a_{n+2})$ directly. The sum splits into four types of terms:
1. Left action of $a_1$ on $(\delta f)(a_2, \ldots)$, which introduces terms with consecutive multiplications $a_i a_{i+1}$ for $i \geq 2$ and a right-action term.
2. Multiplication terms $(\delta f)(\ldots, a_j a_{j+1}, \ldots)$ for $j \geq 1$, each producing further left-action, multiplication, and right-action sub-terms.
3. Right-action terms.

Each pair of consecutive sign indices produces a cancelling pair with opposite signs, by the standard simplicial identity $d_i d_j = d_{j+1} d_i$ for $i \leq j$ (here $d_i$ is the $i$-th face map). Precisely, for $0 \leq i < j \leq n+1$ the contribution from applying face $d_j$ then $d_i$ cancels against the contribution from applying $d_{i+1}$ then $d_j$, since $(-1)^i (-1)^j + (-1)^{j+1} (-1)^i = 0$. The full telescoping yields $\delta^2 = 0$. $\square$

**Definition (Hochschild cohomology).** The *$n$-th Hochschild cohomology group* of $A$ with coefficients in $M$ is

$$HH^n(A, M) := H^n(C^\bullet(A, M), \delta) = \frac{\ker(\delta : C^n \to C^{n+1})}{\mathrm{im}(\delta : C^{n-1} \to C^n)}.$$

Elements of $\ker \delta$ are *$n$-cocycles*; elements of $\mathrm{im} \delta$ are *$n$-coboundaries*.

### 1.4 Connection to Operadic Cohomology

The Hochschild complex is the prototypical example of operadic cohomology for the operad $\mathrm{Ass}$. Recall from [[concepts/category-theory/operads/koszul-duality|Koszul Duality and the Bar Construction]] §8.1 that for an operad $\mathcal{O}$, an $\mathcal{O}$-algebra $A$, and an $A$-module $M$, the *operadic cohomology* $H^\bullet_\mathcal{O}(A, M)$ is computed by the two-sided bar resolution $B(\mathcal{O}^!, A, M)$, where $\mathcal{O}^!$ is the Koszul dual cooperad.

**Theorem (Hochschild = operadic cohomology for Ass).** There is a natural isomorphism

$$HH^\bullet(A, M) \cong H^\bullet_{\mathrm{Ass}}(A, M),$$

where the right-hand side is computed by the two-sided bar complex $B(\mathrm{Ass}^!, A, M) = B(\mathrm{Com}, A, M)$, using the fact that $\mathrm{Ass}^! \cong \mathrm{Ass}$ (the Koszul dual of $\mathrm{Ass}$ as a cooperad is $\mathrm{Com}$, reflecting the duality $\mathrm{Ass}^\vee \cong \mathrm{Ass}$). The explicit generators of the bar complex in arity $n$ are $k$-linear maps $A^{\otimes n} \to M$, recovering $C^n(A,M)$, and the bar differential recovers $\delta$ precisely.

*This is the crucial point*: every special feature of Hochschild cohomology — the Gerstenhaber algebra structure, the deformation interpretation, the HKR theorem — is a manifestation of the rich structure of the operad $\mathrm{Ass}$.

> [!TIP] Which bar complex?
> The two-sided bar complex $B(\mathcal{O}^!, A, M)$ is a *free* resolution of $M$ as an $A$-module, and the Hochschild cochain complex is recovered by applying $\mathrm{Hom}_{A\text{-bimod}}(-, M)$ to the standard bar resolution $B(A) := A \otimes_k \bar{A}^{\otimes \bullet} \otimes_k A$ of $A$ as an $A$-bimodule (where $\bar{A} = A/k \cdot 1$). The augmented normalized bar resolution is the standard tool for explicit computations.

---

> [!QUESTION] Exercise 1: Verifying the Coboundary in Low Degrees
> *This problem establishes that the degree-0 coboundary encodes commutativity, giving a direct check that $\delta^2 = 0$ on $C^0 \to C^1 \to C^2$.*
>
> > **Prerequisites:** [[#1.2 The Coboundary Operator|1.2 The Coboundary Operator]]
>
> Let $m \in M = C^0(A,M)$. Compute $(\delta^2 m)(a_1, a_2) := (\delta(\delta m))(a_1,a_2)$ explicitly and verify it equals zero by expanding both applications of $\delta$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** The two contributions from forming $\delta(\delta m)$ cancel by associativity and bimodule compatibility.
>
> **Sketch:** We have $(\delta m)(a) = a \cdot m - m \cdot a$. Then
> $$(\delta(\delta m))(a_1, a_2) = a_1 \cdot (\delta m)(a_2) - (\delta m)(a_1 a_2) + (\delta m)(a_1) \cdot a_2.$$
> Substituting:
> $$= a_1(a_2 m - m a_2) - (a_1 a_2 m - m a_1 a_2) + (a_1 m - m a_1) a_2$$
> $$= a_1 a_2 m - a_1 m a_2 - a_1 a_2 m + m a_1 a_2 + a_1 m a_2 - m a_1 a_2 = 0.$$
> All six terms cancel in pairs, confirming $\delta^2 = 0$ on $C^0$.

---

## 2. Low-Degree Interpretations 🔑

### 2.1 HH0: The Center

**Proposition (HH0 = center).** There is a natural identification

$$HH^0(A, M) = \{m \in M : a \cdot m = m \cdot a \text{ for all } a \in A\},$$

the *center* $Z(A, M)$ of $M$ as an $A$-bimodule. In particular, $HH^0(A, A) = Z(A)$, the center of the algebra $A$.

*Proof.* Since $C^{-1}(A,M) = 0$, every $m \in C^0(A,M) = M$ is trivially a coboundary-free cochain; the cocycle condition $\delta m = 0$ reads $a \cdot m - m \cdot a = 0$ for all $a \in A$. $\square$

### 2.2 HH1: Outer Derivations

**Definition (Derivation).** A *derivation* of $A$ with values in $M$ is a $k$-linear map $d : A \to M$ satisfying the *Leibniz rule*:

$$d(ab) = a \cdot d(b) + d(a) \cdot b \quad \text{for all } a, b \in A.$$

The $k$-module of all such derivations is denoted $\mathrm{Der}(A, M)$. An *inner derivation* is one of the form $d_m(a) = a \cdot m - m \cdot a$ for some fixed $m \in M$.

**Proposition (HH1 = outer derivations).** There is a natural isomorphism

$$HH^1(A, M) \cong \mathrm{Der}(A, M) / \mathrm{InnDer}(A, M).$$

*Proof.* We show cocycles $= \mathrm{Der}(A,M)$ and coboundaries $= \mathrm{InnDer}(A,M)$.

**Cocycles are derivations.** Let $f \in C^1(A,M) = \mathrm{Hom}_k(A, M)$. Then

$$(\delta f)(a_1, a_2) = a_1 \cdot f(a_2) - f(a_1 a_2) + f(a_1) \cdot a_2.$$

Setting $\delta f = 0$ gives exactly $f(a_1 a_2) = a_1 \cdot f(a_2) + f(a_1) \cdot a_2$, which is the Leibniz rule.

**Coboundaries are inner.** An element $\delta m \in C^1(A,M)$ for $m \in M$ satisfies $(\delta m)(a) = a \cdot m - m \cdot a = d_m(a)$. This is precisely the inner derivation associated to $m$.

Combining: $HH^1(A,M) = \ker \delta / \mathrm{im} \delta = \mathrm{Der}(A,M) / \mathrm{InnDer}(A,M)$. $\square$

> [!INFO] Shannon entropy as a 1-cocycle
> The connection to [[research/entropy-operad-derivation|Entropy as an Operad Derivation]] is immediate: Shannon entropy $H(p_1, \ldots, p_n) = -\sum p_i \log p_i$ satisfies a chain rule that is precisely a derivation condition for the probability simplex algebra. In the operadic setting, entropy is a derivation of the operad $\mathrm{Fin}$ of finite sets; its restriction to associative structure identifies it as a Hochschild 1-cocycle. *See the research thread for the full derivation.*

### 2.3 HH2: Extensions and Infinitesimal Deformations

**Definition (Singular extension).** A *singular extension* of $A$ by $M$ is a short exact sequence of $k$-algebras

$$0 \to M \to E \xrightarrow{\pi} A \to 0,$$

where $M$ is given the zero-product algebra structure ($M^2 = 0$), and the $A$-bimodule structure on $M$ is induced by choosing a $k$-linear section $s : A \to E$ of $\pi$.

Two extensions are *equivalent* if there is an algebra isomorphism $\phi : E \to E'$ commuting with the inclusion of $M$ and projection to $A$.

**Theorem (HH2 classifies extensions).** There is a bijection

$$HH^2(A, M) \xrightarrow{\sim} \{\text{equivalence classes of singular extensions of } A \text{ by } M\}.$$

*Proof sketch.* Given a $k$-linear section $s : A \to E$, the failure of $s$ to be an algebra map is measured by

$$\mu_1(a, b) := s(a) s(b) - s(ab) \in M \quad (M \hookrightarrow E),$$

since $\pi(s(a)s(b) - s(ab)) = ab - ab = 0$. One checks that $\mu_1 \in C^2(A,M)$.

Associativity of $E$ forces $\delta \mu_1 = 0$: indeed, $(s(a)s(b))s(c) = s(a)(s(b)s(c))$ implies

$$\mu_1(a,b) \cdot c - \mu_1(ab, c) + \mu_1(a, bc) - a \cdot \mu_1(b,c) = 0,$$

which is $(\delta \mu_1)(a,b,c) = 0$. A different choice of section $s' = s + \phi$ (where $\phi : A \to M$) changes $\mu_1$ by $\delta\phi$. Thus the cohomology class $[\mu_1] \in HH^2(A,M)$ is well-defined and classifies the extension. One constructs an inverse: given a 2-cocycle $\mu_1$, set $E = M \oplus A$ as $k$-modules with multiplication $(m, a)(n, b) = (m \cdot b + a \cdot n + \mu_1(a,b), ab)$. Associativity of this product is equivalent to $\delta \mu_1 = 0$. $\square$

**Corollary (HH2 classifies infinitesimal deformations).** An *infinitesimal deformation* of $A$ is a $k[\epsilon]/(\epsilon^2)$-algebra $A_\epsilon = (A[\epsilon]/(\epsilon^2), \star)$ with

$$a \star b = ab + \epsilon \mu_1(a, b), \quad \mu_1 \in C^2(A, A).$$

Associativity of $\star$ forces $\delta \mu_1 = 0$. Two deformations $\mu_1, \mu_1'$ are isomorphic (via a $k[\epsilon]/(\epsilon^2)$-algebra isomorphism $\phi_\epsilon = \mathrm{id} + \epsilon \phi_1$) if and only if $\mu_1 - \mu_1' = \delta \phi_1$. **Thus $HH^2(A,A)$ is in bijection with isomorphism classes of infinitesimal deformations of $A$.**

### 2.4 HH3: Obstruction Theory

**Theorem (HH3 = obstructions).** Let $\mu_1 \in Z^2(A,A)$ be a 2-cocycle representing an infinitesimal deformation. An extension to a second-order deformation $\star = ab + t\mu_1(a,b) + t^2 \mu_2(a,b) + O(t^3)$ over $k[[t]]/(t^3)$ exists if and only if a certain *obstruction class* $\mathrm{obs}(\mu_1) \in HH^3(A,A)$ vanishes.

*Proof sketch.* Associativity of $a \star (b \star c) = (a \star b) \star c$ at order $t^2$ yields the equation

$$\delta \mu_2(a,b,c) = \frac{1}{2} [\mu_1, \mu_1]_G(a,b,c),$$

where $[-,-]_G$ is the Gerstenhaber bracket defined in §3. The right-hand side is always a 3-cocycle (by the Jacobi identity for $[-,-]_G$). A solution $\mu_2$ exists if and only if $[\mu_1, \mu_1]_G$ is a coboundary, i.e., $[[\mu_1, \mu_1]_G] = 0 \in HH^3(A,A)$. *If $HH^3(A,A) = 0$ then every infinitesimal deformation extends to all orders.* $\square$

> [!WARNING] HH3 = 0 is sufficient but not necessary
> The vanishing of $HH^3(A,A)$ guarantees that no obstruction can arise at any order — a deformation can always be extended. However, a deformation may extend even when $HH^3 \neq 0$: the obstruction class at each order could individually vanish without the whole group being trivial.

---

> [!QUESTION] Exercise 2: Inner Derivations Form a Submodule
> *This problem establishes that $\mathrm{InnDer}(A,M)$ is genuinely a submodule of $\mathrm{Der}(A,M)$, so the quotient in $HH^1$ is well-defined.*
>
> > **Prerequisites:** [[#2.2 HH1: Outer Derivations|2.2 HH1: Outer Derivations]]
>
> Verify directly (without invoking the cochain complex) that $d_m(a) = am - ma$ satisfies the Leibniz rule for all $m \in M$ and all $a, b \in A$.

> [!TIP]- Solution to Exercise 2
> **Key insight:** The Leibniz rule for $d_m$ follows immediately from associativity and bimodule axioms.
>
> **Sketch:** $d_m(ab) = (ab)m - m(ab) = a(bm) - (ma)b = a(bm - mb) + (am - ma)b = a \cdot d_m(b) + d_m(a) \cdot b$. Every step uses only associativity of $A$ and the bimodule identity $(am)b = a(mb)$.

> [!QUESTION] Exercise 3: The Cocycle Condition for HH2
> *This problem makes the extension-to-cocycle correspondence explicit by deriving the cocycle condition from algebra associativity.*
>
> > **Prerequisites:** [[#2.3 HH2: Extensions and Infinitesimal Deformations|2.3 HH2: Extensions and Infinitesimal Deformations]]
>
> Let $E = M \oplus A$ with multiplication $(m,a)(n,b) = (mb + an + \mu_1(a,b), ab)$ for $\mu_1 \in C^2(A,A)$. Compute $(e_1 e_2) e_3 - e_1 (e_2 e_3)$ for $e_i = (0, a_i)$ and show it equals $(-(\delta \mu_1)(a_1,a_2,a_3), 0)$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** All $M$-components land in $M$ because $M^2 = 0$; associativity in $A$ handles the $A$-component.
>
> **Sketch:** $(e_1 e_2) = (\mu_1(a_1,a_2), a_1 a_2)$. Then $(e_1 e_2)e_3 = (\mu_1(a_1,a_2) a_3 + \mu_1(a_1 a_2, a_3), a_1 a_2 a_3)$. Similarly $e_2 e_3 = (\mu_1(a_2,a_3), a_2 a_3)$ and $e_1(e_2 e_3) = (a_1 \mu_1(a_2,a_3) + \mu_1(a_1, a_2 a_3), a_1 a_2 a_3)$. The difference in $M$-components is $\mu_1(a_1,a_2) a_3 + \mu_1(a_1 a_2, a_3) - a_1 \mu_1(a_2, a_3) - \mu_1(a_1, a_2 a_3) = -(\delta \mu_1)(a_1,a_2,a_3)$, which must vanish for $E$ to be associative.

---

## 3. The Cup Product and Gerstenhaber Algebra 📐

### 3.1 The Cup Product

The special case $M = A$ (coefficients in $A$ itself) endows $C^\bullet(A,A)$ with additional multiplicative structure. Here $A$ is simultaneously the coefficient bimodule and the algebra, and the product in $A$ can be used to compose cochains.

**Definition (Cup product).** For $f \in C^p(A,A)$ and $g \in C^q(A,A)$, define the *cup product* $f \smile g \in C^{p+q}(A,A)$ by

$$(f \smile g)(a_1, \ldots, a_{p+q}) = f(a_1, \ldots, a_p) \cdot g(a_{p+1}, \ldots, a_{p+q}),$$

where the dot on the right is the product in $A$.

**Proposition (Leibniz rule for the cup product).** The coboundary satisfies

$$\delta(f \smile g) = (\delta f) \smile g + (-1)^p f \smile (\delta g).$$

*Proof sketch.* Expand $(\delta(f \smile g))(a_1, \ldots, a_{p+q+1})$ using the definition of $\delta$. The left-action term $a_1 \cdot (f \smile g)(a_2, \ldots)$ gives rise to $a_1 \cdot f(a_2, \ldots, a_{p+1}) \cdot g(a_{p+2}, \ldots)$, which belongs to the $(\delta f) \smile g$ expansion. The middle multiplication terms split at the break point $i = p$: for $i < p$ they contribute to $(\delta f) \smile g$, and for $i > p$ they contribute to $(-1)^p f \smile (\delta g)$. The term at $i = p$ produces $f(a_1, \ldots, a_{p-1}, a_p) \cdot g(a_{p+1}, \ldots)$, which from the left factor side gives $f(\ldots, a_{p-1} a_p)$ times $g$ and from the right factor side gives $f(\ldots, a_p) \cdot g(a_{p+1} a_{p+2}, \ldots)$ — these appear with signs that match the Leibniz sign. The right-action term completes the $(-1)^p f \smile (\delta g)$ side. $\square$

**Corollary.** The cup product descends to cohomology, making $HH^\bullet(A,A)$ into a graded ring. Moreover, the cup product is *graded-commutative* on cohomology: $[f \smile g] = (-1)^{pq} [g \smile f]$.

> [!WARNING] Cup product is not graded-commutative at chain level
> The commutativity $f \smile g \sim (-1)^{pq} g \smile f$ holds only up to a coboundary — not as an equality of cochains. This is analogous to the cup product in singular cohomology of spaces.

### 3.2 The Pre-Lie Product and Gerstenhaber Bracket

To capture the full Lie-algebraic structure, one introduces a finer operation on cochains.

**Definition (Pre-Lie product).** For $f \in C^p(A,A)$ and $g \in C^q(A,A)$, the *pre-Lie product* (or *circle product*) $f \circ g \in C^{p+q-1}(A,A)$ is

$$(f \circ g)(a_1, \ldots, a_{p+q-1}) = \sum_{i=0}^{p-1} (-1)^{i(q-1)} f(a_1, \ldots, a_i, g(a_{i+1}, \ldots, a_{i+q}), a_{i+q+1}, \ldots, a_{p+q-1}).$$

This inserts $g$ into each of the $p$ slots of $f$, with the sign $(-1)^{i(q-1)}$ accounting for the graded nature of the insertion.

**Definition (Gerstenhaber bracket).** The *Gerstenhaber bracket* of $f \in C^p(A,A)$ and $g \in C^q(A,A)$ is

$$[f, g]_G = f \circ g - (-1)^{(p-1)(q-1)} g \circ f \in C^{p+q-1}(A,A).$$

**Proposition (Properties of the bracket).** On $C^\bullet(A,A)$, the bracket $[-,-]_G$ satisfies:
1. *Graded skew-symmetry:* $[f,g]_G = -(-1)^{(p-1)(q-1)} [g,f]_G$.
2. *Graded Jacobi identity:* $[f, [g,h]_G]_G = [[f,g]_G, h]_G + (-1)^{(p-1)(q-1)}[g, [f,h]_G]_G$.
3. *Bracket vs. coboundary:* $\delta[f,g]_G = [\delta f, g]_G + (-1)^{p-1} [f, \delta g]_G$ (i.e., $\delta$ is a derivation of the bracket).

*Proof sketch for (2).* This follows from the pre-Lie associator identity: $(f \circ g) \circ h - f \circ (g \circ h) = (f \circ h) \circ g - f \circ (h \circ g)$ (symmetry of the "failure of associativity"). Antisymmetrizing recovers the Jacobi identity. $\square$

> [!EXAMPLE] The bracket in degree 2
> For $\mu, \nu \in C^2(A,A)$, the pre-Lie product $\mu \circ \nu \in C^3(A,A)$ is
> $$(\mu \circ \nu)(a,b,c) = \mu(\nu(a,b),c) - \mu(a, \nu(b,c)).$$
> Thus $[\mu,\nu]_G(a,b,c) = \mu(\nu(a,b),c) - \mu(a,\nu(b,c)) - \nu(\mu(a,b),c) + \nu(a,\mu(b,c))$.
> This is the expression that appears in the Maurer-Cartan equation in §7.

### 3.3 The Gerstenhaber Algebra Axioms

**Definition (Gerstenhaber algebra).** A *Gerstenhaber algebra* is a graded $k$-module $G = \bigoplus_{n \geq 0} G^n$ equipped with:
- A *graded-commutative* associative product $\smile : G^p \otimes G^q \to G^{p+q}$.
- A degree $(-1)$ Lie bracket $[-,-]_G : G^p \otimes G^q \to G^{p+q-1}$, satisfying graded skew-symmetry and the graded Jacobi identity.
- A *Poisson compatibility* (graded Leibniz rule): for $f \in G^p$,

$$[f, g \smile h]_G = [f, g]_G \smile h + (-1)^{(p-1)|g|} g \smile [f, h]_G.$$

**Theorem (Gerstenhaber, 1963).** The triple $(HH^\bullet(A,A), \smile, [-,-]_G)$ is a Gerstenhaber algebra.

*Proof sketch.* The graded-commutativity of $\smile$ and graded Jacobi identity for $[-,-]_G$ both hold on cohomology (they hold up to coboundary at chain level, and coboundaries vanish in cohomology). The Poisson compatibility follows from the Leibniz property of $\delta$ with respect to both operations. $\square$

**Remark.** The degree shift is conventional: if one sets $\tilde{G}^n = G^{n+1}$, then $[-,-]_G$ becomes a degree $+1$ Lie bracket on $\tilde{G}^\bullet$, matching the convention in which a Gerstenhaber algebra is a graded-commutative algebra with a degree $+1$ Poisson bracket. Both conventions appear in the literature.

---

> [!QUESTION] Exercise 4: The Associator Identity for Pre-Lie Products
> *This problem establishes the pre-Lie identity that underlies the Jacobi identity for the Gerstenhaber bracket.*
>
> > **Prerequisites:** [[#3.2 The Pre-Lie Product and Gerstenhaber Bracket|3.2 The Pre-Lie Product and Gerstenhaber Bracket]]
>
> For $f \in C^2(A,A)$ and $g \in C^2(A,A)$, verify directly that $(f \circ g) \circ h - f \circ (g \circ h) = (f \circ h) \circ g - f \circ (h \circ g)$ holds as an identity in $C^3(A,A)$ when all inputs have degree 2. (Work in degree 2 for concreteness; the general case is the same calculation with signs.)

> [!TIP]- Solution to Exercise 4
> **Key insight:** Both sides count all ways to insert two 2-cochains into a 2-cochain, and the double insertions appear on both sides with matching signs.
>
> **Sketch:** For degree-2 cochains, the pre-Lie product $f \circ g \in C^3(A,A)$ has terms $f(g(a_1,a_2), a_3)$ and $f(a_1, g(a_2,a_3))$ (with sign $(-1)^0 = 1$ and $(-1)^1 = -1$ respectively). Then $(f \circ g) \circ h$ inserts $h$ into $f \circ g$ at all positions; expanding gives eight terms. Similarly for $f \circ (g \circ h)$, $(f \circ h) \circ g$, and $f \circ (h \circ g)$. One checks that the four sets of eight terms satisfy the asserted equality by matching every term on the left with its counterpart on the right.

> [!QUESTION] Exercise 5: Graded Commutativity of the Cup Product
> *This problem proves that $f \smile g - (-1)^{pq} g \smile f$ is always a coboundary, establishing graded commutativity in cohomology.*
>
> > **Prerequisites:** [[#3.1 The Cup Product|3.1 The Cup Product]]
>
> Define the homotopy $H(f, g)(a_1, \ldots, a_{p+q-1}) = \sum_{i=1}^{p} (-1)^{i(q-1)+q} f(a_{i+1}, \ldots, a_{i+p}) g(a_1, \ldots, a_i, a_{i+p+1}, \ldots)$ (this is a specific formula; you need not derive it from scratch). Verify that $\delta H(f,g) = f \smile g - (-1)^{pq} g \smile f$ holds at chain level by a direct sign chase for small $p = q = 1$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** For $p = q = 1$, the homotopy reduces to a single term, and the calculation is a direct coboundary computation.
>
> **Sketch:** With $p = q = 1$, $H(f,g)(a_1) = -f(a_2) \cdot g(a_1)$ (only $i=1$ contributes). Then $(\delta H)(a_1, a_2) = a_1 H(f,g)(a_2) - H(f,g)(a_1 a_2) + H(f,g)(a_1) a_2 = -a_1 f(a_2) g(a_1\ldots) + f(a_1 a_2) g(?) - f(a_2) g(a_1) a_2$. Matching with $(f \smile g)(a_1,a_2) - (g \smile f)(a_1,a_2) = f(a_1) g(a_2) - g(a_1) f(a_2)$ confirms the identity modulo coboundaries by explicit expansion of the degree-1 coboundary formula.

---

## 4. The Deligne Conjecture 💡

### 4.1 Statement

In a 1993 letter, Pierre Deligne asked whether the Hochschild cochain complex $C^\bullet(A,A)$ carries an action of the *little 2-disks operad* $\mathcal{D}_2$ (also called the $E_2$ operad) at the chain level — not merely the Gerstenhaber structure visible on cohomology. This became known as the Deligne conjecture.

**Conjecture/Theorem (Deligne, proven 1998–2000).** For any associative $k$-algebra $A$, the Hochschild cochain complex $C^\bullet(A,A)$ admits a natural structure of an algebra over the little 2-disks operad $\mathcal{D}_2$ (equivalently, an $E_2$-algebra structure). This chain-level structure descends on passing to cohomology to the Gerstenhaber algebra structure on $HH^\bullet(A,A)$.

The *little $n$-disks operad* $\mathcal{D}_n$ has $\mathcal{D}_n(k)$ = the space of configuration of $k$ non-overlapping disks in the unit $n$-disk. An *$E_n$-algebra* is an algebra over (a chain-level model of) $\mathcal{D}_n$. An $E_1$-algebra is an $A_\infty$-algebra; an $E_2$-algebra has both an $A_\infty$-structure and a compatible (homotopy) commutativity one degree down.

### 4.2 Proofs and the E2 Structure

The Deligne conjecture has been proven by multiple teams:

- **McClure–Smith (2002)**: Using the *surjection operad*, a combinatorial model for $E_2$, acting on the Hochschild complex via explicit operations on cochains.
- **Tamarkin (1998)**: Using the formality of $\mathcal{D}_2$ over $\mathbb{Q}$ (equivalently, the fact that $\mathcal{D}_2$ is formal as an operad) to transport the $E_2$-structure to $C^\bullet(A,A)$.
- **Kontsevich–Soibelman**: Via explicit formulas using configuration spaces and compactifications.
- **Voronov**: Using the *Swiss-cheese operad*.

The key mechanism: $C^\bullet(A,A)$ is a *brace algebra*, which is an algebra over a certain combinatorial operad $\mathrm{Br}$ whose spaces of operations $\mathrm{Br}(k)$ are spanned by rooted bipartite planar trees. The brace operations encode all ways to "insert" cochains into one another with correct signs, and the operad $\mathrm{Br}$ receives a quasi-isomorphism from the chains $C_\bullet(\mathcal{D}_2)$ on the little 2-disks operad.

> [!INFO] What is an E2-algebra, concretely?
> An $E_2$-algebra structure on a cochain complex $V$ amounts to: (1) an $A_\infty$-algebra structure (encoding homotopy-associativity), and (2) a collection of higher homotopies witnessing commutativity of the product up to all higher coherences. On homology, $H^\bullet(V)$ inherits a genuine graded-commutative algebra and a degree $(-1)$ Lie bracket — i.e., a Gerstenhaber algebra.

### 4.3 Corollary: Gerstenhaber Structure on Cohomology

**Corollary.** Since the homology of an $E_2$-algebra is a Gerstenhaber algebra (this is a formal consequence of the operad theory: the homology of $C_\bullet(\mathcal{D}_2)$ is the Gerstenhaber operad), the Deligne conjecture implies that $HH^\bullet(A,A)$ is a Gerstenhaber algebra.

*This recovers Gerstenhaber's 1963 theorem as a corollary of the operadic perspective.*

The Deligne conjecture also generalizes: for any $A_\infty$-algebra $(A, \{m_n\})$, its Hochschild cochain complex is an $E_2$-algebra (Kontsevich–Soibelman, Tamarkin). This is the starting point for the formality theorem in deformation quantization.

---

> [!QUESTION] Exercise 6: The Cup Product as an E1 Structure
> *This problem establishes that the cup product alone gives $C^\bullet(A,A)$ the structure of a (homotopy) associative algebra, i.e., an $E_1$-algebra.*
>
> > **Prerequisites:** [[#4.1 Statement|4.1 Statement]], [[#3.1 The Cup Product|3.1 The Cup Product]]
>
> Show that the cup product $\smile$ on $C^\bullet(A,A)$ is associative at chain level (not merely up to homotopy). Conclude that $(C^\bullet(A,A), \smile)$ is already an $E_1$-algebra (a strict dg-associative algebra, not just an $A_\infty$-algebra).

> [!TIP]- Solution to Exercise 6
> **Key insight:** Associativity of $\smile$ is immediate from associativity of the product in $A$.
>
> **Sketch:** $((f \smile g) \smile h)(a_1, \ldots, a_{p+q+r}) = f(a_1,\ldots,a_p) \cdot g(a_{p+1},\ldots,a_{p+q}) \cdot h(a_{p+q+1},\ldots,a_{p+q+r})$, which equals $(f \smile (g \smile h))(a_1,\ldots)$ by associativity in $A$. Hence $\smile$ is strictly associative at chain level, and $(C^\bullet(A,A), \smile, \delta)$ is a dg-algebra, the most basic form of an $E_1$-algebra. The $E_2$-structure of the Deligne conjecture is the *additional* homotopy-commutativity data witnessed by the Gerstenhaber bracket.

---

## 5. Cyclic Cohomology 📐

Cyclic cohomology, introduced independently by Alain Connes and Boris Tsygan in the 1980s, is a refinement of Hochschild cohomology that captures additional information related to traces, index theory, and the de Rham cohomology of the "noncommutative space" represented by $A$.

### 5.1 The Cyclic Operator and Cyclic Cochains

We specialize to $M = A^* := \mathrm{Hom}_k(A, k)$ (the linear dual of $A$), identifying $C^n(A, A^*) \cong \mathrm{Hom}_k(A^{\otimes n+1}, k)$ via $f(a_0, \ldots, a_n) = (f(a_1, \ldots, a_n))(a_0)$.

**Definition (Cyclic operator).** The *cyclic operator* $\lambda : C^n(A, A^*) \to C^n(A, A^*)$ (equivalently, on $\mathrm{Hom}_k(A^{\otimes n+1}, k)$) is defined by

$$(\lambda f)(a_0, a_1, \ldots, a_n) = (-1)^n f(a_n, a_0, a_1, \ldots, a_{n-1}).$$

This implements a cyclic rotation of the inputs, with the sign $(-1)^n$ ensuring compatibility with the coboundary.

**Definition (Cyclic cochains and cyclic cohomology).** The *cyclic cochains* are the fixed-points of $\lambda$:

$$C^n_\lambda(A) := \ker(1 - \lambda) \subset C^n(A, A^*).$$

These are functions $f : A^{\otimes n+1} \to k$ satisfying $f(a_0, \ldots, a_n) = (-1)^n f(a_n, a_0, \ldots, a_{n-1})$ (*cyclic symmetry*). The key fact is that $\delta$ preserves the cyclic eigenspaces, so $(C^\bullet_\lambda(A), \delta)$ is a sub-cochain complex. The *cyclic cohomology* of $A$ is

$$HC^n(A) := H^n(C^\bullet_\lambda(A), \delta).$$

> [!INFO] Connes' original definition via the cyclic bicomplex
> Connes originally defined cyclic cohomology via the double complex with columns given by $C^\bullet(A, A^*)$ and rows given by the operator $1 - \lambda$ and the norm map $N = \sum_{i=0}^n \lambda^i$. The total cohomology of this bicomplex recovers $HC^\bullet(A)$ and encodes the periodicity structure.

### 5.2 The Connes B-Operator

**Definition (Connes' $B$-operator).** The operator $B : C^n(A, A^*) \to C^{n-1}(A, A^*)$ (a map *decreasing* degree by 1) is defined by

$$(Bf)(a_0, \ldots, a_{n-1}) = \sum_{i=0}^{n-1} (-1)^{ni} f(1, a_i, a_{i+1}, \ldots, a_{n-1}, a_0, \ldots, a_{i-1}),$$

where $1 \in A$ is the unit. Equivalently, $B = (1 - (-1)^n \lambda) \circ s \circ N$ where $s$ is the extra degeneracy (insertion of 1) and $N$ is the cyclic norm.

**Proposition.** The operators $b = \delta$ (Hochschild coboundary) and $B$ satisfy:
1. $B^2 = 0$.
2. $b B + B b = 0$.

These make $(C^\bullet(A, A^*), b, B)$ a *mixed complex* — a cochain complex for $b$ equipped with a chain map $B$ satisfying $B^2 = 0$.

*Proof sketch.* $B^2 = 0$ follows from $N(1 - \lambda) = 0$. For $bB + Bb = 0$: this is a direct consequence of the simplicial identities for the extra degeneracy $s$ and the Hochschild differential $b$. $\square$

### 5.3 The SBI Long Exact Sequence

The deepest structural result relating Hochschild and cyclic cohomology is Connes' long exact sequence, often called the *SBI sequence* (Periodicity–$B$–Inclusion).

**Theorem (Connes' SBI sequence).** There is a long exact sequence

$$\cdots \to HC^{n-1}(A) \xrightarrow{S} HC^{n+1}(A) \xrightarrow{B} HH^n(A, A^*) \xrightarrow{I} HC^n(A) \xrightarrow{S} HC^{n+2}(A) \to \cdots$$

where:
- $I : HH^\bullet(A, A^*) \to HC^\bullet(A)$ is the natural map induced by the inclusion of Hochschild cochains with the zero $B$-differential.
- $B : HC^{n+1}(A) \to HH^n(A, A^*)$ is induced by the Connes $B$-operator.
- $S : HC^{n-1}(A) \to HC^{n+1}(A)$ is the *periodicity operator* (cup product with a canonical class in $HC^2(k)$).

*Proof sketch.* Consider the short exact sequence of mixed complexes $0 \to C^\bullet_\lambda \to C^\bullet(A, A^*) \to \Sigma^{-2} C^\bullet_\lambda \to 0$ (where $\Sigma^{-2}$ is a degree shift), whose connecting homomorphism is $S$. The associated long exact sequence in cohomology is the SBI sequence. $\square$

The periodicity operator $S$ has degree $+2$: it relates cyclic cohomology groups two steps apart. Taking the direct limit along $S$ gives *periodic cyclic cohomology* $HP^\bullet(A) = \varinjlim_S HC^\bullet(A)$, which is $\mathbb{Z}/2\mathbb{Z}$-graded.

> [!TIP] The SBI sequence in practice
> The SBI sequence is the main computational tool for cyclic cohomology. Starting from a computation of $HH^\bullet$, one uses the sequence to determine $HC^\bullet$. The map $B$ serves as an analogue of the de Rham differential, connecting Hochschild cohomology (the "forms") to cyclic cohomology (the "closed forms modulo exact forms").

### 5.4 Cyclic Cohomology for Commutative Algebras

For a smooth commutative $k$-algebra $A$, the Hochschild–Kostant–Rosenberg theorem (§6) and the Connes $B$-operator together give a complete picture.

**Theorem (HKR + cyclic cohomology).** For $A$ smooth commutative over $k \supset \mathbb{Q}$:

$$HC^n(A) \cong \frac{\Omega^n_{A/k}}{d\Omega^{n-1}_{A/k}} \oplus H^{n-2}_{\mathrm{dR}}(A) \oplus H^{n-4}_{\mathrm{dR}}(A) \oplus \cdots$$

where $H^\bullet_{\mathrm{dR}}(A) = H^\bullet(\Omega^\bullet_{A/k}, d)$ is the algebraic de Rham cohomology. The Connes $B$-operator corresponds to the de Rham differential $d : \Omega^n \to \Omega^{n+1}$ under the HKR identification.

*This is the algebraic incarnation of the statement that cyclic cohomology of the algebra of functions on a manifold recovers the de Rham cohomology of the manifold.*

---

> [!QUESTION] Exercise 7: B Squares to Zero
> *This problem verifies the key identity $B^2 = 0$ using the relationship between the cyclic norm and the cyclic operator.*
>
> > **Prerequisites:** [[#5.2 The Connes B-Operator|5.2 The Connes B-Operator]]
>
> Using $B = (1 - (-1)^n \lambda) \circ s \circ N$ and the identities $N(1-\lambda) = 0$ and $(1-\lambda)N = 0$ (where $N = \sum_{i=0}^n \lambda^i$ and $\lambda$ is the cyclic operator), prove that $B^2 = 0$.

> [!TIP]- Solution to Exercise 7
> **Key insight:** The two key identities $N(1-\lambda) = 0$ and $(1-\lambda)N = 0$ encode that $1-\lambda$ and $N$ are "complementary" operators, and composing them in either order kills everything.
>
> **Sketch:** $B^2 = [(1-\epsilon\lambda) s N]^2$ where $\epsilon = (-1)^n$. Expanding: $B^2 = (1-\epsilon\lambda) s N (1-\epsilon\lambda) s N$. Now $N(1-\lambda) = 0$ means $N \cdot (1 - \lambda) = 0$; more precisely, $Ns = s N$ commutation and $N(1-\lambda) = 0$ gives $N(1-\epsilon\lambda) = N(1-\lambda) \cdot (\text{sign adjustment}) = 0$ modulo the sign $\epsilon$. A direct computation using $N = \sum_{i=0}^n \lambda^i$ and $\lambda^{n+1} = 1$ on cyclic cochains shows $NsN = 0$, which forces $B^2 = 0$.

> [!QUESTION] Exercise 8: The Cyclic Condition in Degree 1
> *This problem makes the cyclic symmetry condition concrete in degree 1 and connects it to traces.*
>
> > **Prerequisites:** [[#5.1 The Cyclic Operator and Cyclic Cochains|5.1 The Cyclic Operator and Cyclic Cochains]]
>
> Show that a 1-cochain $\phi : A \otimes A \to k$ lies in $C^1_\lambda(A)$ (i.e., $\lambda \phi = \phi$) if and only if $\phi(a, b) = -\phi(b, a)$. Compute $HC^0(A)$ directly from the definition and show it equals the space of trace functionals on $A$ (linear forms $\tau : A \to k$ with $\tau(ab) = \tau(ba)$).

> [!TIP]- Solution to Exercise 8
> **Key insight:** In degree 0, the cyclic condition is exactly the trace identity; in degree 1, it is antisymmetry.
>
> **Sketch:** For degree 1: $(\lambda \phi)(a_0, a_1) = (-1)^1 \phi(a_1, a_0) = -\phi(a_1, a_0)$. So $\lambda \phi = \phi$ iff $\phi(a_0,a_1) = -\phi(a_1,a_0)$, confirming antisymmetry. For degree 0: $C^0(A, A^*) = A^* = \mathrm{Hom}_k(A, k)$, and $(\lambda \tau)(a) = (-1)^0 \tau(a) = \tau(a)$, so $\lambda = \mathrm{id}$ on $C^0$ and every 0-cochain is cyclic. The cocycle condition $\delta \tau = 0$ reads $(\delta \tau)(a,b) = \tau(ab) - \tau(ba) = 0$, i.e., $\tau$ is a trace. There are no $(-1)$-cochains, so $HC^0(A) = \ker(\delta : C^0 \to C^1) = \{\text{traces on } A\}$.

---

## 6. The Hochschild-Kostant-Rosenberg Theorem 🔑

### 6.1 Statement

The HKR theorem is one of the most important structural results in algebra: it identifies Hochschild cohomology with polyvector fields and Hochschild homology with differential forms, for smooth commutative algebras. It is also the theorem that fails in the noncommutative or singular setting, forcing the use of the full machinery of deformation theory.

**Theorem (Hochschild–Kostant–Rosenberg, 1962).** Let $k$ be a field of characteristic zero and let $A$ be a smooth, finitely generated commutative $k$-algebra (equivalently, the coordinate ring $\mathcal{O}(X)$ of a smooth affine variety $X = \mathrm{Spec}(A)$). Then there are natural isomorphisms of graded $A$-modules:

$$HH^n(A, A) \cong \bigwedge^n_A \mathrm{Der}(A, A) \qquad \text{(polyvector fields)},$$

$$HH^n(A, A^*) \cong \Omega^n_{A/k} \qquad \text{(Kähler differential forms)}.$$

### 6.2 The HKR Map

The HKR isomorphism is given by an explicit quasi-isomorphism at chain level.

**Definition (HKR map).** Define $\mathrm{HKR} : \bigwedge^n_A \mathrm{Der}(A) \to C^n(A,A)$ by

$$\mathrm{HKR}(\partial_1 \wedge \cdots \wedge \partial_n)(a_1, \ldots, a_n) = \frac{1}{n!} \sum_{\sigma \in S_n} \mathrm{sgn}(\sigma) \, \partial_{\sigma(1)}(a_1) \cdots \partial_{\sigma(n)}(a_n),$$

where $\partial_1, \ldots, \partial_n \in \mathrm{Der}(A,A)$ are derivations and the product is in $A$.

**Proposition (HKR is a quasi-isomorphism).** The map $\mathrm{HKR}$ is a morphism of cochain complexes (where $\bigwedge^\bullet_A \mathrm{Der}(A)$ has zero differential) and induces the HKR isomorphism on cohomology.

*Proof sketch.* One verifies that the image of $\mathrm{HKR}$ lands in $\ker \delta$: this uses commutativity of $A$ and the Leibniz rule for derivations. The key identity is that $\delta(\mathrm{HKR}(\partial_1 \wedge \cdots \wedge \partial_n)) = 0$ for any tuple of derivations — a consequence of the fact that for commutative $A$, the "non-flat" terms in the Hochschild differential cancel by symmetry. To show $\mathrm{HKR}$ is a quasi-isomorphism: filter both sides by the polynomial degree in $A$, and use the Koszul resolution of $A$ as an $A \otimes_k A^{\mathrm{op}}$-module (which exists and is explicit when $A$ is smooth/flat over $k$). The associated graded computation reduces to an elementary polynomial calculation. $\square$

> [!WARNING] Commutativity is essential
> The HKR theorem is false for noncommutative algebras. For a matrix algebra $A = M_n(k)$, $HH^\bullet(A,A) = HH^0(A,A) = k$ (concentrated in degree 0), while $\bigwedge^\bullet \mathrm{Der}(A)$ is much larger. The theorem critically uses $ab = ba$ to cancel terms in the Hochschild differential.

### 6.3 Consequences

The HKR theorem has structural consequences at the dg level as well.

**Theorem (HKR as dg-algebra quasi-isomorphism, Kontsevich).** For $A = k[x_1, \ldots, x_d]$ (or any smooth affine variety over $k \supset \mathbb{Q}$), there is a quasi-isomorphism of dg-algebras

$$\left(\bigwedge^\bullet_A \mathrm{Der}(A), 0\right) \xrightarrow{\sim} (C^\bullet(A,A), \delta).$$

Moreover, there is a quasi-isomorphism of $L_\infty$-algebras (the *Kontsevich formality theorem*)

$$\left(\bigwedge^\bullet_A \mathrm{Der}(A), 0, \text{Schouten bracket}\right) \xrightarrow{\sim} (C^\bullet(A,A), \delta, [-,-]_G),$$

where the Schouten bracket on polyvector fields is the natural extension of the Lie bracket on $\mathrm{Der}(A)$.

**Corollary (Classification of formal deformations of smooth commutative algebras).** Every formal deformation of a smooth commutative $k$-algebra $A$ (over $k \supset \mathbb{Q}$) is classified, up to gauge equivalence, by a Maurer-Cartan element in $(\bigwedge^\bullet_A \mathrm{Der}(A))[[t]]$ — that is, by a formal *Poisson bivector field* $\pi \in \mathrm{Der}(A)^{\wedge 2}[[t]]$ satisfying $[\pi, \pi]_{\text{Schouten}} = 0$. **This is the algebraic content of Kontsevich's deformation quantization theorem.**

### 6.4 Failure for Singular Varieties

**Remark (Singular case).** When $A$ is not smooth, the HKR isomorphism fails. The correct replacement involves the *cotangent complex* $\mathbb{L}_{A/k}$ of André–Quillen: one has

$$HH^\bullet(A,A) \cong \mathrm{Ext}^\bullet_{A \otimes A^{\mathrm{op}}}(A, A),$$

and this no longer simplifies to polyvector fields unless $\mathbb{L}_{A/k} \simeq \Omega^1_{A/k}[0]$ (i.e., $A$ is smooth). *For singular $A$, deformation theory is controlled by the full derived structure of the cotangent complex, not by Kähler differentials alone.*

---

> [!QUESTION] Exercise 9: HKR in Degree 1
> *This problem makes the degree-1 case of HKR explicit, identifying derivations with Hochschild 1-cocycles for commutative A.*
>
> > **Prerequisites:** [[#6.2 The HKR Map|6.2 The HKR Map]], [[#2.2 HH1: Outer Derivations|2.2 HH1: Outer Derivations]]
>
> For commutative $A$, show that the HKR map in degree 1 is the identity: $\mathrm{HKR}(\partial) = \partial$ as a $k$-linear map $A \to A$. Verify that $\mathrm{HKR}(\partial)$ is a Hochschild 1-cocycle, and that for commutative $A$ all inner derivations $d_a(b) = ab - ba$ are zero. Conclude $HH^1(A,A) = \mathrm{Der}(A,A)$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** Commutativity kills all inner derivations, so $HH^1(A,A)$ is exactly the space of derivations with no quotient needed.
>
> **Sketch:** For $n=1$: $\mathrm{HKR}(\partial)(a) = \frac{1}{1!} \partial(a) = \partial(a)$. So $\mathrm{HKR}(\partial) = \partial$. The cocycle condition $\delta(\partial)(a_1, a_2) = a_1 \partial(a_2) - \partial(a_1 a_2) + \partial(a_1) a_2 = 0$ is exactly the Leibniz rule. For commutative $A$: $d_m(a) = am - ma = 0$ for all $a, m \in A$ since $am = ma$. So $\mathrm{InnDer}(A,A) = 0$ and $HH^1(A,A) = \mathrm{Der}(A,A)$, matching the HKR isomorphism $HH^1(A,A) \cong \mathrm{Der}(A)$.

> [!QUESTION] Exercise 10: HKR for a Polynomial Ring in Degree 2
> *This problem verifies the HKR isomorphism concretely for $A = k[x, y]$, identifying $HH^2$ with bivector fields.*
>
> > **Prerequisites:** [[#6.1 Statement|6.1 Statement]]
>
> For $A = k[x,y]$, the polyvector field $\partial_x \wedge \partial_y$ maps to a Hochschild 2-cochain via $\mathrm{HKR}(\partial_x \wedge \partial_y)(f, g) = \frac{1}{2}(\partial_x f \cdot \partial_y g - \partial_y f \cdot \partial_x g)$. Verify this is a 2-cocycle by computing $\delta(\mathrm{HKR}(\partial_x \wedge \partial_y))(f,g,h)$ and showing it vanishes using commutativity of $A$.

> [!TIP]- Solution to Exercise 10
> **Key insight:** The coboundary of $\mathrm{HKR}(\partial_x \wedge \partial_y)$ vanishes because all terms cancel by commutativity and the product rule for derivatives.
>
> **Sketch:** Let $\phi = \mathrm{HKR}(\partial_x \wedge \partial_y)$, so $\phi(f,g) = \frac{1}{2}(\partial_x f \cdot \partial_y g - \partial_y f \cdot \partial_x g)$. Then $(\delta \phi)(f,g,h) = f \cdot \phi(g,h) - \phi(fg,h) + \phi(f,gh) - \phi(f,g) \cdot h$. Expanding $\phi(fg, h)$ using the product rule: $\partial_x(fg) = f \partial_x g + g \partial_x f$ and similarly for $y$. Substituting and using commutativity ($f \cdot \partial_y g \cdot \partial_x h = \partial_x h \cdot f \cdot \partial_y g$, etc.) shows that all eight terms cancel in pairs, giving $\delta \phi = 0$.

---

## 7. Hochschild Cohomology and Deformation Theory 💡

This section connects to [[research/entropy-operad-derivation|Entropy as an Operad Derivation]] and to the deformation theory portion of the operads curriculum.

### 7.1 Formal Deformations and the Maurer-Cartan Equation

**Definition (Formal deformation).** A *formal deformation* of an associative $k$-algebra $A$ is a $k[[t]]$-algebra $(A[[t]], \star)$ where $A[[t]] = A \otimes_k k[[t]]$ as a $k[[t]]$-module, and the product is

$$a \star b = ab + \sum_{n=1}^\infty t^n \mu_n(a,b), \quad \mu_n \in C^2(A,A),$$

with $ab$ the original product in $A$. Write $\mu = \sum_{n \geq 1} t^n \mu_n \in C^2(A,A)[[t]]$.

Associativity of $\star$ is equivalent to the *Maurer-Cartan equation* in the differential graded Lie algebra $(C^\bullet(A,A)[[t]], \delta, [-,-]_G)$:

$$\delta\mu + \frac{1}{2}[\mu, \mu]_G = 0 \quad \in C^3(A,A)[[t]].$$

> [!NOTE] The dgla perspective
> The Hochschild cochain complex $(C^\bullet(A,A), \delta, [-,-]_G)$ is a *differential graded Lie algebra* (dgla) with the coboundary as differential and the Gerstenhaber bracket as Lie bracket (with appropriate degree shifts). Formal deformations are *Maurer-Cartan elements* of this dgla (tensored with $tk[[t]]$). This is the bridge between Hochschild cohomology and deformation theory — a relationship that applies to any algebraic structure governed by an operad (see [[concepts/category-theory/operads/koszul-duality|Koszul Duality and the Bar Construction]] §8.3).

### 7.2 Order-by-Order Expansion

Expanding the Maurer-Cartan equation order by order in $t$:

**Order $t^1$:** $\delta\mu_1 = 0$. Thus $\mu_1 \in Z^2(A,A)$ is a Hochschild 2-cocycle. Every infinitesimal deformation comes from a 2-cocycle.

**Order $t^2$:** $\delta\mu_2 + \frac{1}{2}[\mu_1, \mu_1]_G = 0$. Since $[\mu_1, \mu_1]_G \in C^3(A,A)$, a solution $\mu_2$ exists if and only if $\frac{1}{2}[\mu_1, \mu_1]_G$ is a coboundary in $C^3(A,A)$, i.e., if and only if $\frac{1}{2}[[\mu_1, \mu_1]_G] = 0$ in $HH^3(A,A)$.

**Definition (Obstruction class).** The class $\mathrm{obs}(\mu_1) := \frac{1}{2}[[\mu_1, \mu_1]_G] \in HH^3(A,A)$ is the *obstruction class* to extending the first-order deformation $\mu_1$ to second order.

**Order $t^n$:** $\delta\mu_n + \frac{1}{2}\sum_{i=1}^{n-1}[\mu_i, \mu_{n-i}]_G = 0$. The obstruction at order $n$ is the class $[\sum_{i=1}^{n-1}[\mu_i, \mu_{n-i}]_G] \in HH^3(A,A)$.

> [!EXAMPLE] Matrix algebras are rigid
> For $A = M_n(k)$ (the $n \times n$ matrix algebra), a classical result (going back to Hochschild) gives $HH^2(M_n(k), M_n(k)) = 0$. By the deformation-theoretic interpretation, **$M_n(k)$ has no nontrivial infinitesimal deformations** — the only deformation is the trivial one. More precisely, every formal deformation is isomorphic (as a $k[[t]]$-algebra) to $M_n(k[[t]])$ with undeformed product. This is called *rigidity*.

### 7.3 Rigidity and Formality

**Definition (Rigidity).** An algebra $A$ is *rigid* if $HH^2(A,A) = 0$. For rigid $A$, every formal deformation is trivial (isomorphic to the undeformed algebra $A[[t]]$).

**Definition (Formality).** A dg-algebra $(V, d)$ is *formal* if there exists a zigzag of dg-algebra quasi-isomorphisms connecting $(V, d)$ to $(H^\bullet(V), 0)$ (its cohomology with zero differential).

**Theorem (Kontsevich formality theorem, 1997).** For any smooth manifold $X$ (or smooth affine variety over $\mathbb{Q}$), the Hochschild cochain complex $C^\bullet(\mathcal{O}(X), \mathcal{O}(X))$ is formal as an $L_\infty$-algebra: there is an $L_\infty$-quasi-isomorphism

$$C^\bullet(\mathcal{O}(X), \mathcal{O}(X)) \xrightarrow{\sim} HH^\bullet(\mathcal{O}(X), \mathcal{O}(X)) \cong \bigwedge^\bullet_{\mathcal{O}(X)} \mathrm{Der}(\mathcal{O}(X))$$

(by HKR). **This implies that deformations of $\mathcal{O}(X)$ as an associative algebra are classified by Maurer-Cartan elements in the Schouten algebra of polyvector fields** — i.e., by Poisson structures on $X$. Every Poisson manifold admits a formal deformation quantization.

> [!INFO] The formality theorem and $\hbar$-deformations
> The formal parameter in the Maurer-Cartan equation is $\hbar$ (Planck's constant) in physics notation. A Poisson manifold $(X, \{-,-\})$ determines a bivector $\pi \in \mathcal{O}(X)^{\wedge 2}$ with $[\pi, \pi]_{\text{Schouten}} = 0$ (Jacobi identity for $\{-,-\}$). The formality theorem says this $\pi$, viewed as an element of $HH^2(\mathcal{O}(X), \mathcal{O}(X))$, lifts via $L_\infty$ to a Maurer-Cartan element $\mu(\hbar)$ in $C^2(\mathcal{O}(X), \mathcal{O}(X))[[\hbar]]$, giving a noncommutative product $\star = \mu(\hbar)$ — the *star product* of deformation quantization.

---

> [!QUESTION] Exercise 11: Associativity at Second Order
> *This problem derives the obstruction class by computing the associativity condition at order $t^2$ directly.*
>
> > **Prerequisites:** [[#7.2 Order-by-Order Expansion|7.2 Order-by-Order Expansion]]
>
> Let $a \star b = ab + t\mu_1(a,b) + t^2 \mu_2(a,b) \pmod{t^3}$. Compute $(a \star b) \star c - a \star (b \star c)$ modulo $t^3$ and identify the coefficient of $t^2$ as $(\delta \mu_2)(a,b,c) + \frac{1}{2}[\mu_1, \mu_1]_G(a,b,c)$ (up to a sign). Hence derive the obstruction equation.

> [!TIP]- Solution to Exercise 11
> **Key insight:** The coefficient of $t^2$ in the associator combines first-order terms quadratically via the Gerstenhaber bracket.
>
> **Sketch:** At order $t^2$: $(a \star b) \star c - a \star (b \star c)$ at $t^2$ contributes from (i) $\mu_1(ab, c) - a \mu_1(b,c) - \mu_1(a,bc) + \mu_1(a,b) c$ at cross-order $t^1 \times t^1$, which is $(\delta \mu_1)$ applied to $(a,b,c)$ (zero since $\mu_1$ is a cocycle, but the cross-terms give $\mu_1(\mu_1(a,b), c) - \mu_1(a, \mu_1(b,c))$), plus (ii) the pure $t^2$ associativity of $\mu_2$ giving $-(\delta \mu_2)(a,b,c)$. The cross-terms are exactly $\frac{1}{2}[\mu_1,\mu_1]_G(a,b,c)$ by the formula from the Example callout in §3.2. Setting the total to zero gives $\delta \mu_2 + \frac{1}{2}[\mu_1,\mu_1]_G = 0$, as claimed.

> [!QUESTION] Exercise 12: Rigidity of the Polynomial Ring
> *This problem shows that $k[x]$ is not rigid, contrasting with the matrix algebra example.*
>
> > **Prerequisites:** [[#7.3 Rigidity and Formality|7.3 Rigidity and Formality]], [[#6.1 Statement|6.1 Statement]]
>
> Using HKR, identify $HH^2(k[x], k[x])$ with the space of bivector fields on $\mathbb{A}^1$, i.e., $\bigwedge^2 \mathrm{Der}(k[x]) = 0$ (since $\mathrm{Der}(k[x]) = k[x] \partial_x$ is rank 1). Conclude $HH^2(k[x], k[x]) = 0$ and thus $k[x]$ is rigid. Now for $k[x,y]$: $HH^2(k[x,y], k[x,y]) \cong k[x,y] \partial_x \wedge \partial_y$ is nonzero. Find the explicit 2-cocycle $\mu_1 = \mathrm{HKR}(\partial_x \wedge \partial_y)$ and verify it satisfies $[\mu_1, \mu_1]_G = 0$ (so the deformation extends to all orders — it is the Moyal product).

> [!TIP]- Solution to Exercise 12
> **Key insight:** Bivector fields in dimension 1 vanish for degree reasons; in dimension 2 the unique bivector $\partial_x \wedge \partial_y$ has $[\pi, \pi]_{\text{Schouten}} = 0$ because the Schouten bracket of a constant bivector with itself vanishes.
>
> **Sketch:** $\bigwedge^2 \mathrm{Der}(k[x]) = \bigwedge^2(k[x]\partial_x) = 0$ since the exterior square of a rank-1 module is zero. So $HH^2(k[x],k[x]) = 0$ and $k[x]$ is rigid. For $k[x,y]$: $\mu_1(f,g) = \partial_x f \cdot \partial_y g - \partial_y f \cdot \partial_x g$ (up to $\frac{1}{2}$). Then $[\mu_1, \mu_1]_G \in C^3(k[x,y], k[x,y])$. Since $\partial_x \wedge \partial_y$ is a constant bivector field (no $x,y$ coefficients), its Schouten self-bracket vanishes: $[\partial_x \wedge \partial_y, \partial_x \wedge \partial_y]_{\text{Schouten}} = 0$. The corresponding Maurer-Cartan equation holds to all orders with $\mu_n = 0$ for $n \geq 2$: the deformation $f \star g = fg + t(\partial_x f \partial_y g - \partial_y f \partial_x g)$ is already associative (it is the Moyal product to first order).

---

## 8. Operadic Cohomology as the General Framework 🔑

This section closes the loop by situating everything in the operadic framework from [[concepts/category-theory/operads/koszul-duality|Koszul Duality and the Bar Construction]] §8.

### 8.1 The Two-Sided Bar Resolution for Ass

Recall from [[concepts/category-theory/operads/algebras-modules|Algebras and Modules]] that the *enveloping algebra* of an $\mathrm{Ass}$-algebra $A$ is $U_{\mathrm{Ass}}(A) = A \otimes_k A^{\mathrm{op}}$, and an $A$-bimodule is a left $U_{\mathrm{Ass}}(A)$-module. The operadic cohomology is computed by a projective resolution of $A$ as a $U_{\mathrm{Ass}}(A)$-module.

**Definition (Bar resolution).** The *two-sided bar resolution* (or *standard bar resolution*) of $A$ over $A \otimes A^{\mathrm{op}}$ is the chain complex

$$B(A) : \cdots \to A \otimes \bar{A}^{\otimes 2} \otimes A \xrightarrow{d} A \otimes \bar{A} \otimes A \xrightarrow{d} A \otimes A \xrightarrow{m} A \to 0,$$

where $\bar{A} = A/k\cdot 1$ (the *augmentation ideal*) and the differential is

$$d(a_0 \otimes [a_1|\cdots|a_n] \otimes a_{n+1}) = \sum_{i=0}^n (-1)^i a_0 \otimes [a_1|\cdots|a_i a_{i+1}|\cdots|a_n] \otimes a_{n+1}$$

(with $a_0 a_1$ replacing $a_0 \otimes [a_1|\cdots]$ at $i=0$ and $a_n a_{n+1}$ at $i=n$). This is a free resolution of $A$ as an $A \otimes A^{\mathrm{op}}$-module.

**Theorem.** Applying $\mathrm{Hom}_{A \otimes A^{\mathrm{op}}}(-, M)$ to the bar resolution $B(A)$ yields the Hochschild cochain complex $C^\bullet(A,M)$. Hence

$$HH^n(A,M) = \mathrm{Ext}^n_{A \otimes A^{\mathrm{op}}}(A, M) = H^n_{\mathrm{Ass}}(A, M).$$

This is the identification of Hochschild cohomology as operadic cohomology for $\mathcal{O} = \mathrm{Ass}$. The two-sided bar complex $B(\mathrm{Ass}^!, A, M)$ of the general operadic framework (with $\mathrm{Ass}^! = \mathrm{Ass}$ as a cooperad, reflecting the Koszul self-duality of $\mathrm{Ass}$) reduces to this bar resolution.

> [!NOTE] Normalized vs. unnormalized bar complex
> The complex $B(A)$ using all of $A$ (not $\bar{A}$) is the unnormalized bar resolution. The normalized version (using $\bar{A} = A/k$) is quasi-isomorphic and more compact. Both compute the same $HH^\bullet$. In the operadic language, the normalization corresponds to working with the *reduced* bar construction $\bar{B}(\mathcal{O})$.

### 8.2 Analogues for Com and Lie

The operadic framework unifies Hochschild cohomology with other classical cohomology theories:

| Operad $\mathcal{O}$ | Algebra type | Cohomology theory | Bimodule resolution |
|---|---|---|---|
| $\mathrm{Ass}$ | Associative algebra $A$ | Hochschild: $HH^\bullet(A,M)$ | $A \otimes A^{\mathrm{op}}$-bar resolution |
| $\mathrm{Com}$ | Commutative algebra $A$ | Harrison: $\mathrm{Harr}^\bullet(A,M)$ | $A$-module cotangent complex |
| $\mathrm{Lie}$ | Lie algebra $\mathfrak{g}$ | Chevalley–Eilenberg: $H^\bullet_{\mathrm{CE}}(\mathfrak{g},M)$ | Chevalley–Eilenberg complex |

The key point: **each cohomology theory is operadic cohomology for the relevant operad**. The Koszul dual pairs govern the form of the bar complex:
- $\mathrm{Ass}^! = \mathrm{Ass}$: the bar complex of $\mathrm{Ass}$ is $\mathrm{Ass}$-shaped (sequences of associative multiplications).
- $\mathrm{Com}^! = \mathrm{Lie}$: Harrison cohomology uses the Lie-type bar complex (antisymmetric tensors); this is why $\mathrm{Harr}^\bullet(A,M)$ is a summand of $HH^\bullet(A,M)$ for commutative $A$.
- $\mathrm{Lie}^! = \mathrm{Com}$: the CE complex uses the exterior algebra (com-type structure).

> [!INFO] Harrison vs. Hochschild for commutative algebras
> For a commutative $k$-algebra $A$, there is a splitting (over $\mathbb{Q}$) of the Hochschild complex into *Adams eigenspaces*: $C^\bullet(A,M) = \bigoplus_{\lambda} C^\bullet_\lambda$, and $\mathrm{Harr}^\bullet(A,M) = C^\bullet_1$ is the eigenspace for the first Adams operation. The HKR theorem is the identification $H^\bullet(C^\bullet_1) = \bigwedge^\bullet \mathrm{Der}(A)$.

### 8.3 Closing the Loop: the Deformation Complex of Ass

The deformation theory of the operad $\mathrm{Ass}$ itself (as an operad, not just as an algebra) produces a complex whose cohomology encodes the space of all deformations of associative algebra structures. This is the *Gerstenhaber–Schack complex* or the *operadic deformation complex* $\mathrm{Def}(\mathrm{Ass})$.

**Theorem.** The deformation complex of $\mathrm{Ass}$ (the dg-Lie algebra governing deformations of $\mathrm{Ass}$ as an operad) is quasi-isomorphic to the full Hochschild cochain complex $C^\bullet(A,A)$ for any $\mathrm{Ass}$-algebra $A$. The Gerstenhaber bracket on $C^\bullet(A,A)$ is the Lie bracket of the deformation complex. The Maurer-Cartan elements of $\mathrm{Def}(\mathrm{Ass})$ are exactly the formal deformations of $\mathrm{Ass}$-algebra structures.

*This unification* — Hochschild cohomology as deformation theory of the operad $\mathrm{Ass}$ — is the deepest structural statement, and it specializes to all the low-degree interpretations (§2) simultaneously: $HH^1$ = infinitesimal automorphisms of $\mathrm{Ass}$-structure, $HH^2$ = infinitesimal deformations, $HH^3$ = obstructions.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=large, row sep=large]
\mathrm{Ass} \arrow[r, "\text{Koszul dual}"] \arrow[d, "\text{bar resolution}"'] & \mathrm{Ass}^! = \mathrm{Ass} \arrow[d, "\text{two-sided bar}"] \\
B(\mathrm{Ass}) \arrow[r, "\text{quasi-iso}"'] \arrow[d, "\mathrm{Hom}_{A \otimes A^{\mathrm{op}}}(-{,}M)"'] & B(\mathrm{Ass}^!, A, M) \arrow[d, "\text{cohomology}"] \\
C^\bullet(A,M) \arrow[r, "\text{iso}"'] & HH^\bullet(A,M)
\end{tikzcd}
\end{document}
```

---

> [!QUESTION] Exercise 13: The Bar Resolution in Low Degrees
> *This problem makes the bar resolution explicit in degrees 0, 1, 2 and derives the Hochschild coboundary formula by applying Hom.*
>
> > **Prerequisites:** [[#8.1 The Two-Sided Bar Resolution for Ass|8.1 The Two-Sided Bar Resolution for Ass]]
>
> Write out the bar resolution $\cdots \to A \otimes A \otimes A \xrightarrow{d_1} A \otimes A \xrightarrow{m} A \to 0$ explicitly. Apply $\mathrm{Hom}_{A \otimes A^{\mathrm{op}}}(-, M)$ to obtain a cochain complex in degrees 0, 1, 2, and verify that the resulting coboundary maps match the Hochschild coboundary $\delta$ of §1.

> [!TIP]- Solution to Exercise 13
> **Key insight:** A left $A \otimes A^{\mathrm{op}}$-module map from $A \otimes A^{\otimes n} \otimes A$ to $M$ is determined by where the generator $1 \otimes a_1 \otimes \cdots \otimes a_n \otimes 1$ goes — so such maps are in bijection with $\mathrm{Hom}_k(A^{\otimes n}, M)$.
>
> **Sketch:** $\mathrm{Hom}_{A \otimes A^{\mathrm{op}}}(A \otimes A^{\otimes n} \otimes A, M) \cong \mathrm{Hom}_k(A^{\otimes n}, M) = C^n(A,M)$. The map $d^* : \mathrm{Hom}(A \otimes A \otimes A, M) \to \mathrm{Hom}(A \otimes A \otimes A \otimes A, M)$ is precomposition with $d_1(a_0 \otimes a_1 \otimes a_2) = a_0 a_1 \otimes a_2 - a_0 \otimes a_1 a_2$. Under the identification, a map $f \in C^1(A,M)$ pulls back to $(d^* f)(a_1, a_2) = a_1 \cdot f(a_2) - f(a_1 a_2) + f(a_1) \cdot a_2 = (\delta f)(a_1, a_2)$, matching the Hochschild coboundary.

> [!QUESTION] Exercise 14: Chevalley-Eilenberg as Operadic Cohomology
> *This problem establishes the analogy between Hochschild and Chevalley-Eilenberg cohomology within the operadic framework.*
>
> > **Prerequisites:** [[#8.2 Analogues for Com and Lie|8.2 Analogues for Com and Lie]]
>
> Let $\mathfrak{g}$ be a Lie algebra and $M$ a $\mathfrak{g}$-module. The Chevalley-Eilenberg cochain complex is $C^n_{\mathrm{CE}}(\mathfrak{g}, M) = \mathrm{Hom}_k(\bigwedge^n \mathfrak{g}, M)$ with coboundary $d_{\mathrm{CE}}$. Explain why this matches the general operadic cohomology formula $H^\bullet_{\mathrm{Lie}}(\mathfrak{g}, M) = H^\bullet(B(\mathrm{Lie}^!, \mathfrak{g}, M))$, noting that $\mathrm{Lie}^! = \mathrm{Com}$ and that $B(\mathrm{Com}, \mathfrak{g}, -)$ uses antisymmetric tensors (the exterior algebra).

> [!TIP]- Solution to Exercise 14
> **Key insight:** Since $\mathrm{Lie}^! = \mathrm{Com}$ (Koszul duality), the two-sided bar complex for $\mathcal{O} = \mathrm{Lie}$ uses the cooperad $\mathrm{Com}$, which picks out the commutative (i.e., symmetric) tensors. For a Lie algebra, antisymmetry of the bracket means the bar complex actually uses antisymmetric tensors, giving $\bigwedge^n \mathfrak{g}$.
>
> **Sketch:** $B(\mathrm{Lie}^!, \mathfrak{g}, M) = B(\mathrm{Com}, \mathfrak{g}, M)$. As a graded $k$-module, $B(\mathrm{Com}, \mathfrak{g}, M)_n = M \otimes \mathrm{Com}(\mathfrak{g})_n = M \otimes S^n(\mathfrak{g}[-1])$. With the degree shift, $S^n(\mathfrak{g}[-1]) \cong \bigwedge^n \mathfrak{g}$ (symmetric powers of the suspension are exterior powers). The differential encodes the Lie bracket, recovering the CE differential. Hence $H^\bullet_{\mathrm{Lie}}(\mathfrak{g}, M) = H^\bullet(C^\bullet_{\mathrm{CE}}(\mathfrak{g}, M))$.

---

## Algorithmic Applications

> [!QUESTION] Exercise 15: Computing HH of a Truncated Polynomial Ring
> *This problem provides a concrete algorithmic computation of Hochschild cohomology for a finite-dimensional algebra.*
>
> > **Prerequisites:** [[#1.2 The Coboundary Operator|1.2 The Coboundary Operator]], [[#2. Low-Degree Interpretations|2. Low-Degree Interpretations]]
>
> Let $A = k[x]/(x^2)$ (the dual numbers). Compute $HH^n(A, A)$ for $n = 0, 1, 2$ directly from the cochain complex. Write Python pseudocode (treating $A$ as a 2-dimensional $k$-vector space with basis $\{1, x\}$ and $x^2 = 0$) to build the matrices of $\delta^0$ and $\delta^1$ and compute their kernels and images.

> [!TIP]- Solution to Exercise 15
> **Key insight:** For the dual numbers, all products involving $x$ twice vanish, making the coboundary maps concrete low-dimensional linear maps computable by linear algebra.
>
> **Sketch:** $A$ has basis $\{1, x\}$ with $x^2 = 0$. $C^n(A,A)$ has dimension $2^{n+1}$. Python pseudocode:
> ```python
> import numpy as np
>
> # Basis elements: e0 = 1, e1 = x in A
> # C^0(A,A) = A, spanned by m0=(1 component), m1=(x component)
> # delta^0(m)(a) = a*m - m*a
> # For commutative A (dual numbers is commutative!): delta^0 = 0
> # So HH^0 = A = k^2
>
> # C^1(A,A) = Hom(A, A), spanned by f_{ij} where f_{ij}(e_i) = e_j
> # delta^1(f)(a,b) = a*f(b) - f(ab) + f(a)*b
> # Build delta^1 as 4x4 matrix (dim C^2 = 4, dim C^1 = 4)
>
> def mult(a, b):
>     # a = (a0, a1) means a0*1 + a1*x, x^2=0
>     return (a[0]*b[0], a[0]*b[1] + a[1]*b[0])
>
> basis = [(1,0), (0,1)]
> delta1 = np.zeros((4, 4))
> for col, f_basis in enumerate([(i,j) for i in range(2) for j in range(2)]):
>     def f(a, fi=f_basis):
>         return (fi[1],) if a == basis[fi[0]] else (0, 0)
>     # ... fill delta1 matrix
>
> # HH^0 = ker(delta^0) = A (all of A, since A commutative)
> # HH^1 = ker(delta^1)/im(delta^0) = Der(A)/0 = Der(A) = k*(x d/dx)
> # HH^2 = ker(delta^2)/im(delta^1) -- periodic for dual numbers
> ```
> Result: $HH^0(A,A) = A$, $HH^1(A,A) \cong k$ (generated by $\partial: x \mapsto 1$), $HH^n(A,A) \cong k$ for all $n \geq 0$ (the cohomology is $k$-periodic by the theory of Tate cohomology for finite group schemes).

> [!QUESTION] Exercise 16: Algorithmic Gerstenhaber Bracket
> *This problem implements the pre-Lie product and Gerstenhaber bracket for low-degree cochains.*
>
> > **Prerequisites:** [[#3.2 The Pre-Lie Product and Gerstenhaber Bracket|3.2 The Pre-Lie Product and Gerstenhaber Bracket]]
>
> Write Python pseudocode implementing the pre-Lie product $f \circ g$ for $f, g \in C^2(A,A)$ (both degree 2), where $A$ is a finite-dimensional algebra given by structure constants $\{c_{ij}^k\}$ (so $e_i \cdot e_j = \sum_k c_{ij}^k e_k$). Use this to compute $[\mu_1, \mu_1]_G$ for the first-order deformation cocycle $\mu_1$, and verify it lands in $C^3(A,A)$.

> [!TIP]- Solution to Exercise 16
> **Key insight:** Structure constants make all multilinear maps into tensors, reducing the bracket computation to finite tensor contractions.
>
> **Sketch:**
> ```python
> import numpy as np
>
> def pre_lie_product(f, g, structure_constants):
>     """
>     f, g: arrays of shape (dim, dim, dim) representing 2-cochains
>           f[i,j,k] = coefficient of e_k in f(e_i, e_j)
>     structure_constants: c[i,j,k] = coeff of e_k in e_i * e_j
>     Returns: f circ g as array of shape (dim, dim, dim) (a 3-cochain)
>     """
>     dim = f.shape[0]
>     result = np.zeros((dim, dim, dim, dim))  # 3-cochain: 3 inputs, 1 output
>     c = structure_constants
>     # (f circ g)(a1, a2, a3) = f(g(a1,a2), a3) - f(a1, g(a2,a3))
>     # sign: (-1)^{(q-1)*i} with q=2, i=0: +1; i=1: -1
>     for a1 in range(dim):
>         for a2 in range(dim):
>             for a3 in range(dim):
>                 for out in range(dim):
>                     # Term i=0: f(g(a1,a2), a3)
>                     for mid in range(dim):
>                         result[a1,a2,a3,out] += g[a1,a2,mid] * f[mid,a3,out]
>                     # Term i=1: -f(a1, g(a2,a3))
>                     for mid in range(dim):
>                         result[a1,a2,a3,out] -= g[a2,a3,mid] * f[a1,mid,out]
>     return result
>
> def gerstenhaber_bracket(f, g, c):
>     """Both f, g are 2-cochains. Returns [f,g]_G = f circ g - g circ f."""
>     return pre_lie_product(f, g, c) - pre_lie_product(g, f, c)
>     # sign (-1)^{(2-1)(2-1)} = 1, so no extra sign
> ```

> [!QUESTION] Exercise 17: The B-Operator as a Differential
> *This problem implements Connes' B-operator and verifies $B^2 = 0$ computationally for a small algebra.*
>
> > **Prerequisites:** [[#5.2 The Connes B-Operator|5.2 The Connes B-Operator]]
>
> For $A = k \oplus k$ (product algebra, 2-dimensional) and $f \in C^2(A, A^*)$ a 2-cochain (equivalently, a trilinear form $f : A^{\otimes 3} \to k$), write Python pseudocode to compute $(Bf)(a_0, a_1)$ using the formula in §5.2 and verify that $B(Bf) = 0$ by computing $B^2 f$ for a randomly chosen $f$.

> [!TIP]- Solution to Exercise 17
> **Key insight:** For a 2-dimensional algebra, all tensors are small enough to verify $B^2 = 0$ by direct enumeration.
>
> **Sketch:**
> ```python
> import numpy as np
>
> dim = 2
> unit = np.array([1, 0])  # e0 = 1 (unit element)
>
> def B_operator(f, unit, dim):
>     """
>     f: array of shape (dim,)*n representing an n-cochain
>        (i.e., (n+1)-linear form A^{n+1} -> k in the cyclic convention)
>     Returns: Bf, an (n-1)-cochain
>     """
>     n = len(f.shape) - 1  # f is an n-cochain
>     result = np.zeros((dim,) * n)
>     # (Bf)(a0,...,a_{n-1}) = sum_{i=0}^{n-1} (-1)^{ni} f(1, a_i,...,a_{n-1},a_0,...,a_{i-1})
>     for idx in np.ndindex((dim,) * (n - 1)):
>         val = 0
>         for i in range(n):
>             # cyclic rotation: (1, a_i, ..., a_{n-1}, a_0, ..., a_{i-1})
>             rotated = (slice(None),) + idx[i:] + idx[:i]
>             sign = (-1) ** (n * i)
>             # insert unit at position 0: contract with unit vector
>             for u in range(dim):
>                 val += sign * unit[u] * f[(u,) + idx[i:] + idx[:i]]
>         result[idx] = val
>     return result
>
> # Test B^2 = 0
> f = np.random.randn(dim, dim, dim)  # random 2-cochain (trilinear form)
> Bf = B_operator(f, unit, dim)       # 1-cochain
> BBf = B_operator(Bf, unit, dim)     # 0-cochain
> assert np.allclose(BBf, 0), f"B^2 != 0: {BBf}"
> print("B^2 = 0 verified.")
> ```

> [!QUESTION] Exercise 18: Solving the Maurer-Cartan Equation Iteratively
> *This problem implements the order-by-order extension of a deformation, checking vanishing of the obstruction class at each order.*
>
> > **Prerequisites:** [[#7.2 Order-by-Order Expansion|7.2 Order-by-Order Expansion]]
>
> Given an associative algebra $A$ with structure constants $c_{ij}^k$ and a Hochschild 2-cocycle $\mu_1$ (verified by checking $\delta \mu_1 = 0$), write Python pseudocode that: (1) computes the obstruction class $\mathrm{obs}_2 = \frac{1}{2}[\mu_1, \mu_1]_G \in C^3(A,A)$; (2) checks whether $\mathrm{obs}_2$ is a coboundary by solving the linear system $\delta \mu_2 = -\frac{1}{2}[\mu_1,\mu_1]_G$; (3) if solvable, outputs $\mu_2$ and continues to order 3.

> [!TIP]- Solution to Exercise 18
> **Key insight:** At each order, the obstruction is a 3-cocycle (by the Jacobi identity for the Gerstenhaber bracket), and extending the deformation reduces to a linear system whose solvability is equivalent to the obstruction class vanishing in $HH^3$.
>
> **Sketch:**
> ```python
> import numpy as np
> from scipy.linalg import lstsq
>
> def extend_deformation(mu_1, delta_matrix, gerstenhaber_bracket, max_order=5):
>     """
>     mu_1: array (dim, dim, dim) -- the first-order deformation 2-cocycle
>     delta_matrix: matrix of the Hochschild coboundary delta: C^2 -> C^3
>     Returns: list of (mu_n, obstruction_n) up to max_order
>     """
>     deformation = [None, mu_1]  # mu_0 = 0, mu_1 given
>     for n in range(2, max_order + 1):
>         # Compute obstruction: sum_{i=1}^{n-1} [mu_i, mu_{n-i}]
>         obs = sum(
>             gerstenhaber_bracket(deformation[i], deformation[n - i])
>             for i in range(1, n)
>             if deformation[i] is not None and deformation[n - i] is not None
>         ) / 2
>         obs_flat = obs.flatten()
>         # Solve delta * mu_n = -obs
>         mu_n_flat, residual, rank, sv = lstsq(delta_matrix, -obs_flat)
>         if residual > 1e-10:
>             print(f"Obstruction at order {n} is non-zero: deformation blocked.")
>             return deformation, obs
>         deformation.append(mu_n_flat.reshape(mu_1.shape))
>         print(f"Order {n}: mu_{n} found, obstruction class = 0.")
>     return deformation, None
> ```

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Gerstenhaber, "The Cohomology Structure of an Associative Ring" | Foundational 1963 paper introducing the Gerstenhaber bracket and proving $HH^\bullet(A,A)$ is a Gerstenhaber algebra | [Ann. of Math. 78 (1963), 267–288](https://www.jstor.org/stable/1970343) |
| Weibel, *An Introduction to Homological Algebra*, Ch. 9 | Standard graduate textbook treatment of Hochschild and cyclic cohomology, bar resolutions, and the operadic perspective | [Cambridge University Press](https://www.cambridge.org/core/books/an-introduction-to-homological-algebra/) |
| Witherspoon, *Hochschild Cohomology for Algebras* | AMS Graduate Studies monograph with complete proofs and modern perspective | [AMS GSM 204](https://www.ams.org/books/gsm/204/) |
| Loday, *Cyclic Homology*, Ch. 1–2 | Comprehensive treatment of cyclic homology and cohomology, the B-operator, and the SBI sequence | [Springer](https://link.springer.com/book/10.1007/978-3-662-21739-9) |
| Loday–Vallette, *Algebraic Operads*, Ch. 12 | Complete operadic perspective on Hochschild cohomology; bar/cobar construction for Ass; deformation theory | [Springer](https://link.springer.com/book/10.1007/978-3-642-30362-3) |
| Ginzburg, "Lectures on Noncommutative Geometry" | Lecture notes covering Hochschild cohomology, Gerstenhaber algebras, HKR, and deformations | [arXiv:math/0506603](https://arxiv.org/abs/math/0506603) |
| Kontsevich, "Deformation Quantization of Poisson Manifolds" | Proves the formality theorem for $C^\bullet(\mathcal{O}(X), \mathcal{O}(X))$; deformation quantization of Poisson manifolds | [arXiv:q-alg/9709040](https://arxiv.org/abs/q-alg/9709040) |
| Kontsevich–Soibelman, "Deformations of Algebras over Operads" | Extension of deformation quantization and formality to algebras over general operads; proof of Deligne conjecture | [arXiv:math/0001151](https://arxiv.org/abs/math/0001151) |
| McClure–Smith, "A Solution of Deligne's Hochschild Cohomology Conjecture" | Direct combinatorial proof of the Deligne conjecture via the surjection operad | [Contemp. Math. 293 (2002)](https://arxiv.org/abs/math/9910126) |
| Tamarkin, "Formality of Chain Operad of Little Discs" | Proves formality of $E_2$ operad over $\mathbb{Q}$; key step in operadic proof of Deligne conjecture | [Lett. Math. Phys. 66 (2003)](https://sites.math.northwestern.edu/~tamarkin/Papers/Formality.pdf) |
| Hochschild–Kostant–Rosenberg, "Differential Forms on Regular Affine Algebras" | Original 1962 paper proving $HH^\bullet(A,A) \cong \bigwedge^\bullet \mathrm{Der}(A)$ for smooth commutative $A$ | [Trans. AMS 102 (1962), 383–408](https://www.jstor.org/stable/1993614) |
| Keller, "Hochschild Cohomology and Derived Categories" | Survey connecting Hochschild cohomology to derived categories and $A_\infty$-categories | [IMRN survey](https://webusers.imj-prg.fr/~bernhard.keller/publ/HochschildCohomologyAndDerivedCategories.pdf) |
