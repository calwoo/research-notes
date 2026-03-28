# 📐 Category Theory IV: Adjoint Functor Theorems and Monads

## Table of Contents

- [[#1. Motivation: When Does a Functor Have an Adjoint|1. Motivation: When Does a Functor Have an Adjoint]]
- [[#2. Adjoints and Infima in Posets|2. Adjoints and Infima in Posets]]
- [[#3. Monads: Definition and First Examples|3. Monads: Definition and First Examples]]
  - [[#3.1 The Definition|3.1 The Definition]]
  - [[#3.2 Every Adjunction Gives a Monad|3.2 Every Adjunction Gives a Monad]]
  - [[#3.3 Every Monad Arises from an Adjunction|3.3 Every Monad Arises from an Adjunction]]
- [[#4. Eilenberg-Moore Algebras|4. Eilenberg-Moore Algebras]]
  - [[#4.1 T-Algebras and Their Morphisms|4.1 T-Algebras and Their Morphisms]]
  - [[#4.2 The Eilenberg-Moore Category|4.2 The Eilenberg-Moore Category]]
  - [[#4.3 The Comparison Functor|4.3 The Comparison Functor]]
- [[#5. Absolute Coequalizers and Split Pairs|5. Absolute Coequalizers and Split Pairs]]
  - [[#5.1 Reflexive Pairs|5.1 Reflexive Pairs]]
  - [[#5.2 Split Coequalizers|5.2 Split Coequalizers]]
- [[#6. Set-Adjoint, Representable, and Limit-Preserving Functors|6. Set-Adjoint, Representable, and Limit-Preserving Functors]]
- [[#7. The General Adjoint Functor Theorem|7. The General Adjoint Functor Theorem]]
  - [[#7.1 The Solution Set Condition|7.1 The Solution Set Condition]]
  - [[#7.2 Proof Sketch|7.2 Proof Sketch]]
- [[#8. Well-Powered Categories and the Subobject Functor|8. Well-Powered Categories and the Subobject Functor]]
- [[#9. The Special Adjoint Functor Theorem|9. The Special Adjoint Functor Theorem]]
  - [[#9.1 Coseparating Sets|9.1 Coseparating Sets]]
  - [[#9.2 Proof Sketch via GAFT|9.2 Proof Sketch via GAFT]]
- [[#10. Applications of GAFT and SAFT|10. Applications of GAFT and SAFT]]
- [[#11. The Eilenberg-Moore Category: Creation Theorems|11. The Eilenberg-Moore Category: Creation Theorems]]
  - [[#11.1 Creation of Limits|11.1 Creation of Limits]]
  - [[#11.2 Creation of Coequalizers for Absolute Pairs|11.2 Creation of Coequalizers for Absolute Pairs]]
- [[#12. Idempotent Monads|12. Idempotent Monads]]
- [[#13. Presheaves Are Monadic|13. Presheaves Are Monadic]]
- [[#14. The Beck Monadicity Theorem|14. The Beck Monadicity Theorem]]
  - [[#14.1 The Huge Monadicity Theorem|14.1 The Huge Monadicity Theorem]]
  - [[#14.2 Proof Strategy|14.2 Proof Strategy]]
- [[#15. The Comparison Functor Revisited|15. The Comparison Functor Revisited]]
- [[#16. Geometric Realization via SAFT|16. Geometric Realization via SAFT]]
- [[#17. Comonads from Adjunctions|17. Comonads from Adjunctions]]
- [[#18. Monadic Categories over Set and Top|18. Monadic Categories over Set and Top]]
- [[#References|References]]

---

> [!INFO] Series context
> This is File 4 of 4 in a self-study series based on Tom Leinster's Part III Cambridge course (Michaelmas 2000). This file presupposes [[concepts/category-theory/01-categories-functors-natural-transformations|File 1]] (categories and functors), [[concepts/category-theory/02-adjoints-representables|File 2]] (adjoints and representables), and [[concepts/category-theory/03-limits-colimits|File 3]] (limits and colimits). All 17 exercises are taken verbatim in spirit from Sheet 4 of the course.

---

## 1. Motivation: When Does a Functor Have an Adjoint 🔍

A central result proved in [[concepts/category-theory/03-limits-colimits|File 3]] is that right adjoints preserve limits. More precisely, if $F \dashv G$ with $G \colon \mathcal{D} \to \mathcal{C}$, then $G$ preserves all limits that exist in $\mathcal{D}$.

This raises the converse question: **when does limit-preservation suffice to guarantee the existence of a left adjoint?** The answer is the content of the *Adjoint Functor Theorems*, the main subject of this file.

> [!INFO] Motivating example: reflective subcategories
> A full subcategory $\mathcal{D} \hookrightarrow \mathcal{C}$ is *reflective* if the inclusion functor $i \colon \mathcal{D} \to \mathcal{C}$ has a left adjoint $L \colon \mathcal{C} \to \mathcal{D}$ (the *reflector* or *localization*). The adjoint functor theorems tell us when a limit-preserving inclusion has a reflector, bypassing the need to construct $L$ by hand.
>
> Classical examples: abelian groups inside all groups, sheaves inside presheaves, compact Hausdorff spaces inside all topological spaces.

The plan is:
1. Understand the poset case, where the theorem is transparent.
2. State and prove the *General Adjoint Functor Theorem* (GAFT) using the *solution set condition*.
3. Derive the *Special Adjoint Functor Theorem* (SAFT) using well-poweredness.
4. Develop the theory of *monads* and prove *Beck's Monadicity Theorem*.

---

## 2. Adjoints and Infima in Posets 📊

A *poset* $(P, \leq)$ is a category whose objects are elements of $P$ and whose hom-sets satisfy $|P(p, q)| \leq 1$, with a morphism $p \to q$ iff $p \leq q$. A functor between posets is exactly a *monotone map*.

**Definition (Adjunction in a Poset).** Let $P$ and $Q$ be posets. Monotone maps $f \colon P \to Q$ and $g \colon Q \to P$ form an adjunction $f \dashv g$ if and only if:
$$f(p) \leq q \iff p \leq g(q) \quad \text{for all } p \in P,\, q \in Q.$$
The map $f$ is called a *Galois connection* lower adjoint and $g$ the upper adjoint.

**Proposition.** Let $g \colon Q \to P$ be a monotone map where $Q$ has all infima. Then $g$ has a left adjoint if and only if $g$ preserves all infima. In that case, the left adjoint is:
$$f(p) = \inf\{q \in Q : p \leq g(q)\}.$$

*Proof sketch.* If $f \dashv g$, then for any $p \in P$ the set $\{q \in Q : p \leq g(q)\} = \{q : f(p) \leq q\}$, and its infimum is $f(p)$. This shows $g$ must preserve infima. Conversely, given that $g$ preserves infima, define $f(p) = \inf\{q : p \leq g(q)\}$. One checks $p \leq g(f(p))$ (using infimum closure and preservation) and $f(g(q)) \leq q$, establishing $f \dashv g$.

> [!EXAMPLE] The power set poset
> Let $P = Q = \mathcal{P}(X)$ ordered by inclusion, and let $g = $ the operation $B \mapsto B^c$ (complement). Then $g$ reverses order and does not preserve infima (which are intersections) in general. The poset example shows that infimum-preservation is the *exact* condition needed.

> [!NOTE] Exercise 1
> Let $g \colon Y \to X$ be a monotone map between posets where $Y$ has all infima and $g$ preserves them. Construct an explicit left adjoint $f \colon X \to Y$ to $g$, and verify the adjunction identity $f(x) \leq y \iff x \leq g(y)$.
>
> (Hint: set $f(x) = \inf\{y \in Y : x \leq g(y)\}$. You will need to use infimum-preservation to establish $x \leq g(f(x))$.)

---

## 3. Monads: Definition and First Examples 🔑

### 3.1 The Definition

A *monad* is the categorical abstraction of a monoid acting on itself — or, as the slogan goes, "a monad is just a monoid in the category of endofunctors."

**Definition (Monad).** A monad on a category $\mathcal{C}$ is a triple $(T, \eta, \mu)$ consisting of:
- an endofunctor $T \colon \mathcal{C} \to \mathcal{C}$,
- a natural transformation $\eta \colon \mathrm{id}_{\mathcal{C}} \Rightarrow T$ called the *unit*,
- a natural transformation $\mu \colon T^2 \Rightarrow T$ (where $T^2 = T \circ T$) called the *multiplication*,

subject to the following two axioms, expressed as commutative diagrams:

**Associativity:**

```tikz
\begin{document}
\begin{tikzcd}
T^3 \arrow[r, "T\mu"] \arrow[d, "\mu T"'] & T^2 \arrow[d, "\mu"] \\
T^2 \arrow[r, "\mu"'] & T
\end{tikzcd}
\end{document}
```

That is: $\mu \circ T\mu = \mu \circ \mu T$ as natural transformations $T^3 \Rightarrow T$.

**Unit laws:**

```tikz
\begin{document}
\begin{tikzcd}
T \arrow[r, "\eta T"] \arrow[dr, "\mathrm{id}_T"'] & T^2 \arrow[d, "\mu"] & T \arrow[l, "T\eta"'] \arrow[dl, "\mathrm{id}_T"] \\
& T &
\end{tikzcd}
\end{document}
```

That is: $\mu \circ \eta T = \mathrm{id}_T = \mu \circ T\eta$.

> [!INFO] Monoids in endofunctor categories
> The monoidal category $[\mathcal{C}, \mathcal{C}]$ of endofunctors of $\mathcal{C}$ under composition has unit object $\mathrm{id}_{\mathcal{C}}$ and tensor product $\circ$. A monoid in this monoidal category is precisely a triple $(T, \eta, \mu)$ satisfying the above axioms — the associativity and unit laws for monoids translate directly. This is the origin of the slogan "a monad is a monoid in the category of endofunctors."

**Examples:**

| Monad $T$ on $\mathbf{Set}$ | Unit $\eta_A$ | Multiplication $\mu_A$ | Eilenberg-Moore algebras |
|---|---|---|---|
| $TA = $ free monoid on $A$ (lists) | $a \mapsto [a]$ | concatenation of lists of lists | monoids |
| $TA = \mathcal{P}(A)$ (power set) | $a \mapsto \{a\}$ | union of a set of sets | complete lattices (sup-semilattices) |
| $TA = A + E$ (exception monad, $E$ fixed) | $a \mapsto \mathrm{inl}(a)$ | flatten double-layer | sets with a distinguished element |
| $TA = $ free $R$-module on $A$ | $a \mapsto 1 \cdot a$ | $R$-linearity of evaluation | $R$-modules |

### 3.2 Every Adjunction Gives a Monad

**Proposition.** Let $F \dashv G$ with $F \colon \mathcal{C} \to \mathcal{D}$ and $G \colon \mathcal{D} \to \mathcal{C}$, unit $\eta \colon \mathrm{id}_{\mathcal{C}} \Rightarrow GF$, and counit $\varepsilon \colon FG \Rightarrow \mathrm{id}_{\mathcal{D}}$. Then $(T, \eta, \mu)$ where $T = GF$ and $\mu = G\varepsilon F \colon GFGF \Rightarrow GF$ is a monad on $\mathcal{C}$.

*Proof sketch.* The unit law $\mu \circ \eta T = \mathrm{id}_T$ follows from one of the triangle identities $G\varepsilon \circ \eta G = \mathrm{id}_G$. The law $\mu \circ T\eta = \mathrm{id}_T$ follows from $\varepsilon F \circ F\eta = \mathrm{id}_F$ and applying $G$. Associativity $\mu \circ T\mu = \mu \circ \mu T$ reduces to naturality of $\varepsilon$.

> [!EXAMPLE] The free-forgetful adjunction for groups
> Let $F \colon \mathbf{Set} \to \mathbf{Grp}$ be the free group functor and $U \colon \mathbf{Grp} \to \mathbf{Set}$ the forgetful functor, $F \dashv U$. The monad $T = UF$ sends a set $A$ to the underlying set of the free group on $A$: words in $A \cup A^{-1}$. The unit $\eta_A$ embeds $A$ as length-one words; the multiplication $\mu_A$ flattens a word of words by concatenation and reduction.

### 3.3 Every Monad Arises from an Adjunction

**Theorem.** Every monad $(T, \eta, \mu)$ on $\mathcal{C}$ arises from at least one adjunction. In fact, it arises from (at least) two canonical ones:
1. The *Kleisli adjunction* $F_T \dashv U_T \colon \mathcal{C}_T \to \mathcal{C}$ via the *Kleisli category* $\mathcal{C}_T$ (objects of $\mathcal{C}$, morphisms $A \to B$ in $\mathcal{C}_T$ are morphisms $A \to TB$ in $\mathcal{C}$).
2. The *Eilenberg-Moore adjunction* $F^T \dashv U^T \colon \mathcal{C}^T \to \mathcal{C}$ via the *Eilenberg-Moore category* $\mathcal{C}^T$ of $T$-algebras (defined in §4).

The Kleisli adjunction is the *initial* adjunction giving rise to $(T, \eta, \mu)$; the Eilenberg-Moore adjunction is the *terminal* one.

> [!NOTE] Exercise 2
> (i) Define a monad $(T, \eta, \mu)$ on a category $\mathcal{C}$. State the monad axioms precisely. Explain how every adjunction $F \dashv G \colon \mathcal{C} \rightleftharpoons \mathcal{D}$ gives rise to a monad on $\mathcal{C}$, verifying the axioms.
>
> (ii) A functor $U \colon \mathcal{D} \to \mathcal{C}$ is *monadic* if it has a left adjoint $F \dashv U$ and the comparison functor $K \colon \mathcal{D} \to \mathcal{C}^T$ (where $\mathcal{C}^T$ is the Eilenberg-Moore category for $T = UF$) is an equivalence of categories. Show that every monadic functor is faithful and reflects isomorphisms.

---

## 4. Eilenberg-Moore Algebras 🏗️

### 4.1 T-Algebras and Their Morphisms

**Definition (T-Algebra).** Let $(T, \eta, \mu)$ be a monad on $\mathcal{C}$. A *$T$-algebra* (or *Eilenberg-Moore algebra*) is a pair $(A, a)$ where $A \in \mathcal{C}$ and $a \colon TA \to A$ is a morphism in $\mathcal{C}$, called the *structure map*, satisfying:

- **Unit law:** $a \circ \eta_A = \mathrm{id}_A$ (the unit is a section of $a$),
- **Multiplication law:** $a \circ \mu_A = a \circ Ta$ (the two ways of collapsing $T^2A$ to $A$ agree).

As commutative diagrams:

```tikz
\begin{document}
\begin{tikzcd}
A \arrow[r, "\eta_A"] \arrow[dr, "\mathrm{id}_A"'] & TA \arrow[d, "a"] \\
& A
\end{tikzcd}
\end{document}
```

```tikz
\begin{document}
\begin{tikzcd}
T^2A \arrow[r, "\mu_A"] \arrow[d, "Ta"'] & TA \arrow[d, "a"] \\
TA \arrow[r, "a"'] & A
\end{tikzcd}
\end{document}
```

> [!EXAMPLE] T-algebras for the list monad
> For the free-monoid monad on $\mathbf{Set}$, a $T$-algebra $(A, a)$ is a set $A$ together with a map $a \colon A^* \to A$ (evaluation of words in $A$) satisfying: $a([x]) = x$ (unit law) and $a$ extends to concatenation in the expected way. These are precisely *monoids*: $a$ gives the multiplication and the unit is $a([\,])$.

**Definition (T-Algebra Morphism).** A *morphism of $T$-algebras* $f \colon (A, a) \to (B, b)$ is a morphism $f \colon A \to B$ in $\mathcal{C}$ such that $f \circ a = b \circ Tf$.

```tikz
\begin{document}
\begin{tikzcd}
TA \arrow[r, "Tf"] \arrow[d, "a"'] & TB \arrow[d, "b"] \\
A \arrow[r, "f"'] & B
\end{tikzcd}
\end{document}
```

### 4.2 The Eilenberg-Moore Category

**Definition (Eilenberg-Moore Category).** The *Eilenberg-Moore category* $\mathcal{C}^T$ has:
- Objects: $T$-algebras $(A, a)$,
- Morphisms: $T$-algebra morphisms,
- Composition and identities inherited from $\mathcal{C}$.

There is a canonical adjunction $F^T \dashv U^T \colon \mathcal{C}^T \rightleftharpoons \mathcal{C}$:
- The *forgetful functor* $U^T \colon \mathcal{C}^T \to \mathcal{C}$ sends $(A, a) \mapsto A$.
- The *free $T$-algebra functor* $F^T \colon \mathcal{C} \to \mathcal{C}^T$ sends $A \mapsto (TA, \mu_A)$.

*Verification that $(TA, \mu_A)$ is a $T$-algebra:* unit law $\mu_A \circ \eta_{TA} = \mathrm{id}_{TA}$ is the monad unit law, and multiplication law $\mu_A \circ \mu_{TA} = \mu_A \circ T\mu_A$ is monad associativity.

The monad induced by $F^T \dashv U^T$ is $U^T F^T = T$ with unit $\eta$ and multiplication $\mu$, recovering the original monad.

> [!TIP] Intuition: free vs. presented algebras
> The free $T$-algebra $(TA, \mu_A)$ is "purely formal" — it has no relations imposed. A general $T$-algebra $(A, a)$ has the same structure but with the map $a \colon TA \to A$ imposing relations (quotienting the free algebra). The unit law says $a$ is a retraction of $\eta_A$.

### 4.3 The Comparison Functor

For any adjunction $F \dashv G \colon \mathcal{C} \rightleftharpoons \mathcal{D}$ with counit $\varepsilon$ and induced monad $T = GF$ on $\mathcal{C}$, the *comparison functor* $K \colon \mathcal{D} \to \mathcal{C}^T$ is defined by:
$$K(d) = (G(d),\, G(\varepsilon_d)) \quad \text{and} \quad K(f) = G(f).$$

One checks that $(G(d), G(\varepsilon_d))$ is indeed a $T$-algebra: the unit law uses the triangle identity $G\varepsilon \circ \eta G = \mathrm{id}_G$, and the multiplication law follows from naturality of $\varepsilon$.

**Definition (Monadic Functor).** $G \colon \mathcal{D} \to \mathcal{C}$ is *monadic* if it has a left adjoint and the comparison functor $K \colon \mathcal{D} \to \mathcal{C}^T$ is an equivalence of categories.

> [!WARNING] Monadicity is a property of $G$, not just of $T$
> Different adjunctions can give the same monad. Being monadic is the assertion that $\mathcal{D}$ is, up to equivalence, the Eilenberg-Moore category for $T$ — i.e., $\mathcal{D}$ is the *terminal* resolution of $T$.

---

## 5. Absolute Coequalizers and Split Pairs ⚙️

### 5.1 Reflexive Pairs

**Definition (Reflexive Pair).** A parallel pair $f, g \colon A \rightrightarrows B$ is *reflexive* if there exists a *common section* $s \colon B \to A$ with $fs = \mathrm{id}_B = gs$.

> [!INFO] Reflexive pairs in algebra
> In $\mathbf{Grp}$, the kernel pair of a group homomorphism $\phi \colon G \to H$ is always reflexive (the diagonal $G \to G \times_H G$ is the section). Reflexive coequalizers are often better-behaved than arbitrary coequalizers in algebra.

### 5.2 Split Coequalizers

**Definition (Split Coequalizer).** A diagram
$$A \overset{f}{\underset{g}{\rightrightarrows}} B \xrightarrow{q} C$$
is a *split coequalizer* (or *absolute coequalizer*) if there exist morphisms $s \colon B \to A$ and $t \colon C \to B$ such that:
1. $q \circ f = q \circ g$,
2. $f \circ s = \mathrm{id}_B$,
3. $q \circ t = \mathrm{id}_C$,
4. $g \circ s = t \circ q$.

```tikz
\begin{document}
\begin{tikzcd}
A \arrow[r, "f", shift left=2] \arrow[r, "g"', shift right=2] & B \arrow[r, "q"] \arrow[l, "s", bend right=40] & C \arrow[l, "t", bend right=40]
\end{tikzcd}
\end{document}
```

*Conditions 1–4 together imply $q$ is a coequalizer of $f$ and $g$.*

**Proposition (Absoluteness).** Every split coequalizer is preserved by every functor.

*Proof.* The conditions 1–4 are purely equational. Applying any functor $H$ preserves all equations, so $H(A) \rightrightarrows H(B) \to H(C)$ with $H(s)$ and $H(t)$ satisfies the same conditions, making $H(q)$ a split coequalizer.

> [!NOTE] Exercise 3
> (i) Define *absolute coequalizer pair* (split coequalizer). Define what it means for a functor $U$ to *preserve*, *reflect*, or *create* coequalizers of such pairs.
>
> (ii) Prove the equivalence of the following three conditions on a functor $U \colon \mathcal{D} \to \mathcal{C}$:
> - (a) $U$ creates coequalizers of $U$-absolute pairs (pairs whose image under $U$ is a split coequalizer).
> - (b) $\mathcal{D}$ has coequalizers of $U$-absolute pairs and $U$ preserves and reflects them.
> - (c) Same as (b) but only $U$ preserves them (not necessarily reflects).

> [!NOTE] Exercise 4
> Let $(T, \eta, \mu)$ be a monad on $\mathcal{C}$ and let $(A, a \colon TA \to A)$ be a $T$-algebra. Consider the parallel pair
> $$T^2A \overset{\mu_A}{\underset{Ta}{\rightrightarrows}} TA.$$
> Show that this is a reflexive pair (exhibit a common section). Then show that with the additional morphisms $s = \eta_{TA} \colon TA \to T^2A$ and $t = \eta_A \colon A \to TA$, the diagram
> $$T^2A \overset{\mu_A}{\underset{Ta}{\rightrightarrows}} TA \xrightarrow{a} A$$
> is a split coequalizer.

---

## 6. Set-Adjoint, Representable, and Limit-Preserving Functors 🧩

Three conditions on a functor $U \colon \mathcal{C} \to \mathbf{Set}$ are naturally ordered by strength:

- **(A) Set-adjoint:** $U$ has a left adjoint $F \dashv U$ (equivalently, for every set $S$ there is a "free object" $F(S) \in \mathcal{C}$ with $\mathcal{C}(F(S), C) \cong \mathbf{Set}(S, UC)$ naturally).
- **(R) Representable:** $U \cong \mathcal{C}(A, -)$ for some $A \in \mathcal{C}$ (naturally in the $\mathcal{C}$ variable).
- **(L) Limit-preserving:** $U$ preserves all small limits.

**Proposition.** $(A) \Rightarrow (R) \Rightarrow (L)$.

*Proof.* $(A) \Rightarrow (R)$: If $F \dashv U$ then $U(C) \cong \mathbf{Set}(1, UC) \cong \mathcal{C}(F(1), C)$ naturally, so $U$ is represented by $F(1)$.

$(R) \Rightarrow (L)$: Representable functors are right adjoints to $\mathcal{C}(-,A) \colon \mathcal{C}^{op} \to \mathbf{Set}$... more directly, $\mathcal{C}(A,-)$ preserves limits since limits in $\mathbf{Set}$ of hom-sets compute correctly (this is the standard fact that hom-functors preserve limits).

**Proposition.** If $\mathcal{C}$ has all small coproducts, then $(R) \Rightarrow (A)$.

*Proof sketch.* Given $U \cong \mathcal{C}(A,-)$, define $F(S) = \coprod_{s \in S} A$ (coproduct of $|S|$ copies of $A$). Then $\mathcal{C}(F(S), C) \cong \prod_{s \in S} \mathcal{C}(A, C) \cong \mathbf{Set}(S, \mathcal{C}(A,C)) \cong \mathbf{Set}(S, UC)$.

> [!NOTE] Exercise 5
> Let $U \colon \mathcal{C} \to \mathbf{Set}$. Consider: (A) $U$ has a left adjoint; (R) $U$ is representable; (L) $U$ preserves all limits. Prove $A \Rightarrow R \Rightarrow L$. If $\mathcal{C}$ has all small coproducts, prove $R \Rightarrow A$. If $\mathcal{C}$ is complete, well-powered, and has a coseparating set, prove that all three conditions are equivalent.

---

## 7. The General Adjoint Functor Theorem 📐

### 7.1 The Solution Set Condition

**Definition (Solution Set Condition).** A functor $G \colon \mathcal{D} \to \mathcal{C}$ satisfies the *solution set condition* if for every object $A \in \mathcal{C}$ there exists a small set $I$ and a family of morphisms $\{f_i \colon A \to G(D_i)\}_{i \in I}$ such that every morphism $h \colon A \to G(D)$ in $\mathcal{C}$ factors as
$$A \xrightarrow{f_i} G(D_i) \xrightarrow{G(k)} G(D)$$
for some $i \in I$ and some $k \colon D_i \to D$ in $\mathcal{D}$.

In categorical language: the full subcategory of $(A \downarrow G)$ on $\{(D_i, f_i)\}_{i \in I}$ is a *weakly initial set* for the comma category.

> [!INFO] Why "solution set"?
> For each object $A$, one seeks a "left adjoint object" $F(A)$ with a universal morphism $A \to GF(A)$. This is an initial object of $(A \downarrow G)$. The solution set condition is a weak version: instead of requiring one initial object, it only requires a small set through which all morphisms factor. The actual initial object is then constructed by taking a limit.

### 7.2 Proof Sketch

**Theorem (General Adjoint Functor Theorem, Freyd).** Let $G \colon \mathcal{D} \to \mathcal{C}$ where $\mathcal{D}$ is locally small and complete. Then $G$ has a left adjoint if and only if:
1. $G$ preserves all small limits, and
2. $G$ satisfies the solution set condition.

*Proof sketch (sufficiency).* Fix $A \in \mathcal{C}$. We must construct an initial object of $(A \downarrow G)$.

**Step 1: Weakly initial object.** By the SSC, there is a small set $\{(D_i, f_i \colon A \to GD_i)\}_{i \in I}$. Let $W = \prod_{i \in I} D_i$ (existing by completeness). The projections $W \to D_i$ give a morphism $A \to GW$ (using limit-preservation of $G$ and the $f_i$). This makes $(W, A \to GW)$ a weakly initial object in $(A \downarrow G)$: every $(D, h)$ is hit via the relevant $D_i \to D$ composed with $W \to D_i$.

**Step 2: Reflexive equalizer to get an initial object.** The comma category $(A \downarrow G)$ is locally small (since $\mathcal{D}$ is locally small). Consider the set of all endomorphisms of $W$ in $(A \downarrow G)$. Take the *equalizer* of all of them simultaneously (a small diagram since $\mathcal{D}(W,W)$ is a set). The resulting object is an initial object of $(A \downarrow G)$, which is the required universal arrow $A \to G(FA)$.

**Necessity.** Right adjoints preserve limits (File 3). For the SSC: take $I = \{\ast\}$ and $f_\ast = \eta_A \colon A \to GF(A)$; every $h \colon A \to GD$ factors through $\eta_A$ by universality.

**Key conclusion: $G$ has a left adjoint if and only if $G$ preserves limits and satisfies the SSC.** This is the most general sufficient condition for existence of a left adjoint.

> [!NOTE] Exercise 6
> (i) State the General Adjoint Functor Theorem precisely, including all hypotheses.
>
> (ii) Let $\mathcal{D}$ be complete and locally small, and suppose $\mathcal{D}$ has a *weakly initial object*: an object $W$ such that for every object $D \in \mathcal{D}$ there exists at least one morphism $W \to D$. Show that $\mathcal{D}$ has an initial object.
>
> (iii) Outline how part (ii) is used in the proof of the GAFT to produce an initial object of $(A \downarrow G)$ from the weakly initial set given by the SSC.

---

## 8. Well-Powered Categories and the Subobject Functor 🔬

**Definition (Subobject).** A *subobject* of $A \in \mathcal{C}$ is an isomorphism class of monomorphisms $m \colon S \hookrightarrow A$. (Two monics $m \colon S \to A$ and $m' \colon S' \to A$ are isomorphic as subobjects if there is an isomorphism $\phi \colon S \xrightarrow{\sim} S'$ with $m' \phi = m$.)

**Definition (Well-Powered).** A category $\mathcal{C}$ is *well-powered* if for every object $A$, the collection $\mathrm{Sub}(A)$ of subobjects of $A$ is a set (not a proper class).

> [!EXAMPLE] Well-powered and non-well-powered categories
>
> | Category | Well-powered? | Reason |
> |---|---|---|
> | $\mathbf{Set}$ | Yes | Subsets of $A$ form a set |
> | $\mathbf{Grp}$ | Yes | Subgroups of $G$ form a set |
> | $\mathbf{Top}$ | Yes | Subspaces of $X$ form a set |
> | $\mathbf{Ord}$ (all ordinals) | No | Every ordinal embeds into every larger ordinal |
> | $\mathbf{Cat}$ | No | A category can have a proper class of full subcategories |

**Proposition.** In a well-powered category $\mathcal{C}$ with pullbacks, there is a well-defined functor
$$\mathrm{Sub} \colon \mathcal{C}^{op} \to \mathbf{Set}$$
sending each object $A$ to $\mathrm{Sub}(A)$, and each morphism $f \colon A \to B$ to the pullback-along-$f$ map $f^* \colon \mathrm{Sub}(B) \to \mathrm{Sub}(A)$.

*Proof sketch.* Given a monic $m \colon S \hookrightarrow B$ representing a subobject of $B$, form the pullback:
$$\begin{array}{ccc} f^*S & \to & S \\ \downarrow & & \downarrow m \\ A & \xrightarrow{f} & B \end{array}$$
The pullback of a monic is monic (a standard lemma), so $f^*S \hookrightarrow A$ is a subobject of $A$. Well-poweredness ensures $f^*$ is a function between sets. Functoriality follows from the universal property of pullbacks.

> [!NOTE] Exercise 7
> Define a well-powered category. For such a category with pullbacks, show that the assignment $A \mapsto \mathrm{Sub}(A)$, $f \mapsto f^*$ defines a functor $\mathrm{Sub} \colon \mathcal{C}^{op} \to \mathbf{Set}$. In particular, verify that $\mathrm{id}^* = \mathrm{id}$ and $(f \circ g)^* = g^* \circ f^*$.

---

## 9. The Special Adjoint Functor Theorem ✨

### 9.1 Coseparating Sets

**Definition (Coseparating Set).** A set $\mathcal{S}$ of objects of $\mathcal{C}$ is a *coseparating set* (or *cogenerating set*) if for every pair of distinct parallel morphisms $f, g \colon A \rightrightarrows B$ in $\mathcal{C}$ with $f \neq g$, there exists $S \in \mathcal{S}$ and a morphism $h \colon B \to S$ with $h \circ f \neq h \circ g$.

Equivalently: the family of functors $\{\mathcal{C}(-, S)\}_{S \in \mathcal{S}}$ is *jointly faithful*.

> [!EXAMPLE] Coseparating sets in standard categories
> - In $\mathbf{Set}$: $\mathcal{S} = \{\{0,1\}\}$ works (the two-element set separates all functions).
> - In $\mathbf{Ab}$: $\mathcal{S} = \{\mathbb{Q}/\mathbb{Z}\}$ works (Pontryagin duality).
> - In $\mathbf{Top}$: $\mathcal{S} = \{[0,1]\}$ works for compact Hausdorff spaces (by Urysohn).
> - In $\mathbf{Grp}$: no finite coseparating set exists, but $\mathcal{S} = $ all finite groups works.

### 9.2 Proof Sketch via GAFT

**Theorem (Special Adjoint Functor Theorem, Freyd).** Let $\mathcal{C}$ be locally small, complete, well-powered, and with a small coseparating set $\mathcal{S}$. Then every limit-preserving functor $G \colon \mathcal{C} \to \mathcal{D}$ has a left adjoint.

*Proof sketch.* By the GAFT, it suffices to verify the SSC. Fix $A \in \mathcal{C}$ and let $f \colon A \to G(D)$ be any morphism.

**Constructing the solution set.** Consider the *equalizer* $E_D = \mathrm{Eq}(G \circ \pi_S)_{S \in \mathcal{S}}$... more concretely:

For each object $D \in \mathcal{D}$, the morphism $f \colon A \to G(D)$ factors through the *intersection of all subobjects of $G(D)$ through which $f$ factors*. Call this $\mathrm{Im}_{\mathcal{S}}(f)$. Well-poweredness ensures $\mathrm{Sub}(G(D))$ is a set, and the coseparating set ensures the image can be computed using $\mathcal{S}$.

Precisely: define $\mathcal{I}$ to be the set of all pairs $(D', f') $ where $D'$ is a subobject of some $\prod_{S \in \mathcal{S}} S^{n}$ for some finite $n$ and $f' \colon A \to G(D')$. Well-poweredness (applied to the product $\prod_{S \in \mathcal{S}} S^n$, which exists by completeness) makes $\mathcal{I}$ a set. The coseparating property ensures every $f \colon A \to G(D)$ factors through some element of $\mathcal{I}$. Hence $\mathcal{I}$ is the required solution set.

*This is the key:* well-poweredness controls the size of $\mathrm{Sub}$, and the coseparating set ensures every object can be separated into this controlled size. Together they imply SSC without needing to know the specific functor $G$.

> [!NOTE] Exercise 8
> State and prove the Special Adjoint Functor Theorem, assuming the General Adjoint Functor Theorem. In your proof, identify explicitly where well-poweredness is used (to bound the size of the solution set) and where the coseparating set is used (to ensure every morphism factors through the solution set).

---

## 10. Applications of GAFT and SAFT 🌐

**Free algebraic structures (GAFT).** For any finitary algebraic theory $\mathbb{T}$ (groups, rings, $R$-modules, lattices, etc.), the forgetful functor $U \colon \mathbf{T}\text{-}\mathbf{Alg} \to \mathbf{Set}$ preserves limits and satisfies the SSC (the solution set for a set $S$ is the single object $F(S)$ = the $\mathbb{T}$-algebra presented by generators $S$ and no relations). Hence $U$ has a left adjoint, the free-algebra functor.

> [!TIP] The GAFT turns existence into a checking game
> To show a left adjoint exists for $G \colon \mathcal{D} \to \mathcal{C}$: (1) verify $G$ preserves limits, and (2) for each $A$, exhibit a small set of morphisms $A \to G(D_i)$ through which all others factor. You never need to construct the left adjoint explicitly.

**Stone-Čech compactification (SAFT).** Let $i \colon \mathbf{KHaus} \hookrightarrow \mathbf{Top}$ be the inclusion of compact Hausdorff spaces into all topological spaces. $\mathbf{KHaus}$ is complete (products and equalizers of compact Hausdorff spaces are compact Hausdorff), well-powered (subspaces of compact Hausdorff spaces form a set), and has a small coseparating set ($\{[0,1]\}$ works by Urysohn's theorem). The inclusion $i$ trivially preserves limits. Hence by the SAFT, $i$ has a left adjoint — the *Stone-Čech compactification* $\beta \colon \mathbf{Top} \to \mathbf{KHaus}$.

**Geometric realization (SAFT).** Covered in §16.

> [!NOTE] Exercise 9
> (i) Apply the GAFT to show that the forgetful functor $\mathbf{Widget} \to \mathbf{Set}$ has a left adjoint (free widget functor), where $\mathbf{Widget}$ is any category of algebras for a finitary algebraic theory.
>
> (ii) Apply similarly to $\mathbf{Chad} \to \mathbf{Set}$ for another finitary theory $\mathbf{Chad}$.
>
> (iii) Using the Beck Monadicity Theorem (proved in §14), show that $\mathbf{Widget}$ is monadic over $\mathbf{Set}$. Which of the three conditions of Beck's theorem do you need to verify for $U \colon \mathbf{Widget} \to \mathbf{Set}$?

---

## 11. The Eilenberg-Moore Category: Creation Theorems 🏗️

### 11.1 Creation of Limits

**Theorem.** The forgetful functor $U^T \colon \mathcal{C}^T \to \mathcal{C}$ *creates* all limits.

*Proof sketch.* Let $D \colon \mathcal{J} \to \mathcal{C}^T$ be a diagram of $T$-algebras $\{(A_j, a_j)\}$. Let $A = \lim_j U^T(A_j) = \lim_j A_j$ in $\mathcal{C}$, with projections $\pi_j \colon A \to A_j$.

Since $T$ is a functor, $TA = T(\lim_j A_j)$ with projections $T\pi_j \colon TA \to TA_j$. Define $a \colon TA \to A$ by the universal property of $A$ (as a limit): the maps $a_j \circ T\pi_j \colon TA \to A_j$ are compatible (using the $T$-algebra morphisms in the diagram), so there is a unique $a \colon TA \to A$ with $\pi_j \circ a = a_j \circ T\pi_j$ for all $j$.

Checking $(A, a)$ is a $T$-algebra: both the unit law and multiplication law hold componentwise (at each $\pi_j$) and are inherited from the $A_j$ by uniqueness of the limit morphism.

**Corollary.** $\mathcal{C}^T$ is complete whenever $\mathcal{C}$ is complete.

### 11.2 Creation of Coequalizers for Absolute Pairs

**Theorem.** $U^T$ creates coequalizers of $U^T$-absolute pairs.

*Proof sketch.* Let $f, g \colon (A, a) \rightrightarrows (B, b)$ be a pair of $T$-algebra morphisms such that $U^T f, U^T g \colon A \rightrightarrows B$ has a split coequalizer $q \colon B \to C$ with splitting $s \colon B \to A$, $t \colon C \to B$ in $\mathcal{C}$. We must produce a $T$-algebra structure on $C$ making $q$ a coequalizer in $\mathcal{C}^T$.

Define $c \colon TC \to C$ by: since $q \colon B \to C$ has a section $t$, we set $c = q \circ b \circ Tt$. One verifies:
- $c$ is well-defined (independent of the choice of factorization, using that $q$ coequalizes $f$ and $g$).
- The unit law for $c$: $c \circ \eta_C = q \circ b \circ Tt \circ \eta_C = q \circ b \circ \eta_B \circ t = q \circ \mathrm{id}_B \circ t = q \circ t = \mathrm{id}_C$.
- The multiplication law: a similar computation using associativity of $b$ and naturality.

**Moreover,** every $T$-algebra is a coequalizer of free algebras:

**Proposition.** For any $T$-algebra $(A, a)$, the diagram
$$F^T(TA) \overset{F^T(a)}{\underset{\mu_A^{\sharp}}{\rightrightarrows}} F^T(A) \xrightarrow{a^{\sharp}} (A, a)$$
is a coequalizer in $\mathcal{C}^T$, where $a^{\sharp}$ is the $T$-algebra map $(TA, \mu_A) \to (A, a)$ corresponding to $a$ under the adjunction, and the parallel pair is identified by Exercise 4.

> [!NOTE] Exercise 10
> (i) Show that the forgetful functor $U^T \colon \mathcal{C}^T \to \mathcal{C}$ creates all limits, and hence $\mathcal{C}^T$ is complete whenever $\mathcal{C}$ is.
>
> (ii) Show that $U^T$ creates coequalizers of $U^T$-absolute coequalizer pairs. (Construct the $T$-algebra structure on the coequalizer object and verify the two algebra axioms.)
>
> (iii) Show that every $T$-algebra $(A, a)$ is a coequalizer of free $T$-algebras; specifically, that $(A, a) = \mathrm{coeq}((TA, \mu_A) \rightrightarrows (TA, \mu_A))$ for a suitable parallel pair arising from the pair $\mu_A, Ta \colon T^2A \rightrightarrows TA$ of Exercise 4.

---

## 12. Idempotent Monads 🔄

**Definition (Idempotent Monad).** A monad $(T, \eta, \mu)$ on $\mathcal{C}$ is *idempotent* if the multiplication $\mu \colon T^2 \Rightarrow T$ is a natural isomorphism.

> [!INFO] Equivalent characterizations
> For a monad $(T, \eta, \mu)$, the following are equivalent:
> - $\mu$ is a natural isomorphism,
> - $\eta T = T\eta$ (the two units $T \Rightarrow T^2$ coincide),
> - $T\eta \colon T \Rightarrow T^2$ is a natural isomorphism.

**Proposition.** For an idempotent monad $(T, \eta, \mu)$:
1. Every $T$-algebra $(A, a)$ is isomorphic to the free $T$-algebra $(TA, \mu_A)$.
2. The Eilenberg-Moore category $\mathcal{C}^T$ is equivalent to the full subcategory of $T$-objects in $\mathcal{C}$: objects $A$ with $\eta_A \colon A \to TA$ an isomorphism.

*Proof.* For (1): given $(A, a \colon TA \to A)$, the unit law gives $a \circ \eta_A = \mathrm{id}_A$, so $\eta_A$ is a split monomorphism. The multiplication law $a \circ \mu_A = a \circ Ta$ with $\mu_A$ an iso gives $a \circ \mu_A \circ (\mu_A)^{-1} = a \circ Ta \circ (\mu_A)^{-1}$, which together with naturality shows $\eta_A$ is also an epimorphism (using $T\eta_A = \eta_{TA}$ and the unit law). Hence $\eta_A \colon A \to TA$ is an isomorphism, and $(A, a) \cong (TA, \mu_A)$.

**Idempotent monads and reflective subcategories.** Idempotent monads on $\mathcal{C}$ correspond bijectively to *reflective subcategories* of $\mathcal{C}$: the Eilenberg-Moore category $\mathcal{C}^T$ is the reflective subcategory, and the reflector is $T$ itself.

> [!EXAMPLE] Sheafification
> The sheafification monad on the category of presheaves $[\mathcal{C}^{op}, \mathbf{Set}]$ is idempotent: sheaves are exactly the presheaves fixed by sheafification ($T(F) \cong F$ iff $F$ is a sheaf). The Eilenberg-Moore category is the full subcategory of sheaves, a reflective subcategory.

> [!NOTE] Exercise 11
> Define an idempotent monad as one where $\mu \colon T^2 \Rightarrow T$ is a natural isomorphism. Prove that every $T$-algebra for an idempotent monad is free, i.e., isomorphic (as a $T$-algebra) to $(TA, \mu_A)$ for some $A \in \mathcal{C}$. Deduce that the forgetful functor $U^T \colon \mathcal{C}^T \to \mathcal{C}$ is fully faithful for idempotent monads.

---

## 13. Presheaves Are Monadic 📦

Let $\mathcal{C}$ be a small category with object set $\mathrm{ob}(\mathcal{C})$. Write $\mathbf{Set}^{\mathrm{ob}(\mathcal{C})}$ for the product category $\prod_{C \in \mathrm{ob}(\mathcal{C})} \mathbf{Set}$ (families of sets indexed by objects of $\mathcal{C}$). There is a forgetful functor
$$V \colon [\mathcal{C}^{op}, \mathbf{Set}] \to \mathbf{Set}^{\mathrm{ob}(\mathcal{C})}$$
sending a presheaf $F$ to the family $(F(C))_{C \in \mathrm{ob}(\mathcal{C})}$ of its values (forgetting the restriction maps).

**Theorem.** $V$ is monadic.

*Proof sketch.* The left adjoint $L \colon \mathbf{Set}^{\mathrm{ob}(\mathcal{C})} \to [\mathcal{C}^{op}, \mathbf{Set}]$ sends a family $(S_C)_C$ to the presheaf $L(S)(D) = \bigsqcup_{C \in \mathrm{ob}(\mathcal{C})} S_C \times \mathcal{C}(D, C)$ (the "free presheaf" on the family). The induced monad $T = VL$ on $\mathbf{Set}^{\mathrm{ob}(\mathcal{C})}$ sends $(S_C)_C$ to $\left(\bigsqcup_{C'} S_{C'} \times \mathcal{C}(C, C')\right)_C$. A $T$-algebra structure precisely encodes the action of all hom-sets $\mathcal{C}(C, C')$ on the family, which is exactly a presheaf on $\mathcal{C}$. The comparison functor is an isomorphism (not just equivalence) of categories.

**Corollaries:**
1. $[\mathcal{C}^{op}, \mathbf{Set}]$ creates limits from $\mathbf{Set}^{\mathrm{ob}(\mathcal{C})}$, hence is complete (limits computed objectwise).
2. $V$ reflects isomorphisms: a natural transformation is an isomorphism iff all components are isomorphisms.
3. Every presheaf is a colimit of representable presheaves (recoverable as a corollary, since the free presheaf $L(S)$ is a coproduct of representables $\mathcal{C}(-,C)$, and every presheaf is a coequalizer of free presheaves by §11).

> [!NOTE] Exercise 12
> (i) Show that the forgetful functor $V \colon [\mathcal{C}^{op}, \mathbf{Set}] \to \mathbf{Set}^{\mathrm{ob}(\mathcal{C})}$ is monadic for small $\mathcal{C}$. (Identify the left adjoint explicitly and verify the three conditions of Beck's theorem.)
>
> (ii) Deduce from monadicity that $[\mathcal{C}^{op}, \mathbf{Set}]$ creates limits and reflects isomorphisms.
>
> (iii) Recover the proof that every presheaf is a colimit of representable presheaves using the coequaliziation result of §11 together with the explicit form of the free presheaf functor.

---

## 14. The Beck Monadicity Theorem 🏆

### 14.1 The Huge Monadicity Theorem

**Definition (U-Split Pair / U-Absolute Pair).** Let $U \colon \mathcal{D} \to \mathcal{C}$. A parallel pair $f, g \colon A \rightrightarrows B$ in $\mathcal{D}$ is *$U$-split* (or *$U$-absolute*) if $Uf, Ug \colon UA \rightrightarrows UB$ has a split coequalizer in $\mathcal{C}$.

**Theorem (Beck Monadicity / Huge Monadicity).** Let $U \colon \mathcal{D} \to \mathcal{C}$ have a left adjoint $F \dashv U$ with induced monad $T = UF$ on $\mathcal{C}$. The following five conditions are equivalent:

| # | Condition |
|---|---|
| (i) | $\mathcal{D}$ has coequalizers of $U$-absolute pairs and $U$ creates them. |
| (ii) | $\mathcal{D}$ has coequalizers of $U$-absolute pairs and $U$ preserves and reflects them. |
| (iii) | $\mathcal{D}$ has coequalizers of $U$-absolute pairs and $U$ preserves them. |
| (iv) | $U$ reflects isomorphisms and $\mathcal{D}$ has coequalizers of $U$-split pairs which $U$ preserves. |
| (v) | $U$ is monadic (the comparison functor $K \colon \mathcal{D} \to \mathcal{C}^T$ is an equivalence). |

**Corollary (Crude Monadicity, Beck).** $U$ is monadic iff $U$ has a left adjoint, $U$ reflects isomorphisms, and $\mathcal{D}$ has and $U$ preserves coequalizers of $U$-split pairs.

> [!INFO] Historical note
> Beck's monadicity theorem first appeared in an untitled manuscript circulated by Jonathan Mock Beck around 1966–1968, published in the proceedings of the 1968 Battelle Memorial Institute conference. It is sometimes called the *Beck tripleability theorem* because *triple* was the earlier term for monad. The theorem is fundamental in *descent theory* and *algebraic geometry*, appearing in Grothendieck's treatment of faithfully flat descent.

### 14.2 Proof Strategy

The proof of the equivalence of (i)–(v) proceeds as follows.

**$(v) \Rightarrow (i)$:** If $K$ is an equivalence, then $\mathcal{D} \simeq \mathcal{C}^T$. By §11, $U^T$ creates coequalizers of $U^T$-absolute pairs. Transporting along $K$ gives the same for $U$.

**$(i) \Rightarrow (ii) \Rightarrow (iii)$:** Creation implies preservation and reflection; preservation and reflection implies preservation. These are formal implications from the definitions.

**$(iii) \Rightarrow (iv)$:** Suppose $U$ preserves coequalizers of $U$-absolute pairs. For any isomorphism-reflecting: if $Uf$ is an isomorphism, then $f, \mathrm{id}$ is a $U$-split pair (with $U\mathrm{id}$ as splitting), so their coequalizer in $\mathcal{D}$ exists and $U$ preserves it — a short diagram chase shows $f$ is an isomorphism.

**$(iv) \Rightarrow (v)$ (key direction).** This is the hardest step. We must show the comparison functor $K \colon \mathcal{D} \to \mathcal{C}^T$ is an equivalence. For every $T$-algebra $(A, a) \in \mathcal{C}^T$, we need an object $d \in \mathcal{D}$ with $K(d) \cong (A, a)$.

By §11, $(A, a)$ is a coequalizer of free $T$-algebras. The corresponding diagram in $\mathcal{D}$ (via $F$) is $F(TA) \rightrightarrows F(A)$, which is a $U$-split pair (since applying $U$ gives the split coequalizer of Exercise 4). By hypothesis (iv), $\mathcal{D}$ has a coequalizer $d = \mathrm{coeq}(F(TA) \rightrightarrows F(A))$, and $U$ preserves it. One checks $K(d) \cong (A, a)$ using that $U$ reflects isomorphisms.

> [!NOTE] Exercise 13
> State and prove the Huge Monadicity Theorem, establishing the equivalence of the five conditions (i)–(v) for a functor $U \colon \mathcal{D} \to \mathcal{C}$ with left adjoint $F$. You may assume the results of §11 (creation theorems for $U^T$). Pay particular attention to the direction (iii) $\Rightarrow$ (iv) (showing $U$ reflects isomorphisms) and (iv) $\Rightarrow$ (v) (showing $K$ is an equivalence).

---

## 15. The Comparison Functor Revisited 🔭

Recall the comparison functor $K \colon \mathcal{D} \to \mathcal{C}^T$ for an adjunction $F \dashv U$ with monad $T = UF$.

**Proposition.** $K$ has a left adjoint if $\mathcal{D}$ has coequalizers of $U$-absolute pairs and $U$ preserves them.

*Proof sketch.* For each $T$-algebra $(A, a)$, we saw in §14 that $(A, a)$ is a coequalizer of free algebras $K(FTA) \rightrightarrows K(FA)$ in $\mathcal{C}^T$. The $U$-preservation hypothesis ensures $\mathcal{D}$ has the corresponding coequalizer, giving a functor $L \colon \mathcal{C}^T \to \mathcal{D}$ with $L(A, a) = \mathrm{coeq}(FTA \rightrightarrows FA)$. One verifies $L \dashv K$ by a universal property argument.

**Proposition.** The unit $\eta^{LK} \colon \mathrm{id}_{\mathcal{D}} \Rightarrow KL$ of $L \dashv K$ is a natural isomorphism iff $U$ reflects isomorphisms.

**Corollary.** Both the unit and counit of $L \dashv K$ are natural isomorphisms (i.e., $K$ is an equivalence of categories) iff $U$ is monadic.

> [!NOTE] Exercise 14
> (i) Show that the comparison functor $K \colon \mathcal{D} \to \mathcal{C}^T$ has a left adjoint $L \colon \mathcal{C}^T \to \mathcal{D}$ whenever $\mathcal{D}$ has coequalizers of $U$-absolute pairs and $U$ preserves them. Describe the action of $L$ on objects.
>
> (ii) Show that the unit $\eta \colon \mathrm{id}_{\mathcal{D}} \Rightarrow KL$ is a natural isomorphism iff $U$ reflects isomorphisms.
>
> (iii) Show that both unit and counit of $L \dashv K$ are natural isomorphisms (so $K$ is an equivalence) iff $U$ is monadic.

---

## 16. Geometric Realization via SAFT 🌐

The *simplex category* $\Delta$ has:
- Objects: finite non-empty ordinals $[n] = \{0 < 1 < \cdots < n\}$ for $n \geq 0$,
- Morphisms: order-preserving (non-strictly) functions $[m] \to [n]$.

A *simplicial set* is a functor $X \colon \Delta^{op} \to \mathbf{Set}$. The category of simplicial sets is $\mathbf{sSet} = [\Delta^{op}, \mathbf{Set}]$.

The *standard geometric simplices* are $|\Delta^n| = \{(t_0, \ldots, t_n) \in \mathbb{R}^{n+1}_{\geq 0} : \sum t_i = 1\}$. These assemble into a functor $|\Delta^{\bullet}| \colon \Delta \to \mathbf{Top}$.

The *singular complex* functor $\mathrm{Sing} \colon \mathbf{Top} \to \mathbf{sSet}$ is defined by:
$$\mathrm{Sing}(X)_n = \mathbf{Top}(|\Delta^n|, X).$$
Its restriction and degeneracy maps are precomposition with the face and degeneracy maps of $\Delta$.

**Theorem.** $\mathrm{Sing}$ has a left adjoint $|-| \colon \mathbf{sSet} \to \mathbf{Top}$, called *geometric realization*.

*Proof via SAFT.* Apply the SAFT to $G = \mathrm{Sing} \colon \mathbf{Top} \to \mathbf{sSet}$... but wait: we want the left adjoint to $\mathrm{Sing}$, so we apply SAFT to the opposite category situation, or more directly apply GAFT:

$\mathrm{Sing}$ preserves limits (since $\mathbf{Top}(|\Delta^n|, -)$ preserves limits for each $n$, limits in $\mathbf{sSet}$ are computed pointwise). For the SSC: for any simplicial set $X$, the solution set consists of a single object — the geometric realization $|X|$ if we already know it exists, or more carefully: the set of all quotients of coproducts of simplices $\bigsqcup_\sigma |\Delta^{n_\sigma}|$ satisfying the $n$-skeleta conditions. Since $X$ has only a set of simplices, this set is small.

*Alternatively*, one constructs $|X|$ directly as the coend:
$$|X| = \int^{[n] \in \Delta} X_n \cdot |\Delta^n| = \left(\bigsqcup_{n \geq 0} X_n \times |\Delta^n|\right) \big/ \sim$$
where $(x, \phi^*(t)) \sim (\phi_*(x), t)$ for $\phi \colon [m] \to [n]$ order-preserving. This formula shows $|-|$ is well-defined and one verifies the adjunction bijection $\mathbf{sSet}(X, \mathrm{Sing}(Y)) \cong \mathbf{Top}(|X|, Y)$ by the universal property of the coend.

> [!WARNING] Conditions on Top
> For the SAFT to apply directly to $\mathbf{Top}$ (as opposed to a nice subcategory), one needs to verify that $\mathbf{Top}$ is locally small (yes), complete (yes, limits are concrete limits of sets with appropriate topology), well-powered (yes, open subsets form a set), and has a coseparating set (yes, $\{[0,1]\}$ is a coseparating set for $T_0$ spaces by Urysohn, and working with all topological spaces one can use $\{D\}$ for $D$ the Sierpinski space for the appropriate version). These conditions are all satisfied.

> [!NOTE] Exercise 15
> Let $\Delta$ be the simplex category. Consider the functor $\mathrm{Sing} \colon \mathbf{Top}^{op} \to [\Delta^{op}, \mathbf{Set}]$ sending $X$ to its singular complex $\mathrm{Sing}(X)_n = \mathbf{Top}(|\Delta^n|, X)$. Using the Special Adjoint Functor Theorem, show that $\mathrm{Sing}$ has a left adjoint (geometric realization $|-|$). State explicitly which conditions on $\mathbf{Top}$ you need to verify and why each holds. What is the formula for $|X|$?

---

## 17. Comonads from Adjunctions ⬇️

**Definition (Comonad).** A *comonad* on $\mathcal{C}$ is a triple $(G, \varepsilon, \delta)$ where $G \colon \mathcal{C} \to \mathcal{C}$ is an endofunctor, $\varepsilon \colon G \Rightarrow \mathrm{id}_{\mathcal{C}}$ is the *counit*, and $\delta \colon G \Rightarrow G^2$ is the *comultiplication*, satisfying coassociativity $\delta G \circ \delta = G\delta \circ \delta$ and counit laws $\varepsilon G \circ \delta = \mathrm{id}_G = G\varepsilon \circ \delta$.

Dually, a *$G$-coalgebra* is a pair $(A, \alpha \colon A \to GA)$ satisfying $\varepsilon_A \circ \alpha = \mathrm{id}_A$ and $\delta_A \circ \alpha = G\alpha \circ \alpha$.

**Theorem.** Let $(T, \eta, \mu)$ be a monad on $\mathcal{C}$ and suppose $T$ has a right adjoint $G \colon \mathcal{C} \to \mathcal{C}$ (i.e., $T \dashv G$). Then:
1. $G$ carries a canonical comonad structure $(G, \varepsilon^G, \delta^G)$.
2. The category of $G$-coalgebras is isomorphic to the category of $T$-algebras $\mathcal{C}^T$.

*Proof sketch.* Let $\phi_{A,B} \colon \mathcal{C}(TA, B) \xrightarrow{\sim} \mathcal{C}(A, GB)$ be the adjunction bijection for $T \dashv G$.

**Counit of $G$:** Define $\varepsilon^G_A \colon GA \to A$ by $\varepsilon^G_A = \phi^{-1}_{A,A}(\mathrm{id}_{GA})$...

*More precisely:* The comonad structure on $G$ is obtained by dualizing the monad structure on $T$ via the adjunction. Specifically:
- The counit $\varepsilon^G \colon G \Rightarrow \mathrm{id}$ corresponds under $\phi$ to the unit $\eta \colon \mathrm{id} \Rightarrow T$ (via the bijection $\mathcal{C}(T \cdot \mathrm{id}, \mathrm{id}) \cong \mathcal{C}(\mathrm{id}, G \cdot \mathrm{id})$).
- The comultiplication $\delta^G \colon G \Rightarrow G^2$ corresponds to the multiplication $\mu \colon T^2 \Rightarrow T$ (via $\mathcal{C}(T^2, \mathrm{id}) \cong \mathcal{C}(\mathrm{id}, G^2)$).

**Isomorphism of algebra categories:** Given a $T$-algebra $(A, a \colon TA \to A)$, define a $G$-coalgebra structure $\bar{a} \colon A \to GA$ by $\bar{a} = \phi_{A,A}(a)$. The unit law $a \circ \eta_A = \mathrm{id}_A$ translates to the coalgebra counit law $\varepsilon^G_A \circ \bar{a} = \mathrm{id}_A$, and the multiplication law translates to the comultiplication law. This correspondence is functorial and gives an isomorphism $\mathcal{C}^T \cong \mathcal{C}_G$ (Eilenberg-Moore algebras for $T$ vs. coalgebras for $G$).

> [!EXAMPLE] The continuation monad and its comonad
> In Haskell, the continuation monad $T(A) = (A \to R) \to R$ has a right adjoint $G(A) = A \times R$ (for a fixed type $R$). The comonad structure on $G$ is the *store comonad*: $G(A) = A \times R$ with counit $(a, r) \mapsto a$ and comultiplication $(a, r) \mapsto ((a, r), r)$.

> [!NOTE] Exercise 16
> Let $(T, \eta, \mu)$ be a monad on $\mathcal{C}$ and suppose $T$ has a right adjoint $G \colon \mathcal{C} \to \mathcal{C}$ (so $T \dashv G$). Show that $G$ canonically carries the structure of a comonad $(G, \varepsilon^G, \delta^G)$, and that the category of $G$-coalgebras is isomorphic to the category of $T$-algebras $\mathcal{C}^T$. (Define $\varepsilon^G$ and $\delta^G$ explicitly using the adjunction bijection, and verify the comonad axioms.)

---

## 18. Monadic Categories over Set and Top 📋

We apply Beck's theorem to determine monadicity for various concrete categories.

| Category $\mathcal{D}$ | Forgetful functor $U$ | Has left adjoint? | Reflects isomorphisms? | Coequalizers of $U$-split pairs? | Monadic over? |
|---|---|---|---|---|---|
| $\mathbf{Grp}$ (groups) | $U \colon \mathbf{Grp} \to \mathbf{Set}$ | Yes (free groups) | Yes | Yes | $\mathbf{Set}$ ✓ |
| $\mathbf{Ab}$ (abelian groups) | $U \colon \mathbf{Ab} \to \mathbf{Set}$ | Yes (free abelian) | Yes | Yes | $\mathbf{Set}$ ✓ |
| $\mathbf{Lat}$ (lattices) | $U \colon \mathbf{Lat} \to \mathbf{Set}$ | Yes (free lattice) | Yes | Yes | $\mathbf{Set}$ ✓ |
| $\mathbf{Field}$ (fields) | $U \colon \mathbf{Field} \to \mathbf{Set}$ | **No** | — | — | Not monadic |
| $\mathbf{Pos}$ (posets) | $U \colon \mathbf{Pos} \to \mathbf{Set}$ | Yes (discrete preorder) | **No** | — | Not monadic |
| $\mathbf{TOrd}$ (totally ordered sets) | $U \colon \mathbf{TOrd} \to \mathbf{Set}$ | Yes | **No** | — | Not monadic |
| $\mathbf{TopGp}$ (topological groups) | $U \colon \mathbf{TopGp} \to \mathbf{Top}$ | No in general | — | — | Not monadic over $\mathbf{Top}$ |

**Key justifications:**

**Fields are not monadic over $\mathbf{Set}$:** The forgetful functor $U \colon \mathbf{Field} \to \mathbf{Set}$ has no left adjoint — there is no free field on a set (the "free field on $\{x, y\}$" would need to include $x/y$ but $y$ might be zero; this leads to contradiction). Formally: in any monad on $\mathbf{Set}$, the algebra over the empty set is nonempty (it must contain the unit), but there is no field with 0 elements and the field with 1 element does not exist (fields require $0 \neq 1$). So the initial algebra, which would have to be the "field on 0 generators", does not exist.

> [!DANGER] Common misconception about posets
> One might think $U \colon \mathbf{Pos} \to \mathbf{Set}$ is monadic because it has a left adjoint (the discrete poset functor — giving every set the discrete partial order where $x \leq y$ iff $x = y$). But $U$ does not reflect isomorphisms: a bijective order-preserving map between posets need not have an order-preserving inverse. (E.g., the identity function from the discrete poset $\{0,1\}$ to $\{0 \leq 1\}$ is bijective and order-preserving, but its inverse is not.) Hence Beck's theorem fails at condition (iv), and the category of posets is not monadic over sets.

**Totally ordered sets are not monadic over $\mathbf{Set}$:** The same failure of reflection of isomorphisms applies — a bijective order-preserving map between total orders need not be an isomorphism (think of the map $\{0 < 1 < 2\} \to \{0 < 1 < 2\}$ given by $0 \mapsto 1, 1 \mapsto 2, 2 \mapsto 0$; this is a bijection but not order-preserving in that direction... actually, let us note that any bijection that is order-preserving between total orders IS an isomorphism — a bijective order-preserving map between finite total orders is an isomorphism. For infinite total orders: $\mathbb{Z} \to \mathbb{N}$ via a bijection can be order-preserving with non-order-preserving inverse. The key issue is that there is no free totally ordered set on more than 1 element, since all orderings on a finite set are a choice and none is canonical.)

*Alternatively:* $\mathbf{TOrd}$ does not have coequalizers of all $U$-split pairs — certain coequalizer constructions are not totally ordered.

**Topological groups are not monadic over $\mathbf{Top}$:** The forgetful functor $U \colon \mathbf{TopGp} \to \mathbf{Top}$ does not have a left adjoint in general (there is no free topological group on a topological space with the expected universal property in the category of all topological groups, without further conditions on the space).

> [!NOTE] Exercise 17
> For each of the following categories, determine whether it is monadic over $\mathbf{Set}$ (or over $\mathbf{Top}$ where appropriate), and briefly justify using Beck's theorem (or the failure of one of its conditions):
>
> (i) $\mathbf{Field}$ (fields and ring homomorphisms) over $\mathbf{Set}$.
>
> (ii) $\mathbf{Pos}$ (posets and monotone maps) over $\mathbf{Set}$.
>
> (iii) $\mathbf{TOrd}$ (totally ordered sets and order-preserving maps) over $\mathbf{Set}$.
>
> (iv) $\mathbf{Lat}$ (lattices and lattice homomorphisms) over $\mathbf{Set}$.
>
> (v) $\mathbf{TopGp}$ (topological groups and continuous group homomorphisms) over $\mathbf{Top}$.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Tom Leinster, Part III Category Theory (2000) | Original Cambridge course whose Sheet 4 exercises this note covers; includes sections on SAFT and monadicity | [Course page](https://webhomes.maths.ed.ac.uk/~tl/categories/index.html) |
| Leinster, Section E2: SAFT | 7-page examinable section on the Special Adjoint Functor Theorem | [saft.ps](https://webhomes.maths.ed.ac.uk/~tl/categories/saft.ps) |
| Leinster, Section F4: Monadicity | 10-page examinable section on the Beck Monadicity Theorem | [monadicity.ps](https://webhomes.maths.ed.ac.uk/~tl/categories/monadicity.ps) |
| Mac Lane, *Categories for the Working Mathematician* | The standard reference; Chapter VI covers monads, Chapter V covers limits and adjoints | [Springer](https://link.springer.com/book/10.1007/978-1-4612-9839-7) |
| Riehl, *Category Theory in Context* | Modern treatment; Chapter 5 covers adjoint functor theorems and Chapter 5.5 Beck's theorem | [Free PDF](https://math.jhu.edu/~eriehl/context.pdf) |
| Leinster, *Basic Category Theory* (arXiv:1612.09375) | Short introductory text; Chapter 4 (adjoint functors) and Chapter 5 (limits) are relevant | [arXiv](https://arxiv.org/abs/1612.09375) |
| Barr & Wells, *Toposes, Triples and Theories* | Classic monograph on monads (triples); Chapters 3–4 give full proofs of Beck's theorem and Eilenberg-Moore constructions | [Free PDF](http://www.tac.mta.ca/tac/reprints/articles/12/tr12.pdf) |
| Beck, untitled manuscript (1966–1968) | Original source of the monadicity theorem, republished in Battelle 1968 proceedings | [Scan at TAC Reprints](http://www.tac.mta.ca/tac/reprints/articles/12/tr12.pdf) |
| Wikipedia, Beck's Monadicity Theorem | Accessible overview with statement of crude and precise monadicity | [Wikipedia](https://en.wikipedia.org/wiki/Beck%27s_monadicity_theorem) |
| Struck, *An Alternative Approach to the Monadicity Theorem* (REU 2022) | Concise proof of Beck's theorem with clear exposition of split coequalizers | [UChicago REU](http://math.uchicago.edu/~may/REU2022/REUPapers/Struck.pdf) |
| Milewski, *Freyd's Adjoint Functor Theorem* (2020) | Accessible blog exposition of the GAFT proof via comma categories | [Blog post](https://bartoszmilewski.com/2020/07/22/freyds-adjoint-functor-theorem/) |
