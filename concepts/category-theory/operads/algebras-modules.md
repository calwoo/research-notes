# Algebras and Modules over Operads

## Table of Contents

- [[#1. Algebras over an Operad|1. Algebras over an Operad]]
  - [[#1.1 The Structure Maps|1.1 The Structure Maps]]
  - [[#1.2 Compatibility Axioms|1.2 Compatibility Axioms]]
  - [[#1.3 Algebras as Operad Maps|1.3 Algebras as Operad Maps]]
  - [[#1.4 Recovering Classical Examples|1.4 Recovering Classical Examples]]
  - [[#1.5 The Category of O-Algebras|1.5 The Category of O-Algebras]]
- [[#2. Free Algebras|2. Free Algebras]]
  - [[#2.1 Construction and Universal Property|2.1 Construction and Universal Property]]
  - [[#2.2 Classical Cases|2.2 Classical Cases]]
  - [[#2.3 The Operad Monad|2.3 The Operad Monad]]
- [[#3. Left and Right Modules|3. Left and Right Modules]]
  - [[#3.1 Left Modules|3.1 Left Modules]]
  - [[#3.2 Right Modules|3.2 Right Modules]]
  - [[#3.3 Bimodules|3.3 Bimodules]]
  - [[#3.4 Classical Recovery for Ass|3.4 Classical Recovery for Ass]]
- [[#4. The Enveloping Algebra|4. The Enveloping Algebra]]
  - [[#4.1 Modules over an O-Algebra|4.1 Modules over an O-Algebra]]
  - [[#4.2 Construction of the Enveloping Algebra|4.2 Construction of the Enveloping Algebra]]
  - [[#4.3 Classical Cases|4.3 Classical Cases]]
- [[#5. Derivations|5. Derivations]]
  - [[#5.1 Definition and the Leibniz Rule|5.1 Definition and the Leibniz Rule]]
  - [[#5.2 The Module of Derivations|5.2 The Module of Derivations]]
  - [[#5.3 Operad Derivations and Algebra Derivations|5.3 Operad Derivations and Algebra Derivations]]
- [[#6. Kahler Differentials|6. Kahler Differentials]]
  - [[#6.1 The Universal Property|6.1 The Universal Property]]
  - [[#6.2 Construction via Generators and Relations|6.2 Construction via Generators and Relations]]
  - [[#6.3 Recovery for Com|6.3 Recovery for Com]]
- [[#7. A-infinity Algebras|7. A-infinity Algebras]]
  - [[#7.1 The Stasheff Relations|7.1 The Stasheff Relations]]
  - [[#7.2 Low-Degree Consequences|7.2 Low-Degree Consequences]]
  - [[#7.3 Infinity-Morphisms|7.3 Infinity-Morphisms]]
  - [[#7.4 Operadic Perspective|7.4 Operadic Perspective]]
- [[#8. The Probability Operad Revisited|8. The Probability Operad Revisited]]
  - [[#8.1 P-Algebra Structure on the Positive Reals|8.1 P-Algebra Structure on the Positive Reals]]
  - [[#8.2 Shannon Entropy as a Derivation|8.2 Shannon Entropy as a Derivation]]
  - [[#8.3 Connection to BFL Entropy|8.3 Connection to BFL Entropy]]
- [[#References|References]]

---

## 1. Algebras over an Operad

📐 Throughout, $k$ is a commutative unital ring, $\mathbf{Mod}_k$ is the category of $k$-modules, and $\mathcal{O}$ is an operad in $(\mathbf{SymSeq}, \circ, \mathbf{1})$ as developed in [[concepts/category-theory/operads/definitions|Definitions and Examples]]. We recall that $\mathcal{O}$ is equipped with a composition morphism $\gamma: \mathcal{O} \circ \mathcal{O} \to \mathcal{O}$ and a unit $\eta: \mathbf{1} \to \mathcal{O}$, satisfying the monoid axioms in $(\mathbf{SymSeq}, \circ, \mathbf{1})$.

### 1.1 The Structure Maps

**Definition 1.1 (Algebra over an Operad).** An *algebra* over $\mathcal{O}$, or an *$\mathcal{O}$-algebra*, is a $k$-module $A$ together with a collection of $k$-linear *structure maps*
$$\gamma_A(n) \colon \mathcal{O}(n) \otimes_k A^{\otimes n} \longrightarrow A, \qquad n \geq 0,$$
satisfying $S_n$-equivariance and the two compatibility axioms stated below.

The $S_n$-equivariance condition means: for every $\mu \in \mathcal{O}(n)$, every permutation $\sigma \in S_n$, and every tuple $(a_1, \ldots, a_n) \in A^n$,
$$\gamma_A(n)(\mu \cdot \sigma;\, a_1, \ldots, a_n) = \gamma_A(n)(\mu;\, a_{\sigma(1)}, \ldots, a_{\sigma(n)}).$$
Here $\mu \cdot \sigma$ is the right $S_n$-action on $\mathcal{O}(n)$.

> [!NOTE] Notation
> We write $\gamma_A(\mu; a_1, \ldots, a_n)$ for $\gamma_A(n)(\mu \otimes a_1 \otimes \cdots \otimes a_n)$, suppressing the arity when clear from context. This notation emphasizes that $\mu$ is an $n$-ary operation acting on the inputs $a_1, \ldots, a_n$.

### 1.2 Compatibility Axioms

Let $\mu \in \mathcal{O}(n)$ and $\nu_1 \in \mathcal{O}(k_1), \ldots, \nu_n \in \mathcal{O}(k_n)$, so that $\gamma(\mu; \nu_1, \ldots, \nu_n) \in \mathcal{O}(k_1 + \cdots + k_n)$ is the composite operation.

**Axiom A (Associativity).** For all arities and all elements as above:
$$\gamma_A\!\left(\gamma(\mu;\nu_1,\ldots,\nu_n);\; a_{1,1},\ldots,a_{n,k_n}\right) = \gamma_A\!\left(\mu;\; \gamma_A(\nu_1; a_{1,1},\ldots,a_{1,k_1}),\ldots,\gamma_A(\nu_n; a_{n,1},\ldots,a_{n,k_n})\right).$$

**Axiom B (Unitality).** Writing $\mathrm{id} = \eta(1_k) \in \mathcal{O}(1)$ for the image of the unit map:
$$\gamma_A(\mathrm{id};\, a) = a \quad \text{for all } a \in A.$$

These axioms are most cleanly expressed as commutative diagrams. Let $\overline{\mathcal{O}}(A)$ denote the symmetric sequence $\{n \mapsto \mathcal{O}(n) \otimes A^{\otimes n}\}$, so that $\gamma_A$ is collectively a map of the Schur functor evaluation $\mathcal{O}(A) \to A$.

The associativity axiom is the outer square of:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=4em, row sep=3em]
(\mathcal{O} \circ \mathcal{O})(A)
  \arrow[r, "\gamma \circ \mathrm{id}_A"]
  \arrow[d, "\mathrm{id}_{\mathcal{O}} \circ \gamma_A"']
& \mathcal{O}(A)
  \arrow[d, "\gamma_A"] \\
\mathcal{O}(A)
  \arrow[r, "\gamma_A"']
& A
\end{tikzcd}
\end{document}
```

The unitality axiom is:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=4em, row sep=3em]
\mathbf{1}(A)
  \arrow[r, "\eta \circ \mathrm{id}_A"]
  \arrow[dr, "\sim"']
& \mathcal{O}(A)
  \arrow[d, "\gamma_A"] \\
& A
\end{tikzcd}
\end{document}
```

Here $\mathbf{1}(A) = A$ canonically, and $\sim$ is this canonical isomorphism.

### 1.3 Algebras as Operad Maps

🔑 There is a fundamental reformulation that makes the structure entirely explicit.

**Proposition 1.2.** The data of an $\mathcal{O}$-algebra structure on $A$ is equivalent to an operad morphism
$$\rho \colon \mathcal{O} \longrightarrow \mathrm{End}_A,$$
where $\mathrm{End}_A$ is the *endomorphism operad* of $A$, defined by $\mathrm{End}_A(n) := \mathrm{Hom}_k(A^{\otimes n}, A)$ with $S_n$ acting by precomposition with permutations of tensor factors.

*Proof sketch.* Given structure maps $\gamma_A(n)$, define $\rho_n \colon \mathcal{O}(n) \to \mathrm{Hom}_k(A^{\otimes n}, A)$ by
$$\rho_n(\mu)(a_1 \otimes \cdots \otimes a_n) := \gamma_A(\mu; a_1, \ldots, a_n).$$
Equivariance of $\gamma_A$ is equivalent to $S_n$-equivariance of $\rho_n$. Axiom A translates to $\rho$ commuting with the operad compositions $\gamma$ and $\gamma_{\mathrm{End}_A}$. Axiom B translates to $\rho$ sending $\mathrm{id} \in \mathcal{O}(1)$ to $\mathrm{id}_A \in \mathrm{End}_A(1)$. The correspondence is bijective. $\square$

> [!TIP] Why this perspective matters
> The operad map formulation $\rho: \mathcal{O} \to \mathrm{End}_A$ exposes the functoriality: a morphism of operads $f: \mathcal{O} \to \mathcal{O}'$ induces a functor $f^*: \mathcal{O}'\text{-}\mathbf{Alg} \to \mathcal{O}\text{-}\mathbf{Alg}$ by precomposition. This is the mechanism by which a map $\mathrm{Com} \to \mathrm{Ass}$ makes every associative algebra into a commutative algebra after forgetting the commutativity.

### 1.4 Recovering Classical Examples

**Ass-algebras.** The associative operad has $\mathrm{Ass}(n) = k[S_n]$ (the free $k[S_n]$-module of rank 1). The equivariance condition and composition axioms force $\gamma_A$ to give exactly one $n$-ary operation for each ordering of inputs, subject only to associativity. Concretely: the generator $\mathrm{id}_{S_n} \in \mathrm{Ass}(n)$ acts as the $n$-fold iterated product $a_1 \cdots a_n$. An $\mathrm{Ass}$-algebra is therefore precisely a unital *associative algebra*.

**Com-algebras.** Since $\mathrm{Com}(n) = k$ with trivial $S_n$-action, equivariance forces $\gamma_A(\mu; a_1, \ldots, a_n) = \gamma_A(\mu; a_{\sigma(1)}, \ldots, a_{\sigma(n)})$ for all $\sigma$. Combined with associativity, this gives exactly a *commutative associative algebra*.

**Lie-algebras.** The Lie operad encodes the antisymmetry $[a,b] = -[b,a]$ and Jacobi identity; a $\mathrm{Lie}$-algebra is precisely a *Lie algebra* in the classical sense.

> [!EXAMPLE] A Lie algebra from an associative algebra
> If $A$ is an $\mathrm{Ass}$-algebra, the commutator bracket $[a,b] := ab - ba$ makes $A$ into a $\mathrm{Lie}$-algebra. This corresponds to the operad morphism $\mathrm{Lie} \to \mathrm{Ass}$ (via the antisymmetrizer map), which induces the forgetful functor $\mathrm{Ass}\text{-}\mathbf{Alg} \to \mathrm{Lie}\text{-}\mathbf{Alg}$.

### 1.5 The Category of O-Algebras

**Definition 1.3 (Morphism of $\mathcal{O}$-Algebras).** A *morphism* $f \colon (A, \gamma_A) \to (B, \gamma_B)$ of $\mathcal{O}$-algebras is a $k$-linear map $f \colon A \to B$ that is compatible with all structure maps:
$$f\!\left(\gamma_A(\mu; a_1, \ldots, a_n)\right) = \gamma_B\!\left(\mu; f(a_1), \ldots, f(a_n)\right)$$
for all $n \geq 0$, $\mu \in \mathcal{O}(n)$, $a_i \in A$.

The resulting category $\mathcal{O}\text{-}\mathbf{Alg}$ has $\mathcal{O}$-algebras as objects and such morphisms as morphisms. It is complete and cocomplete (limits computed in $\mathbf{Mod}_k$, colimits via reflexive coequalizers). When $\mathcal{O} = \mathrm{Com}$, this recovers the category of commutative $k$-algebras; when $\mathcal{O} = \mathrm{Lie}$, it recovers the category of Lie algebras.

> [!QUESTION] Exercise 1: Limits in O-Alg
> *This exercise establishes that limits in $\mathcal{O}\text{-}\mathbf{Alg}$ are computed in $\mathbf{Mod}_k$.*
>
> > **Prerequisites:** [[#1.2 Compatibility Axioms|1.2 Compatibility Axioms]]
>
> Let $\{A_i\}_{i \in I}$ be a diagram of $\mathcal{O}$-algebras. Show that the limit $\varprojlim A_i$ in $\mathbf{Mod}_k$, equipped with structure maps induced by the projections, forms an $\mathcal{O}$-algebra, and that this is the limit in $\mathcal{O}\text{-}\mathbf{Alg}$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** The structure maps $\gamma_A(n): \mathcal{O}(n) \otimes A^{\otimes n} \to A$ are natural in $A$; a cone of algebra morphisms is precisely a cone of $k$-module maps compatible with all $\gamma$-maps.
>
> **Sketch:** Given projections $\pi_i: L \to A_i$ (the limit in $\mathbf{Mod}_k$), define $\gamma_L(\mu; \ell_1, \ldots, \ell_n)$ to be the unique element of $L$ with $\pi_i$-coordinate $\gamma_{A_i}(\mu; \pi_i(\ell_1), \ldots, \pi_i(\ell_n))$ for each $i$. Equivariance and the axioms follow componentwise. The universal property of $L$ as a limit in $\mathbf{Mod}_k$ implies $L$ is the limit in $\mathcal{O}\text{-}\mathbf{Alg}$.

> [!QUESTION] Exercise 2: The Forgetful Functor is Faithful
> *This problem shows that morphisms of $\mathcal{O}$-algebras are entirely determined by the underlying $k$-linear maps.*
>
> > **Prerequisites:** [[#1.5 The Category of O-Algebras|1.5 The Category of O-Algebras]]
>
> Show that the forgetful functor $U: \mathcal{O}\text{-}\mathbf{Alg} \to \mathbf{Mod}_k$ is faithful. Is it full? Provide a counterexample if not.

> [!TIP]- Solution to Exercise 2
> **Key insight:** Faithfulness is immediate; fullness fails because not every $k$-linear map respects the algebra structure.
>
> **Sketch:** Faithfulness: if $f, g: A \to B$ are $\mathcal{O}$-algebra maps with $U(f) = U(g)$, then $f = g$ as $k$-linear maps, hence as $\mathcal{O}$-algebra maps. Not full: take $\mathcal{O} = \mathrm{Com}$ and $A = k[x]$, $B = k$. The projection $p: k[x] \to k$ sending $x \mapsto 0$ is a $k$-linear map, but so is the evaluation $\mathrm{ev}_1: p(x) \mapsto 1$. The latter is a ring map (hence Com-algebra map) only when it respects multiplication; the map $f(x) = x + 1$ does not respect $f(x^2) = f(x)^2$ unless it's specifically an algebra morphism. More concretely, the $k$-linear map $x \mapsto 1$ is not a $k$-algebra map $k[x] \to k$ since $f(x^2) = 1 \neq f(x)^2 = 1$ (this particular case works), so take $f(x) = 2$ instead: then $f(x^2) = 4 \neq f(x)^2 = 4$; use $\mathcal{O} = \mathrm{Ass}$ and the map $f: k \oplus k \to k$ given by $(a,b) \mapsto a - b$ which is $k$-linear but not a ring map since $f((1,0)(0,1)) = f(0,0) = 0 \neq f(1,0)f(0,1) = 1 \cdot (-1) = -1$.

---

## 2. Free Algebras

💡 The category $\mathcal{O}\text{-}\mathbf{Alg}$ has enough structure to guarantee the existence of free objects, constructed explicitly via *Schur functors* (see [[concepts/category-theory/operads/definitions|Definitions and Examples]], Section 1.3).

### 2.1 Construction and Universal Property

**Definition 2.1 (Free $\mathcal{O}$-Algebra).** Given a $k$-module $V$, the *free $\mathcal{O}$-algebra* on $V$ is
$$\mathcal{O}(V) := \bigoplus_{n \geq 0} \mathcal{O}(n) \otimes_{S_n} V^{\otimes n},$$
equipped with the structure maps
$$\gamma_{\mathcal{O}(V)}(\mu;\, [\nu_1 \otimes v_1],\, \ldots,\, [\nu_k \otimes v_k]) := [\gamma(\mu;\nu_1,\ldots,\nu_k) \otimes v_1 \otimes \cdots \otimes v_k]$$
where $[\cdot]$ denotes the $S_n$-coinvariant class and $v_i \in V^{\otimes k_i}$.

> [!WARNING] Notation clash
> We write $\mathcal{O}(V)$ for the free algebra — the same symbol as the Schur functor evaluation. These coincide: the Schur functor of $\mathcal{O}$, evaluated at $V$, is precisely the underlying $k$-module of the free $\mathcal{O}$-algebra on $V$. The $\mathcal{O}$-algebra structure is the extra data.

**Theorem 2.2 (Adjunction).** The assignment $V \mapsto \mathcal{O}(V)$ extends to a functor $\mathcal{O}(-): \mathbf{Mod}_k \to \mathcal{O}\text{-}\mathbf{Alg}$ which is left adjoint to the forgetful functor $U: \mathcal{O}\text{-}\mathbf{Alg} \to \mathbf{Mod}_k$:
$$\mathrm{Hom}_{\mathcal{O}\text{-}\mathbf{Alg}}(\mathcal{O}(V), A) \cong \mathrm{Hom}_{\mathbf{Mod}_k}(V, U(A)),$$
natural in $V$ and $A$.

*Proof sketch.* The unit of the adjunction is the inclusion $\iota_V: V \hookrightarrow \mathcal{O}(V)$ into the $n=1$ summand: $V = \mathcal{O}(1) \otimes_{S_1} V \subseteq \mathcal{O}(V)$ (since $\mathcal{O}(1) \ni \mathrm{id}$ acts as identity). Given $f: V \to U(A)$, the extension $\tilde{f}: \mathcal{O}(V) \to A$ is defined on generators by $\tilde{f}[\mu \otimes v_1 \otimes \cdots \otimes v_n] := \gamma_A(\mu; f(v_1), \ldots, f(v_n))$. Equivariance and the axioms for $\gamma_A$ ensure $\tilde{f}$ is a well-defined $\mathcal{O}$-algebra morphism, and it is the unique such extension. $\square$

### 2.2 Classical Cases

**Com.** Since $\mathrm{Com}(n) = k$ with trivial $S_n$-action, $\mathrm{Com}(n) \otimes_{S_n} V^{\otimes n} = k \otimes_{S_n} V^{\otimes n} \cong V^{\otimes n}/S_n = \mathrm{Sym}^n(V)$. Therefore:
$$\mathrm{Com}(V) = \bigoplus_{n \geq 0} \mathrm{Sym}^n(V) = \mathrm{Sym}(V),$$
the *symmetric algebra* on $V$. **This recovers the fact that the free commutative algebra is the symmetric algebra.**

**Ass.** Since $\mathrm{Ass}(n) = k[S_n]$ with the regular right $S_n$-action, $k[S_n] \otimes_{S_n} V^{\otimes n} \cong V^{\otimes n}$. Therefore:
$$\mathrm{Ass}(V) = \bigoplus_{n \geq 0} V^{\otimes n} = T(V),$$
the *tensor algebra* on $V$. **This recovers the fact that the free associative algebra is the tensor algebra.**

**Lie.** The Lie operad is more subtle: $\mathrm{Lie}(n)$ is the multilinear part of the free Lie algebra on $n$ generators. One has $\mathrm{Lie}(V) \cong \mathrm{FreeLie}(V)$, the free Lie algebra, embedded in $T(V)$ via the Poincaré–Birkhoff–Witt basis.

### 2.3 The Operad Monad

📐 The free-algebra construction assembles into a *monad* on $\mathbf{Mod}_k$.

**Definition 2.3 (Operad Monad).** The *monad associated to $\mathcal{O}$* is the endofunctor $T_\mathcal{O} := U \circ \mathcal{O}(-): \mathbf{Mod}_k \to \mathbf{Mod}_k$, i.e., $T_\mathcal{O}(V) = \mathcal{O}(V)$ as a $k$-module, equipped with:
- *Unit* $\eta_V: V \to T_\mathcal{O}(V)$: the inclusion into the $n=1$ summand.
- *Multiplication* $\mu_V: T_\mathcal{O}(T_\mathcal{O}(V)) \to T_\mathcal{O}(V)$: the extension of $\mathrm{id}: T_\mathcal{O}(V) \to T_\mathcal{O}(V)$ via the adjunction applied with $A = \mathcal{O}(V)$.

Concretely, $\mu_V$ is given by "substituting trees into trees": an element of $\mathcal{O}(\mathcal{O}(V))$ is a tree of operations with leaves in $\mathcal{O}(V)$, and $\mu_V$ composes them using the operad structure $\gamma$.

**Proposition 2.4.** The category of Eilenberg-Moore algebras for the monad $T_\mathcal{O}$ is isomorphic to $\mathcal{O}\text{-}\mathbf{Alg}$.

*Proof sketch.* An Eilenberg-Moore algebra is a $k$-module $A$ with a map $h: T_\mathcal{O}(A) \to A$ satisfying $h \circ \eta_A = \mathrm{id}_A$ and $h \circ \mu_A = h \circ T_\mathcal{O}(h)$. Setting $\gamma_A(n) := h|_{\mathcal{O}(n) \otimes_{S_n} A^{\otimes n}}$ recovers Definition 1.1. The axioms match precisely. $\square$

> [!EXAMPLE] The monad for Com
> $T_{\mathrm{Com}}(V) = \mathrm{Sym}(V)$. The monad multiplication $\mu: \mathrm{Sym}(\mathrm{Sym}(V)) \to \mathrm{Sym}(V)$ is the algebra map extending the inclusion $\mathrm{Sym}(V) \hookrightarrow \mathrm{Sym}(V)$ as the identity. This is just the "flattening" of symmetric polynomials in symmetric polynomials.

> [!QUESTION] Exercise 3: Free Lie Algebra Dimension
> *This exercise computes the dimension of the free Lie algebra on $n$ generators in each degree, using the operadic description.*
>
> > **Prerequisites:** [[#2.2 Classical Cases|2.2 Classical Cases]]
>
> Let $V$ be a free $k$-module of rank $r$ (so $V \cong k^r$). Use the formula $\dim_k \mathrm{Lie}(n) = (n-1)!$ and the coinvariant description to compute $\dim_k \mathrm{Lie}(n) \otimes_{S_n} V^{\otimes n}$ as a function of $r$ and $n$. Recover the Witt formula: $\dim(\mathrm{FreeLie}(V))_n = \frac{1}{n} \sum_{d \mid n} \mu(n/d) r^d$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** The dimension of $\mathrm{Lie}(n) \otimes_{S_n} V^{\otimes n}$ counts $r$-colored $n$-element Lie brackets via Möbius inversion.
>
> **Sketch:** We have $\mathrm{Lie}(n) \otimes_{S_n} V^{\otimes n} = \mathrm{Lie}(n) \otimes_{S_n} (k^r)^{\otimes n}$. As a $k$-module, $(k^r)^{\otimes n}$ has a basis of tuples $(e_{i_1}, \ldots, e_{i_n})$ with $i_j \in \{1,\ldots,r\}$. The coinvariant $\otimes_{S_n}$ identifies $\ell \cdot \sigma \otimes v$ with $\ell \otimes \sigma^{-1} \cdot v$. The dimension equals $\sum_{\lambda \vdash n} \dim(\mathrm{Lie}(n))^{S_\lambda} \cdot r^{\ell(\lambda)}$ by Burnside/Molien theory, which by Möbius inversion on the divisor lattice gives the Witt formula $\frac{1}{n}\sum_{d|n} \mu(n/d) r^d$.

> [!QUESTION] Exercise 4: Adjunction Counit
> *This exercise makes the counit of the free-forgetful adjunction explicit.*
>
> > **Prerequisites:** [[#2.1 Construction and Universal Property|2.1 Construction and Universal Property]]
>
> Describe the counit $\varepsilon: \mathcal{O}(U(A)) \to A$ of the adjunction $\mathcal{O}(-) \dashv U$ explicitly in terms of the structure maps $\gamma_A(n)$.

> [!TIP]- Solution to Exercise 4
> **Key insight:** The counit is the "evaluation map" that applies all pending operations.
>
> **Sketch:** The counit $\varepsilon_A: \mathcal{O}(A) \to A$ corresponds, under the adjunction, to the identity map $\mathrm{id}_{U(A)}: U(A) \to U(A)$. Explicitly, $\varepsilon_A[\mu \otimes a_1 \otimes \cdots \otimes a_n] = \gamma_A(\mu; a_1, \ldots, a_n)$. This is well-defined because $\gamma_A$ is $S_n$-equivariant. The triangle identities for the adjunction follow from Axiom B (unitality of $\gamma_A$).

---

## 3. Left and Right Modules

📐 Having algebras in hand, we next define *modules* — objects on which an operad acts from one side. The natural setting is again symmetric sequences, with actions defined via the composition product $\circ$.

### 3.1 Left Modules

**Definition 3.1 (Left $\mathcal{O}$-Module).** A *left $\mathcal{O}$-module* is a symmetric sequence $M \in \mathbf{SymSeq}$ together with a morphism of symmetric sequences
$$\lambda: \mathcal{O} \circ M \longrightarrow M$$
satisfying:
1. *Associativity:* $\lambda \circ (\gamma \circ \mathrm{id}_M) = \lambda \circ (\mathrm{id}_\mathcal{O} \circ \lambda)$ as maps $\mathcal{O} \circ \mathcal{O} \circ M \to M$.
2. *Unitality:* $\lambda \circ (\eta \circ \mathrm{id}_M) = \mathrm{id}_M$ as maps $\mathbf{1} \circ M \cong M \to M$.

In terms of components, recalling that $(\mathcal{O} \circ M)(n) = \bigoplus_{k \geq 0} \mathcal{O}(k) \otimes_{S_k} \bigoplus_{n_1+\cdots+n_k = n} \mathrm{Ind}_{S_{n_1} \times \cdots \times S_{n_k}}^{S_n}(M(n_1) \otimes \cdots \otimes M(n_k))$, the map $\lambda$ specifies how $\mathcal{O}$-operations compose with operations already in $M$.

The axioms read as the following commutative diagrams:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=4em, row sep=3em]
\mathcal{O} \circ \mathcal{O} \circ M
  \arrow[r, "\gamma \circ \mathrm{id}_M"]
  \arrow[d, "\mathrm{id}_{\mathcal{O}} \circ \lambda"']
& \mathcal{O} \circ M
  \arrow[d, "\lambda"] \\
\mathcal{O} \circ M
  \arrow[r, "\lambda"']
& M
\end{tikzcd}
\end{document}
```

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=4em, row sep=3em]
\mathbf{1} \circ M
  \arrow[r, "\eta \circ \mathrm{id}_M"]
  \arrow[dr, "\sim"']
& \mathcal{O} \circ M
  \arrow[d, "\lambda"] \\
& M
\end{tikzcd}
\end{document}
```

> [!NOTE] Left modules vs. algebras
> Every $\mathcal{O}$-algebra $A$ determines a left $\mathcal{O}$-module: take the *constant* symmetric sequence $\underline{A}$ with $\underline{A}(0) = A$ and $\underline{A}(n) = 0$ for $n \geq 1$. The module map $\lambda: \mathcal{O} \circ \underline{A} \to \underline{A}$ on the $(n=0)$-summand is $\gamma_A(n): \mathcal{O}(n) \otimes A^{\otimes n} \to A$. Thus the notion of left module generalizes that of algebra to symmetric-sequence-valued actions.

### 3.2 Right Modules

**Definition 3.2 (Right $\mathcal{O}$-Module).** A *right $\mathcal{O}$-module* is a symmetric sequence $M$ together with
$$\rho: M \circ \mathcal{O} \longrightarrow M$$
satisfying the mirror axioms:
1. *Associativity:* $\rho \circ (\rho \circ \mathrm{id}_\mathcal{O}) = \rho \circ (\mathrm{id}_M \circ \gamma)$ as maps $M \circ \mathcal{O} \circ \mathcal{O} \to M$.
2. *Unitality:* $\rho \circ (\mathrm{id}_M \circ \eta) = \mathrm{id}_M$ as maps $M \circ \mathbf{1} \cong M \to M$.

> [!WARNING] Asymmetry of the composition product
> The composition product $\circ$ is not symmetric: $\mathcal{O} \circ M \neq M \circ \mathcal{O}$ in general. This is why left and right modules over an operad are genuinely different structures, unlike modules over a commutative ring. The non-symmetry of $\circ$ reflects the non-symmetric way $\mathcal{O}$ composes with itself: the outer operation ranges over all arities while inner operations slot into leaves.

### 3.3 Bimodules

**Definition 3.3 ($\mathcal{O}$-Bimodule).** An *$\mathcal{O}$-bimodule* is a symmetric sequence $M$ equipped with both a left action $\lambda: \mathcal{O} \circ M \to M$ and a right action $\rho: M \circ \mathcal{O} \to M$, satisfying the compatibility condition:
$$\lambda \circ (\mathrm{id}_\mathcal{O} \circ \rho) = \rho \circ (\lambda \circ \mathrm{id}_\mathcal{O})$$
as maps $\mathcal{O} \circ M \circ \mathcal{O} \to M$.

This is the operadic analogue of the left-right compatibility condition $r(am) = a(rm)$ for a classical bimodule.

We denote the categories:
- ${}_\mathcal{O}\mathbf{Mod}$: left $\mathcal{O}$-modules.
- $\mathbf{Mod}_\mathcal{O}$: right $\mathcal{O}$-modules.
- ${}_\mathcal{O}\mathbf{Mod}_\mathcal{O}$: $\mathcal{O}$-bimodules.

### 3.4 Classical Recovery for Ass

🔑 The cleanest verification of these definitions is recovering classical module theory.

**Proposition 3.4.** An $\mathrm{Ass}$-bimodule in $\mathbf{SymSeq}$ concentrated in arity 0 is precisely a classical bimodule over an associative algebra.

*Proof sketch.* A left $\mathrm{Ass}$-module structure on a symmetric sequence $M$ concentrated in arity 0 (i.e., $M(0) = N$, $M(k) = 0$ for $k \geq 1$) reduces, by unfolding $(\mathrm{Ass} \circ M)(0) = \bigoplus_k \mathrm{Ass}(k) \otimes_{S_k} N^{\otimes 0}$ — wait, one must be careful: $M(0) \otimes \cdots$ for $k$ slots is $M(0)^{\otimes 0} = k$ for $k=0$. The action on arity $0$ from $\mathrm{Ass}(k)$ gives maps $A^{\otimes k} \otimes N \to N$ (left multiplication), and the right action from $N \otimes A^{\otimes k} \to N$ (right multiplication). Associativity and unitality of $\lambda$ and $\rho$ are equivalent to the bimodule axioms $(a_1 a_2)n = a_1(a_2 n)$ and $n(a_1 a_2) = (na_1)a_2$ and $1 \cdot n = n \cdot 1 = n$. $\square$

> [!EXAMPLE] The operad O as a bimodule over itself
> The operad $\mathcal{O}$ is naturally an $\mathcal{O}$-bimodule via $\gamma$: left action is $\gamma: \mathcal{O} \circ \mathcal{O} \to \mathcal{O}$ and right action is likewise $\gamma$. The bimodule compatibility is exactly the associativity of $\gamma$.

> [!QUESTION] Exercise 5: Arity-Concentration and Classical Modules
> *This exercise makes the identification of Proposition 3.4 fully explicit at the component level.*
>
> > **Prerequisites:** [[#3.4 Classical Recovery for Ass|3.4 Classical Recovery for Ass]]
>
> Let $A$ be an associative algebra and $N$ a classical $A$-bimodule. Define a symmetric sequence $M$ by $M(0) = N$, $M(k) = 0$ for $k \geq 1$. Write down explicitly the left action map $\lambda(1): \mathrm{Ass}(1) \otimes_{S_1} M(0) \to M(0)$ and the right action map $\rho(1): M(0) \otimes_{S_1} \mathrm{Ass}(1) \to M(0)$ in terms of the $A$-bimodule structure on $N$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** The arity-1 components of the action maps encode the left and right $A$-module multiplications.
>
> **Sketch:** $\mathrm{Ass}(1) = k[S_1] = k \cdot e$, so $\mathrm{Ass}(1) \otimes_{S_1} M(0) \cong N$. The left action $\lambda(1): N \to N$ must equal the identity (by unitality, since $e \in \mathrm{Ass}(1)$ maps to $\mathrm{id} \in \mathrm{End}_A(1)$). For arity $k$: $(\mathrm{Ass} \circ M)(0)$ receives contributions from $\mathrm{Ass}(1) \otimes_{S_1} M(0) \cong N$ at $k=1$ only (since $M(j) = 0$ for $j \geq 1$ forces the only nonzero contribution to $M(0)$ from $k=1$). Thus $\lambda$ on the relevant summand is $a \cdot n$ (left $A$-action). Similarly $\rho$ on the relevant summand is $n \cdot a$ (right $A$-action).

---

## 4. The Enveloping Algebra

📐 For a $k$-algebra $A$, a *module over $A$* in the classical sense is a module over the ring $A$. For an $\mathcal{O}$-algebra, the correct analogue of "module" must be sensitive to the operadic structure; the device encoding this is the *enveloping algebra*.

### 4.1 Modules over an O-Algebra

**Definition 4.1 (Representation / Module).** Let $(A, \gamma_A)$ be an $\mathcal{O}$-algebra. A *representation of $A$* (or *$A$-module* in the $\mathcal{O}$-sense) is a $k$-module $M$ together with structure maps
$$\gamma_M(n, i) \colon \mathcal{O}(n) \otimes A^{\otimes (i-1)} \otimes M \otimes A^{\otimes (n-i)} \longrightarrow M, \quad 1 \leq i \leq n,\; n \geq 1,$$
satisfying:
- $S_n$-equivariance (permuting all slots, with $M$ in position $i$ permuted accordingly to position $\sigma(i)$).
- Compatibility with $\gamma$: the maps $\gamma_M(n, i)$ express "applying $\mu \in \mathcal{O}(n)$ to inputs where all but the $i$-th are in $A$ and the $i$-th is in $M$."
- When all inputs are in $A$, recovery of the algebra action: $\gamma_M(n, i)|_{M=A} = \gamma_A(n)$.

> [!NOTE] Symmetry of the slot
> For $\mathrm{Com}$-algebras, equivariance forces all $\gamma_M(n, i)$ to be equal for different $i$, giving the single commutative module map $A^{\otimes(n-1)} \otimes M \to M$. For $\mathrm{Ass}$-algebras, the $n=2$ slot gives left and right multiplication, recovering $am$ and $ma$.

### 4.2 Construction of the Enveloping Algebra

**Definition 4.2 (Enveloping Algebra).** The *enveloping algebra* of the $\mathcal{O}$-algebra $A$, denoted $U_\mathcal{O}(A)$, is the associative $k$-algebra defined as the quotient:
$$U_\mathcal{O}(A) := T\!\left(\bigoplus_{n \geq 1} \mathcal{O}(n) \otimes_{S_{n-1}} A^{\otimes(n-1)}\right) \Big/ \sim$$
where $T(-)$ denotes the tensor algebra, and $\sim$ is the equivalence relation generated by:
1. $(\mu \cdot \sigma) \otimes a_1 \otimes \cdots \otimes a_{n-1} \sim \mu \otimes a_{\sigma^{-1}(1)} \otimes \cdots \otimes a_{\sigma^{-1}(n-1)}$ for $\sigma \in S_{n-1}$ (embedded in $S_n$ as the stabilizer of position $n$).
2. $\gamma(\mu; \nu_1, \ldots, \nu_n) \otimes \cdots \sim \mu \otimes \cdots \otimes \nu_j \otimes \cdots$ (operadic composition relations).
3. $\mathrm{id} \otimes 1_k \sim 1_{U}$ (unitality).

The *universal property* of $U_\mathcal{O}(A)$ is:
$$\mathrm{Der}_\mathcal{O}(A, M) \cong \mathrm{Hom}_{U_\mathcal{O}(A)\text{-}\mathbf{Mod}}(\Omega^1_\mathcal{O}(A), M),$$
naturally in the $A$-module $M$ (see Section 6 for $\Omega^1_\mathcal{O}(A)$). Equivalently, $U_\mathcal{O}(A)$-modules are in bijection with representations of $A$ in the sense of Definition 4.1.

### 4.3 Classical Cases

**For $\mathrm{Ass}$.** The $n$-th summand in the generating module is $\mathrm{Ass}(n) \otimes_{S_{n-1}} A^{\otimes(n-1)} = k[S_n] \otimes_{S_{n-1}} A^{\otimes(n-1)}$. By Mackey's induction, $k[S_n] \otimes_{S_{n-1}} A^{\otimes(n-1)} \cong k[S_n/S_{n-1}] \otimes A^{\otimes(n-1)} \cong A^{\otimes(n-1)} \oplus A^{\otimes(n-1)}$ (left and right cosets). The quotient by the operadic relations gives $A \otimes A^{\mathrm{op}}$. **Thus $U_{\mathrm{Ass}}(A) = A \otimes_k A^{\mathrm{op}}$, the classical enveloping algebra of an associative algebra.**

**For $\mathrm{Lie}$.** Here $\mathrm{Lie}(n) \otimes_{S_{n-1}} A^{\otimes(n-1)}$ assembles under the PBW theorem to the universal enveloping algebra of the Lie algebra $A$. **Thus $U_{\mathrm{Lie}}(\mathfrak{g}) = U(\mathfrak{g})$, the classical universal enveloping algebra of a Lie algebra.**

> [!EXAMPLE] Enveloping algebra for Com
> For a commutative algebra $A$ (i.e., $\mathrm{Com}$-algebra), $\mathrm{Com}(n) \otimes_{S_{n-1}} A^{\otimes(n-1)} = k \otimes_{S_{n-1}} A^{\otimes(n-1)} = \mathrm{Sym}^{n-1}(A)/(S_{n-1})$... actually since $\mathrm{Com}(n) = k$ with trivial action, $U_{\mathrm{Com}}(A) = A$ itself: the module category over a commutative algebra $A$ in the operadic sense is simply the category of $A$-modules in the classical sense, so no additional algebra structure is needed beyond $A$.

> [!QUESTION] Exercise 6: Enveloping Algebra for Ass
> *This exercise derives the isomorphism $U_{\mathrm{Ass}}(A) \cong A \otimes A^{\mathrm{op}}$ directly from the generators-and-relations construction.*
>
> > **Prerequisites:** [[#4.2 Construction of the Enveloping Algebra|4.2 Construction of the Enveloping Algebra]]
>
> Work through the generators-and-relations construction for $\mathcal{O} = \mathrm{Ass}$. The generating module at arity $n=2$ is $\mathrm{Ass}(2) \otimes_{S_1} A^{\otimes 1} = k[S_2] \otimes_{S_1} A$. Show this is isomorphic to $A \oplus A$ (indexed by left and right multiplication). Then show the operadic composition relations force the algebra structure to be $A \otimes A^{\mathrm{op}}$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** $k[S_2]$ as a right $S_1$-module is free of rank 2, with basis $\{e, (12)\}$.
>
> **Sketch:** $k[S_2] \otimes_{S_1} A \cong (k \cdot e \otimes A) \oplus (k \cdot (12) \otimes A) \cong A \oplus A$. An element $e \otimes a$ corresponds to "left-multiply by $a$" and $(12) \otimes a$ to "right-multiply by $a$." The operadic composition relation for $\gamma(\mu_2; \nu_2, \mathrm{id})$ (composing the product with itself in the first slot) gives $(a_1 a_2) \cdot m = a_1 \cdot (a_2 \cdot m)$ for left multiplication, and $(12) \otimes a_1, (12) \otimes a_2$ compose to give $m \cdot (a_1 a_2) = (m \cdot a_1) \cdot a_2$ for right multiplication. The left and right actions commute: $a \cdot (m \cdot b) = (a \cdot m) \cdot b$. This is exactly the $A \otimes A^{\mathrm{op}}$-module structure via $(a \otimes b) \cdot m = a \cdot m \cdot b$.

> [!QUESTION] Exercise 7: PBW and the Lie Enveloping Algebra
> *This exercise connects the operadic enveloping algebra construction to the classical PBW theorem.*
>
> > **Prerequisites:** [[#4.3 Classical Cases|4.3 Classical Cases]]
>
> Verify that for $\mathcal{O} = \mathrm{Lie}$ and $A = \mathfrak{g}$ a Lie algebra, the relation (2) in the construction of $U_\mathcal{O}(A)$ (operadic composition) implies the relation $[x, y] = xy - yx$ in $U(\mathfrak{g})$. More precisely, show that the Lie bracket $\ell_2 \in \mathrm{Lie}(2)$ and the composition axiom force $\ell_2(x, y) - \ell_2(y, x) = \ell_2(\gamma(\ell_2; x, y), -)$ to recover the Jacobi identity.

> [!TIP]- Solution to Exercise 7
> **Key insight:** The generator $\ell_2 \in \mathrm{Lie}(2)$ is antisymmetric; composing with the composition map recovers the Jacobi identity as an associativity relation in $U(\mathfrak{g})$.
>
> **Sketch:** In $U_{\mathrm{Lie}}(\mathfrak{g})$, the generator corresponding to $\ell_2 \otimes_{S_1} x \in \mathrm{Lie}(2) \otimes_{S_1} \mathfrak{g}$ (with $x$ in position 1, the free slot in position 2) acts as "bracket with $x$ from the left." The antisymmetry of $\ell_2$ (i.e., $\ell_2 \cdot (12) = -\ell_2$ in $\mathrm{Lie}(2)$) gives $[x,y] = -[y,x]$. The operadic composition relation for $\gamma(\ell_2; \ell_2, \mathrm{id})$ reads $[[x,y],z] + \text{cyclic} = 0$ (Jacobi). In $T(\mathfrak{g})$, modding out by $xy - yx - [x,y]$ recovers $U(\mathfrak{g})$.

---

## 5. Derivations

📐 *Derivations* are the fundamental infinitesimal symmetries of algebras; for operadic algebras, the correct notion of derivation is dictated by the multi-linear structure.

### 5.1 Definition and the Leibniz Rule

**Definition 5.1 ($\mathcal{O}$-Derivation).** Let $(A, \gamma_A)$ be an $\mathcal{O}$-algebra and $M$ a representation of $A$. A *derivation* from $A$ to $M$ is a $k$-linear map $d: A \to M$ satisfying the *operadic Leibniz rule*: for all $n \geq 1$, all $\mu \in \mathcal{O}(n)$, and all $a_1, \ldots, a_n \in A$,
$$d\!\left(\gamma_A(\mu;\, a_1,\ldots,a_n)\right) = \sum_{i=1}^{n} \gamma_M\!\left(\mu;\, a_1,\ldots,a_{i-1},\, da_i,\, a_{i+1},\ldots,a_n\right).$$
Here $\gamma_M(\mu; a_1, \ldots, a_{i-1}, m, a_{i+1}, \ldots, a_n)$ uses the representation maps of Definition 4.1 with $m = da_i \in M$ in position $i$.

> [!NOTE] Recovering the classical Leibniz rule
> For $\mathcal{O} = \mathrm{Ass}$, $n=2$, $\mu = $ product, and $M = A$ (with the bimodule structure): the Leibniz rule becomes $d(a_1 a_2) = d(a_1) \cdot a_2 + a_1 \cdot d(a_2)$, exactly the classical derivation condition. For $n=1$, $\mu = \mathrm{id}$: $d(\mathrm{id}(a)) = d(a)$, trivially satisfied.

### 5.2 The Module of Derivations

**Definition 5.2.** The set $\mathrm{Der}_\mathcal{O}(A, M)$ of all $\mathcal{O}$-derivations from $A$ to $M$ is naturally a $k$-module (pointwise addition and scalar multiplication). When $M = A$ with its own $A$-bimodule structure (via the representation maps coming from $\gamma_A$ itself), we write $\mathrm{Der}_\mathcal{O}(A) := \mathrm{Der}_\mathcal{O}(A, A)$.

For fixed $A$, the assignment $M \mapsto \mathrm{Der}_\mathcal{O}(A, M)$ is a functor $A\text{-}\mathbf{Mod} \to \mathbf{Mod}_k$. The universal object representing this functor is the module of Kähler differentials $\Omega^1_\mathcal{O}(A)$ (Section 6).

### 5.3 Operad Derivations and Algebra Derivations

💡 Bradley's paper [[research/entropy-operad-derivation|Entropy as an Operad Derivation]] works with derivations of the operad $\mathcal{P}$ itself (not of a $\mathcal{P}$-algebra). These two notions are related by a natural evaluation map.

**Definition 5.3 (Derivation of an Operad).** A *derivation of the operad $\mathcal{O}$* with values in an $\mathcal{O}$-bimodule $M$ is a collection of $k$-linear maps $\partial_n: \mathcal{O}(n) \to M(n)$ satisfying the operadic Leibniz rule:
$$\partial_{k_1+\cdots+k_n}\!\left(\gamma(\mu;\nu_1,\ldots,\nu_n)\right) = \sum_{i=1}^{n} \gamma_M\!\left(\mu;\nu_1,\ldots,\nu_{i-1},\partial_{k_i}(\nu_i),\nu_{i+1},\ldots,\nu_n\right)$$
where $\gamma_M$ uses the bimodule action.

**Proposition 5.4 (Evaluation Map).** Let $\mathcal{O}$ be an operad, $A$ an $\mathcal{O}$-algebra, and $N$ an $A$-module. There is a natural $k$-linear map (the *evaluation*)
$$\mathrm{ev}_A: \mathrm{Der}(\mathcal{O}, \mathrm{End}_A) \longrightarrow \mathrm{Der}_\mathcal{O}(A, N)$$
defined as follows: given a derivation $\partial: \mathcal{O} \to \mathrm{End}_A$ of the operad (with values in the endomorphism bimodule), form $d: A \to N$ by $d(a) := \sum_n (\partial_n \mathrm{id}_A)(a)$ in degree $n=1$.

*The significance:* this map explains how the operadic derivation of the probability operad $\mathcal{P}$ (Shannon entropy $H$, shown by Bradley to be a derivation $\mathcal{P} \to \mathbb{R}$-valued bimodule) descends to a derivation of $\mathcal{P}$-algebras in the classical sense. See Section 8.

> [!QUESTION] Exercise 8: Leibniz Rule for Com-Algebras
> *This exercise verifies that the operadic derivation condition for $\mathrm{Com}$-algebras reduces to the classical derivation rule.*
>
> > **Prerequisites:** [[#5.1 Definition and the Leibniz Rule|5.1 Definition and the Leibniz Rule]]
>
> Let $\mathcal{O} = \mathrm{Com}$ and $A$ a commutative $k$-algebra. Show that a $\mathrm{Com}$-derivation $d: A \to M$ (where $M$ is an $A$-module) satisfies $d(ab) = a \cdot d(b) + b \cdot d(a)$ — the classical derivation condition. Use equivariance of $\gamma_M$ to show both terms are equal.

> [!TIP]- Solution to Exercise 8
> **Key insight:** For $\mathrm{Com}$, equivariance forces $\gamma_M(n, i) = \gamma_M(n, j)$ for all $i, j$, so the two terms in the Leibniz rule are both $\gamma_M(\mu_2; da, b)$ and $\gamma_M(\mu_2; a, db)$, recovering $a \cdot db + b \cdot da$.
>
> **Sketch:** With $n=2$, $\mu_2 \in \mathrm{Com}(2) = k$ the generator: $d(\gamma_A(\mu_2; a, b)) = \gamma_M(\mu_2; da, b) + \gamma_M(\mu_2; a, db)$. Since $\mathrm{Com}(2)$ has trivial $S_2$-action, $\gamma_M(\mu_2; da, b) = \gamma_M(\mu_2 \cdot (12); da, b) = \gamma_M(\mu_2; b, da)$. Thus both $\gamma_M(\mu_2; da, b) = b \cdot da$ and $\gamma_M(\mu_2; a, db) = a \cdot db$, giving $d(ab) = b \cdot da + a \cdot db$.

> [!QUESTION] Exercise 9: Inner Derivations
> *This exercise identifies the operadic analogue of inner derivations for Lie algebras.*
>
> > **Prerequisites:** [[#5.2 The Module of Derivations|5.2 The Module of Derivations]]
>
> For $\mathcal{O} = \mathrm{Lie}$ and a Lie algebra $A = \mathfrak{g}$, show that for any $x \in \mathfrak{g}$, the map $\mathrm{ad}_x: \mathfrak{g} \to \mathfrak{g}$ defined by $\mathrm{ad}_x(y) = [x, y]$ is a $\mathrm{Lie}$-derivation. Verify the operadic Leibniz rule for $\mu = \ell_2 \in \mathrm{Lie}(2)$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** The adjoint action is the archetypical derivation; the Jacobi identity is precisely the Leibniz rule for $\mathrm{ad}_x$.
>
> **Sketch:** We must check $\mathrm{ad}_x([y,z]) = [\mathrm{ad}_x(y), z] + [y, \mathrm{ad}_x(z)]$, i.e., $[x,[y,z]] = [[x,y],z] + [y,[x,z]]$. This is the Jacobi identity: $[x,[y,z]] + [y,[z,x]] + [z,[x,y]] = 0$, rearranged as $[x,[y,z]] = [[x,y],z] + [y,[x,z]]$ (using antisymmetry $[z,x] = -[x,z]$).

---

## 6. Kahler Differentials

📐 The *module of Kähler differentials* $\Omega^1_\mathcal{O}(A)$ is the universal linearization of $\mathcal{O}$-derivations — it converts derivations into module homomorphisms.

### 6.1 The Universal Property

**Definition 6.1 (Kähler Differentials).** Let $(A, \gamma_A)$ be an $\mathcal{O}$-algebra. The *module of Kähler differentials* $\Omega^1_\mathcal{O}(A)$ is a $U_\mathcal{O}(A)$-module equipped with a universal derivation $d: A \to \Omega^1_\mathcal{O}(A)$, characterized by the natural isomorphism
$$\mathrm{Der}_\mathcal{O}(A, M) \cong \mathrm{Hom}_{U_\mathcal{O}(A)}(\Omega^1_\mathcal{O}(A), M)$$
for every $A$-module (i.e., $U_\mathcal{O}(A)$-module) $M$.

In other words, every $\mathcal{O}$-derivation $\partial: A \to M$ factors uniquely as $A \xrightarrow{d} \Omega^1_\mathcal{O}(A) \xrightarrow{\tilde\partial} M$ where $\tilde\partial$ is a $U_\mathcal{O}(A)$-module map:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=4em, row sep=3em]
A
  \arrow[r, "d"]
  \arrow[dr, "\partial"']
& \Omega^1_{\mathcal{O}}(A)
  \arrow[d, "\exists!\,\tilde{\partial}"] \\
& M
\end{tikzcd}
\end{document}
```

### 6.2 Construction via Generators and Relations

**Proposition 6.2 (Explicit Construction).** The module $\Omega^1_\mathcal{O}(A)$ is the $U_\mathcal{O}(A)$-module
$$\Omega^1_\mathcal{O}(A) = U_\mathcal{O}(A) \otimes_k A \Big/ \langle \text{Leibniz relations} \rangle,$$
where the *Leibniz relations* are: for all $n$, $\mu \in \mathcal{O}(n)$, and $a_1, \ldots, a_n \in A$,
$$d\!\left(\gamma_A(\mu; a_1,\ldots,a_n)\right) \sim \sum_{i=1}^n [\text{element of } U_\mathcal{O}(A) \text{ corresponding to } (\mu; a_1,\ldots,\hat{a}_i,\ldots,a_n)] \cdot d(a_i).$$
The universal derivation is $d: A \to \Omega^1_\mathcal{O}(A)$, $a \mapsto [1 \otimes a]$.

*Proof sketch.* The universal property follows from the Freyd adjoint functor theorem applied to the forgetful functor from pairs $(M, \partial: A \to M)$ to $A$-modules $M$; the explicit construction is the coend formula realizing this left adjoint. The Leibniz relations ensure $d$ is an $\mathcal{O}$-derivation; universality follows from the freeness of $U_\mathcal{O}(A) \otimes A$ as a $U_\mathcal{O}(A)$-module. $\square$

### 6.3 Recovery for Com

**Proposition 6.3.** For $\mathcal{O} = \mathrm{Com}$ and $A$ a commutative $k$-algebra:
$$\Omega^1_{\mathrm{Com}}(A) \cong \Omega^1_{A/k},$$
the *classical module of Kähler differentials* — the $A$-module generated by symbols $da$ for $a \in A$, subject to $d(ab) = a \cdot db + b \cdot da$ and $d(\lambda) = 0$ for $\lambda \in k$.

*Proof sketch.* Since $U_{\mathrm{Com}}(A) = A$ (as noted in Section 4.3), the $U_{\mathrm{Com}}(A)$-module $\Omega^1_{\mathrm{Com}}(A)$ is an $A$-module. The Leibniz relations for $\mathrm{Com}$ at $n=2$ are exactly $d(ab) = a \cdot db + b \cdot da$ (using the commutativity of $A$ to identify both $\gamma_M(2,1)$ and $\gamma_M(2,2)$). The generators $\{da : a \in A\}$ and the relations match the classical definition. $\square$

> [!EXAMPLE] Kahler differentials for the polynomial algebra
> Let $A = k[x_1, \ldots, x_n]$. Then $\Omega^1_{A/k}$ is the free $A$-module generated by $dx_1, \ldots, dx_n$, with $d(f) = \sum_i \frac{\partial f}{\partial x_i} dx_i$. This is the module of algebraic 1-forms on affine $n$-space — the algebraic analogue of the cotangent bundle.

> [!QUESTION] Exercise 10: Kahler Differentials for Ass
> *This exercise computes $\Omega^1_{\mathrm{Ass}}(A)$ and identifies it with the noncommutative 1-forms.*
>
> > **Prerequisites:** [[#6.1 The Universal Property|6.1 The Universal Property]], [[#4.3 Classical Cases|4.3 Classical Cases]]
>
> For $\mathcal{O} = \mathrm{Ass}$ and an associative algebra $A$, show that $\Omega^1_{\mathrm{Ass}}(A) \cong \ker(\mu: A \otimes A \to A)$ as an $A \otimes A^{\mathrm{op}}$-module, where $\mu$ is the multiplication map. (This is the $A$-bimodule of noncommutative Kähler differentials, also denoted $\Omega^1_{\mathrm{nc}}(A)$.)

> [!TIP]- Solution to Exercise 10
> **Key insight:** The kernel of $\mu: A \otimes A \to A$ represents noncommutative 1-forms via $da = 1 \otimes a - a \otimes 1$.
>
> **Sketch:** The map $d: A \to \ker(\mu)$ given by $d(a) = 1 \otimes a - a \otimes 1$ satisfies the Leibniz rule: $d(ab) = 1 \otimes ab - ab \otimes 1 = (1 \otimes a - a \otimes 1)(1 \otimes b) + (a \otimes 1)(1 \otimes b - b \otimes 1) = d(a) \cdot b + a \cdot d(b)$. By the universal property, $\Omega^1_{\mathrm{Ass}}(A) \cong \ker(\mu)$ as $A \otimes A^{\mathrm{op}}$-modules. The isomorphism sends the generator $da$ to $1 \otimes a - a \otimes 1$.

> [!QUESTION] Exercise 11: Functoriality of Kahler Differentials
> *This exercise establishes that $\Omega^1_\mathcal{O}$ is functorial in the algebra map.*
>
> > **Prerequisites:** [[#6.1 The Universal Property|6.1 The Universal Property]]
>
> Let $f: A \to B$ be a morphism of $\mathcal{O}$-algebras. Construct a natural $U_\mathcal{O}(B)$-module map $f^*: B \otimes_{U_\mathcal{O}(A)} \Omega^1_\mathcal{O}(A) \to \Omega^1_\mathcal{O}(B)$ (the relative cotangent map). Show it is an isomorphism when $f$ is étale (in the algebraic sense).

> [!TIP]- Solution to Exercise 11
> **Key insight:** The map $f^*$ is induced by the universal property applied to $d_B \circ f: A \to \Omega^1_\mathcal{O}(B)$.
>
> **Sketch:** The composition $A \xrightarrow{f} B \xrightarrow{d_B} \Omega^1_\mathcal{O}(B)$ is an $\mathcal{O}$-derivation from $A$ to $\Omega^1_\mathcal{O}(B)$ viewed as an $A$-module via $f$. By the universal property, this factors through $\Omega^1_\mathcal{O}(A)$, giving $\Omega^1_\mathcal{O}(A) \to \Omega^1_\mathcal{O}(B)$. Extending scalars via $B \otimes_{A} -$ gives $f^*$. In the étale case (formally: the kernel and cokernel of $\hat{f}^*$ vanish in the derived sense), $f^*$ is an isomorphism — this is the algebraic analogue of the fact that étale maps have trivial relative cotangent.

---

## 7. A-infinity Algebras

📐 $A_\infty$-algebras, introduced by Stasheff in 1963, are algebras that are *homotopy associative* in a precise, controlled sense — associative up to an infinite tower of coherent homotopies.

### 7.1 The Stasheff Relations

**Definition 7.1 ($A_\infty$-Algebra).** Let $k$ be a field of characteristic zero. An *$A_\infty$-algebra* is a $\mathbb{Z}$-graded $k$-module $A = \bigoplus_{n \in \mathbb{Z}} A^n$ equipped with graded $k$-linear maps
$$m_n: A^{\otimes n} \longrightarrow A, \quad n \geq 1,$$
of degree $|m_n| = 2 - n$ (i.e., $m_n$ maps $A^{\otimes n}$ to $A$ with a degree shift of $2-n$), satisfying the *Stasheff identities*: for all $n \geq 1$,
$$\sum_{\substack{r+s+t = n \\ r, t \geq 0,\; s \geq 1}} (-1)^{rs+t}\; m_{r+1+t}\!\left(\mathrm{id}^{\otimes r} \otimes m_s \otimes \mathrm{id}^{\otimes t}\right) = 0.$$

The sign $(-1)^{rs+t}$ arises from the Koszul sign convention: moving $m_s$ of degree $2-s$ past $r$ inputs each of total degree (tracked cumulatively).

> [!NOTE] Sign conventions
> There is a cleaner formulation via the *suspended* module $sA = A[1]$ (shift: $(sA)^n = A^{n+1}$). All $m_n$ become degree $+1$ maps on $sA$. The coderivation $D = \sum_n D_n$ on the tensor coalgebra $T^c(sA) = \bigoplus_n (sA)^{\otimes n}$ (where $D_n|_{(sA)^{\otimes n}} = s m_n s^{-\otimes n}$) has degree $+1$, and the Stasheff identities are equivalent to the single equation $D^2 = 0$.

### 7.2 Low-Degree Consequences

Let us unpack the Stasheff identities for small $n$.

**$n=1$:** The only term has $r=t=0$, $s=1$: $m_1 \circ m_1 = 0$. Thus $m_1$ is a *differential* on $A$, making $(A, m_1)$ a chain complex.

**$n=2$:** Terms have $(r,s,t) \in \{(1,1,0),(0,1,1),(0,2,0)\}$:
$$(-1)^1 m_2(m_1 \otimes \mathrm{id}) + (-1)^0 m_2(\mathrm{id} \otimes m_1) + (-1)^0 m_1 m_2 = 0.$$
Rearranging: $m_1(m_2(a,b)) = m_2(m_1(a),b) + (-1)^{|a|} m_2(a, m_1(b))$. *This says $m_1$ is a derivation with respect to $m_2$* — the product $m_2$ is compatible with the differential.

**$n=3$:** The $n=3$ identity says that the *associator* $m_2(m_2(a,b),c) - m_2(a,m_2(b,c))$ equals the boundary of $m_3(a,b,c)$. **Thus $m_2$ is homotopy-associative, with $m_3$ the explicit homotopy for associativity.**

**$n=4$:** The $n=4$ identity says the pentagon: $m_3$ satisfies its own coherence up to a homotopy given by $m_4$.

> [!EXAMPLE] Differential graded algebras
> A *dga* (differential graded algebra) is an $A_\infty$-algebra with $m_n = 0$ for all $n \geq 3$. Then: $m_1$ is the differential, $m_2$ is the (strictly associative) product. The $n=3$ Stasheff identity reduces to strict associativity of $m_2$. Conversely, any $A_\infty$-algebra is quasi-isomorphic (via an $\infty$-morphism) to a dga when $k$ is a field of characteristic zero.

### 7.3 Infinity-Morphisms

**Definition 7.2 ($\infty$-Morphism).** An *$\infty$-morphism* $f: A \rightsquigarrow B$ of $A_\infty$-algebras is a collection of graded $k$-linear maps
$$f_n: A^{\otimes n} \longrightarrow B, \quad n \geq 1,$$
of degree $1-n$, satisfying: for all $n \geq 1$,
$$\sum_{\substack{r+s+t=n \\ s \geq 1}} (-1)^{rs+t} f_{r+1+t}\!\left(\mathrm{id}^{\otimes r} \otimes m_s^A \otimes \mathrm{id}^{\otimes t}\right) = \sum_{\substack{k \geq 1 \\ i_1+\cdots+i_k=n}} (-1)^\epsilon\, m_k^B\!\left(f_{i_1} \otimes \cdots \otimes f_{i_k}\right),$$
where $(-1)^\epsilon$ is a Koszul sign.

💡 *Surprisingly*, an $\infty$-morphism with $f_n = 0$ for $n \geq 2$ is precisely a strict morphism of $A_\infty$-algebras. The full notion is substantially more flexible: it allows morphisms that are only homotopy-compatible with the $A_\infty$-structures.

**Definition 7.3 ($\infty$-Quasi-Isomorphism).** An $\infty$-morphism $f: A \rightsquigarrow B$ is an *$\infty$-quasi-isomorphism* if $f_1: (A, m_1^A) \to (B, m_1^B)$ is a quasi-isomorphism of chain complexes.

> [!WARNING] Non-invertibility of strict morphisms
> A strict quasi-isomorphism of $A_\infty$-algebras need not be invertible as a strict morphism. However, every $\infty$-quasi-isomorphism is invertible as an $\infty$-morphism. This is the key homotopical property that makes $A_\infty$-algebras well-behaved.

### 7.4 Operadic Perspective

🔑 $A_\infty$-algebras fit into the operadic framework via the *bar-cobar construction* and Koszul duality.

The associative operad $\mathrm{Ass}$ is Koszul, with Koszul dual cooperad $\mathrm{Ass}^! = \mathrm{Ass}^{\scriptstyle\vee}$ (the linear dual). The *cobar construction* $\Omega(\mathrm{Ass}^!)$ produces a dg-operad whose algebras are precisely $A_\infty$-algebras. This is made precise in [[concepts/category-theory/operads/koszul-duality|Koszul Duality and the Bar Construction]].

**Proposition 7.4.** An $A_\infty$-algebra structure on $A$ is equivalent to a morphism of dg-operads
$$\Omega(\mathrm{Ass}^!) \longrightarrow \mathrm{End}_A.$$

*Heuristic sketch.* The cobar construction $\Omega(\mathrm{Ass}^!)$ is freely generated as an operad by elements $m_n$ in arity $n$ and degree $2-n$, subject to the differential $\partial(m_n) = \sum_{r+s+t=n} \pm m_{r+1+t} \circ_i m_s$ (the Stasheff relations arising from $\partial^2 = 0$ in the cobar complex). An operad map $\Omega(\mathrm{Ass}^!) \to \mathrm{End}_A$ is therefore exactly a choice of maps $\{m_n \in \mathrm{End}_A(n)\}$ satisfying the Stasheff identities.

> [!QUESTION] Exercise 12: The n=3 Stasheff Identity
> *This exercise expands the Stasheff identity for $n=3$ explicitly and interprets each term.*
>
> > **Prerequisites:** [[#7.1 The Stasheff Relations|7.1 The Stasheff Relations]]
>
> Write out all six terms $(r, s, t)$ with $r + s + t = 3$, $s \geq 1$ in the Stasheff identity for $n = 3$. Organize them to show that the identity states: the sum of all ways of applying one $m_2$ and one $m_2$ (with appropriate signs) equals the boundary of $m_3$, i.e., $m_3$ is a homotopy for the failure of $m_2$-associativity.

> [!TIP]- Solution to Exercise 12
> **Key insight:** The six terms split into three pairs: those involving $m_2 \circ m_2$ (failure of associativity) and those involving $m_1 \circ m_3$ and $m_3 \circ m_1$ (boundary of $m_3$).
>
> **Sketch:** For $n=3$, $(r,s,t)$ runs over $(0,1,2),(0,2,1),(0,3,0),(1,1,1),(1,2,0),(2,1,0)$. The terms:
> - $(0,1,2)$: $(-1)^{0\cdot1+2} m_3(m_1 \otimes \mathrm{id} \otimes \mathrm{id}) = m_3(m_1 a_1, a_2, a_3)$ (sign $(-1)^2=+1$)
> - $(0,2,1)$: $(-1)^{0\cdot2+1} m_2(m_2 a_1 a_2, a_3)$ (sign $-1$)... wait, $s=2$, so the term is $m_{0+1+1}(m_2 \otimes \mathrm{id}) = m_2(m_2(a_1,a_2), a_3)$ with sign $(-1)^{0\cdot2+1} = -1$.
> - $(0,3,0)$: $(-1)^0 m_1(m_3(a_1,a_2,a_3))$ (sign $+1$).
> - $(1,1,1)$: $(-1)^{1+1} m_3(\mathrm{id} \otimes m_1 \otimes \mathrm{id}) = m_3(a_1, m_1 a_2, a_3)$ (sign $+1$).
> - $(1,2,0)$: $(-1)^{1\cdot2+0} m_2(a_1, m_2(a_2,a_3))$ (sign $-1$... wait $(-1)^{rs+t} = (-1)^{1\cdot2+0} = +1$). So $+m_2(a_1,m_2(a_2,a_3))$.
> - $(2,1,0)$: $(-1)^{2\cdot1+0} m_3(\mathrm{id} \otimes \mathrm{id} \otimes m_1)$ (sign $+1$). So the identity: $m_1 m_3 + m_3(m_1,-,-) + m_3(-,m_1,-) + m_3(-,-,m_1) = m_2(m_2 \cdot, \cdot) - m_2(\cdot, m_2 \cdot)$, i.e., $\partial m_3 = $ associator, confirming $m_3$ is a homotopy.

> [!QUESTION] Exercise 13: Infinity-Morphisms Composing
> *This exercise establishes that $\infty$-morphisms compose to give $\infty$-morphisms.*
>
> > **Prerequisites:** [[#7.3 Infinity-Morphisms|7.3 Infinity-Morphisms]]
>
> Let $f: A \rightsquigarrow B$ and $g: B \rightsquigarrow C$ be $\infty$-morphisms of $A_\infty$-algebras. Define the composite $(g \circ f)_n = \sum_{k \geq 1} \sum_{i_1+\cdots+i_k=n} \pm g_k(f_{i_1} \otimes \cdots \otimes f_{i_k})$. Verify that $g \circ f$ satisfies the $\infty$-morphism identities, showing that $A_\infty$-algebras and $\infty$-morphisms form a category.

> [!TIP]- Solution to Exercise 13
> **Key insight:** Composition of $\infty$-morphisms corresponds to composition of coderivations on tensor coalgebras; the coalgebra category is strict.
>
> **Sketch:** Lift $f$ and $g$ to coalgebra maps $F: T^c(sA) \to T^c(sB)$ and $G: T^c(sB) \to T^c(sC)$, each determined by their components $\{f_n\}$ and $\{g_n\}$ respectively. The composite $G \circ F: T^c(sA) \to T^c(sC)$ is again a coalgebra map (coalgebra maps compose), and its $n$-th component is exactly $(g \circ f)_n$ as defined. The $\infty$-morphism identities for $G \circ F$ follow from $G \circ D_A = D_C \circ G$ and $F \circ D_A = D_B \circ F$ (where $D_A, D_B, D_C$ are the coderivations encoding the $A_\infty$-structures), giving $G \circ F \circ D_A = G \circ D_B \circ F = D_C \circ G \circ F$.

---

## 8. The Probability Operad Revisited

📐 We return to the probability operad $\mathcal{P}$ introduced in [[concepts/category-theory/operads/definitions|Definitions and Examples]] (Section 5.7). Recall: $\mathcal{P}(n)$ is the topological simplex $\Delta_n = \{(p_1, \ldots, p_{n+1}) : p_i \geq 0, \sum p_i = 1\}$ (in the topological version of Bradley) or $\mathcal{P}(n) = \{(p_1, \ldots, p_n) : p_i \geq 0, \sum p_i = 1\}$, with operad structure given by operadic composition of probability distributions.

### 8.1 P-Algebra Structure on the Positive Reals

**Definition 8.1 ($\mathcal{P}$-Algebra).** A *$\mathcal{P}$-algebra* is a set $X$ (or $k$-module) equipped with structure maps $\gamma_X(n): \mathcal{P}(n) \times X^n \to X$ satisfying the algebra axioms of Definition 1.1.

**Example 8.2 ($[0, \infty)$ as a $\mathcal{P}$-algebra).** The set $[0, \infty)$ (or $[0,1]$, or $\mathbb{R}_{\geq 0}$) becomes a $\mathcal{P}$-algebra via the *weighted mean* structure map:
$$\gamma(p_1, \ldots, p_n;\, x_1, \ldots, x_n) := \sum_{i=1}^n p_i x_i.$$
The algebra axioms reduce to standard properties of convex combinations:
- *Associativity:* $\sum_i p_i (\sum_j q_{ij} x_{ij}) = \sum_{i,j} p_i q_{ij} x_{ij}$, i.e., iterated weighted averages compose correctly.
- *Unitality:* $1 \cdot x = x$.

> [!NOTE] Convexity
> The $\mathcal{P}$-algebra structure on $[0,\infty)$ is precisely the structure of a *convex space*: a set with well-defined convex combinations satisfying the natural coherences. Entropy is a function on such a convex space.

### 8.2 Shannon Entropy as a Derivation

🔑 The central result, proved by Bradley in [[research/entropy-operad-derivation|Entropy as an Operad Derivation]], is:

**Theorem 8.3 (Bradley 2021).** Shannon entropy $H: \Delta_n \to \mathbb{R}$, defined by
$$H(p_1, \ldots, p_n) = -\sum_{i=1}^n p_i \log p_i,$$
is a *derivation of the operad $\mathcal{P}$* with values in the $\mathcal{P}$-bimodule $\mathbb{R}$ (with the bimodule structure $\gamma_\mathbb{R}(p; x) = \sum_i p_i x_i$). Moreover, up to a positive constant, $H$ is the unique such derivation.

The operadic Leibniz rule for $H$ (Definition 5.3) gives:
$$H(p \circ_i q) = H(p) + p_i H(q),$$
where $p \circ_i q$ denotes the operadic composition inserting distribution $q$ in position $i$ of distribution $p$. **This is precisely the chain rule for Shannon entropy.**

*Proof sketch.* By the operadic Leibniz rule applied to the composition $\gamma_\mathcal{P}(p; q^1, \ldots, q^n) = p \circ (q^1, \ldots, q^n)$:
$$H(p \circ (q^1,\ldots,q^n)) = H(p) \cdot \gamma_\mathbb{R}(q^1, \ldots, q^n) + \sum_i p_i H(q^i).$$
When only $q^i \neq \mathrm{id}$ (i.e., all other $q^j$ are pure point distributions), this reduces to $H(p \circ_i q) = H(p) + p_i H(q)$. The uniqueness follows from Faddeev's characterization theorem: any continuous function satisfying this chain rule is a constant multiple of $H$. $\square$

> [!QUESTION] Exercise 14: Leibniz Rule for Entropy
> *This exercise verifies the chain rule identity directly from the operadic Leibniz rule.*
>
> > **Prerequisites:** [[#5.1 Definition and the Leibniz Rule|5.1 Definition and the Leibniz Rule]], [[#8.2 Shannon Entropy as a Derivation|8.2 Shannon Entropy as a Derivation]]
>
> Let $p = (p_1, p_2) \in \Delta_2$ and $q = (q_1, q_2) \in \Delta_2$. The composition $p \circ_1 q \in \Delta_3$ is the distribution $(p_1 q_1, p_1 q_2, p_2)$. Verify the chain rule $H(p \circ_1 q) = H(p) + p_1 H(q)$ by direct computation using $H(r_1, r_2, r_3) = -r_1 \log r_1 - r_2 \log r_2 - r_3 \log r_3$.

> [!TIP]- Solution to Exercise 14
> **Key insight:** The cross-terms from $-p_1 q_j \log(p_1 q_j) = -p_1 q_j \log p_1 - p_1 q_j \log q_j$ factor as $H(p)$ (weighted by total weight $p_1$) plus $p_1 H(q)$.
>
> **Sketch:** $H(p_1 q_1, p_1 q_2, p_2) = -p_1 q_1 \log(p_1 q_1) - p_1 q_2 \log(p_1 q_2) - p_2 \log p_2$.
> Expand: $-p_1 q_j \log(p_1 q_j) = -p_1 q_j \log p_1 - p_1 q_j \log q_j$.
> Sum over $j \in \{1,2\}$: $-p_1 \log p_1 (q_1 + q_2) - p_1(q_1 \log q_1 + q_2 \log q_2) = -p_1 \log p_1 - p_1 H(q)$.
> Adding $-p_2 \log p_2$: $H(p \circ_1 q) = -(p_1 \log p_1 + p_2 \log p_2) + p_1 H(q) = H(p) + p_1 H(q)$. $\checkmark$

### 8.3 Connection to BFL Entropy

The broader research thread on entropy and operads is developed in [[research/entropy-operad-derivation|Entropy as an Operad Derivation]] and [[research/categorical-entropy|Categorical Entropy]].

The *BFL (Baez-Fritz-Leinster) entropy* is a categorification of Shannon entropy using a monoidal functor from a category of finite probability spaces to $[0, \infty)$. The operadic derivation perspective illuminates the BFL result: **the chain rule $H(p \circ_i q) = H(p) + p_i H(q)$ is not a property of entropy but its definition** — entropy is the unique (up to scaling) derivation of $\mathcal{P}$.

The evaluation map of Proposition 5.4 connects these: the derivation of the operad $\mathcal{P}$ (entropy as an operad-level map) descends, via evaluation on the $\mathcal{P}$-algebra $([0,\infty), \gamma_\mathbb{R})$, to a classical derivation of the $\mathcal{P}$-algebra structure. This is the bridge between the operadic characterization and the classical information-theoretic one.

> [!QUESTION] Exercise 15: Uniqueness via Faddeev
> *This exercise outlines the uniqueness argument showing that entropy is the unique operad derivation of $\mathcal{P}$.*
>
> > **Prerequisites:** [[#8.2 Shannon Entropy as a Derivation|8.2 Shannon Entropy as a Derivation]]
>
> Assume $D: \mathcal{P} \to \mathbb{R}$ is a continuous operad derivation (satisfying the chain rule $D(p \circ_i q) = D(p) + p_i D(q)$). By setting $q = (t, 1-t) \in \Delta_2$ and $p = (1) \in \Delta_1$, derive a functional equation for $f(t) := D(t, 1-t)$ and show it implies $f(t) = -c(t \log t + (1-t)\log(1-t))$ for some constant $c \geq 0$.

> [!TIP]- Solution to Exercise 15
> **Key insight:** The chain rule with $p = (1)$ forces $D(p \circ_1 q) = D(q)$, giving the functional equation for binary entropy; the only continuous solutions are scalar multiples of binary entropy.
>
> **Sketch:** With $p = (1) \in \Delta_1$ (the unique 1-element distribution), $p \circ_1 q = q$, so $D(q) = D(p) + 1 \cdot D(q)$, implying $D(p) = 0$ (consistent since $D$ of a 1-element distribution is zero by unitality). Apply the chain rule to $(s, 1-s) \circ_1 (t/(s), (s-t)/s)$: $D(t, s-t, 1-s) = D(s, 1-s) + s \cdot D(t/s, 1-t/s)$. Setting $g(t) = D(t, 1-t)$, this gives $g(t) + g(s-t) = g(s) + s \cdot g(t/s)$ (Cauchy-type functional equation). Under continuity, the only solutions are $g(t) = c(-t\log t - (1-t)\log(1-t))$, the binary entropy.

---

## Algorithmic Applications

> [!QUESTION] Exercise 16: Computing the Free Ass-Algebra
> *This exercise implements the tensor algebra as the free Ass-algebra and explores its degree decomposition.*
>
> > **Prerequisites:** [[#2.2 Classical Cases|2.2 Classical Cases]]
>
> Let $V = k^r$. Write Python pseudocode for a function `free_ass_algebra(r, max_degree)` that returns the graded components $T(V)_n = V^{\otimes n}$ for $0 \leq n \leq$ `max_degree` as lists of basis monomials (tuples of indices in $\{1,\ldots,r\}$). Implement the multiplication map $T(V)_m \times T(V)_n \to T(V)_{m+n}$ by concatenation.

> [!TIP]- Solution to Exercise 16
> **Key insight:** The tensor algebra has basis consisting of all words in the alphabet $\{1, \ldots, r\}$; multiplication is concatenation.
>
> **Sketch:**
> ```python
> from itertools import product
>
> def free_ass_algebra(r: int, max_degree: int) -> dict[int, list[tuple]]:
>     """
>     Returns graded components of T(k^r) up to max_degree.
>     Each basis element is a tuple of indices in {1, ..., r}.
>     """
>     components = {}
>     for n in range(max_degree + 1):
>         if n == 0:
>             components[0] = [()]  # unit element (empty word)
>         else:
>             components[n] = list(product(range(1, r + 1), repeat=n))
>     return components
>
> def multiply(u: tuple, v: tuple) -> tuple:
>     """Multiplication in T(k^r) is concatenation."""
>     return u + v
>
> # Example: r=2, basis of degree 2 is [(1,1),(1,2),(2,1),(2,2)]
> alg = free_ass_algebra(2, 3)
> print(f"dim T(k^2)_3 = {len(alg[3])}")  # 8 = 2^3
> ```

> [!QUESTION] Exercise 17: Operad Monad Multiplication
> *This exercise implements the monad multiplication for the Com operad, i.e., the algebra map $\mathrm{Sym}(\mathrm{Sym}(V)) \to \mathrm{Sym}(V)$.*
>
> > **Prerequisites:** [[#2.3 The Operad Monad|2.3 The Operad Monad]]
>
> Write Python pseudocode for the monad multiplication $\mu: \mathrm{Sym}(\mathrm{Sym}(V)) \to \mathrm{Sym}(V)$ when $V = k$ (so $\mathrm{Sym}(V) = k[x]$ and $\mathrm{Sym}(\mathrm{Sym}(V)) = k[y_0, y_1, y_2, \ldots]$ where $y_n$ corresponds to $x^n$). Represent elements of $\mathrm{Sym}(V) = k[x]$ as coefficient lists and implement $\mu$ as polynomial substitution.

> [!TIP]- Solution to Exercise 17
> **Key insight:** The monad multiplication for $\mathrm{Sym}$ is polynomial composition: a polynomial in $\{y_n\}$ where each $y_n$ represents $x^n$ is flattened by substituting $y_n \mapsto x^n$.
>
> **Sketch:**
> ```python
> import numpy as np
> from numpy.polynomial import polynomial as P
>
> def sym_sym_to_sym(poly_of_polys: list[list[float]]) -> list[float]:
>     """
>     Monad multiplication mu: Sym(Sym(V)) -> Sym(V).
>     poly_of_polys: list of coefficient lists for each y_n.
>     Element of Sym(Sym(k)) = k[y_0, y_1, ...] is a polynomial in the y_n's.
>     We evaluate at y_n = x^n.
>
>     Input: poly_of_polys[d] = coefficients of the degree-d monomial's
>            polynomial-coefficient in x, i.e., the coefficient of y_0^{d_0} * y_1^{d_1}...
>     Simplified: represent an element as sum_n c_n * y_n, a linear polynomial.
>     Then mu(sum_n c_n * y_n) = sum_n c_n * x^n.
>     """
>     # For linear elements sum_n c_n * y_n:
>     # result polynomial has coefficient c_n for x^n
>     coeffs = poly_of_polys  # [c_0, c_1, c_2, ...]
>     return coeffs  # direct substitution: y_n -> x^n
>
> def monad_mult_nonlinear(outer_poly_coeffs, inner_polys):
>     """
>     General case: outer_poly is a polynomial in y-variables,
>     inner_polys[n] = coefficients of x^n polynomial replacing y_n.
>     Evaluate by substitution.
>     """
>     result = [0.0]
>     for n, coeff in enumerate(outer_poly_coeffs):
>         if abs(coeff) > 1e-12:
>             x_to_n = [0.0] * n + [1.0]  # x^n
>             term = [c * coeff for c in x_to_n]
>             result = P.polyadd(result, term)
>     return result
> ```

> [!QUESTION] Exercise 18: Computing Kahler Differentials
> *This exercise implements the module of Kähler differentials for a polynomial algebra.*
>
> > **Prerequisites:** [[#6.3 Recovery for Com|6.3 Recovery for Com]]
>
> For $A = k[x, y]/(xy)$ (the coordinate ring of the union of two coordinate axes), compute $\Omega^1_{A/k}$ explicitly: find generators and relations as an $A$-module. Write Python pseudocode to represent elements of $\Omega^1_{A/k}$ as pairs $(f, g)$ corresponding to $f \cdot dx + g \cdot dy$, and implement the differential $d: A \to \Omega^1_{A/k}$ via formal partial derivatives.

> [!TIP]- Solution to Exercise 18
> **Key insight:** $\Omega^1_{A/k}$ for $A = k[x,y]/(xy)$ is generated by $dx, dy$ with the relation $d(xy) = x\,dy + y\,dx = 0$ (from $xy = 0$ in $A$).
>
> **Sketch:**
> ```python
> from sympy import symbols, Rational, diff, expand
> from sympy import Symbol
>
> x, y = symbols('x y')
>
> def differential(f_expr):
>     """
>     Compute d(f) in Omega^1_{k[x,y]/(xy)/k}.
>     Returns (coeff_dx, coeff_dy) representing df = coeff_dx*dx + coeff_dy*dy.
>     Apply the relation xy=0 before computing partial derivatives.
>     """
>     # Partial derivatives
>     df_dx = diff(f_expr, x)
>     df_dy = diff(f_expr, y)
>     return (df_dx, df_dy)
>
> def apply_relation(coeff):
>     """Apply xy = 0 in the quotient ring."""
>     return expand(coeff).subs(x * y, 0)
>
> def omega_element_add(w1, w2):
>     """Add two Kahler differential elements."""
>     return (apply_relation(w1[0] + w2[0]),
>             apply_relation(w1[1] + w2[1]))
>
> # The relation from d(xy) = 0: y*dx + x*dy = 0
> # So in Omega^1, we have the relation (y, x) ~ 0
> # This means x*dy = -y*dx, a key relation between the generators
>
> # Example: d(x^2) = 2x*dx, d(y^2) = 2y*dy
> print(differential(x**2))  # (2*x, 0) -- this is 2x*dx
> print(differential(y**2))  # (0, 2*y) -- this is 2y*dy
> ```

> [!QUESTION] Exercise 19: A-infinity Structure from a dga
> *This exercise constructs an $A_\infty$-algebra from a differential graded algebra by identifying the structure maps.*
>
> > **Prerequisites:** [[#7.2 Low-Degree Consequences|7.2 Low-Degree Consequences]]
>
> Let $(A, d, \cdot)$ be a differential graded algebra (dga): $d: A^n \to A^{n+1}$ with $d^2=0$, and $\cdot: A^m \otimes A^n \to A^{m+n}$ strictly associative. Write pseudocode for a function `dga_to_ainfty(A, d, mult)` that returns the $A_\infty$-structure maps $\{m_1, m_2, m_3, \ldots\}$ where $m_1 = d$, $m_2 = $ product, $m_n = 0$ for $n \geq 3$. Verify the $n=1,2,3$ Stasheff identities for this input.

> [!TIP]- Solution to Exercise 19
> **Key insight:** A dga is a special $A_\infty$-algebra with all higher structure maps zero; the Stasheff identities reduce to $d^2=0$ and the Leibniz rule.
>
> **Sketch:**
> ```python
> def dga_to_ainfty(dim: int, d_matrix, mult_tensor):
>     """
>     dim: dimension of underlying graded vector space
>     d_matrix: (dim x dim) matrix representing differential d
>     mult_tensor: (dim x dim x dim) tensor representing multiplication
>
>     Returns list of structure maps m_n for n >= 1.
>     m_1 = d, m_2 = mult, m_n = 0 for n >= 3.
>     """
>     import numpy as np
>
>     def m1(a):
>         return d_matrix @ a
>
>     def m2(a, b):
>         # m2(a,b)_k = sum_{i,j} mult_tensor[k,i,j] * a[i] * b[j]
>         return np.einsum('kij,i,j->k', mult_tensor, a, b)
>
>     def m_n(n):
>         if n >= 3:
>             return lambda *args: np.zeros(dim)
>         elif n == 2:
>             return m2
>         elif n == 1:
>             return m1
>
>     # Verify n=1 Stasheff: m1(m1(a)) = 0 (i.e., d^2 = 0)
>     def verify_n1(a):
>         return np.allclose(m1(m1(a)), 0)
>
>     # Verify n=2: m1(m2(a,b)) = m2(m1(a),b) + (-1)^|a| m2(a,m1(b))
>     # (sign depends on degree; simplified here for degree 0 inputs)
>     def verify_n2(a, b):
>         lhs = m1(m2(a, b))
>         rhs = m2(m1(a), b) + m2(a, m1(b))
>         return np.allclose(lhs, rhs)
>
>     # Verify n=3: m3=0, so the identity reduces to:
>     # m2(m2(a,b),c) - m2(a,m2(b,c)) = 0 (strict associativity)
>     def verify_n3(a, b, c):
>         return np.allclose(m2(m2(a,b),c), m2(a,m2(b,c)))
>
>     return {'m1': m1, 'm2': m2, 'verify': (verify_n1, verify_n2, verify_n3)}
> ```

> [!QUESTION] Exercise 20: Entropy as a Derivation: Numerical Check
> *This exercise numerically verifies that Shannon entropy satisfies the operadic Leibniz rule.*
>
> > **Prerequisites:** [[#8.2 Shannon Entropy as a Derivation|8.2 Shannon Entropy as a Derivation]]
>
> Write Python pseudocode for a function `verify_entropy_derivation(p, q, i)` that: (a) computes the operadic composition $p \circ_i q$ (inserting distribution $q$ at position $i$ of distribution $p$), (b) computes $H(p \circ_i q)$ and $H(p) + p[i] \cdot H(q)$, and (c) verifies they are numerically equal to within floating-point precision.

> [!TIP]- Solution to Exercise 20
> **Key insight:** The chain rule is an exact algebraic identity; numerical verification confirms the operadic Leibniz rule with machine precision.
>
> **Sketch:**
> ```python
> import numpy as np
>
> def entropy(p: np.ndarray) -> float:
>     """Shannon entropy H(p) = -sum p_i log p_i."""
>     p = p[p > 0]  # avoid 0 * log(0) = 0 by convention
>     return -np.sum(p * np.log(p))
>
> def operad_compose(p: np.ndarray, q: np.ndarray, i: int) -> np.ndarray:
>     """
>     Compute p circ_i q: insert distribution q at position i of p.
>     Result has len(p) - 1 + len(q) entries.
>     """
>     before = p[:i]
>     p_i = p[i]
>     after = p[i+1:]
>     inserted = p_i * q
>     return np.concatenate([before, inserted, after])
>
> def verify_entropy_derivation(p: np.ndarray,
>                                q: np.ndarray,
>                                i: int,
>                                tol: float = 1e-10) -> bool:
>     """
>     Verify H(p circ_i q) = H(p) + p[i] * H(q).
>     """
>     composed = operad_compose(p, q, i)
>     lhs = entropy(composed)
>     rhs = entropy(p) + p[i] * entropy(q)
>     print(f"H(p ∘_i q) = {lhs:.8f}")
>     print(f"H(p) + p_i*H(q) = {rhs:.8f}")
>     print(f"Difference: {abs(lhs - rhs):.2e}")
>     return abs(lhs - rhs) < tol
>
> # Test: p = (0.5, 0.5), q = (0.3, 0.7), i = 0
> p = np.array([0.5, 0.5])
> q = np.array([0.3, 0.7])
> result = verify_entropy_derivation(p, q, 0)
> print(f"Leibniz rule holds: {result}")
> # Expected: H(0.15, 0.35, 0.5) = H(0.5, 0.5) + 0.5 * H(0.3, 0.7)
> ```

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Loday & Vallette, *Algebraic Operads*](https://link.springer.com/book/10.1007/978-3-642-30362-3) | Comprehensive treatment of operads, algebras, Koszul duality; Chapters 5–6 cover algebras and modules over operads; Chapter 12 covers cohomology, enveloping algebras, and Kähler differentials | [Springer](https://link.springer.com/book/10.1007/978-3-642-30362-3) |
| [Fresse, *Modules over Operads and Functors*](https://link.springer.com/book/10.1007/978-3-540-89056-0) | Systematic treatment of left/right modules over operads as symmetric sequences; homotopy theory of module categories | [Springer](https://link.springer.com/book/10.1007/978-3-540-89056-0) |
| [Kriz & May, *Operads, Algebras, Modules, and Motives*](https://www.math.uchicago.edu/~may/BOOKS/KM.pdf) | Early foundational reference connecting operads to stable homotopy; module theory over operads in spectra | [PDF](https://www.math.uchicago.edu/~may/BOOKS/KM.pdf) |
| [Keller, *Introduction to A-infinity Algebras and Modules*](https://arxiv.org/abs/math/9910179) | Expanded lecture notes defining $A_\infty$-algebras, modules, derived categories; coderivation formulation of the Stasheff identities | [arXiv](https://arxiv.org/abs/math/9910179) |
| [Bradley, *Entropy as a Topological Operad Derivation*](https://arxiv.org/abs/2107.09581) | Proves Shannon entropy is a derivation of the probability operad $\mathcal{P}$; the chain rule as the Leibniz rule | [arXiv](https://arxiv.org/abs/2107.09581) |
| [Stasheff, *Homotopy Associativity of H-Spaces I, II*](https://www.ams.org/journals/tran/1963-108-02/S0002-9947-1963-0158400-5/) | Original 1963 paper introducing $A_\infty$-spaces and the associahedra; foundational for all homotopy-coherent algebra | [AMS](https://www.ams.org/journals/tran/1963-108-02/S0002-9947-1963-0158400-5/) |
| [Quillen, *On the (co-)homology of commutative rings*](https://www.ams.org/books/pspum/017/) | Introduces André-Quillen cohomology and the correct derived functor of derivations; foundational for Kähler differentials in the operadic setting | [AMS Proceedings](https://www.ams.org/books/pspum/017/) |
| [nLab, *A-infinity algebra*](https://ncatlab.org/nlab/show/A-infinity-algebra) | Concise categorical treatment; coderivation and Maurer-Cartan formulations; infinity-morphisms | [nLab](https://ncatlab.org/nlab/show/A-infinity-algebra) |
