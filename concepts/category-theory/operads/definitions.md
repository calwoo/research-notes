# Operads: Foundations and Definitions

## Table of Contents

- [[#1. Symmetric Sequences|1. Symmetric Sequences]]
  - [[#1.1 The Category SymSeq|1.1 The Category SymSeq]]
  - [[#1.2 The Unit Object|1.2 The Unit Object]]
  - [[#1.3 Schur Functors|1.3 Schur Functors]]
- [[#2. The Composition Product|2. The Composition Product]]
  - [[#2.1 Definition of the Substitution Product|2.1 Definition of the Substitution Product]]
  - [[#2.2 Monoidal Structure on SymSeq|2.2 Monoidal Structure on SymSeq]]
  - [[#2.3 Non-Symmetry of the Composition Product|2.3 Non-Symmetry of the Composition Product]]
- [[#3. Operads as Monoids|3. Operads as Monoids]]
  - [[#3.1 Monoid Definition|3.1 Monoid Definition]]
  - [[#3.2 Partial Composition Formulation|3.2 Partial Composition Formulation]]
  - [[#3.3 Equivalence of the Two Formulations|3.3 Equivalence of the Two Formulations]]
- [[#4. Morphisms and the Category Op|4. Morphisms and the Category Op]]
- [[#5. Key Examples|5. Key Examples]]
  - [[#5.1 The Associative Operad Ass|5.1 The Associative Operad Ass]]
  - [[#5.2 The Commutative Operad Com|5.2 The Commutative Operad Com]]
  - [[#5.3 The Lie Operad Lie|5.3 The Lie Operad Lie]]
  - [[#5.4 The Endomorphism Operad EndV|5.4 The Endomorphism Operad EndV]]
  - [[#5.5 Operad Morphisms and Forgetful Functors|5.5 Operad Morphisms and Forgetful Functors]]
  - [[#5.6 The Little n-Disks Operad|5.6 The Little n-Disks Operad]]
  - [[#5.7 The Probability Operad|5.7 The Probability Operad]]
- [[#6. Colored Operads and Multicategories|6. Colored Operads and Multicategories]]
  - [[#6.1 Definition of Colored Operads|6.1 Definition of Colored Operads]]
  - [[#6.2 Examples and Connections|6.2 Examples and Connections]]
- [[#References|References]]

---

## 1. Symmetric Sequences

📐 We work throughout over a commutative unital ring $k$. All tensor products are over $k$ unless otherwise noted. The category of $k$-modules is denoted $\mathbf{Mod}_k$.

**Definition 1.1 (Symmetric Sequence).** A *symmetric sequence* (also called an *$S$-module*) is a collection
$$\mathcal{M} = \{\mathcal{M}(n)\}_{n \geq 0}$$
where each $\mathcal{M}(n)$ is a right $k[S_n]$-module — that is, a $k$-module equipped with a right action of the *symmetric group* $S_n$. We write the action as $m \cdot \sigma$ for $m \in \mathcal{M}(n)$ and $\sigma \in S_n$.

> [!NOTE] Convention on Arity Zero
> The component $\mathcal{M}(0)$ is often called the *nullary* component. For many operads of interest (e.g., $\mathrm{Ass}$, $\mathrm{Com}$, $\mathrm{Lie}$) one takes $\mathcal{M}(0) = 0$, though the general theory permits nonzero nullary components, which correspond to *constants* in the algebra.

Equivalently, letting $\mathbb{S}$ denote the *groupoid* whose objects are finite sets $\mathbf{n} = \{1, \ldots, n\}$ for $n \geq 0$ and whose morphisms are bijections (so $\mathrm{Hom}_\mathbb{S}(\mathbf{n}, \mathbf{n}) = S_n$ and $\mathrm{Hom}_\mathbb{S}(\mathbf{m}, \mathbf{n}) = \emptyset$ for $m \neq n$), a symmetric sequence is precisely a functor
$$\mathcal{M} \colon \mathbb{S}^{\mathrm{op}} \longrightarrow \mathbf{Mod}_k.$$
The contravariant functor perspective makes the right $S_n$-action manifest: a bijection $\sigma \colon \mathbf{n} \to \mathbf{n}$ is sent to the map $\mathcal{M}(\sigma) \colon \mathcal{M}(n) \to \mathcal{M}(n)$, giving the right action by setting $m \cdot \sigma := \mathcal{M}(\sigma)(m)$.

### 1.1 The Category SymSeq

**Definition 1.2 (Morphism of Symmetric Sequences).** A *morphism* $f \colon \mathcal{M} \to \mathcal{N}$ of symmetric sequences is a collection of $k$-linear maps $f_n \colon \mathcal{M}(n) \to \mathcal{N}(n)$, $n \geq 0$, each of which is $S_n$-equivariant:
$$f_n(m \cdot \sigma) = f_n(m) \cdot \sigma \quad \text{for all } m \in \mathcal{M}(n),\ \sigma \in S_n.$$

The collection of symmetric sequences and their morphisms assembles into a category, which we denote $\mathbf{SymSeq}$ (or $\mathbf{Mod}_k[\mathbb{S}]$, the functor category $[\mathbb{S}^{\mathrm{op}}, \mathbf{Mod}_k]$). This category is abelian (being a functor category into an abelian category) and inherits the projective model structure when $k$ is a field of characteristic zero.

> [!TIP] Schur Functors
> Every symmetric sequence $\mathcal{M}$ determines a *Schur functor* $\overline{\mathcal{M}} \colon \mathbf{Mod}_k \to \mathbf{Mod}_k$ by the formula
> $$\overline{\mathcal{M}}(V) := \bigoplus_{n \geq 0} \mathcal{M}(n) \otimes_{S_n} V^{\otimes n}.$$
> Here $\otimes_{S_n}$ denotes the coinvariant (quotient) tensor product: $M \otimes_{S_n} N := (M \otimes_k N) / \langle m \cdot \sigma \otimes v - m \otimes \sigma \cdot v \rangle$. The Schur functor perspective reveals that composition of Schur functors corresponds to the composition product defined in Section 2.

### 1.2 The Unit Object

**Definition 1.3 (Unit Symmetric Sequence).** The *unit* symmetric sequence $\mathbf{1}$ is defined by
$$\mathbf{1}(n) := \begin{cases} k & n = 1 \\ 0 & n \neq 1. \end{cases}$$
The group $S_1 = \{e\}$ acts trivially on $k$, so $\mathbf{1}(1) = k$ carries the unique (trivial) $S_1$-module structure.

This object will be the monoidal unit for the composition product $\circ$ defined in Section 2. Intuitively, $\mathbf{1}$ represents the "identity operation" of arity 1.

### 1.3 Schur Functors

**Definition 1.4 (Schur Functor).** To each $\mathcal{M} \in \mathbf{SymSeq}$ we associate the *Schur functor* $F_\mathcal{M} \colon \mathbf{Mod}_k \to \mathbf{Mod}_k$ defined by
$$F_\mathcal{M}(V) := \bigoplus_{n \geq 0} \mathcal{M}(n) \otimes_{S_n} V^{\otimes n}.$$
When $\mathcal{M}(0) = 0$, this sum starts at $n = 1$ and the functor is *reduced*. One checks that $F_\mathbf{1}(V) = V$ for all $V$.

> [!EXAMPLE] Schur Functor for Com
> Let $\mathrm{Com}(n) = k$ with trivial $S_n$-action. Then
> $$F_{\mathrm{Com}}(V) = \bigoplus_{n \geq 0} k \otimes_{S_n} V^{\otimes n} = \bigoplus_{n \geq 0} V^{\otimes n} / S_n = \bigoplus_{n \geq 0} \mathrm{Sym}^n(V) = \mathrm{Sym}(V),$$
> the full symmetric algebra. This foreshadows why $\mathrm{Com}$-algebras are commutative algebras.

> [!QUESTION] Exercise 1: Schur Functor for Ass
> *This problem establishes that the Schur functor of the associative operad recovers the tensor algebra.*
>
> > **Prerequisites:** [[#1.3 Schur Functors|1.3 Schur Functors]]
>
> Let $\mathrm{Ass}(n) = k[S_n]$ with $S_n$ acting by right multiplication. Compute $F_{\mathrm{Ass}}(V)$ and identify it as a familiar algebra.

> [!TIP]- Solution to Exercise 1
> **Key insight:** The coinvariant quotient $k[S_n] \otimes_{S_n} V^{\otimes n}$ kills the $S_n$-action, but since $S_n$ acts freely on $k[S_n]$, the quotient is simply $k \otimes V^{\otimes n} \cong V^{\otimes n}$.
>
> **Sketch:** We have $k[S_n] \otimes_{S_n} V^{\otimes n} \cong V^{\otimes n}$ via the map $[\sigma] \otimes v_1 \otimes \cdots \otimes v_n \mapsto v_{\sigma^{-1}(1)} \otimes \cdots \otimes v_{\sigma^{-1}(n)}$, checking that this descends to the quotient. Summing over $n \geq 0$ gives $F_{\mathrm{Ass}}(V) = \bigoplus_{n \geq 0} V^{\otimes n} = T(V)$, the tensor algebra.

> [!QUESTION] Exercise 2: Functoriality of SymSeq
> *This problem verifies that SymSeq is a well-defined category with composition of morphisms.*
>
> > **Prerequisites:** [[#1.1 The Category SymSeq|1.1 The Category SymSeq]]
>
> Let $f \colon \mathcal{L} \to \mathcal{M}$ and $g \colon \mathcal{M} \to \mathcal{N}$ be morphisms of symmetric sequences. Show that $g \circ f$ (componentwise composition) is again a morphism of symmetric sequences.

> [!TIP]- Solution to Exercise 2
> **Key insight:** Equivariance is preserved under composition of equivariant maps.
>
> **Sketch:** For $\ell \in \mathcal{L}(n)$ and $\sigma \in S_n$: $(g_n \circ f_n)(\ell \cdot \sigma) = g_n(f_n(\ell \cdot \sigma)) = g_n(f_n(\ell) \cdot \sigma) = g_n(f_n(\ell)) \cdot \sigma$, using equivariance of $f_n$ then $g_n$.

---

## 2. The Composition Product

📐 The central monoidal structure on $\mathbf{SymSeq}$ is the *composition product* (or *substitution product*) $\circ$, which formalizes the idea of "plugging the outputs of one family of operations into the inputs of another."

### 2.1 Definition of the Substitution Product

We first define the $k$-th *tensor power* of a symmetric sequence $\mathcal{P}$ as another symmetric sequence:

**Definition 2.1 (Tensor Power of a Symmetric Sequence).** For $\mathcal{P} \in \mathbf{SymSeq}$ and $k \geq 0$, define $\mathcal{P}^{\otimes k} \in \mathbf{SymSeq}$ by
$$\mathcal{P}^{\otimes k}(n) := \bigoplus_{\substack{n_1, \ldots, n_k \geq 0 \\ n_1 + \cdots + n_k = n}} \mathcal{P}(n_1) \otimes_k \mathcal{P}(n_2) \otimes_k \cdots \otimes_k \mathcal{P}(n_k).$$
The right $S_n$-action on $\mathcal{P}^{\otimes k}(n)$ is by *block permutations*: the summand indexed by $(n_1, \ldots, n_k)$ is permuted to the summand indexed by $(n_{\sigma^{-1}(1)}, \ldots, n_{\sigma^{-1}(k)})$ for $\sigma \in S_n$, together with the induced internal permutations. More precisely, define the *block permutation* $\mathrm{bl}(n_1,\ldots,n_k;\sigma) \in S_n$ for $\sigma \in S_k$ as the permutation that maps the $j$-th block of size $n_j$ to the $\sigma(j)$-th block, preserving relative order within each block. Then
$$(p_1 \otimes \cdots \otimes p_k) \cdot \sigma := (p_{\sigma^{-1}(1)} \otimes \cdots \otimes p_{\sigma^{-1}(k)}) \cdot \mathrm{bl}(n_{\sigma^{-1}(1)}, \ldots, n_{\sigma^{-1}(k)}; \sigma)$$
for $p_j \in \mathcal{P}(n_j)$.

> [!WARNING] Convention Conflict
> Some authors define $\mathcal{P}^{\otimes k}(n)$ with the direct sum over *ordered* compositions $n_1 + \cdots + n_k = n$ of non-negative integers. Others use *positive* integers (excluding $n_i = 0$). The two conventions lead to slightly different composition products and different notions of "unitary" operad. We follow Loday–Vallette and allow $n_i \geq 0$.

**Definition 2.2 (Composition Product).** For $\mathcal{O}, \mathcal{P} \in \mathbf{SymSeq}$, the *composition product* $\mathcal{O} \circ \mathcal{P} \in \mathbf{SymSeq}$ is defined by
$$(\mathcal{O} \circ \mathcal{P})(n) := \bigoplus_{k \geq 0} \mathcal{O}(k) \otimes_{S_k} \mathcal{P}^{\otimes k}(n).$$
Explicitly, using the distributivity of $\otimes$ over $\oplus$:
$$(\mathcal{O} \circ \mathcal{P})(n) = \bigoplus_{k \geq 0} \mathcal{O}(k) \otimes_{S_k} \left( \bigoplus_{\substack{n_1, \ldots, n_k \geq 0 \\ n_1 + \cdots + n_k = n}} \mathcal{P}(n_1) \otimes \cdots \otimes \mathcal{P}(n_k) \right).$$

The coinvariant quotient $\otimes_{S_k}$ identifies $(\mu \cdot \sigma) \otimes (p_1 \otimes \cdots \otimes p_k)$ with $\mu \otimes (p_{\sigma^{-1}(1)} \otimes \cdots \otimes p_{\sigma^{-1}(k)})$ for $\sigma \in S_k$. The right $S_n$-action on $(\mathcal{O} \circ \mathcal{P})(n)$ is induced from the block-permutation action on $\mathcal{P}^{\otimes k}(n)$.

> [!EXAMPLE] Explicit Composition at Low Arities
> Consider the arity-2 component $(\mathcal{O} \circ \mathcal{P})(2)$. The relevant $k$-values contribute:
> - $k = 0$: $\mathcal{O}(0) \otimes_{S_0} \mathcal{P}^{\otimes 0}(2)$. Since $\mathcal{P}^{\otimes 0} = \mathbf{1}$ (the monoidal unit), this is $\mathcal{O}(0) \otimes k = \mathcal{O}(0)$ when $n = 0$, but $0$ when $n = 2$ (as $\mathbf{1}(2) = 0$). So this term vanishes.
> - $k = 1$: $\mathcal{O}(1) \otimes_{S_1} \mathcal{P}(2) = \mathcal{O}(1) \otimes \mathcal{P}(2)$ (since $S_1$ is trivial).
> - $k = 2$: $\mathcal{O}(2) \otimes_{S_2} (\mathcal{P}(1) \otimes \mathcal{P}(1) \oplus \mathcal{P}(0) \otimes \mathcal{P}(2) \oplus \mathcal{P}(2) \otimes \mathcal{P}(0))$.
> - $k = 3$: Only the summand $n_1 + n_2 + n_3 = 2$ with all $n_i \geq 0$; the terms $\mathcal{O}(3) \otimes_{S_3} (\mathcal{P}(0) \otimes \mathcal{P}(0) \otimes \mathcal{P}(2) \oplus \cdots)$.
>
> If $\mathcal{P}(0) = 0$ (reduced case), the series truncates: $(\mathcal{O} \circ \mathcal{P})(2) = \mathcal{O}(1) \otimes \mathcal{P}(2) \oplus (\mathcal{O}(2) \otimes_{S_2} \mathcal{P}(1)^{\otimes 2})$.

### 2.2 Monoidal Structure on SymSeq

🔑 The composition product makes $\mathbf{SymSeq}$ into a monoidal category.

**Proposition 2.3 (Monoidal Structure).** $(\mathbf{SymSeq}, \circ, \mathbf{1})$ is a monoidal category. Specifically:
1. *(Associativity)* There is a natural isomorphism $(\mathcal{O} \circ \mathcal{P}) \circ \mathcal{Q} \cong \mathcal{O} \circ (\mathcal{P} \circ \mathcal{Q})$.
2. *(Left unit)* There is a natural isomorphism $\mathbf{1} \circ \mathcal{O} \cong \mathcal{O}$.
3. *(Right unit)* There is a natural isomorphism $\mathcal{O} \circ \mathbf{1} \cong \mathcal{O}$.

*Proof sketch of associativity.* Both $(\mathcal{O} \circ \mathcal{P}) \circ \mathcal{Q}$ and $\mathcal{O} \circ (\mathcal{P} \circ \mathcal{Q})$ at arity $n$ can be identified as direct sums indexed by *planar rooted trees* with $n$ leaves: one sums over all ways to partition $[n]$ into $k$ groups, then each group into $k_i$ sub-groups, and so on. The coinvariant quotients and block permutations assemble consistently, giving a natural isomorphism. More precisely, one verifies on elements: an element of $((\mathcal{O} \circ \mathcal{P}) \circ \mathcal{Q})(n)$ is an equivalence class $[\mu \otimes (\nu_1, \ldots, \nu_j) \otimes (\rho_1, \ldots, \rho_{k_1+\cdots+k_j})]$ with $\mu \in \mathcal{O}(j)$, $\nu_i \in \mathcal{P}(k_i)$, $\rho_\ell \in \mathcal{Q}(n_\ell)$; this manifestly equals an element of $(\mathcal{O} \circ (\mathcal{P} \circ \mathcal{Q}))(n)$ via the isomorphism $\mathcal{P}(k_i) \otimes_{S_{k_i}} \mathcal{Q}^{\otimes k_i} \cong (\mathcal{P} \circ \mathcal{Q})(\sum_\ell n_{i,\ell})$.

*Proof of unit laws.* For the left unit, $(\mathbf{1} \circ \mathcal{O})(n) = \bigoplus_{k} \mathbf{1}(k) \otimes_{S_k} \mathcal{O}^{\otimes k}(n)$. Since $\mathbf{1}(k) = 0$ for $k \neq 1$ and $\mathbf{1}(1) = k$, only the $k = 1$ term survives: $k \otimes_{S_1} \mathcal{O}(n) = \mathcal{O}(n)$. For the right unit, $(\mathcal{O} \circ \mathbf{1})(n) = \bigoplus_k \mathcal{O}(k) \otimes_{S_k} \mathbf{1}^{\otimes k}(n)$. Now $\mathbf{1}^{\otimes k}(n) = 0$ unless $k = n$ (since $\mathbf{1}(n_i) \neq 0$ iff $n_i = 1$, and $n_1 + \cdots + n_k = n$ with all $n_i = 1$ iff $k = n$), in which case $\mathbf{1}^{\otimes k}(k) = k^{\otimes k} = k$ with $S_k$ acting trivially. Thus $(\mathcal{O} \circ \mathbf{1})(n) = \mathcal{O}(n) \otimes_{S_n} k = \mathcal{O}(n) / \langle m \cdot \sigma - m \rangle$. Wait — this is the coinvariant quotient, which is *not* $\mathcal{O}(n)$ unless the $S_n$-action is trivial. The resolution is that the $k = n$ summand carries $S_n$ acting by block permutations on $\mathbf{1}^{\otimes n}(n) = k$, and since the $S_n$-action on $k$ is through the isomorphism $\sigma \mapsto 1$ (trivial on the $k$ factor), the coinvariant quotient is $\mathcal{O}(n) \otimes_{S_n} (k[S_n] \cdot 1)$. More carefully: $\mathbf{1}^{\otimes n}(n) \cong k[S_n]$ as a right $S_n$-module (by block permutations, which give all permutations since all blocks have size 1), and $\mathcal{O}(n) \otimes_{S_n} k[S_n] \cong \mathcal{O}(n)$ via the canonical isomorphism $M \otimes_G k[G] \cong M$. $\square$

> [!NOTE] Schur Functor Perspective
> The isomorphism $\mathcal{O} \circ (\mathcal{P} \circ \mathcal{Q}) \cong (\mathcal{O} \circ \mathcal{P}) \circ \mathcal{Q}$ is exactly the statement that the Schur functor $F_{\mathcal{O} \circ \mathcal{P}} = F_\mathcal{O} \circ F_\mathcal{P}$ (composition of functors), and functor composition is associative. This gives a slick proof but conceals the combinatorial content.

### 2.3 Non-Symmetry of the Composition Product

⚠️ The composition product $\circ$ is *not* symmetric: in general $\mathcal{O} \circ \mathcal{P} \not\cong \mathcal{P} \circ \mathcal{O}$.

**Example 2.4.** Let $\mathcal{O}(n) = k$ for all $n$ (trivial $S_n$-action) and $\mathcal{P}(1) = k$, $\mathcal{P}(n) = 0$ for $n \neq 1$. Then:
$$(\mathcal{O} \circ \mathcal{P})(n) = \bigoplus_k \mathcal{O}(k) \otimes_{S_k} \mathcal{P}^{\otimes k}(n) = \mathcal{O}(n) \otimes_{S_n} \mathcal{P}(1)^{\otimes n} = k \otimes_{S_n} k^{\otimes n} = k.$$
$$(\mathcal{P} \circ \mathcal{O})(n) = \bigoplus_k \mathcal{P}(k) \otimes_{S_k} \mathcal{O}^{\otimes k}(n) = \mathcal{P}(1) \otimes_{S_1} \mathcal{O}(n) = k \otimes \mathcal{O}(n) = k.$$
In this case they agree, but consider instead $\mathcal{O}(2) = k$, $\mathcal{O}(n) = 0$ otherwise (binary $\mathcal{O}$), and $\mathcal{P}(3) = k$, $\mathcal{P}(n) = 0$ otherwise (ternary $\mathcal{P}$). Then $(\mathcal{O} \circ \mathcal{P})(6) = \mathcal{O}(2) \otimes_{S_2} \mathcal{P}(3)^{\otimes 2} = k \otimes_{S_2} k = k$ (one element), but $(\mathcal{P} \circ \mathcal{O})(5) = \mathcal{P}(3) \otimes_{S_3} \mathcal{O}(2)^{\otimes 3}$, which concentrates at arity 5, not 6. More starkly: $(\mathcal{P} \circ \mathcal{O})(6) = \mathcal{P}(3) \otimes_{S_3} (\mathcal{O}(2)^{\otimes 3})(6) = k \otimes_{S_3} k = k$, but $(\mathcal{O} \circ \mathcal{P})(7) = 0 \neq (\mathcal{P} \circ \mathcal{O})(7)$. The dimensions disagree across arities, so no graded isomorphism exists.

> [!WARNING] Consequences for Algebras
> The non-symmetry of $\circ$ means that the category of operads (monoids in $\mathbf{SymSeq}$) is not symmetric monoidal; there is no general tensor product of operads in the same sense as tensor product of rings. One must instead work with the Hadamard product or other ad-hoc constructions.

> [!QUESTION] Exercise 3: Composition at Arity 2
> *This problem develops facility with the composition product formula.*
>
> > **Prerequisites:** [[#2.1 Definition of the Substitution Product|2.1 Definition of the Substitution Product]]
>
> Let $\mathcal{O}(n) = k$ for all $n \geq 0$ (trivial $S_n$-action) and $\mathcal{P}(2) = k$, $\mathcal{P}(n) = 0$ for $n \neq 2$. Compute $(\mathcal{O} \circ \mathcal{P})(n)$ for $n = 0, 1, 2, 3, 4$ explicitly.

> [!TIP]- Solution to Exercise 3
> **Key insight:** Only the terms with all $n_i = 2$ (giving $k$ operations each of arity 2) can contribute.
>
> **Sketch:** $(\mathcal{O} \circ \mathcal{P})(n) = \bigoplus_k \mathcal{O}(k) \otimes_{S_k} \mathcal{P}^{\otimes k}(n)$. Since $\mathcal{P}(m) = 0$ unless $m = 2$, the only non-vanishing summand in $\mathcal{P}^{\otimes k}(n)$ is when $n_1 = \cdots = n_k = 2$, so $n = 2k$. Thus $(\mathcal{O} \circ \mathcal{P})(n) = 0$ for odd $n$, and for even $n = 2k$ it equals $\mathcal{O}(k) \otimes_{S_k} k = k \otimes_{S_k} k = k/(\text{trivial action}) = k$. In particular: $(\mathcal{O} \circ \mathcal{P})(0) = k$ (from $k=0$ term), $(\mathcal{O} \circ \mathcal{P})(1) = 0$, $(\mathcal{O} \circ \mathcal{P})(2) = k$, $(\mathcal{O} \circ \mathcal{P})(3) = 0$, $(\mathcal{O} \circ \mathcal{P})(4) = k$.

> [!QUESTION] Exercise 4: Non-Symmetry Proof
> *This problem proves the non-symmetry of* $\circ$ *via an explicit counterexample.*
>
> > **Prerequisites:** [[#2.3 Non-Symmetry of the Composition Product|2.3 Non-Symmetry of the Composition Product]]
>
> Let $\mathrm{Com}(n) = k$ (trivial action) and $\mathrm{Ass}(n) = k[S_n]$ (regular right action). Compute $(\mathrm{Com} \circ \mathrm{Ass})(2)$ and $(\mathrm{Ass} \circ \mathrm{Com})(2)$. Show they are isomorphic as $k$-modules but that the $S_2$-actions differ, ruling out an equivariant isomorphism.

> [!TIP]- Solution to Exercise 4
> **Key insight:** The $S_2$-actions on the two products permute elements differently.
>
> **Sketch:** $(\mathrm{Com} \circ \mathrm{Ass})(2) = \bigoplus_k k \otimes_{S_k} k[S_k]^{\otimes k}(2)$. For $k = 1$: $k \otimes k[S_1](2) = 0$ (since $k[S_1](2) = 0$). For $k = 2$: $k \otimes_{S_2} (k[S_1] \otimes k[S_1] \oplus k[S_2] \otimes k[S_2] \otimes \cdots)$. With $\mathrm{Ass}(1) = k[S_1] = k$ and $n_1 + n_2 = 2$, the partition $(1,1)$ gives $k[S_1] \otimes k[S_1] = k$, and $S_2$ acts by swapping, so the coinvariant is $k$. For $(\mathrm{Ass} \circ \mathrm{Com})(2)$: $k[S_1] \otimes k \oplus k[S_2] \otimes_{S_2} k^{\otimes 2}(2)$. The $k=1$ piece gives $k[S_1] \otimes_{S_1} k(2) = 0$; the $k=2$ piece gives $k[S_2] \otimes_{S_2} k = k[S_2] / \langle\sigma v - v\rangle = k$. As $k$-modules both are $k$; but the $S_2$-action on the first comes from conjugation on $k[S_2]$ while the second has trivial action inherited from $\mathrm{Com}$, so they are not equivariantly isomorphic.

---

## 3. Operads as Monoids

### 3.1 Monoid Definition

🔑 The cleanest modern definition packages all the structure of an operad into a single algebraic object.

**Definition 3.1 (Operad).** An *operad* (over $k$) is a *monoid* in the monoidal category $(\mathbf{SymSeq}, \circ, \mathbf{1})$. Explicitly, it is a triple $(\mathcal{O}, \gamma, \eta)$ consisting of:
- A symmetric sequence $\mathcal{O} \in \mathbf{SymSeq}$,
- A *composition map* $\gamma \colon \mathcal{O} \circ \mathcal{O} \to \mathcal{O}$ (morphism of symmetric sequences),
- A *unit map* $\eta \colon \mathbf{1} \to \mathcal{O}$ (morphism of symmetric sequences),

subject to the following axioms, expressed as commutative diagrams:

**Associativity:** The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
  (\mathcal{O} \circ \mathcal{O}) \circ \mathcal{O} \arrow[r, "\alpha"] \arrow[d, "\gamma \circ \mathrm{id}"'] & \mathcal{O} \circ (\mathcal{O} \circ \mathcal{O}) \arrow[d, "\mathrm{id} \circ \gamma"] \\
  \mathcal{O} \circ \mathcal{O} \arrow[r, "\gamma"'] & \mathcal{O}
\end{tikzcd}
\end{document}
```

where $\alpha \colon (\mathcal{O} \circ \mathcal{O}) \circ \mathcal{O} \xrightarrow{\sim} \mathcal{O} \circ (\mathcal{O} \circ \mathcal{O})$ is the associativity constraint of $(\mathbf{SymSeq}, \circ, \mathbf{1})$.

**Left and Right Unit Laws:** The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
  \mathbf{1} \circ \mathcal{O} \arrow[r, "\eta \circ \mathrm{id}"] \arrow[dr, "\lambda"'] & \mathcal{O} \circ \mathcal{O} \arrow[d, "\gamma"] & \mathcal{O} \circ \mathbf{1} \arrow[l, "\mathrm{id} \circ \eta"'] \arrow[dl, "\rho"] \\
  & \mathcal{O} &
\end{tikzcd}
\end{document}
```

where $\lambda \colon \mathbf{1} \circ \mathcal{O} \xrightarrow{\sim} \mathcal{O}$ and $\rho \colon \mathcal{O} \circ \mathbf{1} \xrightarrow{\sim} \mathcal{O}$ are the unit isomorphisms from Proposition 2.3.

The unit map $\eta \colon \mathbf{1} \to \mathcal{O}$ picks out a distinguished element $\mathrm{id} := \eta(1_k) \in \mathcal{O}(1)$, called the *identity operation*.

In terms of components, $\gamma$ at arity $n$ gives:
$$\gamma_n \colon (\mathcal{O} \circ \mathcal{O})(n) = \bigoplus_{k \geq 0} \mathcal{O}(k) \otimes_{S_k} \mathcal{O}^{\otimes k}(n) \longrightarrow \mathcal{O}(n).$$
Restricting to the summand where $k$ is fixed and $n_1 + \cdots + n_k = n$, this gives a family of maps
$$\mathcal{O}(k) \otimes_{S_k} (\mathcal{O}(n_1) \otimes \cdots \otimes \mathcal{O}(n_k)) \longrightarrow \mathcal{O}(n),$$
which we write as $(\mu; \nu_1, \ldots, \nu_k) \mapsto \gamma(\mu; \nu_1, \ldots, \nu_k)$.

> [!NOTE] Operadic Intuition
> An element $\mu \in \mathcal{O}(k)$ is an abstract $k$-ary operation. The composition $\gamma(\mu; \nu_1, \ldots, \nu_k)$ represents feeding the output of $\nu_i \in \mathcal{O}(n_i)$ into the $i$-th input of $\mu$, producing an operation of arity $n_1 + \cdots + n_k$. The $S_k$-coinvariant in the formula accounts for the fact that permuting how we label the $\nu_i$'s and permuting the $S_k$-action on $\mu$ simultaneously gives the same composition.

### 3.2 Partial Composition Formulation

💡 The monoid definition packages all compositions simultaneously. An equivalent formulation works with *binary* partial compositions, which are often easier to verify axiom by axiom.

**Definition 3.2 (Partial Composition).** Given a symmetric sequence $\mathcal{O}$, a *partial composition structure* consists of $k$-linear maps
$$\circ_i \colon \mathcal{O}(m) \otimes \mathcal{O}(n) \longrightarrow \mathcal{O}(m + n - 1)$$
for all $m, n \geq 1$ and $1 \leq i \leq m$, satisfying the following axioms. Write $\mu \circ_i \nu$ for the image of $\mu \otimes \nu$:

**Axiom PC1 (Sequential Associativity).** For $1 \leq i \leq m$ and $1 \leq j \leq n$:
$$(\mu \circ_i \nu) \circ_{i + j - 1} \rho = \mu \circ_i (\nu \circ_j \rho).$$
This says: first plug $\nu$ into the $i$-th slot of $\mu$, then plug $\rho$ into the $j$-th slot of the result (which has become the $(i+j-1)$-th slot of the whole) equals first composing $\rho$ into $\nu$ at position $j$, then plugging $\nu \circ_j \rho$ into $\mu$ at position $i$.

**Axiom PC2 (Parallel Associativity).** For $1 \leq j < i \leq m$ and $\rho \in \mathcal{O}(p)$:
$$(\mu \circ_i \nu) \circ_j \rho = (\mu \circ_j \rho) \circ_{i + p - 1} \nu.$$
This says: plugging into two *different* input slots of $\mu$ commutes (up to index shift).

**Axiom PC3 (Equivariance).** The partial compositions are equivariant with respect to the symmetric group actions. Precisely, for $\sigma \in S_m$, $\tau \in S_n$:
$$(\mu \cdot \sigma) \circ_{\sigma^{-1}(i)} (\nu \cdot \tau) = (\mu \circ_i \nu) \cdot (\sigma \circ_i \tau)$$
where $\sigma \circ_i \tau \in S_{m+n-1}$ is the *insertion permutation*: the element that permutes $\{1, \ldots, m+n-1\}$ by letting $\sigma$ act on the $m$ "block positions" (with the $i$-th block expanded to $n$ elements by $\tau$). Concretely, for $s \in \{1,\ldots,m+n-1\}$, define
$$(\sigma \circ_i \tau)(s) := \begin{cases} \sigma(j)(= \sigma(j)) + (\text{block-size adjustment}) & \text{if } s \text{ is in the } j\text{-th block and } j \neq i \\ \text{position of } \tau(s - i + 1) \text{ in the }\sigma(i)\text{-th block} & \text{if } s \text{ is in the }i\text{-th block.} \end{cases}$$

> [!NOTE] Explicit Formula for the Insertion Permutation
> More precisely, for $\sigma \in S_m$ and $\tau \in S_n$, the permutation $\sigma \circ_i \tau \in S_{m+n-1}$ acts on $\{1, \ldots, m+n-1\}$ as follows. Number the inputs of $\mu \circ_i \nu$ from $1$ to $m+n-1$ by replacing the $i$-th input with $n$ new inputs labeled $i, i+1, \ldots, i+n-1$. Then:
> - Input $s$ with $s < i$: sent to position $\sigma(s)$ adjusted for block sizes
> - Inputs $s \in \{i, \ldots, i+n-1\}$: sent to position $\tau(s-i+1)$ within the $\sigma(i)$-th block
> - Input $s$ with $s > i+n-1$: sent to block $\sigma(s-n+1)$, adjusted

**Unit axiom:** There exists a distinguished element $\mathrm{id} \in \mathcal{O}(1)$ such that $\mu \circ_i \mathrm{id} = \mu$ for all $\mu \in \mathcal{O}(m)$ and $1 \leq i \leq m$, and $\mathrm{id} \circ_1 \mu = \mu$ for all $\mu \in \mathcal{O}(n)$.

### 3.3 Equivalence of the Two Formulations

**Proposition 3.3.** The data of a monoid $(\mathcal{O}, \gamma, \eta)$ in $(\mathbf{SymSeq}, \circ, \mathbf{1})$ is equivalent to a symmetric sequence with a partial composition structure $(\mathcal{O}, \{\circ_i\}, \mathrm{id})$ satisfying axioms PC1, PC2, PC3, and the unit axiom.

*Proof.* Given $(\mathcal{O}, \gamma, \eta)$:

**Monoid $\Rightarrow$ Partial compositions.** Define $\mu \circ_i \nu$ for $\mu \in \mathcal{O}(m)$, $\nu \in \mathcal{O}(n)$ by
$$\mu \circ_i \nu := \gamma(\mu; \underbrace{\mathrm{id}, \ldots, \mathrm{id}}_{i-1}, \nu, \underbrace{\mathrm{id}, \ldots, \mathrm{id}}_{m-i})$$
where $\mathrm{id} = \eta(1) \in \mathcal{O}(1)$. This is the composition of $\mu$ with $\nu$ in the $i$-th slot and the identity in all other slots.

**Partial compositions $\Rightarrow$ Monoid.** Given partial compositions, reconstruct the full composition by:
$$\gamma(\mu; \nu_1, \ldots, \nu_k) := (\cdots((\mu \circ_k \nu_k) \circ_{k-1} \nu_{k-1}) \cdots) \circ_1 \nu_1.$$
One must verify this is well-defined (independent of the order in which we compose), which follows from axioms PC1 and PC2. Axiom PC3 ensures the result is $S_n$-equivariant.

The two constructions are inverse: the monoid axioms (associativity, unit) correspond precisely to PC1, PC2, PC3. The full verification is a straightforward but lengthy diagram chase; we refer to Loday–Vallette Proposition 5.3.4. $\square$

> [!EXAMPLE] Verifying PC2 from the Monoid Axioms
> From the monoid definition, $(\mu \circ_i \nu) \circ_j \rho$ for $j < i$ is
> $$\gamma(\mu; \mathrm{id}, \ldots, \rho, \ldots, \mathrm{id}, \ldots, \nu, \ldots, \mathrm{id})$$
> where $\rho$ occupies position $j$ and $\nu$ occupies position $i$. By the symmetry of $\gamma$ (it only depends on the $S_k$-coinvariant class), this equals $\gamma(\mu; \mathrm{id}, \ldots, \nu, \ldots, \rho, \ldots, \mathrm{id})$ composed with the appropriate block permutation, which is exactly $(\mu \circ_j \rho) \circ_{i+p-1} \nu$ after reindexing.

> [!QUESTION] Exercise 5: Partial Compositions and Sequential Associativity
> *This problem verifies axiom PC1 from the monoid associativity diagram.*
>
> > **Prerequisites:** [[#3.1 Monoid Definition|3.1 Monoid Definition]], [[#3.2 Partial Composition Formulation|3.2 Partial Composition Formulation]]
>
> Using the definition $\mu \circ_i \nu = \gamma(\mu; \mathrm{id}, \ldots, \nu, \ldots, \mathrm{id})$ with $\nu$ in position $i$, verify that the monoid associativity axiom implies PC1: $(\mu \circ_i \nu) \circ_{i+j-1} \rho = \mu \circ_i (\nu \circ_j \rho)$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** Both sides equal $\gamma(\mu; \mathrm{id}, \ldots, \gamma(\nu; \mathrm{id}, \ldots, \rho, \ldots, \mathrm{id}), \ldots, \mathrm{id})$ by applying associativity.
>
> **Sketch:** LHS: $(\mu \circ_i \nu) \circ_{i+j-1} \rho = \gamma(\mu \circ_i \nu; \mathrm{id}, \ldots, \rho, \ldots, \mathrm{id})$ with $\rho$ in position $i+j-1$. Substituting $\mu \circ_i \nu = \gamma(\mu; \mathrm{id},\ldots,\nu,\ldots,\mathrm{id})$ and expanding using associativity of $\gamma$ gives $\gamma(\mu; \mathrm{id},\ldots, \gamma(\nu;\mathrm{id},\ldots,\rho,\ldots,\mathrm{id}),\ldots,\mathrm{id})$ with $\rho$ in the $j$-th slot of $\nu$. RHS: $\mu \circ_i (\nu \circ_j \rho) = \gamma(\mu;\mathrm{id},\ldots,\nu\circ_j\rho,\ldots,\mathrm{id}) = \gamma(\mu;\mathrm{id},\ldots,\gamma(\nu;\mathrm{id},\ldots,\rho,\ldots,\mathrm{id}),\ldots,\mathrm{id})$. These are identical.

> [!QUESTION] Exercise 6: The Unit Element is Unique
> *This problem shows that the identity operation is determined by the operad structure.*
>
> > **Prerequisites:** [[#3.2 Partial Composition Formulation|3.2 Partial Composition Formulation]]
>
> Suppose $(\mathcal{O}, \{\circ_i\}, \mathrm{id})$ and $(\mathcal{O}, \{\circ_i\}, \mathrm{id}')$ are two partial-composition structures on the same $\mathcal{O}$ with the same $\circ_i$ maps. Show that $\mathrm{id} = \mathrm{id}'$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** Standard monoid argument: two units in a monoid must coincide.
>
> **Sketch:** $\mathrm{id} = \mathrm{id} \circ_1 \mathrm{id}' = \mathrm{id}'$, using the left unit axiom ($\mathrm{id} \circ_1 \mathrm{id}' = \mathrm{id}'$) and the right unit axiom ($\mathrm{id} \circ_1 \mathrm{id}' = \mathrm{id}$).

---

## 4. Morphisms and the Category Op

**Definition 4.1 (Operad Morphism).** A *morphism of operads* $f \colon (\mathcal{O}, \gamma_\mathcal{O}, \eta_\mathcal{O}) \to (\mathcal{P}, \gamma_\mathcal{P}, \eta_\mathcal{P})$ is a morphism of symmetric sequences $f = \{f_n\}_{n \geq 0}$ (each $f_n \colon \mathcal{O}(n) \to \mathcal{P}(n)$ being $S_n$-equivariant) such that:

1. *(Compatibility with composition)* The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
  \mathcal{O} \circ \mathcal{O} \arrow[r, "f \circ f"] \arrow[d, "\gamma_\mathcal{O}"'] & \mathcal{P} \circ \mathcal{P} \arrow[d, "\gamma_\mathcal{P}"] \\
  \mathcal{O} \arrow[r, "f"'] & \mathcal{P}
\end{tikzcd}
\end{document}
```

where $(f \circ f)$ denotes the induced map on the composition product.

2. *(Compatibility with unit)* The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
  \mathbf{1} \arrow[r, "\eta_\mathcal{O}"] \arrow[dr, "\eta_\mathcal{P}"'] & \mathcal{O} \arrow[d, "f"] \\
  & \mathcal{P}
\end{tikzcd}
\end{document}
```

In terms of partial compositions, $f$ is an operad morphism iff $f_n$ is $S_n$-equivariant for all $n$, $f_1(\mathrm{id}_\mathcal{O}) = \mathrm{id}_\mathcal{P}$, and $f_{m+n-1}(\mu \circ_i \nu) = f_m(\mu) \circ_i f_n(\nu)$ for all $\mu \in \mathcal{O}(m)$, $\nu \in \mathcal{O}(n)$, $1 \leq i \leq m$.

**Definition 4.2 (Category of Operads).** The *category of operads* $\mathbf{Op}$ (over $k$) has objects the operads and morphisms the operad morphisms. Composition is given componentwise.

**Remark 4.3.** By the general theory of monoids in monoidal categories, $\mathbf{Op}$ is precisely the category of monoids in $(\mathbf{SymSeq}, \circ, \mathbf{1})$, i.e., the full subcategory of the category of monoid objects in $\mathbf{SymSeq}$ spanned by those monoids whose underlying symmetric sequence structure is appropriately defined. The forgetful functor $U \colon \mathbf{Op} \to \mathbf{SymSeq}$ has a left adjoint (the free operad functor $\mathcal{F}$), giving rise to the free operad monad.

> [!NOTE] Free Operad
> For a symmetric sequence $\mathcal{M}$, the *free operad* $\mathcal{F}(\mathcal{M})$ is computed via the colimit of the sequence $\mathcal{M}, \mathcal{M} \circ \mathcal{M}, \mathcal{M} \circ \mathcal{M} \circ \mathcal{M}, \ldots$ with appropriate maps. As a $k$-module, $\mathcal{F}(\mathcal{M})(n)$ is spanned by all planar rooted trees with $n$ leaves, where each internal vertex of valence $k$ is decorated by an element of $\mathcal{M}(k)$. This tree description underlies all generator-and-relation presentations of operads (see Section 5).

> [!QUESTION] Exercise 7: Operad Morphisms and Algebra Homomorphisms
> *This problem shows that operad morphisms induce functors between algebra categories.*
>
> > **Prerequisites:** [[#4. Morphisms and the Category Op|4. Morphisms and the Category Op]]
>
> Let $f \colon \mathcal{O} \to \mathcal{P}$ be a morphism of operads and $A$ a $\mathcal{P}$-algebra (i.e., a $k$-module with a morphism $\rho \colon \mathcal{P} \to \mathrm{End}_A$). Show that $A$ acquires an $\mathcal{O}$-algebra structure via $\rho \circ f \colon \mathcal{O} \to \mathrm{End}_A$.

> [!TIP]- Solution to Exercise 7
> **Key insight:** Composition of operad morphisms is an operad morphism.
>
> **Sketch:** The composite $\rho \circ f \colon \mathcal{O} \to \mathrm{End}_A$ is a composite of two operad morphisms, hence an operad morphism by functoriality. The $S_n$-equivariance and compatibility with $\gamma$ and $\eta$ all follow from the corresponding properties of $f$ and $\rho$.

---

## 5. Key Examples

### 5.1 The Associative Operad Ass

**Definition 5.1 (Associative Operad).** The *associative operad* $\mathrm{Ass}$ is defined by
$$\mathrm{Ass}(n) := k[S_n], \quad n \geq 1, \qquad \mathrm{Ass}(0) := 0.$$
The $k$-module $k[S_n]$ is the *group algebra* of $S_n$: the free $k$-module on the set $S_n$, with basis $\{e_\sigma\}_{\sigma \in S_n}$. The right $S_n$-action is by right multiplication: $e_\sigma \cdot \tau = e_{\sigma\tau}$.

The unit is $\mathrm{id} = e_{\mathrm{id}_{S_1}} = e_e \in \mathrm{Ass}(1) = k[S_1] = k$.

**Composition in Ass.** The composition map
$$\gamma \colon \mathrm{Ass}(k) \otimes_{S_k} \mathrm{Ass}(n_1) \otimes \cdots \otimes \mathrm{Ass}(n_k) \longrightarrow \mathrm{Ass}(n_1 + \cdots + n_k)$$
is defined on basis elements by *block permutation concatenation*:
$$\gamma(e_\sigma; e_{\tau_1}, \ldots, e_{\tau_k}) := e_{\sigma\langle\tau_1,\ldots,\tau_k\rangle}$$
where $\sigma\langle\tau_1,\ldots,\tau_k\rangle \in S_n$ (with $n = n_1 + \cdots + n_k$) is the permutation obtained by:
1. Applying the block permutation $\mathrm{bl}(\sigma; n_1,\ldots,n_k)$ induced by $\sigma$: permute the $k$ blocks of sizes $n_1,\ldots,n_k$ according to $\sigma$, preserving internal order.
2. Within the $i$-th block (now at position $\sigma(i)$), apply the permutation $\tau_i$ internally.

Concretely: the $j$-th position within the $i$-th block of the output maps to the position determined by $\tau_i(j)$ within the $\sigma(i)$-th block of the target grouping. This is the *block-sum* of permutations.

> [!EXAMPLE] Explicit Computation in Ass(2)
> Take $k = 2$, $n_1 = n_2 = 1$, so $n = 2$. We have $\mathrm{Ass}(2) = k[S_2] = k \cdot e_{(12)} \oplus k \cdot e_e$ (where $(12)$ denotes the transposition). The composition:
> - $\gamma(e_e; e_e, e_e) = e_e$ (identity composed with identities = identity)
> - $\gamma(e_{(12)}; e_e, e_e) = e_{(12)}$ (swap the two blocks, each of size 1)
>
> For $k=2$, $n_1=1$, $n_2=2$ ($n=3$): $\gamma(e_e; e_e, e_{(12)}) = e_{(23)} \in S_3$ (the transposition $(23)$ acts within the second block). And $\gamma(e_{(12)}; e_e, e_{(12)}) = e_{(12)(23)} = e_{(123)} \in S_3$.

**Algebras over Ass.** A $\mathrm{Ass}$-algebra is a $k$-module $A$ with an operad morphism $\mathrm{Ass} \to \mathrm{End}_A$. The image of $e_\sigma \in \mathrm{Ass}(n) = k[S_n]$ is a linear map $A^{\otimes n} \to A$, and the images of $e_e \in \mathrm{Ass}(2)$ and $e_{(12)} \in \mathrm{Ass}(2)$ give the product and its "transposed" version. The operadic axioms force these to make $A$ into an *associative* algebra (with $e_e$ giving the product $\mu(a,b) = ab$ and $e_{(12)}$ giving $\mu(b,a) = ba$, so the full $k[S_2]$-action just tracks the ordering). Associativity of $\mu$ follows from the operad axioms.

> [!QUESTION] Exercise 8: Ass-Algebra Structure
> *This problem verifies that Ass-algebras are precisely associative algebras.*
>
> > **Prerequisites:** [[#5.1 The Associative Operad Ass|5.1 The Associative Operad Ass]]
>
> Let $A$ be a $k$-module with an operad morphism $f \colon \mathrm{Ass} \to \mathrm{End}_A$. Set $\mu := f_2(e_e) \colon A \otimes A \to A$. Using the operad morphism axioms and the composition rule in $\mathrm{Ass}$, show that $\mu \circ (\mu \otimes \mathrm{id}_A) = \mu \circ (\mathrm{id}_A \otimes \mu)$ (associativity of $\mu$).

> [!TIP]- Solution to Exercise 8
> **Key insight:** The composition $\gamma(e_e; e_e, e_e) = e_e$ in $\mathrm{Ass}(3)$ encodes associativity.
>
> **Sketch:** In $\mathrm{Ass}(3)$, we have $e_e \circ_1 e_e = \gamma(e_e; e_e, \mathrm{id}) = e_e \in \mathrm{Ass}(3)$ (the identity on 3 letters). Similarly $e_e \circ_2 e_e = e_e$. So $e_e \circ_1 e_e = e_e \circ_2 e_e$ in $\mathrm{Ass}(3)$. Applying $f_3$ and using $f$ is an operad morphism: $f_3(e_e \circ_1 e_e) = f_2(e_e) \circ_1 f_2(e_e) = \mu \circ (\mu \otimes \mathrm{id})$, and $f_3(e_e \circ_2 e_e) = \mu \circ (\mathrm{id} \otimes \mu)$. Since both equal $f_3(e_e)$, we get associativity.

### 5.2 The Commutative Operad Com

**Definition 5.2 (Commutative Operad).** The *commutative operad* $\mathrm{Com}$ is defined by
$$\mathrm{Com}(n) := k, \quad n \geq 1, \qquad \mathrm{Com}(0) := 0.$$
Each $\mathrm{Com}(n) = k$ carries the *trivial* right $S_n$-action: $1_k \cdot \sigma = 1_k$ for all $\sigma \in S_n$.

The unit is $\mathrm{id} = 1_k \in \mathrm{Com}(1) = k$.

**Composition in Com.** The composition map is
$$\gamma \colon \mathrm{Com}(k) \otimes_{S_k} \mathrm{Com}^{\otimes k}(n_1 + \cdots + n_k) \to \mathrm{Com}(n_1 + \cdots + n_k),$$
$$\gamma(1_k; 1_k, \ldots, 1_k) = 1_k.$$
That is, composition is multiplication of $1$'s — it is forced since every component is one-dimensional with trivial action. The coinvariant quotient $k \otimes_{S_k} k^{\otimes k} = k \otimes_{S_k} k = k$ (since $S_k$ acts trivially on both sides).

**Algebras over Com.** A $\mathrm{Com}$-algebra is an operad morphism $\mathrm{Com} \to \mathrm{End}_V$. Since $\mathrm{Com}(n) = k$ and the only map is determined by the image of $1_k \in \mathrm{Com}(n)$, we get a single linear map $\mu_n \colon V^{\otimes n} \to V$ for each arity $n$, with all $S_n$-equivariance and composition axioms. This forces: (1) $\mu_n$ is $S_n$-equivariant (hence symmetric/commutative), (2) $\mu_n$ is determined by $\mu_2$ via operadic composition, and (3) the binary operation is associative and commutative. So $\mathrm{Com}$-algebras are precisely *commutative (and associative) $k$-algebras*.

> [!TIP] Quotient Relationship
> There is a natural surjection $\mathrm{Ass}(n) = k[S_n] \twoheadrightarrow \mathrm{Com}(n) = k$ sending every basis element $e_\sigma \mapsto 1$. This is $S_n$-equivariant (since $S_n$ acts trivially on $k$) and compatible with compositions, giving an operad morphism $\mathrm{Ass} \to \mathrm{Com}$. The induced functor on algebras is the forgetful functor $\mathrm{CAlg}_k \to \mathrm{Alg}_k$ (from commutative to associative algebras).

### 5.3 The Lie Operad Lie

**Definition 5.3 (Lie Operad).** The *Lie operad* $\mathrm{Lie}$ is the operad encoding Lie algebras. It is most naturally defined via generators and relations. As a symmetric sequence:
$$\mathrm{Lie}(n) := \text{(free } k\text{-module on Lie monomials in } n \text{ variables)} / (\text{antisymmetry} + \text{Jacobi}),$$
with $S_n$ acting by permuting variables.

More precisely, define the *free Lie algebra* $\mathcal{L}(n)$ in $n$ generators $x_1, \ldots, x_n$ (over $k$) as the $k$-module spanned by all iterated brackets $[x_{i_1}, [\ldots, [x_{i_{n-1}}, x_{i_n}] \ldots]]$ subject to:
- **Antisymmetry:** $[a, b] = -[b, a]$, i.e., $[a,b] + [b,a] = 0$
- **Jacobi identity:** $[[a,b],c] + [[b,c],a] + [[c,a],b] = 0$

Then $\mathrm{Lie}(n) := \mathcal{L}(n)$. The symmetric group $S_n$ acts on $\mathrm{Lie}(n)$ by permuting the labels of the generators.

**Low-arity components:**
- $\mathrm{Lie}(1) = k$, generated by $x_1$ (the identity operation)
- $\mathrm{Lie}(2) = k \cdot [x_1, x_2] \cong k$ as a $k$-module, with $S_2 = \{e, (12)\}$ acting by $(12) \cdot [x_1,x_2] = [x_2,x_1] = -[x_1,x_2]$. So $\mathrm{Lie}(2) = k$ with the *sign* representation of $S_2$.
- $\mathrm{Lie}(3) = k^2$, generated by $[x_1,[x_2,x_3]]$ and $[x_2,[x_1,x_3]]$ (the third is determined by Jacobi).

**Composition in Lie.** The partial composition $\ell \circ_i \ell'$ for $\ell \in \mathrm{Lie}(m)$ and $\ell' \in \mathrm{Lie}(n)$ is defined by *substitution*: replace the $i$-th variable in $\ell$ by the Lie expression $\ell'(x_i, \ldots, x_{i+n-1})$, then relabel. This preserves the Lie relations (antisymmetry and Jacobi) by the universal property of free Lie algebras.

**Algebras over Lie.** A $\mathrm{Lie}$-algebra structure on a $k$-module $\mathfrak{g}$ is an operad morphism $\mathrm{Lie} \to \mathrm{End}_\mathfrak{g}$. The image of $[x_1, x_2] \in \mathrm{Lie}(2)$ gives a bilinear map $[-,-] \colon \mathfrak{g} \otimes \mathfrak{g} \to \mathfrak{g}$, and the operad axioms force antisymmetry and the Jacobi identity. So $\mathrm{Lie}$-algebras are precisely *Lie algebras*.

> [!NOTE] Dimension Formula
> By a theorem of Hall and Witt, $\dim_k \mathrm{Lie}(n) = \frac{1}{n} \sum_{d | n} \mu(n/d) \cdot d! \cdot \binom{d}{n/d \text{ ... }}$... more precisely, the generating function is $\sum_n \dim \mathrm{Lie}(n) \cdot t^n/n! = \ln(1/(1-t))$, where the right side is the exponential generating function of the sequence $0, 1, 1, 2, 6, 24, \ldots$ (i.e., $(n-1)!$ for $n \geq 1$). So $\dim_k \mathrm{Lie}(n) = (n-1)!$.

> [!QUESTION] Exercise 9: Dimension of Lie(3)
> *This problem verifies the dimension formula for the Lie operad at arity 3.*
>
> > **Prerequisites:** [[#5.3 The Lie Operad Lie|5.3 The Lie Operad Lie]]
>
> Show directly from the antisymmetry and Jacobi relations that $\dim_k \mathrm{Lie}(3) = 2 = (3-1)!$. Write down an explicit basis.

> [!TIP]- Solution to Exercise 9
> **Key insight:** The Jacobi identity expresses one nested bracket in terms of two others, leaving a 2-dimensional space.
>
> **Sketch:** The arity-3 Lie monomials are all nestings of brackets in $\{x_1, x_2, x_3\}$: $[x_1,[x_2,x_3]]$, $[x_2,[x_1,x_3]]$, $[x_3,[x_1,x_2]]$, $[[x_1,x_2],x_3]$, $[[x_1,x_3],x_2]$, $[[x_2,x_3],x_1]$. By antisymmetry, $[[a,b],c] = -[[b,a],c]$ etc., reducing to at most 3 generators (say, the first three above). By Jacobi: $[x_1,[x_2,x_3]] + [x_2,[x_3,x_1]] + [x_3,[x_1,x_2]] = 0$, or equivalently $[x_3,[x_1,x_2]] = -[x_1,[x_2,x_3]] - [x_2,[x_3,x_1]] = -[x_1,[x_2,x_3]] + [x_2,[x_1,x_3]]$. So the third is determined by the first two, giving a basis $\{[x_1,[x_2,x_3]], [x_2,[x_1,x_3]]\}$ and $\dim = 2$.

### 5.4 The Endomorphism Operad EndV

**Definition 5.4 (Endomorphism Operad).** For a $k$-module $V$, the *endomorphism operad* $\mathrm{End}_V$ is defined by
$$\mathrm{End}_V(n) := \mathrm{Hom}_k(V^{\otimes n}, V), \quad n \geq 0$$
with $\mathrm{End}_V(0) = \mathrm{Hom}_k(k, V) \cong V$. The right $S_n$-action on $\mathrm{Hom}_k(V^{\otimes n}, V)$ is by *precomposition with permutation of tensor factors*:
$$(f \cdot \sigma)(v_1 \otimes \cdots \otimes v_n) := f(v_{\sigma^{-1}(1)} \otimes \cdots \otimes v_{\sigma^{-1}(n)}).$$

The unit is $\mathrm{id}_V \in \mathrm{Hom}_k(V, V) = \mathrm{End}_V(1)$.

**Composition in $\mathrm{End}_V$.** The composition map
$$\gamma \colon \mathrm{End}_V(k) \otimes_{S_k} \mathrm{End}_V^{\otimes k}(n_1 + \cdots + n_k) \longrightarrow \mathrm{End}_V(n_1 + \cdots + n_k)$$
is defined for $f \in \mathrm{Hom}(V^{\otimes k}, V)$ and $g_i \in \mathrm{Hom}(V^{\otimes n_i}, V)$ by *substitution*:
$$\gamma(f; g_1, \ldots, g_k)(v_1 \otimes \cdots \otimes v_n) := f\bigl(g_1(v_1 \otimes \cdots \otimes v_{n_1}), g_2(v_{n_1+1} \otimes \cdots \otimes v_{n_1+n_2}), \ldots, g_k(v_{n-n_k+1} \otimes \cdots \otimes v_n)\bigr).$$
The coinvariant quotient by $S_k$ is compatible with this definition because permuting the $g_i$'s while adjusting $f$ by the corresponding permutation gives the same composite.

🔑 The endomorphism operad is the *universal* or *tautological* example: every operad structure on a $k$-module $V$ is an operad morphism to $\mathrm{End}_V$.

**Proposition 5.5.** Let $\mathcal{O}$ be an operad and $V$ a $k$-module. An *$\mathcal{O}$-algebra structure* on $V$ is precisely an operad morphism $\rho \colon \mathcal{O} \to \mathrm{End}_V$.

*Proof.* Given $\rho$, for each $n$ we have $\rho_n \colon \mathcal{O}(n) \to \mathrm{Hom}(V^{\otimes n}, V)$, which by adjunction gives a $k$-linear map $\mathcal{O}(n) \otimes V^{\otimes n} \to V$. The $S_n$-equivariance of $\rho_n$ means this factors through $\mathcal{O}(n) \otimes_{S_n} V^{\otimes n}$, giving an action map. The morphism conditions then ensure: (1) the identity operation $\mathrm{id} \in \mathcal{O}(1)$ acts as the identity on $V$; (2) the composition $\gamma$ in $\mathcal{O}$ is compatible with composition in $\mathrm{End}_V$ (i.e., the action of a composed operation is the composition of the actions). These are precisely the axioms of an $\mathcal{O}$-algebra. $\square$

> [!QUESTION] Exercise 10: Endomorphism Operad Axioms
> *This problem verifies that* $\mathrm{End}_V$ *satisfies the operad axioms.*
>
> > **Prerequisites:** [[#5.4 The Endomorphism Operad EndV|5.4 The Endomorphism Operad EndV]]
>
> Verify that the composition $\gamma$ in $\mathrm{End}_V$ is associative and unital. Specifically, show that $\gamma(\gamma(f;g_1,\ldots,g_k); h_{11},\ldots,h_{km_k}) = \gamma(f; \gamma(g_1;h_{11},\ldots,h_{1m_1}), \ldots, \gamma(g_k;h_{k1},\ldots,h_{km_k}))$.

> [!TIP]- Solution to Exercise 10
> **Key insight:** Both sides equal $f \circ (g_1 \otimes \cdots \otimes g_k) \circ (\text{all the }h_{ij})$, which is associativity of function composition.
>
> **Sketch:** LHS applies $f$ to the outputs of $\gamma(g_i; h_{i1},\ldots,h_{im_i})$ for each $i$. Each $\gamma(g_i; h_{i1},\ldots)$ is function composition (plug all $h_{ij}$'s into $g_i$). So LHS is $f$ applied to $g_1(\text{inputs from }h_{1j}\text{'s}), \ldots, g_k(\text{inputs from }h_{kj}\text{'s})$. RHS computes similarly. Both reduce to $f \circ (g_1 \circ (h_{11} \otimes \cdots) \otimes \cdots \otimes g_k \circ (h_{k1} \otimes \cdots))$, which is the same by associativity of $\circ$.

### 5.5 Operad Morphisms and Forgetful Functors

The sequence of operad morphisms
$$\mathrm{Lie} \longrightarrow \mathrm{Ass} \longrightarrow \mathrm{Com}$$
has the following explicit descriptions:

**$\mathrm{Lie} \to \mathrm{Ass}$:** The map $f \colon \mathrm{Lie}(n) \to \mathrm{Ass}(n) = k[S_n]$ sends a Lie monomial $\ell(x_1,\ldots,x_n)$ to the corresponding *antisymmetrized sum* in $k[S_n]$. At arity 2: $f([x_1,x_2]) = e_e - e_{(12)} \in k[S_2]$. This corresponds to the fact that the Lie bracket $[a,b] = ab - ba$ in an associative algebra. The map is $S_n$-equivariant by construction.

**$\mathrm{Ass} \to \mathrm{Com}$:** The map $g \colon \mathrm{Ass}(n) = k[S_n] \to \mathrm{Com}(n) = k$ is $g(e_\sigma) = 1$ for all $\sigma \in S_n$ (sum over all permutations, normalized). This is equivariant since $S_n$ acts trivially on $k$, and it is compatible with composition.

**Induced Functors.** Applying Proposition 5.5 (via postcomposition with $\mathrm{End}_V$): an operad morphism $\mathcal{O} \to \mathcal{P}$ induces a functor $\mathcal{P}\text{-}\mathbf{Alg} \to \mathcal{O}\text{-}\mathbf{Alg}$ (in the reverse direction: restricting structure along the morphism). Therefore:
- $\mathrm{Com} \to \mathrm{Ass}$: wrong direction. The morphism is $\mathrm{Ass} \to \mathrm{Com}$, inducing $\mathrm{Com}\text{-}\mathbf{Alg} \to \mathrm{Ass}\text{-}\mathbf{Alg}$, i.e., $\mathbf{CAlg}_k \hookrightarrow \mathbf{Alg}_k$ (every commutative algebra is, in particular, associative).
- $\mathrm{Lie} \to \mathrm{Ass}$: induces $\mathrm{Ass}\text{-}\mathbf{Alg} \to \mathrm{Lie}\text{-}\mathbf{Alg}$, i.e., for every associative algebra $A$, the commutator bracket $[a,b] = ab - ba$ makes $A$ a Lie algebra.

*Surprisingly,* this means the "inclusions" of algebra categories ($\mathbf{CAlg} \subset \mathbf{Alg} \subset \mathbf{LieAlg}$ — in the sense of forgetful functors going from more to less structure) correspond to operad morphisms going the *opposite* direction ($\mathrm{Lie} \to \mathrm{Ass} \to \mathrm{Com}$).

> [!QUESTION] Exercise 11: Lie to Ass at Arity 3
> *This problem explicitly computes the operad morphism* $\mathrm{Lie} \to \mathrm{Ass}$ *at arity 3.*
>
> > **Prerequisites:** [[#5.5 Operad Morphisms and Forgetful Functors|5.5 Operad Morphisms and Forgetful Functors]]
>
> Compute $f([x_1,[x_2,x_3]]) \in k[S_3]$ for the operad morphism $f \colon \mathrm{Lie} \to \mathrm{Ass}$. Express your answer as a sum of basis elements $e_\sigma$.

> [!TIP]- Solution to Exercise 11
> **Key insight:** Nested Lie brackets correspond to nested commutators in the group algebra.
>
> **Sketch:** $[x_1,[x_2,x_3]]$ maps to $f(x_1) \cdot f([x_2,x_3]) - f([x_2,x_3]) \cdot f(x_1)$ in an associative algebra, and $f([x_2,x_3]) = e_e - e_{(12)}$ at the $\{2,3\}$ positions. The full antisymmetrization: $[x_1,[x_2,x_3]] = x_1 x_2 x_3 - x_1 x_3 x_2 - x_2 x_3 x_1 + x_3 x_2 x_1$, giving $f([x_1,[x_2,x_3]]) = e_e - e_{(23)} - e_{(123)} + e_{(132)} \in k[S_3]$ (identifying permutations: $e$ = identity, $(23)$ swaps 2 and 3, $(123)$ is the 3-cycle, $(132)$ its inverse).

### 5.6 The Little n-Disks Operad

The *little $n$-disks operad* (or *little $n$-cubes operad*) is a fundamental example of a *topological operad* — an operad in the category $\mathbf{Top}$ of topological spaces (replacing $k$-modules by spaces, $\otimes$ by cartesian product $\times$, and $\bigoplus$ by disjoint union $\sqcup$).

**Definition 5.6 (Little $n$-Disks Operad).** Let $D^n \subset \mathbb{R}^n$ denote the closed unit disk. Define
$$\mathcal{D}_n(k) := \left\{ \text{configurations of } k \text{ disjoint "little" } n\text{-disks } D_1, \ldots, D_k \subset D^n \right\}$$
where a "little $n$-disk" is the image of a *linear embedding* $D^n \to D^n$ (i.e., a composition of a dilation and a translation, with no rotation). So $\mathcal{D}_n(k)$ is the space of $k$-tuples of *rectilinear embeddings* $D^n \hookrightarrow D^n$ with pairwise disjoint images.

The $S_k$-action on $\mathcal{D}_n(k)$ is by permuting the labels of the little disks.

The unit is $\mathrm{id}_{D^n} \in \mathcal{D}_n(1)$.

**Composition.** For a configuration $(D_1,\ldots,D_k) \in \mathcal{D}_n(k)$ and configurations $(E_{i,1},\ldots,E_{i,m_i}) \in \mathcal{D}_n(m_i)$ for each $i$, the composed configuration $\gamma((D_i); (E_{i,j})) \in \mathcal{D}_n(m_1+\cdots+m_k)$ is obtained by *rescaling*: map each $E_{i,j} \subset D^n$ by the linear embedding $D_i \colon D^n \to D^n$ to get a little disk $D_i(E_{i,j}) \subset D_i \subset D^n$.

> [!NOTE] Homotopy Type of Dn(k)
> The space $\mathcal{D}_n(k)$ is homotopy equivalent to the *configuration space* $\mathrm{Conf}_k(\mathbb{R}^n)$ of $k$ distinct ordered points in $\mathbb{R}^n$. For $n=1$, $\mathrm{Conf}_k(\mathbb{R}) \simeq S_k$ (discrete), so $\mathcal{D}_1$ is the operad of *associativity* (up to homotopy). For $n=2$, $\mathrm{Conf}_k(\mathbb{C})$ is connected with a rich braid-group action — algebras over $H_*(\mathcal{D}_2)$ are exactly *Gerstenhaber algebras*.

**Theorem 5.7 (May's Recognition Principle, 1972).** A connected topological space $X$ is weakly homotopy equivalent to an $n$-fold loop space $\Omega^n Y$ (for some $Y$) if and only if $X$ is a *grouplike $\mathcal{D}_n$-algebra* (a $\mathcal{D}_n$-algebra in $\mathbf{Top}$ whose induced $\pi_0$-monoid structure is a group).

*Heuristic:* The little $n$-disks operad parameterizes the homotopy-coherent ways to compose $n$-fold loops. An $n$-fold loop space $\Omega^n Y = \mathrm{Map}_*(S^n, Y)$ is naturally a $\mathcal{D}_n$-algebra: given a configuration of $k$ disks in $D^n$ and $k$ loops in $Y$, one can "stack" the loops on the disk configuration to get a new loop. May's theorem says this completely characterizes $n$-fold loop spaces.

> [!INFO] Historical Note
> May introduced operads in his 1972 monograph precisely to give a clean proof of the recognition principle. Earlier work by Stasheff (1963) on $A_\infty$-spaces and Boardman–Vogt (1973) on homotopy-invariant structures were key precursors.

### 5.7 The Probability Operad

💡 The *probability operad* gives an unexpected bridge between operad theory and information theory. See also [[research/categorical-entropy|Categorical Entropy]] and [[research/entropy-operad-derivation|Entropy as an Operad Derivation]].

**Definition 5.8 (Probability Operad).** The probability operad $\mathcal{P}$ is a topological operad defined by
$$\mathcal{P}(n) := \Delta^{n-1} = \left\{ (p_1, \ldots, p_n) \in \mathbb{R}^n \;\middle|\; p_i \geq 0,\ \sum_{i=1}^n p_i = 1 \right\}, \quad n \geq 1,$$
the standard $(n-1)$-dimensional *simplex*. Set $\mathcal{P}(0) := \emptyset$.

The symmetric group $S_n$ acts on $\mathcal{P}(n) = \Delta^{n-1}$ by permuting coordinates: $(p_1,\ldots,p_n) \cdot \sigma = (p_{\sigma^{-1}(1)}, \ldots, p_{\sigma^{-1}(n)})$.

The unit is the unique point $1 \in \Delta^0 = \mathcal{P}(1)$.

**Composition.** For $p = (p_1,\ldots,p_k) \in \mathcal{P}(k)$ and $q^{(i)} = (q^{(i)}_1, \ldots, q^{(i)}_{n_i}) \in \mathcal{P}(n_i)$ for $i = 1, \ldots, k$, the *full composition* is:
$$\gamma(p; q^{(1)}, \ldots, q^{(k)}) := (p_1 q^{(1)}_1, \ldots, p_1 q^{(1)}_{n_1}, p_2 q^{(2)}_1, \ldots, p_k q^{(k)}_{n_k}) \in \mathcal{P}(n_1 + \cdots + n_k).$$
In partial composition notation, for $p \in \mathcal{P}(m)$, $q \in \mathcal{P}(n)$:
$$p \circ_i q = (p_1, \ldots, p_{i-1}, p_i q_1, \ldots, p_i q_n, p_{i+1}, \ldots, p_m) \in \mathcal{P}(m + n - 1).$$

*Operadic interpretation:* $\gamma(p; q^{(1)},\ldots,q^{(k)})$ is the *mixture* distribution: choose component $i$ with probability $p_i$, then sample from $q^{(i)}$.

> [!EXAMPLE] Composition in P(2)
> Take $p = (p_1, p_2) \in \mathcal{P}(2)$ and $q = (q_1, q_2, q_3) \in \mathcal{P}(3)$. Then
> $$p \circ_1 q = (p_1 q_1, p_1 q_2, p_1 q_3, p_2) \in \mathcal{P}(4),$$
> $$p \circ_2 q = (p_1, p_2 q_1, p_2 q_2, p_2 q_3) \in \mathcal{P}(4).$$
> These are genuine probability distributions since $p_1 q_1 + p_1 q_2 + p_1 q_3 + p_2 = p_1(q_1+q_2+q_3) + p_2 = p_1 + p_2 = 1$.

**$\mathcal{P}$-Algebra Structure.** A $\mathcal{P}$-algebra (in the topological category, with spaces replaced by "sets with measure-theoretic structure") is a set $A$ equipped with maps $\alpha_n \colon \mathcal{P}(n) \times A^n \to A$ for all $n$, compatible with the operadic composition. For $A = [0,\infty)$ with $\alpha_n(p; a_1,\ldots,a_n) := \sum_{i=1}^n p_i a_i$ (weighted average), one verifies:
$$\alpha_{n_1+\cdots+n_k}(\gamma(p;q^{(1)},\ldots,q^{(k)}); a_1, \ldots, a_n) = \alpha_k(p; \alpha_{n_1}(q^{(1)};a_1,\ldots), \ldots, \alpha_{n_k}(q^{(k)};\ldots,a_n))$$
since both sides equal $\sum_i p_i \sum_j q^{(i)}_j a_{i,j}$. So $([0,\infty), \{\alpha_n\})$ is a $\mathcal{P}$-algebra.

**Shannon Entropy as an Operadic Derivation.** Define $H \colon \mathcal{P}(n) \to \mathbb{R}$ by
$$H(p_1, \ldots, p_n) := -\sum_{i=1}^n p_i \log p_i.$$
(with the convention $0 \log 0 = 0$.) Then $H$ satisfies the *chain rule*:
$$H(p \circ_i q) = H(p) + p_i H(q)$$
or more generally
$$H(\gamma(p; q^{(1)}, \ldots, q^{(k)})) = H(p) + \sum_{i=1}^k p_i H(q^{(i)}).$$

This is precisely the condition that $H$ is an *operadic derivation* of $\mathcal{P}$ (with values in a suitable bimodule). By a theorem of Faddeev (1956) and its operadic refinement by Leinster–Bradley, Shannon entropy is (up to scalar) the *unique* continuous derivation of the probability operad satisfying this chain rule.

> [!WARNING] Algebraic vs Topological Operad
> The probability operad $\mathcal{P}$ is a topological operad (enriched in $\mathbf{Top}$) but *not* an algebraic operad in the sense of Section 3: its components $\mathcal{P}(n) = \Delta^{n-1}$ are geometric simplices, not $k$-modules. To apply the algebraic machinery, one passes to the singular chain complex $C_*(\mathcal{P})$ or homology $H_*(\mathcal{P})$, obtaining an operad in chain complexes or graded $k$-modules respectively.

> [!QUESTION] Exercise 12: Verification of P-Operad Axioms
> *This problem verifies that the probability operad satisfies the partial-composition axioms.*
>
> > **Prerequisites:** [[#5.7 The Probability Operad|5.7 The Probability Operad]]
>
> Let $p \in \mathcal{P}(m)$, $q \in \mathcal{P}(n)$, $r \in \mathcal{P}(\ell)$, and suppose $j < i$. Verify axiom PC2 (parallel associativity): $(p \circ_i q) \circ_j r = (p \circ_j r) \circ_{i + \ell - 1} q$.

> [!TIP]- Solution to Exercise 12
> **Key insight:** Both sides rescale the same coordinates, and scalar multiplication commutes.
>
> **Sketch:** LHS: $p \circ_i q = (\ldots, p_i q_1, \ldots, p_i q_n, \ldots)$ with $q$-block at positions $i,\ldots,i+n-1$. Then $\circ_j r$ (for $j < i$) replaces $p_j$ by $p_j r_1,\ldots,p_j r_\ell$ at positions $j,\ldots,j+\ell-1$: result is $(\ldots, p_j r_1,\ldots,p_j r_\ell, \ldots, p_i q_1,\ldots,p_i q_n, \ldots) \in \mathcal{P}(m+n+\ell-2)$. RHS: $p \circ_j r = (\ldots,p_j r_1,\ldots,p_j r_\ell,\ldots,p_i,\ldots)$, then $\circ_{i+\ell-1} q$ replaces $p_i$ (now at position $i+\ell-1$) by $p_i q_1,\ldots,p_i q_n$: same result.

> [!QUESTION] Exercise 13: Entropy Chain Rule
> *This problem derives the operadic character of Shannon entropy from the chain rule.*
>
> > **Prerequisites:** [[#5.7 The Probability Operad|5.7 The Probability Operad]]
>
> Using the partial composition $p \circ_i q$ in $\mathcal{P}$, verify the chain rule $H(p \circ_i q) = H(p) + p_i H(q)$ directly from the definition $H(p) = -\sum_j p_j \log p_j$.

> [!TIP]- Solution to Exercise 13
> **Key insight:** The $\log$ of a product splits as a sum of logs, and the constraint $\sum_j q_j = 1$ allows separating terms.
>
> **Sketch:** $p \circ_i q = (p_1,\ldots,p_{i-1},p_i q_1,\ldots,p_i q_n,p_{i+1},\ldots,p_m)$. So:
> $$H(p \circ_i q) = -\sum_{j \neq i} p_j \log p_j - \sum_{s=1}^n p_i q_s \log(p_i q_s)$$
> $$= -\sum_{j \neq i} p_j \log p_j - \sum_s p_i q_s (\log p_i + \log q_s)$$
> $$= -\sum_{j \neq i} p_j \log p_j - p_i \log p_i \underbrace{\sum_s q_s}_{=1} - p_i \sum_s q_s \log q_s$$
> $$= H(p) + p_i H(q).$$

> [!QUESTION] Exercise 14 (Algorithmic): Computing Operad Compositions
> *This problem develops a recursive algorithm for computing operadic compositions in terms of partial compositions.*
>
> > **Prerequisites:** [[#3.3 Equivalence of the Two Formulations|3.3 Equivalence of the Two Formulations]]
>
> Write a Python function `full_compose(mu, nus, gamma_partial)` that computes $\gamma(\mu; \nu_1, \ldots, \nu_k)$ from partial compositions using the reconstruction formula. What is its time complexity in terms of $k$ (number of inputs)?

> [!TIP]- Solution to Exercise 14
> **Key insight:** The sequential reconstruction applies partial compositions one at a time, from right to left.
>
> **Sketch:**
> ```python
> def full_compose(mu, nus, circ_i):
>     """
>     mu: element of O(k)
>     nus: list [nu_1, ..., nu_k], nu_i in O(n_i)
>     circ_i(x, y, i): partial composition x circ_i y
>     Returns gamma(mu; nu_1, ..., nu_k)
>     """
>     result = mu
>     # Compose from left: insert nu_1 at position 1, then nu_2 at 1, etc.
>     # After inserting nu_1, ..., nu_j, the arity of result is 1 + sum(n_i - 1 for i <= j)
>     # nu_{j+1} is inserted at position 1 + sum(n_i for i < j+1) - j ... 
>     # Simpler: insert from right to left
>     offset = 1
>     for nu in nus:
>         result = circ_i(result, nu, offset)
>         offset += 1  # next nu goes into the slot after the previous
>     return result
>     # Actually: insert nu_1 at pos 1, nu_2 at pos n_1 + 1, etc.
> ```
> *Correction:* Insert $\nu_j$ at position $n_1 + \cdots + n_{j-1} + 1$ of the current result. Time complexity: $O(k)$ partial composition calls, each taking $O(1)$ to specify (assuming $O(1)$ per call), so overall $O(k)$.

---

## 6. Colored Operads and Multicategories

### 6.1 Definition of Colored Operads

The single-color operad theory of the previous sections generalizes naturally to *colored operads*, which allow operations to have inputs and outputs of different *types* or *colors*. This connects operads to [[concepts/category-theory/symmetric-monoidal-categories|Symmetric Monoidal Categories]] and to the theory of multicategories.

**Definition 6.1 (Colored Operad).** Let $C$ be a set, called the *set of colors* (or *set of objects*). A *$C$-colored operad* (also called a *multicategory* or *symmetric multicategory*) $\mathcal{O}$ consists of:
- For each $n \geq 0$, each tuple of *input colors* $(c_1, \ldots, c_n) \in C^n$, and each *output color* $c \in C$, a $k$-module
$$\mathcal{O}(c_1, \ldots, c_n; c)$$
of *operations from $(c_1,\ldots,c_n)$ to $c$*.

- For each color $c \in C$, a *unit element* $\mathrm{id}_c \in \mathcal{O}(c; c)$.

- *Composition maps*: for all $n, m_1, \ldots, m_n \geq 0$, all input/output color data, a map
$$\gamma \colon \mathcal{O}(c_1,\ldots,c_n; c) \otimes \mathcal{O}(d_{1,1},\ldots,d_{1,m_1}; c_1) \otimes \cdots \otimes \mathcal{O}(d_{n,1},\ldots,d_{n,m_n}; c_n) \longrightarrow \mathcal{O}(d_{1,1},\ldots,d_{n,m_n}; c).$$

- A *right action of $S_n$ on $\mathcal{O}(c_1,\ldots,c_n; c)$* by simultaneous permutation of inputs: $\sigma \in S_n$ acts by
$$\mu \cdot \sigma \in \mathcal{O}(c_{\sigma^{-1}(1)},\ldots,c_{\sigma^{-1}(n)}; c).$$

These data satisfy associativity, unit, and equivariance axioms analogous to those of Definition 3.1 (with color compatibility conditions added: all compositions are only defined when input and output colors match).

> [!NOTE] Single-Color Case
> When $|C| = 1$ (one color), all color-matching conditions are vacuous, and a $C$-colored operad reduces to an ordinary (symmetric) operad. So the single-color theory is a special case of the colored theory.

### 6.2 Examples and Connections

**Small Categories as Colored Operads.** A *small category* $\mathcal{C}$ with object set $C$ is a $C$-colored operad concentrated in arity 1: $\mathcal{O}(c; d) = \mathrm{Hom}_\mathcal{C}(c, d)$ and $\mathcal{O}(c_1,\ldots,c_n; d) = 0$ for $n \neq 1$. Composition in $\mathcal{C}$ gives the operadic composition at arity 1. The unit $\mathrm{id}_c$ is the identity morphism.

**Symmetric Monoidal Categories as Colored Operads.** A *symmetric monoidal category* $(\mathcal{C}, \otimes, \mathbf{1})$ can be encoded as a colored operad where $\mathcal{O}(c_1,\ldots,c_n; c) = \mathrm{Hom}_\mathcal{C}(c_1 \otimes \cdots \otimes c_n, c)$. The operadic composition corresponds to the bifunctoriality of $\otimes$, and the $S_n$-action corresponds to the symmetric monoidal structure (braiding). This connection is the starting point for Lurie's theory of $\infty$-operads.

**PROPs.** A *PROP* (PROduct and Permutation category) generalizes colored operads by allowing multiple outputs as well as multiple inputs. Operations take the form $P(\mathbf{m}; \mathbf{n})$ for $\mathbf{m}, \mathbf{n}$ finite sets; composition is by horizontal (sequential) and vertical (parallel) composition. PROPs encode structures like bialgebras and Hopf algebras that cannot be described by operads.

> [!TIP] Connection to Lurie's Theory
> In Lurie's *Higher Algebra*, $\infty$-operads are defined as $\infty$-categories $\mathcal{O}^\otimes \to \mathbf{Fin}_*$ over the category of pointed finite sets, where the fiber over $\langle n \rangle$ gives $\mathcal{O}(n)$. This is the $(\infty,1)$-categorical version of colored operads, and symmetric monoidal $\infty$-categories are precisely commutative algebra objects in the $(\infty,1)$-category of $\infty$-categories.

> [!QUESTION] Exercise 15 (Algorithmic): Checking Color Compatibility
> *This problem sketches an algorithm for verifying that a colored operad composition is well-typed.*
>
> > **Prerequisites:** [[#6.1 Definition of Colored Operads|6.1 Definition of Colored Operads]]
>
> Write a Python function `check_composition_typed(mu_colors, nu_colors_list)` that takes the input/output colors of an operation $\mu$ and a list of operations $\nu_1, \ldots, \nu_k$ and checks whether the composition $\gamma(\mu; \nu_1, \ldots, \nu_k)$ is well-typed (i.e., the output color of each $\nu_i$ matches the $i$-th input color of $\mu$).

> [!TIP]- Solution to Exercise 15
> **Key insight:** Color-matching is simply checking equality of output color of each $\nu_i$ against the corresponding input color of $\mu$.
>
> **Sketch:**
> ```python
> def check_composition_typed(mu_colors, nu_colors_list):
>     """
>     mu_colors: dict with keys 'inputs' (list of colors) and 'output' (color)
>     nu_colors_list: list of dicts, each with 'inputs' and 'output'
>     Returns True if composition is well-typed, False otherwise.
>     """
>     mu_inputs = mu_colors['inputs']
>     if len(mu_inputs) != len(nu_colors_list):
>         return False
>     for i, (expected_color, nu) in enumerate(zip(mu_inputs, nu_colors_list)):
>         if nu['output'] != expected_color:
>             return False
>     return True
> # O(k) time where k = len(mu_inputs)
> ```

> [!QUESTION] Exercise 16: Categories Are Colored Operads
> *This problem makes precise the embedding of small categories into colored operads.*
>
> > **Prerequisites:** [[#6.2 Examples and Connections|6.2 Examples and Connections]]
>
> Let $\mathcal{C}$ be a small category with objects $C = \{a, b, c, \ldots\}$. Define a $C$-colored operad $\mathcal{O}$ by $\mathcal{O}(c_1; c) = \mathrm{Hom}_\mathcal{C}(c_1, c)$ and $\mathcal{O}(c_1,\ldots,c_n;c) = 0$ for $n \neq 1$. Verify that the operadic composition and unit axioms reduce exactly to the composition and unit axioms of $\mathcal{C}$.

> [!TIP]- Solution to Exercise 16
> **Key insight:** All non-trivial compositions in this operad involve only arity-1 operations, which are exactly morphisms in $\mathcal{C}$.
>
> **Sketch:** The only non-zero composition is at $(n;m_1) = (1;1)$: $\mathcal{O}(c_1;c) \otimes \mathcal{O}(d_1;c_1) \to \mathcal{O}(d_1;c)$, i.e., $\mathrm{Hom}(c_1,c) \otimes \mathrm{Hom}(d_1,c_1) \to \mathrm{Hom}(d_1,c)$, which is composition in $\mathcal{C}$. Unit: $\mathrm{id}_c \in \mathcal{O}(c;c) = \mathrm{Hom}(c,c)$ is the identity morphism. Associativity in $\mathcal{O}$ reduces to associativity of composition in $\mathcal{C}$.

> [!QUESTION] Exercise 17: The Free Colored Operad
> *This problem establishes the universal property of free colored operads.*
>
> > **Prerequisites:** [[#4. Morphisms and the Category Op|4. Morphisms and the Category Op]], [[#6.1 Definition of Colored Operads|6.1 Definition of Colored Operads]]
>
> Let $C$ be a set of colors and $\mathcal{M}$ a collection of generating operations (a $C$-colored symmetric sequence). Define the free $C$-colored operad $\mathcal{F}_C(\mathcal{M})$ and state its universal property: for any $C$-colored operad $\mathcal{O}$ and any color-preserving map $f \colon \mathcal{M} \to \mathcal{O}$ of colored symmetric sequences, there is a unique operad morphism $\tilde{f} \colon \mathcal{F}_C(\mathcal{M}) \to \mathcal{O}$ extending $f$.

> [!TIP]- Solution to Exercise 17
> **Key insight:** Free objects are constructed from trees decorated by generators, just as in the single-color case.
>
> **Sketch:** $\mathcal{F}_C(\mathcal{M})(c_1,\ldots,c_n;c)$ is spanned by all rooted trees with $n$ leaves, where each leaf is labeled by a color $c_i$, the root is labeled by $c$, and each internal vertex of valence $k$ with input colors $a_1,\ldots,a_k$ and output color $a$ carries a generator from $\mathcal{M}(a_1,\ldots,a_k;a)$. The universal property follows: $\tilde{f}$ is defined by sending each decorated tree to the corresponding iterated composition in $\mathcal{O}$.

> [!QUESTION] Exercise 18 (Algorithmic): Implementing the Probability Operad
> *This problem implements the probability operad's composition map as a numerical routine.*
>
> > **Prerequisites:** [[#5.7 The Probability Operad|5.7 The Probability Operad]]
>
> Implement a Python function `prob_operad_compose(p, qs)` that computes $\gamma(p; q^{(1)}, \ldots, q^{(k)}) \in \Delta^{n-1}$ where $p \in \Delta^{k-1}$ and $q^{(i)} \in \Delta^{n_i - 1}$. Verify computationally that the result sums to 1.

> [!TIP]- Solution to Exercise 18
> **Key insight:** The composed distribution is the concatenation of scaled sub-distributions.
>
> **Sketch:**
> ```python
> import numpy as np
>
> def prob_operad_compose(p, qs):
>     """
>     p: array of shape (k,), a probability distribution
>     qs: list of k arrays, qs[i] is a probability distribution of shape (n_i,)
>     Returns: probability distribution of shape (n_1 + ... + n_k,)
>     """
>     p = np.array(p)
>     result = np.concatenate([p[i] * np.array(q) for i, q in enumerate(qs)])
>     assert abs(result.sum() - 1.0) < 1e-10, f"Not a distribution: sum = {result.sum()}"
>     return result
>
> # Example: p = (0.3, 0.7), q1 = (0.5, 0.5), q2 = (1/3, 1/3, 1/3)
> # Result: (0.15, 0.15, 0.7/3, 0.7/3, 0.7/3)
> print(prob_operad_compose([0.3, 0.7], [[0.5, 0.5], [1/3, 1/3, 1/3]]))
> ```

---

## References

| Reference Name | Brief Summary | Link to Reference |
|----------------|--------------|-------------------|
| Loday & Vallette, *Algebraic Operads* | The definitive reference for algebraic operads: symmetric sequences, composition product, Koszul duality, bar/cobar constructions; Chapters 5 and 13 are most relevant here | [Springer](https://link.springer.com/book/10.1007/978-3-642-30362-3) |
| May, *The Geometry of Iterated Loop Spaces* | Original monograph introducing operads and proving the recognition principle for $n$-fold loop spaces via the little $n$-cubes operad | [PDF](https://www.math.uchicago.edu/~may/BOOKS/gils.pdf) |
| Markl, *Operads and PROPs* | Concise survey of operads and PROPs, covering symmetric sequences, the composition product, examples, and PROPs; excellent secondary reference | [arXiv:math/0601129](https://arxiv.org/abs/math/0601129) |
| Vallette, "Algebra + Homotopy = Operad" | Elementary survey of operads with emphasis on homotopical applications; accessible entry point before Loday–Vallette | [arXiv:1202.3245](https://arxiv.org/abs/1202.3245) |
| Leinster, *Entropy and Diversity* | Develops the probability operad and proves Shannon entropy is the unique continuous derivation; connects information theory to operadic algebra | [arXiv:2012.02113](https://arxiv.org/abs/2012.02113) |
| Ginzburg & Kapranov, "Koszul Duality for Operads" | Foundational paper establishing Koszul duality for quadratic operads; defines $\mathrm{Ass}$, $\mathrm{Com}$, $\mathrm{Lie}$ as Koszul dual pairs | [arXiv:0709.1228](https://arxiv.org/abs/0709.1228) |
| nLab: operad | Comprehensive wiki article on operads with precise definitions, examples, and categorical perspectives | [nLab](https://ncatlab.org/nlab/show/operad) |
