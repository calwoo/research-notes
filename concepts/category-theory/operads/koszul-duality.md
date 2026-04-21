# Operads: Koszul Duality and the Bar Construction

## Table of Contents

- [[#1. Cooperads|1. Cooperads]]
  - [[#1.1 Definition|1.1 Definition]]
  - [[#1.2 Conilpotency|1.2 Conilpotency]]
- [[#2. The Operadic Bar Construction|2. The Operadic Bar Construction]]
  - [[#2.1 Augmented Operads|2.1 Augmented Operads]]
  - [[#2.2 The Bar Complex|2.2 The Bar Complex]]
  - [[#2.3 The Differential|2.3 The Differential]]
- [[#3. The Operadic Cobar Construction|3. The Operadic Cobar Construction]]
  - [[#3.1 Definition|3.1 Definition]]
  - [[#3.2 The Bar-Cobar Adjunction|3.2 The Bar-Cobar Adjunction]]
- [[#4. Twisting Morphisms|4. Twisting Morphisms]]
  - [[#4.1 The Maurer-Cartan Equation|4.1 The Maurer-Cartan Equation]]
  - [[#4.2 The Universal Twisting Morphisms|4.2 The Universal Twisting Morphisms]]
- [[#5. Quadratic Operads and the Koszul Dual|5. Quadratic Operads and the Koszul Dual]]
  - [[#5.1 Quadratic Operads|5.1 Quadratic Operads]]
  - [[#5.2 The Koszul Dual Cooperad|5.2 The Koszul Dual Cooperad]]
  - [[#5.3 The Canonical Twisting Morphism|5.3 The Canonical Twisting Morphism]]
  - [[#5.4 Examples: Ass, Com, and Lie|5.4 Examples: Ass, Com, and Lie]]
- [[#6. The Koszul Criterion|6. The Koszul Criterion]]
  - [[#6.1 Statement|6.1 Statement]]
  - [[#6.2 Koszulness of Ass and Com|6.2 Koszulness of Ass and Com]]
- [[#7. Homotopy Algebras from Koszul Duality|7. Homotopy Algebras from Koszul Duality]]
  - [[#7.1 The Resolution O-infinity|7.1 The Resolution O-infinity]]
  - [[#7.2 A-infinity and L-infinity Algebras|7.2 A-infinity and L-infinity Algebras]]
  - [[#7.3 The Homotopy Transfer Theorem|7.3 The Homotopy Transfer Theorem]]
- [[#8. Operadic Cohomology and Deformation Theory|8. Operadic Cohomology and Deformation Theory]]
  - [[#8.1 The Two-Sided Bar Resolution|8.1 The Two-Sided Bar Resolution]]
  - [[#8.2 Classical Cohomology Theories as Special Cases|8.2 Classical Cohomology Theories as Special Cases]]
  - [[#8.3 The Deformation Complex of an Operad|8.3 The Deformation Complex of an Operad]]
- [[#References|References]]

---

## 1. Cooperads 📐

### 1.1 Definition

Dualizing the notion of an operad yields a *cooperad*. Recall from [[concepts/category-theory/operads/definitions|Definitions and Examples]] that an operad is a monoid in the monoidal category $(\mathbf{SymSeq}, \circ, \mathbf{1})$. A cooperad is the corresponding comonoid.

**Definition (Cooperad).** A *cooperad* is a triple $(\mathcal{C}, \Delta, \epsilon)$ consisting of a symmetric sequence $\mathcal{C} \in \mathbf{SymSeq}$ together with:

- A *decomposition map* $\Delta: \mathcal{C} \to \mathcal{C} \circ \mathcal{C}$, coassociative in the sense that the following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\mathcal{C} \arrow[r, "\Delta"] \arrow[d, "\Delta"'] & \mathcal{C} \circ \mathcal{C} \arrow[d, "\Delta \circ \mathrm{id}"] \\
\mathcal{C} \circ \mathcal{C} \arrow[r, "\mathrm{id} \circ \Delta"'] & \mathcal{C} \circ \mathcal{C} \circ \mathcal{C}
\end{tikzcd}
\end{document}
```

- A *counit* $\epsilon: \mathcal{C} \to \mathbf{1}$, satisfying the counit axioms:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\mathcal{C} \arrow[r, "\Delta"] \arrow[dr, "\sim"'] & \mathcal{C} \circ \mathcal{C} \arrow[d, "\epsilon \circ \mathrm{id}"] \\
& \mathbf{1} \circ \mathcal{C} \cong \mathcal{C}
\end{tikzcd}
\qquad
\begin{tikzcd}
\mathcal{C} \arrow[r, "\Delta"] \arrow[dr, "\sim"'] & \mathcal{C} \circ \mathcal{C} \arrow[d, "\mathrm{id} \circ \epsilon"] \\
& \mathcal{C} \circ \mathbf{1} \cong \mathcal{C}
\end{tikzcd}
\end{document}
```

> [!NOTE] Decomposition in low arities
> In arity $n$, the decomposition map $\Delta$ has components
> $$\Delta_n : \mathcal{C}(n) \to \bigoplus_{k \geq 1} \bigoplus_{n_1+\cdots+n_k=n} \mathcal{C}(k) \otimes_{S_k} \bigl(\mathcal{C}(n_1) \otimes \cdots \otimes \mathcal{C}(n_k)\bigr).$$
> This encodes all ways to split an $n$-ary cooperation into a $k$-ary cooperation followed by $k$ cooperations of arities $n_1, \ldots, n_k$.

**Definition (Morphism of cooperads).** A morphism $f: \mathcal{C} \to \mathcal{D}$ of cooperads is a morphism of symmetric sequences commuting with $\Delta$ and $\epsilon$.

**Example.** The *dual* of a finite-dimensional operad $\mathcal{O}$ is a cooperad: set $\mathcal{O}^\vee(n) = \mathcal{O}(n)^* = \mathrm{Hom}_k(\mathcal{O}(n), k)$ with right $S_n$-action via $(f \cdot \sigma)(x) = f(x \cdot \sigma^{-1})$. The composition $\gamma: \mathcal{O} \circ \mathcal{O} \to \mathcal{O}$ dualizes to $\Delta: \mathcal{O}^\vee \to \mathcal{O}^\vee \circ \mathcal{O}^\vee$.

### 1.2 Conilpotency

The bar construction requires a finiteness condition on cooperads.

**Definition (Conilpotency).** A cooperad $\mathcal{C}$ is *conilpotent* if for every $c \in \mathcal{C}(n)$ there exists $N$ such that all iterated decompositions of $c$ of depth $> N$ vanish. Explicitly, define the *coradical filtration* $F_0\mathcal{C} \subseteq F_1\mathcal{C} \subseteq \cdots$ by $F_0\mathcal{C}(n) = 0$ for $n > 1$, $F_0\mathcal{C}(1) = k \cdot \mathrm{id}$, and $F_r\mathcal{C} = \Delta^{-1}(\sum_{s+t=r} F_s\mathcal{C} \circ F_t\mathcal{C})$. The cooperad is conilpotent if $\mathcal{C} = \bigcup_r F_r\mathcal{C}$.

> [!WARNING] Why conilpotency matters
> Without conilpotency, the cobar construction $\Omega(\mathcal{C})$ produces an operad with infinitely many generators in each weight, making the quasi-isomorphism $\Omega B(\mathcal{O}) \xrightarrow{\sim} \mathcal{O}$ ill-formed. All cooperads arising as Koszul duals $\mathcal{O}^!$ of quadratic operads are automatically conilpotent.

> [!QUESTION] Exercise 1: The cooperad $\mathrm{Ass}^c$
> *This establishes the cooperad structure on $\mathrm{Ass}^c = \mathrm{Ass}^\vee$, the Koszul dual cooperad of $\mathrm{Ass}$.*
>
> > **Prerequisites:** [[#1.1 Definition|1.1 Definition]], [[concepts/category-theory/operads/definitions#5.1 The Associative Operad Ass|Ass definition]]
>
> The associative cooperad $\mathrm{Ass}^c$ has $\mathrm{Ass}^c(n) = k[S_n]$ for $n \geq 1$. Write down the decomposition map $\Delta: \mathrm{Ass}^c(n) \to \bigoplus_{k} \mathrm{Ass}^c(k) \otimes_{S_k} \mathrm{Ass}^c(n_1) \otimes \cdots \otimes \mathrm{Ass}^c(n_k)$ explicitly for $n = 3$ and verify coassociativity.

> [!TIP]- Solution to Exercise 1
> **Key insight:** $\Delta$ on $\mathrm{Ass}^c$ sums over all ways to partition a permutation into a block structure, with appropriate block-sum permutations.
>
> **Sketch:** For $\sigma \in S_3$, $\Delta(\sigma) = \sum_{k; n_1+\cdots+n_k=3} \tau \otimes (\sigma_1 \otimes \cdots \otimes \sigma_k)$ where $\tau \in S_k$ is the block permutation induced by $\sigma$ on blocks of sizes $n_i$, and $\sigma_i \in S_{n_i}$ is the restriction of $\sigma$ to the $i$-th block. For example, $\Delta(\mathrm{id}_3) = \mathrm{id}_1 \otimes \mathrm{id}_3 + \mathrm{id}_2 \otimes (\mathrm{id}_1 \otimes \mathrm{id}_2 + \mathrm{id}_2 \otimes \mathrm{id}_1) + \mathrm{id}_3 \otimes \mathrm{id}_1^{\otimes 3}$. Coassociativity follows from associativity of the block-sum decomposition of permutations.

---

## 2. The Operadic Bar Construction 📐

### 2.1 Augmented Operads

The bar construction requires an augmentation.

**Definition (Augmented operad).** An *augmented operad* is an operad $\mathcal{O}$ together with an operad morphism $\epsilon: \mathcal{O} \to \mathbf{1}$ (the *augmentation*). Since $\mathbf{1}(1) = k$ and $\mathbf{1}(n) = 0$ for $n \neq 1$, this forces $\epsilon$ to be zero on all $\mathcal{O}(n)$ for $n \neq 1$ and the identity on the unit $\mathrm{id} \in \mathcal{O}(1)$.

Set $\bar{\mathcal{O}} = \ker(\epsilon)$, so that $\mathcal{O} \cong \mathbf{1} \oplus \bar{\mathcal{O}}$ as symmetric sequences. The components of $\bar{\mathcal{O}}$ are $\bar{\mathcal{O}}(n) = \mathcal{O}(n)$ for $n \geq 2$ and $\bar{\mathcal{O}}(1) = \ker(\mathcal{O}(1) \to k)$.

> [!EXAMPLE] Standard augmentations
> The operads $\mathrm{Ass}$, $\mathrm{Com}$, and $\mathrm{Lie}$ are naturally augmented: the augmentation $\epsilon: \mathcal{O} \to \mathbf{1}$ sends the unit element $\mathrm{id} \in \mathcal{O}(1)$ to $1 \in k$ and kills all higher-arity operations. So $\overline{\mathrm{Ass}}(n) = k[S_n]$ for $n \geq 2$, zero for $n = 1$.

### 2.2 The Bar Complex

**Definition (Bar construction).** For an augmented dg-operad $(\mathcal{O}, d, \gamma, \eta, \epsilon)$, the *bar construction* $B(\mathcal{O})$ is the conilpotent dg-cooperad defined as follows.

As a graded symmetric sequence (ignoring the differential),
$$B(\mathcal{O}) = \mathcal{F}^c(\bar{\mathcal{O}}[1]),$$
the *cofree conilpotent cooperad* on the suspension $\bar{\mathcal{O}}[1]$. Concretely, $\mathcal{F}^c(M)$ is the symmetric sequence whose $n$-th component consists of all rooted trees with $n$ leaves whose internal vertices are labeled by elements of $M$, modulo the $S_n$-equivariance. As a graded object,
$$B(\mathcal{O})(n) = \bigoplus_{T} \bigotimes_{v \in V(T)} \bar{\mathcal{O}}(|v|)[1],$$
where the sum is over isomorphism classes of rooted trees $T$ with $n$ leaves, $V(T)$ is the set of internal vertices, and $|v|$ is the number of inputs of $v$.

> [!NOTE] Weight grading
> The bar construction carries a *weight grading* by $\mathrm{wt}(B(\mathcal{O})) = k$ when $k$ internal vertices are used. This is separate from the homological grading. The piece of weight $k$ is $\bar{\mathcal{O}}^{\circ k}[k]$ (the $k$-fold composition product of $\bar{\mathcal{O}}$, shifted by $k$).

### 2.3 The Differential

The differential on $B(\mathcal{O})$ has two components: $d_{B(\mathcal{O})} = d_1 + d_2$.

- $d_1$: the *internal differential*, induced by the differential $d$ on $\mathcal{O}$.
- $d_2$: the *composition differential*, which decreases weight by 1. On a tree-monomial $(\mu_1, \ldots, \mu_k)$ labeling the vertices, $d_2$ sums over all internal edges $e$ of the tree and contracts $e$ by applying the operadic composition $\gamma$ to the two vertices adjacent to $e$, with a sign $(-1)^{\deg}$ determined by the Koszul sign rule.

🔑 **The key fact:** $d_{B(\mathcal{O})}^2 = 0$ follows from the associativity of $\gamma$: the terms in $d_2^2$ pair up with opposite signs by the associativity axiom for $\mathcal{O}$, and $d_1^2 = 0$ since $d$ is a differential on $\mathcal{O}$.

> [!QUESTION] Exercise 2: The differential in weight 2
> *This makes the abstract definition of $d_2$ concrete in the first nontrivial case.*
>
> > **Prerequisites:** [[#2.3 The Differential|2.3 The Differential]], [[#2.1 Augmented Operads|2.1 Augmented Operads]]
>
> Let $\mathcal{O}$ be a non-dg augmented operad (so $d_1 = 0$). For $\mu \in \bar{\mathcal{O}}(2)$ and $\nu \in \bar{\mathcal{O}}(2)$, the weight-2 part $B(\mathcal{O})^{(2)}(3) = (\bar{\mathcal{O}}[1])^{\circ 2}(3)$ consists of elements $[s\mu; s\nu]$ corresponding to binary trees with 3 leaves. Write down $d_2([s\mu; s\nu])$ explicitly and verify that $d_2^2 = 0$ using the associativity axiom.

> [!TIP]- Solution to Exercise 2
> **Key insight:** There are three binary trees with 3 leaves, corresponding to the three ways to parenthesize $x_1 x_2 x_3$; $d_2$ maps to a sum over weight-1 elements.
>
> **Sketch:** In $B(\mathcal{O})^{(2)}(3)$, there are two planar binary trees: the left-leaning tree (computing $(\mu \circ_1 \nu)$) and the right-leaning tree ($\mu \circ_2 \nu$). The differential $d_2$ contracts the internal edge of each tree using $\gamma$: $d_2([s\mu \circ_1 s\nu]) = s(\mu \circ_1 \nu)$ and $d_2([s\mu \circ_2 s\nu]) = s(\mu \circ_2 \nu)$ (up to sign from the suspension). Then $d_2^2 = 0$ amounts to $\mu \circ_1 (\nu \circ_1 -) = (\mu \circ_1 \nu) \circ_1 -$ and $\mu \circ_2 (\nu \circ_1 -) = (\mu \circ_1 \nu) \circ_2 -$ (sequential and parallel associativity axioms for $\circ_i$), which hold by definition of an operad.

> [!QUESTION] Exercise 3: Augmentation of the bar construction
> *This verifies that $B(\mathcal{O})$ is naturally a cooperad, not just a graded symmetric sequence.*
>
> > **Prerequisites:** [[#1.1 Definition|1.1 Definition]], [[#2.2 The Bar Complex|2.2 The Bar Complex]]
>
> Construct the decomposition map $\Delta: B(\mathcal{O}) \to B(\mathcal{O}) \circ B(\mathcal{O})$ explicitly on tree-monomials by explaining how a rooted tree $T$ decomposes into a "trunk" tree and a collection of "branch" subtrees. Verify this makes $B(\mathcal{O})$ a cooperad.

> [!TIP]- Solution to Exercise 3
> **Key insight:** The decomposition map cuts each tree at a chosen internal edge, producing a "root component" tree and a "leaf component" subtree.
>
> **Sketch:** Given a tree $T$ with vertex labels $(\mu_v)_{v \in V(T)}$, $\Delta(T) = \sum_{v \in V(T)} T_v^{\mathrm{root}} \otimes (T_{v,1}^{\mathrm{leaf}} \otimes \cdots \otimes T_{v,k}^{\mathrm{leaf}})$ where $T_v^{\mathrm{root}}$ is the subtree above $v$ (with $v$ as a new leaf) and $T_{v,i}^{\mathrm{leaf}}$ is the subtree rooted at the $i$-th child of $v$. Coassociativity follows because both $(\Delta \circ \mathrm{id}) \circ \Delta$ and $(\mathrm{id} \circ \Delta) \circ \Delta$ sum over all pairs of edges in $T$, with the two orderings producing the same set of terms.

---

## 3. The Operadic Cobar Construction 📐

### 3.1 Definition

**Definition (Cobar construction).** For an augmented conilpotent dg-cooperad $(\mathcal{C}, d, \Delta, \epsilon)$, the *cobar construction* $\Omega(\mathcal{C})$ is the augmented dg-operad defined as follows.

As a graded operad (ignoring the differential), $\Omega(\mathcal{C}) = \mathcal{F}(\bar{\mathcal{C}}[-1])$, the *free operad* on the desuspension $\bar{\mathcal{C}}[-1]$. Elements are linear combinations of rooted trees with internal vertices labeled by elements of $\bar{\mathcal{C}}[-1]$.

The differential $d_{\Omega(\mathcal{C})} = d_1 + d_2$ where:
- $d_1$: induced by the internal differential $d$ on $\mathcal{C}$.
- $d_2$: the *cocomposition differential*, which increases weight by 1. On a single vertex labeled by $c \in \bar{\mathcal{C}}(n)$, $d_2(s^{-1}c) = \sum s^{-1}c' \circ_i s^{-1}c''$ summing over the components of $\Delta(c)$ (split into a tree with one interior edge).

> [!EXAMPLE] Cobar of Com-dual is Lie-infinity
> The cooperad $\mathrm{Com}^c$ (dual of $\mathrm{Com}$) has $\mathrm{Com}^c(n) = k$ with trivial $S_n$-action. The cobar construction $\Omega(\mathrm{Com}^c)$ is the operad encoding $L_\infty$-algebras, with generators $\ell_n$ in each arity $n \geq 2$ of degree $2-n$. This is the content of the Koszul duality $\mathrm{Com}^! = \mathrm{Lie}^c$.

### 3.2 The Bar-Cobar Adjunction

**Theorem (Bar-cobar adjunction).** There is an adjunction
$$\Omega : \mathbf{dg\text{-}Coop}^{\mathrm{aug, conil}} \rightleftharpoons \mathbf{dg\text{-}Op}^{\mathrm{aug}} : B$$
with unit $\mathcal{C} \to B\Omega(\mathcal{C})$ and counit $\Omega B(\mathcal{O}) \to \mathcal{O}$.

🔑 **The counit is a quasi-isomorphism:** For any augmented dg-operad $\mathcal{O}$, the natural map $\Omega B(\mathcal{O}) \xrightarrow{\sim} \mathcal{O}$ is a quasi-isomorphism. This makes $\Omega B(\mathcal{O})$ a *cofibrant resolution* of $\mathcal{O}$ — a free dg-operad with differential encoding all operadic compositions in $\mathcal{O}$.

> [!QUESTION] Exercise 4: The counit map
> *This unpacks the universal property of the bar-cobar adjunction in the simplest case.*
>
> > **Prerequisites:** [[#3.2 The Bar-Cobar Adjunction|3.2 The Bar-Cobar Adjunction]]
>
> Let $\mathcal{O}$ be a non-dg augmented operad. Describe the counit map $\Omega B(\mathcal{O}) \to \mathcal{O}$ explicitly on generators: what does it send a weight-1 generator $s^{-1}(s\mu) \in \bar{\mathcal{C}}[-1]$ (for $\mu \in \bar{\mathcal{O}}(n)$) to? What is the image of a weight-2 generator?

> [!TIP]- Solution to Exercise 4
> **Key insight:** The counit sends weight-1 generators to the corresponding operation in $\mathcal{O}$, and kills all weight $\geq 2$ generators — which is why $\Omega B(\mathcal{O}) \to \mathcal{O}$ is a surjection onto $\mathcal{O}$ with acyclic kernel.
>
> **Sketch:** A weight-1 generator of $\Omega B(\mathcal{O})$ is $s^{-1}(s\mu) = \mu \in \bar{\mathcal{O}}(n)$ (the suspension and desuspension cancel). The counit sends this to $\mu \in \mathcal{O}(n)$. A weight-2 generator is $s^{-1}(s\mu) \circ_i s^{-1}(s\nu)$ for $\mu \in \bar{\mathcal{O}}(m)$, $\nu \in \bar{\mathcal{O}}(n)$; the counit sends this to $\mu \circ_i \nu \in \mathcal{O}(m+n-1)$. But in $\Omega B(\mathcal{O})$ there is also the differential term $d_2(s^{-1}s(\mu \circ_i \nu)) = s^{-1}(s\mu) \circ_i s^{-1}(s\nu) - s^{-1}s(\mu \circ_i \nu)$, which shows these weight-2 elements are exact in the acyclic complex $\ker(\Omega B(\mathcal{O}) \to \mathcal{O})$.

> [!QUESTION] Exercise 5: Acyclicity of the bar complex of Com
> *This is a concrete computation showing $B(\mathrm{Com})$ has the right homology, illustrating the bar-cobar quasi-isomorphism.*
>
> > **Prerequisites:** [[#2.2 The Bar Complex|2.2 The Bar Complex]], [[#2.3 The Differential|2.3 The Differential]]
>
> Compute the homology of $B(\mathrm{Com})(n)$ in weight 1 and weight 2 for $n = 2, 3$. (Recall $\bar{\mathrm{Com}}(n) = k$ for $n \geq 2$.) Identify the pattern and state what it predicts for the Koszul criterion.

> [!TIP]- Solution to Exercise 5
> **Key insight:** $H_\bullet(B(\mathrm{Com})(n))$ is concentrated in weight $n-1$, the minimum possible weight, with $H = k$ in that weight and zero elsewhere — exactly the Koszul condition.
>
> **Sketch:** $B(\mathrm{Com})(2)$: only weight 1, one generator $[s\mu_2]$ where $\mu_2 \in \bar{\mathrm{Com}}(2) = k$. No differential. $H = k$ in weight 1. For $n=3$, weight 1: one generator $[s\mu_3]$. Weight 2: three generators $[s\mu_2 \circ_i s\mu_2]$ for $i=1,2$ (two binary trees with 3 leaves), but $\mathrm{Com}$ is symmetric so all $S_3$-orbits collapse to two basis elements. The differential $d_2([s\mu_3]) = 0$ (weight 1 has no decomposition) and $d_2([s\mu_2 \circ_1 s\mu_2]) = [s\mu_3]$, $d_2([s\mu_2 \circ_2 s\mu_2]) = [s\mu_3]$ (up to sign). Thus $d_2$ on weight 2 maps onto weight 1; the kernel in weight 2 is the difference $[s\mu_2 \circ_1 s\mu_2] - [s\mu_2 \circ_2 s\mu_2]$, which has no further boundary. So $H$ is concentrated in weight 2 = $n-1$ for $n=3$, as required.

---

## 4. Twisting Morphisms 💡

### 4.1 The Maurer-Cartan Equation

The bar-cobar adjunction is best understood through *twisting morphisms*, which are the "structure constants" mediating between cooperads and operads.

**Definition (Twisting morphism).** Let $\mathcal{C}$ be an augmented conilpotent dg-cooperad and $\mathcal{O}$ an augmented dg-operad. A *twisting morphism* $\alpha \in \mathrm{Hom}_{\mathbf{SymSeq}}(\bar{\mathcal{C}}, \bar{\mathcal{O}})$ of degree $-1$ is a collection of $S_n$-equivariant maps $\alpha(n): \bar{\mathcal{C}}(n) \to \bar{\mathcal{O}}(n)$ of degree $-1$ satisfying the *Maurer–Cartan equation*:
$$\partial(\alpha) + \gamma \circ (\alpha \circ \alpha) \circ \Delta = 0 \quad \in \mathrm{Hom}_{\mathbf{SymSeq}}(\bar{\mathcal{C}}, \bar{\mathcal{O}}).$$

Here $\partial(\alpha) = d_\mathcal{O} \circ \alpha - (-1)^{|\alpha|} \alpha \circ d_\mathcal{C}$ is the chain-complex differential on $\mathrm{Hom}$, and $\gamma \circ (\alpha \circ \alpha) \circ \Delta$ encodes the sum over all binary decompositions of $\mathcal{C}$ followed by applying $\alpha$ twice and composing in $\mathcal{O}$.

The set of twisting morphisms is denoted $\mathrm{Tw}(\mathcal{C}, \mathcal{O})$.

> [!NOTE] The convolution pre-Lie algebra
> The space $\mathrm{Hom}_{\mathbf{SymSeq}}(\bar{\mathcal{C}}, \bar{\mathcal{O}})$ carries a *convolution pre-Lie algebra* structure: $(f \star g)(c) = \gamma(\mathrm{id} \circ (f \otimes g))(\Delta(c))$. The Maurer–Cartan equation is $\partial(\alpha) + \alpha \star \alpha = 0$ in this algebra. This is the operadic analogue of the Maurer–Cartan equation $dA + A \wedge A = 0$ in gauge theory.

### 4.2 The Universal Twisting Morphisms

**Theorem (Representability).** The functor $\mathrm{Tw}(-, \mathcal{O})$ is representable by $B(\mathcal{O})$, and $\mathrm{Tw}(\mathcal{C}, -)$ is representable by $\Omega(\mathcal{C})$:
$$\mathrm{Tw}(\mathcal{C}, \mathcal{O}) \cong \mathrm{Hom}_{\mathbf{dg\text{-}Op}}(\Omega\mathcal{C}, \mathcal{O}) \cong \mathrm{Hom}_{\mathbf{dg\text{-}Coop}}(\mathcal{C}, B\mathcal{O}).$$

The *universal twisting morphism from the bar construction* is $\pi: B(\mathcal{O}) \to \mathcal{O}$, the map projecting onto weight-1 trees. The *universal twisting morphism into the cobar construction* is $\iota: \mathcal{C} \to \Omega(\mathcal{C})$, the inclusion as weight-1 generators.

🔑 **A twisting morphism $\alpha: \mathcal{C} \to \mathcal{O}$ is a quasi-isomorphism iff the twisted composite $\mathcal{C} \circ_\alpha \mathcal{O}$ (with differential $d_\alpha = d + \alpha$) is acyclic.** This is the key criterion underlying Koszul duality.

> [!QUESTION] Exercise 6: The Maurer-Cartan equation for the identity
> *This shows that the identity morphism $\mathrm{id}: \mathcal{O}^! \to \mathcal{O}^!$ does not in general solve the Maurer-Cartan equation, distinguishing the canonical twisting morphism $\kappa$ from naive candidates.*
>
> > **Prerequisites:** [[#4.1 The Maurer-Cartan Equation|4.1 The Maurer-Cartan Equation]]
>
> Let $\mathcal{O}$ be a non-dg quadratic operad with $\mathcal{O}^!$ its Koszul dual. Explain why the degree-0 identity map $\mathrm{id}: \bar{\mathcal{O}}^! \to \bar{\mathcal{O}}^!$ cannot be a twisting morphism $\mathcal{O}^! \to \mathcal{O}^!$. Then explain why the canonical $\kappa: \mathcal{O}^! \to \mathcal{O}$ (of degree $-1$) can satisfy the Maurer–Cartan equation.

> [!TIP]- Solution to Exercise 6
> **Key insight:** Twisting morphisms must have degree $-1$; the identity has degree 0. The degree shift from the Koszul dual construction ($sE^\vee$ instead of $E^\vee$) is exactly what gives $\kappa$ the correct degree.
>
> **Sketch:** The Maurer–Cartan equation requires $\alpha$ to have degree $-1$ (so that $\alpha \star \alpha$ has degree $-2$ and $\partial(\alpha)$ has degree $-1 - 1 = -2$, making the equation degree-homogeneous at $-2$). The identity has degree 0, so $\partial(\mathrm{id}) = 0$ but $\mathrm{id} \star \mathrm{id}$ has degree 0, and the equation $0 + \mathrm{id} \star \mathrm{id} = 0$ says $\gamma \circ \Delta = 0$, which is false for non-trivial cooperads. The canonical $\kappa: \mathcal{O}^!(n) \to \mathcal{O}(n)$ is defined on generators $sE^\vee \to E$ by the degree-$(-1)$ evaluation pairing, and satisfies MC because the quadratic relations of $\mathcal{O}^!$ are the orthogonal complement of those of $\mathcal{O}$.

---

## 5. Quadratic Operads and the Koszul Dual 📐

### 5.1 Quadratic Operads

Most operads encountered in practice are *quadratic*: generated by binary operations with relations of degree 2.

**Definition (Quadratic operad).** A *quadratic operad* is an operad of the form $\mathcal{O} = \mathcal{F}(E)/(R)$, where:
- $E \in \mathbf{SymSeq}$ is a symmetric sequence concentrated in arity 2 (the *generators*),
- $R \subseteq \mathcal{F}(E)(3)$ is an $S_3$-submodule (the *relations*), and
- $(R)$ is the operadic ideal generated by $R$.

We write $\mathcal{O} = \mathcal{F}(E, R)$ for the quadratic operad with generators $E$ and relations $R$.

> [!EXAMPLE] Ass, Com, Lie as quadratic operads
> - $\mathrm{Ass} = \mathcal{F}(E_\mathrm{Ass}, R_\mathrm{Ass})$ with $E_\mathrm{Ass}(2) = k[S_2]$ (two binary operations $\mu$ and $\mu \cdot (12)$) and $R_\mathrm{Ass} = \langle \mu \circ_1 \mu - \mu \circ_2 \mu \rangle$ (associativity).
> - $\mathrm{Com} = \mathcal{F}(E_\mathrm{Com}, R_\mathrm{Com})$ with $E_\mathrm{Com}(2) = k$ (one binary symmetric operation) and $R_\mathrm{Com} = \langle \mu \circ_1 \mu - \mu \circ_2 \mu \rangle$ (associativity; commutativity is imposed by the trivial $S_2$-action).
> - $\mathrm{Lie} = \mathcal{F}(E_\mathrm{Lie}, R_\mathrm{Lie})$ with $E_\mathrm{Lie}(2) = k \cdot [-, -]$ (antisymmetric bracket, $S_2$ acts by $-1$) and $R_\mathrm{Lie}$ the Jacobi identity $[[a,b],c] + [[b,c],a] + [[c,a],b] = 0$.

### 5.2 The Koszul Dual Cooperad

**Definition (Koszul dual cooperad).** For a quadratic operad $\mathcal{O} = \mathcal{F}(E, R)$, the *Koszul dual cooperad* is
$$\mathcal{O}^! = \mathcal{F}^c(sE^\vee, s^2 R^\perp),$$
the quadratic cooperad with cogenerators $sE^\vee$ (the suspension of the linear dual of $E$) and corelations $s^2 R^\perp$, where $R^\perp \subseteq \mathcal{F}(E^\vee)(3)$ is the annihilator of $R$ under the natural pairing $\mathcal{F}(E^\vee)(3) \otimes \mathcal{F}(E)(3) \to k$.

The suspension shifts are: $(sE^\vee)(2) = E^\vee(2)[-1]$, so if $E(2)$ is concentrated in degree 0, then $(sE^\vee)(2)$ is in degree $+1$. This gives $\mathcal{O}^!(n)$ a homological degree of $n-1$ (for $n$-ary cooperations).

> [!WARNING] The Koszul dual operad $\mathcal{O}^*$
> There is a related but distinct notion, the *Koszul dual operad* $\mathcal{O}^* = \mathcal{F}(E^\vee, R^\perp)$ (without the suspension). The cooperad $\mathcal{O}^!$ is related by $\mathcal{O}^! = s\mathcal{O}^*$ (suspension of the dual operad, viewed as a cooperad). Loday–Vallette use $\mathcal{O}^!$ for the cooperad; some sources write $\mathcal{O}^*$ for the dual operad. *Be careful not to confuse them.*

### 5.3 The Canonical Twisting Morphism

There is a canonical degree-$(-1)$ map $\kappa: \mathcal{O}^! \to \mathcal{O}$, defined on cogenerators $sE^\vee \to E$ by the evaluation pairing $(se^\vee)(e) = e^\vee(e) \in k$, extended to all of $\mathcal{O}^!$ by the universal property of the cofree cooperad, and zero on all weight $\geq 2$.

**Proposition.** $\kappa: \mathcal{O}^! \to \mathcal{O}$ is a twisting morphism: it satisfies the Maurer–Cartan equation $\partial(\kappa) + \gamma \circ (\kappa \circ \kappa) \circ \Delta = 0$.

*Proof sketch.* The equation in weight 2 reduces to $\gamma(e^\vee_1 \otimes e^\vee_2)(\Delta(c)) = 0$ for all $c \in \mathcal{O}^!(3)$. Since $c \in s^2 R^\perp$ by definition, and $R^\perp$ annihilates $R$, this is exactly the condition that the image of $\gamma \circ (\kappa \circ \kappa)$ lies in the image of the relations of $\mathcal{O}$, which is zero in $\mathcal{O}$. □

### 5.4 Examples: Ass, Com, and Lie

| Operad | Generators | Relations | Koszul dual $\mathcal{O}^!$ |
|--------|-----------|-----------|--------------------------|
| $\mathrm{Ass}$ | $k[S_2]$ | Associativity | $\mathrm{Ass}^c$ (self-dual) |
| $\mathrm{Com}$ | $k$ (trivial) | Associativity + commutativity | $\mathrm{Lie}^c$ |
| $\mathrm{Lie}$ | $k \cdot [-,-]$ (sign rep) | Jacobi | $\mathrm{Com}^c$ |

🔑 **The Com–Lie Koszul duality** $\mathrm{Com}^! = \mathrm{Lie}^c$ and $\mathrm{Lie}^! = \mathrm{Com}^c$ is the operadic shadow of the classical PBW theorem and the Chevalley–Eilenberg complex.

> [!QUESTION] Exercise 7: The Koszul dual of Ass
> *This establishes the self-duality of $\mathrm{Ass}$ and clarifies why it is special among the classical operads.*
>
> > **Prerequisites:** [[#5.2 The Koszul Dual Cooperad|5.2 The Koszul Dual Cooperad]], [[#5.1 Quadratic Operads|5.1 Quadratic Operads]]
>
> Let $\mathrm{Ass} = \mathcal{F}(k[S_2], R)$ with $R = \langle \mu \circ_1 \mu - \mu \circ_2 \mu \rangle \subseteq \mathcal{F}(k[S_2])(3) \cong k[S_3]^{\oplus 2}$. Compute $R^\perp \subseteq \mathcal{F}(k[S_2]^\vee)(3)$ and show that $\mathrm{Ass}^! \cong \mathrm{Ass}^c$ as cooperads.

> [!TIP]- Solution to Exercise 7
> **Key insight:** $R$ and $R^\perp$ have the same dimension (both $\frac{1}{2}$-dimensional in each $S_3$-isotypic component), and the pairing swaps them, giving $\mathrm{Ass}^* \cong \mathrm{Ass}$.
>
> **Sketch:** $\mathcal{F}(k[S_2])(3) \cong k[S_3]$ (the free associative operad in arity 3 has basis all parenthesizations of 3 inputs, i.e., $S_3$ elements). The relation space $R = \ker(\mathcal{F}(k[S_2])(3) \to \mathrm{Ass}(3))$: since $\mathrm{Ass}(3) = k[S_3]$ and the map is an isomorphism (no relations kill things), we have $R = 0$! Wait — more carefully, $\mathrm{Ass}(3) = k[S_3]$ and $\mathcal{F}(k[S_2])(3) = k[S_3]^2$ (two planar binary trees), so $R = \langle (\sigma, -\sigma) : \sigma \in S_3\rangle$, a rank-6 subspace of the rank-12 space $k[S_3]^2$. Then $R^\perp$ has rank 6 as well and one checks the natural pairing identifies it with a copy of $k[S_3]$ — i.e., $\mathrm{Ass}^*$ is again generated by $k[S_2]^\vee \cong k[S_2]$ with the same relations. So $\mathrm{Ass}^! \cong \mathrm{Ass}^c$.

> [!QUESTION] Exercise 8: Koszul dual of Com and the Lie operad
> *This establishes the Com-Lie Koszul duality at the level of generators and relations.*
>
> > **Prerequisites:** [[#5.2 The Koszul Dual Cooperad|5.2 The Koszul Dual Cooperad]]
>
> Show that $\mathrm{Com}^! \cong \mathrm{Lie}^c$: identify the cogenerators of $\mathrm{Com}^!$ (as a cooperad) with the generators of $\mathrm{Lie}$ (as an operad), and show that the corelations of $\mathrm{Com}^!$ are the dual of the Jacobi identity.

> [!TIP]- Solution to Exercise 8
> **Key insight:** The annihilator of the associativity+commutativity relations in $\mathcal{F}(k)(3) = k[S_3]$ is spanned by the antisymmetrizer and its cyclic permutations — which is exactly the Jacobi identity in antisymmetric form.
>
> **Sketch:** $E_\mathrm{Com} = k$ with trivial $S_2$-action, so $E_\mathrm{Com}^\vee = k$ with trivial $S_2$-action, but $sE_\mathrm{Com}^\vee = k[-1]$ with $S_2$ acting by the sign representation (the suspension of a trivial representation under the Koszul sign rule becomes the sign representation). This is exactly $E_\mathrm{Lie}$. The corelation space $s^2 R_\mathrm{Com}^\perp \subseteq \mathcal{F}(E_\mathrm{Lie})(3)$: one computes $R_\mathrm{Com}^\perp = \langle e_1 \otimes e_2 \circ_1 e_3 + e_2 \otimes e_3 \circ_1 e_1 + e_3 \otimes e_1 \circ_1 e_2 \rangle$, which is the Jacobi identity. So the quadratic operad with these generators and relations is $\mathrm{Lie}$, confirming $\mathrm{Com}^* \cong \mathrm{Lie}$ and $\mathrm{Com}^! \cong \mathrm{Lie}^c$.

---

## 6. The Koszul Criterion 🔑

### 6.1 Statement

**Definition (Koszul operad).** A quadratic operad $\mathcal{O}$ is *Koszul* if the canonical twisting morphism $\kappa: \mathcal{O}^! \to \mathcal{O}$ is a quasi-isomorphism, i.e., if $H_\bullet(\mathcal{O}^! \circ_\kappa \mathcal{O}) \cong \mathbf{1}$ (the unit symmetric sequence).

**Theorem (Equivalent criteria).** The following are equivalent for a quadratic operad $\mathcal{O}$:

1. $\kappa: \mathcal{O}^! \to \mathcal{O}$ is a quasi-isomorphism.
2. The twisted composite product $\mathcal{O}^! \circ_\kappa \mathcal{O}$ (the *Koszul complex*) is acyclic.
3. $H_\bullet(B(\mathcal{O}))$ is concentrated in weight $= $ homological degree (i.e., pure weight).
4. For each $n$, the component $\mathcal{O}(n)$ has a basis given by a PBW-type basis of decorated trees.

> [!INFO] Why Koszulness is hard to check in general
> Criteria 1–3 are equivalent but not directly computable without knowing the full bar complex. Criterion 4 (the PBW/rewriting criterion, due to Dotsenko–Khoroshkin and Loday–Vallette) gives an algorithmic check: $\mathcal{O}$ is Koszul iff its relations form a Gröbner basis for the free operad, equivalently iff a quadratic PBW basis exists. This is how one proves that $\mathrm{Perm}$, $\mathrm{PreLie}$, and $\mathrm{Dend}$ are Koszul.

### 6.2 Koszulness of Ass and Com

**Theorem.** The operads $\mathrm{Ass}$, $\mathrm{Com}$, and $\mathrm{Lie}$ are Koszul.

*Proof for $\mathrm{Ass}$.* The Koszul complex $\mathrm{Ass}^c \circ_\kappa \mathrm{Ass}$ has components $(\mathrm{Ass}^c \circ_\kappa \mathrm{Ass})(n)$. By direct computation using the fact that $\mathrm{Ass}^c(k) = k[S_k]$ and $\mathrm{Ass}(n) = k[S_n]$, the complex in arity $n$ is isomorphic to the augmented chain complex of the $(n-2)$-simplex $\Delta^{n-2}$ (tensored with $k[S_n]$). Since $\Delta^{n-2}$ is contractible, the complex is acyclic. □

*Proof sketch for $\mathrm{Com}$.* The Koszul complex $\mathrm{Lie}^c \circ_\kappa \mathrm{Com}$ in arity $n$ is isomorphic (as a chain complex of $S_n$-modules) to the top-weight part of the homology of the partition lattice $\Pi_n$. By a theorem of Sundaram, this is acyclic in all degrees except the top, where $H_{n-2}(\Pi_n) = \mathrm{Lie}(n)$ (the representation of $S_n$ on the free Lie algebra). This confirms $H_\bullet(\mathrm{Lie}^c \circ_\kappa \mathrm{Com}) \cong \mathbf{1}$, so $\mathrm{Com}$ is Koszul. □

> [!QUESTION] Exercise 9: The Koszul complex of Ass in arity 3
> *This makes the proof of Koszulness of Ass concrete in the simplest nontrivial case.*
>
> > **Prerequisites:** [[#6.1 Statement|6.1 Statement]], [[#5.4 Examples: Ass, Com, and Lie|5.4 Examples: Ass, Com, and Lie]]
>
> Write out the Koszul complex $(\mathrm{Ass}^c \circ_\kappa \mathrm{Ass})(3)$ explicitly: identify the chain groups in each weight, write the differential $d_\kappa$, and verify it is acyclic. Identify it as the chain complex of the 1-simplex $\Delta^1$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** The complex has two chain groups (weight 1 and weight 2), and the differential is an isomorphism — the chain complex of an interval.
>
> **Sketch:** $(\mathrm{Ass}^c \circ_\kappa \mathrm{Ass})(3)$: weight 1 part is $\mathrm{Ass}^c(3) \otimes_{S_3} \mathrm{Ass}(1)^{\otimes 3} \oplus \cdots$, but the leading term is $\mathrm{Ass}^c(1) \cdot \mathrm{Ass}(3) = k[S_3]$ (one vertex, all 3 inputs entering directly) and $\mathrm{Ass}^c(3) \cdot \mathrm{Ass}(1)^3 = k[S_3]$ in weight 1. Actually weight 1 is $\bar{\mathrm{Ass}}^c(3) \otimes 1 \cong k[S_3]$ and weight 2 is $\bigoplus_{n_1+n_2=3} \bar{\mathrm{Ass}}^c(2) \otimes (\bar{\mathrm{Ass}}(n_1) \otimes \bar{\mathrm{Ass}}(n_2)) \cong k[S_3] \oplus k[S_3]$ (two splittings: $(2,1)$ and $(1,2)$). The differential $d_\kappa$ on the weight-2 generators is $\kappa$ applied to the cooperadic generator: $d_\kappa(\mu^* \otimes \nu_1 \otimes \nu_2) = \kappa(\mu^*) \circ (\nu_1 \otimes \nu_2) = \mu \circ (\nu_1 \otimes \nu_2)$. Computing the chain groups as $k[S_3]$ in weight 1 and $k[S_3]^2$ in weight 2, the differential is the map $(\mathrm{id}, -\mathrm{id}): k[S_3]^2 \to k[S_3]$, which is surjective with kernel $k[S_3]$ (the diagonal). The kernel is the image of the weight-3 differential (which is zero since there is no weight 3 for $n=3$). So $H_0 = 0$ and $H_{-1} = 0$ except at weight = homological degree, confirming acyclicity. This is the chain complex $k[S_3] \xrightarrow{\sim} k[S_3]$ of $\Delta^1 \otimes k[S_3]$.

> [!QUESTION] Exercise 10: A non-Koszul quadratic operad
> *This gives an example where the Koszul criterion fails, illustrating that Koszulness is a non-trivial condition.*
>
> > **Prerequisites:** [[#6.1 Statement|6.1 Statement]]
>
> The *pre-Lie operad* $\mathrm{PreLie}$ is generated by one binary operation $a \rhd b$ with the single relation $(a \rhd b) \rhd c - a \rhd (b \rhd c) = (a \rhd c) \rhd b - a \rhd (c \rhd b)$. Show this is a quadratic operad. State (without proof) whether it is Koszul, and identify its Koszul dual.

> [!TIP]- Solution to Exercise 10
> **Key insight:** $\mathrm{PreLie}$ is Koszul, with Koszul dual $\mathrm{Perm}$ (the permutation operad), reflecting the fact that pre-Lie algebras are the correct framework for Rota–Baxter structures.
>
> **Sketch:** The pre-Lie relation is a quadratic relation in $\mathcal{F}(E)(3)$ where $E(2) = k \cdot \mu$ (one non-symmetric binary operation). The Koszul dual has generators $k \cdot \mu^\vee$ with the orthogonal complement relation, which turns out to be $\mathrm{Perm}$: the operad encoding algebras with a bilinear product satisfying $(a \cdot b) \cdot c = a \cdot (b \cdot c)$ and $(a \cdot b) \cdot c = (a \cdot c) \cdot b$ (the permutation associativity). $\mathrm{PreLie}$ is Koszul by the rewriting criterion (Dotsenko–Khoroshkin): its generators form a Gröbner basis for the free operad with the appropriate monomial order.

---

## 7. Homotopy Algebras from Koszul Duality 💡

### 7.1 The Resolution O-infinity

For a Koszul operad $\mathcal{O}$, the bar-cobar adjunction produces a canonical cofibrant resolution.

**Definition ($\mathcal{O}_\infty$).** For a Koszul operad $\mathcal{O}$, the *homotopy $\mathcal{O}$-operad* is
$$\mathcal{O}_\infty := \Omega(\mathcal{O}^!),$$
the cobar construction on the Koszul dual cooperad.

**Theorem.** The canonical map $\mathcal{O}_\infty = \Omega(\mathcal{O}^!) \xrightarrow{\sim} \mathcal{O}$ induced by $\kappa: \mathcal{O}^! \to \mathcal{O}$ is a quasi-isomorphism of dg-operads. Moreover, $\mathcal{O}_\infty$ is a *free* graded operad (as a graded operad, ignoring the differential), hence *cofibrant* in the model structure on dg-operads.

> [!NOTE] Why the cobar resolution is better than the bar-cobar resolution
> Both $\Omega B(\mathcal{O}) \xrightarrow{\sim} \mathcal{O}$ and $\Omega(\mathcal{O}^!) \xrightarrow{\sim} \mathcal{O}$ are cofibrant resolutions. The bar-cobar resolution $\Omega B(\mathcal{O})$ always exists but is enormous (exponential in the weight). The Koszul resolution $\Omega(\mathcal{O}^!)$ is *much smaller* — it uses only $\mathcal{O}^!$ rather than all of $B(\mathcal{O})$. Koszulness is precisely the condition that these two resolutions are quasi-isomorphic.

### 7.2 A-infinity and L-infinity Algebras

The two most important instances:

**$A_\infty$-algebras.** Since $\mathrm{Ass}$ is Koszul and $\mathrm{Ass}^! = \mathrm{Ass}^c$,
$$A_\infty = \Omega(\mathrm{Ass}^c).$$
A graded module $A$ is an $A_\infty$-algebra iff it carries operations $m_n: A^{\otimes n} \to A$ of degree $2-n$ ($n \geq 1$) satisfying the *Stasheff relations*:
$$\sum_{r+s+t=n} (-1)^{rs+t} m_{r+1+t}(\mathrm{id}^{\otimes r} \otimes m_s \otimes \mathrm{id}^{\otimes t}) = 0 \quad \forall n \geq 1.$$
The first few: $m_1^2 = 0$ (so $d = m_1$ is a differential), $m_1 m_2 + m_2(m_1 \otimes \mathrm{id} + \mathrm{id} \otimes m_1) = 0$ (so $m_2$ is a chain map), and $m_2(\mathrm{id} \otimes m_2) - m_2(m_2 \otimes \mathrm{id}) = m_1 m_3 + m_3(\text{boundary}) + \cdots$ (so $m_2$ is homotopy-associative with $m_3$ as the homotopy).

See [[concepts/category-theory/operads/algebras-modules|Algebras and Modules]] §7 for the full treatment.

**$L_\infty$-algebras.** Since $\mathrm{Lie}$ is Koszul and $\mathrm{Lie}^! = \mathrm{Com}^c$,
$$L_\infty = \Omega(\mathrm{Com}^c).$$
An $L_\infty$-algebra is a graded module with antisymmetric brackets $\ell_n: \Lambda^n A \to A$ of degree $2-n$ satisfying generalized Jacobi identities. An ordinary dg-Lie algebra is the special case $\ell_n = 0$ for $n \geq 3$. $L_\infty$-algebras are the correct notion of a "Lie algebra up to all higher homotopies" and govern formal deformation problems via the Maurer–Cartan equation $\sum_n \frac{1}{n!}\ell_n(\gamma^{\otimes n}) = 0$.

### 7.3 The Homotopy Transfer Theorem

The key application of $\mathcal{O}_\infty$-algebras is that they transfer along deformation retracts.

**Theorem (Homotopy Transfer).** Let $A$ be an $\mathcal{O}_\infty$-algebra and suppose there is a diagram
$$\begin{tikzcd} H \arrow[r, shift left, "i"] & A \arrow[l, shift left, "p"] \end{tikzcd} \quad \text{with } pi = \mathrm{id}_H, \quad ip \simeq \mathrm{id}_A \text{ (chain homotopy } h: A \to A[-1]).$$
Then $H$ carries an induced $\mathcal{O}_\infty$-algebra structure, and $i$ extends to an $\infty$-quasi-isomorphism of $\mathcal{O}_\infty$-algebras.

The transferred operations on $H$ are given by explicit *tree formulas*: the $n$-ary operation on $H$ is a sum over binary trees with $n$ leaves, where each leaf is labeled by $i$, each internal vertex by an operation of $A$, and each internal edge by the homotopy $h$.

> [!EXAMPLE] Minimal model of a dg-algebra
> If $A$ is a dg-associative algebra and $H = H^\bullet(A)$ with the zero differential, the homotopy transfer theorem gives $H$ the structure of an $A_\infty$-algebra (the *minimal model* of $A$) with $m_1 = 0$, $m_2 = $ induced product, and $m_n$ ($n \geq 3$) the higher Massey products. The $A_\infty$-algebra $H$ is quasi-isomorphic to $A$ as an $A_\infty$-algebra. *This is why cohomology "knows" the quasi-isomorphism type of $A$.*

> [!QUESTION] Exercise 11: Transfer of an A-infinity structure
> *This makes the homotopy transfer theorem concrete for the simplest nontrivial case.*
>
> > **Prerequisites:** [[#7.3 The Homotopy Transfer Theorem|7.3 The Homotopy Transfer Theorem]], [[#7.2 A-infinity and L-infinity Algebras|7.2 A-infinity and L-infinity Algebras]]
>
> Let $A$ be a dg-associative algebra with a deformation retract $(i, p, h)$ onto $H = H^\bullet(A)$. Write down the formula for the transferred operation $m_3: H^{\otimes 3} \to H$ using the tree formula, and identify it as a Massey product.

> [!TIP]- Solution to Exercise 11
> **Key insight:** $m_3(x,y,z) = p(h(i(x) \cdot i(y)) \cdot i(z)) \pm p(i(x) \cdot h(i(y) \cdot i(z)))$, which is exactly the triple Massey product $\langle [x], [y], [z] \rangle$.
>
> **Sketch:** The tree formula for $m_3$ sums over the two binary trees with 3 leaves and one internal edge. Tree 1 (left-leaning): $p \circ m_2^A \circ (h \circ m_2^A \circ (i \otimes i) \otimes i)$, giving $p(h(i(x) \cdot i(y)) \cdot i(z))$. Tree 2 (right-leaning): $p \circ m_2^A \circ (i \otimes h \circ m_2^A \circ (i \otimes i))$, giving $(-1)^\star p(i(x) \cdot h(i(y) \cdot i(z)))$ with sign from the Koszul rule. The sum is the Massey product $\langle x, y, z \rangle$ (when $xy = yz = 0$ in cohomology, so the Massey product is defined); more precisely, $m_3$ computes the indeterminacy-free Massey product, which is always defined for an $A_\infty$-algebra.

> [!QUESTION] Exercise 12: L-infinity algebras and deformation functors
> *This connects L-infinity algebras to the deformation theory of algebraic structures.*
>
> > **Prerequisites:** [[#7.2 A-infinity and L-infinity Algebras|7.2 A-infinity and L-infinity Algebras]]
>
> Let $\mathfrak{g}$ be an $L_\infty$-algebra (concentrated in non-negative degrees). The *Maurer–Cartan set* is $\mathrm{MC}(\mathfrak{g}) = \{\gamma \in \mathfrak{g}^1 : \sum_{n \geq 1} \frac{1}{n!} \ell_n(\gamma^{\otimes n}) = 0\}$. Show that when $\mathfrak{g}$ is an ordinary dg-Lie algebra (i.e., $\ell_n = 0$ for $n \geq 3$), this reduces to the classical Maurer–Cartan equation $d\gamma + \frac{1}{2}[\gamma, \gamma] = 0$.

> [!TIP]- Solution to Exercise 12
> **Key insight:** For a dg-Lie algebra, only the $n=1$ and $n=2$ terms survive in the $L_\infty$ MC equation, giving exactly $\ell_1(\gamma) + \frac{1}{2}\ell_2(\gamma, \gamma) = d\gamma + \frac{1}{2}[\gamma, \gamma] = 0$.
>
> **Sketch:** The $L_\infty$ MC equation is $\sum_{n=1}^\infty \frac{1}{n!} \ell_n(\gamma^{\otimes n}) = 0$. For $n=1$: $\ell_1 = d$ (the differential). For $n=2$: $\frac{1}{2}\ell_2(\gamma,\gamma) = \frac{1}{2}[\gamma,\gamma]$. For $n \geq 3$: $\ell_n = 0$ by assumption. So the equation is $d\gamma + \frac{1}{2}[\gamma,\gamma] = 0$. The factor of $\frac{1}{2}$ absorbs the antisymmetry $[\gamma,\gamma] = -[\gamma,\gamma]$: since $\gamma \in \mathfrak{g}^1$ is degree 1, the Koszul sign gives $[\gamma,\gamma] \neq 0$ in general (it equals $2\gamma^2$ in the universal enveloping algebra).

---

## 8. Operadic Cohomology and Deformation Theory 🔑

### 8.1 The Two-Sided Bar Resolution

The bar construction extends to a two-sided version that resolves an algebra $A$ as a module over itself.

**Definition (Two-sided bar resolution).** For an $\mathcal{O}$-algebra $A$ and $A$-modules $M$, $N$ (left and right respectively), the *two-sided bar complex* is
$$B(\mathcal{O}, A, A) := \mathcal{O} \circ_\kappa A,$$
with $A$ playing the role of both left and right arguments. As a graded symmetric sequence,
$$B(\mathcal{O}, A, A)(n) = \bigoplus_k \mathcal{O}^!(k) \otimes_{S_k} A^{\otimes k} \otimes \cdots$$
with differential combining the internal differential on $A$ with the twisted differential from $\kappa$.

🔑 **$B(\mathcal{O}, A, A)$ is a free resolution of $A$ as an $\mathcal{O}$-algebra** (when $\mathcal{O}$ is Koszul). This replaces the classical bar resolution $A \otimes A^{\otimes \bullet} \otimes A \to A$ for associative algebras.

**Definition (Operadic cohomology).** The *operadic cohomology* of an $\mathcal{O}$-algebra $A$ with coefficients in an $A$-module $M$ is
$$H^\bullet_\mathcal{O}(A, M) := H^\bullet\bigl(\mathrm{Hom}_{\mathcal{O}\text{-bimod}}(B(\mathcal{O}, A, A), M)\bigr).$$

### 8.2 Classical Cohomology Theories as Special Cases

| Operad $\mathcal{O}$ | $\mathcal{O}$-algebra $A$ | $H^\bullet_\mathcal{O}(A, M)$ |
|---------------------|--------------------------|-------------------------------|
| $\mathrm{Ass}$ | Associative algebra | Hochschild cohomology $HH^\bullet(A, M)$ |
| $\mathrm{Com}$ | Commutative algebra | Harrison cohomology $\cong$ André–Quillen cohomology $AQ^\bullet(A, M)$ |
| $\mathrm{Lie}$ | Lie algebra $\mathfrak{g}$ | Chevalley–Eilenberg cohomology $H^\bullet_\mathrm{CE}(\mathfrak{g}, M)$ |

The operadic framework unifies all three under a single definition and makes transparent why they satisfy the same formal properties (long exact sequences, cup products when $M = A$, Gerstenhaber algebra structure, etc.).

> [!NOTE] Harrison vs André-Quillen
> Harrison cohomology (defined via the anti-symmetrization of the Hochschild complex) and André–Quillen cohomology (the derived functor of derivations for commutative algebras) agree over $\mathbb{Q}$ but differ in characteristic $p$ (where Harrison lacks the higher-order terms). The operadic cohomology $H^\bullet_\mathrm{Com}(A,M)$ recovers Harrison cohomology; André–Quillen cohomology arises from the full cotangent complex $\mathbb{L}_{A/k}$.

### 8.3 The Deformation Complex of an Operad

The bar construction also computes deformations of the operad $\mathcal{O}$ itself.

**Definition (Deformation complex).** The *deformation complex* of a Koszul operad $\mathcal{O}$ is
$$\mathrm{Def}(\mathcal{O}) := \mathrm{Hom}_{\mathbf{SymSeq}}(\mathcal{O}^!, \mathcal{O})$$
with the *convolution $L_\infty$-algebra structure*: the bracket $[\alpha, \beta] = \alpha \star \beta - (-1)^{|\alpha||\beta|} \beta \star \alpha$ where $\star$ is the convolution pre-Lie product from §4.1, and higher brackets from the $L_\infty$ structure on the convolution algebra.

**Theorem.** The Maurer–Cartan elements of $\mathrm{Def}(\mathcal{O})$ are in bijection with *deformations of the operad structure on $\mathcal{O}$*: dg-operad structures on $\mathcal{O}[[t]]$ that reduce to $\mathcal{O}$ modulo $t$.

The cohomology groups of $\mathrm{Def}(\mathcal{O})$ control:
- $H^0$: symmetries/automorphisms of $\mathcal{O}$ (infinitesimal)
- $H^1$: infinitesimal deformations of $\mathcal{O}$ as an operad
- $H^2$: obstructions to extending deformations

> [!EXAMPLE] The Kontsevich formality theorem via deformation complexes
> The Kontsevich formality theorem states that the dg-Lie algebra of polyvector fields on $\mathbb{R}^n$ (the Hochschild complex $C^\bullet(C^\infty, C^\infty)$ with Gerstenhaber bracket) is formal — quasi-isomorphic to its cohomology. In the operadic language, this is a quasi-isomorphism of $L_\infty$-algebras $\mathrm{Def}(\mathcal{O}) \xrightarrow{\sim} H^\bullet(\mathrm{Def}(\mathcal{O}))$. The formality implies Kontsevich's deformation quantization: every Poisson manifold has a canonical $\star$-product deformation of its function algebra.

> [!QUESTION] Exercise 13: Hochschild cohomology from the deformation complex of Ass
> *This recovers the classical Hochschild complex as a special case of the operadic deformation complex.*
>
> > **Prerequisites:** [[#8.3 The Deformation Complex of an Operad|8.3 The Deformation Complex of an Operad]], [[#5.4 Examples: Ass, Com, and Lie|5.4 Examples: Ass, Com, and Lie]]
>
> Show that $\mathrm{Def}(\mathrm{Ass}) = \mathrm{Hom}_{\mathbf{SymSeq}}(\mathrm{Ass}^c, \mathrm{Ass})$ is isomorphic (as a graded vector space) to the Hochschild cochain complex $C^\bullet(A, A) = \prod_{n \geq 0} \mathrm{Hom}(A^{\otimes n}, A)$ for the free associative algebra $A = T(V)$. Identify which part of the Gerstenhaber bracket on $HH^\bullet$ comes from the convolution pre-Lie product.

> [!TIP]- Solution to Exercise 13
> **Key insight:** $\mathrm{Hom}_{\mathbf{SymSeq}}(\mathrm{Ass}^c, \mathrm{Ass}) = \prod_n \mathrm{Hom}_{S_n}(k[S_n], k[S_n]) = \prod_n \mathrm{End}(k[S_n])^{S_n}$, which for the free algebra $T(V)$ specializes to $\prod_n \mathrm{Hom}(V^{\otimes n}, V^{\otimes n})^{S_n} \supseteq \prod_n \mathrm{Hom}(V^{\otimes n}, V)$, recovering the Hochschild complex.
>
> **Sketch:** An element $f \in \mathrm{Hom}_{\mathbf{SymSeq}}(\mathrm{Ass}^c(n), \mathrm{Ass}(n)) = \mathrm{Hom}_{S_n}(k[S_n], k[S_n])$ is an $S_n$-equivariant endomorphism of $k[S_n]$, i.e., an element of $k[S_n]^{\mathrm{op}} = k[S_n]$ itself (by Schur's lemma for the regular representation). For the free algebra $T(V) = \bigoplus_n V^{\otimes n}$, a Hochschild $n$-cochain $f: A^{\otimes n} \to A$ corresponds to an $S_n$-equivariant map $k[S_n] \to k[S_n]$ (since $\mathrm{Ass}(n) = k[S_n]$ acts on $T(V)(n) = V^{\otimes n}$). The convolution pre-Lie product $f \star g(n) = \sum_{i} f \circ_i g$ (inserting $g$ at the $i$-th input of $f$) recovers the Gerstenhaber pre-Lie product on Hochschild cochains, whose antisymmetrization is the Gerstenhaber bracket $[f,g] = f \star g - (-1)^{|f||g|} g \star f$.

> [!QUESTION] Exercise 14: Infinitesimal deformations of Com and Harrison cohomology
> *This shows that deformations of the Com operad are classified by Harrison cohomology, connecting operad deformation theory to commutative algebra.*
>
> > **Prerequisites:** [[#8.3 The Deformation Complex of an Operad|8.3 The Deformation Complex of an Operad]], [[#8.2 Classical Cohomology Theories as Special Cases|8.2 Classical Cohomology Theories as Special Cases]]
>
> Show that $H^1(\mathrm{Def}(\mathrm{Com})) = H^1_\mathrm{Harrison}(k, k)$, where the Harrison cohomology is taken with coefficients in the ground field $k$ viewed as a $k$-module. Interpret the Maurer–Cartan elements of $\mathrm{Def}(\mathrm{Com})$ in terms of deformations of the commutative operad structure.

> [!TIP]- Solution to Exercise 14
> **Key insight:** $H^1(\mathrm{Def}(\mathrm{Com}))$ classifies deformations of $\mathrm{Com}$ that remain quadratic — i.e., deformations within the world of quadratic operads. These correspond to antisymmetric "Lie-type" corrections, linking to the Com–Lie Koszul duality.
>
> **Sketch:** $\mathrm{Def}(\mathrm{Com}) = \mathrm{Hom}_{\mathbf{SymSeq}}(\mathrm{Lie}^c, \mathrm{Com})$. In degree 1, this is $\mathrm{Hom}_{\mathbf{SymSeq}}(\mathrm{Lie}^c(2), \mathrm{Com}(2)) = \mathrm{Hom}_{S_2}(k \cdot \mathrm{sgn}, k) = 0$ (since sgn and trivial representations don't map nontrivially). This reflects the fact that $\mathrm{Com}$ is rigid (no nontrivial first-order deformations), consistent with the classical result $H^2_\mathrm{Harrison}(A,A) = 0$ for polynomial algebras. At degree 2, $\mathrm{Def}(\mathrm{Com})^2 = \mathrm{Hom}_{S_3}(\mathrm{Lie}^c(3), \mathrm{Com}(3))$; a Maurer–Cartan element here is an antisymmetric ternary operation satisfying a Jacobi-like identity — a deformation toward $\mathrm{Lie}$.

---

**Mathematical Development Exercises (continued)**

> [!QUESTION] Exercise 15: The bar complex of a free operad
> *This shows that the bar construction of a free operad has trivial homology, consistent with the fact that free operads are "Koszul" in a degenerate sense.*
>
> > **Prerequisites:** [[#2.2 The Bar Complex|2.2 The Bar Complex]], [[#2.3 The Differential|2.3 The Differential]]
>
> Let $\mathcal{O} = \mathcal{F}(E)$ be a free operad on a symmetric sequence $E$ concentrated in arity 2. Show that $B(\mathcal{F}(E))$ has homology $H_\bullet(B(\mathcal{F}(E))) = \mathbf{1}$ (the unit cooperad). What does this say about the Koszul criterion for free operads?

> [!TIP]- Solution to Exercise 15
> **Key insight:** The bar complex of a free operad is contractible: the generating trees provide an explicit contracting homotopy via "expanding" each arity-2 composition back into a free tree.
>
> **Sketch:** In $B(\mathcal{F}(E))$, the weight-$n$ component consists of trees-of-trees, which can be "straightened" into single trees using the free operad multiplication. The contracting homotopy $h: B(\mathcal{F}(E)) \to B(\mathcal{F}(E))[-1]$ is defined by picking a splitting of the composition $\gamma: \mathcal{F}(E) \circ \mathcal{F}(E) \to \mathcal{F}(E)$, i.e., a map $s: \mathcal{F}(E) \to \mathcal{F}(E) \circ \mathcal{F}(E)$ sending each tree to its "top vertex" and the remaining forest. Then $[d_\gamma, h] = \mathrm{id}$ on the positive-weight part, showing acyclicity. This is consistent with the Koszul criterion: free operads satisfy the criterion trivially since all their relations are zero.

> [!QUESTION] Exercise 16: Koszul duality and Poincare-Birkhoff-Witt bases
> *This connects the PBW basis theorem for universal enveloping algebras to Koszulness of the Lie operad.*
>
> > **Prerequisites:** [[#6.1 Statement|6.1 Statement]], [[#6.2 Koszulness of Ass and Com|6.2 Koszulness of Ass and Com]]
>
> The PBW theorem states that $U(\mathfrak{g}) \cong S(\mathfrak{g})$ as filtered vector spaces (for a Lie algebra $\mathfrak{g}$). Show how this follows from the Koszulness of $\mathrm{Lie}$, using the fact that $\mathrm{Lie}(n) \cong k[S_n] \otimes_{S_n} \mathrm{sgn}$ (the free Lie algebra) as an $S_n$-module.

> [!TIP]- Solution to Exercise 16
> **Key insight:** The Koszul complex $\mathrm{Com}^c \circ_\kappa \mathrm{Lie}$ being acyclic is exactly the PBW theorem: it says that the associated graded of $U(\mathfrak{g})$ is $S(\mathfrak{g})$.
>
> **Sketch:** The Koszul complex for $\mathrm{Lie}$ is $\mathrm{Com}^c \circ_\kappa \mathrm{Lie}$. In arity $n$, this is a complex of $S_n$-modules $\bigoplus_k (\mathrm{Com}^c(k) \otimes \mathrm{Lie}(n_1) \otimes \cdots \otimes \mathrm{Lie}(n_k))_{n_1+\cdots+n_k=n}$ with differential from $\kappa$. Acyclicity of this complex implies that $\mathrm{Lie}(n)$ has a PBW basis indexed by the same combinatorial data as $\mathrm{Sym}^n$ — i.e., $\dim \mathrm{Lie}(n) = (n-1)!$ (the well-known dimension formula), and the associated graded of $U(\mathfrak{g})$ is $S(\mathfrak{g})$ for any $\mathfrak{g}$.

---

**Algorithmic Applications**

> [!QUESTION] Exercise 17: Computing the bar construction algorithmically
> *This gives a concrete algorithm for computing $B(\mathcal{O})(n)$ in low weights.*
>
> > **Prerequisites:** [[#2.2 The Bar Complex|2.2 The Bar Complex]], [[#2.3 The Differential|2.3 The Differential]]
>
> Write a Python pseudocode algorithm that, given a quadratic operad $\mathcal{O}$ specified by its generators $E$ and relations $R$ (as matrices over $k$), computes the chain groups and differential of $B(\mathcal{O})(n)$ for a given $n$ and weight $\leq W$.

> [!TIP]- Solution to Exercise 17
> **Key insight:** The weight-$w$ part of $B(\mathcal{O})(n)$ is a direct sum over rooted trees with $w$ internal vertices and $n$ leaves, with each vertex labeled by a basis element of $\bar{\mathcal{O}}$.
>
> **Sketch:**
> ```python
> def bar_construction(O_gens, O_rels, n, max_weight):
>     """
>     O_gens: dict {arity: basis_list} for the generators of O-bar
>     O_rels: list of linear relations as dicts {monomial: coefficient}
>     Returns: chain_groups, differential_matrices
>     """
>     from itertools import product
>
>     def rooted_trees(n_leaves, weight):
>         """Enumerate rooted trees with n_leaves leaves and `weight` internal vertices."""
>         if weight == 1:
>             yield [n_leaves]  # single vertex of arity n_leaves
>         else:
>             for arity in range(2, n_leaves):
>                 for partition in integer_partitions(n_leaves, arity):
>                     for subtrees in product(
>                         *[rooted_trees(p, w)
>                           for p, w in distribute_weight(partition, weight-1)]
>                     ):
>                         yield (arity, subtrees)
>
>     chain_groups = {}
>     for w in range(1, max_weight + 1):
>         basis = []
>         for tree in rooted_trees(n, w):
>             for labels in label_tree(tree, O_gens):
>                 basis.append((tree, labels))
>         chain_groups[w] = basis
>
>     # Differential: contract each internal edge using O_rels
>     differentials = {}
>     for w in range(2, max_weight + 1):
>         mat = {}
>         for idx, (tree, labels) in enumerate(chain_groups[w]):
>             for edge in internal_edges(tree):
>                 contracted = contract_edge(tree, labels, edge, O_rels)
>                 sign = koszul_sign(tree, labels, edge)
>                 mat[idx] = mat.get(idx, {})
>                 for jdx, coeff in contracted:
>                     mat[idx][jdx] = mat[idx].get(jdx, 0) + sign * coeff
>         differentials[w] = mat
>
>     return chain_groups, differentials
> ```
> The key subroutines are `rooted_trees` (enumerate by weight), `label_tree` (assign operad generators to vertices), `contract_edge` (apply $\gamma$ using the relations), and `koszul_sign` (track the degree-shift signs from suspension). The differential matrix has shape `len(chain_groups[w]) × len(chain_groups[w-1])`.

> [!QUESTION] Exercise 18: Checking the Koszul criterion computationally
> *This turns the Koszul criterion into a linear algebra computation.*
>
> > **Prerequisites:** [[#6.1 Statement|6.1 Statement]], [[#17|Exercise 17]]
>
> Using the output of the algorithm in Exercise 17, describe how to check the Koszul criterion for a quadratic operad $\mathcal{O}$ in arity $n \leq N$. What linear algebra operations are needed, and what is the computational complexity in $n$?

> [!TIP]- Solution to Exercise 18
> **Key insight:** Koszulness is equivalent to rank conditions on the differential matrices: $\mathrm{rank}(d_w) + \mathrm{rank}(d_{w+1}) = \dim(C_w)$ for all $w$ — i.e., the complex is exact at each weight.
>
> **Sketch:** After computing `chain_groups[w]` and `differentials[w]` for $w = 1, \ldots, W$ and $n = 1, \ldots, N$:
> ```python
> def check_koszul(chain_groups, differentials, n, max_weight):
>     for w in range(1, max_weight):
>         C_w = len(chain_groups[w])
>         d_w = matrix(differentials.get(w, {}), C_w)
>         d_wp1 = matrix(differentials.get(w+1, {}), len(chain_groups.get(w+1, [])))
>         # Exactness: im(d_{w+1}) = ker(d_w)
>         rank_w = rank(d_w)
>         rank_wp1 = rank(d_wp1)
>         if rank_w + rank_wp1 != C_w:
>             return False  # Not Koszul at weight w
>     return True
> ```
> Complexity: the number of rooted trees with $n$ leaves and weight $w$ grows as $n^w / w!$ (Catalan-like), so the chain groups have dimension $O((n/e)^n)$ in the worst case. Computing ranks via Gaussian elimination costs $O(\dim^3)$. In practice, the $S_n$-equivariance can be used to block-diagonalize (decompose by irreducible representations of $S_n$), reducing to checking each isotypic component separately. For small $n$ (say $n \leq 6$), this is feasible in seconds.

> [!QUESTION] Exercise 19: Implementing the cobar construction
> *This connects the abstract definition of $\Omega(\mathcal{C})$ to a practical data structure.*
>
> > **Prerequisites:** [[#3.1 Definition|3.1 Definition]]
>
> Write Python pseudocode for computing $\Omega(\mathcal{O}^!)(n)$ for a Koszul quadratic operad $\mathcal{O}$ (specified by $E$ and $R$). The output should be the underlying graded operad (generators and their degrees) and the differential matrix.

> [!TIP]- Solution to Exercise 19
> **Key insight:** $\Omega(\mathcal{O}^!)$ is the free operad on $\mathcal{O}^![-1]$, so its generators are just the cogenerators of $\mathcal{O}^!$ in one degree lower, and the differential expands each cogenerator using the coproduct $\Delta$ of $\mathcal{O}^!$.
>
> **Sketch:**
> ```python
> def cobar_construction(E, R, n, max_weight):
>     """
>     E: dict {arity: basis} for generators of O
>     R: relations in F(E)(3)
>     Returns generators and differential of Omega(O^!)(n)
>     """
>     # Step 1: compute O^! cogenerators = sE^vee with corelations s^2 R^perp
>     E_dual = {arity: dual_basis(E[arity]) for arity in E}
>     R_perp = annihilator(R, free_operad(E_dual, arities=[3]))
>
>     # Step 2: desuspend to get Omega(O^!) generators in degree -(arity-1)
>     generators = {}
>     for arity, basis in E_dual.items():
>         degree = -(arity - 1) + 1  # desuspension: s^{-1}(sE^vee) = E^vee in degree 0
>         generators[arity] = [(b, degree) for b in basis]
>
>     # Step 3: free operad on generators (planar rooted trees with labels from generators)
>     free_op = free_operad_on(generators, arity=n, max_weight=max_weight)
>
>     # Step 4: differential from the coproduct of O^!
>     def differential(tree_monomial):
>         result = []
>         for vertex in tree_monomial.vertices:
>             cogen = tree_monomial.label(vertex)
>             # Apply coproduct Delta: cogen -> sum of (cogen' o cogen'')
>             for (c_prime, c_double, i, sign) in coproduct(cogen, R_perp):
>                 new_tree = tree_monomial.expand_vertex(vertex, c_prime, c_double, i)
>                 result.append((sign, new_tree))
>         return result
>
>     diff_matrix = build_matrix(free_op, differential)
>     return free_op, diff_matrix
> ```
> The main subtlety is `coproduct`: it reads off the corelations $s^2 R^\perp$ to compute $\Delta$ on cogenerators. For $\mathcal{O} = \mathrm{Ass}$, this reproduces the Stasheff $A_\infty$ operad with operations $m_n$ in degree $2-n$.

> [!QUESTION] Exercise 20: Minimal model of a dg-algebra via transfer
> *This implements the homotopy transfer theorem as a practical algorithm for computing minimal $A_\infty$-models.*
>
> > **Prerequisites:** [[#7.3 The Homotopy Transfer Theorem|7.3 The Homotopy Transfer Theorem]]
>
> Given a dg-associative algebra $A$ (specified by its multiplication table and differential $d: A \to A$) and a deformation retract $(i, p, h)$ onto $H = H^\bullet(A)$, write Python pseudocode for computing the transferred $A_\infty$-operations $m_n: H^{\otimes n} \to H$ for $n \leq N$ using the tree formula.

> [!TIP]- Solution to Exercise 20
> **Key insight:** The transferred $m_n$ is a sum over all binary trees with $n$ leaves, with $i$ at leaves, $m_2^A$ at internal vertices, and $h$ at internal edges, projected to $H$ by $p$ at the root.
>
> **Sketch:**
> ```python
> def homotopy_transfer(A_mult, d, i, p, h, n_max):
>     """
>     A_mult: function (a, b) -> a*b for the algebra A
>     d: differential on A
>     i, p, h: deformation retract maps (i: H->A, p: A->H, h: A->A, degree -1)
>     Returns: dict {n: m_n} for n = 1, ..., n_max
>     """
>     from functools import lru_cache
>
>     @lru_cache(maxsize=None)
>     def phi(tree):
>         """
>         Compute the A-valued operation phi_tree: H^(leaves) -> A
>         defined by the tree formula, without the final p.
>         """
>         if tree.is_leaf():
>             return i  # base case: leaf applies i
>         left, right = tree.children()
>         phi_L = phi(left)  # left subtree map H^k -> A
>         phi_R = phi(right)  # right subtree map H^l -> A
>         # Compose with A_mult and then apply h (homotopy on internal edge)
>         return lambda *args: h(A_mult(
>             phi_L(*args[:left.n_leaves]),
>             phi_R(*args[left.n_leaves:])
>         ))
>
>     m = {}
>     for n in range(1, n_max + 1):
>         if n == 1:
>             m[1] = lambda x: p(d(i(x)))  # m_1 = p d i (boundary of i)
>         else:
>             total = zero_map(n)
>             for tree in binary_trees(n):
>                 sign = koszul_sign_tree(tree)
>                 # Apply p to the root
>                 total += sign * (lambda *args, t=tree: p(A_mult(
>                     phi(t.left)(*args[:t.left.n_leaves]),
>                     phi(t.right)(*args[t.left.n_leaves:])
>                 )))
>             m[n] = total
>     return m
> ```
> The key insight is the recursive `phi(tree)`: for a leaf it is just $i$; for an internal node it applies $h$ (the homotopy) after the multiplication, threading the deformation retract data through the tree structure. The number of binary trees with $n$ leaves is the Catalan number $C_{n-1}$, so the complexity is $O(C_{n-1} \cdot \dim(H)^n)$ — exponential in $n$, but manageable for small $n$.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Loday & Vallette, *Algebraic Operads*](https://link.springer.com/book/10.1007/978-3-642-30362-3) | The canonical graduate reference; Chapters 6–7 cover the bar/cobar construction and twisting morphisms, Chapters 10–12 cover Koszul duality in full generality. | [Springer](https://link.springer.com/book/10.1007/978-3-642-30362-3) |
| [Ginzburg & Kapranov, "Koszul Duality for Operads"](https://arxiv.org/abs/0709.1228) | The foundational paper introducing Koszul duality for operads: quadratic operads, Koszul dual cooperad, the criterion, and the Com–Lie duality. Duke Math. J. 1994. | [arXiv:0709.1228](https://arxiv.org/abs/0709.1228) |
| [Getzler & Jones, "Operads, Homotopy Algebra and Iterated Integrals"](https://arxiv.org/abs/hep-th/9403055) | Extends Koszul duality to the dg setting; introduces the dg bar construction for operads, homotopy algebras over operads, and applications to 2d TFT. | [arXiv:hep-th/9403055](https://arxiv.org/abs/hep-th/9403055) |
| [Vallette, "Algebra + Homotopy = Operad"](https://arxiv.org/abs/1202.3245) | Accessible 43-page survey with figures; §3–4 cover bar/cobar and Koszul duality with many worked examples. Ideal first read before Loday–Vallette. | [arXiv:1202.3245](https://arxiv.org/abs/1202.3245) |
| [Keller, "Introduction to A∞ Algebras and Modules"](https://arxiv.org/abs/math/9910179) | Self-contained lecture notes on $A_\infty$-algebras; covers the bar construction, minimal models, and the homotopy transfer theorem in full detail. | [arXiv:math/9910179](https://arxiv.org/abs/math/9910179) |
| [Markl, Shnider & Stasheff, *Operads in Algebra, Topology and Physics*](https://bookstore.ams.org/surv-96-s) | AMS monograph; Chapter 3 covers Koszul duality, and Chapter 4 covers deformation theory of operads and the Kontsevich formality theorem. | [AMS](https://bookstore.ams.org/surv-96-s) |
| [nLab: Koszul duality](https://ncatlab.org/nlab/show/Koszul+duality) | Encyclopedic treatment with multiple equivalent formulations and connections to rational homotopy theory and derived algebraic geometry. | [nLab](https://ncatlab.org/nlab/show/Koszul+duality) |
| [nLab: bar and cobar construction](https://ncatlab.org/nlab/show/bar+and+cobar+construction) | Precise definitions of the operadic bar and cobar constructions, with the adjunction and the quasi-isomorphism statement. | [nLab](https://ncatlab.org/nlab/show/bar+and+cobar+construction) |
