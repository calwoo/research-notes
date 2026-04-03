# Mackey Functors

## Table of Contents

- [[#1. From Coefficient Systems to Mackey Functors|1. From Coefficient Systems to Mackey Functors]]
  - [[#1.1 Coefficient Systems|1.1 Coefficient Systems]]
  - [[#1.2 Why Restriction Alone Is Insufficient|1.2 Why Restriction Alone Is Insufficient]]
  - [[#1.3 The Span Perspective|1.3 The Span Perspective]]
- [[#2. The Burnside Category|2. The Burnside Category]]
  - [[#2.1 Spans of Finite G-Sets|2.1 Spans of Finite G-Sets]]
  - [[#2.2 Composition via Fiber Product|2.2 Composition via Fiber Product]]
  - [[#2.3 Additive Structure and the Burnside Ring|2.3 Additive Structure and the Burnside Ring]]
- [[#3. Mackey Functors: Formal Definition|3. Mackey Functors: Formal Definition]]
  - [[#3.1 Additive Functors on the Burnside Category|3.1 Additive Functors on the Burnside Category]]
  - [[#3.2 Unpacking into Restriction, Transfer, and Conjugation|3.2 Unpacking into Restriction, Transfer, and Conjugation]]
  - [[#3.3 Axiomatic Presentation|3.3 Axiomatic Presentation]]
- [[#4. The Mackey Double Coset Formula|4. The Mackey Double Coset Formula]]
  - [[#4.1 The Pullback Decomposition|4.1 The Pullback Decomposition]]
  - [[#4.2 Derivation from Spans|4.2 Derivation from Spans]]
- [[#5. Key Examples|5. Key Examples]]
  - [[#5.1 The Constant Mackey Functor|5.1 The Constant Mackey Functor]]
  - [[#5.2 The Burnside Ring Mackey Functor|5.2 The Burnside Ring Mackey Functor]]
  - [[#5.3 Fixed-Point Mackey Functors|5.3 Fixed-Point Mackey Functors]]
  - [[#5.4 The C2 Classification|5.4 The C2 Classification]]
- [[#6. The Box Product and Green/Tambara Functors|6. The Box Product and Green/Tambara Functors]]
  - [[#6.1 Day Convolution and the Box Product|6.1 Day Convolution and the Box Product]]
  - [[#6.2 Green Functors|6.2 Green Functors]]
  - [[#6.3 Tambara Functors and Multiplicative Norms|6.3 Tambara Functors and Multiplicative Norms]]
- [[#7. Projective Mackey Functors and Resolutions|7. Projective Mackey Functors and Resolutions]]
  - [[#7.1 Representable Mackey Functors|7.1 Representable Mackey Functors]]
  - [[#7.2 Global Dimension and Resolutions|7.2 Global Dimension and Resolutions]]
- [[#8. Spectral Mackey Functors: Barwick's Theorem|8. Spectral Mackey Functors: Barwick's Theorem]]
  - [[#8.1 The Effective Burnside Infinity-Category|8.1 The Effective Burnside Infinity-Category]]
  - [[#8.2 Spectral Mackey Functors|8.2 Spectral Mackey Functors]]
  - [[#8.3 Barwick's Equivalence|8.3 Barwick's Equivalence]]
  - [[#8.4 Homotopy Groups as Classical Mackey Functors|8.4 Homotopy Groups as Classical Mackey Functors]]
- [[#References|References]]

---

## 1. From Coefficient Systems to Mackey Functors 📐

### 1.1 Coefficient Systems

To motivate Mackey functors, we begin with the simpler notion that arises in Bredon cohomology. Fix a finite group $G$ throughout. Recall the *orbit category* $\mathcal{O}_G$ from [[concepts/equivariant-stable-homotopy/g-spaces-and-equivariant-maps|G-Spaces and Equivariant Maps]]: its objects are the *transitive* $G$-sets $G/H$ for $H \leq G$, and its morphisms $\mathcal{O}_G(G/H, G/K)$ are $G$-equivariant maps $G/H \to G/K$. Every such map has the form $gH \mapsto gxK$ for some fixed $x$ with $x^{-1}Hx \leq K$.

**Definition (Coefficient System).** A *coefficient system* for $G$ is a contravariant functor

$$M^*: \mathcal{O}_G^{\mathrm{op}} \longrightarrow \mathbf{Ab}.$$

Concretely, a coefficient system assigns an abelian group $M(G/H)$ to each orbit $G/H$, and to each $G$-map $f: G/H \to G/K$ a *restriction map* $f^*: M(G/K) \to M(G/H)$, functorially. In particular, conjugation by $g \in G$ gives an isomorphism

$$c_g: M(G/H) \xrightarrow{\sim} M(G/{}^gH), \quad {}^gH = gHg^{-1}.$$

The group $H$ acts on $M(G/H)$ via $c_h$ for $h \in H$ (since ${}^hH = H$), but this action is trivial because $c_h = \mathrm{id}$ as a map $G/H \to G/H$ (the only equivariant self-map of $G/H$ is the identity up to the $G$-action).

> [!INFO] Bredon's Original Setup
> Bredon introduced coefficient systems in 1967 as the coefficient objects for his equivariant cohomology theory. The chain complex of a $G$-CW complex $X$ is a diagram of abelian groups indexed by $\mathcal{O}_G$, and cohomology $H^*_G(X; M^*)$ is computed by applying $\mathrm{Nat}(\mathcal{C}_*(X), M^*)$. This is well-behaved but only captures contravariant data.

### 1.2 Why Restriction Alone Is Insufficient 💡

Coefficient systems suffice for defining cohomology, but they are too coarse to capture the full structure of equivariant invariants. Consider a $G$-space $X$. The natural assignment $G/H \mapsto \pi_n(X^H)$ is contravariant with respect to the orbit category via restriction: if $f: G/H \to G/K$ comes from inclusion $H \leq K$, the map $X^K \hookrightarrow X^H$ (more fixed points under a smaller group) gives a restriction

$$\mathrm{res}_H^K: \pi_n(X^K) \to \pi_n(X^H).$$

But there is also a *transfer map* running in the other direction:

$$\mathrm{tr}_H^K: \pi_n(X^H) \longrightarrow \pi_n(X^K).$$

For $K = G$ and $n = 0$, this is the equivariant analogue of summing over coset representatives: $\mathrm{tr}_H^G(m) = \sum_{[g] \in G/H} g_* m$. This map is covariant, going from smaller to larger group, and has no analogue in the coefficient system framework.

*The failure of coefficient systems is that they admit only restriction maps.* A Mackey functor is precisely the structure that captures both restrictions and transfers simultaneously, subject to the compatibility imposed by the Mackey double coset formula.

> [!WARNING] Direction Convention
> Transfer maps $\mathrm{tr}_H^K: M(G/H) \to M(G/K)$ go from the smaller orbit to the larger one — opposite to the restriction direction. Some sources write $I_H^K$ (induction) for transfer and $R_H^K$ for restriction.

### 1.3 The Span Perspective 🔑

The key insight of Lindner (1976) and Dress (1973) is that both restriction and transfer arise from a single *span* (correspondence) of $G$-sets. A span from $X$ to $Y$ is a diagram

$$X \xleftarrow{p} Z \xrightarrow{q} Y$$

of $G$-equivariant maps. Given such a span and a functor $M$, one can:
- Apply $M$ contravariantly to $p$ to get $M(p): M(X) \to M(Z)$,
- Apply $M$ covariantly to $q$ to get $M(q): M(Z) \to M(Y)$,

and compose to get a map $M(X) \to M(Y)$. The composite

$$M(X) \xrightarrow{M(p)} M(Z) \xrightarrow{M(q)} M(Y)$$

depends on $M$ having both contravariant and covariant functoriality. This is the content of defining $M$ as a functor on the *span category*, where every morphism $X \to Y$ is a span $X \leftarrow Z \rightarrow Y$.

**The Burnside category** $\mathcal{A}(G)$ is the $\mathbb{Z}$-linear category whose morphism groups are Grothendieck completions of the monoid of isomorphism classes of spans, with composition given by fiber product. Mackey functors are additive functors $\mathcal{A}(G) \to \mathbf{Ab}$.

---

## 2. The Burnside Category 🧮

### 2.1 Spans of Finite G-Sets

**Definition (Pre-Burnside Category).** Let $\mathbf{FSets}_G$ denote the category of finite $G$-sets. Define a category $\mathcal{A}^+(G)$ — the *pre-Burnside category* — as follows:

- **Objects:** finite $G$-sets $X, Y, Z, \ldots$
- **Morphisms:** $\mathcal{A}^+(G)(X, Y)$ is the set of *isomorphism classes* of spans $X \xleftarrow{} Z \xrightarrow{} Y$ of $G$-equivariant maps, where two spans $(X \leftarrow Z \rightarrow Y)$ and $(X \leftarrow Z' \rightarrow Y)$ are isomorphic if there is a $G$-equivariant bijection $Z \xrightarrow{\sim} Z'$ commuting with both projection maps.

The set $\mathcal{A}^+(G)(X, Y)$ is a *commutative monoid* under disjoint union of spans: $[X \leftarrow Z \rightarrow Y] + [X \leftarrow Z' \rightarrow Y] = [X \leftarrow Z \sqcup Z' \rightarrow Y]$.

> [!NOTE] Objects as Orbits
> It suffices to take objects to be the transitive $G$-sets $G/H$ for conjugacy classes of subgroups $H \leq G$, since every finite $G$-set decomposes as a disjoint union of orbits. Thus objects of $\mathcal{A}(G)$ are indexed by the *table of marks* data of $G$.

**Definition (Burnside Category).** The *Burnside category* $\mathcal{A}(G)$ is the *additive* (i.e., $\mathbf{Ab}$-enriched) category obtained from $\mathcal{A}^+(G)$ by applying the Grothendieck group construction to each morphism monoid:

$$\mathcal{A}(G)(X, Y) = K_0\bigl(\mathcal{A}^+(G)(X,Y)\bigr).$$

Concretely, $\mathcal{A}(G)(X, Y)$ consists of formal differences $[X \leftarrow Z \rightarrow Y] - [X \leftarrow Z' \rightarrow Y]$ of isomorphism classes of spans.

### 2.2 Composition via Fiber Product 📐

Composition in the Burnside category is defined on the pre-Burnside level and extended bilinearly to the group-completed version.

**Definition (Span Composition).** Given spans $\sigma = (X \xleftarrow{p} Z \xrightarrow{q} Y)$ and $\tau = (Y \xleftarrow{r} W \xrightarrow{s} V)$, their composite $\tau \circ \sigma$ is the span

$$X \xleftarrow{p \circ \pi_Z} Z \times_Y W \xrightarrow{s \circ \pi_W} V,$$

where $Z \times_Y W$ is the *fiber product* (pullback) of $q: Z \to Y$ and $r: W \to Y$ in $\mathbf{FSets}_G$:

$$Z \times_Y W = \{(z, w) \in Z \times W : q(z) = r(w)\}.$$

The $G$-action on the fiber product is diagonal: $g \cdot (z, w) = (gz, gw)$.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
 & Z \times_Y W \arrow[dl, "\pi_Z"'] \arrow[dr, "\pi_W"] & \\
Z \arrow[dl, "p"'] \arrow[dr, "q"] & & W \arrow[dl, "r"'] \arrow[dr, "s"] \\
X & Y & Y & V
\end{tikzcd}
\end{document}
```

> [!INFO] Associativity
> Composition is associative because the fiber product of spans is associative up to canonical isomorphism (by the universal property of pullbacks in $\mathbf{FSets}_G$). The identity span on $X$ is $X \xleftarrow{\mathrm{id}} X \xrightarrow{\mathrm{id}} X$.

### 2.3 Additive Structure and the Burnside Ring 🔑

The category $\mathcal{A}(G)$ is *additive*: the coproduct of objects $X$ and $Y$ is the disjoint union $X \sqcup Y$, and this makes $\mathcal{A}(G)$ into a $\mathbb{Z}$-linear category.

**The Burnside ring** $A(G)$ appears as the endomorphism ring of the terminal $G$-set:

$$A(G) = \mathcal{A}(G)(G/G, G/G).$$

Indeed, $\mathcal{A}^+(G)(G/G, G/G)$ is the monoid of isomorphism classes of finite $G$-sets (since a span $G/G \leftarrow Z \rightarrow G/G$ is simply a finite $G$-set $Z$ with no additional structure beyond the two maps to the point), so $K_0$ recovers the Burnside ring.

More generally,

$$\mathcal{A}(G)(G/H, G/K) \cong K_0\bigl(\mathbf{FSets}_{G/H \times G/K}\bigr)$$

where the right side is the Grothendieck group of finite $G$-sets over $G/H \times G/K$ — equivalently, finite $(H \times K)$-sets with the $(H,K)$-biset structure.

> [!EXAMPLE] Morphisms for G = C2
> Let $G = C_2 = \{e, \tau\}$. The objects of $\mathcal{A}(C_2)$ are $C_2/e \cong C_2$ (the free orbit) and $C_2/C_2 \cong *$ (the fixed point). The morphism groups are:
> - $\mathcal{A}(C_2)(*, *) \cong A(C_2) \cong \mathbb{Z}^2$ (generated by $[*]$ and $[C_2]$),
> - $\mathcal{A}(C_2)(C_2, *) \cong \mathbb{Z}$ (generated by the transfer span $* \leftarrow C_2 \rightarrow *$),
> - $\mathcal{A}(C_2)(*, C_2) \cong \mathbb{Z}$ (generated by the restriction span $C_2 \leftarrow C_2 \rightarrow *$... wait: actually $* \leftarrow * \rightarrow C_2$, the inclusion),
> - $\mathcal{A}(C_2)(C_2, C_2) \cong \mathbb{Z}^2$ (generated by the identity span and the $C_2$-set $C_2 \times C_2$).

---

## 3. Mackey Functors: Formal Definition 🔑

### 3.1 Additive Functors on the Burnside Category

**Definition (Mackey Functor).** A *Mackey functor* for $G$ is an *additive functor*

$$M: \mathcal{A}(G) \longrightarrow \mathbf{Ab},$$

i.e., a functor of $\mathbb{Z}$-linear categories from the Burnside category to abelian groups.

This is Lindner's 1976 reformulation of Dress's original definition. The additivity condition means that $M$ preserves finite direct sums: $M(X \sqcup Y) \cong M(X) \oplus M(Y)$.

> [!NOTE] Equivalent Characterizations
> Additive functors $\mathcal{A}(G) \to \mathbf{Ab}$ are equivalently: (1) additive functors out of the full span category $\mathbf{Span}(\mathbf{FSets}_G)$ that invert the Grothendieck completion, or (2) the data described in §3.3 below satisfying the Mackey formula.

The category of Mackey functors $\mathrm{Mack}(G) = \mathrm{Fun}^{\mathrm{add}}(\mathcal{A}(G), \mathbf{Ab})$ is an *abelian category* (since $\mathbf{Ab}$ is abelian and limits/colimits of additive functors are computed pointwise). In particular, it has enough injectives and projectives.

### 3.2 Unpacking into Restriction, Transfer, and Conjugation

Since $M: \mathcal{A}(G) \to \mathbf{Ab}$ is a covariant functor, and a span $X \leftarrow Z \rightarrow Y$ is a morphism $X \to Y$ in $\mathcal{A}(G)$, applying $M$ to a span yields a map $M(X) \to M(Y)$. The restriction and transfer arise from two different spans between the same pair of orbits.

Let $p: G/H \to G/K$ denote the canonical $G$-equivariant projection (defined for $H \leq K$, sending $gH \mapsto gK$).

- **Restriction** $\mathrm{res}_H^K: M(G/K) \to M(G/H)$ comes from the span
$$G/K \xleftarrow{p} G/H \xrightarrow{\mathrm{id}} G/H,$$
which is a morphism $G/K \to G/H$ in $\mathcal{A}(G)$. Applying $M$ gives the map $M(G/K) \to M(G/H)$.

- **Transfer** $\mathrm{tr}_H^K: M(G/H) \to M(G/K)$ comes from the span
$$G/H \xleftarrow{\mathrm{id}} G/H \xrightarrow{p} G/K,$$
which is a morphism $G/H \to G/K$ in $\mathcal{A}(G)$. Applying $M$ gives the map $M(G/H) \to M(G/K)$.

- **Conjugation** $c_g: M(G/H) \to M(G/{}^gH)$ for $g \in G$ comes from the span
$$G/H \xleftarrow{c_g} G/{}^gH \xrightarrow{\mathrm{id}} G/{}^gH,$$
where $c_g: G/{}^gH \xrightarrow{\sim} G/H$ sends $x{}^gH \mapsto xg^{-1}H$.

> [!NOTE] Variance Summary
> Restriction and transfer are both *covariant* in $M$, but they come from spans pointing in opposite directions. The restriction span has the projection $p$ on its left leg (source side); the transfer span has $p$ on its right leg (target side). This asymmetry is why both maps coexist in a single additive functor on $\mathcal{A}(G)$.

> [!NOTE] Functoriality
> The composite of two restriction spans is the restriction span for the composite inclusion, and similarly for transfers. The Mackey formula arises when a transfer is composed with a restriction — which requires composing spans via fiber product.

### 3.3 Axiomatic Presentation

The additive functor definition unpacks into the following axiomatic presentation, which is often taken as the classical definition of a Mackey functor.

**Definition (Mackey Functor, Axiomatic).** A Mackey functor $M$ for $G$ consists of:
- For each $H \leq G$: an abelian group $M(H)$ (abbreviated $M(G/H)$),
- For each $H \leq K \leq G$: restriction $\mathrm{res}_H^K: M(K) \to M(H)$ and transfer $\mathrm{tr}_H^K: M(H) \to M(K)$,
- For each $g \in G$ and $H \leq G$: conjugation $c_g: M(H) \xrightarrow{\sim} M({}^gH)$,

satisfying:
1. **Transitivity of restriction:** $\mathrm{res}_H^K \circ \mathrm{res}_K^L = \mathrm{res}_H^L$ for $H \leq K \leq L$.
2. **Transitivity of transfer:** $\mathrm{tr}_K^L \circ \mathrm{tr}_H^K = \mathrm{tr}_H^L$ for $H \leq K \leq L$.
3. **Conjugation compatibility:** $c_{gh} = c_g \circ c_h$; $\mathrm{res}_{{}^gH}^{{}^gK} \circ c_g = c_g \circ \mathrm{res}_H^K$; similarly for transfers.
4. **Mackey double coset formula:** For $H, K \leq G$,
$$\mathrm{res}_K^G \circ \mathrm{tr}_H^G(m) = \sum_{[g] \in K\backslash G/H} \mathrm{tr}_{K \cap {}^gH}^K \circ c_g \circ \mathrm{res}_{K^g \cap H}^H(m),$$
where $K^g = g^{-1}Kg$ and the sum is over double coset representatives $[g] \in K\backslash G/H$.

> [!INFO] Lindner's Theorem
> **Lindner's theorem** (1976) states that the category of additive functors $\mathcal{A}(G) \to \mathbf{Ab}$ is equivalent to the category of pairs $(M^*, M_*)$ — one contravariant and one covariant functor on $\mathcal{O}_G$ agreeing on objects — satisfying the Mackey formula. This equivalence identifies the abstract span-functor definition with the concrete axiomatic one.

---

## 4. The Mackey Double Coset Formula 📐

### 4.1 The Pullback Decomposition

The double coset formula is not an additional axiom imposed by fiat — it is a theorem forced by the span composition in the Burnside category. We derive it here from first principles.

Fix subgroups $H, K \leq G$. The transfer $\mathrm{tr}_H^G: M(G/H) \to M(G/G)$ corresponds to the span (morphism $G/H \to G/G$ in $\mathcal{A}(G)$)

$$G/H \xleftarrow{\mathrm{id}} G/H \xrightarrow{p} G/G,$$

where $p: G/H \to G/G$ is the canonical projection. The restriction $\mathrm{res}_K^G: M(G/G) \to M(G/K)$ corresponds to the span (morphism $G/G \to G/K$ in $\mathcal{A}(G)$)

$$G/G \xleftarrow{q} G/K \xrightarrow{\mathrm{id}} G/K,$$

where $q: G/K \to G/G$ is the canonical projection.

To compute $\mathrm{res}_K^G \circ \mathrm{tr}_H^G$, we compose the span for $\mathrm{tr}_H^G: G/H \to G/G$ with the span for $\mathrm{res}_K^G: G/G \to G/K$, obtaining a composite span $G/H \to G/K$. The middle $G/G$ is the object over which we take the fiber product.

**Step 1:** The middle piece of the composite span is the fiber product $G/H \times_{G/G} G/K$ (the pullback of $p: G/H \to G/G$ and $q: G/K \to G/G$ in $\mathbf{FSets}_G$). Since $G/G = \{*\}$ is the terminal object, the fiber product is simply the Cartesian product:

$$G/H \times_{G/G} G/K \cong G/H \times G/K.$$

**Step 2:** Decompose $G/H \times G/K$ into $G$-orbits. The diagonal $G$-action is $g \cdot (xH, yK) = (gxH, gyK)$. Two pairs $(xH, yK)$ and $(x'H, y'K)$ lie in the same orbit iff there exists $g \in G$ with $gxH = x'H$ and $gyK = y'K$. Setting $x = e$, the orbit of $(H, g_0 K)$ is indexed by $g_0 \in H\backslash G/K$, and the stabilizer of the pair $(H, g_0 K)$ under the diagonal action is $H \cap {}^{g_0}K$. Equivalently, writing with our convention of $K\backslash G/H$ double cosets, the stabilizer of $(K, g_0 H)$ is $K \cap {}^{g_0}H$. Therefore:

$$G/K \times_{G/G} G/H \cong \bigsqcup_{[g] \in K\backslash G/H} G/(K \cap {}^gH).$$

This is the *key decomposition*. The fiber product of the two spans decomposes as a disjoint union of orbits, one for each double coset $KgH$.

### 4.2 Derivation from Spans

Each summand $G/(K \cap {}^gH)$ in the decomposition above contributes a span

$$G/K \xleftarrow{} G/(K \cap {}^gH) \xrightarrow{} G/H.$$

The left leg is the projection $G/(K \cap {}^gH) \to G/K$ (corresponding to $K \cap {}^gH \leq K$), so it contributes a transfer $\mathrm{tr}_{K \cap {}^gH}^K$. The right leg goes to $G/H$ by the map $x(K \cap {}^gH) \mapsto xg^{-1}H$ (which uses conjugation by $g^{-1}$ to move from $K \cap {}^gH = K \cap gHg^{-1}$ to $g^{-1}Kg \cap H$), contributing $c_g \circ \mathrm{res}_{g^{-1}Kg \cap H}^H$.

Summing over all double coset representatives and applying $M$:

$$\boxed{\mathrm{res}_K^G \circ \mathrm{tr}_H^G(m) = \sum_{[g] \in K\backslash G/H} \mathrm{tr}_{K \cap {}^gH}^K \circ c_g \circ \mathrm{res}_{K^g \cap H}^H(m)}$$

where $K^g = g^{-1}Kg$.

**This is the Mackey double coset formula, derived purely from the fiber product decomposition of spans in $\mathbf{FSets}_G$.**

> [!EXAMPLE] Specialization: G = C2
> Take $G = C_2$, $H = K = \{e\}$. Then $K\backslash G/H = C_2\backslash C_2/\{e\} = \{[e], [\tau]\}$, two double cosets. For $[g] = [e]$: $K \cap {}^eH = \{e\} \cap \{e\} = \{e\}$, contributing $\mathrm{tr}_e^e \circ c_e \circ \mathrm{res}_e^e = \mathrm{id}$. For $[g] = [\tau]$: $K \cap {}^\tau H = \{e\}$, contributing $c_\tau$. So $\mathrm{res}_e^{C_2} \circ \mathrm{tr}_e^{C_2} = \mathrm{id} + \tau_*$, the norm map. For $G = C_2$ acting on itself, this says $\mathrm{res} \circ \mathrm{tr}(m) = m + \tau m$.

> [!EXAMPLE] Specialization: Disjoint Subgroups
> If $H$ and $K$ are subgroups with $HK = G$ (e.g., $G$ is the direct product $H \times K$), then $K\backslash G/H$ has a single element $[e]$, and the formula reduces to $\mathrm{res}_K^G \circ \mathrm{tr}_H^G = \mathrm{tr}_{K \cap H}^K \circ c_e \circ \mathrm{res}_{K \cap H}^H = \mathrm{tr}_{K \cap H}^K \circ \mathrm{res}_{K \cap H}^H$. This is the *base change formula* for products of groups.

---

## 5. Key Examples 🧮

### 5.1 The Constant Mackey Functor

**Definition (Constant Mackey Functor).** For an abelian group $A$, the *constant Mackey functor* $\underline{A}$ is defined by:
- $\underline{A}(G/H) = A$ for all $H \leq G$,
- $\mathrm{res}_H^K = \mathrm{id}_A$ for all $H \leq K$,
- $\mathrm{tr}_H^K = [K:H] \cdot \mathrm{id}_A$ (multiplication by the index $[K:H]$),
- $c_g = \mathrm{id}_A$ for all $g$.

Let us verify the Mackey formula. Take $H, K \leq G$ and $m \in \underline{A}(G/H) = A$. Then:

$$\mathrm{res}_K^G \circ \mathrm{tr}_H^G(m) = \mathrm{id}_A([G:H] \cdot m) = [G:H] \cdot m.$$

The right side of the Mackey formula gives:

$$\sum_{[g] \in K\backslash G/H} \mathrm{tr}_{K \cap {}^gH}^K \circ c_g \circ \mathrm{res}_{K^g \cap H}^H(m) = \sum_{[g] \in K\backslash G/H} [K : K \cap {}^gH] \cdot m.$$

Since $\sum_{[g] \in K\backslash G/H} [K : K \cap {}^gH] = [G:H]$ (this is the standard double coset counting formula), the Mackey formula holds. ✓

> [!WARNING] Transfers Are Not Identities
> A common error is to confuse the constant Mackey functor with the constant coefficient system (where transfers are simply identity maps). In the constant Mackey functor, transfers are multiplication by index — *not* the identity. The constant coefficient system does not extend to a Mackey functor unless $A = 0$ or all index multiplicities are trivially 1.

### 5.2 The Burnside Ring Mackey Functor

The most fundamental Mackey functor is the *Burnside ring Mackey functor* $\underline{A}$.

**Definition (Burnside Ring Mackey Functor).** Define $\underline{A}$ by:
- $\underline{A}(G/H) = A(H)$, the Burnside ring of $H$,
- $\mathrm{res}_H^K: A(K) \to A(H)$ is restriction of the $K$-action to $H$: $[S] \mapsto [S|_H]$,
- $\mathrm{tr}_H^K: A(H) \to A(K)$ is induction: $[S] \mapsto [K \times_H S]$ where $K \times_H S = K \times S / (kh, s) \sim (k, hs)$,
- $c_g: A(H) \to A({}^gH)$ sends $[S]$ to $[{}^gS]$ where ${}^gS$ has the action $h \cdot s = g^{-1}hg \cdot s$.

The Burnside ring Mackey functor is the *representable* Mackey functor $\mathcal{A}(G)(G/G, -)$ — it is represented by the terminal $G$-set. In terms of the Yoneda lemma for additive categories, $\underline{A}(G/H) = \mathcal{A}(G)(G/G, G/H) = A(H)$.

> [!INFO] Universal Property
> The Burnside ring Mackey functor $\underline{A}$ is the *unit* for the box product (see §6). Every Mackey functor $M$ admits a unique unital map $\underline{A} \to M$ of Green functors (when $M$ has a Green functor structure). In this sense $\underline{A}$ plays the role of $\mathbb{Z}$ among abelian groups.

### 5.3 Fixed-Point Mackey Functors

**Definition (Fixed-Point Mackey Functor).** For a $G$-space $X \in G\mathbf{Top}$ (see [[concepts/equivariant-stable-homotopy/g-spaces-and-equivariant-maps|G-Spaces and Equivariant Maps]]) and $n \geq 0$, define the *homotopy Mackey functor* $\underline{\pi}_n(X)$ by:

$$\underline{\pi}_n(X)(G/H) = \pi_n(X^H),$$

where $X^H = \{x \in X : hx = x \text{ for all } h \in H\}$ is the $H$-fixed-point subspace.

The restriction maps $\mathrm{res}_H^K: \pi_n(X^K) \to \pi_n(X^H)$ are induced by the inclusion $X^K \hookrightarrow X^H$ (for $H \leq K$). The transfer maps are more subtle and require the equivariant transfer map in homotopy theory — they are not simply functorial in the obvious sense.

For a genuine $G$-spectrum $E$ (see [[concepts/equivariant-stable-homotopy/g-spectra|G-Spectra]] *(no note yet)*), the homotopy groups

$$\underline{\pi}_n(E)(G/H) = \pi_n(E^H)$$

form a Mackey functor for every $n \in \mathbb{Z}$, and the collection $\{\underline{\pi}_n(E)\}_{n \in \mathbb{Z}}$ is the primary algebraic invariant of $E$.

> [!NOTE] The Role in the Slice Spectral Sequence
> In the [[concepts/equivariant-stable-homotopy/equivariant-postnikov-and-slice|Equivariant Postnikov and Slice]] context, the $E_2$-page of the slice spectral sequence takes values in Mackey functors: $E_2^{s,t} = H_{\mathrm{Bredon}}^s(G; \underline{\pi}_t(E))$. The Mackey functor $\underline{\pi}_t(E)$ captures the correct coefficient system for RO(G)-graded Bredon cohomology.

### 5.4 The C2 Classification

For $G = C_2 = \{e, \tau\}$, there is a simple classification of all $C_2$-Mackey functors. A $C_2$-Mackey functor consists of a diagram

$$M(C_2/C_2) \underset{\mathrm{tr}}{\overset{\mathrm{res}}{\rightleftharpoons}} M(C_2/e)$$

where $\tau$ acts on $M(C_2/e)$ (by $c_\tau$), subject to:

1. **Mackey formula:** $\mathrm{res} \circ \mathrm{tr}(m) = m + \tau \cdot m$ for all $m \in M(C_2/e)$.
2. **Trace formula:** $\mathrm{tr} \circ \mathrm{res}(n) = \mathrm{tr}(\mathrm{res}(n))$ for $n \in M(C_2/C_2)$ (automatic from transitivity).

(The conjugation axiom for $M(C_2/C_2)$ is trivial since $C_2/C_2$ has only one coset.)

The four most important $C_2$-Mackey functors are:

| Name | $M(C_2/C_2)$ | $M(C_2/e)$ | res | tr | $\tau$-action |
|------|-------------|-----------|-----|-----|---------------|
| $\underline{\mathbb{Z}}$ | $\mathbb{Z}$ | $\mathbb{Z}$ | id | $\times 2$ | id |
| $\underline{\mathbb{Z}}^-$ | $0$ | $\mathbb{Z}$ | $0$ | $0$ | $\times(-1)$ |
| $A(C_2)$ | $\mathbb{Z}^2$ | $\mathbb{Z}$ | $[1,1]$ | $[1;1]^T$ | id |
| $\mathbb{Z}/2$ | $\mathbb{Z}/2$ | $0$ | $0$ | $0$ | id |

Let us verify the Mackey formula for $\underline{\mathbb{Z}}$: $\mathrm{res} \circ \mathrm{tr}(m) = \mathrm{id}(2m) = 2m = m + \tau \cdot m = m + m = 2m$ ✓ (since $\tau$ acts by identity on $M(C_2/e) = \mathbb{Z}$).

> [!EXAMPLE] The Lewis Diagram for Underline Z
> The constant Mackey functor $\underline{\mathbb{Z}}$ is often drawn as:
> $$\mathbb{Z} \underset{2}{\overset{1}{\rightleftharpoons}} \mathbb{Z}$$
> where the top arrow is $\mathrm{res} = \mathrm{id}$ and the bottom arrow is $\mathrm{tr} = \times 2$. This Lewis diagram notation — abelian groups at the nodes with res/tr as arrows — is the standard way to specify a $C_2$-Mackey functor.

> [!WARNING] Constant Functor Is Not the Same as the Constant Coefficient System
> In the constant Mackey functor $\underline{\mathbb{Z}}$, the transfer $C_2/e \to C_2/C_2$ is multiplication by 2 (the index $[C_2 : e] = 2$), *not* the identity. The constant coefficient system (restriction = id, transfer = id) does not satisfy the Mackey formula.

---

## 6. The Box Product and Green/Tambara Functors 💡

### 6.1 Day Convolution and the Box Product

The category $\mathrm{Mack}(G)$ carries a symmetric monoidal product, the *box product* $\square$, which is the analogue of the tensor product of abelian groups.

**Definition (Box Product).** For Mackey functors $M$ and $N$, their *box product* $M \square N$ is the *Day convolution* of $M$ and $N$ with respect to the symmetric monoidal structure on $\mathcal{A}(G)$ given by Cartesian product of $G$-sets:

$$(M \square N)(G/H) = \int^{X, Y \in \mathcal{A}(G)} \mathcal{A}(G)(X \times Y, G/H) \otimes_{\mathbb{Z}} M(X) \otimes_{\mathbb{Z}} N(Y).$$

Here the coend $\int^{X,Y}$ runs over pairs of objects of $\mathcal{A}(G)$, and $\otimes_\mathbb{Z}$ denotes the usual tensor product of abelian groups.

Concretely, $(M \square N)(G/H)$ is generated by symbols $m \otimes_H n$ for $m \in M(G/H)$, $n \in N(G/H)$, subject to the bilinearity relations coming from the $\mathcal{A}(G)$-enrichment. The key relations imposed are:

$$\mathrm{tr}_K^H(m) \otimes_H n = \mathrm{tr}_K^H(m \otimes_K \mathrm{res}_K^H(n))$$

(the *Frobenius reciprocity*-type relation in the definition of the box product).

The unit for $\square$ is the Burnside ring Mackey functor $\underline{A}$: there are natural isomorphisms $\underline{A} \square M \cong M \cong M \square \underline{A}$.

> [!INFO] Comparison with Tensor Product of Modules
> The box product $\square$ is the correct analogue of $\otimes_\mathbb{Z}$, but it is *not* the pointwise tensor product. The pointwise tensor product (taking $M(G/H) \otimes N(G/H)$ at each orbit) does not satisfy the Mackey formula for the resulting transfer maps. The Day convolution corrects this by building in the span structure.

### 6.2 Green Functors

**Definition (Green Functor).** A *Green functor* for $G$ is a commutative monoid in $(\mathrm{Mack}(G), \square, \underline{A})$. Explicitly, it is a Mackey functor $R$ equipped with:
- A unit $\eta: \underline{A} \to R$,
- A multiplication $\mu: R \square R \to R$,

satisfying commutativity, associativity, and unitality in $\mathrm{Mack}(G)$.

Unpacking the definition: a Green functor $R$ is a Mackey functor such that each $R(G/H)$ is a commutative ring with unit, the restriction maps $\mathrm{res}_H^K: R(K) \to R(H)$ are ring homomorphisms, and the *Frobenius reciprocity* condition holds:

$$\mathrm{tr}_H^K(a) \cdot b = \mathrm{tr}_H^K\bigl(a \cdot \mathrm{res}_H^K(b)\bigr) \quad \text{for all } a \in R(G/H), b \in R(G/K).$$

> [!EXAMPLE] The Burnside Ring is a Green Functor
> The Burnside ring Mackey functor $\underline{A}$ is a Green functor: $A(H)$ is a commutative ring under Cartesian product of $H$-sets, restriction of scalars is a ring map, and the Frobenius condition follows from the fact that $K \times_H (S \times \mathrm{res}_H^K(T)) \cong (K \times_H S) \times T$ as $K$-sets.

> [!EXAMPLE] Representation Ring Green Functor
> The *representation ring Mackey functor* $\underline{R}$ defined by $\underline{R}(G/H) = R(H)$ (the complex representation ring of $H$) is a Green functor: restriction of representations is a ring map, induction is the transfer, and Frobenius reciprocity is the classical statement $\mathrm{Ind}_H^K(V) \otimes W \cong \mathrm{Ind}_H^K(V \otimes \mathrm{Res}_H^K(W))$.

### 6.3 Tambara Functors and Multiplicative Norms 🔑

Green functors capture additive ring structure with transfers. But genuine equivariant commutative ring spectra have *multiplicative norms* — maps $N_H^G: M(G/H) \to M(G/G)$ that are *multiplicative*, not just additive. This extra structure is axiomatized by Tambara functors.

**Definition (Tambara Functor).** A *Tambara functor* for $G$ is a commutative monoid in a suitable symmetric monoidal $\infty$-category of Mackey functors that also carries *multiplicative norm maps* $N_H^K: M(G/H) \to M(G/K)$ for $H \leq K$, satisfying:

1. $N_H^K$ is a (not necessarily additive) multiplicative map: $N_H^K(1) = 1$ and $N_H^K(ab) = N_H^K(a) N_H^K(b)$,
2. **Distributivity:** $\mathrm{tr}_H^K(a) \cdot b = \mathrm{tr}_H^K(a \cdot \mathrm{res}_H^K(b))$ (Frobenius) and $N_H^K(\mathrm{res}_H^K(b) \cdot a) = b^{[K:H]} \cdot N_H^K(a)$ (norm-restriction compatibility),
3. **Norm-transfer compatibility:** $N_H^K(\mathrm{tr}_{H'}^H(a)) = \mathrm{tr}_{K'}^K(N_{H'}^{K'}(a) \cdot c)$ for suitable $H' \leq H$ and $K' = \mathrm{ind}_H^K H'$ (this is the *norm-transfer formula*, which encodes the Tambara polynomial functor structure).

The key example: if $E$ is a genuine equivariant commutative ring $G$-spectrum (an $E_\infty$-algebra in $\mathrm{Sp}^G$), then $\underline{\pi}_0(E)$ is a Tambara functor. The norm maps $N_H^G$ on $\underline{\pi}_0(E)$ come from the multiplicative norm maps $N_H^G: E^H \to E^G$ in spectra (the Hill-Hopkins-Ravenel norm).

> [!INFO] Historical Context
> Tambara functors were introduced by D. Tambara in 1993 as "TNR-functors" (Transfer-Norm-Restriction). They were later recognized as the algebraic structure carried by $\pi_0$ of genuine equivariant commutative ring spectra. The connection to $N_\infty$-operads (Blumberg-Hill 2015) clarifies which norm maps a given commutative ring spectrum is required to admit.

> [!WARNING] Not Every Green Functor is Tambara
> The inclusion $\{\text{Tambara functors}\} \subsetneq \{\text{Green functors}\}$ is strict. A Green functor has additive transfers but no multiplicative norms. A Tambara functor has both additive transfers and multiplicative norms, plus the distributivity law relating them. The extra structure is highly non-trivial and imposes strong constraints.

---

## 7. Projective Mackey Functors and Resolutions 🧮

### 7.1 Representable Mackey Functors

The category $\mathrm{Mack}(G)$ is an abelian category, and like any functor category, it has a natural supply of projective objects coming from the Yoneda lemma.

**Definition (Representable Mackey Functor).** For a subgroup $H \leq G$, define the *representable Mackey functor* $\mathbb{Z}[G/H, -] = \mathcal{A}(G)(G/H, -)$ by:

$$\mathbb{Z}[G/H, -](G/K) = \mathcal{A}(G)(G/H, G/K).$$

Since $\mathcal{A}(G)$ is additive, $\mathbb{Z}[G/H, -]$ is an additive (hence Mackey) functor. By the Yoneda lemma for additive categories, there is a natural isomorphism

$$\mathrm{Hom}_{\mathrm{Mack}(G)}(\mathbb{Z}[G/H, -], M) \cong M(G/H)$$

for any Mackey functor $M$.

**Proposition.** *The representable Mackey functor $\mathbb{Z}[G/H, -]$ is projective in $\mathrm{Mack}(G)$.*

*Proof sketch.* The Yoneda isomorphism shows that $\mathrm{Hom}(\mathbb{Z}[G/H, -], -)$ is naturally isomorphic to evaluation at $G/H$, which is exact (since limits/colimits in $\mathrm{Mack}(G)$ are computed pointwise). $\square$

The representable Mackey functor $\mathbb{Z}[G/H, -]$ evaluates explicitly as:

$$\mathbb{Z}[G/H, -](G/K) = \mathcal{A}(G)(G/H, G/K) \cong \bigoplus_{[x] \in H\backslash G/K} \mathbb{Z} \cdot [KxH],$$

i.e., the free abelian group on the set of double cosets $H\backslash G/K$.

> [!EXAMPLE] Representable for H = G
> The representable Mackey functor $\mathbb{Z}[G/G, -]$ is the Burnside ring Mackey functor $\underline{A}$: $\mathbb{Z}[G/G, -](G/K) = \mathcal{A}(G)(G/G, G/K) = A(K)$ (the Burnside ring of $K$). This is consistent with $\underline{A}$ being the unit for the box product.

### 7.2 Global Dimension and Resolutions 🔑

**Theorem (Projectivity and Generators).** The representable Mackey functors $\{\mathbb{Z}[G/H, -] : H \leq G\}$ (one for each conjugacy class of subgroups) form a *generating set* of projectives for $\mathrm{Mack}(G)$. Every Mackey functor $M$ admits a projective resolution:

$$0 \longleftarrow M \longleftarrow P_0 \longleftarrow P_1 \longleftarrow \cdots$$

where each $P_i$ is a direct sum of representable Mackey functors.

**Theorem (Global Dimension).** *The global dimension of $\mathrm{Mack}(G)$ is finite.* Specifically, if $G$ has virtual cohomological dimension $\mathrm{vcd}(G)$, then every Mackey functor has a projective resolution of length at most $\mathrm{vcd}(G) + 1$.

For a finite group $G$ of order $n$, the global dimension is bounded: projective resolutions exist of length $\leq 2$ after inverting $|G|$ (since $|G|$-torsion is the obstruction). Over $\mathbb{Z}[1/|G|]$, the category $\mathrm{Mack}(G)[\frac{1}{|G|}]$ is semisimple.

The derived functors of $\mathrm{Hom}_{\mathrm{Mack}(G)}$ give *Ext groups for Mackey functors*:

$$\mathrm{Ext}^n_{\mathrm{Mack}(G)}(M, N) = H^n(\mathrm{Hom}_{\mathrm{Mack}(G)}(P_\bullet, N))$$

and similarly for $\mathrm{Tor}_n^{\mathrm{Mack}(G)}(M, N)$ using the box product. **These Ext groups appear as the $E_2$-page of the slice spectral sequence for a genuine $G$-spectrum $E$:**

$$E_2^{s,t} \cong \mathrm{Ext}^s_{\mathrm{Mack}(G)}(\underline{\pi}_t(E), \underline{A}) \Rightarrow \pi_{t-s}(E^G).$$

> [!WARNING] Projective vs. Free
> Unlike the category of abelian groups where projective = free for countably generated modules, projective Mackey functors are *not* representable in general — they are direct summands of representable ones. The distinction matters for computing projective resolutions.

> [!INFO] Thévenaz-Webb Structure Theorem
> Thévenaz and Webb (1995) showed that $\mathrm{Mack}(G)$ is equivalent (via the *Brauer quotient* functor) to a product of module categories over twisted group rings. This gives a complete structural classification: the simple Mackey functors $S_{H,V}$ are indexed by pairs $(H, V)$ with $H$ a subgroup of $G$ up to conjugacy and $V$ a simple $\mathbb{Z}[N_G(H)/H]$-module.

---

## 8. Spectral Mackey Functors: Barwick's Theorem 💡

### 8.1 The Effective Burnside Infinity-Category

The classical Burnside category $\mathcal{A}(G)$ is a 1-category. To relate Mackey functors to genuine $G$-spectra, we need the $\infty$-categorical lift due to Barwick (2014).

**Definition (Effective Burnside $\infty$-Category).** Let $\mathcal{F}_G$ denote the $\infty$-category of finite $G$-sets (regarded as a 1-category, hence an $\infty$-category with only homotopy-discrete morphism spaces). The *effective Burnside $\infty$-category* $\mathcal{A}^{\mathrm{eff}}_\infty(G)$ is the $\infty$-category of *spans* in $\mathcal{F}_G$: its objects are finite $G$-sets, and its morphisms are spans $X \leftarrow Z \rightarrow Y$ where composition is via homotopy pullback (which for sets is the ordinary pullback, but the $\infty$-categorical framework tracks all higher coherences).

More precisely, $\mathcal{A}^{\mathrm{eff}}_\infty(G)$ is constructed as the *effective Burnside $\infty$-category* of the *disjunctive triple* $(\mathcal{F}_G, \mathcal{F}_G, \mathcal{F}_G)$ in Barwick's sense: a triple $(\mathcal{C}, \mathcal{C}^\dagger, \mathcal{C}_\dagger)$ where both the "ingressive" maps $\mathcal{C}^\dagger$ and "egressive" maps $\mathcal{C}_\dagger$ are taken to be all maps (since all maps of finite $G$-sets may appear as either leg of a span).

The key property is that $\mathcal{A}^{\mathrm{eff}}_\infty(G)$ is the *homotopy-coherent* version of span composition: the associativity of fiber products is handled up to coherent homotopy, not just up to isomorphism.

> [!INFO] Relation to the Classical Burnside Category
> The homotopy category $h(\mathcal{A}^{\mathrm{eff}}_\infty(G))$ recovers the pre-Burnside category $\mathcal{A}^+(G)$ (before group completion of the morphism monoids). Taking $\pi_0$ of the mapping spaces and group-completing recovers the classical Burnside category $\mathcal{A}(G)$.

### 8.2 Spectral Mackey Functors

**Definition (Spectral Mackey Functor).** A *spectral Mackey functor* for $G$ is an *additive* (equivalently, finite-coproduct-preserving) functor

$$\mathcal{M}: \mathcal{A}^{\mathrm{eff}}_\infty(G) \longrightarrow \mathbf{Sp}$$

where $\mathbf{Sp}$ is the $\infty$-category of spectra. Additivity means $\mathcal{M}(X \sqcup Y) \simeq \mathcal{M}(X) \vee \mathcal{M}(Y)$ (wedge of spectra).

The $\infty$-category of spectral Mackey functors is

$$\mathrm{SpMack}(G) = \mathrm{Fun}^{\oplus}(\mathcal{A}^{\mathrm{eff}}_\infty(G), \mathbf{Sp}).$$

This is a presentable stable $\infty$-category and carries a symmetric monoidal structure (the spectral Day convolution).

> [!NOTE] Why Spectra, Not Abelian Groups?
> Replacing $\mathbf{Ab}$ with $\mathbf{Sp}$ is essential for capturing the full stable equivariant theory. Abelian groups embed into spectra as Eilenberg-MacLane spectra (via $A \mapsto HA$), so classical Mackey functors embed into spectral ones. But the stable equivariant information — transfer maps, RO(G)-graded homotopy groups — requires the full spectrum-valued picture.

### 8.3 Barwick's Equivalence

The central theorem of Barwick's 2014 paper is:

**Theorem (Barwick 2014).** *There is an equivalence of $\infty$-categories*

$$\mathrm{Fun}^{\oplus}(\mathcal{A}^{\mathrm{eff}}_\infty(G), \mathbf{Sp}) \simeq \mathrm{Sp}^G,$$

*where $\mathrm{Sp}^G$ is the $\infty$-category of genuine $G$-spectra (orthogonal $G$-spectra localized at genuine equivalences).*

*Proof sketch.* The forward direction sends a spectral Mackey functor $\mathcal{M}$ to the genuine $G$-spectrum $\mathcal{M}(G/G)$ together with the action of the Burnside category on the collection $\{\mathcal{M}(G/H)\}$. The reverse direction sends a genuine $G$-spectrum $E$ to the functor $G/H \mapsto E^H$ (categorical fixed-point spectrum), with the restriction and transfer maps encoded by the span functoriality.

The key technical input is that the span functoriality on $\{E^H\}$ — specifically the existence of coherent transfer maps — is exactly what genuine $G$-spectra have and naive $G$-spectra lack.

**Corollary.** *Eilenberg-MacLane spectral Mackey functors $H\underline{M}$ for classical Mackey functors $\underline{M} \in \mathrm{Mack}(G)$ correspond under Barwick's equivalence to the genuine Eilenberg-MacLane $G$-spectra of equivariant stable homotopy theory.* These are the coefficient objects for RO(G)-graded Bredon cohomology.

> [!INFO] Guillou-May Comparison
> Barwick's theorem recovers (and gives a conceptually clean proof of) the earlier Guillou-May theorem (2011) identifying naive spectral Mackey functors with a specific model for genuine $G$-spectra. Barwick's version is more conceptual: it identifies the *universal property* of genuine $G$-spectra as "spectral-valued functors on the Burnside $\infty$-category."

### 8.4 Homotopy Groups as Classical Mackey Functors 🔑

Under Barwick's equivalence, the relationship between spectral and classical Mackey functors is mediated by the truncation functors on spectra.

**Proposition.** *Let $\mathcal{M}: \mathcal{A}^{\mathrm{eff}}_\infty(G) \to \mathbf{Sp}$ be a spectral Mackey functor, and let $E = \mathcal{M}(G/G)$ be the corresponding genuine $G$-spectrum. Then for each $n \in \mathbb{Z}$, the classical Mackey functor*

$$\underline{\pi}_n(\mathcal{M}): G/H \longmapsto \pi_n(\mathcal{M}(G/H))$$

*is the $n$-th homotopy Mackey functor of $E$.*

*Proof sketch.* Since $\mathcal{M}$ sends spans to maps of spectra, and restriction/transfer in the $G$-spectrum $E$ are encoded by specific spans in $\mathcal{A}^{\mathrm{eff}}_\infty(G)$, applying $\pi_n$ levelwise gives a classical Mackey functor with the correct restrictions and transfers. The Mackey formula holds because it holds at the level of $\pi_n$ applied to the span decomposition of pullbacks. $\square$

**The homotopy groups of a genuine $G$-spectrum $E$ are not merely abelian groups — they are Mackey functors.** This is the fundamental reason Mackey functors appear as the coefficient objects for equivariant cohomology: they encode the simultaneous data of all fixed-point homotopy groups together with the restriction and transfer maps between them.

> [!INFO] Postnikov Towers and Slices
> The slice filtration (discussed in [[concepts/equivariant-stable-homotopy/equivariant-postnikov-and-slice|Equivariant Postnikov and Slice]]) is an equivariant analogue of the Postnikov tower where the fibers are genuine Eilenberg-MacLane spectra $H\underline{M}$ for Mackey functors $\underline{M}$. The $k$-invariants of this filtration lie in RO(G)-graded Bredon cohomology with Mackey functor coefficients.

> [!QUESTION] Open Problem: Higher Segal Conditions
> Barwick's theorem characterizes genuine $G$-spectra as additive functors on $\mathcal{A}^{\mathrm{eff}}_\infty(G)$. A natural question: what does the *non-additive* (semi-additive or $n$-semiadditive) version give? Recent work of Carmeli-Schlank-Yanovski on higher semiadditivity suggests a rich generalization. The precise relationship to $p$-typical equivariant theories and cyclotomic spectra remains an active area.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|----------------|--------------|------------------|
| [A Guide to Mackey Functors (Webb 2000)](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/WebbMF.pdf) | Comprehensive handbook survey of Mackey functor algebra: projective resolutions, Green functors, connections to representation theory | [PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/WebbMF.pdf) |
| [Spectral Mackey Functors and Equivariant Algebraic K-Theory (Barwick 2014)](https://arxiv.org/abs/1404.0108) | Defines the effective Burnside $\infty$-category and proves genuine $G$-spectra $\simeq$ spectral Mackey functors; establishes unfurling construction for equivariant K-theory | [arXiv:1404.0108](https://arxiv.org/abs/1404.0108) |
| [M392C Equivariant Homotopy Theory Lecture Notes (Blumberg/Debray 2017)](https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf) | Best single survey: G-spaces, genuine G-spectra, Mackey functors, RO(G)-grading, HHR theorem | [PDF](https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf) |
| [Equivariant Homotopy and Cohomology Theory (May et al. 1996)](https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf) | The Alaska notes: complete classical treatment of G-spectra, RO(G)-graded theories, Mackey functors in §§IX–X | [PDF](https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf) |
| [The Structure of Mackey Functors (Thévenaz-Webb 1995)](https://www.ams.org/journals/tran/1995-347-06/S0002-9947-1995-1261590-5/) | Complete structural classification of Mackey functors via Brauer quotient; simple Mackey functors indexed by $(H, V)$ | [AMS TRAN](https://www.ams.org/journals/tran/1995-347-06/S0002-9947-1995-1261590-5/) |
| [Contributions to the Theory of Induced Representations (Dress 1973)](https://link.springer.com/chapter/10.1007/BFb0061381) | Original introduction of Mackey functors as "functors with two structures" (Batelle Institute conference proceedings) | [Springer](https://link.springer.com/chapter/10.1007/BFb0061381) |
| [A Remark on Mackey-Functors (Lindner 1976)](https://link.springer.com/article/10.1007/BF01245921) | Proves equivalence between Mackey functors and additive functors on the span category (Lindner's theorem) | [Springer](https://link.springer.com/article/10.1007/BF01245921) |
| [Mackey Functor (nLab)](https://ncatlab.org/nlab/show/Mackey+functor) | Online encyclopedic reference with modern categorical perspective; links to related structures | [nLab](https://ncatlab.org/nlab/show/Mackey+functor) |
| [Burnside Category (Wikipedia)](https://en.wikipedia.org/wiki/Burnside_category) | Concise definition of the Burnside category as the span category of finite G-sets; relation to Mackey functors | [Wikipedia](https://en.wikipedia.org/wiki/Burnside_category) |
| [Equivariant Spectra and Mackey Functors (Rubin 2019)](https://iwoat.github.io/2019/notes/Lecture-10.pdf) | Lecture notes connecting Mackey functors to genuine G-spectra; good modern exposition | [PDF](https://iwoat.github.io/2019/notes/Lecture-10.pdf) |
| [Operadic Multiplications in Equivariant Spectra, Norms, and Transfers (Blumberg-Hill 2015)](https://arxiv.org/abs/1309.1750) | $N_\infty$-operads and the relationship between Tambara functors and multiplicative norm maps in genuine equivariant commutative ring spectra | [arXiv:1309.1750](https://arxiv.org/abs/1309.1750) |
| [On the Non-Existence of Elements of Kervaire Invariant One (Hill-Hopkins-Ravenel 2016)](https://arxiv.org/abs/0908.3724) | Uses Mackey functor-valued homotopy groups of genuine $C_8$-spectra as central tool; the norm $N_{C_2}^{C_8}$ is a key example of a Tambara functor map | [arXiv:0908.3724](https://arxiv.org/abs/0908.3724) |
