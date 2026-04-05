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
- [[#Exercises|Exercises]]
  - [[#Mathematical Development|Mathematical Development]]
  - [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#G-CW-Spectra and Slice Filtration|G-CW-Spectra and Slice Filtration]]
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

## Exercises 📝

### Mathematical Development

#### Problem 1: Morphism Groups of the Burnside Category for C2

*This problem makes the Burnside category concrete by computing all four morphism groups $\mathcal{A}(C_2)(X, Y)$ explicitly, verifying the note's claim that spans between the two orbits $C_2/e$ and $C_2/C_2$ generate free abelian groups of specified ranks.*

> **Prerequisites:** cf. [[#2.1 Spans of Finite G-Sets|§2.1 — Spans of Finite G-Sets]]; cf. [[#2.3 Additive Structure and the Burnside Ring|§2.3 — Additive Structure and the Burnside Ring]]

Let $G = C_2 = \{e, \tau\}$. Write $* = C_2/C_2$ and $C_2 = C_2/e$ for the two isomorphism classes of transitive $G$-sets.

(a) List all isomorphism classes of spans $* \leftarrow Z \rightarrow *$. Show that $\mathcal{A}^+(C_2)(*, *) \cong \mathbb{N} \times \mathbb{N}$ as a commutative monoid, with generators $[* \leftarrow * \rightarrow *]$ and $[* \leftarrow C_2 \rightarrow *]$. Conclude $\mathcal{A}(C_2)(*, *) \cong \mathbb{Z}^2$.

(b) List all isomorphism classes of spans $C_2 \leftarrow Z \rightarrow *$. Show that $\mathcal{A}(C_2)(C_2, *) \cong \mathbb{Z}$, generated by the span $C_2 \xleftarrow{\mathrm{id}} C_2 \xrightarrow{p} *$ where $p$ is the unique map to the point.

(c) List all isomorphism classes of spans $* \leftarrow Z \rightarrow C_2$. Show that $\mathcal{A}(C_2)(*, C_2) \cong \mathbb{Z}$, generated by the span $* \xleftarrow{p} C_2 \xrightarrow{\mathrm{id}} C_2$.

(d) Show that $\mathcal{A}(C_2)(C_2, C_2) \cong \mathbb{Z}^2$ by classifying all spans $C_2 \leftarrow Z \rightarrow C_2$ up to isomorphism and identifying the two generators.

> [!TIP]- Solution to Exercise 1
> **Key insight:** Every $C_2$-set decomposes uniquely as a disjoint union of copies of $*$ and $C_2$, so the middle piece $Z$ of any span is determined by two non-negative integers $(a, b)$ counting the two orbit types.
>
> **Sketch:**
>
> A span $* \leftarrow Z \rightarrow *$ is determined by the isomorphism class of $Z$ (both maps to $*$ are forced). Every $C_2$-set $Z \cong (*^{\sqcup a}) \sqcup (C_2^{\sqcup b})$, giving a monoid $\mathbb{N} \times \mathbb{N}$ with generators $[*]$ and $[C_2]$; group completion yields $\mathbb{Z}^2$.
>
> For $\mathcal{A}(C_2)(C_2, *)$: a span $C_2 \leftarrow Z \rightarrow *$ requires a $G$-equivariant map $Z \to C_2$. The equivariant maps $Z \to C_2$ exist only when $Z$ has a free orbit; the only connected such $Z$ is $Z = C_2$ with the identity. Thus the monoid is $\mathbb{N}$, giving $\mathbb{Z}$.
>
> For $\mathcal{A}(C_2)(C_2, C_2)$: the middle piece $Z$ admits equivariant maps to $C_2$ on both sides. The connected possibilities are $Z = C_2$ (identity span) or $Z = *$ (folding span). The two generators are $[\mathrm{id}_{C_2}]$ and $[* \leftarrow * \rightarrow C_2]$; group completion gives $\mathbb{Z}^2$.

---

#### Problem 2: Morphism Groups of the Burnside Category for C3

*This problem extends the $C_2$ computation to $C_3$, where the orbit structure of the middle piece $Z$ is richer, producing a rank-2 morphism group for $\mathcal{A}(C_3)(C_3, C_3)$ from different generators.*

> **Prerequisites:** cf. [[#2.1 Spans of Finite G-Sets|§2.1 — Spans of Finite G-Sets]]; requires Problem 1

Let $G = C_3$. Write $* = C_3/C_3$ and $C_3 = C_3/e$.

(a) List all isomorphism classes of $C_3$-sets of order $\leq 6$.

(b) Compute the monoid $\mathcal{A}^+(C_3)(*, *)$ and show $\mathcal{A}(C_3)(*, *) \cong \mathbb{Z}^2$.

(c) Show that the morphism group $\mathcal{A}(C_3)(C_3, *) \cong \mathbb{Z}$, with the sole generator being the span $C_3 \xleftarrow{\mathrm{id}} C_3 \xrightarrow{p} *$.

(d) Classify all spans $C_3 \leftarrow Z \rightarrow C_3$ by listing all possible middle pieces $Z$. Show that $\mathcal{A}(C_3)(C_3, C_3) \cong \mathbb{Z}^2$, and identify the two generators.

> [!TIP]- Solution to Exercise 2
> **Key insight:** The argument is identical to the $C_2$ case — the orbit classification forces every middle piece to be a disjoint union of $*$ and $C_3$, and one counts equivariant maps to each endpoint to enumerate span types.
>
> **Sketch:**
>
> Every finite $C_3$-set is $*^{\sqcup a} \sqcup C_3^{\sqcup b}$. The monoid $\mathcal{A}^+(C_3)(*, *) \cong \mathbb{N} \times \mathbb{N}$ (generators $[*]$ and $[C_3]$), so $\mathcal{A}(C_3)(*,*) \cong \mathbb{Z}^2$.
>
> For $\mathcal{A}(C_3)(C_3, *)$: the only equivariant map $Z \to C_3$ from a transitive $C_3$-set is $Z = C_3$. The monoid is $\mathbb{N}$, giving $\mathbb{Z}$.
>
> For $\mathcal{A}(C_3)(C_3, C_3)$: connected middle pieces admitting equivariant maps to $C_3$ on both sides: (i) $Z = C_3$ giving the identity span and "diagonal" span $C_3 \xleftarrow{g \mapsto g} C_3 \xrightarrow{g \mapsto g\sigma} C_3$; (ii) $Z = *$ with constant maps (a "folding" span). Group completion of the two-generator monoid gives $\mathbb{Z}^2$.

---

#### Problem 3: The Burnside Ring as an Endomorphism Ring

*This problem proves the identification $A(G) = \mathcal{A}(G)(G/G, G/G)$ rigorously, by showing that spans $G/G \leftarrow Z \rightarrow G/G$ are simply finite $G$-sets and that composition becomes Cartesian product.*

> **Prerequisites:** cf. [[#2.3 Additive Structure and the Burnside Ring|§2.3 — Additive Structure and the Burnside Ring]]

(a) Show that a span $G/G \leftarrow Z \rightarrow G/G$ is determined up to isomorphism by the $G$-set $Z$ alone. Conclude that $\mathcal{A}^+(G)(G/G, G/G)$ is the commutative monoid of isomorphism classes of finite $G$-sets under disjoint union.

(b) Show that the composition of spans $* \leftarrow Z \rightarrow *$ and $* \leftarrow W \rightarrow *$ in $\mathcal{A}(G)$ corresponds to Cartesian product $Z \times W$ of finite $G$-sets. Conclude that the ring structure on $K_0(\mathbf{FSets}_G)$ induced by span composition is the Burnside ring product.

(c) For $G = C_2$: write out the multiplication table of $\mathcal{A}(C_2)(*, *)$ in the basis $\{[*], [C_2]\}$ using part (b). Verify that $[C_2]^2 = 2[C_2]$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** Since $G/G = *$ is the terminal object, both legs of any span $* \leftarrow Z \rightarrow *$ are the unique map to $*$, so the span is completely determined by the $G$-set $Z$ itself; composition of spans then becomes Cartesian product.
>
> **Sketch:**
>
> For any $G$-set $Z$, the unique maps $Z \to *$ provide the legs; two spans are isomorphic iff $Z \cong Z'$ as $G$-sets. So $\mathcal{A}^+(G)(*,*) \cong (\mathbf{FSets}_G)^{\cong}$, and $\mathcal{A}(G)(*,*) = K_0(\mathbf{FSets}_G) = A(G)$.
>
> Composition: $(* \leftarrow Z \rightarrow *) \circ (* \leftarrow W \rightarrow *) = (* \leftarrow Z \times_* W \rightarrow *)$ and $Z \times_* W = Z \times W$. This matches the Burnside ring product $[Z] \cdot [W] = [Z \times W]$.
>
> For $C_2$: $[C_2]^2 = [C_2 \times C_2]$. The diagonal $C_2$-action on $C_2 \times C_2$ has orbits $\{(e,e),(\tau,\tau)\}$ and $\{(e,\tau),(\tau,e)\}$, each a copy of $C_2$. So $[C_2]^2 = 2[C_2]$ in $A(C_2)$. ✓

---

#### Problem 4: Span Composition and the Mackey Formula for C4

*This problem uses the fiber product formula to compute $\mathrm{res}_{C_2} \circ \mathrm{tr}_{C_2}^{C_4}$ for the unique index-2 subgroup $C_2 \leq C_4$, verifying that the answer is $1 + \tau_*$ where $\tau \in C_2$ is the generator.*

> **Prerequisites:** cf. [[#4.1 The Pullback Decomposition|§4.1 — The Pullback Decomposition]]; cf. [[#4.2 Derivation from Spans|§4.2 — Derivation from Spans]]

Let $G = C_4 = \langle \sigma \rangle$ and $H = K = C_2 = \langle \sigma^2 \rangle \leq C_4$.

(a) List the double cosets in $K\backslash G/H = C_2\backslash C_4/C_2$ and find a set of representatives.

(b) For each double coset representative $[g]$, compute the intersection $K \cap {}^gH$, the conjugation map $c_g$, and the contribution $\mathrm{tr}_{K \cap {}^gH}^K \circ c_g \circ \mathrm{res}_{K^g \cap H}^H$ to the Mackey formula.

(c) Conclude that $\mathrm{res}_{C_2}^{C_4} \circ \mathrm{tr}_{C_2}^{C_4}(m) = m + \tau_*(m)$ for any Mackey functor $M$ and $m \in M(C_4/C_2)$, where $\tau = \sigma^2$.

(d) Verify this formula on the constant Mackey functor $\underline{\mathbb{Z}}$: confirm that $\mathrm{res} \circ \mathrm{tr}(m) = 2m$ and that $m + \tau_*(m) = 2m$.

> [!TIP]- Solution to Exercise 4
> **Key insight:** The double coset space $C_2\backslash C_4/C_2$ has exactly two classes — $[e]$ and $[\sigma]$ — because $C_4/C_2$ has two elements; the corresponding Mackey formula contributions are $\mathrm{id}$ and $\tau_*$.
>
> **Sketch:**
>
> $C_2\backslash C_4/C_2$: the orbits of $C_2 = \{e, \sigma^2\}$ acting on $C_4/C_2 = \{eC_2, \sigma C_2\}$ by left multiplication are each singletons. So $K\backslash C_4/H = \{[e], [\sigma]\}$.
>
> For $[g] = [e]$: $K \cap {}^eH = C_2 \cap C_2 = C_2$, so the contribution is $\mathrm{tr}_{C_2}^{C_2} \circ c_e \circ \mathrm{res}_{C_2}^{C_2} = \mathrm{id}$.
>
> For $[g] = [\sigma]$: $K \cap {}^\sigma H = C_2 \cap \sigma C_2 \sigma^{-1} = C_2$ (since $C_4$ is abelian). The contribution is $\mathrm{tr}_{C_2}^{C_2} \circ c_\sigma \circ \mathrm{res}_{C_2}^{C_2} = c_{\sigma^2} = \tau_*$.
>
> Summing: $\mathrm{res}_{C_2} \circ \mathrm{tr}_{C_2}^{C_4} = \mathrm{id} + \tau_*$. For $\underline{\mathbb{Z}}$: $\tau = \mathrm{id}$, so $\mathrm{id} + \tau_* = 2\cdot\mathrm{id}$, which matches $\mathrm{res}(\mathrm{tr}(m)) = 2m$. ✓

---

#### Problem 5: The Mackey Formula for S3

*This problem applies the double coset formula to a non-abelian group, computing $\mathrm{res}_K^{S_3} \circ \mathrm{tr}_H^{S_3}$ for two non-conjugate subgroups $H = \langle (12) \rangle$ and $K = \langle (13) \rangle$ of order 2, a case where the double coset decomposition is non-trivial.*

> **Prerequisites:** cf. [[#4. The Mackey Double Coset Formula|§4 — The Mackey Double Coset Formula]]

Let $G = S_3$, $H = \langle (12) \rangle \cong C_2$, and $K = \langle (13) \rangle \cong C_2$.

(a) Compute the double coset decomposition $K\backslash G/H$. List representative elements and the corresponding double cosets $KgH$ explicitly as subsets of $S_3$.

(b) For each double coset representative $[g]$, compute $K \cap {}^gH$ and the conjugate ${}^gH = g H g^{-1}$.

(c) Write out the full Mackey formula:
$$\mathrm{res}_K^{S_3} \circ \mathrm{tr}_H^{S_3}(m) = \sum_{[g] \in K\backslash S_3/H} \mathrm{tr}_{K \cap {}^gH}^K \circ c_g \circ \mathrm{res}_{K^g \cap H}^H(m).$$

(d) Specialize to the constant Mackey functor $\underline{\mathbb{Z}}$ and verify numerically that both sides give the same value.

> [!TIP]- Solution to Exercise 5
> **Key insight:** The two subgroups $H = \langle(12)\rangle$ and $K = \langle(13)\rangle$ are non-conjugate in $S_3$ but both of order 2; the double coset decomposition $K\backslash S_3/H$ has two elements, one with trivial intersection and one where $K \cap {}^gH = K$.
>
> **Sketch:**
>
> Partitioning $S_3$ into double cosets $KgH$: by the double coset size formula $|KgH| = |K||H|/|K \cap {}^gH|$:
> - $[g] = e$: $K \cap H = \{e\}$, so $|KeH| = 4$.
> - $[g] = [(132)]$: $(132)(12)(123) = (13)$, so ${}^{(132)}H = K$. Then $K \cap K = K \cong C_2$, so $|K(132)H| = 2$.
> Total: $4 + 2 = 6 = |S_3|$. ✓ Two double cosets.
>
> Mackey formula:
> $$\mathrm{res}_K \circ \mathrm{tr}_H = (c_e \circ \mathrm{res}_{\{e\}}^H) + (\mathrm{tr}_K^K \circ c_{(132)} \circ \mathrm{res}_{K^{(132)} \cap H}^H).$$
> Here $K^{(132)} = \langle(12)\rangle = H$, so $K^{(132)} \cap H = H$, and the second term is $c_{(132)} \circ \mathrm{id}$.
>
> For $\underline{\mathbb{Z}}$: first term gives $\mathrm{id}$ (restriction from $H$ to $e$ then covariant), contributing $m$ at each of the $[K:e]$ indices... full count gives $3m = [S_3:H] \cdot m$, consistent with $\mathrm{res}_K(\mathrm{tr}_H^{S_3}(m)) = 3m$.

---

#### Problem 6: The Norm Map for Cp

*This problem establishes the general formula $\mathrm{res}_e^{C_p} \circ \mathrm{tr}_e^{C_p} = \sum_{g \in C_p} g_*$ for any prime $p$, which is the fundamental example of the Mackey formula for a cyclic group of prime order.*

> **Prerequisites:** cf. [[#4.2 Derivation from Spans|§4.2 — Derivation from Spans]]

Let $G = C_p$ for a prime $p$, $H = K = \{e\}$.

(a) List all double cosets in $\{e\}\backslash C_p/\{e\}$. Show there are exactly $p$ double cosets, one for each $g \in C_p$.

(b) For each double coset representative $g \in C_p$, compute $K \cap {}^gH = \{e\}$ and show each contributes the map $c_g: M(C_p/e) \to M(C_p/e)$.

(c) Conclude that for any $C_p$-Mackey functor $M$: $\mathrm{res}_e^{C_p} \circ \mathrm{tr}_e^{C_p}(m) = \sum_{g \in C_p} c_g(m) = \mathrm{Nm}(m)$, the norm map.

(d) For the constant Mackey functor $\underline{\mathbb{Z}}$, verify this gives $pm$. For the fixed-point Mackey functor of the regular representation $M(C_p/e) = \mathbb{Z}[C_p]$, compute the norm $\sum_{g \in C_p} g \cdot m$ explicitly for $m = e$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** When both $H$ and $K$ are trivial, every element of $C_p$ is its own double coset, so the Mackey formula is a sum of $p$ conjugation maps — by definition the norm map.
>
> **Sketch:**
>
> $\{e\}\backslash C_p/\{e\} = C_p$ (each element $g$ is the singleton double coset $\{g\}$). There are exactly $p$ double cosets.
>
> For each $g \in C_p$: $K \cap {}^gH = \{e\}$, so $\mathrm{tr}_{\{e\}}^{\{e\}} = \mathrm{id}$ and $\mathrm{res}_{\{e\}}^{\{e\}} = \mathrm{id}$. The contribution is $c_g$.
>
> Summing: $\mathrm{res}_e^{C_p} \circ \mathrm{tr}_e^{C_p}(m) = \sum_{g \in C_p} c_g(m) = \mathrm{Nm}(m)$.
>
> For $\underline{\mathbb{Z}}$: all $c_g = \mathrm{id}$, so $\mathrm{Nm}(m) = pm$. Also $\mathrm{tr}_e^{C_p}(m) = pm$ and $\mathrm{res}_e^{C_p}(pm) = pm$. ✓
>
> For $\mathbb{Z}[C_p]$: $c_g(e) = g$, so $\mathrm{Nm}(e) = \sum_{g \in C_p} g = N$ (the norm element of the group ring).

---

#### Problem 7: The C2 Mackey Functor Classification

*This problem classifies all $C_2$-Mackey functors with $M(C_2/e) = \mathbb{Z}$ by determining which pairs (res, tr) are compatible with the Mackey formula.*

> **Prerequisites:** cf. [[#5.4 The C2 Classification|§5.4 — The C2 Classification]]

Let $G = C_2$ and suppose $M$ is a $C_2$-Mackey functor with $M(C_2/e) = \mathbb{Z}$.

(a) Write $A = M(C_2/C_2)$, $r = \mathrm{res}: A \to \mathbb{Z}$, $t = \mathrm{tr}: \mathbb{Z} \to A$, and $\tau = c_\tau: \mathbb{Z} \to \mathbb{Z}$. Show $\tau$ must be an involution, so either $\tau = \mathrm{id}$ or $\tau = -\mathrm{id}$.

(b) In the case $\tau = \mathrm{id}$: show the Mackey formula forces $r \circ t = 2 \cdot \mathrm{id}_\mathbb{Z}$. Show that $A = \mathbb{Z}$, $r = \mathrm{id}$, $t = \times 2$ gives the constant Mackey functor $\underline{\mathbb{Z}}$.

(c) In the case $\tau = -\mathrm{id}$: show the Mackey formula forces $r \circ t = 0$. Show the choice $A = \mathbb{Z}$, $r = 0$, $t = 0$, $\tau = -1$ gives $\underline{\mathbb{Z}}^-$.

(d) Are there Mackey functors with $M(C_2/e) = \mathbb{Z}$ and $A = M(C_2/C_2) = \mathbb{Z}/2$? Identify any example or prove none exist.

> [!TIP]- Solution to Exercise 7
> **Key insight:** The Mackey formula $\mathrm{res} \circ \mathrm{tr} = \mathrm{id} + \tau$ is the single constraint; with $\tau \in \{\pm\mathrm{id}\}$, it forces $r \circ t = 2$ or $r \circ t = 0$, each case determining a one-parameter family of extensions.
>
> **Sketch:**
>
> Since $c_\tau: \mathbb{Z} \to \mathbb{Z}$ must satisfy $c_\tau^2 = \mathrm{id}$ and be a group automorphism of $\mathbb{Z}$, we get $c_\tau \in \{\mathrm{id}, -\mathrm{id}\}$.
>
> Case $\tau = \mathrm{id}$: Mackey gives $r \circ t = 2$. The pair $(r, t)$ is characterized by $m = t(1) \in A$ with $r(m) = 2$. The minimal choice $A = \mathbb{Z}$, $t(n) = 2n$, $r = \mathrm{id}$ gives $\underline{\mathbb{Z}}$.
>
> Case $\tau = -\mathrm{id}$: Mackey gives $r \circ t = \mathrm{id} + (-\mathrm{id}) = 0$. The choice $A = \mathbb{Z}$, $r = 0$, $t = 0$ gives $\underline{\mathbb{Z}}^-$.
>
> For $A = \mathbb{Z}/2$: $\mathrm{Hom}(\mathbb{Z}/2, \mathbb{Z}) = 0$ (since $\mathbb{Z}$ is torsion-free), so $r = 0$. The Mackey formula in the case $\tau = -\mathrm{id}$ gives $0 = 0$, which is consistent. So there do exist Mackey functors with $A = \mathbb{Z}/2$, $r = 0$, $t: \mathbb{Z} \to \mathbb{Z}/2$ any homomorphism, and $\tau = -\mathrm{id}$.

---

#### Problem 8: The Constant Mackey Functor and Bredon Cohomology

*This problem computes $H^0_{C_2}(C_2; \underline{\mathbb{Z}})$ using the Lewis diagram of $\underline{\mathbb{Z}}$, providing a concrete first example of Bredon cohomology computed directly from the Mackey functor structure.*

> **Prerequisites:** cf. [[#5.1 The Constant Mackey Functor|§5.1 — The Constant Mackey Functor]]; cf. [[#5.4 The C2 Classification|§5.4 — The C2 Classification]]

For $G = C_2$, the constant Mackey functor $\underline{\mathbb{Z}}$ has Lewis diagram $\mathbb{Z} \underset{2}{\overset{1}{\rightleftharpoons}} \mathbb{Z}$.

(a) Verify the Mackey formula for $\underline{\mathbb{Z}}$: confirm that $\mathrm{res} \circ \mathrm{tr}(m) = 2m$ agrees with $m + \tau \cdot m = 2m$.

(b) Recall that $H^0_{C_2}(C_2; M)$ is computed from the complex $0 \to M(C_2/C_2) \xrightarrow{\mathrm{res}} M(C_2/e) \to 0$. Compute $H^0_{C_2}(C_2; \underline{\mathbb{Z}}) = \ker(1 - \tau)/\mathrm{im}(\mathrm{res})$.

(c) Since $\tau = \mathrm{id}$ on $M(C_2/e) = \mathbb{Z}$, the map $1 - \tau = 0$, so $\ker(1-\tau) = \mathbb{Z}$. Compute $\mathrm{im}(\mathrm{res})$ and conclude. Explain geometrically why the 0th Bredon cohomology of a free orbit should vanish.

> [!TIP]- Solution to Exercise 8
> **Key insight:** For the free orbit $C_2/e$, the fixed-point space is empty at the group level; the $C_2$-equivariant Bredon cohomology in degree 0 sees only $C_2$-invariants, which for a free orbit are trivial.
>
> **Sketch:**
>
> Mackey formula check: $\mathrm{res}(\mathrm{tr}(m)) = \mathrm{id}(2m) = 2m$ and $m + \tau m = m + m = 2m$ (since $\tau = \mathrm{id}$ on $M(C_2/e) = \mathbb{Z}$). ✓
>
> For Bredon $H^0$: $H^0 = \ker(1 - \tau \mid_{M(C_2/e)}) / \mathrm{im}(\mathrm{res})$. With $\tau = \mathrm{id}$: $1 - \tau = 0$ so $\ker = \mathbb{Z}$; $\mathrm{im}(\mathrm{res}) = \mathrm{im}(\mathrm{id}: \mathbb{Z} \to \mathbb{Z}) = \mathbb{Z}$.
>
> $H^0_{C_2}(C_2; \underline{\mathbb{Z}}) = \mathbb{Z}/\mathbb{Z} = 0$.
>
> Geometric interpretation: Bredon $H^0_{C_2}(C_2; \underline{\mathbb{Z}})$ computes connected components of the orbit space $C_2/C_2 = *$, weighted by the Mackey structure. Since $C_2$ acts freely on itself, there are no genuine fixed points to contribute, and the single orbit $*$ has $H^0 = 0$ (vanishing due to the free action).

---

#### Problem 9: The Burnside Ring Mackey Functor for C2

*This problem computes $\underline{A}(C_2)$ explicitly as a Lewis diagram, verifies the Mackey formula for the Burnside ring Mackey functor, and confirms that $A(C_2) \cong \mathbb{Z}^2$.*

> **Prerequisites:** cf. [[#5.2 The Burnside Ring Mackey Functor|§5.2 — The Burnside Ring Mackey Functor]]

Let $G = C_2$. Recall $\underline{A}(G/H) = A(H)$ (the Burnside ring of $H$).

(a) Compute $\underline{A}(C_2/C_2) = A(C_2)$ and $\underline{A}(C_2/e) = A(e) \cong \mathbb{Z}$. Write the Lewis diagram with explicit groups.

(b) Compute the restriction map $\mathrm{res}_e^{C_2}: A(C_2) \to A(e) = \mathbb{Z}$: for each generator $[*]$ and $[C_2]$ of $A(C_2) \cong \mathbb{Z}^2$, restrict the $C_2$-action to $e$. Express $\mathrm{res}$ as a matrix.

(c) Compute the transfer map $\mathrm{tr}_e^{C_2}: A(e) \to A(C_2)$: the transfer of an $e$-set $S$ is $C_2 \times_e S \cong C_2 \times S$. Compute $\mathrm{tr}([*]) = [C_2]$ and write the Lewis diagram.

(d) Verify the Mackey formula: $\mathrm{res} \circ \mathrm{tr}(n) = 2n$ and $n + \tau_*(n) = 2n$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** The Burnside ring of $C_2$ is $A(C_2) = \mathbb{Z}\{[*],[C_2]\} \cong \mathbb{Z}^2$, and the restriction/transfer are precisely the "forget action" and "induce" operations on $0$-dimensional $C_2$-sets.
>
> **Sketch:**
>
> $\underline{A}(C_2/C_2) = A(C_2) \cong \mathbb{Z}^2$ (generators $[*]$ and $[C_2]$). $\underline{A}(C_2/e) = A(e) \cong \mathbb{Z}$.
>
> Restriction $\mathrm{res}_e^{C_2}: A(C_2) \to A(e)$: forget the $C_2$-action. $\mathrm{res}([*]) = 1$ and $\mathrm{res}([C_2]) = 2$ (two isolated points). So $\mathrm{res}(a, b) = a + 2b$ as a map $\mathbb{Z}^2 \to \mathbb{Z}$.
>
> Transfer $\mathrm{tr}_e^{C_2}: A(e) \to A(C_2)$: $\mathrm{tr}(1) = [C_2]$. So $\mathrm{tr}(n) = n[C_2]$, i.e., $\mathrm{tr} = (0, 1)^T: \mathbb{Z} \to \mathbb{Z}^2$.
>
> Lewis diagram: $\mathbb{Z}^2 \underset{(0,1)^T}{\overset{(1,2)}{\rightleftharpoons}} \mathbb{Z}$.
>
> Mackey check: $\mathrm{res} \circ \mathrm{tr}(n) = (1,2)(0,n)^T = 2n$ and $n + \tau_*(n) = n + n = 2n$. ✓

---

#### Problem 10: The Mackey Formula from Fiber Products

*This problem re-derives the Mackey double coset formula from the explicit fiber product decomposition in $\mathbf{FSets}_G$, following the derivation in the note but working out all intermediate steps for $G = C_4$, $H = e$, $K = C_2$.*

> **Prerequisites:** cf. [[#4.1 The Pullback Decomposition|§4.1 — The Pullback Decomposition]]; cf. [[#4.2 Derivation from Spans|§4.2 — Derivation from Spans]]

Let $G = C_4 = \langle \sigma \rangle$, $H = \{e\}$, $K = C_2 = \langle \sigma^2 \rangle$.

(a) Write the two spans corresponding to $\mathrm{tr}_e^{C_4}$ and $\mathrm{res}_{C_2}^{C_4}$.

(b) Compute the fiber product $G/e \times_{G/G} G/K = C_4 \times_* (C_4/C_2)$ as a $C_4$-set. Decompose into orbits under the diagonal $C_4$-action.

(c) Identify each orbit in the decomposition with a coset space $G/(K \cap {}^gH)$ for appropriate $g$. Write the full orbit decomposition and read off the Mackey formula.

(d) Verify the result matches the double coset formula: $\mathrm{res}_{C_2}^{C_4} \circ \mathrm{tr}_e^{C_4}(m) = \sum_{g \in C_2\backslash C_4/\{e\}} c_g(m)$. How many terms appear?

> [!TIP]- Solution to Exercise 10
> **Key insight:** The fiber product $C_4 \times_* (C_4/C_2)$ is just the Cartesian product (since the base is a point), and the diagonal $C_4$-orbits are indexed by double cosets in $C_2\backslash C_4/\{e\} = C_2\backslash C_4$.
>
> **Sketch:**
>
> The span for $\mathrm{tr}_e^{C_4}$ is $C_4 \xleftarrow{\mathrm{id}} C_4 \xrightarrow{p} *$; for $\mathrm{res}_{C_2}^{C_4}$ it is $* \xleftarrow{q} C_4/C_2 \xrightarrow{\mathrm{id}} C_4/C_2$.
>
> Fiber product: $C_4 \times_* (C_4/C_2) = C_4 \times C_4/C_2$ with diagonal $C_4$-action.
>
> Orbits: for each $\sigma^i C_2 \in C_4/C_2$, the orbit of $(e, \sigma^i C_2)$ has stabilizer $\{e\} \cap \sigma^i C_2 \sigma^{-i} = \{e\} \cap C_2 = \{e\}$. So each orbit is $\cong C_4/\{e\} = C_4$.
>
> There are $|C_4/C_2| = 2$ orbits, each $\cong C_4$. Reading off: $\mathrm{res}_{C_2}^{C_4} \circ \mathrm{tr}_e^{C_4}(m) = c_e(m) + c_\sigma(m)$, a sum of 2 terms. ✓

---

#### Problem 11: The Yoneda Lemma for Mackey Functors

*This problem proves the Yoneda isomorphism $\mathrm{Hom}_{\mathrm{Mack}(G)}(\mathbb{Z}[G/H,-], M) \cong M(G/H)$ in detail, establishing that representable Mackey functors are the fundamental projective objects.*

> **Prerequisites:** cf. [[#7.1 Representable Mackey Functors|§7.1 — Representable Mackey Functors]]

Recall the representable Mackey functor $\mathbb{Z}[G/H,-] = \mathcal{A}(G)(G/H,-)$.

(a) For a natural transformation $\phi: \mathbb{Z}[G/H,-] \to M$, let $m = \phi_{G/H}(\mathrm{id}_{G/H}) \in M(G/H)$. Show that $\phi_{G/K}(\alpha) = M(\alpha)(m)$ for every $\alpha \in \mathcal{A}(G)(G/H, G/K)$ and every orbit $G/K$.

(b) Deduce that the assignment $\phi \mapsto \phi_{G/H}(\mathrm{id}_{G/H})$ defines a bijection $\mathrm{Hom}_{\mathrm{Mack}(G)}(\mathbb{Z}[G/H,-], M) \xrightarrow{\sim} M(G/H)$, natural in both $M$ and $G/H$.

(c) Conclude that $\mathbb{Z}[G/H,-]$ is projective: use the Yoneda isomorphism to show that $\mathrm{Hom}_{\mathrm{Mack}(G)}(\mathbb{Z}[G/H,-], -)$ is exact.

(d) Compute $\mathrm{Hom}_{\mathrm{Mack}(G)}(\mathbb{Z}[G/G,-], M) \cong M(G/G)$ and identify $\mathbb{Z}[G/G,-]$ with the Burnside ring Mackey functor $\underline{A}$.

> [!TIP]- Solution to Exercise 11
> **Key insight:** The Yoneda lemma for additive categories applies verbatim: a natural transformation out of a representable is determined by the image of the identity morphism.
>
> **Sketch:**
>
> Given $\phi: \mathbb{Z}[G/H,-] \to M$ and $\alpha \in \mathcal{A}(G)(G/H, G/K) = \mathbb{Z}[G/H,-](G/K)$, naturality gives:
> $$\phi_{G/K}(\alpha) = \phi_{G/K}(\mathcal{A}(G)(G/H,-)(\alpha)(\mathrm{id}_{G/H})) = M(\alpha)(\phi_{G/H}(\mathrm{id}_{G/H})).$$
> So $\phi$ is determined by $m = \phi_{G/H}(\mathrm{id}_{G/H}) \in M(G/H)$; conversely any $m \in M(G/H)$ defines a valid $\phi$ by this formula.
>
> Projectivity: $\mathrm{Hom}(\mathbb{Z}[G/H,-], -)$ is naturally isomorphic to evaluation at $G/H$. Evaluation is exact because limits/colimits in $\mathrm{Mack}(G)$ are computed pointwise.
>
> For $H = G$: $\mathbb{Z}[G/G,-](G/K) = \mathcal{A}(G)(G/G, G/K) = A(K) = \underline{A}(G/K)$, so $\mathbb{Z}[G/G,-] = \underline{A}$.

---

#### Problem 12: Representable Mackey Functors and Double Cosets

*This problem makes the formula $\mathbb{Z}[G/H,-](G/K) \cong \bigoplus_{[x] \in H\backslash G/K} \mathbb{Z}$ explicit for the case $G = S_3$, $H = \langle(12)\rangle$, computing the representable Mackey functor at each orbit.*

> **Prerequisites:** cf. [[#7.1 Representable Mackey Functors|§7.1 — Representable Mackey Functors]]; requires Problem 11

Let $G = S_3$, $H = \langle (12) \rangle \cong C_2$. Consider the representable Mackey functor $P = \mathbb{Z}[S_3/H, -]$.

(a) Compute $P(S_3/S_3)$. Use the double coset formula $\mathbb{Z}[G/H,-](G/K) \cong \bigoplus_{[x]\in H\backslash G/K} \mathbb{Z}$ with $K = S_3$.

(b) Compute $P(S_3/e)$. Use the double coset formula with $K = \{e\}$.

(c) Compute $P(S_3/H') $ where $H' = \langle (123) \rangle \cong C_3$.

(d) Write the resulting Mackey functor as a diagram with groups at the three orbits $S_3/S_3$, $S_3/H$, $S_3/H'$, $S_3/e$. Identify the restriction and transfer maps.

> [!TIP]- Solution to Exercise 12
> **Key insight:** The formula $\mathbb{Z}[G/H,-](G/K) \cong \bigoplus_{H\backslash G/K} \mathbb{Z}$ says the representable evaluated at $G/K$ is the free abelian group on double cosets; counting double cosets in each case gives the rank.
>
> **Sketch:**
>
> $P(S_3/S_3)$: double cosets $H\backslash S_3/S_3 = H\backslash S_3$ — a single double coset (since $K = S_3$ is the whole group). Rank 1: $P(S_3/S_3) \cong \mathbb{Z}$.
>
> $P(S_3/e)$: double cosets $H\backslash S_3/\{e\} = H\backslash S_3$ (left cosets of $H$) — $|S_3|/|H| = 3$ elements. So $P(S_3/e) \cong \mathbb{Z}^3$.
>
> $P(S_3/H')$ with $H' = \langle(123)\rangle$: $H\backslash S_3/H'$. Since $H \cap H' = \{e\}$, all double cosets have size $|H||H'|/1 = 6 = |S_3|$, so there is only one double coset. Rank 1: $P(S_3/H') \cong \mathbb{Z}$.
>
> $P(S_3/H)$: double cosets $H\backslash S_3/H$ — 2 double cosets ($[e]$ of size 2 and $[(132)]$ of size 4). Rank 2: $P(S_3/H) \cong \mathbb{Z}^2$.

---

#### Problem 13: Projective Resolution of the Constant Functor for C2

*This problem constructs an explicit length-2 projective resolution of $\underline{\mathbb{Z}}$ as a $C_2$-Mackey functor using the representable projectives $\mathbb{Z}[C_2/H,-]$, making the global dimension bound concrete.*

> **Prerequisites:** cf. [[#7.2 Global Dimension and Resolutions|§7.2 — Global Dimension and Resolutions]]; requires Problems 9 and 11

Let $G = C_2$. Write $P_0 = \mathbb{Z}[C_2/C_2, -]$ and $P_1 = \mathbb{Z}[C_2/e, -]$ for the two indecomposable projectives.

(a) Compute the Lewis diagrams of $P_0$ and $P_1$ using the formula $\mathbb{Z}[G/H,-](G/K) \cong \bigoplus_{H\backslash G/K} \mathbb{Z}$.

(b) Define a surjection $\epsilon: P_0 \to \underline{\mathbb{Z}}$ using the Yoneda lemma (maps $P_0 \to \underline{\mathbb{Z}}$ are in bijection with $\underline{\mathbb{Z}}(C_2/C_2) = \mathbb{Z}$). Pick the generator $1 \in \mathbb{Z}$ and write $\epsilon$ explicitly.

(c) Compute the kernel $K_0 = \ker(\epsilon: P_0 \to \underline{\mathbb{Z}})$ as a Lewis diagram.

(d) Show $K_0$ is projective by finding an isomorphism $K_0 \cong P_1^{\oplus n}$ for some $n$. Write the resulting projective resolution $0 \to K_0 \to P_0 \to \underline{\mathbb{Z}} \to 0$.

> [!TIP]- Solution to Exercise 13
> **Key insight:** The two representable projectives for $C_2$ have Lewis diagrams determined by the double coset formula; the kernel of the augmentation $P_0 \to \underline{\mathbb{Z}}$ is isomorphic to a shift of $P_1$, giving a length-1 resolution.
>
> **Sketch:**
>
> $P_0 = \mathbb{Z}[C_2/C_2,-]$: $P_0(*) = \mathcal{A}(C_2)(*,*) \cong \mathbb{Z}^2$, $P_0(C_2/e) = \mathcal{A}(C_2)(*,C_2) \cong \mathbb{Z}$.
>
> $P_1 = \mathbb{Z}[C_2/e,-]$: $P_1(*) = \mathcal{A}(C_2)(C_2,*) \cong \mathbb{Z}$, $P_1(C_2/e) = \mathcal{A}(C_2)(C_2,C_2) \cong \mathbb{Z}^2$.
>
> Augmentation $\epsilon: P_0 \to \underline{\mathbb{Z}}$: the generator $1 \in \mathbb{Z}$ defines $\epsilon_{*}: \mathbb{Z}^2 \to \mathbb{Z}$ by $(a,b) \mapsto a$ (mapping $[*] \mapsto 1$, $[C_2] \mapsto 0$) and $\epsilon_{C_2/e}: \mathbb{Z} \to \mathbb{Z}$ by $n \mapsto n$.
>
> $K_0 = \ker(\epsilon)$: at level $*$, $\ker(\epsilon_{*}) = \mathbb{Z} \cdot [C_2] \cong \mathbb{Z}$; at level $C_2/e$, $\ker(\epsilon_{C_2/e}) = 0$. The Mackey functor $K_0$ with $K_0(*) = \mathbb{Z}$, $K_0(C_2/e) = 0$ is projective, giving a length-1 resolution $0 \to K_0 \to P_0 \xrightarrow{\epsilon} \underline{\mathbb{Z}} \to 0$.

---

#### Problem 14: Box Product Computation for C2

*This problem computes $\underline{\mathbb{Z}} \square \underline{\mathbb{Z}}$ for $G = C_2$ using the coend formula, establishing that the box product of the constant functor with itself is again $\underline{\mathbb{Z}}$.*

> **Prerequisites:** cf. [[#6.1 Day Convolution and the Box Product|§6.1 — Day Convolution and the Box Product]]

Let $G = C_2$ and $M = N = \underline{\mathbb{Z}}$.

(a) Recall the coend formula $(M \square N)(G/H) = \int^{X,Y} \mathcal{A}(G)(X \times Y, G/H) \otimes M(X) \otimes N(Y)$. For $H = C_2$, show that the coend reduces to a sum over pairs $(X, Y) \in \{*, C_2\}^2$ and compute each summand.

(b) Apply the Frobenius reciprocity relation in the box product. For $M = N = \underline{\mathbb{Z}}$, write out this relation explicitly at level $H = C_2$.

(c) Show that $(\underline{\mathbb{Z}} \square \underline{\mathbb{Z}})(C_2/C_2) \cong \mathbb{Z}$ and $(\underline{\mathbb{Z}} \square \underline{\mathbb{Z}})(C_2/e) \cong \mathbb{Z}$.

(d) Identify the restriction and transfer maps and conclude $\underline{\mathbb{Z}} \square \underline{\mathbb{Z}} \cong \underline{\mathbb{Z}}$.

> [!TIP]- Solution to Exercise 14
> **Key insight:** The Day convolution coend for $\underline{\mathbb{Z}} \square \underline{\mathbb{Z}}$ reduces at each level to a tensor product modulo Frobenius relations, and those relations force $(\underline{\mathbb{Z}} \square \underline{\mathbb{Z}})(G/H) \cong \mathbb{Z}$ for both $H$.
>
> **Sketch:**
>
> At level $* = C_2/C_2$: contributions from $(X,Y) \in \{(*,*), (*,C_2), (C_2,*), (C_2,C_2)\}$ — all pairs contribute since all $C_2$-sets map to $*$.
>
> - $(*, *)$: $\mathcal{A}(*,*) \otimes \mathbb{Z} \otimes \mathbb{Z} = \mathbb{Z}^2$.
> - $(C_2, *)$: $\mathcal{A}(C_2 \times *, *) \otimes \mathbb{Z} \otimes \mathbb{Z} = \mathbb{Z}$.
>
> The Frobenius relation $\mathrm{tr}(m) \otimes_* n = \mathrm{tr}(m \cdot n)$ imposes $2m \otimes n = 2mn$, automatically satisfied. After imposing all coend relations, the result at $*$ is $\mathbb{Z}$.
>
> At level $C_2/e$: similarly gives $\mathbb{Z}$.
>
> The restriction and transfer agree with those of $\underline{\mathbb{Z}}$ (res $= \mathrm{id}$, tr $= \times 2$), so $\underline{\mathbb{Z}} \square \underline{\mathbb{Z}} \cong \underline{\mathbb{Z}}$.

---

#### Problem 15: The Burnside Ring Mackey Functor is the Unit for Box Product

*This problem proves the unit isomorphism $\underline{A} \square M \cong M$ for any $C_2$-Mackey functor $M$, establishing that $\underline{A}$ plays the role of $\mathbb{Z}$ in the monoidal structure.*

> **Prerequisites:** cf. [[#6.1 Day Convolution and the Box Product|§6.1 — Day Convolution and the Box Product]]; cf. [[#5.2 The Burnside Ring Mackey Functor|§5.2 — The Burnside Ring Mackey Functor]]; requires Problem 11

(a) Use the Yoneda lemma (Problem 11) to show $\mathbb{Z}[G/G,-] \square M \cong M$ for any Mackey functor $M$.

(b) Use the identification $\underline{A} = \mathbb{Z}[G/G,-]$ to conclude $\underline{A} \square M \cong M$.

(c) Verify the isomorphism explicitly for $G = C_2$ and $M = \underline{\mathbb{Z}}$: use the Lewis diagrams of $\underline{A}$ (from Problem 9) and $\underline{\mathbb{Z}}$ to compute $(\underline{A} \square \underline{\mathbb{Z}})(C_2/C_2)$ and $(\underline{A} \square \underline{\mathbb{Z}})(C_2/e)$ from the coend formula, and check they agree with $\underline{\mathbb{Z}}$.

> [!TIP]- Solution to Exercise 15
> **Key insight:** Since $\underline{A} = \mathbb{Z}[G/G,-]$ is the representable at the terminal object, Day convolution with a representable recovers evaluation by the enriched Yoneda lemma.
>
> **Sketch:**
>
> For any presheaf $M$ and representable $\mathbb{Z}[X,-]$, the coend formula and Yoneda lemma give:
> $$(\mathbb{Z}[X,-] \square M)(Y) \cong \int^B \mathcal{A}(X \times B, Y) \otimes M(B).$$
> With $X = G/G$ (so $G/G \times B = B$): $\int^B \mathcal{A}(B, Y) \otimes M(B) \cong M(Y)$ by the enriched Yoneda lemma.
>
> Explicit check for $G = C_2$, $M = \underline{\mathbb{Z}}$: $(\underline{A} \square \underline{\mathbb{Z}})(*) = A(C_2) \otimes_{A(C_2)} \mathbb{Z} \cong \mathbb{Z}$ and at $C_2/e$: $A(e) \otimes_{A(e)} \mathbb{Z} \cong \mathbb{Z}$. ✓

---

#### Problem 16: Frobenius Reciprocity and the Green Functor Structure on A(G)

*This problem verifies Frobenius reciprocity $\mathrm{tr}(a) \cdot b = \mathrm{tr}(a \cdot \mathrm{res}(b))$ for the Burnside ring Mackey functor $\underline{A}$, using the concrete description in terms of induced $G$-sets.*

> **Prerequisites:** cf. [[#6.2 Green Functors|§6.2 — Green Functors]]

For $G = C_2$, $H = \{e\}$, $K = C_2$.

(a) Write out the restriction $\mathrm{res}_e^{C_2}: A(C_2) \to A(e) \cong \mathbb{Z}$ and transfer $\mathrm{tr}_e^{C_2}: A(e) \to A(C_2)$ explicitly on generators $[*]$ and $[C_2]$.

(b) For $a = [*] \in A(e) = \mathbb{Z}$ and $b = [C_2] \in A(C_2)$, compute both sides of $\mathrm{tr}(a) \cdot b = \mathrm{tr}(a \cdot \mathrm{res}(b))$.

(c) Prove Frobenius reciprocity for $\underline{A}$ in general: show that $K \times_H (S \times \mathrm{res}_H^K(T)) \cong (K \times_H S) \times T$ as $K$-sets. Define the isomorphism explicitly and check $K$-equivariance.

(d) Verify the same identity for the constant Mackey functor $\underline{\mathbb{Z}}$.

> [!TIP]- Solution to Exercise 16
> **Key insight:** Frobenius reciprocity $K \times_H (S \times \mathrm{res}(T)) \cong (K \times_H S) \times T$ is the set-level distributivity of Cartesian product over induced sets; the isomorphism $(k, s, t) \mapsto ([k,s], kt)$ is the explicit $K$-equivariant bijection.
>
> **Sketch:**
>
> For $G = C_2$, $H = \{e\}$, $K = C_2$, $a = [*] \in A(\{e\})$, $b = [C_2] \in A(C_2)$:
>
> LHS: $\mathrm{tr}([*]) \cdot [C_2] = [C_2] \cdot [C_2] = [C_2 \times C_2] = 2[C_2]$ (in $A(C_2)$).
> RHS: $\mathrm{tr}([*] \cdot \mathrm{res}([C_2])) = \mathrm{tr}([*] \cdot 2) = \mathrm{tr}(2[*]) = 2[C_2]$. ✓
>
> General proof: define $\phi: K \times_H (S \times T|_H) \to (K \times_H S) \times T$ by $\phi([k, (s,t)]) = ([k,s], kt)$. Check $K$-equivariance: $k' \cdot \phi([k,(s,t)]) = ([k'k,s], k't)$ and $\phi(k' \cdot [k,(s,t)]) = \phi([k'k,(s,t)]) = ([k'k,s],k't)$. ✓
>
> For $\underline{\mathbb{Z}}$: $\mathrm{tr}(a) \cdot b = [K:H] \cdot a \cdot b$ and $\mathrm{tr}(a \cdot \mathrm{res}(b)) = [K:H] \cdot (a \cdot b)$. Equal. ✓

---

#### Problem 17: Green Functor Structures on the Constant Mackey Functor

*This problem classifies all Green functor structures on the constant $C_2$-Mackey functor $\underline{\mathbb{Z}}$, determining which ring structures are compatible with the given restriction and transfer.*

> **Prerequisites:** cf. [[#6.2 Green Functors|§6.2 — Green Functors]]; requires Problem 7

(a) For $G = C_2$: show any Green functor structure on $\underline{\mathbb{Z}}$ must make $\mathrm{res} = \mathrm{id}: \mathbb{Z} \to \mathbb{Z}$ a ring homomorphism, forcing the unique standard ring structure on $\mathbb{Z}$ at both levels.

(b) Verify Frobenius reciprocity for $\underline{\mathbb{Z}}$: $\mathrm{tr}(a) \cdot b = \mathrm{tr}(a \cdot \mathrm{res}(b))$ becomes $2a \cdot b = 2(ab)$.

(c) Show that $\underline{\mathbb{Z}}^-$ (with $M(C_2/e) = \mathbb{Z}$, $\tau = -\mathrm{id}$, $\mathrm{res} = 0$, $\mathrm{tr} = 0$) admits no Green functor structure.

(d) Show there is exactly one Green functor structure on $\underline{\mathbb{Z}}$ (the standard one).

> [!TIP]- Solution to Exercise 17
> **Key insight:** The restriction map of $\underline{\mathbb{Z}}$ is the identity $\mathbb{Z} \to \mathbb{Z}$, which forces the ring structure on both levels to coincide; Frobenius then holds automatically from the commutativity of $\mathbb{Z}$, giving a unique Green functor structure.
>
> **Sketch:**
>
> A Green functor structure requires $\mathrm{res} = \mathrm{id}: \mathbb{Z} \to \mathbb{Z}$ to be a ring map. Since $\mathrm{id}$ is a ring isomorphism for any ring structure, both levels must carry the same ring structure. The only unital commutative ring structure on $\mathbb{Z}$ is the standard one.
>
> Frobenius: $\mathrm{tr}(a) \cdot b = 2a \cdot b = 2(ab) = \mathrm{tr}(ab) = \mathrm{tr}(a \cdot \mathrm{res}(b))$. ✓
>
> For $\underline{\mathbb{Z}}^-$: $\mathrm{res} = 0: \mathbb{Z} \to \mathbb{Z}$ must be a ring map, requiring $0 = 0(1) = 1$, so $0 = 1$ in $\mathbb{Z}$. Contradiction. So $\underline{\mathbb{Z}}^-$ admits no Green functor structure.

---

#### Problem 18: Fixed-Point Mackey Functor Axiom Verification

*This problem verifies that the fixed-point assignment $G/H \mapsto \pi_n(X^H)$ for a $G$-CW complex $X$ satisfies the Mackey functor axioms, in particular the Mackey formula, assuming transfer maps are defined via equivariant transfer.*

> **Prerequisites:** cf. [[#5.3 Fixed-Point Mackey Functors|§5.3 — Fixed-Point Mackey Functors]]; cf. [[#3.3 Axiomatic Presentation|§3.3 — Axiomatic Presentation]]

Let $G = C_2$ and $X = EC_2$ (the free contractible $C_2$-space).

(a) Compute $\pi_0(X^{C_2})$ and $\pi_0(X^e) = \pi_0(EC_2)$.

(b) Now let $X = S^1$ with $C_2$ acting by the antipodal map $z \mapsto -z$. Compute $\underline{\pi}_0(S^1)$ as a partial Lewis diagram.

(c) Let $X = S^1$ with $C_2$ acting by complex conjugation $z \mapsto \bar{z}$. Compute both levels of $\underline{\pi}_0(S^1)$ and write the Lewis diagram.

(d) For $X = C_2 \times Y$ (the free $C_2$-space on $Y$): determine both fixed-point spaces and the restriction map for $Y$ path-connected.

> [!TIP]- Solution to Exercise 18
> **Key insight:** The two $C_2$-actions on $S^1$ — antipodal versus conjugation — give fixed-point sets $\emptyset$ and $S^0$ respectively, producing Mackey functors with different structures at the two levels.
>
> **Sketch:**
>
> $X = EC_2$: $EC_2^{C_2} = \emptyset$ (free action), $\pi_0(EC_2^e) = \pi_0(EC_2) = 0$. So $\underline{\pi}_0(EC_2)$ is the zero Mackey functor.
>
> $X = S^1$ with antipodal action: $(S^1)^{C_2} = \emptyset$, $(S^1)^e = S^1$ (connected). So $\underline{\pi}_0(S^1)(C_2/e) = 0$ and the top level is undefined/zero.
>
> $X = S^1$ with conjugation action: $(S^1)^{C_2} = \{+1,-1\} = S^0$ (two fixed points), $(S^1)^e = S^1$. So $\underline{\pi}_0(S^1)(C_2/C_2) = \pi_0(S^0) = \mathbb{Z}/2$ and $\underline{\pi}_0(S^1)(C_2/e) = \pi_0(S^1) = 0$. Lewis diagram: $\mathbb{Z}/2 \rightleftharpoons 0$.
>
> $X = C_2 \times Y$ (free $C_2$-space): $(C_2 \times Y)^{C_2} = \emptyset$ and $(C_2 \times Y)^e = C_2 \times Y$ (two disjoint copies of $Y$). For $Y$ path-connected: $\underline{\pi}_n(C_2 \times Y)(C_2/e) \cong \pi_n(Y) \oplus \pi_n(Y)$ and the top level is zero.

---

### Algorithmic Applications

#### Problem 19: Burnside Ring Multiplication Table

*This problem asks for a Python implementation that computes the Burnside ring multiplication table $[G/H] \cdot [G/K] = \sum_L n_L [G/L]$, where $n_L$ counts orbits of $L$ on $G/H \times G/K$.*

> **Prerequisites:** cf. [[#2.3 Additive Structure and the Burnside Ring|§2.3 — Additive Structure and the Burnside Ring]]

(a) **Inputs and data structures**: Describe the input to the algorithm: a finite group $G$ given by its multiplication table, a list of conjugacy classes of subgroups $H_0 = G, H_1, \ldots, H_k = \{e\}$, and for each $H_i$ its index $[G:H_i]$ and the list of left cosets $G/H_i$. Specify how to represent a $G$-set as a Python list with a group action function.

(b) **Counting fixed points**: Write a Python function `count_orbits(G_set, G_action, subgroup)` that, given a finite $G$-set and a subgroup $L$, counts the number of $L$-orbits using Burnside's lemma $|X / L| = \frac{1}{|L|} \sum_{l \in L} |X^l|$.

(c) **Multiplication table**: Write pseudocode for `burnside_product(H_idx, K_idx, subgroup_list, G)` that forms $G/H \times G/K$, applies the diagonal action, and returns the coefficient vector $(n_L)_L$.

(d) **Verification for C2**: Run the algorithm on $G = C_2$ and verify $[C_2]^2 = 2[C_2]$, $[*]^2 = [*]$, $[*] \cdot [C_2] = [C_2]$.

> [!TIP]- Solution to Exercise 19
> **Key insight:** The Burnside ring product $[G/H] \cdot [G/K] = [G/H \times G/K]$ is computed by decomposing the $G$-set $G/H \times G/K$ into orbits via Burnside's lemma applied to the diagonal action.
>
> **Sketch:**
>
> ```python
> from itertools import product
> from collections import defaultdict
>
> def count_fixed_points(g_set, g_action, group_element):
>     return sum(1 for x in g_set if g_action(group_element, x) == x)
>
> def count_orbits_burnside(g_set, g_action, subgroup):
>     total_fixed = sum(count_fixed_points(g_set, g_action, l) for l in subgroup)
>     return total_fixed // len(subgroup)
>
> def burnside_product(H_idx, K_idx, subgroup_list, coset_lists, g_action_on_coset):
>     GH = coset_lists[H_idx]
>     GK = coset_lists[K_idx]
>     product_set = list(product(GH, GK))
>
>     def diagonal_action(g, pair):
>         x, y = pair
>         return (g_action_on_coset(g, H_idx, x),
>                 g_action_on_coset(g, K_idx, y))
>
>     coefficients = defaultdict(int)
>     remaining = list(product_set)
>     G_elements = ...  # full group
>     while remaining:
>         x0 = remaining[0]
>         orbit = {diagonal_action(g, x0) for g in G_elements}
>         stab_size = len(G_elements) // len(orbit)
>         for L_idx, L in enumerate(subgroup_list):
>             if len(L) == stab_size:
>                 coefficients[L_idx] += 1
>                 break
>         for x in orbit:
>             remaining.remove(x)
>     return coefficients
>
> # Verification for C2:
> # [C2]^2 = [C2 x C2]: orbits {(e,e),(tau,tau)} and {(e,tau),(tau,e)}
> # each isomorphic to C2 => [C2]^2 = 2*[C2]  checkmark
> ```

---

#### Problem 20: Mackey Functor Axiom Checker from Lewis Diagram

*This problem asks for a Python class that takes a $C_2$-Mackey functor specified as a Lewis diagram and verifies all four axioms: the Mackey formula, transitivity, conjugation, and the trace formula.*

> **Prerequisites:** cf. [[#3.3 Axiomatic Presentation|§3.3 — Axiomatic Presentation]]; cf. [[#5.4 The C2 Classification|§5.4 — The C2 Classification]]

(a) **Data representation**: Define a Python dataclass `C2MackeyFunctor` with fields for `rank0`, `rank1`, `res` (matrix), `tr` (matrix), and `tau` (matrix for the $C_2$-action on $M_1$).

(b) **Mackey formula check**: Write a method `check_mackey(self) -> bool` that verifies $\mathrm{res} \circ \mathrm{tr} = \mathrm{id} + \tau$ as a matrix equation.

(c) **Involution check**: Write `check_tau_involution(self) -> bool` (verifies $\tau^2 = \mathrm{id}$).

(d) **Example**: Instantiate the checker for each row of the table in §5.4 ($\underline{\mathbb{Z}}$, $\underline{\mathbb{Z}}^-$, $\underline{A}(C_2)$, $\mathbb{Z}/2$) and show `check_mackey()` returns `True` for each. Show one example where the check fails.

> [!TIP]- Solution to Exercise 20
> **Key insight:** For $C_2$, all Mackey axioms reduce to a handful of matrix equations; checking them is a finite linear algebra computation over $\mathbb{Z}$.
>
> **Sketch:**
>
> ```python
> import numpy as np
> from dataclasses import dataclass
>
> @dataclass
> class C2MackeyFunctor:
>     rank0: int
>     rank1: int
>     res: np.ndarray   # shape (rank1, rank0)
>     tr:  np.ndarray   # shape (rank0, rank1)
>     tau: np.ndarray   # shape (rank1, rank1)
>
>     def check_mackey(self) -> bool:
>         lhs = self.res @ self.tr
>         rhs = np.eye(self.rank1, dtype=int) + self.tau
>         return np.array_equal(lhs, rhs)
>
>     def check_tau_involution(self) -> bool:
>         return np.array_equal(self.tau @ self.tau,
>                                np.eye(self.rank1, dtype=int))
>
> # Underline Z: res=[[1]], tr=[[2]], tau=[[1]]
> # check_mackey: [[1]]@[[2]] = [[2]]; id+tau = [[2]]. True.
>
> # Underline Z^-: res=[[0]], tr=[[0]], tau=[[-1]]
> # check_mackey: [[0]]@[[0]] = [[0]]; id+tau = [[0]]. True.
>
> # Failing example: res=[[3]], tr=[[1]], tau=[[1]]
> # lhs = [[3]], rhs = [[2]]. False.
> ```

---

#### Problem 21: Projective Resolution Algorithm for C2-Mackey Functors

*This problem asks for pseudocode implementing a step-by-step projective resolution of a $C_2$-Mackey functor, using the two representable projectives $P_0 = \mathbb{Z}[C_2/C_2,-]$ and $P_1 = \mathbb{Z}[C_2/e,-]$.*

> **Prerequisites:** cf. [[#7.2 Global Dimension and Resolutions|§7.2 — Global Dimension and Resolutions]]; requires Problems 13 and 20

(a) **Surjection step**: Write pseudocode for `surject_projective(M)` that reads off $\mathrm{rank}(M_0)$ and $\mathrm{rank}(M_1)$, returns $P = P_0^{m_0} \oplus P_1^{m_1}$, and the surjection $\epsilon: P \to M$.

(b) **Kernel computation**: Write pseudocode for `compute_kernel(eps, P, M)` that computes $\ker(\epsilon)$ as a Lewis diagram using Smith normal form to compute integral kernels at each level.

(c) **Full resolution**: Write a loop `resolve(M, max_length=4)` that iterates $M \leftarrow \ker(\epsilon)$ at each step, stopping when $M = 0$.

(d) **Termination**: Argue why the algorithm terminates in at most 2 steps for any $C_2$-Mackey functor $M$ with finitely generated values.

> [!TIP]- Solution to Exercise 21
> **Key insight:** Because $\mathrm{Mack}(C_2)$ has global dimension 1 (for finitely generated torsion-free Mackey functors), the kernel of any surjection from a projective is itself projective, terminating the resolution after at most 2 steps.
>
> **Sketch:**
>
> ```python
> def surject_projective(M):
>     r0, r1 = M.rank0, M.rank1
>     eps_top = np.eye(r0, dtype=int)
>     eps_bot = np.eye(r1, dtype=int)
>     P = build_direct_sum_projective(r0, r1)
>     return P, (eps_top, eps_bot)
>
> def compute_kernel(eps_top, eps_bot, P, M):
>     from sympy import Matrix
>     E0 = Matrix(eps_top)
>     K0_basis = E0.nullspace()
>     rank_K0 = len(K0_basis)
>     E1 = Matrix(eps_bot)
>     K1_basis = E1.nullspace()
>     rank_K1 = len(K1_basis)
>     # Restrict res and tr to kernel submodules
>     return C2MackeyFunctor(rank_K0, rank_K1, ...)
>
> def resolve(M, max_length=4):
>     resolution = []
>     current = M
>     for step in range(max_length):
>         if current.rank0 == 0 and current.rank1 == 0:
>             break
>         P, eps = surject_projective(current)
>         resolution.append((P, eps))
>         current = compute_kernel(*eps, P, current)
>     return resolution
>
> # Termination: after step 0, kernel K0 is always projective for finitely
> # generated torsion-free C2-Mackey functors (global dim <= 1), so step 1
> # gives K0 = 0. Torsion at one level can extend resolution by 1 step.
> ```

---

#### Problem 22: Box Product Algorithm for Cp

*This problem asks for pseudocode computing the box product $M \square N$ for $G = C_p$ from Lewis diagrams, implementing the coend formula and imposing the Frobenius relations.*

> **Prerequisites:** cf. [[#6.1 Day Convolution and the Box Product|§6.1 — Day Convolution and the Box Product]]

For $G = C_p$, a Mackey functor is given by groups $M_0 = M(C_p/C_p)$, $M_1 = M(C_p/e)$, and maps $r: M_0 \to M_1$, $t: M_1 \to M_0$, $\tau: M_1 \to M_1$.

(a) **Coend at level $C_p/C_p$**: Show the coend formula reduces to $(M \square N)_0 = (M_0 \otimes N_0) \oplus (M_1 \otimes_{C_p} N_1)$. Write pseudocode to compute this from the integer matrices.

(b) **Coend at level $C_p/e$**: Show $(M \square N)_1 = M_1 \otimes N_1$ with $\tau_{M\square N} = \tau_M \otimes \tau_N$.

(c) **Restriction and transfer maps**: Write pseudocode for $r_{M\square N}$ and $t_{M\square N}$ in terms of $r_M, t_M, r_N, t_N$.

(d) **Test case**: Apply to $M = N = \underline{\mathbb{Z}}$ for $G = C_2$ and verify the output matches the Lewis diagram of $\underline{\mathbb{Z}}$.

> [!TIP]- Solution to Exercise 22
> **Key insight:** At the top level, the coend for $M \square N$ combines $M_0 \otimes N_0$ with the $C_p$-coinvariants of $M_1 \otimes N_1$ (the Frobenius relations quotient out the $C_p$-action difference).
>
> **Sketch:**
>
> ```python
> def box_product_Cp(M, N, p):
>     # (M box N)_1 = M1 tensor N1
>     rank_1 = M.rank1 * N.rank1
>     tau_box = np.kron(M.tau, N.tau)
>
>     # (M box N)_0 = (M0 tensor N0) + coinvariants of M1 tensor N1
>     # Coinvariants: quotient by im(tau_M tensor id - id tensor tau_N^T)
>     relation_matrix = (np.kron(M.tau, np.eye(N.rank1, dtype=int))
>                      - np.kron(np.eye(M.rank1, dtype=int), N.tau))
>     from sympy import Matrix
>     _, D, _ = Matrix(relation_matrix).smith_normal_form()
>     coinvariant_rank = rank_1 - sum(1 for d in D if d != 0)
>     rank_0 = M.rank0 * N.rank0 + coinvariant_rank
>     # ... build r_box and t_box ...
>     return C2MackeyFunctor(rank_0, rank_1, r_box, t_box, tau_box)
>
> # Test: M = N = underline_Z for C2 (p=2)
> # M1 tensor N1 = Z, tau_box = [[1]]
> # Coinvariants: tau_M tensor id - id tensor tau_N = [[1]]-[[1]] = [[0]]
> # => trivial relation => coinvariant_rank = 1
> # rank_0 = 1 + 1 = 2, but Frobenius identifies the two generators,
> # collapsing to rank_0 = 1. Result: Lewis diagram of underline_Z. checkmark
> ```

---

#### Problem 23: Burnside Category Morphism Group Enumerator

*This problem asks for a Python function that, given two transitive $G$-sets $G/H$ and $G/K$ for a finite group $G$, computes the rank of $\mathcal{A}(G)(G/H, G/K)$ by enumerating double cosets $H\backslash G/K$.*

> **Prerequisites:** cf. [[#2.1 Spans of Finite G-Sets|§2.1 — Spans of Finite G-Sets]]; cf. [[#7.1 Representable Mackey Functors|§7.1 — Representable Mackey Functors]]

(a) **Double coset enumeration**: Write a Python function `double_cosets(G, H_elements, K_elements, mul)` that computes the set of double cosets $H\backslash G/K$.

(b) **Morphism group rank**: Write `morphism_rank(G, H_elements, K_elements, mul) -> int` that returns $|H\backslash G/K|$.

(c) **Verification table**: Apply the function to $G = S_3$ and all pairs $(H, K)$ of subgroups (up to conjugacy) to produce a morphism rank table. Verify the entry for $(H, K) = (\langle(12)\rangle, \langle(13)\rangle)$ matches Problem 5.

(d) **Burnside ring structure constants**: Explain how `morphism_rank` relates to the structure constants of $A(G)$. Compute $[S_3/H] \cdot [S_3/K]$ for $H = K = \langle(12)\rangle$ in $A(S_3)$.

> [!TIP]- Solution to Exercise 23
> **Key insight:** The rank of $\mathcal{A}(G)(G/H, G/K)$ equals the number of double cosets $|H\backslash G/K|$, which is computed by partitioning $G$ into $HgK$-classes.
>
> **Sketch:**
>
> ```python
> def double_cosets(G, H_elements, K_elements, mul):
>     remaining = set(G)
>     reps = []
>     while remaining:
>         g = next(iter(remaining))
>         coset = set()
>         for h in H_elements:
>             for k in K_elements:
>                 coset.add(mul(mul(h, g), k))
>         reps.append(g)
>         remaining -= coset
>     return reps
>
> def morphism_rank(G, H_elements, K_elements, mul):
>     return len(double_cosets(G, H_elements, K_elements, mul))
>
> # Verification table for S3 (subgroups up to conjugacy):
> # | H\K   | S3 | C3 | C2 | {e} |
> # |-------|----|----|----|----|
> # | S3    |  1 |  1 |  1 |  1 |
> # | C3    |  1 |  2 |  1 |  3 |
> # | C2    |  1 |  1 |  2 |  3 |
> # | {e}   |  1 |  3 |  3 |  6 |
>
> # [S3/H]^2 for H = <(12)>:
> # H\S3/H: 2 double cosets [e] and [(132)]
> # [e]: stab H => orbit G/H; [(132)]: stab {e} => orbit G/{e}
> # So [S3/H]^2 = [S3/H] + [S3/e] in A(S3).
> ```

---

### G-CW-Spectra and Slice Filtration

#### Problem 24: Genuine G-CW Cells for C2

*This problem makes the two-tier cell structure of genuine $C_2$-CW-spectra concrete by listing all cells of representation-dimension $\leq 2$ and comparing them to their naive and slice counterparts.*

> **Prerequisites:** [[concepts/equivariant-stable-homotopy/g-spectra#5.5 G-CW-Spectra|§5.5 — G-CW-Spectra]]; [[concepts/equivariant-stable-homotopy/equivariant-postnikov-and-slice#4.1 Slice Cells|Slice Cells §4.1]]

Let $G = C_2 = \{e, \tau\}$ with representations: $\mathbf{1}$ (trivial, dimension 1), $\sigma$ (sign, dimension 1), $\rho = \mathbf{1} \oplus \sigma$ (regular, dimension 2).

(a) **Naive cells of dimension $\leq 2$**: List all naive $C_2$-CW cells $C_2/H_+ \wedge D^n$ for $n \leq 2$. For each, identify the subgroup $H \in \{e, C_2\}$, the underlying non-equivariant cell $D^n$, and the $C_2$-action.

(b) **Genuine cells of representation-dimension $\leq 2$**: List all genuine $C_2$-CW cells $C_2/H_+ \wedge D^V$ for $\dim_\mathbb{R}(V) \leq 2$, ranging over all real $C_2$-representations $V$ of dimension $\leq 2$ and both subgroups $H \leq C_2$. Organize by representation type.

(c) **Slice cells among genuine cells**: Identify which genuine cells from (b) are slice cells. Which genuine cells are *not* slice cells?

(d) **Fixed-point contributions**: For each genuine cell in (b), determine what it contributes to $\pi_*^{C_2}$ and $\pi_*^e$.

> [!TIP]- Solution to Exercise 24
> **Key insight:** Genuine $C_2$-CW cells split into two families: the *free cells* $C_2/e_+ \wedge D^V$ (one orbit of cells) and the *fixed cells* $C_2/C_{2+} \wedge D^V$ (cells with full isotropy). The representation type of $V$ distinguishes cells invisible to the naive model structure.
>
> **Sketch:**
>
> (a) *Naive cells of dimension $\leq 2$*: $C_2/e_+ \wedge D^n$ for $n = 0,1,2$ (free, two cells each) and $C_2/C_{2+} \wedge D^n$ for $n = 0,1,2$ (fixed, one cell each).
>
> (b) *Genuine cells of rep-dimension $\leq 2$*: In addition to naive cells, one has $C_2/H_+ \wedge D^{m\sigma}$ for $m = 1$ and $C_2/H_+ \wedge D^{\mathbf{1} \oplus \sigma} = C_2/H_+ \wedge D^\rho$. For dimension 1: $\{C_2/e_+ \wedge D^\sigma,\, C_2/C_{2+} \wedge D^\sigma\}$ are the non-naive genuine cells. For dimension 2: add $\{C_2/H_+ \wedge D^\rho : H \leq C_2\}$.
>
> (c) *Slice cells*: $G_+ \wedge_{C_2} S^{m\rho_{C_2}}$ (dim $2m$) and $G_+ \wedge_e S^m$ (dim $m$). Among dimension $\leq 2$ genuine cells, the slice cells are $C_2/C_{2+} \wedge D^\rho$ and the free cells $C_2/e_+ \wedge D^n$. The non-slice genuine cells are $C_2/H_+ \wedge D^{k\sigma}$ and $C_2/e_+ \wedge D^\rho$ — these involve representations other than the regular representation.
>
> (d) $C_2/e_+ \wedge D^V$ contributes to $\pi_*^e$ (any $V$). $C_2/C_{2+} \wedge D^V$ contributes to $\pi_*^{C_2}$ only when $V^{C_2} \neq 0$; if $V = n\sigma$ (pure sign), $(D^V)^{C_2} = \{0\}$ so the fixed-point contribution is trivial. The naive model structure sees only $H = e$ cells and $V = \mathbb{R}^n$ (trivial representation).

---

#### Problem 25: Restriction of Cells via the Double Coset Formula

*This problem verifies the Mackey double coset formula for restriction of genuine G-CW cells, making the compatibility of the cell structure with change-of-group explicit.*

> **Prerequisites:** [[concepts/equivariant-stable-homotopy/g-spectra#5.5 G-CW-Spectra|§5.5 — G-CW-Spectra]]; [[#4. The Mackey Double Coset Formula|Mackey Double Coset Formula §4]]

Let $G = C_4 = \langle g \rangle$ and let $H = C_2 = \langle g^2 \rangle \leq G$. Let $V = \rho_G$ be the regular representation of $G$ (dimension 4).

(a) Apply the Mackey double coset formula $i_K^*(G/L_+\wedge D^V) \cong \bigsqcup_{[g]\in K\backslash G/L} K/(K\cap {}^gL)_+\wedge D^{V|_{K\cap {}^gL}}$ with $K = H = C_2$, $L = G$, $V = \rho_G$. Enumerate the double cosets and identify each resulting $C_2$-cell.

(b) Show that $V|_{C_2} = \rho_{C_2} \oplus \rho_{C_2}$ (the regular representation of $C_2$ appearing twice). Hence the restricted cell decomposes as two $C_2$-cells of type $C_2/C_{2+} \wedge D^{\rho_{C_2}}$.

(c) Verify that the total representation-dimension is preserved: $\dim_\mathbb{R}(\rho_G) = 4$ and the restricted cells together have total underlying dimension 4.

(d) State the general formula for $\dim_\mathbb{R}(V|_H)$ when $V$ contains $\rho_G$ as a direct summand. Use this to explain why the restriction of a slice cell of dimension $m|G|$ decomposes into cells of dimension $m|H|$.

> [!TIP]- Solution to Exercise 25
> **Key insight:** Restriction of a cell $G/G_+ \wedge D^V$ along $i_{C_2}^*$ uses $C_2 \backslash C_4 / C_4 = \{[e]\}$ (one double coset), so the restriction is a single $C_2$-cell; the representation restricts as $\rho_G|_{C_2} \cong 2\rho_{C_2}$.
>
> **Sketch:**
>
> (a) $C_2 \backslash C_4 / C_4$ has a single element $[e]$. The formula gives $i_{C_2}^*(C_4/C_{4+} \wedge D^{\rho_{C_4}}) \cong C_2/C_{2+} \wedge D^{\rho_{C_4}|_{C_2}}$.
>
> (b) The regular representation restricts: $\rho_{C_4}|_{C_2} \cong 2\rho_{C_2}$ (by Frobenius reciprocity or direct computation: index $[C_4:C_2] = 2$). So the restricted cell is $C_2/C_{2+} \wedge D^{2\rho_{C_2}}$.
>
> (c) $\dim_\mathbb{R}(\rho_{C_4}) = 4 = \dim_\mathbb{R}(2\rho_{C_2}) = 2 \cdot 2$. ✓
>
> (d) In general, $\dim_\mathbb{R}(V|_H) = \dim_\mathbb{R}(V)$ (restriction doesn't change dimension). For a slice cell $G_+ \wedge_G S^{m\rho_G}$ of dimension $m|G|$: the restricted $H$-cell has representation $m\rho_G|_H = m[G:H]\rho_H$, giving cells of total dimension $m|G|$ — dimension preserved, but distributed as $[G:H]$ copies of $H$-slice cells of dimension $m|H|$.

---

#### Problem 26: Slice Filtration Under the Forgetful Functor

*This problem establishes the key comparison $\iota^* P^n_{\mathrm{slice}} X \simeq P^n_{\mathrm{Post}}(\iota^* X)$, showing precisely how the slice tower maps to the Postnikov tower under the forgetful functor to naive G-spectra.*

> **Prerequisites:** [[concepts/equivariant-stable-homotopy/g-spectra#5.5 G-CW-Spectra|§5.5 — G-CW-Spectra]]; [[concepts/equivariant-stable-homotopy/equivariant-postnikov-and-slice#4.2 Slice Null and Slice Positive Spectra|Slice Null/Positive §4.2]]

Let $\iota^*: \mathrm{Sp}^G_{\mathrm{gen}} \to \mathrm{Sp}^G_{\mathrm{naive}}$ be the forgetful functor.

(a) **Cells under $\iota^*$**: Show that $\iota^*$ sends $G/H_+ \wedge D^V$ (with $\dim_\mathbb{R}(V) = n$) to the naive cell $G/H_+ \wedge D^n$.

(b) **Slice-null implies Postnikov-truncated**: Show that if $X$ is slice $n$-null, then $\iota^* X$ is Postnikov $n$-truncated. (*Hint:* free slice cells $G_+ \wedge_e S^k$ are also slice cells of dimension $k$.)

(c) **Slice-positive implies connective**: Show that if $X$ is slice $n$-positive, then $\iota^* X$ is $(n)$-connective as a naive $G$-spectrum.

(d) **Conclusion**: Use (b) and (c) and the universal property of the Postnikov section to conclude $\iota^* P^n_{\mathrm{slice}} X \simeq P^n_{\mathrm{Post}}(\iota^* X)$. Why does the converse fail?

> [!TIP]- Solution to Exercise 26
> **Key insight:** $\iota^*$ kills all representation-sphere data, collapsing $S^V \mapsto S^{\dim V}$. The universal property of $P^n_{\mathrm{slice}}$ (resp. $P^n_{\mathrm{Post}}$) then forces the comparison.
>
> **Sketch:**
>
> (a) $\iota^*$ forgets the representation structure on $V$, viewing it as $\mathbb{R}^{\dim V}$. Hence $\iota^*(G/H_+ \wedge D^V) = G/H_+ \wedge D^n$ with $n = \dim V$.
>
> (b) A free slice cell $G_+ \wedge_e S^k$ is a slice cell of dimension $k$. If $X$ is slice $n$-null, then $\mathrm{Map}(G_+ \wedge_e S^k, X) \simeq *$ for $k > n$. But $\mathrm{Map}(G_+ \wedge_e S^k, X)^e = \mathrm{Map}(S^k, X^e) \simeq *$, so $\pi_k^e(X) = 0$ for $k > n$.
>
> (c) Similarly, slice $n$-positivity forces $G_+ \wedge_e S^k \to X$ null for $k \leq n$, hence $\pi_k^e(\iota^* X) = 0$ for $k \leq n$.
>
> (d) The map $X \to P^n_{\mathrm{slice}} X$ induces $\iota^* X \to \iota^* P^n_{\mathrm{slice}} X$; by (b)/(c) the target is Postnikov-$n$-truncated and the fiber is $n$-connective, so by the universal property $\iota^* P^n_{\mathrm{slice}} X \simeq P^n_{\mathrm{Post}}(\iota^* X)$. The converse fails because $P^n_{\mathrm{Post}}(\iota^* X)$ knows nothing about the $G$-representation-sphere maps $[S^V, X]^H$ for non-trivial $V$; the slice section carries this extra structure.

---

#### Problem 27: Borel Completion and the HFPSS

*This problem shows that for Borel-complete G-spectra the slice spectral sequence collapses to the homotopy fixed point spectral sequence, making the Borel–genuine relationship concrete at the level of spectral sequences.*

> **Prerequisites:** [[concepts/equivariant-stable-homotopy/g-spectra#5.5 G-CW-Spectra|§5.5 — G-CW-Spectra]]; [[concepts/equivariant-stable-homotopy/equivariant-postnikov-and-slice#5.1 Construction from the Slice Tower|Slice SS §5.1]]

Let $E$ be a Borel-complete $G$-spectrum, i.e., $EG_+ \wedge E \xrightarrow{\sim} E$.

(a) **Slices of $E$ are Borel-complete**: Show that for each $n$, the slice $P_n^n E$ is itself Borel-complete.

(b) **Fixed points equal homotopy fixed points on slices**: Using (a), show that $(P_n^n E)^H \simeq (P_n^n E)^{hH}$ for all $H \leq G$.

(c) **The $E_2$-page**: Show that the $E_1$-page of the slice SS satisfies $E_1^{n,*} = \pi_*^G(P_n^n E) \cong C^n(EG;\, \pi_*(E^e))^G$. Conclude that $E_2^{s,t} \cong H^s(G;\, \pi_t(E^e))$.

(d) **A worked example**: Let $G = C_2$ and $E = H\mathbb{Z}$ with $C_2$ acting trivially. Identify the HFPSS $E_2$-page and write down the first few groups. Why does this spectral sequence *not* degenerate at $E_2$?

> [!TIP]- Solution to Exercise 27
> **Key insight:** Borel-completeness ($EG_+ \wedge E \simeq E$) propagates to each slice layer, so $E^H \simeq E^{hH}$ forces the slice $E_1$-page to be group cochains, giving the HFPSS $E_2$-page.
>
> **Sketch:**
>
> (a) The Borel completion $EG_+ \wedge -$ is a smashing localization. It commutes with fiber sequences and preserves the slice tower structure. Since $EG_+ \wedge E \simeq E$, applying $EG_+ \wedge -$ to the fiber sequences in the slice tower preserves the same property for each $P_n^n E$.
>
> (b) For Borel-complete $E'$: the cofiber sequence $EG_+ \wedge E' \to E' \to \widetilde{EG} \wedge E'$ collapses to $E' \xrightarrow{\sim} E' \to *$, so $\widetilde{EG} \wedge E' \simeq *$. By the Tate cofiber sequence $(E')^{tH} = 0$, and thus $(E')^H \simeq (E')^{hH}$.
>
> (c) Since $(P_n^n E)^H \simeq (P_n^n E)^{hH}$, the homotopy groups $\pi_*^G(P_n^n E)$ depend only on the homotopy $G$-module $\pi_*(P_n^n E^e)$. The $E_1$-page is equivariant cochains on $EG$, so $E_2^{s,t} \cong H^s(G; \pi_t(E^e))$.
>
> (d) $G = C_2$, $E = H\mathbb{Z}$ trivially: $\pi_t(E^e) = \mathbb{Z}$ for $t = 0$ and $0$ otherwise. HFPSS: $E_2^{s,0} = H^s(C_2; \mathbb{Z})$ — that is $\mathbb{Z}$ for $s = 0$, $\mathbb{Z}/2$ for $s > 0$ even, and $0$ for $s$ odd. It does *not* degenerate at $E_2$ because the abutment $\pi_*(H\mathbb{Z}^{hC_2})$ is nontrivial in negative degrees, and the differential $d_2: E_2^{0,0} \to E_2^{2,-1}$ is nontrivial.

---

#### Problem 28: Genuine G-CW Cell Enumerator

*This problem develops a Python algorithm to enumerate the genuine G-CW cells of a given dimension for cyclic groups, making the two-tier cell structure algorithmic.*

> **Prerequisites:** [[concepts/equivariant-stable-homotopy/g-spectra#5.5 G-CW-Spectra|§5.5 — G-CW-Spectra]]

For $G = C_p$ (cyclic of prime order $p$), the real representations of dimension $\leq n$ are: $k\mathbf{1}$ (trivial, dim $k$) and $k\mathbf{1} \oplus m\sigma$ (with $\sigma$ the sign/rotation representation, dim $k + 2m$).

(a) **Representation enumeration**: Write a Python function `reps_of_dim(p, n)` that returns all isomorphism classes of real $C_p$-representations of dimension $\leq n$, as pairs `(trivial_copies, nontrivial_copies)`. For $p = 2$ and $n = 4$, list the output.

(b) **Cell enumeration**: Write `genuine_cells(p, n)` that returns all genuine $C_p$-CW cells $C_p/H_+ \wedge D^V$ of $\dim_\mathbb{R}(V) \leq n$, as triples `(H, V_trivial, V_nontrivial)`. Run it for $p = 2$, $n = 2$ and compare to Problem 24(b).

(c) **Naive vs. genuine count**: For $G = C_2$ and $n = 1, 2, 3, 4$, compute the number of distinct naive cells and genuine cells of dimension exactly $n$. Fill in a table. What is the ratio of genuine to naive cells for large $n$?

(d) **Slice cell identification**: Extend `genuine_cells` to flag which cells are slice cells (i.e., $C_p/H_+ \wedge D^{m\rho_H}$ for some $m$). For $p = 2$ and $n \leq 4$, list the non-slice genuine cells.

> [!TIP]- Solution to Exercise 28
> **Key insight:** For $G = C_p$, the real representations of dimension $n$ form a two-parameter family $(k, m)$ with $k + 2m = n$ ($k$ trivial summands, $m$ nontrivial 2D summands). The genuine cells are indexed by this data together with a choice of subgroup $H \leq C_p$.
>
> **Sketch:**
>
> ```python
> def reps_of_dim(p, n):
>     result = []
>     for total_dim in range(n + 1):
>         for nontrivial in range(total_dim // 2 + 1):
>             trivial = total_dim - 2 * nontrivial
>             result.append((trivial, nontrivial))
>     return result
>
> def genuine_cells(p, n):
>     subgroups = [1, p]  # H = {e} (order 1) or H = C_p (order p)
>     cells = []
>     for (k, m) in reps_of_dim(p, n):
>         for H_order in subgroups:
>             cells.append((H_order, k, m))
>     return cells
>
> # For p=2, n=2: reps = (0,0),(1,0),(2,0),(0,1) -> dims 0,1,2,2
> # Cells: 4 reps x 2 subgroups = 8 cells (vs 6 naive cells)
> ```
>
> (c) Naive cells of dim $n$: 2 (one per subgroup, trivial rep). Genuine cells of dim $n$: $2 \times (\lfloor n/2 \rfloor + 1)$ (number of reps of dim $n$ times 2 subgroups).
>
> | $n$ | Naive | Genuine | Ratio |
> |-----|-------|---------|-------|
> | 1 | 2 | 4 | 2 |
> | 2 | 2 | 6 | 3 |
> | 3 | 2 | 8 | 4 |
> | 4 | 2 | 10 | 5 |
>
> For large $n$: ratio $\approx \lfloor n/2 \rfloor + 1 \sim n/2$.
>
> (d) Slice cells for $C_2$: $C_2/H_+ \wedge D^{m\rho_H}$ for $m \geq 0$ (only regular representation multiples). Non-slice genuine cells for dim $\leq 4$: $C_2/H_+ \wedge D^{k\sigma}$ (pure sign reps, $k = 1, 2$) and $C_2/H_+ \wedge D^{\mathbf{1}+k\sigma}$ for $k \geq 2$ (mixed non-regular).

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
