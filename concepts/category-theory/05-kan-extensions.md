# 📐 Category Theory V: Kan Extensions

## Table of Contents

- [[#1. Motivation: Extending Functors Along a Third Category|1. Motivation: Extending Functors Along a Third Category]]
- [[#2. Definition of Kan Extensions|2. Definition of Kan Extensions]]
  - [[#2.1 Left Kan Extension|2.1 Left Kan Extension]]
  - [[#2.2 Right Kan Extension|2.2 Right Kan Extension]]
  - [[#2.3 Kan Extensions as Adjoints to Precomposition|2.3 Kan Extensions as Adjoints to Precomposition]]
- [[#3. The Pointwise Formula|3. The Pointwise Formula]]
  - [[#3.1 Comma Categories Recalled|3.1 Comma Categories Recalled]]
  - [[#3.2 The Colimit Formula for Left Kan Extensions|3.2 The Colimit Formula for Left Kan Extensions]]
  - [[#3.3 The Limit Formula for Right Kan Extensions|3.3 The Limit Formula for Right Kan Extensions]]
  - [[#3.4 The Coend Formula|3.4 The Coend Formula]]
  - [[#3.5 Proof Sketch: The Colimit Formula Satisfies the Universal Property|3.5 Proof Sketch: The Colimit Formula Satisfies the Universal Property]]
- [[#4. Pointwise Kan Extensions|4. Pointwise Kan Extensions]]
  - [[#4.1 Definition and Characterisation|4.1 Definition and Characterisation]]
  - [[#4.2 Preservation by Right Adjoints|4.2 Preservation by Right Adjoints]]
- [[#5. All Concepts are Kan Extensions|5. All Concepts are Kan Extensions]]
  - [[#5.1 Limits and Colimits as Kan Extensions|5.1 Limits and Colimits as Kan Extensions]]
  - [[#5.2 Adjoints as Kan Extensions|5.2 Adjoints as Kan Extensions]]
  - [[#5.3 The Yoneda Lemma and the Density Theorem|5.3 The Yoneda Lemma and the Density Theorem]]
- [[#6. Derived Functors as Kan Extensions|6. Derived Functors as Kan Extensions]]
- [[#7. Ends, Coends, and the Ninja Yoneda Lemma|7. Ends, Coends, and the Ninja Yoneda Lemma]]
  - [[#7.1 Ends|7.1 Ends]]
  - [[#7.2 Coends|7.2 Coends]]
  - [[#7.3 The Ninja Yoneda Lemma|7.3 The Ninja Yoneda Lemma]]
  - [[#7.4 Natural Transformations as Ends|7.4 Natural Transformations as Ends]]
- [[#References|References]]

---

> [!INFO] Series context
> This is File 5 of 5 in a self-study series based on Emily Riehl's *Category Theory in Context* (Chapter 6). This file presupposes [[concepts/category-theory/01-categories-functors-natural-transformations|File 1]] (categories and functors), [[concepts/category-theory/02-adjoints-representables|File 2]] (adjoints and representables), [[concepts/category-theory/03-limits-colimits|File 3]] (limits and colimits), and [[concepts/category-theory/04-adjoint-functor-theorems-monads|File 4]] (adjoint functor theorems and monads). The central theme is Mac Lane's maxim: *"All concepts are Kan extensions."*

---

## 1. Motivation: Extending Functors Along a Third Category 🔍

Suppose $F: \mathcal{C} \to \mathcal{E}$ is a functor and $K: \mathcal{C} \to \mathcal{D}$ is another functor. The *extension problem* asks: does there exist a functor $\mathcal{D} \to \mathcal{E}$ that, when precomposed with $K$, agrees with $F$ (or is as close to agreeing as possible)?

$$\begin{array}{c}
\mathcal{C} \xrightarrow{K} \mathcal{D} \\
\searrow^{F} \quad \downarrow^{?} \\
\mathcal{E}
\end{array}$$

More precisely: if $K$ is injective on objects (e.g. a full inclusion), can we extend $F$ to a functor defined on all of $\mathcal{D}$? If $K$ is not injective, the question is subtler: we ask for the *best approximation* to an extension, in a universal sense. This is the *Kan extension* problem.

> [!INFO] Historical context
> Daniel Kan introduced these constructions in his 1958 paper "Adjoint functors," studying them in the context of simplicial sets and homotopy theory. The phrase "all concepts are Kan extensions" appears in Mac Lane's *Categories for the Working Mathematician* (Chapter X, §7) as a programmatic slogan. Riehl's Chapter 6, whose title borrows this phrase, makes it precise by showing that limits, colimits, and adjoints are all instances.

**Motivating example: geometric realization.** The standard $n$-simplex $\Delta^n \subset \mathbb{R}^{n+1}$ defines a functor $\Delta^\bullet: \mathbf{\Delta} \to \mathbf{Top}$ from the simplex category. The Yoneda embedding $y: \mathbf{\Delta} \to [\mathbf{\Delta}^{op}, \mathbf{Set}]$ sends $[n]$ to the representable presheaf $\mathbf{\Delta}(-,[n])$. Geometric realization $|{-}|: [\mathbf{\Delta}^{op}, \mathbf{Set}] \to \mathbf{Top}$ is the *left Kan extension* of $\Delta^\bullet$ along $y$: it extends the functor $\Delta^\bullet$ from the simplex category to all simplicial sets in the "most efficient" way compatible with colimits.

**Notation for this file.** Throughout:
- $\mathcal{C}, \mathcal{D}, \mathcal{E}$ denote categories (locally small unless stated otherwise).
- $[\mathcal{C}, \mathcal{E}]$ denotes the functor category with objects the functors $\mathcal{C} \to \mathcal{E}$ and morphisms the natural transformations.
- $K^*: [\mathcal{D}, \mathcal{E}] \to [\mathcal{C}, \mathcal{E}]$ denotes the *precomposition functor* sending $G \mapsto G \circ K$.
- For objects $d \in \mathcal{D}$, $(K \downarrow d)$ denotes the *comma category* of $K$ over $d$; see §3.1.

---

## 2. Definition of Kan Extensions 📐

### 2.1 Left Kan Extension

**Definition (Left Kan Extension).** Let $K: \mathcal{C} \to \mathcal{D}$ and $F: \mathcal{C} \to \mathcal{E}$ be functors. A *left Kan extension of $F$ along $K$* is a functor $\mathrm{Lan}_K F: \mathcal{D} \to \mathcal{E}$ together with a natural transformation

$$\eta: F \Rightarrow (\mathrm{Lan}_K F) \circ K,$$

called the *unit*, satisfying the following universal property: for any functor $G: \mathcal{D} \to \mathcal{E}$ and any natural transformation $\alpha: F \Rightarrow G \circ K$, there exists a *unique* natural transformation $\bar{\alpha}: \mathrm{Lan}_K F \Rightarrow G$ such that $(\bar{\alpha} \star K) \circ \eta = \alpha$, i.e. the composite

$$F \xRightarrow{\eta} (\mathrm{Lan}_K F) \circ K \xRightarrow{\bar{\alpha} \star K} G \circ K$$

equals $\alpha$. Here $\bar{\alpha} \star K$ denotes the *whiskering* of $\bar{\alpha}$ by $K$.

In other words, $\mathrm{Lan}_K F$ is the *initial* functor under $K$ equipped with a natural transformation from $F$: it is initial in the category of pairs $(G, \alpha: F \Rightarrow G \circ K)$.

The situation is depicted by the following diagram, in which the dashed arrow is the unique fill-in:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\mathcal{C} \arrow[r, "K"] \arrow[dr, "F"'] & \mathcal{D} \arrow[d, "{\mathrm{Lan}_K F}", dashed] \arrow[d, bend left=50, "G" description] \\
& \mathcal{E}
\end{tikzcd}
\end{document}
```

The natural transformation $\eta$ fills the triangle on the left (from $F$ to $(\mathrm{Lan}_K F) \circ K$), and any other filling $\alpha$ factors uniquely through $\eta$ via $\bar{\alpha}$.

> [!NOTE] Uniqueness up to isomorphism
> As with any construction defined by a universal property, $\mathrm{Lan}_K F$ is unique up to canonical isomorphism. If $(L, \eta)$ and $(L', \eta')$ both satisfy the universal property, the unique maps $\bar{\eta}': L \to L'$ and $\bar{\eta}: L' \to L$ are mutually inverse by uniqueness.

> [!NOTE] Riehl Exercise: Unit of a discrete Kan extension
> Let $K: \mathbf{1} \to \mathcal{D}$ pick out an object $d_0 \in \mathcal{D}$, and let $F: \mathbf{1} \to \mathcal{E}$ pick out an object $e_0 \in \mathcal{E}$. Describe $(\mathrm{Lan}_K F)(d)$ explicitly for any $d \in \mathcal{D}$, and verify the universal property directly. What is the unit $\eta: e_0 \Rightarrow (\mathrm{Lan}_K F)(d_0)$?

### 2.2 Right Kan Extension

**Definition (Right Kan Extension).** A *right Kan extension of $F$ along $K$* is a functor $\mathrm{Ran}_K F: \mathcal{D} \to \mathcal{E}$ together with a natural transformation

$$\varepsilon: (\mathrm{Ran}_K F) \circ K \Rightarrow F,$$

called the *counit*, satisfying the dual universal property: for any functor $G: \mathcal{D} \to \mathcal{E}$ and any natural transformation $\beta: G \circ K \Rightarrow F$, there exists a unique natural transformation $\bar{\beta}: G \Rightarrow \mathrm{Ran}_K F$ such that $\varepsilon \circ (\bar{\beta} \star K) = \beta$.

That is, $\mathrm{Ran}_K F$ is the *terminal* functor over $K$ equipped with a natural transformation to $F$: it is terminal in the category of pairs $(G, \beta: G \circ K \Rightarrow F)$.

> [!TIP] The mnemonic
> Left extension: $\eta$ goes *from* $F$ (upward), so $\mathrm{Lan}_K F$ is "larger than" $F$. Right extension: $\varepsilon$ goes *to* $F$ (downward), so $\mathrm{Ran}_K F$ is "smaller than" $F$. This mirrors the left–right convention for adjoints: left adjoints have units $\eta: \mathrm{id} \Rightarrow GF$, right adjoints have counits $\varepsilon: FG \Rightarrow \mathrm{id}$.

### 2.3 Kan Extensions as Adjoints to Precomposition

The defining universal properties state precisely that $\mathrm{Lan}_K$ and $\mathrm{Ran}_K$ are adjoints to the precomposition functor.

**Proposition (Kan Extensions as Adjoints).** If $\mathrm{Lan}_K F$ exists for every $F: \mathcal{C} \to \mathcal{E}$, then $\mathrm{Lan}_K: [\mathcal{C}, \mathcal{E}] \to [\mathcal{D}, \mathcal{E}]$ is left adjoint to the precomposition functor $K^*: [\mathcal{D}, \mathcal{E}] \to [\mathcal{C}, \mathcal{E}]$:

$$\mathrm{Lan}_K \dashv K^*.$$

More precisely, the universal property of $\mathrm{Lan}_K F$ gives a natural bijection:

$$[\mathcal{D}, \mathcal{E}](\mathrm{Lan}_K F,\, G) \;\cong\; [\mathcal{C}, \mathcal{E}](F,\, G \circ K) = [\mathcal{C}, \mathcal{E}](F,\, K^* G).$$

Dually, if $\mathrm{Ran}_K F$ exists for every $F$, then $\mathrm{Ran}_K \dashv$ no — rather, $K^* \dashv \mathrm{Ran}_K$:

$$K^* \dashv \mathrm{Ran}_K,$$

with the natural bijection $[\mathcal{C}, \mathcal{E}](K^* G,\, F) \cong [\mathcal{D}, \mathcal{E}](G,\, \mathrm{Ran}_K F)$.

*Proof sketch.* The unit of the adjunction $\mathrm{Lan}_K \dashv K^*$ at $F$ is exactly $\eta: F \Rightarrow K^*(\mathrm{Lan}_K F)$, and the counit at $G$ is the unique map $\overline{\mathrm{id}}: \mathrm{Lan}_K(G \circ K) \Rightarrow G$ induced by the identity $\mathrm{id}: G \circ K \Rightarrow G \circ K$ and the universal property of $\mathrm{Lan}_K(G \circ K)$. One checks the triangle identities. $\square$

> [!INFO] Size issues
> The functor categories $[\mathcal{C}, \mathcal{E}]$ and $[\mathcal{D}, \mathcal{E}]$ may be large. The adjunction $\mathrm{Lan}_K \dashv K^*$ holds at the level of categories of *small* functors, or when all relevant Kan extensions exist. We ignore set-theoretic subtleties throughout, as is standard in this context.

---

## 3. The Pointwise Formula ⚙️

The abstract universal property tells us *what* $\mathrm{Lan}_K F$ is, but not *how to compute it*. When the target category $\mathcal{E}$ has enough colimits, there is an explicit formula.

### 3.1 Comma Categories Recalled

The *comma category* $(K \downarrow d)$ for a fixed $d \in \mathcal{D}$ has:
- **Objects:** pairs $(c, f)$ with $c \in \mathcal{C}$ and $f: Kc \to d$ a morphism in $\mathcal{D}$.
- **Morphisms:** a morphism $(c, f) \to (c', f')$ is a morphism $g: c \to c'$ in $\mathcal{C}$ such that $f' \circ Kg = f$, i.e. the triangle

$$Kc \xrightarrow{Kg} Kc' \xrightarrow{f'} d \quad \text{equals} \quad Kc \xrightarrow{f} d.$$

There is a canonical *projection functor* $\Pi_d: (K \downarrow d) \to \mathcal{C}$ sending $(c, f) \mapsto c$ and $g \mapsto g$. The composite $F \circ \Pi_d: (K \downarrow d) \to \mathcal{E}$ is the diagram whose colimit will compute $(\mathrm{Lan}_K F)(d)$.

Dually, the comma category $(d \downarrow K)$ has objects $(c, f)$ with $f: d \to Kc$, and the composite $F \circ \Pi_d^{op}: (d \downarrow K) \to \mathcal{E}$ is the diagram whose limit computes $(\mathrm{Ran}_K F)(d)$.

> [!EXAMPLE] Comma category in a concrete case
> Let $\mathcal{C} = \mathbf{\Delta}$, $\mathcal{D} = [\mathbf{\Delta}^{op}, \mathbf{Set}]$, $K = y$ (Yoneda embedding), and $d = X$ a simplicial set. An object of $(y \downarrow X)$ is a pair $([n], f: y[n] \to X)$; by Yoneda, $f$ corresponds to an element $x \in X_n$. So $(y \downarrow X)$ is the *category of elements* of $X$, whose objects are pairs $([n], x \in X_n)$. The colimit formula then says: $(\mathrm{Lan}_y \Delta^\bullet)(X) = \mathrm{colim}_{([n], x) \in \int X} \Delta^n = |X|$ (geometric realization).

### 3.2 The Colimit Formula for Left Kan Extensions

**Theorem (Pointwise Left Kan Extension).** Let $K: \mathcal{C} \to \mathcal{D}$ and $F: \mathcal{C} \to \mathcal{E}$. If $\mathcal{C}$ is small and $\mathcal{E}$ has all colimits of shape $(K \downarrow d)$ for each $d \in \mathcal{D}$, then the left Kan extension $\mathrm{Lan}_K F$ exists and is computed *pointwise* by:

$$(\mathrm{Lan}_K F)(d) \;=\; \mathrm{colim}\bigl(F \circ \Pi_d: (K \downarrow d) \to \mathcal{E}\bigr).$$

More explicitly, $(\mathrm{Lan}_K F)(d)$ is the colimit of the diagram that sends each pair $(c, f: Kc \to d)$ to $Fc \in \mathcal{E}$.

**Functoriality:** For a morphism $g: d \to d'$ in $\mathcal{D}$, the map $(\mathrm{Lan}_K F)(g): (\mathrm{Lan}_K F)(d) \to (\mathrm{Lan}_K F)(d')$ is induced by the functor $(K \downarrow d) \to (K \downarrow d')$ sending $(c, f: Kc \to d) \mapsto (c, g \circ f: Kc \to d')$, which gives a morphism of diagrams, hence a unique map between their colimits by universality.

> [!NOTE] Riehl Exercise: Computing a left Kan extension explicitly
> Let $K: \mathbf{2} \to \mathbf{3}$ be the inclusion of the two-object discrete category $\{0, 1\}$ into the three-object discrete category $\{0, 1, 2\}$ sending $0 \mapsto 0$, $1 \mapsto 1$. Let $F: \mathbf{2} \to \mathbf{Ab}$ send $0 \mapsto \mathbb{Z}$ and $1 \mapsto \mathbb{Z}/2$. Compute $(\mathrm{Lan}_K F)(2)$ explicitly using the comma category formula, and verify it satisfies the universal property.

### 3.3 The Limit Formula for Right Kan Extensions

**Theorem (Pointwise Right Kan Extension).** Dually, if $\mathcal{C}$ is small and $\mathcal{E}$ has all limits of shape $(d \downarrow K)$ for each $d \in \mathcal{D}$, then:

$$(\mathrm{Ran}_K F)(d) \;=\; \mathrm{lim}\bigl(F \circ \Pi_d^{op}: (d \downarrow K) \to \mathcal{E}\bigr).$$

The limit is over all pairs $(c, f: d \to Kc)$ with value $Fc$.

> [!WARNING] Existence is not automatic
> The formulas require $\mathcal{E}$ to have specific classes of colimits (resp. limits). If $\mathcal{E}$ lacks these, neither the formula nor the Kan extension need exist. When $\mathcal{C}$ is small and $\mathcal{E}$ is cocomplete, the left Kan extension always exists. When $\mathcal{E}$ is complete, the right Kan extension always exists.

### 3.4 The Coend Formula

The pointwise formula has a cleaner reformulation in terms of *coends* (to be defined in §7). When $\mathcal{E}$ is *tensored* over $\mathbf{Set}$ — meaning for each set $S$ and object $e \in \mathcal{E}$ there is an object $S \otimes e = \coprod_{s \in S} e$ (the *copower* or *tensor* of $S$ with $e$) — the left Kan extension formula takes the form:

$$(\mathrm{Lan}_K F)(d) \;=\; \int^{c \in \mathcal{C}} \mathcal{D}(Kc, d) \otimes Fc.$$

Here $\mathcal{D}(Kc, d)$ is a set, $\otimes$ is the copower, and $\int^c$ denotes the coend (§7.2). In the special case $\mathcal{E} = \mathbf{Set}$:

$$(\mathrm{Lan}_K F)(d) \;=\; \int^{c \in \mathcal{C}} \mathcal{D}(Kc, d) \times Fc.$$

This is the *density formula* for left Kan extensions in $\mathbf{Set}$.

Dually, the right Kan extension in a *cotensored* category uses an end:

$$(\mathrm{Ran}_K F)(d) \;=\; \int_{c \in \mathcal{C}} Fc^{\mathcal{D}(d, Kc)},$$

where $e^S = \prod_{s \in S} e$ denotes the *power* (cotensor) of $e$ with $S$.

> [!EXAMPLE] Left Kan extension in Set
> Let $K = \mathrm{id}_{\mathcal{C}}: \mathcal{C} \to \mathcal{C}$ and $F: \mathcal{C} \to \mathbf{Set}$. The comma category $(\mathrm{id} \downarrow c)$ has objects $(c', f: c' \to c)$, and $F \circ \Pi_c$ sends $(c', f) \mapsto Fc'$. The colimit of this diagram is the *category of elements coend*: $\int^{c'} \mathcal{C}(c', c) \times Fc'$. The Yoneda lemma (in coend form, §7.3) tells us this is exactly $Fc$. So $\mathrm{Lan}_{\mathrm{id}} F \cong F$, as expected.

### 3.5 Proof Sketch: The Colimit Formula Satisfies the Universal Property

*Claim:* Setting $L(d) := \mathrm{colim}_{(K \downarrow d)} (F \circ \Pi_d)$ defines a functor $L: \mathcal{D} \to \mathcal{E}$, and there is a natural transformation $\eta: F \Rightarrow L \circ K$ such that $(L, \eta)$ is the left Kan extension.

*Proof sketch (in three steps):*

**Step 1 (Unit construction).** For each $c \in \mathcal{C}$, the pair $(c, \mathrm{id}_{Kc}: Kc \to Kc)$ is an object of $(K \downarrow Kc)$. The colimit cocone at $d = Kc$ therefore includes a canonical leg $\lambda^{Kc}_{(c, \mathrm{id}_{Kc})}: Fc \to L(Kc)$. Set $\eta_c := \lambda^{Kc}_{(c, \mathrm{id}_{Kc})}$. Naturality of $\eta$ in $c$ follows from the naturality of the colimit construction.

**Step 2 (Universality).** Given any $G: \mathcal{D} \to \mathcal{E}$ and $\alpha: F \Rightarrow G \circ K$, we must produce a unique $\bar{\alpha}: L \Rightarrow G$. For each $d \in \mathcal{D}$, a cocone from the diagram $F \circ \Pi_d$ to $Gd$ is given by the family indexed by $(c, f: Kc \to d)$:

$$Fc \xrightarrow{\alpha_c} G(Kc) \xrightarrow{G(f)} Gd.$$

This is indeed a cocone (check: for $g: (c,f) \to (c', f')$ in $(K \downarrow d)$, the composites agree by naturality of $\alpha$ and functoriality of $G$). By the universal property of $L(d) = \mathrm{colim}_{(K \downarrow d)} F \circ \Pi_d$, there is a unique map $\bar{\alpha}_d: L(d) \to Gd$ such that $\bar{\alpha}_d \circ \lambda^d_{(c,f)} = G(f) \circ \alpha_c$ for all $(c,f)$.

**Step 3 (Uniqueness).** Any $\bar{\alpha}': L \Rightarrow G$ with $(\bar{\alpha}' \star K) \circ \eta = \alpha$ satisfies $\bar{\alpha}'_d \circ \lambda^d_{(c,f)} = G(f) \circ \alpha_c$ for all $(c,f)$ (by a short diagram chase using the cocone property). Hence $\bar{\alpha}' = \bar{\alpha}$ by uniqueness of colimit maps. $\square$

---

## 4. Pointwise Kan Extensions 🎯

### 4.1 Definition and Characterisation

**Definition (Pointwise Kan Extension).** A left Kan extension $(L, \eta)$ of $F$ along $K$ is *pointwise* if it is computed by the colimit formula of §3.2: $L(d) = \mathrm{colim}_{(K \downarrow d)} F \circ \Pi_d$ for every $d \in \mathcal{D}$.

An equivalent intrinsic characterisation: $(L, \eta)$ is pointwise if and only if it is *preserved by all representable functors*: for every object $e \in \mathcal{E}$, the functor $\mathcal{E}(e, -): \mathcal{E} \to \mathbf{Set}$ sends $(L, \eta)$ to the left Kan extension of $\mathcal{E}(e, F-): \mathcal{C} \to \mathbf{Set}$ along $K$. That is, the natural map

$$\mathcal{E}(e, L(d)) \;\longrightarrow\; \mathrm{Lan}_K\bigl[\mathcal{E}(e, F-)\bigr](d)$$

is an isomorphism for all $e \in \mathcal{E}$ and $d \in \mathcal{D}$.

> [!WARNING] Non-pointwise Kan extensions exist
> It is possible for a Kan extension to exist without being pointwise. This occurs when the unit $\eta: F \Rightarrow L \circ K$ is not a colimit-computing cocone at each $d$. Non-pointwise Kan extensions satisfy the abstract universal property but cannot be computed by the local formula, and they are typically less well-behaved. In practice, virtually every Kan extension arising in mathematics is pointwise.

**Theorem (Existence of Pointwise Left Kan Extensions).** If $\mathcal{C}$ is small and $\mathcal{E}$ is cocomplete, then for any $K: \mathcal{C} \to \mathcal{D}$ and $F: \mathcal{C} \to \mathcal{E}$, the pointwise left Kan extension $\mathrm{Lan}_K F$ exists.

### 4.2 Preservation by Right Adjoints

**Proposition (Pointwise Kan Extensions are Preserved by Right Adjoints).** If $(L, \eta)$ is a pointwise left Kan extension of $F: \mathcal{C} \to \mathcal{E}$ along $K: \mathcal{C} \to \mathcal{D}$, and $H: \mathcal{E} \to \mathcal{E}'$ is a functor that preserves the relevant colimits (e.g. $H$ has a right adjoint), then $(H \circ L, H \star \eta)$ is the pointwise left Kan extension of $H \circ F$ along $K$:

$$H \circ \mathrm{Lan}_K F \;\cong\; \mathrm{Lan}_K(H \circ F).$$

*Proof sketch.* Since right adjoints preserve limits (by [[concepts/category-theory/03-limits-colimits|File 3]], §5), left adjoints preserve colimits by duality. The formula $(H \circ L)(d) = H(\mathrm{colim}_{(K \downarrow d)} F \circ \Pi_d) \cong \mathrm{colim}_{(K \downarrow d)} H \circ F \circ \Pi_d$ follows. $\square$

> [!NOTE] Riehl Exercise: Preservation and composition of Kan extensions
> Let $K: \mathcal{C} \to \mathcal{D}$, $J: \mathcal{D} \to \mathcal{D}'$, and $F: \mathcal{C} \to \mathcal{E}$. Assuming all relevant Kan extensions exist and are pointwise, prove that $\mathrm{Lan}_{J \circ K} F \cong \mathrm{Lan}_J (\mathrm{Lan}_K F)$. (Hint: use the comma category $(J \circ K \downarrow d')$ and show it is equivalent to $(K \downarrow -)$ integrated over $(J \downarrow d')$.)

---

## 5. All Concepts are Kan Extensions 🔑

Mac Lane's slogan becomes a theorem once we identify the correct choice of $K$, $F$, and $\mathcal{E}$ in each case.

### 5.1 Limits and Colimits as Kan Extensions

Let $\mathbf{1}$ denote the *terminal category* with one object $*$ and one (identity) morphism. For any category $\mathcal{I}$, there is a unique functor $!: \mathcal{I} \to \mathbf{1}$.

**Proposition (Limits and Colimits as Kan Extensions).** Let $F: \mathcal{I} \to \mathcal{C}$ be a diagram.

1. *Colimits as left Kan extensions:* $\mathrm{colim}\, F = (\mathrm{Lan}_! F)(*)$. That is, the left Kan extension of $F$ along $!: \mathcal{I} \to \mathbf{1}$, evaluated at the unique object $*$, is the colimit of $F$.

2. *Limits as right Kan extensions:* $\mathrm{lim}\, F = (\mathrm{Ran}_! F)(*)$.

*Proof sketch.* The comma category $(! \downarrow *) \cong \mathcal{I}$ (since there is only one object in $\mathbf{1}$, every map in $\mathbf{1}$ is $\mathrm{id}_*$, and the objects of $(! \downarrow *)$ are pairs $(i \in \mathcal{I},\, !i = * \to *)$ i.e. just objects $i \in \mathcal{I}$). Therefore:

$$(\mathrm{Lan}_! F)(*) = \mathrm{colim}_{(! \downarrow *)} F \circ \Pi_* = \mathrm{colim}_{\mathcal{I}} F.$$

The right Kan extension case is dual. $\square$

**Corollary.** A category $\mathcal{C}$ is *cocomplete* if and only if $\mathrm{Lan}_!$ exists for every small diagram shape $\mathcal{I}$ and every $F: \mathcal{I} \to \mathcal{C}$.

> [!EXAMPLE] The empty colimit
> The colimit of the empty diagram $F: \mathbf{0} \to \mathcal{C}$ (empty category) is the initial object of $\mathcal{C}$. As a Kan extension: $\mathrm{Lan}_! F$ where $!: \mathbf{0} \to \mathbf{1}$. The comma category $(! \downarrow *) = \mathbf{0}$, so the colimit is the empty colimit — the initial object. This recovers the standard fact that the initial object is the colimit of the empty diagram.

> [!NOTE] Riehl Exercise: Limits as right Kan extensions
> Prove carefully that the limit of $F: \mathcal{I} \to \mathcal{C}$ is $(\mathrm{Ran}_! F)(*)$ by working through the universal property. Specifically, show that a cone over $F$ with vertex $c \in \mathcal{C}$ corresponds precisely to a natural transformation $\Delta_c \Rightarrow F$ (where $\Delta_c: \mathcal{I} \to \mathcal{C}$ is the constant functor), and verify this matches the universal property of $\mathrm{Ran}_! F$.

### 5.2 Adjoints as Kan Extensions

This is the deepest unification result. Every adjoint functor *is* a Kan extension — of the identity functor.

**Theorem (Adjoints as Kan Extensions).** Let $F: \mathcal{C} \to \mathcal{D}$ and $G: \mathcal{D} \to \mathcal{C}$ with $F \dashv G$. Then:

1. $G \cong \mathrm{Ran}_F \mathrm{id}_{\mathcal{D}}$: the right adjoint $G$ is the right Kan extension of the identity functor on $\mathcal{D}$ along $F$.
2. $F \cong \mathrm{Lan}_G \mathrm{id}_{\mathcal{C}}$: the left adjoint $F$ is the left Kan extension of the identity functor on $\mathcal{C}$ along $G$.

*Proof sketch (for part 1).* We show $G$ satisfies the universal property of $\mathrm{Ran}_F \mathrm{id}_{\mathcal{D}}$. The counit is $\varepsilon: G \circ F \Rightarrow \mathrm{id}_{\mathcal{D}} \circ F = F$... wait, let us be careful about variance. We need a natural transformation $\varepsilon': (\mathrm{Ran}_F \mathrm{id}_{\mathcal{D}}) \circ F \Rightarrow \mathrm{id}_{\mathcal{D}}$. Take $\varepsilon': GF \Rightarrow \mathrm{id}_{\mathcal{D}}$ to be the counit of the adjunction $F \dashv G$. For any $H: \mathcal{D} \to \mathcal{C}$ and $\beta: HF \Rightarrow \mathrm{id}_{\mathcal{D}}$, the adjunction bijection gives a natural transformation $\bar{\beta}: H \Rightarrow G$ as the composite:

$$H \xrightarrow{\eta \star H} GFH \xrightarrow{G \star \beta} G \cdot \mathrm{id}_{\mathcal{D}} = G.$$

One verifies $\varepsilon' \circ (\bar{\beta} \star F) = \beta$ using the triangle identities for $F \dashv G$. Uniqueness of $\bar{\beta}$ follows from the bijectivity of the adjunction hom-set bijection. $\square$

> [!INFO] Conceptual meaning
> This theorem says: given $F: \mathcal{C} \to \mathcal{D}$, the right adjoint (if it exists) is the "best approximation" to extending the identity $\mathrm{id}_{\mathcal{D}}$ backwards along $F$. This is the abstract sense in which $G$ is a "pseudo-inverse" to $F$: it extends $\mathrm{id}$ as far as possible.

> [!NOTE] Riehl Exercise: Adjoints as Kan extensions
> Prove part 2 of the theorem: if $F \dashv G$, then $F \cong \mathrm{Lan}_G \mathrm{id}_\mathcal{C}$. Use the pointwise formula: for each $d \in \mathcal{D}$, compute $\mathrm{colim}_{(G \downarrow d)} \mathrm{id}_\mathcal{C} \circ \Pi_d$ and show it equals $Fd$. (Hint: the comma category $(G \downarrow d)$ has a terminal object $(Fd, \varepsilon_d: GFd \to d)$; colimits over categories with a terminal object are evaluated at the terminal object.)

**Converse.** The converse also holds: if $\mathrm{Ran}_F \mathrm{id}_{\mathcal{D}}$ exists and is *preserved by $F$* (i.e. $F \circ \mathrm{Ran}_F \mathrm{id}_{\mathcal{D}} \cong \mathrm{id}_{\mathcal{D}}$ in the appropriate sense), then $\mathrm{Ran}_F \mathrm{id}_{\mathcal{D}}$ is a right adjoint to $F$. This gives a Kan-extension-based criterion for the existence of adjoints, complementing the adjoint functor theorems of [[concepts/category-theory/04-adjoint-functor-theorems-monads|File 4]].

### 5.3 The Yoneda Lemma and the Density Theorem

Let $y: \mathcal{C} \to [\mathcal{C}^{op}, \mathbf{Set}]$ denote the *Yoneda embedding* $c \mapsto \mathcal{C}(-, c)$ (as established in [[concepts/category-theory/02-adjoints-representables|File 2]], §4).

**Theorem (Density Theorem).** For any small category $\mathcal{C}$:

$$\mathrm{id}_{[\mathcal{C}^{op}, \mathbf{Set}]} \;\cong\; \mathrm{Lan}_y\, y.$$

That is, the identity functor on the presheaf category is (isomorphic to) the left Kan extension of the Yoneda embedding along itself.

*Equivalently:* every presheaf $P: \mathcal{C}^{op} \to \mathbf{Set}$ is canonically a colimit of representable presheaves:

$$P \;\cong\; \mathrm{colim}\bigl(y \circ \pi: \textstyle\int_\mathcal{C} P \to [\mathcal{C}^{op}, \mathbf{Set}]\bigr),$$

where $\int_\mathcal{C} P$ is the *category of elements* of $P$ (objects are pairs $(c, x)$ with $x \in P(c)$; morphisms $(c,x) \to (c', x')$ are $f: c \to c'$ with $P(f)(x') = x$) and $\pi: \int_\mathcal{C} P \to \mathcal{C}$ is the projection.

*Proof sketch.* The pointwise formula gives:

$$(\mathrm{Lan}_y\, y)(P) = \mathrm{colim}_{(y \downarrow P)} y \circ \Pi_P.$$

By the Yoneda lemma, an object of the comma category $(y \downarrow P)$ is a pair $(c, \alpha: y(c) \Rightarrow P)$, which by Yoneda corresponds to an element $x \in P(c)$. Thus $(y \downarrow P) \cong \int_\mathcal{C} P$ (category of elements). The colimit of $y \circ \Pi_P: \int_\mathcal{C} P \to [\mathcal{C}^{op}, \mathbf{Set}]$ is the colimit of representables indexed by elements of $P$, which is $P$ by the density theorem. $\square$

**The co-Yoneda lemma** (§7.3 below) gives the coend form:

$$P(c) \;\cong\; \int^{c' \in \mathcal{C}} \mathcal{C}(c', c) \times P(c'),$$

which says: the presheaf $P$ evaluated at $c$ is recovered as the coend of all hom-sets $\mathcal{C}(c', c)$ weighted by $P(c')$.

> [!NOTE] Riehl Exercise: Density theorem via the universal property
> Let $y: \mathcal{C} \to [\mathcal{C}^{op}, \mathbf{Set}]$ be the Yoneda embedding, and let $L = \mathrm{Lan}_y\, y$. Without using the pointwise formula, verify that $L \cong \mathrm{id}$ by showing: (a) any functor $G: [\mathcal{C}^{op}, \mathbf{Set}] \to [\mathcal{C}^{op}, \mathbf{Set}]$ with $G \circ y \cong y$ must receive a natural transformation $L \Rightarrow G$, and (b) the identity functor has this property with $\eta = \mathrm{id}_y$. Conclude $L \cong \mathrm{id}$.

---

## 6. Derived Functors as Kan Extensions 🧮

The framework of Kan extensions subsumes the classical theory of derived functors in homological algebra. This section sketches the connection; see Riehl §6.4 for a full treatment.

**Setup.** Let $\mathcal{A}$ be an abelian category with enough projectives (e.g. $\mathbf{Ab}$, $R\text{-}\mathbf{Mod}$). Let $F: \mathcal{A} \to \mathcal{B}$ be a *right-exact* functor. The *left derived functors* $L_n F: \mathcal{A} \to \mathcal{B}$ are defined classically by choosing a projective resolution $P_\bullet \xrightarrow{\sim} A$ and setting $L_n F(A) = H_n(F(P_\bullet))$.

**Kan extension formulation.** Let $\gamma: \mathcal{A} \to \mathbf{D}^-(\mathcal{A})$ be the *localization functor* from $\mathcal{A}$ to its bounded-above *derived category* (obtained by inverting quasi-isomorphisms). The total left derived functor $\mathbf{L}F: \mathbf{D}^-(\mathcal{A}) \to \mathbf{D}^-(\mathcal{B})$ is, when it exists, the left Kan extension of the composite $\mathcal{A} \xrightarrow{F} \mathcal{B} \xrightarrow{\gamma_\mathcal{B}} \mathbf{D}^-(\mathcal{B})$ along $\gamma_\mathcal{A}$:

$$\mathbf{L}F \;=\; \mathrm{Lan}_{\gamma_\mathcal{A}}\,(\gamma_\mathcal{B} \circ F).$$

The individual classical derived functors are recovered as $L_n F(A) = H_n(\mathbf{L}F(A))$.

**Key property.** The Kan extension $\mathbf{L}F$ is characterized by the fact that it *agrees with $F$ on projective objects*: if $P \in \mathcal{A}$ is projective, then $\gamma_\mathcal{A}(P)$ already has a projective resolution $P \xrightarrow{\mathrm{id}} P$ (a one-term resolution), so $(\mathbf{L}F)(\gamma P) \cong \gamma_\mathcal{B}(FP)$. This is the abstract counterpart to the classical fact that "derived functors of $F$ agree with $F$ on projective objects."

**Quillen's perspective.** In a *model category* $\mathcal{M}$ (Quillen 1967), the localization $\gamma: \mathcal{M} \to \mathrm{Ho}(\mathcal{M})$ inverts all *weak equivalences*. For a left Quillen functor $F: \mathcal{M} \to \mathcal{N}$, the total left derived functor $\mathbf{L}F: \mathrm{Ho}(\mathcal{M}) \to \mathrm{Ho}(\mathcal{N})$ is the left Kan extension of $\gamma_\mathcal{N} \circ F$ along $\gamma_\mathcal{M}$, computed by *cofibrant replacement*: $\mathbf{L}F(X) = \gamma_\mathcal{N}(F(QX))$ where $QX \xrightarrow{\sim} X$ is a cofibrant replacement.

> [!INFO] Grothendieck and Verdier's contribution
> Alexander Grothendieck and Jean-Louis Verdier introduced derived categories around 1960 as the correct framework for sheaf cohomology and duality. Verdier's key observation was that morphisms in the derived category are computed by a *calculus of fractions*, making the localization manageable. The Kan extension perspective — that derived functors are universal among functors inverting quasi-isomorphisms — formalizes the sense in which they are the "best possible" extensions of exact functors to the derived category.

> [!NOTE] Riehl Exercise: Derived functors via Kan extensions
> Let $\mathcal{A} = \mathbf{Ab}$ and $F = \mathbf{Hom}_{\mathbb{Z}}(A, -): \mathbf{Ab} \to \mathbf{Ab}$ for a fixed abelian group $A$. The classical right derived functors are $R^n F = \mathrm{Ext}^n_{\mathbb{Z}}(A, -)$. Identify the localization functor $\gamma: \mathbf{Ab} \to \mathbf{D}^+(\mathbf{Ab})$ and state (without full proof) how $\mathbf{R}F = \mathrm{Ran}_\gamma (\gamma \circ F)$ recovers the Ext groups. What property of $F$ ensures the right (rather than left) Kan extension is relevant?

> [!WARNING] Subtlety: existence vs. Kan extension
> A subtlety noted by Maltsiniotis and others: a functor can have a Kan extension along a localization without that Kan extension being the "correct" derived functor (it may fail to preserve quasi-isomorphisms on all objects). The correct derived functor requires the Kan extension to be *computed on resolutions*, which is an additional condition equivalent to requiring that the Kan extension is preserved by $F$ on cofibrant/fibrant objects.

---

## 7. Ends, Coends, and the Ninja Yoneda Lemma 🔬

Ends and coends are the "weighted" generalisations of limits and colimits that handle functors of mixed variance, $T: \mathcal{C}^{op} \times \mathcal{C} \to \mathcal{E}$. They are the natural language for the Kan extension coend formula of §3.4.

### 7.1 Ends

**Definition (Wedge).** Let $T: \mathcal{C}^{op} \times \mathcal{C} \to \mathcal{E}$ be a functor. A *wedge* from an object $e \in \mathcal{E}$ to $T$ is a family of morphisms $\{w_c: e \to T(c, c)\}_{c \in \mathcal{C}}$ such that for every morphism $f: c \to c'$ in $\mathcal{C}$, the following *dinatural* (diagonal naturality) condition holds:

$$T(\mathrm{id}_c, f) \circ w_c = T(f, \mathrm{id}_{c'}) \circ w_{c'}: e \to T(c, c').$$

That is, both composites $e \to T(c, c) \to T(c, c')$ and $e \to T(c', c') \to T(c, c')$ agree.

**Definition (End).** The *end* of $T$, written $\int_{c \in \mathcal{C}} T(c, c)$ or $\int_c T(c, c)$, is the *universal* wedge: an object $\int_c T(c,c) \in \mathcal{E}$ together with wedge maps $\{\pi_c: \int_c T(c,c) \to T(c,c)\}$ such that any other wedge $\{w_c: e \to T(c,c)\}$ factors uniquely through a map $e \to \int_c T(c,c)$.

When $\mathcal{C}$ is small and $\mathcal{E}$ is complete, $\int_c T(c,c)$ is computed as the *equaliser* of two maps between products:

$$\int_c T(c,c) = \mathrm{eq}\Biggl(\prod_{c \in \mathcal{C}} T(c,c) \rightrightarrows \prod_{f: c \to c' \in \mathcal{C}} T(c, c')\Biggr),$$

where the two maps send the component indexed by $c$ to the two morphisms $T(\mathrm{id},f)$ and $T(f, \mathrm{id})$.

### 7.2 Coends

**Definition (Coend).** The *coend* of $T: \mathcal{C}^{op} \times \mathcal{C} \to \mathcal{E}$, written $\int^{c \in \mathcal{C}} T(c,c)$ or $\int^c T(c,c)$, is the *initial* cowedge: an object $\int^c T(c,c)$ with morphisms $\{i_c: T(c,c) \to \int^c T(c,c)\}$ such that for every $f: c \to c'$:

$$i_{c'} \circ T(f, \mathrm{id}_{c'}) = i_c \circ T(\mathrm{id}_c, f): T(c, c') \to \int^c T(c,c),$$

and any other such cowedge factors uniquely through $\int^c T(c,c)$.

When $\mathcal{C}$ is small and $\mathcal{E}$ is cocomplete, $\int^c T(c,c)$ is computed as the *coequaliser*:

$$\mathrm{coeq}\Biggl(\coprod_{f: c \to c'} T(c, c') \rightrightarrows \coprod_{c} T(c,c)\Biggr) = \int^c T(c,c),$$

where the two maps are $T(\mathrm{id},f)$ and $T(f,\mathrm{id})$.

> [!INFO] Twisted arrow category
> The coend $\int^c T(c,c)$ is the colimit of $T$ over the *twisted arrow category* $\mathrm{tw}(\mathcal{C})$, whose objects are morphisms $f: c \to c'$ in $\mathcal{C}$ and whose morphisms $(f: c \to c') \to (g: d \to d')$ are pairs $(h: d \to c, k: c' \to d')$ with $g = k \circ f \circ h$. This makes the mixed-variance "twisting" precise.

### 7.3 The Ninja Yoneda Lemma

The *co-Yoneda lemma* (sometimes called the *ninja Yoneda lemma* in the coend-calculus literature) is the coend version of the Yoneda lemma.

**Theorem (Co-Yoneda / Ninja Yoneda Lemma).** For any functor $F: \mathcal{C} \to \mathcal{E}$ (with $\mathcal{C}$ small and $\mathcal{E}$ cocomplete) and any $c \in \mathcal{C}$:

$$\int^{c' \in \mathcal{C}} \mathcal{C}(c', c) \otimes Fc' \;\cong\; Fc,$$

naturally in $c$. Here $S \otimes e = \coprod_{s \in S} e$ is the copower of $e$ with a set $S$.

*Proof sketch.* This is exactly the left Kan extension formula with $K = \mathrm{id}_\mathcal{C}$, $d = c$: $(\mathrm{Lan}_{\mathrm{id}} F)(c) = \int^{c'} \mathcal{C}(c', c) \otimes Fc'$. But $\mathrm{Lan}_{\mathrm{id}} F \cong F$ (extending along the identity recovers the functor), so the coend equals $Fc$. $\square$

*Alternatively*, this follows directly from the ordinary Yoneda lemma: $[\mathcal{C}, \mathbf{Set}](y(c), F) \cong Fc$ where $y(c) = \mathcal{C}(-, c)$... but via coend calculus, the bijection is expressed as an isomorphism of coends.

> [!NOTE] Riehl Exercise: The co-Yoneda lemma
> Verify the co-Yoneda lemma directly for $\mathcal{E} = \mathbf{Set}$ (no copowers needed, as $S \otimes X = S \times X$ for sets). Show that the natural map $\int^{c'} \mathcal{C}(c', c) \times Fc' \to Fc$ sending $(f: c' \to c, x \in Fc')$ to $F(f)(x) \in Fc$ is an isomorphism. (Hint: construct the inverse using $x \in Fc \mapsto (\mathrm{id}_c, x) \in \mathcal{C}(c, c) \times Fc$, and check the coend relations.)

### 7.4 Natural Transformations as Ends

**Proposition.** For functors $F, G: \mathcal{C} \to \mathcal{D}$ (with $\mathcal{C}$ small), the set of natural transformations is computed as an end:

$$[\mathcal{C}, \mathcal{D}](F, G) \;=\; \int_{c \in \mathcal{C}} \mathcal{D}(Fc, Gc).$$

*Proof sketch.* A natural transformation $\alpha: F \Rightarrow G$ is a family $\{\alpha_c: Fc \to Gc\}_{c \in \mathcal{C}}$ such that for every $f: c \to c'$, the naturality square $G(f) \circ \alpha_c = \alpha_{c'} \circ F(f)$ commutes. This is precisely the condition for $\{\alpha_c\}$ to be a wedge into $T(c, c') = \mathcal{D}(Fc, Gc')$ (where $T(c,c) = \mathcal{D}(Fc, Gc)$). The universal such wedge is the set of natural transformations. $\square$

> [!TIP] The coend formula for Kan extensions, revisited
> Putting the pieces together: the left Kan extension formula $(\mathrm{Lan}_K F)(d) = \int^c \mathcal{D}(Kc, d) \otimes Fc$ now reads as "coend over $c$ of the copower of $Fc$ by the hom-set $\mathcal{D}(Kc, d)$." The dual end formula $(\mathrm{Ran}_K F)(d) = \int_c Fc^{\mathcal{D}(d, Kc)}$ reads as "end over $c$ of the power of $Fc$ by the hom-set $\mathcal{D}(d, Kc)$." These are the canonical expressions for Kan extensions in terms of the coend calculus.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Riehl, *Category Theory in Context*, Ch. 6](https://emilyriehl.github.io/files/context.pdf) | The primary source for this file; Chapter 6 "All Concepts are Kan Extensions" covers the full theory of Kan extensions including the pointwise formula, derived functors, and density theorem | https://emilyriehl.github.io/files/context.pdf |
| [Mac Lane, *Categories for the Working Mathematician*, Ch. X](https://link.springer.com/book/10.1007/978-1-4757-4721-8) | Original development of Kan extensions including the slogan "all concepts are Kan extensions"; Chapter X §7 is the locus classicus | https://link.springer.com/book/10.1007/978-1-4757-4721-8 |
| [Leinster, *Basic Category Theory*](https://arxiv.org/abs/1612.09375) | Concise treatment of category theory through Kan extensions; Chapter 6 covers Kan extensions in a streamlined style | https://arxiv.org/abs/1612.09375 |
| [nLab: Kan extension](https://ncatlab.org/nlab/show/Kan+extension) | Comprehensive reference with formal definitions, pointwise characterisations, and many examples | https://ncatlab.org/nlab/show/Kan+extension |
| [nLab: co-Yoneda lemma](https://ncatlab.org/nlab/show/co-Yoneda+lemma) | Precise statement and proof of the ninja Yoneda lemma in terms of coends | https://ncatlab.org/nlab/show/co-Yoneda+lemma |
| [Loregian, *Coend Calculus* (arXiv:1501.02503)](https://arxiv.org/abs/1501.02503) | Comprehensive reference for ends, coends, the Fubini theorem for coends, and the coend formula for Kan extensions | https://arxiv.org/abs/1501.02503 |
| [Lehner, "All Concepts are Kan Extensions" (Harvard Senior Thesis)](https://legacy-www.math.harvard.edu/theses/senior/lehner/lehner.pdf) | Expository treatment of Kan extensions focusing on their role as the universal concept in category theory | https://legacy-www.math.harvard.edu/theses/senior/lehner/lehner.pdf |
| [Keller, "Derived Categories and Their Uses"](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/keller.pdf) | Reference for derived categories, derived functors, and their relationship to Kan extensions via localization | https://webhomes.maths.ed.ac.uk/~v1ranick/papers/keller.pdf |
| [Maltsiniotis, "So, What is a Derived Functor?" (arXiv:1811.12255)](https://arxiv.org/abs/1811.12255) | Modern treatment clarifying the relationship between derived functors and Kan extensions, with subtleties about existence | https://arxiv.org/abs/1811.12255 |
| [nLab: density theorem](https://ncatlab.org/nlab/show/density+theorem) | Statement and proof of the density theorem (every presheaf is a colimit of representables) in Kan extension language | https://ncatlab.org/nlab/show/density+theorem |
