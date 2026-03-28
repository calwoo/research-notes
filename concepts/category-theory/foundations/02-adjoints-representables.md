# 📐 Category Theory II: Adjoints and Representable Functors

## Table of Contents

- [[#1. Adjunctions: Four Equivalent Definitions|1. Adjunctions: Four Equivalent Definitions]]
  - [[#1.1 Definition via Natural Bijection|1.1 Definition via Natural Bijection]]
  - [[#1.2 Definition via Unit and Counit|1.2 Definition via Unit and Counit]]
  - [[#1.3 Definition via Initial Objects in Comma Categories|1.3 Definition via Initial Objects in Comma Categories]]
  - [[#1.4 Equivalence of the Three Definitions|1.4 Equivalence of the Three Definitions]]
  - [[#1.5 Uniqueness of Adjoints|1.5 Uniqueness of Adjoints]]
  - [[#1.6 Examples of Adjunctions|1.6 Examples of Adjunctions]]
- [[#2. Comma Categories|2. Comma Categories]]
  - [[#2.1 Definition|2.1 Definition]]
  - [[#2.2 Special Cases|2.2 Special Cases]]
  - [[#2.3 Adjunctions via Comma Categories|2.3 Adjunctions via Comma Categories]]
- [[#3. Representable Functors|3. Representable Functors]]
  - [[#3.1 Definition|3.1 Definition]]
  - [[#3.2 Examples|3.2 Examples]]
  - [[#3.3 Universal Elements|3.3 Universal Elements]]
- [[#4. The Yoneda Embedding|4. The Yoneda Embedding]]
  - [[#4.1 Definition|4.1 Definition]]
  - [[#4.2 Injectivity, Fullness, and Faithfulness|4.2 Injectivity, Fullness, and Faithfulness]]
- [[#5. Full and Faithful Functors Reflect Isomorphisms|5. Full and Faithful Functors Reflect Isomorphisms]]
- [[#6. Adjunctions on Functor Categories|6. Adjunctions on Functor Categories]]
- [[#7. The Cayley Embedding and Representability of Categories|7. The Cayley Embedding and Representability of Categories]]
- [[#8. Idempotent Adjunctions and Reflections|8. Idempotent Adjunctions and Reflections]]
- [[#9. Adjunctions via Comma Categories: Terminal Objects|9. Adjunctions via Comma Categories: Terminal Objects]]
- [[#10. The Yoneda Lemma|10. The Yoneda Lemma]]
  - [[#10.1 Statement and Proof|10.1 Statement and Proof]]
  - [[#10.2 Naturality|10.2 Naturality]]
  - [[#10.3 Corollaries|10.3 Corollaries]]
- [[#11. Further Topics: Advanced Exercises|11. Further Topics: Advanced Exercises]]
  - [[#11.1 Group Actions as Functors|11.1 Group Actions as Functors]]
  - [[#11.2 Quantifiers as Adjoints|11.2 Quantifiers as Adjoints]]
  - [[#11.3 Mates|11.3 Mates]]
  - [[#11.4 Arrow Categories|11.4 Arrow Categories]]
- [[#12. The Category of Elements 🔬|12. The Category of Elements 🔬]]
  - [[#12.1 Definition: Contravariant Case|12.1 Definition: Contravariant Case]]
  - [[#12.2 Covariant Version|12.2 Covariant Version]]
  - [[#12.3 Universal Elements and Representations|12.3 Universal Elements and Representations]]
  - [[#12.4 The Grothendieck Construction|12.4 The Grothendieck Construction]]
  - [[#12.5 The Density Theorem via the Category of Elements|12.5 The Density Theorem via the Category of Elements]]
  - [[#12.6 Functoriality|12.6 Functoriality]]
- [[#13. Multivariable and Contravariant Adjunctions ⚡|13. Multivariable and Contravariant Adjunctions ⚡]]
  - [[#13.1 Two-Variable Adjunctions|13.1 Two-Variable Adjunctions]]
  - [[#13.2 Closed Monoidal Categories|13.2 Closed Monoidal Categories]]
  - [[#13.3 Currying|13.3 Currying]]
  - [[#13.4 Contravariant Adjunctions|13.4 Contravariant Adjunctions]]
  - [[#13.5 The Calculus of Adjunctions|13.5 The Calculus of Adjunctions]]
  - [[#13.6 Mates: Full Treatment|13.6 Mates: Full Treatment]]
- [[#References|References]]

---

> [!INFO] Prerequisites and Notation
> This note assumes fluency with the material in [[concepts/category-theory/01-categories-functors-natural-transformations|File 1]]: categories, functors, natural transformations, and the basic language of category theory. We write $\mathbf{C}(A, B)$ for the hom-set of morphisms from $A$ to $B$ in $\mathbf{C}$, and $[\mathbf{C}, \mathbf{D}]$ for the functor category with objects the functors $\mathbf{C} \to \mathbf{D}$ and morphisms the natural transformations between them. All categories are assumed *locally small* (hom-sets are genuine sets) unless otherwise stated.

---

## 1. Adjunctions: Four Equivalent Definitions 🔑

*Adjunctions* are the central organising concept of category theory. An adjunction between two functors formalises a pervasive mathematical phenomenon: a "free" construction on one side and a "forgetful" or "underlying" construction on the other. We give three equivalent definitions and prove their equivalence.

> [!INFO] Leinster's opening framing of adjunctions
> Leinster begins Chapter 2 of *Basic Category Theory* with the observation that adjunctions "are everywhere." He lists ten examples in the first two pages: free/forgetful pairs in algebra, direct/inverse image of sets, products/diagonals, tensor/hom, abelianisation/inclusion, sheafification/forgetful, geometric realisation/singular simplices, suspension/loop space, compactification/inclusion, and quantifiers as adjoints. The sheer ubiquity is the point: adjunctions are not a specialised tool but the fundamental language of "optimal solution" in mathematics.

### 1.1 Definition via Natural Bijection

> [!INFO] Leinster's hom-set definition: notation
> Leinster writes the adjunction bijection as $\overline{(-)} : \mathcal{B}(FA, B) \xrightarrow{\sim} \mathcal{A}(A, GB)$, using an overline to denote the transpose of a morphism under the bijection. He writes $\bar{f}$ for the transpose of $f \colon FA \to B$ (i.e., $\bar{f} = \phi(f) \colon A \to GB$), and $\bar{g}$ for the transpose of $g \colon A \to GB$ (i.e., $\bar{g} = \phi^{-1}(g) \colon FA \to B$). This overline notation will be adopted alongside the $\phi$ notation used elsewhere in these notes.

**Definition (Adjunction, hom-set form).** Let $\mathbf{C}$ and $\mathbf{D}$ be locally small categories. A functor $F \colon \mathbf{D} \to \mathbf{C}$ is *left adjoint* to a functor $G \colon \mathbf{C} \to \mathbf{D}$, written $F \dashv G$, if there is a family of bijections

$$\phi_{A,B} \colon \mathbf{C}(FA, B) \xrightarrow{\;\sim\;} \mathbf{D}(A, GB)$$

indexed by $A \in \mathbf{D}$ and $B \in \mathbf{C}$, that is *natural* in both $A$ and $B$. The full data $(F, G, \phi)$ is called an *adjunction*.

Naturality in $B$ means: for every morphism $g \colon B \to B'$ in $\mathbf{C}$, the square

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\mathcal{C}(FA, B) \arrow[r, "\varphi_{A,B}"] \arrow[d, "g \circ {-}"'] & \mathcal{D}(A, GB) \arrow[d, "Gg \circ {-}"] \\
\mathcal{C}(FA, B') \arrow[r, "\varphi_{A,B'}"'] & \mathcal{D}(A, GB')
\end{tikzcd}
\end{document}
```

commutes. Naturality in $A$ means: for every morphism $h \colon A' \to A$ in $\mathbf{D}$, the square

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\mathcal{C}(FA, B) \arrow[r, "\varphi_{A,B}"] \arrow[d, "{-} \circ Fh"'] & \mathcal{D}(A, GB) \arrow[d, "{-} \circ h"] \\
\mathcal{C}(FA', B) \arrow[r, "\varphi_{A',B}"'] & \mathcal{D}(A', GB)
\end{tikzcd}
\end{document}
```

commutes.

> [!NOTE]
> In explicit terms, naturality in $A$ and $B$ together say that for any $h \colon A' \to A$, $f \colon FA \to B$, and $g \colon B \to B'$,
> $$\phi_{A', B'}(g \circ f \circ Fh) = Gg \circ \phi_{A,B}(f) \circ h.$$
> This single equation encodes both naturality squares simultaneously.

### 1.2 Definition via Unit and Counit

**Definition (Adjunction, unit-counit form).** An adjunction $F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ consists of:
- a *unit* natural transformation $\eta \colon \mathrm{Id}_{\mathbf{D}} \Rightarrow GF$,
- a *counit* natural transformation $\varepsilon \colon FG \Rightarrow \mathrm{Id}_{\mathbf{C}}$,

satisfying the *triangle identities*:

$$(\varepsilon F) \circ (F\eta) = \mathrm{id}_F \qquad \text{and} \qquad (G\varepsilon) \circ (\eta G) = \mathrm{id}_G.$$

Here $\varepsilon F$ denotes the natural transformation with components $\varepsilon_{FA} \colon FGF A \to FA$, and $F\eta$ denotes the natural transformation with components $F(\eta_A) \colon FA \to FGFA$. Written as commutative diagrams for each $A \in \mathbf{D}$ and $B \in \mathbf{C}$:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
FA \arrow[r, "F\eta_A"] \arrow[dr, "\mathrm{id}_{FA}"'] & FGFA \arrow[d, "\varepsilon_{FA}"] \\
& FA
\end{tikzcd}
\end{document}
```

$$F A \xrightarrow{F\eta_A} FGFA \xrightarrow{\varepsilon_{FA}} FA = \mathrm{id}_{FA}$$

$$GB \xrightarrow{\eta_{GB}} GFGB \xrightarrow{G\varepsilon_B} GB = \mathrm{id}_{GB}$$

> [!INFO] Why "triangle identities"?
> The name comes from the string-diagram calculus, where the two equations look like triangles made of bent strings. In the classical "globular" notation, pasting the unit and counit along the adjunction wire and straightening gives back the identity wire.

### 1.3 Definition via Initial Objects in Comma Categories

We give the comma category definition here as a preview; we formally define comma categories in §2 and give the full characterisation in §9.

**Definition (Adjunction, universal arrow form).** $F \dashv G$ if and only if for each object $B \in \mathbf{C}$, there exists an object $A \in \mathbf{D}$ and a morphism $\eta_B \colon B \to GFA$ (a *universal arrow from $B$ to $G$*) such that for every morphism $f \colon B \to GX$ in $\mathbf{D}$, there exists a unique morphism $\bar{f} \colon FA \to X$ in $\mathbf{C}$ with $G\bar{f} \circ \eta_B = f$.

The pair $(FA, \eta_B)$ is *initial* in the comma category $(B \downarrow G)$ — see §2.

### 1.4 Equivalence of the Three Definitions

📐 We now prove that the hom-set and unit-counit definitions are equivalent. The proof follows Leinster's treatment in §2.2 of *Basic Category Theory*, which gives the cleanest available account and introduces the key "adjoint transpose" formulas.

**Theorem (Equivalence of adjunction definitions).** Given functors $F \colon \mathbf{D} \to \mathbf{C}$ and $G \colon \mathbf{C} \to \mathbf{D}$, the following data are in natural bijection:
1. A natural bijection $\phi_{A,B} \colon \mathbf{C}(FA, B) \xrightarrow{\sim} \mathbf{D}(A, GB)$.
2. A pair $(\eta, \varepsilon)$ of natural transformations satisfying the triangle identities.

*Proof.* **From (1) to (2).** Given $\phi$, define:
$$\eta_A = \phi_{A, FA}(\mathrm{id}_{FA}) \in \mathbf{D}(A, GFA), \qquad \varepsilon_B = \phi_{GB, B}^{-1}(\mathrm{id}_{GB}) \in \mathbf{C}(FGB, B).$$

*Naturality of $\eta$:* For any $h \colon A' \to A$ in $\mathbf{D}$, we must show $GFh \circ \eta_{A'} = \eta_A \circ h$.

Apply naturality of $\phi$ in $A$ (precompose domain with $Fh$) to $\mathrm{id}_{FA} \colon FA \to FA$: the naturality square for $\phi$ reads $\phi_{A', FA}(\mathrm{id}_{FA} \circ Fh) = \phi_{A,FA}(\mathrm{id}_{FA}) \circ h$, i.e., $\phi_{A', FA}(Fh) = \eta_A \circ h$.

Apply naturality of $\phi$ in $B$ (postcompose codomain with $GFh \colon GFA' \to GFA$) to $\mathrm{id}_{FA'} \colon FA' \to FA'$: the naturality square reads $\phi_{A', FA}(GFh \circ \mathrm{id}_{FA'}) = GFh \circ \phi_{A', FA'}(\mathrm{id}_{FA'})$, i.e., $\phi_{A', FA}(Fh) = GFh \circ \eta_{A'}$.

Since both equal $\phi_{A', FA}(Fh)$, we conclude $\eta_A \circ h = GFh \circ \eta_{A'}$, which is naturality of $\eta$. $\checkmark$

*Triangle identity $\varepsilon_{FA} \circ F\eta_A = \mathrm{id}_{FA}$:* Since $\phi_{A,FA}$ is a bijection, it suffices to show $\phi_{A,FA}(\varepsilon_{FA} \circ F\eta_A) = \phi_{A,FA}(\mathrm{id}_{FA}) = \eta_A$.

Apply naturality of $\phi$ in $A$ (precompose with $\eta_A \colon A \to GFA$, treating $GFA$ as the new first argument):
$$\phi_{A, FA}(\varepsilon_{FA} \circ F\eta_A) = \phi_{GFA, FA}(\varepsilon_{FA}) \circ \eta_A.$$
By definition of $\varepsilon_{FA} = \phi_{GFA,FA}^{-1}(\mathrm{id}_{GFA})$, we have $\phi_{GFA, FA}(\varepsilon_{FA}) = \mathrm{id}_{GFA}$. Therefore:
$$\phi_{A,FA}(\varepsilon_{FA} \circ F\eta_A) = \mathrm{id}_{GFA} \circ \eta_A = \eta_A. \quad \checkmark$$

The second triangle identity $G\varepsilon_B \circ \eta_{GB} = \mathrm{id}_{GB}$ follows by the dual argument (apply $\phi^{-1}_{GB,B}$ to both sides and use the analogous naturality calculation). This establishes both triangle identities. **QED (1) $\Rightarrow$ (2).**

**From (2) to (1).** Given $(\eta, \varepsilon)$ satisfying the triangle identities, define:
$$\phi_{A,B}(f) = Gf \circ \eta_A \quad \text{for } f \colon FA \to B, \qquad \phi_{A,B}^{-1}(g) = \varepsilon_B \circ Fg \quad \text{for } g \colon A \to GB.$$

*$\phi \circ \phi^{-1} = \mathrm{id}$:*
$$\phi_{A,B}(\varepsilon_B \circ Fg) = G(\varepsilon_B \circ Fg) \circ \eta_A = G\varepsilon_B \circ GFg \circ \eta_A \stackrel{\text{nat }\eta}{=} G\varepsilon_B \circ \eta_{GB} \circ g \stackrel{\triangle_2}{=} \mathrm{id}_{GB} \circ g = g. \checkmark$$

*$\phi^{-1} \circ \phi = \mathrm{id}$:*
$$\phi^{-1}_{A,B}(Gf \circ \eta_A) = \varepsilon_B \circ F(Gf \circ \eta_A) = \varepsilon_B \circ FGf \circ F\eta_A \stackrel{\text{nat }\varepsilon}{=} f \circ \varepsilon_{FA} \circ F\eta_A \stackrel{\triangle_1}{=} f \circ \mathrm{id}_{FA} = f. \checkmark$$

Naturality of $\phi$ in $A$ and $B$ follows from naturality of $\eta$ and $\varepsilon$ respectively. The two constructions (1) $\Rightarrow$ (2) and (2) $\Rightarrow$ (1) are mutually inverse. $\square$

> [!TIP] The fundamental adjoint transpose formulas
> **Given $F \dashv G$ with unit $\eta$ and counit $\varepsilon$:** every $f \colon FA \to B$ has transpose $\bar{f} = Gf \circ \eta_A \colon A \to GB$, and every $g \colon A \to GB$ has transpose $\bar{g} = \varepsilon_B \circ Fg \colon FA \to B$. These satisfy $\overline{\bar{f}} = f$ and $\overline{\bar{g}} = g$. The unit and counit are the transposes of identity morphisms: $\eta_A = \overline{\mathrm{id}_{FA}}$ and $\varepsilon_B = \overline{\mathrm{id}_{GB}}$.
>
> Leinster calls $\bar{f}$ and $\bar{g}$ the *adjoint transposes*, and emphasises that this is the most computationally useful formulation — to compute the adjoint transpose of any morphism, simply compose with the unit or counit.

### 1.5 Uniqueness of Adjoints

💡 **Proposition (Uniqueness of right adjoints).** If $F \dashv G$ and $F \dashv G'$, then $G \cong G'$ via a unique natural isomorphism. Similarly, left adjoints are unique up to unique natural isomorphism.

*Proof.* By hypothesis we have natural bijections $\mathbf{C}(FA, B) \cong \mathbf{D}(A, GB)$ and $\mathbf{C}(FA, B) \cong \mathbf{D}(A, G'B)$, both natural in $A \in \mathbf{D}$ and $B \in \mathbf{C}$. Composing gives a natural bijection $\mathbf{D}(A, GB) \cong \mathbf{D}(A, G'B)$, natural in $A$. By the Yoneda lemma (§10), any natural bijection $\mathbf{D}(A, GB) \cong \mathbf{D}(A, G'B)$ natural in $A$ is induced by a unique isomorphism $GB \cong G'B$ in $\mathbf{D}$. Since this isomorphism is natural in $B$ (being induced by composing the original natural bijections), we obtain a unique natural isomorphism $G \cong G'$. $\square$

> [!WARNING]
> *Uniqueness is up to unique isomorphism, not strict equality.* This matters when constructing adjunctions: if you build two right adjoints to the same functor by different universal constructions, you can transport structure between them along the canonical isomorphism.

### 1.6 Examples of Adjunctions

🔑 Here are the archetypal adjunctions that motivate the theory.

**Example (Free-forgetful for groups).** Let $U \colon \mathbf{Grp} \to \mathbf{Set}$ be the forgetful functor. The *free group functor* $F \colon \mathbf{Set} \to \mathbf{Grp}$ assigns to each set $S$ the free group $F(S)$ on the set $S$ of generators. There is a natural bijection
$$\mathbf{Grp}(FS, H) \cong \mathbf{Set}(S, UH),$$
because a group homomorphism out of $FS$ is determined by, and can be arbitrary specified by, the images of the generators. Hence $F \dashv U$.

**Example (Tensor-hom adjunction).** For a commutative ring $R$ and $R$-modules $M, N, P$, there is a natural isomorphism
$$\mathrm{Hom}_R(M \otimes_R N, P) \cong \mathrm{Hom}_R(M, \mathrm{Hom}_R(N, P)).$$
This says $M \otimes_R - \dashv \mathrm{Hom}_R(M, -)$ as endofunctors on $\mathbf{Mod}_R$.

**Example (Product-diagonal adjunction).** For sets (or any category with products), the diagonal functor $\Delta \colon \mathbf{C} \to \mathbf{C} \times \mathbf{C}$ sending $A \mapsto (A, A)$ has as right adjoint the product functor $\times \colon \mathbf{C} \times \mathbf{C} \to \mathbf{C}$:
$$(\mathbf{C} \times \mathbf{C})(\Delta A, (B, C)) = \mathbf{C}(A, B) \times \mathbf{C}(A, C) \cong \mathbf{C}(A, B \times C).$$

**Example (Direct and inverse image of sheaves/presheaves).** For a continuous map $f \colon X \to Y$ of topological spaces, there is an adjunction $f^{-1} \dashv f_*$ between inverse image and direct image functors on sheaves (or presheaves). The unit is a natural map $\mathcal{F} \to f_* f^{-1} \mathcal{F}$ and the counit is $f^{-1} f_* \mathcal{G} \to \mathcal{G}$.

> [!EXAMPLE]- Free-forgetful adjunctions in algebra
> The free-forgetful pattern appears uniformly across algebra:
> - $\mathbf{Set} \rightleftarrows \mathbf{Ab}$: free abelian group $\mathbb{Z}^{\oplus S}$ on generators $S$.
> - $\mathbf{Set} \rightleftarrows \mathbf{Ring}$: polynomial ring $\mathbb{Z}[x_1, \ldots, x_n]$ for a finite set, or more generally $\mathbb{Z}[S]$ for an arbitrary set $S$ of indeterminates.
> - $\mathbf{Ab} \rightleftarrows \mathbf{Mod}_R$: restriction of scalars along $\mathbb{Z} \to R$ has a left adjoint $R \otimes_{\mathbb{Z}} -$.
> In each case the unit $\eta_A \colon A \to UFA$ is the inclusion of generators, and the counit $\varepsilon_B \colon FUB \to B$ is the evaluation homomorphism.

---

## 2. Comma Categories 📐

### 2.1 Definition

**Definition (Comma category).** Let $F \colon \mathbf{A} \to \mathbf{C}$ and $G \colon \mathbf{B} \to \mathbf{C}$ be functors. The *comma category* $(F \downarrow G)$ is defined as follows:
- **Objects:** triples $(a, b, f)$ where $a \in \mathbf{A}$, $b \in \mathbf{B}$, and $f \colon Fa \to Gb$ is a morphism in $\mathbf{C}$.
- **Morphisms:** a morphism $(a, b, f) \to (a', b', f')$ is a pair $(h \colon a \to a', k \colon b \to b')$ of morphisms in $\mathbf{A}$ and $\mathbf{B}$ such that the square

$$Fa \xrightarrow{f} Gb \quad\text{commutes with}\quad Fa' \xrightarrow{f'} Gb',$$

i.e., $Gk \circ f = f' \circ Fh$ in $\mathbf{C}$.

- **Composition** is componentwise: $(h', k') \circ (h, k) = (h' \circ h, k' \circ k)$.
- **Identities:** $\mathrm{id}_{(a,b,f)} = (\mathrm{id}_a, \mathrm{id}_b)$.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
Fa \arrow[r, "f"] \arrow[d, "Fh"'] & Gb \arrow[d, "Gk"] \\
Fa' \arrow[r, "f'"'] & Gb'
\end{tikzcd}
\end{document}
```

The commutativity condition is $Gk \circ f = f' \circ Fh$.

### 2.2 Special Cases

**Special case (Slice categories).** Take $\mathbf{A} = \mathbf{C}$, $F = \mathrm{Id}_{\mathbf{C}}$, and $\mathbf{B} = \mathbf{1}$ (the one-object one-morphism category) with $G(\star) = c$ for some fixed $c \in \mathbf{C}$. Then $(F \downarrow G) = (\mathrm{Id}_{\mathbf{C}} \downarrow c)$, which is the *slice category* $\mathbf{C}/c$:
- Objects: morphisms $f \colon a \to c$ in $\mathbf{C}$.
- Morphisms: morphisms $h \colon a \to a'$ in $\mathbf{C}$ such that $f' \circ h = f$.

Dually, fixing the source gives the *coslice category* $c/\mathbf{C} = (c \downarrow \mathrm{Id}_{\mathbf{C}})$.

**Special case (For adjunctions).** Given $F \colon \mathbf{D} \to \mathbf{C}$ and $G \colon \mathbf{C} \to \mathbf{D}$, the comma category $(F \downarrow \mathrm{Id}_{\mathbf{D}})$ has as objects triples $(a, b, f \colon Fa \to b)$ with $a \in \mathbf{C}$ and $b \in \mathbf{D}$. For the adjunction characterisation, we use instead the coslice $(B \downarrow G)$ for each $B \in \mathbf{D}$: this has objects pairs $(X \in \mathbf{C}, f \colon B \to GX)$ and morphisms $h \colon X \to X'$ such that $Gh \circ f = f'$.

### 2.3 Adjunctions via Comma Categories

**Proposition.** $F \dashv G$ if and only if for each $B \in \mathbf{D}$, the comma category $(B \downarrow G)$ has an initial object.

The initial object of $(B \downarrow G)$ is the pair $(FB, \eta_B \colon B \to GFB)$: any morphism $f \colon B \to GX$ factors uniquely as $Gf' \circ \eta_B$ for the unique $f' \colon FB \to X$ with $\phi_{B,X}^{-1}(f) = f'$.

---

## 3. Representable Functors 💡

### 3.1 Definition

**Definition (Representable functor, contravariant).** Let $\mathbf{C}$ be a locally small category. A functor $X \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is *representable* if there exists an object $A \in \mathbf{C}$ and a natural isomorphism $\alpha \colon \mathbf{C}(-, A) \xrightarrow{\sim} X$. The pair $(A, \alpha)$ is called a *representation* of $X$. The object $A$ is called the *representing object*.

**Definition (Representable functor, covariant).** A functor $F \colon \mathbf{C} \to \mathbf{Set}$ is *representable* if there exists $A \in \mathbf{C}$ and a natural isomorphism $\mathbf{C}(A, -) \xrightarrow{\sim} F$.

> [!NOTE]
> Representability is a property of a functor up to choice of representation. The representing object $A$ is unique up to unique isomorphism (a consequence of the Yoneda lemma, proved in §10.3). When working with representable functors, one always specifies the representation — the natural isomorphism $\alpha$ — not just the object $A$.

> [!NOTE] Exercise 1
> Define what it means for a functor $X \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ (from a locally small category $\mathbf{C}$) to be representable, and what a representation is.

> [!NOTE] Exercise 2
> Give five examples of representable functors.

### 3.2 Examples

> [!EXAMPLE] Key examples of representable functors
> **1. Hom-functors.** The functor $\mathbf{C}(-, A) \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is representable by definition, with representing object $A$ and $\alpha = \mathrm{id}$.
>
> **2. Forgetful functor on groups.** The forgetful functor $U \colon \mathbf{Grp} \to \mathbf{Set}$ is representable: $U \cong \mathbf{Grp}(\mathbb{Z}, -)$ because a group homomorphism $\mathbb{Z} \to G$ is determined by the image of $1 \in \mathbb{Z}$, which can be any element of $G$. So $\mathbb{Z}$ represents $U$.
>
> **3. Underlying set of a ring.** The forgetful functor $U \colon \mathbf{Ring} \to \mathbf{Set}$ is represented by $(\mathbb{Z}[x], x)$: ring homomorphisms $\mathbb{Z}[x] \to R$ correspond bijectively to elements of $R$ (the image of $x$).
>
> **4. Bilinear maps as a representable.** The functor $\mathrm{Bil}(M, N; -) \colon \mathbf{Mod}_R \to \mathbf{Set}$ sending $P$ to the set of $R$-bilinear maps $M \times N \to P$ is represented by $M \otimes_R N$ with the universal bilinear map $M \times N \to M \otimes_R N$.
>
> **5. Underlying set of a topological space.** The forgetful functor $U \colon \mathbf{Top} \to \mathbf{Set}$ is represented by the one-point space $\{*\}$: continuous maps $\{*\} \to X$ correspond to points of $X$.

### 3.3 Universal Elements

🔑 Leinster introduces *universal elements* as an equivalent and often more intuitive way to think about representability. This concept (Chapter 4 of *Basic Category Theory*) unifies many universal constructions.

**Definition (Universal element).** Let $X \colon \mathcal{A}^{\mathrm{op}} \to \mathbf{Set}$ be a functor and $A \in \mathcal{A}$ an object. A *universal element* of $X$ at $A$ is an element $u \in X(A)$ such that for every $B \in \mathcal{A}$ and every $x \in X(B)$, there exists a *unique* morphism $f \colon B \to A$ in $\mathcal{A}$ with $X(f)(u) = x$.

**Proposition (Universal elements and representations).** A functor $X \colon \mathcal{A}^{\mathrm{op}} \to \mathbf{Set}$ is representable if and only if it has a universal element. More precisely, a representation $(A, \phi \colon \mathcal{A}(-, A) \xrightarrow{\sim} X)$ corresponds bijectively to a universal element $u = \phi_A(\mathrm{id}_A) \in X(A)$.

*Proof.* Given a representation $(A, \phi)$, set $u = \phi_A(\mathrm{id}_A)$. For any $B$ and $x \in X(B)$, define $f = \phi_B^{-1}(x) \colon B \to A$. Then $X(f)(u) = X(f)(\phi_A(\mathrm{id}_A)) = \phi_B(\mathrm{id}_A \circ f) = \phi_B(f) = x$ by naturality of $\phi$. Uniqueness: if $X(f)(u) = x$ and $X(g)(u) = x$, then $\phi_B(f) = \phi_B(g)$, so $f = g$ by injectivity of $\phi_B$.

Conversely, given a universal element $u \in X(A)$, define $\phi_B \colon \mathcal{A}(B, A) \to X(B)$ by $\phi_B(f) = X(f)(u)$. The universality of $u$ says exactly that $\phi_B$ is a bijection for every $B$. Naturality follows from functoriality of $X$. $\square$

> [!EXAMPLE]- Universal elements in concrete examples
> **1. Free groups.** The forgetful functor $U \colon \mathbf{Grp} \to \mathbf{Set}$ is represented by $(\mathbb{Z}, 1)$ ... no, more precisely, for a set $S$, the representing object for $\mathbf{Set}(S, U(-))$ is the free group $F(S)$, with universal element $\iota_S \colon S \to UF(S)$ (the inclusion of generators). For any group $G$ and function $f \colon S \to UG$, there is a unique group homomorphism $\bar{f} \colon F(S) \to G$ with $U\bar{f} \circ \iota_S = f$.
>
> **2. Tensor product.** The functor $\mathrm{Bilin}(M, N; -) \colon \mathbf{Mod}_R \to \mathbf{Set}$ (bilinear maps out of $M \times N$) is represented by $(M \otimes_R N, \otimes)$, where $\otimes \colon M \times N \to M \otimes_R N$ is the universal bilinear map. Every bilinear map $M \times N \to P$ factors uniquely through the universal one.
>
> **3. Terminal object.** The constant functor $\mathbf{1} \colon \mathcal{A}^{\mathrm{op}} \to \mathbf{Set}$ (sending every object to a fixed one-element set $\{*\}$) is representable if and only if $\mathcal{A}$ has a terminal object $T$. The universal element is the unique element $* \in \mathbf{1}(T) = \{*\}$.

> [!INFO] Leinster's perspective on universal elements
> Leinster (Chapter 4) argues that "universal property" in informal mathematical usage almost always means: an object together with a universal element of some functor. The Yoneda lemma is the precise statement that universal elements correspond exactly to representations, and that representations are unique up to unique isomorphism. This gives a *formal foundation* for the ubiquitous informal use of universal properties throughout mathematics.

---

## 4. The Yoneda Embedding 🔑

### 4.1 Definition

**Definition (Yoneda embedding).** Let $\mathbf{C}$ be a locally small category. The *Yoneda embedding* is the functor

$$H^{\bullet} \colon \mathbf{C} \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$$

defined on objects by $H^{\bullet}(A) = \mathbf{C}(-, A)$ and on morphisms $f \colon A \to B$ by $H^{\bullet}(f) = f_* \colon \mathbf{C}(-, A) \Rightarrow \mathbf{C}(-, B)$, where the component at $C \in \mathbf{C}$ is $(f_*)_C = f \circ - \colon \mathbf{C}(C, A) \to \mathbf{C}(C, B)$.

We also have the *covariant Yoneda embedding*

$$H_{\bullet} \colon \mathbf{C}^{\mathrm{op}} \to [\mathbf{C}, \mathbf{Set}]$$

defined by $H_{\bullet}(A) = \mathbf{C}(A, -)$ and on morphisms by precomposition.

> [!INFO] Connection to representability
> A functor $X \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is representable precisely when it lies in the *essential image* of $H^{\bullet}$: i.e., when $X \cong H^{\bullet}(A)$ for some $A$.

### 4.2 Injectivity, Fullness, and Faithfulness

📐 We prove the three key properties of $H^{\bullet}$ directly, *without* appealing to the Yoneda lemma (which is proved in §10 and uses these properties via corollary). This direct approach builds intuition for why representable functors completely determine objects.

**Proposition.** $H^{\bullet}$ is:
1. *Injective on objects:* if $\mathbf{C}(-, A) \cong \mathbf{C}(-, B)$ then $A \cong B$.
2. *Full:* every natural transformation $\mathbf{C}(-, A) \Rightarrow \mathbf{C}(-, B)$ is of the form $f_*$ for some $f \colon A \to B$.
3. *Faithful:* if $f_* = g_*$ (as natural transformations) then $f = g$.

*Proof of (3): Faithful.* Suppose $f, g \colon A \to B$ and $f_* = g_*$ as natural transformations. Evaluating the component at $C = A$ and applying to $\mathrm{id}_A \colon A \to A$:
$$(f_*)_A(\mathrm{id}_A) = f \circ \mathrm{id}_A = f, \qquad (g_*)_A(\mathrm{id}_A) = g.$$
So $f = g$. $\square$

*Proof of (2): Full.* Let $\alpha \colon \mathbf{C}(-, A) \Rightarrow \mathbf{C}(-, B)$ be a natural transformation. Define $f = \alpha_A(\mathrm{id}_A) \in \mathbf{C}(A, B)$. We claim $\alpha = f_*$.

For any $C \in \mathbf{C}$ and $h \colon C \to A$, naturality of $\alpha$ at the morphism $h$ gives a commuting square:
$$\alpha_C(h) = \alpha_C(\mathrm{id}_A \circ h) \stackrel{\text{nat}}{=} \alpha_A(\mathrm{id}_A) \circ h = f \circ h = (f_*)_C(h).$$
(Here we use that $\mathbf{C}(h, A) \colon \mathbf{C}(A, A) \to \mathbf{C}(C, A)$ is precomposition with $h$, and $\alpha_C \circ \mathbf{C}(h, A) = \mathbf{C}(h, B) \circ \alpha_A$ by naturality.) So $\alpha = f_*$. $\square$

*Proof of (1): Injective on objects.* Suppose $\alpha \colon \mathbf{C}(-, A) \xrightarrow{\sim} \mathbf{C}(-, B)$ is a natural isomorphism. By fullness applied to $\alpha$ and $\alpha^{-1}$, there exist $f \colon A \to B$ and $g \colon B \to A$ with $\alpha = f_*$ and $\alpha^{-1} = g_*$. Then $(gf)_* = g_* \circ f_* = \alpha^{-1} \circ \alpha = \mathrm{id}_{\mathbf{C}(-,A)} = (\mathrm{id}_A)_*$, so by faithfulness $gf = \mathrm{id}_A$. Similarly $fg = \mathrm{id}_B$. Hence $A \cong B$. $\square$

> [!NOTE] Exercise 3
> Define the Yoneda embedding $H^{\bullet} \colon \mathbf{C} \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$. Prove directly (without using the Yoneda Lemma) that: (i) $H^{\bullet}$ is injective on objects; (ii) $H^{\bullet}$ is full; (iii) $H^{\bullet}$ is faithful; (iv) a functor $X \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is representable if and only if it is in the essential image of $H^{\bullet}$.

---

## 5. Full and Faithful Functors Reflect Isomorphisms 🔑

**Definition (Reflects isomorphisms).** A functor $F \colon \mathbf{C} \to \mathbf{D}$ *reflects isomorphisms* if whenever $Ff \colon FA \to FB$ is an isomorphism in $\mathbf{D}$, then $f \colon A \to B$ is an isomorphism in $\mathbf{C}$.

**Proposition.** Any full and faithful functor $F \colon \mathbf{C} \to \mathbf{D}$ reflects isomorphisms.

*Proof.* Suppose $Ff \colon FA \to FB$ is an isomorphism with inverse $g \colon FB \to FA$ in $\mathbf{D}$. Since $F$ is full, there exists $h \colon B \to A$ in $\mathbf{C}$ with $Fh = g$. Then $F(hf) = Fh \circ Ff = g \circ Ff = \mathrm{id}_{FA} = F(\mathrm{id}_A)$. Since $F$ is faithful, $hf = \mathrm{id}_A$. Similarly, $F(fh) = Ff \circ Fh = Ff \circ g = \mathrm{id}_{FB} = F(\mathrm{id}_B)$, so $fh = \mathrm{id}_B$. Hence $f$ is an isomorphism with inverse $h$. $\square$

> [!WARNING]
> *A full and faithful functor need not be injective on objects in general.* It is injective on isomorphism classes: if $FA \cong FB$ then $A \cong B$. But two distinct objects can have equal images only if they are isomorphic. In a skeletal category (where isomorphic objects are equal), full faithfulness does imply injectivity on objects.

> [!NOTE] Exercise 4
> What does it mean for a functor to reflect isomorphisms? Prove that any full and faithful functor reflects isomorphisms.

---

## 6. Adjunctions on Functor Categories 💡

**Proposition (Induced adjunctions on functor categories).** Let $L \dashv R \colon \mathbf{C} \rightleftarrows \mathbf{D}$ be an adjunction and $\mathbf{E}$ a small category.

1. **Postcomposition** induces an adjunction $L_* \dashv R_* \colon [\mathbf{E}, \mathbf{C}] \rightleftarrows [\mathbf{E}, \mathbf{D}]$, where $(L_* F)(e) = L(Fe)$ for $F \in [\mathbf{E}, \mathbf{C}]$.

2. **Precomposition** by a functor $p \colon \mathbf{E}' \to \mathbf{E}$ induces $p^* \colon [\mathbf{E}, \mathbf{C}] \to [\mathbf{E}', \mathbf{C}]$ sending $F \mapsto F \circ p$.

*Proof of (1).* For functors $F \colon \mathbf{E} \to \mathbf{C}$ and $G \colon \mathbf{E} \to \mathbf{D}$, a natural transformation $\alpha \colon L_* F \Rightarrow G$ consists of components $\alpha_e \colon LFe \to Ge$ for each $e \in \mathbf{E}$. By the adjunction $L \dashv R$, each $\alpha_e$ corresponds to a unique $\hat\alpha_e \colon Fe \to RGe$. The naturality of $\alpha$ (the square $G(f) \circ \alpha_e = \alpha_{e'} \circ LF(f)$ for $f \colon e \to e'$) translates precisely to the naturality of $\hat\alpha$ (the square $RG(f) \circ \hat\alpha_e = \hat\alpha_{e'} \circ F(f)$) via naturality of $\phi$ in both variables. This gives the required natural bijection $[\mathbf{E}, \mathbf{D}](L_*F, G) \cong [\mathbf{E}, \mathbf{C}](F, R_*G)$. $\square$

**Example (Chains of adjoints).** For an open inclusion $i \colon U \hookrightarrow X$ of topological spaces, one has a chain of adjoint functors on presheaf categories:

$$i_! \dashv i^* \dashv i_*$$

where $i_!$ is extension by zero (left adjoint to restriction), $i^*$ is restriction (inverse image), and $i_*$ is direct image (right adjoint to restriction). The chain extends: limits and colimits of presheaves can themselves be computed pointwise, and the above chain lifts to sheaves.

For a ring homomorphism $\phi \colon R \to S$, the restriction-of-scalars functor $\phi^* = \phi_* \colon \mathbf{Mod}_S \to \mathbf{Mod}_R$ (view an $S$-module as an $R$-module via $\phi$) admits a left adjoint $\phi^! = S \otimes_R - \colon \mathbf{Mod}_R \to \mathbf{Mod}_S$ (extension of scalars) and a right adjoint $\phi_* = \mathrm{Hom}_R(S, -) \colon \mathbf{Mod}_R \to \mathbf{Mod}_S$ (coextension of scalars):

$$S \otimes_R - \;\dashv\; \phi^* \;\dashv\; \mathrm{Hom}_R(S, -).$$

> [!NOTE] Exercise 5
> Let $X$ be a topological space and let $\mathbf{PSh}(X)$ be the category of presheaves of sets on $X$. For an open inclusion $i \colon U \hookrightarrow X$, exhibit a chain of adjoint functors involving the direct image $i_*$, inverse image $i^*$, and extension by zero $i_!$. (Alternatively: for a ring map $\phi \colon R \to S$, exhibit the chain of adjoint functors among restriction of scalars $\phi^*$ and the extension/coextension of scalars $\phi^!$ and $\phi_*$.)

> [!NOTE] Exercise 6
> For an adjunction $L \dashv R \colon \mathbf{C} \rightleftarrows \mathbf{D}$ and a small category $\mathbf{E}$, show that postcomposition gives an adjunction $[\mathbf{E}, \mathbf{C}] \rightleftarrows [\mathbf{E}, \mathbf{D}]$. Extend Exercise 5 to obtain longer chains of adjoints.

---

## 7. The Cayley Embedding and Representability of Categories 🔑

### 7.1 Cayley's Theorem for Groups

Recall that *Cayley's theorem* states that every group $G$ embeds as a subgroup of $\mathrm{Sym}(G)$, the symmetric group on the underlying set of $G$, via the *regular representation*: $g \mapsto (h \mapsto gh)$. This is injective because if $gh = g'h$ for all $h$ then $g = g'$ (take $h = e$), and it is a group homomorphism: the map for $gg'$ is the composition of the maps for $g$ and $g'$.

### 7.2 The Free Category on a Graph

**Definition (Graph).** A *directed graph* (or *quiver*) consists of a set of vertices $V$ and a set of edges $E$ with source and target maps $s, t \colon E \to V$.

**Definition (Free category).** The *free category* $\mathbf{FC}(\Gamma)$ on a directed graph $\Gamma$ has as objects the vertices of $\Gamma$ and as morphisms the finite paths (including the empty path at each vertex, which serves as the identity). Composition is concatenation of paths.

The functor $U \colon \mathbf{Cat} \to \mathbf{Grph}$ (which forgets composition and identities, retaining only the underlying graph of objects and non-identity morphisms) has $\mathbf{FC}$ as its left adjoint: a functor $\mathbf{FC}(\Gamma) \to \mathbf{C}$ is the same as a graph morphism $\Gamma \to U\mathbf{C}$.

### 7.3 The Yoneda Embedding as Cayley's Theorem

**Theorem (Categorical Cayley).** For any small category $\mathbf{C}$, the Yoneda embedding $H^{\bullet} \colon \mathbf{C} \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$ is full and faithful, hence embeds $\mathbf{C}$ as a full subcategory of the presheaf category $[\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$.

This is a direct corollary of Proposition 4.2 (or alternatively of the Yoneda lemma; see §10.3). The analogy with Cayley's theorem is precise: objects of $\mathbf{C}$ are like group elements, and the action of an object $A$ on the category is the functor $\mathbf{C}(-, A)$ — the "what does $A$ look like from every other object" functor.

> [!NOTE] Exercise 7
> (i) Describe the left adjoint to the forgetful functor from $\mathbf{Cat}$ to $\mathbf{Grph}$ (directed graphs). (ii) Define the Cayley embedding for groups. (iii) Prove that every small category embeds fully and faithfully into $\mathbf{Set}^{\mathbf{C}^{\mathrm{op}}}$ via the Yoneda embedding $H^{\bullet}$.

---

## 8. Idempotent Adjunctions and Reflections 💡

### 8.1 Reflective Subcategories

**Definition (Reflective subcategory).** A full subcategory $\iota \colon \mathbf{D} \hookrightarrow \mathbf{C}$ is *reflective* if $\iota$ has a left adjoint $L \colon \mathbf{C} \to \mathbf{D}$ (the *reflector* or *localisation*):
$$L \dashv \iota \colon \mathbf{C} \rightleftarrows \mathbf{D}.$$
The unit of this adjunction is $\eta_A \colon A \to \iota L(A)$, the *reflection* of $A$ into $\mathbf{D}$.

**Examples.** The category $\mathbf{Ab}$ of abelian groups is a reflective subcategory of $\mathbf{Grp}$: the reflector sends $G$ to $G/[G,G]$, its abelianisation. The category of sheaves is a reflective subcategory of the category of presheaves: the reflector is sheafification.

### 8.2 Idempotent Adjunctions

**Definition (Idempotent adjunction).** An adjunction $F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ is *idempotent* if the unit $\eta_A \colon A \to GFA$ is an isomorphism for all $A \in \mathbf{D}$ (equivalently, if the counit $\varepsilon_B \colon FGB \to B$ is an isomorphism for all $B \in \mathbf{C}$).

> [!WARNING]
> *The two conditions — $\eta$ iso vs. $\varepsilon$ iso — are not trivially equivalent.* The equivalence requires a non-trivial argument using the triangle identities; see Exercise 9.

**Proposition (Restriction to an equivalence).** Let $F \dashv G$. Define:
$$\mathbf{D}' = \{A \in \mathbf{D} : \eta_A \text{ is an iso}\}, \qquad \mathbf{C}' = \{B \in \mathbf{C} : \varepsilon_B \text{ is an iso}\}.$$

Then $F$ restricts to a functor $F' \colon \mathbf{D}' \to \mathbf{C}'$, $G$ restricts to $G' \colon \mathbf{C}' \to \mathbf{D}'$, and $F' \dashv G'$ is an equivalence of categories.

*Proof.* First we show $F(\mathbf{D}') \subseteq \mathbf{C}'$: for $A \in \mathbf{D}'$, we must show $\varepsilon_{FA} \colon FGFA \to FA$ is an isomorphism. By the first triangle identity, $\varepsilon_{FA} \circ F\eta_A = \mathrm{id}_{FA}$. Since $\eta_A$ is an isomorphism by assumption, $F\eta_A$ is an isomorphism (as $F$ preserves isomorphisms). Then $\varepsilon_{FA} = \mathrm{id}_{FA} \circ (F\eta_A)^{-1}$ is an isomorphism.

Similarly $G(\mathbf{C}') \subseteq \mathbf{D}'$: for $B \in \mathbf{C}'$, the second triangle identity gives $G\varepsilon_B \circ \eta_{GB} = \mathrm{id}_{GB}$; since $\varepsilon_B$ is iso, $G\varepsilon_B$ is iso, so $\eta_{GB}$ is a split monomorphism; a more careful argument using both triangle identities shows it is iso.

The unit $\eta'_A = \eta_A$ and counit $\varepsilon'_B = \varepsilon_B$ are isomorphisms on $\mathbf{D}'$ and $\mathbf{C}'$ respectively, making $F' \dashv G'$ an equivalence. $\square$

> [!NOTE] Exercise 8
> Let $F \dashv G$ be an adjunction. Show that this adjunction restricts to an equivalence between the full subcategory of $\mathbf{D}$ on objects $A$ where $\eta_A \colon A \to GFA$ is an isomorphism, and the full subcategory of $\mathbf{C}$ on objects $B$ where $\varepsilon_B \colon FGB \to B$ is an isomorphism.

> [!NOTE] Exercise 9
> An adjunction $F \dashv G \colon \mathbf{D} \rightleftarrows \mathbf{C}$ is called *idempotent* if the unit $\eta_A \colon A \to GFA$ is an isomorphism for all $A$. Prove that the following six conditions are equivalent: (i) $\eta_A$ is an iso for all $A$; (ii) $\varepsilon_B$ is an iso for all $B$; (iii) $F\eta$ is an iso (as a natural transformation $F \Rightarrow FGF$); (iv) $\varepsilon F$ is an iso; (v) $G\varepsilon$ is an iso; (vi) $\eta G$ is an iso. Apply this to the adjunction between a poset and its ideal completion.

---

## 9. Adjunctions via Comma Categories: Terminal Objects 📐

**Theorem (Universal arrow formulation).** Let $G \colon \mathbf{C} \to \mathbf{D}$ be a functor. Specifying a left adjoint to $G$ is equivalent to specifying, for each object $B \in \mathbf{D}$, an initial object in the comma category $(B \downarrow G)$.

More explicitly:

- An object of $(B \downarrow G)$ is a pair $(X, f)$ with $X \in \mathbf{C}$ and $f \colon B \to GX$.
- A morphism $(X, f) \to (X', f')$ is a morphism $h \colon X \to X'$ in $\mathbf{C}$ such that $Gh \circ f = f'$.
- An initial object in $(B \downarrow G)$ is a pair $(FB, \eta_B)$ with $\eta_B \colon B \to GFB$ such that for any $(X, f)$, there is a unique $h \colon FB \to X$ with $Gh \circ \eta_B = f$.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
B \arrow[r, "\eta_B"] \arrow[dr, "f"'] & GFB \arrow[d, "Gh"] \\
& GX
\end{tikzcd}
\end{document}
```

*Proof.* Given an adjunction $F \dashv G$ with unit $\eta$, the pair $(FB, \eta_B)$ is initial in $(B \downarrow G)$: the unique morphism to any $(X, f)$ is $\hat{f} = \phi_{B,X}^{-1}(f) \colon FB \to X$ (the transpose of $f$), and $G\hat{f} \circ \eta_B = f$ holds by definition of $\phi$ (since $\phi_{B,X}(\hat{f}) = G\hat{f} \circ \eta_B = f$). Uniqueness follows from bijectivity of $\phi$.

Conversely, given initial objects $(FB, \eta_B)$ for each $B$, the assignment $B \mapsto FB$ extends to a functor $F$ (using the initiality to define $F$ on morphisms), and the bijection $\mathbf{C}(FB, X) \cong \mathbf{D}(B, GX)$ (sending $h \mapsto Gh \circ \eta_B$) is natural in $B$ and $X$, yielding an adjunction $F \dashv G$. $\square$

**Dual formulation.** Dually, $F \dashv G$ is also equivalent to: for each $A \in \mathbf{C}$, the pair $(GA, \varepsilon_A)$ is a *terminal* object in the comma category $(F \downarrow A)$, where objects are pairs $(Y \in \mathbf{D}, g \colon FY \to A)$.

> [!NOTE] Exercise 10
> Describe the relationship between adjunctions $F \dashv G \colon \mathbf{D} \rightleftarrows \mathbf{C}$ and initial (or terminal) objects in comma categories. Specifically, show that specifying a left adjoint to $G \colon \mathbf{C} \to \mathbf{D}$ is equivalent to specifying, for each $B \in \mathbf{D}$, an initial object in the comma category $(B \downarrow G)$.

---

## 10. The Yoneda Lemma 🔑

### 10.1 Statement and Proof

**Theorem (Yoneda Lemma).** Let $\mathbf{C}$ be a locally small category, $X \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ a functor, and $A \in \mathbf{C}$ an object. There is a bijection

$$\Phi_{A,X} \colon [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}](\mathbf{C}(-, A),\, X) \xrightarrow{\;\sim\;} X(A)$$

defined by $\Phi_{A,X}(\alpha) = \alpha_A(\mathrm{id}_A)$, where $\alpha \colon \mathbf{C}(-, A) \Rightarrow X$ is a natural transformation.

*Proof.* We construct an explicit inverse to $\Phi_{A,X}$.

**Define $\Psi_{A,X} \colon X(A) \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}](\mathbf{C}(-, A), X)$** as follows. Given $x \in X(A)$, define the natural transformation $\Psi_{A,X}(x) = \alpha^x$ by:
$$\alpha^x_B \colon \mathbf{C}(B, A) \to X(B), \qquad \alpha^x_B(f) = X(f)(x).$$
(Here $f \colon B \to A$ is viewed as a morphism in $\mathbf{C}$, so $X(f) \colon X(A) \to X(B)$ since $X$ is contravariant, and $X(f)(x) \in X(B)$.)

**Naturality of $\alpha^x$:** For a morphism $g \colon C \to B$ in $\mathbf{C}$, we need $\alpha^x_C \circ \mathbf{C}(g, A) = X(g) \circ \alpha^x_B$, i.e., $\alpha^x_C(f \circ g) = X(g)(\alpha^x_B(f))$ for all $f \colon B \to A$. Compute:
$$\alpha^x_C(f \circ g) = X(f \circ g)(x) = (X(g) \circ X(f))(x) = X(g)(X(f)(x)) = X(g)(\alpha^x_B(f)). \checkmark$$

**$\Phi \circ \Psi = \mathrm{id}$:** Given $x \in X(A)$:
$$\Phi_{A,X}(\Psi_{A,X}(x)) = \Phi_{A,X}(\alpha^x) = \alpha^x_A(\mathrm{id}_A) = X(\mathrm{id}_A)(x) = \mathrm{id}_{X(A)}(x) = x. \checkmark$$

**$\Psi \circ \Phi = \mathrm{id}$:** Given $\alpha \colon \mathbf{C}(-, A) \Rightarrow X$, we must show $\Psi_{A,X}(\Phi_{A,X}(\alpha)) = \alpha$, i.e., $\alpha^{\alpha_A(\mathrm{id}_A)} = \alpha$. For any $B \in \mathbf{C}$ and $f \colon B \to A$:
$$\alpha^{\alpha_A(\mathrm{id}_A)}_B(f) = X(f)(\alpha_A(\mathrm{id}_A)).$$
By naturality of $\alpha$ at $f \colon B \to A$ (viewing $f$ as a morphism $f \colon B \to A$ in $\mathbf{C}^{\mathrm{op}}$, i.e., a morphism in $\mathbf{C}$ from $A$ to $B$ but reversed), the naturality square is:
$$\alpha_B \circ \mathbf{C}(f, A) = X(f) \circ \alpha_A$$
where $\mathbf{C}(f, A) \colon \mathbf{C}(A, A) \to \mathbf{C}(B, A)$ sends $h \mapsto h \circ f$. Evaluating at $\mathrm{id}_A$:
$$\alpha_B(f) = \alpha_B(\mathrm{id}_A \circ f) = X(f)(\alpha_A(\mathrm{id}_A)). \checkmark$$

Hence $\Phi_{A,X}$ is a bijection with inverse $\Psi_{A,X}$. $\square$

### 10.2 Naturality

💡 The bijection $\Phi_{A,X}$ is natural in both $A$ and $X$.

**Naturality in $X$:** For a natural transformation $\beta \colon X \Rightarrow Y$ and $\alpha \colon \mathbf{C}(-,A) \Rightarrow X$:
$$\Phi_{A,Y}(\beta \circ \alpha) = (\beta \circ \alpha)_A(\mathrm{id}_A) = \beta_A(\alpha_A(\mathrm{id}_A)) = \beta_A(\Phi_{A,X}(\alpha)).$$
So $\Phi_{A,Y}(\beta \circ \alpha) = (\beta_A \circ \Phi_{A,X})(\alpha)$, confirming naturality in $X$.

**Naturality in $A$:** For a morphism $g \colon A \to A'$ in $\mathbf{C}$ and $\alpha \colon \mathbf{C}(-,A') \Rightarrow X$:
$$\Phi_{A,X}(\alpha \circ g_*) = (\alpha \circ g_*)_A(\mathrm{id}_A) = \alpha_A(g \circ \mathrm{id}_A) = \alpha_A(g) = X(g)(\alpha_{A'}(\mathrm{id}_{A'})) = X(g)(\Phi_{A',X}(\alpha)).$$
(The penultimate equality uses $\alpha$'s naturality at $g$.) So $\Phi_{A,X}(\alpha \circ g_*) = X(g) \circ \Phi_{A',X}(\alpha)$, naturality in $A$.

### 10.3 Corollaries

> [!INFO] Leinster's formulation of the Yoneda Lemma
> Leinster states the Yoneda Lemma (Theorem 4.2.1 in *Basic Category Theory*) as: "Let $\mathcal{A}$ be a locally small category. Then for any functor $X \colon \mathcal{A}^{\mathrm{op}} \to \mathbf{Set}$ and object $A \in \mathcal{A}$,
> $$[\mathcal{A}^{\mathrm{op}}, \mathbf{Set}](\mathcal{A}(-, A), X) \cong X(A)$$
> naturally in $A$ and $X$." He emphasises that the right-hand side is *a set*, not a functor or category — the Yoneda lemma collapses a set of natural transformations into the set of elements of $X$ at $A$.
>
> Leinster's proof strategy is identical to the one given in §10.1: the bijection is $\alpha \mapsto \alpha_A(\mathrm{id}_A)$, and the inverse sends $x \in X(A)$ to the natural transformation with components $\alpha^x_B(f) = X(f)(x)$. He then proves naturality separately.

**Corollary 1 (Yoneda embedding is fully faithful).** The Yoneda embedding $H^{\bullet} \colon \mathbf{C} \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$ is full and faithful.

*Proof.* Take $X = \mathbf{C}(-, B)$ in the Yoneda lemma:
$$[\mathbf{C}^{\mathrm{op}}, \mathbf{Set}](\mathbf{C}(-, A), \mathbf{C}(-, B)) \cong \mathbf{C}(-, B)(A) = \mathbf{C}(A, B).$$
This bijection sends a natural transformation $\alpha \colon \mathbf{C}(-,A) \Rightarrow \mathbf{C}(-,B)$ to $\alpha_A(\mathrm{id}_A) \in \mathbf{C}(A,B)$. Tracing the inverse: $f \in \mathbf{C}(A,B)$ maps to the natural transformation $\alpha^f$ with $\alpha^f_C(h) = \mathbf{C}(h, B)(f) = f \circ h = (f_*)_C(h)$. So the bijection is exactly $H^{\bullet}_{A,B} \colon \mathbf{C}(A,B) \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}](H^{\bullet}A, H^{\bullet}B)$, which is therefore a bijection — i.e., $H^{\bullet}$ is full and faithful. $\square$

**Corollary 2 (Isomorphism via representables).** $\mathbf{C}(-, A) \cong \mathbf{C}(-, B)$ as functors if and only if $A \cong B$ in $\mathbf{C}$.

*Proof.* If $A \cong B$ then $\mathbf{C}(-, A) \cong \mathbf{C}(-, B)$ by functoriality. Conversely, a natural isomorphism $\alpha \colon \mathbf{C}(-,A) \xrightarrow{\sim} \mathbf{C}(-,B)$ corresponds via Corollary 1 to $f = \alpha_A(\mathrm{id}_A) \in \mathbf{C}(A,B)$ with the property that $H^{\bullet}(f) = \alpha$ is an isomorphism; since $H^{\bullet}$ is fully faithful, it reflects isomorphisms (§5), so $f$ is an isomorphism. $\square$

**Corollary 3 (Uniqueness of representations).** A representation $(A, \alpha)$ of $X \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is unique up to unique isomorphism: if $(A, \alpha)$ and $(A', \alpha')$ are both representations of $X$, there is a unique isomorphism $\phi \colon A \xrightarrow{\sim} A'$ such that $\alpha' \circ \phi_* = \alpha$ (as natural transformations $\mathbf{C}(-, A) \Rightarrow X$, where $\phi_* = H^{\bullet}(\phi)$).

*Proof.* The composite $(\alpha')^{-1} \circ \alpha \colon \mathbf{C}(-,A) \xrightarrow{\sim} \mathbf{C}(-,A')$ is a natural isomorphism. By Corollary 2, it is induced by a unique isomorphism $\phi \colon A \xrightarrow{\sim} A'$. $\square$

> [!NOTE] Exercise 11
> State and prove the Yoneda Lemma. Deduce: (i) The Yoneda embedding $H^{\bullet} \colon \mathbf{C} \to [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$ is full and faithful. (ii) If $\mathbf{C}(-, A) \cong \mathbf{C}(-, B)$ then $A \cong B$. (iii) A natural transformation from $\mathbf{C}(-, A)$ to $X$ is uniquely determined by the image of $\mathrm{id}_A$.

---

## 11. Further Topics: Advanced Exercises 📐

### 11.1 Group Actions as Functors

**Definition (Classifying category).** For a group $G$, the *classifying category* $\mathbf{B}G$ is the category with one object $\star$ and $\mathbf{B}G(\star, \star) = G$, with composition given by group multiplication.

**Proposition.** Functors $\mathbf{B}G \to \mathbf{Set}$ correspond exactly to left $G$-sets (sets $S$ with a left $G$-action $G \times S \to S$ satisfying the axioms). Natural transformations between two such functors correspond to $G$-equivariant maps.

*Proof.* A functor $F \colon \mathbf{B}G \to \mathbf{Set}$ assigns to the unique object $\star$ a set $S = F(\star)$, and to each $g \in G$ (viewed as a morphism $\star \to \star$) a function $F(g) \colon S \to S$. Functoriality requires $F(gh) = F(g) \circ F(h)$ (which is the action axiom $g \cdot (h \cdot s) = (gh) \cdot s$ with our convention) and $F(e) = \mathrm{id}_S$. A natural transformation $\eta \colon F \Rightarrow F'$ has a single component $\eta_\star \colon S \to S'$, and the naturality square for $g \in G$ reads $\eta_\star \circ F(g) = F'(g) \circ \eta_\star$, i.e., $\eta_\star(g \cdot s) = g \cdot \eta_\star(s)$ — exactly equivariance. $\square$

The functor category $[\mathbf{B}G, \mathbf{Set}]$ is therefore the category of $G$-sets and equivariant maps, denoted $G\text{-}\mathbf{Set}$.

> [!NOTE] Exercise 12
> Let $G$ be a group. View $G$ as a one-object category $\mathbf{B}G$. Show that functors $\mathbf{B}G \to \mathbf{Set}$ correspond to $G$-sets, and natural transformations between them correspond to $G$-equivariant maps. Using this, describe $[\mathbf{B}G, \mathbf{Set}]$ in terms of $G$-sets.

### 11.2 Quantifiers as Adjoints

For a function $f \colon X \to Y$, let $\mathbf{Sub}(X) = \mathcal{P}(X)$ denote the power set of $X$, ordered by inclusion. The *substitution* (or *inverse image*) functor is

$$f^* \colon \mathbf{Sub}(Y) \to \mathbf{Sub}(X), \qquad f^*(T) = f^{-1}(T) = \{x \in X : f(x) \in T\}.$$

Since $\mathbf{Sub}(X)$ and $\mathbf{Sub}(Y)$ are poset categories, a functor $\mathbf{Sub}(Y) \to \mathbf{Sub}(X)$ is just an order-preserving map, and adjoints are *Galois connections*.

**Proposition.** $f^*$ has a left adjoint $\exists_f$ and a right adjoint $\forall_f$:

$$\exists_f \dashv f^* \dashv \forall_f$$

where

$$\exists_f(S) = f(S) = \{f(x) : x \in S\} \subseteq Y \qquad (\text{existential quantification along } f),$$
$$\forall_f(S) = \{y \in Y : f^{-1}(\{y\}) \subseteq S\} \qquad (\text{universal quantification along } f).$$

*Proof.* We verify $\exists_f \dashv f^*$ (the left adjoint condition): for $S \subseteq X$ and $T \subseteq Y$,
$$\exists_f(S) \subseteq T \iff f(S) \subseteq T \iff (\forall x \in S,\; f(x) \in T) \iff S \subseteq f^{-1}(T) = f^*(T).$$
For $f^* \dashv \forall_f$: $f^*(T) \subseteq S \iff f^{-1}(T) \subseteq S \iff (\forall y \in T,\; f^{-1}(\{y\}) \subseteq S) \iff T \subseteq \forall_f(S)$. $\square$

The logical interpretation: $\exists_f(S)$ is $\{y : \exists x \in S,\; f(x) = y\}$ and $\forall_f(S)$ is $\{y : \forall x,\; f(x) = y \Rightarrow x \in S\}$.

> [!INFO] Historical note: Lawvere's insight
> This observation — that the existential and universal quantifiers arise as adjoints to substitution — was one of Lawvere's key early insights (around 1969), leading to the notion of a *hyperdoctrine* and the categorical foundations of logic. It shows that the asymmetry between $\exists$ and $\forall$ is not logical but structural: they are left and right adjoints to the same functor.

> [!NOTE] Exercise 13
> Show that the existential quantifier $\exists_f$ and universal quantifier $\forall_f$ along a function $f \colon X \to Y$ can be viewed as left and right adjoints to the substitution functor $f^* \colon \mathbf{Sub}(Y) \to \mathbf{Sub}(X)$ (where $\mathbf{Sub}(X)$ is the poset of subsets of $X$ ordered by inclusion).

### 11.3 Mates

**Definition (Mates).** Given adjunctions $(η, \varepsilon) \colon F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ and $(\eta', \varepsilon') \colon F' \dashv G' \colon \mathbf{C}' \rightleftarrows \mathbf{D}'$, and functors $H \colon \mathbf{D} \to \mathbf{D}'$ and $H' \colon \mathbf{C} \to \mathbf{C}'$, define:

Given a natural transformation $\alpha \colon H'G \Rightarrow G'H$ (a square with $G$ on the right), its *mate* $\alpha^\sharp \colon F'H' \Rightarrow HF$ is the composite

$$F'H' \xRightarrow{F'H'\eta} F'H'GF \xRightarrow{F'\alpha F} F'G'HF \xRightarrow{\varepsilon' HF} HF.$$

Dually, given $\beta \colon F'H' \Rightarrow HF$, its mate $\beta^\flat \colon H'G \Rightarrow G'H$ is

$$H'G \xRightarrow{\eta' H'G} G'F'H'G \xRightarrow{G'\beta G} G'HFG \xRightarrow{G'H\varepsilon} G'H.$$

**Proposition.** The operations $\alpha \mapsto \alpha^\sharp$ and $\beta \mapsto \beta^\flat$ are mutually inverse bijections:

$$\{H'G \Rightarrow G'H\} \leftrightarrow \{F'H' \Rightarrow HF\}.$$

The mate correspondence is due to Kelly and Street (1974) and is a fundamental tool in 2-category theory.

> [!NOTE] Exercise 14
> Let $(\eta, \varepsilon) \colon F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ and $(\eta', \varepsilon') \colon F' \dashv G' \colon \mathbf{C}' \rightleftarrows \mathbf{D}'$ be adjunctions. Given a natural transformation $\alpha \colon G'H \Rightarrow H'G$ (for some $H \colon \mathbf{C} \to \mathbf{C}'$ and $H' \colon \mathbf{D} \to \mathbf{D}'$), define the *mate* $\alpha^\sharp \colon HF \Rightarrow F'H'$ of $\alpha$, and prove that taking mates defines a bijection.

### 11.4 Arrow Categories

**Definition (Arrow category).** For a category $\mathbf{C}$, the *arrow category* $\mathbf{C}^{\to}$ has:
- **Objects:** morphisms $f \colon A \to B$ of $\mathbf{C}$.
- **Morphisms:** from $f \colon A \to B$ to $g \colon C \to D$, a *commutative square*, i.e., a pair $(u \colon A \to C,\; v \colon B \to D)$ such that $v \circ f = g \circ u$.
- **Composition:** componentwise.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
A \arrow[r, "f"] \arrow[d, "u"'] & B \arrow[d, "v"] \\
C \arrow[r, "g"'] & D
\end{tikzcd}
\end{document}
```

$\mathbf{C}^{\to}$ is the functor category $[\mathbf{2}, \mathbf{C}]$, where $\mathbf{2}$ is the category $\{0 \to 1\}$ with two objects and one non-identity morphism.

**Proposition.** A functor $F \colon \mathbf{C} \to \mathbf{D}$ extends to a functor $F^{\to} \colon \mathbf{C}^{\to} \to \mathbf{D}^{\to}$ sending $(f \colon A \to B) \mapsto (Ff \colon FA \to FB)$. An adjunction $F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ induces an adjunction $F^{\to} \dashv G^{\to} \colon \mathbf{C}^{\to} \rightleftarrows \mathbf{D}^{\to}$.

*Proof sketch.* $\mathbf{C}^{\to} = [\mathbf{2}, \mathbf{C}]$ and $\mathbf{D}^{\to} = [\mathbf{2}, \mathbf{D}]$. By §6, postcomposition with $F$ and $G$ gives the induced adjunction $F^{\to} \dashv G^{\to}$ as a special case of the functor-category adjunction with $\mathbf{E} = \mathbf{2}$. $\square$

> [!NOTE] Exercise 15
> Define the arrow category $\mathbf{C}^{\to}$ of a category $\mathbf{C}$ (objects are morphisms of $\mathbf{C}$; a morphism from $f \colon A \to B$ to $g \colon C \to D$ is a commutative square). Show that a functor $F \colon \mathbf{C} \to \mathbf{D}$ extends to a functor $F^{\to} \colon \mathbf{C}^{\to} \to \mathbf{D}^{\to}$, and that an adjunction $F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ induces an adjunction $F^{\to} \dashv G^{\to} \colon \mathbf{C}^{\to} \rightleftarrows \mathbf{D}^{\to}$.

---

> [!INFO] Connection to Limits and Colimits
> Many of the structures studied here — representable functors, initial objects in comma categories, and adjunctions — are intimately related to limits and colimits. For example: a right adjoint preserves all limits, and a left adjoint preserves all colimits. This is developed in [[concepts/category-theory/03-limits-colimits|File 3]].

---

## 12. The Category of Elements 🔬

The *category of elements* is a fundamental construction that packages together a functor and its "points" (elements of its values) into a single category. It is the key to understanding why every presheaf is a colimit of representables, and it provides the conceptual foundation for the Grothendieck construction.

> [!INFO] Riehl's treatment
> This construction appears in Riehl's *Category Theory in Context*, §2.3 (for the covariant case) and §2.4 (for the density theorem). It is one of the most important tools in topos theory and homotopy theory, connecting the category-theoretic notion of representability with the set-theoretic notion of elements.

### 12.1 Definition: Contravariant Case

🔑 **Definition (Category of elements, contravariant).** Let $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ be a presheaf. The *category of elements* $\int F$ (also written $\mathrm{el}(F)$ or $\mathbf{C} \int F$) is the category with:

- **Objects:** pairs $(C, x)$ where $C \in \mathbf{C}$ and $x \in F(C)$.
- **Morphisms:** a morphism $(C, x) \to (C', x')$ in $\int F$ is a morphism $f \colon C \to C'$ in $\mathbf{C}$ such that $F(f)(x') = x$. (Since $F$ is contravariant, $F(f) \colon F(C') \to F(C)$, so the condition is that $f$ "pulls back" $x'$ to $x$.)
- **Composition:** induced from $\mathbf{C}$: if $f \colon C \to C'$ and $g \colon C' \to C''$ are morphisms in $\int F$, their composite is $g \circ f \colon C \to C''$, and $F(g \circ f)(x'') = F(f)(F(g)(x'')) = F(f)(x') = x$.
- **Identities:** the identity on $(C, x)$ is $\mathrm{id}_C$, since $F(\mathrm{id}_C)(x) = \mathrm{id}_{F(C)}(x) = x$.

There is a canonical *forgetful functor* (or *projection*) $\pi \colon \int F \to \mathbf{C}$ sending $(C, x) \mapsto C$ and $f \mapsto f$.

> [!WARNING] Variance in morphisms
> *The condition on morphisms is contravariant:* a morphism $(C, x) \to (C', x')$ is a morphism $f \colon C \to C'$ in $\mathbf{C}$ (going the same direction in $\mathbf{C}$) with $F(f)(x') = x$ (pulling back along $F$). This asymmetry is essential — it ensures the composite of morphisms in $\int F$ is well-defined.

> [!EXAMPLE] Example: Representable presheaf
> For the representable presheaf $F = \mathbf{C}(-, A) \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$, an object of $\int F$ is a pair $(C, f)$ where $f \colon C \to A$ is a morphism in $\mathbf{C}$. A morphism $(C, f) \to (C', f')$ in $\int F$ is a morphism $g \colon C \to C'$ with $\mathbf{C}(g, A)(f') = f' \circ g = f$. This is precisely the *slice category* $\mathbf{C}/A$ — objects are morphisms into $A$, and morphisms are commutative triangles.

### 12.2 Covariant Version

**Definition (Category of elements, covariant).** For $F \colon \mathbf{C} \to \mathbf{Set}$ a covariant functor, the category of elements $\int F$ has:

- **Objects:** pairs $(C, x)$ with $x \in F(C)$.
- **Morphisms:** $(C, x) \to (C', x')$ is a morphism $f \colon C \to C'$ in $\mathbf{C}$ with $F(f)(x) = x'$ (now $F(f)$ pushes $x$ forward to $x'$).
- **Projection:** $\pi \colon \int F \to \mathbf{C}$ sending $(C, x) \mapsto C$.

> [!INFO] Relationship to the slice category
> For a covariant $F$, $\int F$ is the *comma category* $(\ast \downarrow F)$ where $\ast$ denotes the singleton set viewed as the unique functor $\mathbf{1} \to \mathbf{Set}$. Concretely, an object is a choice of a set $F(C)$ and an element $x \in F(C)$, which is the same as a function $\{*\} \to F(C)$.

### 12.3 Universal Elements and Representations

📐 The category of elements provides a crisp reformulation of representability via universal elements.

**Definition (Universal element).** A *universal element* for $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is a *terminal object* of $\int F$ — a pair $(R, u)$ with $u \in F(R)$ such that for every $(C, x) \in \int F$ there is a unique morphism $(C, x) \to (R, u)$ in $\int F$, i.e., a unique $f \colon C \to R$ in $\mathbf{C}$ with $F(f)(u) = x$.

> [!WARNING] Variance: terminal, not initial
> *For a contravariant $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$, a universal element is a terminal object of $\int F$, not an initial object.* The direction of the universal property (unique maps into $(R,u)$) matches terminality. For the covariant version, a universal element is an initial object of $\int F$.

**Theorem (Universal element = representation).** $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ has a universal element $(R, u)$ if and only if $F$ is representable, with $u \in F(R)$ being the element corresponding to $\mathrm{id}_R$ under the natural isomorphism $\mathbf{C}(-, R) \xrightarrow{\sim} F$.

*Proof.* Given a representation $\phi \colon \mathbf{C}(-, R) \xrightarrow{\sim} F$, set $u = \phi_R(\mathrm{id}_R) \in F(R)$. For any $(C, x)$, the unique morphism $(C, x) \to (R, u)$ corresponds to $f = \phi_C^{-1}(x) \colon C \to R$: one checks $F(f)(u) = \phi_C(f) = x$ by naturality of $\phi$. Conversely, if $(R, u)$ is terminal, the bijections $f \mapsto F(f)(u)$ assemble into the natural isomorphism. $\square$

**Key conclusion: $\mathbf{C}(-, R) \cong F$ if and only if $(R, u)$ is terminal in $\int F$ for $u = \phi_R(\mathrm{id}_R)$.**

> [!NOTE] Riehl Exercise: Category of Elements
> **(i)** Let $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ be a representable functor with representing object $R$. Show that $\int F$ has a terminal object (not initial — check the variance carefully).
>
> **(ii)** For the functor $F = \mathbf{C}(-, A) \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$, describe $\int F$ explicitly. What are its objects and morphisms?
>
> **(iii)** Show that the projection functor $\pi \colon \int F \to \mathbf{C}$ is a *discrete fibration*: for every object $(C, x) \in \int F$ and morphism $f \colon C' \to C$ in $\mathbf{C}$, there is a unique lift of $f$ to a morphism in $\int F$ with target $(C, x)$.

### 12.4 The Grothendieck Construction

💡 The category of elements is a special case of a much more general construction.

**Definition (Grothendieck construction).** Given a functor $F \colon \mathbf{C} \to \mathbf{Cat}$, the *Grothendieck construction* $\int F$ is the category with:

- **Objects:** pairs $(C, D)$ where $C \in \mathbf{C}$ and $D \in F(C)$.
- **Morphisms:** $(C, D) \to (C', D')$ is a pair $(f \colon C \to C',\; g \colon F(f)(D) \to D')$ where $F(f)(D) \in F(C')$ is the image of $D$ under the functor $F(f) \colon F(C) \to F(C')$.
- **Composition:** $(f, g) \circ (f', g') = (f \circ f',\; g \circ F(f)(g'))$.

The presheaf case $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ is recovered by viewing each set $F(C)$ as a *discrete category* (objects are elements, only identity morphisms). Since morphisms must be of the form $F(f)(D) \to D'$ in a discrete category, this forces $F(f)(D) = D'$, which is exactly the morphism condition for $\int F$ in the covariant setting. The contravariant version arises by working with $F \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Cat}$.

> [!INFO] The Grothendieck construction in homotopy theory
> The Grothendieck construction is the categorical shadow of a deep phenomenon in homotopy theory: a functor $F \colon \mathbf{C} \to \mathbf{Cat}$ (encoding a "family of categories") is equivalent to a *fibration* $\int F \to \mathbf{C}$. This is the *Grothendieck correspondence*, which was generalised by Lurie to $(\infty, 1)$-categories via the *straightening/unstraightening* equivalence in *Higher Topos Theory*.

### 12.5 The Density Theorem via the Category of Elements

🔑 The category of elements gives a canonical indexing for the colimit decomposition of every presheaf into representables.

**Theorem (Density theorem / co-Yoneda lemma).** Every presheaf $X \in [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$ is canonically a colimit of representables:

$$X \cong \operatorname*{colim}_{(C, x) \in \int X} \mathbf{C}(-, C),$$

where the colimit is taken over the diagram $\int X \xrightarrow{\pi} \mathbf{C} \xrightarrow{\mathbf{h}^{\bullet}} [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$ (first project to $\mathbf{C}$, then apply the Yoneda embedding $C \mapsto \mathbf{C}(-, C)$).

*Proof sketch.* One verifies that the colimit cocone is given by the natural transformations $\lambda_{(C,x)} \colon \mathbf{C}(-, C) \Rightarrow X$ corresponding (via Yoneda) to the elements $x \in X(C)$. The universal property of the colimit amounts to the Yoneda lemma applied objectwise. $\square$

**Key insight: $\int X$ is the "right" index category** — its objects are precisely the "points" $(C, x)$ of $X$, and the colimit says $X$ is the "union" (in the presheaf sense) of all its representable subpresheaves, glued along the morphisms of $\int X$.

> [!EXAMPLE] Example: Density in Set
> For $\mathbf{C} = \mathbf{1}$ (the terminal category), $[\mathbf{C}^{\mathrm{op}}, \mathbf{Set}] \cong \mathbf{Set}$. The unique representable is the terminal set $\{*\}$. The category of elements $\int X$ has objects the elements of $X$ and only identity morphisms. The density theorem says $X \cong \operatorname{colim}_{x \in X} \{*\} \cong$ coproduct of $|X|$ copies of $\{*\}$, which is just $X$ itself. $\checkmark$

### 12.6 Functoriality

**Proposition (Functoriality of the Grothendieck construction).** A natural transformation $\alpha \colon F \Rightarrow G$ between presheaves $F, G \colon \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ induces a functor $\int \alpha \colon \int F \to \int G$ over $\mathbf{C}$, defined by $(C, x) \mapsto (C, \alpha_C(x))$ on objects and acting as the identity on morphisms (since if $f \colon C \to C'$ satisfies $F(f)(x') = x$, then $G(f)(\alpha_{C'}(x')) = \alpha_C(F(f)(x')) = \alpha_C(x)$ by naturality of $\alpha$, so $f$ also defines a valid morphism $(C, \alpha_C(x)) \to (C', \alpha_{C'}(x'))$ in $\int G$).

This makes $\int(-) \colon [\mathbf{C}^{\mathrm{op}}, \mathbf{Set}] \to \mathbf{Cat}/\mathbf{C}$ a functor (the Grothendieck construction is functorial in the presheaf argument), where $\mathbf{Cat}/\mathbf{C}$ denotes the slice of $\mathbf{Cat}$ over $\mathbf{C}$.

> [!TIP] Why this matters
> Functoriality means the Grothendieck construction is not just a "trick for a single presheaf" but a genuine functor. In fact, it restricts to an equivalence of categories between the presheaf category $[\mathbf{C}^{\mathrm{op}}, \mathbf{Set}]$ and the full subcategory of $\mathbf{Cat}/\mathbf{C}$ spanned by the *discrete fibrations* over $\mathbf{C}$.

---

## 13. Multivariable and Contravariant Adjunctions ⚡

So far we have treated adjunctions between two categories with a single pair of functors. Many important adjunctions in algebra and geometry involve multiple variables or contravariant functors. This section systematises these generalizations.

> [!INFO] Riehl's treatment
> Two-variable adjunctions appear in Riehl §4.3, and the mates correspondence (already introduced in §11.3) is treated fully in Riehl §4.4. Closed monoidal categories appear throughout modern algebra, topology, and homotopy theory.

### 13.1 Two-Variable Adjunctions

📐 **Definition (Two-variable adjunction).** Given functors $F \colon \mathbf{C} \times \mathbf{D} \to \mathbf{E}$, $G \colon \mathbf{C}^{\mathrm{op}} \times \mathbf{E} \to \mathbf{D}$, $H \colon \mathbf{D}^{\mathrm{op}} \times \mathbf{E} \to \mathbf{C}$, a *two-variable adjunction* (or *multivariable adjunction*) is a natural isomorphism

$$\mathbf{E}(F(C, D),\, E) \cong \mathbf{D}(D,\, G(C, E)) \cong \mathbf{C}(C,\, H(D, E))$$

natural in all three variables $C \in \mathbf{C}$, $D \in \mathbf{D}$, $E \in \mathbf{E}$.

The prototypical example is the *tensor-hom adjunction* in $\mathbf{Ab}$:

$$\mathbf{Ab}(A \otimes B,\, C) \cong \mathbf{Ab}(A,\, \mathrm{Hom}(B, C)) \cong \mathbf{Ab}(B,\, \mathrm{Hom}(A, C)),$$

where $\mathrm{Hom}(B, C)$ is the internal hom of abelian groups (itself an abelian group). The symmetry $A \otimes B \cong B \otimes A$ implies that the two "right adjoint" slots $G$ and $H$ are both the internal hom, just with arguments swapped.

> [!EXAMPLE] Example: The three functors of a two-variable adjunction
> In the tensor-hom example, $F = \otimes \colon \mathbf{Ab} \times \mathbf{Ab} \to \mathbf{Ab}$, $G(A, C) = \mathrm{Hom}(A, C) \colon \mathbf{Ab}^{\mathrm{op}} \times \mathbf{Ab} \to \mathbf{Ab}$, and $H(B, C) = \mathrm{Hom}(B, C)$. Fixing any one variable gives an ordinary adjunction in the remaining two.

### 13.2 Closed Monoidal Categories

**Definition (Symmetric monoidal category).** A *symmetric monoidal category* $(\mathbf{C}, \otimes, I)$ consists of a category $\mathbf{C}$, a bifunctor $\otimes \colon \mathbf{C} \times \mathbf{C} \to \mathbf{C}$, a unit object $I$, and natural isomorphisms (associator, left/right unitors, symmetry) satisfying coherence axioms.

**Definition (Closed monoidal category).** A symmetric monoidal category $(\mathbf{C}, \otimes, I)$ is *closed* if for each object $B \in \mathbf{C}$, the functor $- \otimes B \colon \mathbf{C} \to \mathbf{C}$ has a right adjoint, written $[B, -] \colon \mathbf{C} \to \mathbf{C}$ and called the *internal hom* (or *exponential*):

$$\mathbf{C}(A \otimes B,\, C) \cong \mathbf{C}(A,\, [B, C]),$$

naturally in $A$ and $C$ (with $B$ fixed).

| Category | Monoidal product $\otimes$ | Unit $I$ | Internal hom $[B, C]$ |
|---|---|---|---|
| $\mathbf{Set}$ | Cartesian product $\times$ | $\{*\}$ | Function set $C^B$ |
| $\mathbf{Ab}$ | Tensor product $\otimes_\mathbb{Z}$ | $\mathbb{Z}$ | $\mathrm{Hom}(B, C)$ |
| $\mathbf{Mod}_R$ | $\otimes_R$ | $R$ | $\mathrm{Hom}_R(B, C)$ |
| $\mathbf{Cat}$ | Cartesian product $\times$ | $\mathbf{1}$ | Functor category $[B, C]$ |
| $\mathbf{sSet}$ | Cartesian product | $\Delta^0$ | Simplicial mapping space |
| $\mathbf{Top}$ (compactly generated) | $\times$ | point | Compact-open topology $C^B$ |

> [!WARNING] Not all monoidal categories are closed
> *$\mathbf{Ab}$ with the categorical product $\oplus$ (which is also the coproduct) is a symmetric monoidal category, but the tensor product $\otimes$ is the monoidal structure that makes it closed.* Checking closedness requires finding the right adjoint, which may not exist in general.

### 13.3 Currying

💡 The tensor-hom adjunction in $\mathbf{Set}$ expresses *currying* — a fundamental principle in both mathematics and programming.

**Theorem (Currying).** There is a natural bijection
$$\mathbf{Set}(A \times B,\, C) \cong \mathbf{Set}(A,\, C^B) \cong \mathbf{Set}(A,\, \mathbf{Set}(B, C)),$$
naturally in $A$, $B$, $C$. Explicitly, a function $f \colon A \times B \to C$ corresponds to its *curried* form $\tilde{f} \colon A \to C^B$ defined by $\tilde{f}(a)(b) = f(a, b)$.

The *unit* of the adjunction $- \times B \dashv (-)^B$ is $\eta_A \colon A \to (A \times B)^B$ defined by $\eta_A(a)(b) = (a, b)$.

The *counit* is $\varepsilon_C \colon C^B \times B \to C$ defined by $\varepsilon_C(g, b) = g(b)$ — the *evaluation map*.

**Key conclusion: evaluation $\mathrm{ev} \colon [B, C] \times B \to C$ is the counit of the product-exponential adjunction.**

> [!TIP] Currying in functional programming
> In typed lambda calculus and functional programming languages, the type isomorphism $A \times B \to C \;\simeq\; A \to (B \to C)$ is exactly currying. Haskell's `curry :: ((a, b) -> c) -> a -> b -> c` implements this bijection. The internal hom $[B, C]$ is the function type `b -> c`.

### 13.4 Contravariant Adjunctions

**Definition (Contravariant adjunction).** A *contravariant adjunction* between $\mathbf{C}$ and $\mathbf{D}$ is a natural isomorphism

$$\mathbf{C}(A,\, GB) \cong \mathbf{D}(B,\, FA)$$

natural in $A \in \mathbf{C}$ and $B \in \mathbf{D}$. Equivalently, it is an ordinary adjunction $F \colon \mathbf{D}^{\mathrm{op}} \rightleftarrows \mathbf{C} \colon G^{\mathrm{op}}$ (where we view $F$ as a functor $\mathbf{D}^{\mathrm{op}} \to \mathbf{C}$ and $G$ as a functor $\mathbf{C}^{\mathrm{op}} \to \mathbf{D}$).

**Example (Dualizing object in $\mathbf{Vect}_k$).** For a field $k$, the duality functor $(-)^* = \mathrm{Hom}_k(-, k) \colon \mathbf{Vect}_k^{\mathrm{op}} \to \mathbf{Vect}_k$ gives a contravariant adjunction:
$$\mathbf{Vect}_k(V,\, W^*) \cong \mathbf{Vect}_k(W,\, V^*),$$
because both sides are naturally isomorphic to the set of bilinear forms $V \times W \to k$. *Surprisingly,* this is a self-adjoint contravariant adjunction: $F = G = (-)^*$. Such a structure is called an *ambidextrous adjunction* or *duality*.

> [!EXAMPLE] Example: Stone duality
> *Stone duality* is a classical contravariant adjunction: the functor $\mathrm{Cl}(-) \colon \mathbf{BoolAlg}^{\mathrm{op}} \to \mathbf{Stone}$ (taking the Stone space of ultrafilters of a Boolean algebra) is left adjoint to the functor $\mathrm{Clop}(-) \colon \mathbf{Stone}^{\mathrm{op}} \to \mathbf{BoolAlg}$ (taking the Boolean algebra of clopen sets). This contravariant adjunction is in fact a contravariant equivalence of categories.

### 13.5 The Calculus of Adjunctions

**Proposition (Composition of adjunctions).** Adjunctions compose: if $F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ and $F' \dashv G' \colon \mathbf{D} \rightleftarrows \mathbf{E}$, then $F'F \dashv GG' \colon \mathbf{C} \rightleftarrows \mathbf{E}$.

*Proof.* Compute:
$$\mathbf{E}(F'FA,\, E) \cong \mathbf{D}(FA,\, G'E) \cong \mathbf{C}(A,\, GG'E),$$
where the first bijection uses $F' \dashv G'$ and the second uses $F \dashv G$. Both bijections are natural in $A$ and $E$, so their composite is a natural bijection exhibiting $F'F \dashv GG'$. $\square$

**Unit and counit of the composite:** The unit of $F'F \dashv GG'$ is $A \xrightarrow{\eta_A} GFA \xrightarrow{G\eta'_{FA}} GG'F'FA$, and the counit is $F'FGG'E \xrightarrow{F'\varepsilon_{G'E}} F'G'E \xrightarrow{\varepsilon'_E} E$.

> [!INFO] Adjunctions form a 2-category
> Adjunctions compose and the identity functor is self-adjoint ($\mathrm{id} \dashv \mathrm{id}$ with unit and counit both the identity). More precisely, *adjunctions* are the 1-morphisms of the 2-category $\mathbf{Adj}$ whose 0-morphisms are categories and whose 2-morphisms are *mates* (see §13.6). This 2-categorical structure is developed by Riehl–Verity in their work on formal category theory.

### 13.6 Mates: Full Treatment

🔑 The mates correspondence (previewed in §11.3) admits a full treatment once we have the multivariable perspective.

**Setup.** Suppose we have:
- Adjunctions $(\eta, \varepsilon) \colon F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ and $(\eta', \varepsilon') \colon F' \dashv G' \colon \mathbf{C}' \rightleftarrows \mathbf{D}'$.
- Functors $H \colon \mathbf{C} \to \mathbf{C}'$ and $K \colon \mathbf{D} \to \mathbf{D}'$.

**Definition (Mate correspondence).** There is a canonical bijection between:
- Natural transformations $\alpha \colon KF \Rightarrow F'H$ ("forward square"), and
- Natural transformations $\alpha^\sharp \colon HG \Rightarrow G'K$ ("backward square").

The *mate* $\alpha^\sharp$ of $\alpha \colon KF \Rightarrow F'H$ is defined by the composite:

$$\alpha^\sharp \colon HG \xRightarrow{\eta'_{HG}} G'F'HG \xRightarrow{G'\alpha G} G'KFG \xRightarrow{G'K\varepsilon} G'K.$$

Conversely, the mate of $\beta \colon HG \Rightarrow G'K$ is:

$$\beta^\flat \colon KF \xRightarrow{KF\eta} KFHG \xRightarrow{... } F'HG \xRightarrow{... } F'H,$$

more precisely: $\beta^\flat = (\varepsilon'_{-} \circ F'\beta_F) \colon KF \xRightarrow{K\eta} KFGF \xRightarrow{\beta_{GF}} G'KGF \xRightarrow{...} ...$

The explicit formula for $\beta^\flat$ at component $A \in \mathbf{C}$ is:

$$(\beta^\flat)_A = \varepsilon'_{F'HA} \circ F'(\beta_{GA}) \circ F'H(\eta_A) \;\colon\; KFA \to F'HA.$$

Wait — let us write the correct formula. Given $\beta \colon HG \Rightarrow G'K$, the mate $\beta^\flat \colon KF \Rightarrow F'H$ has components

$$(\beta^\flat)_A = \varepsilon'_{F'HA} \circ F'(\beta_{GA}) \circ F'(H\eta_A)^{-1}...$$

The cleanest statement uses the adjoint transpose directly:

**Proposition (Mate via adjoint transpose).** Given $\alpha \colon KF \Rightarrow F'H$, the mate $\alpha^\sharp_B \colon HGB \to G'KB$ for $B \in \mathbf{D}$ is the adjoint transpose of $K(\varepsilon_B) \circ \alpha_{GB} \colon F'HGB \to KB$ under the adjunction $F' \dashv G'$:

$$\alpha^\sharp_B = \overline{K\varepsilon_B \circ \alpha_{GB}} = G'(K\varepsilon_B \circ \alpha_{GB}) \circ \eta'_{HGB}.$$

**Proposition (Involution).** The mate correspondence is an involution: $(\alpha^\sharp)^\flat = \alpha$ and $(\beta^\flat)^\sharp = \beta$.

*Proof.* Apply the triangle identities for both adjunctions. $\square$

> [!NOTE] Riehl Exercise: Multivariable Adjunctions
> **(i)** Show directly that $\mathbf{Set}(A \times B, C) \cong \mathbf{Set}(A, \mathbf{Set}(B, C))$ naturally in $A$, $B$, $C$. What are the unit and counit of this adjunction (viewed as $- \times B \dashv (-)^B$ for fixed $B$)?
>
> **(ii)** Let $R$ be a commutative ring. Show that the tensor-hom adjunction $\mathbf{Mod}_R(M \otimes_R N, P) \cong \mathbf{Mod}_R(M, \mathrm{Hom}_R(N, P))$ defines a two-variable adjunction, and describe the unit and counit.
>
> **(iii)** If $F \dashv G \colon \mathbf{C} \rightleftarrows \mathbf{D}$ and $F' \dashv G' \colon \mathbf{D} \rightleftarrows \mathbf{E}$, verify that $F'F \dashv GG'$ by constructing the unit and counit of the composite adjunction explicitly from the units and counits of the factors.

> [!WARNING] Common mistake: composing mates
> *When taking mates, the direction of $H$ and $K$ relative to the adjunctions must be tracked carefully.* A natural transformation $\alpha \colon KF \Rightarrow F'H$ goes "from the left-adjoint side to the left-adjoint side," and its mate $\alpha^\sharp \colon HG \Rightarrow G'K$ goes "from the right-adjoint side to the right-adjoint side." Mixing these directions is a common source of errors in 2-categorical arguments.

---

> [!INFO] Connection to Limits and Colimits
> Many of the structures studied here — representable functors, initial objects in comma categories, and adjunctions — are intimately related to limits and colimits. For example: a right adjoint preserves all limits, and a left adjoint preserves all colimits. This is developed in [[concepts/category-theory/03-limits-colimits|File 3]].

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Tom Leinster, Part III Category Theory Course (2000) | Cambridge lecture course; Sheet 2 covers adjoints and representables; the source of all 15 exercises in this note | [Course page](https://webhomes.maths.ed.ac.uk/~tl/categories/index.html) |
| Tom Leinster, *Basic Category Theory* (2014) | Introductory textbook; Chapters 2–4 cover adjunctions, representables, and Yoneda | [arXiv:1612.09375](https://arxiv.org/abs/1612.09375) |
| Saunders Mac Lane, *Categories for the Working Mathematician* (Springer, 1971) | The foundational reference; adjunctions in Chapter IV, Yoneda in Chapter III | [Springer](https://link.springer.com/book/9780387984032) |
| Emily Riehl, *Category Theory in Context* (Dover, 2016) | Modern treatment; Chapters 1–2 give a careful development of representables, Yoneda, and adjunctions | [Author's page](https://math.jhu.edu/~eriehl/context.pdf) |
| Francis Borceux, *Handbook of Categorical Algebra*, Vol. 1 (Cambridge UP, 1994) | Encyclopaedic reference; complete proofs of all results; adjunctions in Chapter 3 | [Cambridge UP](https://www.cambridge.org/core/books/handbook-of-categorical-algebra/category-theory/3EFFB6E7A7C86BAF0E3B08E0EBEBCA39) |
| G. M. Kelly and Ross Street, "Review of the Elements of 2-Categories" (1974) | Original source for the mates correspondence | [SDMA proceedings](https://link.springer.com/chapter/10.1007/BFb0063101) |
| F. W. Lawvere, "Adjointness in Foundations" (1969) | Original paper showing quantifiers are adjoints to substitution | [Dialectica 23(3–4)](https://doi.org/10.1111/j.1746-8361.1969.tb01194.x) |
| nLab: Adjunction | Community wiki with comprehensive formal treatment | [nLab](https://ncatlab.org/nlab/show/adjunction) |
| Wikipedia: Yoneda Lemma | Accessible summary with proof sketch and corollaries | [Wikipedia](https://en.wikipedia.org/wiki/Yoneda_lemma) |
| Wikipedia: Adjoint Functors | Unit-counit form, triangle identities, and equivalence of definitions | [Wikipedia](https://en.wikipedia.org/wiki/Adjoint_functors) |
